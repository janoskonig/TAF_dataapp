import ast
import copy
import io
import json
import os
import types
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from flask import Flask, jsonify, request
from ftplib import all_errors, error_perm
from io import BytesIO
from werkzeug.utils import secure_filename

from followup import create_followup_blueprint


class ScriptedCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = None
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def close(self):
        pass

    def execute(self, sql, params=()):
        self.connection.executions.append((sql, params))
        if self.connection.fail_on_execute == len(self.connection.executions):
            raise RuntimeError("forced transaction failure")
        if 'FROM cohort c' in sql:
            self.description = [("patient_id",), ("taj",)]
            self._rows = [(1, "123456789")]

    def fetchall(self):
        return list(self._rows)


class ScriptedConnection:
    def __init__(self, fail_on_execute=None):
        self.fail_on_execute = fail_on_execute
        self.executions = []
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return ScriptedCursor(self)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def build_followup_app(connections):
    remaining = list(connections)

    def connection_factory():
        return remaining.pop(0)

    app = Flask(__name__, root_path=str(Path(__file__).parent))
    app.secret_key = "test-secret"
    app.testing = True
    app.register_blueprint(
        create_followup_blueprint(
            connection_factory=connection_factory,
            hue_calculator=lambda *_: 1.0,
            nas_uploader=lambda *_: "ftp://test/image.tiff",
            model_stl_inventory_loader=lambda: ({}, True),
            model_file_lister=lambda *_: ([], True),
            model_file_uploader=lambda *_: "stored",
            model_file_downloader=lambda *_: None,
            upload_folder="/private/tmp",
            allowed_file=lambda *_: True,
        )
    )
    return app


def authenticate(client):
    with client.session_transaction() as session:
        session["followup_authenticated"] = True
        session["followup_csrf"] = "csrf-test"


def test_blank_intake_fields_cannot_erase_saved_values():
    read_connection = ScriptedConnection()
    write_connection = ScriptedConnection()
    app = build_followup_app([read_connection, write_connection])

    with patch.dict(os.environ, {"FOLLOWUP_ACCESS_CODE": "configured"}):
        with app.test_client() as client:
            authenticate(client)
            response = client.post(
                "/followup/patient/1/intake",
                data={"csrf_token": "csrf-test", "f9": ""},
            )

    assert response.status_code == 302
    assert write_connection.committed
    sql, params = write_connection.executions[0]
    assert "f9 = COALESCE(EXCLUDED.f9, followup_visits.f9)" in sql
    assert "data_collector = COALESCE(EXCLUDED.data_collector" in sql
    assert "consent_confirmed = followup_visits.consent_confirmed OR" in sql
    assert params[2] == "arrived"


def test_logistics_and_contact_history_are_one_atomic_transaction():
    read_connection = ScriptedConnection()
    write_connection = ScriptedConnection(fail_on_execute=2)
    app = build_followup_app([read_connection, write_connection])

    with patch.dict(os.environ, {"FOLLOWUP_ACCESS_CODE": "configured"}):
        with app.test_client() as client:
            authenticate(client)
            with pytest.raises(RuntimeError, match="forced transaction failure"):
                client.post(
                    "/followup/patient/1/logistics",
                    data={
                        "csrf_token": "csrf-test",
                        "visit_status": "scheduled",
                        "contact_note": "test",
                    },
                )

    assert len(write_connection.executions) == 2
    assert write_connection.rolled_back
    assert not write_connection.committed


def test_questionnaire_answer_is_saved_as_a_draft_immediately():
    read_connection = ScriptedConnection()
    write_connection = ScriptedConnection()
    app = build_followup_app([read_connection, write_connection])

    with patch.dict(os.environ, {"FOLLOWUP_ACCESS_CODE": "configured"}):
        with app.test_client() as client:
            authenticate(client)
            response = client.post(
                "/followup/patient/1/questionnaire/draft",
                data={"csrf_token": "csrf-test", "ohip_1_recall": "3"},
            )

    assert response.status_code == 200
    assert response.get_json() == {"saved": True}
    sql, params = write_connection.executions[0]
    assert "ohip_1_recall" in sql
    assert "questionnaire_completed_at" not in sql
    assert params[-1] == 3
    assert write_connection.committed


class ApiCursor:
    def __init__(self, connection):
        self.connection = connection

    def execute(self, sql, params=()):
        self.connection.executions.append((sql, params))

    def fetchone(self):
        return (48,)

    def close(self):
        pass


class ApiConnection:
    def __init__(self):
        self.executions = []
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return ApiCursor(self)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def load_main_functions(names, namespace):
    source_path = Path(__file__).with_name("main.py")
    tree = ast.parse(source_path.read_text())
    functions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            copied = copy.deepcopy(node)
            copied.decorator_list = []
            functions.append(copied)
    missing = set(names) - {node.name for node in functions}
    assert not missing, f"Missing functions in main.py: {sorted(missing)}"
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


@pytest.fixture
def main_storage():
    namespace = {
        "json": json,
        "uuid": uuid,
        "os": os,
        "BytesIO": BytesIO,
        "all_errors": all_errors,
        "error_perm": error_perm,
        "secure_filename": secure_filename,
        "datetime": datetime,
        "BUDAPEST_TZ": ZoneInfo("Europe/Budapest"),
        "nas_host": "test-host",
        "nas_user": "test-user",
        "nas_password": "test-pass",
        "nas_folder": "test-root",
        "nas_patients_folder": "patients",
        "model_analysis_folder": "model-analysis",
        "model_manifest_filename": ".model-analysis-manifest.json",
    }
    load_main_functions(
        {
            "ensure_ftp_subdirectory",
            "read_model_manifest",
            "fallback_model_file_listing",
            "model_file_listing",
            "write_model_manifest",
            "update_model_manifest",
            "safe_patient_folder",
            "upload_patient_model_stream",
        },
        namespace,
    )
    return types.SimpleNamespace(**namespace)


def test_morphometry_api_preserves_omitted_values_and_updates_latest_row():
    connection = ApiConnection()
    namespace = {
        "request": request,
        "jsonify": jsonify,
        "json": json,
        "create_db_connection": lambda: connection,
        "_valid_blender_api_key": lambda: True,
    }
    load_main_functions({"_float_or_none", "api_morphometria"}, namespace)
    app = Flask(__name__)
    app.add_url_rule(
        "/api/morphometria",
        view_func=namespace["api_morphometria"],
        methods=["POST"],
    )
    response = app.test_client().post(
        "/api/morphometria", json={"TAJ": "123456789", "F1": 12.5}
    )

    assert response.status_code == 200
    select_sql, _ = connection.executions[0]
    update_sql, update_params = connection.executions[1]
    assert 'ORDER BY "id" DESC LIMIT 1 FOR UPDATE' in select_sql
    assert 'WHERE "id" = %s' in update_sql
    assert update_sql.count("COALESCE(") >= 13
    assert update_params[0] == 12.5
    assert update_params[1] is None
    assert update_params[-1] == 48
    assert connection.committed and connection.closed


class MemoryFTP:
    def __init__(self, files, fail_manifest_install=False):
        self.files = dict(files)
        self.fail_manifest_install = fail_manifest_install

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def login(self, *_args):
        pass

    def cwd(self, *_args):
        pass

    def mkd(self, *_args):
        pass

    def quit(self):
        pass

    def mlsd(self):
        return [(name, {"type": "file", "size": str(len(data))}) for name, data in self.files.items()]

    def nlst(self):
        return list(self.files)

    def retrbinary(self, command, callback):
        callback(self.files[command.split(" ", 1)[1]])

    def storbinary(self, command, source, **_kwargs):
        self.files[command.split(" ", 1)[1]] = source.read()

    def rename(self, source, destination):
        if self.fail_manifest_install and destination == ".model-analysis-manifest.json":
            self.fail_manifest_install = False
            raise error_perm("forced rename failure")
        if source not in self.files:
            raise error_perm("missing source")
        self.files[destination] = self.files.pop(source)

    def delete(self, name):
        if name not in self.files:
            raise error_perm("missing file")
        del self.files[name]


def test_same_name_model_upload_creates_a_version_without_overwrite(main_storage):
    manifest_name = main_storage.model_manifest_filename
    old_manifest = json.dumps({"files": [{"name": "model.stl", "size": 3}]}).encode()
    ftp = MemoryFTP({"model.stl": b"old", manifest_name: old_manifest})

    main_storage.upload_patient_model_stream.__globals__["FTP"] = lambda *_args: ftp
    stored = main_storage.upload_patient_model_stream(
        "123456789", io.BytesIO(b"new"), "model.stl", 3
    )

    assert stored != "model.stl"
    assert ftp.files["model.stl"] == b"old"
    assert ftp.files[stored] == b"new"


def test_manifest_rename_failure_restores_previous_manifest(main_storage):
    manifest_name = main_storage.model_manifest_filename
    old_manifest = b'{"files": [{"name": "old.stl", "size": 3}]}'
    ftp = MemoryFTP({manifest_name: old_manifest}, fail_manifest_install=True)

    with pytest.raises(error_perm):
        main_storage.write_model_manifest(ftp, [{"name": "new.stl", "size": 3}])

    assert ftp.files[manifest_name] == old_manifest


def test_addon_never_sends_unmeasured_defaults_and_keeps_local_backup():
    source = Path("addon/taf_addon.py").read_text()
    assert 'payload_dict = {"TAJ": props.patient_id.strip()}' in source
    assert 'if props.f1_n_pairs > 0 or props.f1_profile_json:' in source
    assert '_A10_FELSO_NAME in bpy.data.objects' in source
    assert "with open(backup_path, 'x'" in source
    assert "os.remove(backup_path)" not in source
