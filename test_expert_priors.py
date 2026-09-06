import csv
import io
from pathlib import Path

import pytest
from flask import Blueprint, Flask
from psycopg2.extras import Json

from expert_priors import (
    BACKGROUND_CSV_COLUMNS,
    ITEM_CODES,
    ITEMS,
    PRIOR_CSV_COLUMNS,
    completeness_errors,
    create_expert_blueprint,
    item_progress,
    parse_field_name,
    prior_rows,
)

RESPONSE_COLUMNS = [
    "id", "expert_code", "expert_name", "expert_affiliation", "token", "status", "consent_confirmed",
    "background", "calibration", "items", "closing", "form_version", "submitted_at", "created_at", "updated_at",
]
SCHEMA_OK = ("to_regclass('public.expert_prior_responses')", ["responses_table", "name_column"], [("expert_prior_responses", True)])
INSERT_OK = ("INSERT INTO expert_prior_responses", ["id"], [(7,)])


def response_row(**overrides):
    row = {
        "id": 7, "expert_code": "SZ07", "expert_name": "Dr. Teszt Elek", "expert_affiliation": "SE Fogpótlástani Klinika",
        "token": "tok", "status": "draft", "consent_confirmed": True,
        "background": {}, "calibration": {}, "items": {}, "closing": {}, "form_version": "v1.0",
        "submitted_at": None, "created_at": "2026-09-06 10:00", "updated_at": "2026-09-06 10:05",
    }
    row.update(overrides)
    return row


def token_lookup(row):
    return ("WHERE token = %s", RESPONSE_COLUMNS, [row])


def id_lookup(row):
    return ("WHERE id = %s", RESPONSE_COLUMNS, [row])


def listing(rows):
    return ("FROM expert_prior_responses", RESPONSE_COLUMNS, rows)


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = None
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=()):
        self.connection.executions.append((sql, params))
        self.description = None
        self._rows = []
        for matcher, columns, rows in self.connection.script:
            if matcher in sql:
                self.description = [(column,) for column in columns]
                self._rows = [
                    tuple(row[column] for column in columns) if isinstance(row, dict) else row
                    for row in rows
                ]
                break

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConnection:
    def __init__(self, script):
        self.script = script
        self.executions = []
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def build_app(script):
    connections = []

    def factory():
        connection = FakeConnection(script)
        connections.append(connection)
        return connection

    app = Flask(__name__, root_path=str(Path(__file__).parent))
    app.secret_key = "test-secret"
    app.testing = True
    clinical = Blueprint("followup", __name__, url_prefix="/followup")

    @clinical.route("/login")
    def login():
        return "klinikai belépés"

    app.register_blueprint(clinical)
    app.register_blueprint(create_expert_blueprint(connection_factory=factory))
    return app, connections


def executed_sql(connections):
    return [sql for connection in connections for sql, _ in connection.executions]


def auth_expert(client):
    with client.session_transaction() as session:
        session["expert_authenticated"] = True
        session["expert_csrf"] = "csrf-test"


def auth_admin(client):
    with client.session_transaction() as session:
        session["followup_authenticated"] = True
        session["expert_csrf"] = "csrf-test"


def complete_form():
    form = {
        "csrf_token": "csrf-test",
        "bg__evek_gyakorlat": "22",
        "bg__fogsorok_szama_kat": "500-1000",
        "cal__alap_siker_100": "75",
        "cal__anatomia_sulya_pct": "40",
        "cl__onertekeles": "4",
    }
    for code in ITEM_CODES:
        form[f"item__{code}__irany"] = "B_kedvezotlenebb"
        form[f"item__{code}__p_irany"] = "90"
    form["item__A11__siker_A"] = "85"
    form["item__A11__siker_B"] = "60"
    form["item__A11__kulonbseg_min"] = "10"
    form["item__A11__kulonbseg_max"] = "40"
    form["item__F5__sub__lokalizacio"] = "frontalis"
    return form


@pytest.fixture(autouse=True)
def no_access_code(monkeypatch):
    monkeypatch.delenv("EXPERT_ACCESS_CODE", raising=False)


def test_item_codes_match_r_prior_template():
    with open(Path(__file__).parent / "predict_expert_priorok.csv", encoding="utf-8-sig") as handle:
        template_codes = {row["tetel"] for row in csv.DictReader(handle)}
    assert template_codes == set(ITEM_CODES)
    assert len(ITEMS) == 17


def test_parse_field_name_keeps_sub_question_suffix():
    assert parse_field_name("item__F5__sub__lokalizacio") == ("item", "F5", "sub__lokalizacio")
    assert parse_field_name("bg__oktat") == ("bg", None, "oktat")
    with pytest.raises(ValueError):
        parse_field_name("csrf_token")


def test_start_page_is_open_without_access_code():
    app, _ = build_app([SCHEMA_OK])
    client = app.test_client()
    response = client.get("/expert")
    assert response.status_code == 200
    assert "Kitöltés indítása" in response.get_data(as_text=True)
    assert "Kilépés" not in response.get_data(as_text=True)
    login = client.get("/expert/login")
    assert login.status_code == 302 and login.headers["Location"].endswith("/expert")


def test_optional_access_code_guards_start_page(monkeypatch):
    monkeypatch.setenv("EXPERT_ACCESS_CODE", "szakerto-kod")
    app, _ = build_app([SCHEMA_OK])
    client = app.test_client()
    response = client.get("/expert")
    assert response.status_code == 302
    assert "/expert/login" in response.headers["Location"]


def test_login_with_correct_code_opens_session(monkeypatch):
    monkeypatch.setenv("EXPERT_ACCESS_CODE", "szakerto-kod")
    app, _ = build_app([SCHEMA_OK])
    client = app.test_client()
    with client.session_transaction() as session:
        session["expert_csrf"] = "csrf-test"
    response = client.post("/expert/login", data={"csrf_token": "csrf-test", "access_code": "szakerto-kod"})
    assert response.status_code == 302
    with client.session_transaction() as session:
        assert session.get("expert_authenticated") is True


def test_start_response_inserts_row_and_assigns_expert_code():
    app, connections = build_app([SCHEMA_OK, INSERT_OK])
    client = app.test_client()
    auth_expert(client)
    response = client.post("/expert/start", data={"csrf_token": "csrf-test", "consent": "on", "expert_name": " Dr. Teszt Elek ", "expert_affiliation": "SE"})
    assert response.status_code == 302
    assert "/expert/urlap/" in response.headers["Location"]
    statements = [(sql, params) for connection in connections for sql, params in connection.executions]
    assert any("INSERT INTO expert_prior_responses" in sql and params[1] == "Dr. Teszt Elek" and params[2] == "SE" for sql, params in statements)
    assert any("SET expert_code = %s" in sql and params[0] == "SZ07" for sql, params in statements)
    assert connections[-1].committed


def test_start_without_consent_or_name_is_refused():
    app, connections = build_app([SCHEMA_OK, SCHEMA_OK])
    client = app.test_client()
    auth_expert(client)
    response = client.post("/expert/start", data={"csrf_token": "csrf-test", "expert_name": "Dr. Teszt Elek"})
    assert response.status_code == 302
    response = client.post("/expert/start", data={"csrf_token": "csrf-test", "consent": "on", "expert_name": "  "})
    assert response.status_code == 302
    assert not any("INSERT" in sql for sql in executed_sql(connections))


def test_form_page_renders_all_items():
    app, _ = build_app([SCHEMA_OK, token_lookup(response_row())])
    client = app.test_client()
    auth_expert(client)
    html = client.get("/expert/urlap/tok").get_data(as_text=True)
    for code in ITEM_CODES:
        assert f'id="item-{code}"' in html
    assert 'id="itemsDone">0</span> / 17 tétel' in html


def test_autosave_writes_sub_question_into_item_json():
    app, connections = build_app([token_lookup(response_row())])
    client = app.test_client()
    auth_expert(client)
    response = client.post(
        "/expert/urlap/tok/mentes",
        data={"csrf_token": "csrf-test", "field": "item__F5__sub__lokalizacio", "value": "frontalis"},
    )
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    update = [(sql, params) for connection in connections for sql, params in connection.executions if "jsonb_set" in sql]
    assert len(update) == 1
    sql, params = update[0]
    assert params[0] == "F5"
    assert isinstance(params[2], Json)
    assert params[2].adapted == {"sub_lokalizacio": "frontalis"}


def test_autosave_rejects_certainty_outside_scale():
    app, connections = build_app([token_lookup(response_row())])
    client = app.test_client()
    auth_expert(client)
    response = client.post(
        "/expert/urlap/tok/mentes",
        data={"csrf_token": "csrf-test", "field": "item__F1__p_irany", "value": "73"},
    )
    assert response.status_code == 400
    assert not any("UPDATE" in sql for sql in executed_sql(connections))


def test_autosave_refuses_submitted_response():
    app, _ = build_app([token_lookup(response_row(status="submitted"))])
    client = app.test_client()
    auth_expert(client)
    response = client.post(
        "/expert/urlap/tok/mentes",
        data={"csrf_token": "csrf-test", "field": "bg__oktat", "value": "igen"},
    )
    assert response.status_code == 409


def test_submit_incomplete_form_is_rejected_with_error_list():
    app, connections = build_app([token_lookup(response_row())])
    client = app.test_client()
    auth_expert(client)
    response = client.post("/expert/urlap/tok/bekuldes", data={"csrf_token": "csrf-test", "item__F1__irany": "B_kedvezotlenebb"})
    assert response.status_code == 400
    html = response.get_data(as_text=True)
    assert "F1 · Felső állcsontgerinc magassága (profilja): bizonyosság" in html
    assert "B1. Alap-sikerarány" in html
    assert not any("status = 'submitted'" in sql for sql in executed_sql(connections))


def test_submit_complete_form_marks_response_submitted():
    app, connections = build_app([token_lookup(response_row())])
    client = app.test_client()
    auth_expert(client)
    response = client.post("/expert/urlap/tok/bekuldes", data=complete_form())
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/expert/kesz/tok")
    submits = [(sql, params) for connection in connections for sql, params in connection.executions if "status = 'submitted'" in sql]
    assert len(submits) == 1
    items = submits[0][1][2].adapted
    assert items["A11"]["siker_A"] == 85 and items["A11"]["kulonbseg_max"] == 40
    assert items["F5"]["sub_lokalizacio"] == "frontalis"
    assert submits[0][1][0].adapted["evek_gyakorlat"] == 22


def test_export_requires_clinical_session():
    app, _ = build_app([SCHEMA_OK])
    client = app.test_client()
    auth_expert(client)
    response = client.get("/expert/admin/export/priorok.csv")
    assert response.status_code == 302
    assert "/followup/login" in response.headers["Location"]


def test_prior_export_matches_r_script_template():
    submitted = response_row(
        status="submitted",
        items={
            "F5": {"irany": "B_kedvezotlenebb", "p_irany": 80, "siker_A": 80, "siker_B": 55,
                   "kulonbseg_min": 10, "kulonbseg_max": 40, "mechanizmus": ["alatamasztas", "fajdalom"],
                   "sub_lokalizacio": "frontalis", "megjegyzes": "ritka"},
            "F2": {"irany": "nem_monoton"},
        },
    )
    app, _ = build_app([listing([submitted])])
    client = app.test_client()
    auth_admin(client)
    response = client.get("/expert/admin/export/priorok.csv")
    assert response.status_code == 200
    text = response.get_data(as_text=True).lstrip("﻿")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0].keys()) == PRIOR_CSV_COLUMNS
    assert len(rows) == len(ITEM_CODES)
    by_code = {row["tetel"]: row for row in rows}
    assert by_code["F5"]["irany"] == "B_kedvezotlenebb" and by_code["F5"]["p_irany"] == "80"
    assert by_code["F5"]["mechanizmus"] == "alatamasztas; fajdalom"
    assert "lokalizacio=frontális" in by_code["F5"]["megjegyzes"]
    assert by_code["F2"]["alak"] == "optimum"
    assert by_code["A1"]["irany"] == ""


def test_background_export_columns():
    submitted = response_row(status="submitted", background={"evek_gyakorlat": 30, "fogsorok_szama_kat": ">3000"},
                             calibration={"alap_siker_100": 70, "anatomia_sulya_pct": 35}, closing={"rang_1": "A11", "onertekeles": "4"})
    app, _ = build_app([listing([submitted])])
    client = app.test_client()
    auth_admin(client)
    text = client.get("/expert/admin/export/hatter.csv").get_data(as_text=True).lstrip("﻿")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0].keys()) == BACKGROUND_CSV_COLUMNS
    assert rows[0]["rang_1"] == "A11" and rows[0]["alap_siker_100"] == "70"
    assert rows[0]["nev"] == "Dr. Teszt Elek" and rows[0]["intezmeny"] == "SE Fogpótlástani Klinika"


def test_admin_pages_render():
    submitted = response_row(status="submitted", items={"A1": {"irany": "B_kedvezotlenebb", "p_irany": 95}})
    app, _ = build_app([SCHEMA_OK, id_lookup(submitted), listing([submitted, response_row(id=8, expert_code="SZ08", token="tok8")])])
    client = app.test_client()
    auth_admin(client)
    html = client.get("/expert/admin").get_data(as_text=True)
    assert "SZ07" in html and "SZ08" in html and "Tételenkénti gyorsösszesítés" in html
    assert "Dr. Teszt Elek" in html
    view = client.get("/expert/admin/7").get_data(as_text=True)
    assert "a B pólus a kedvezőtlenebb" in view and "Válasz törlése" in view


def test_admin_delete_requires_code_confirmation():
    row = response_row(status="submitted")
    app, connections = build_app([id_lookup(row)])
    client = app.test_client()
    auth_admin(client)
    refused = client.post("/expert/admin/7/torles", data={"csrf_token": "csrf-test", "confirm": "rossz"})
    assert refused.status_code == 302
    assert not any("DELETE" in sql for sql in executed_sql(connections))
    accepted = client.post("/expert/admin/7/torles", data={"csrf_token": "csrf-test", "confirm": "SZ07"})
    assert accepted.status_code == 302
    assert any("DELETE FROM expert_prior_responses" in sql for sql in executed_sql(connections))


def test_helpers_progress_and_completeness():
    items = {"F1": {"irany": "B_kedvezotlenebb", "p_irany": 90}, "F2": {"irany": "nem_monoton"}, "F3": {"irany": "A_kedvezotlenebb"}}
    assert item_progress(items) == 2
    problems = completeness_errors({"items": items, "background": {}, "calibration": {}, "closing": {}})
    assert any("F3" in problem and "bizonyosság" in problem for problem in problems)
    rows = prior_rows([response_row(status="submitted", items=items)])
    assert len(rows) == 17
