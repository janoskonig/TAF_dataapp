import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from flask import Blueprint, Flask

from followup import (
    VISIT_STATUSES,
    VISIT_STATUS_LABELS,
    create_followup_blueprint,
    normalise_taj,
)
from upload_model_analysis_to_nas import flattened_name, uploadable_files


class ModelStlInventoryTests(unittest.TestCase):
    def test_normalise_taj_keeps_leading_zeroes(self):
        self.assertEqual(normalise_taj("009-907-384"), "009907384")
        self.assertEqual(normalise_taj(9907384), "009907384")

    def test_every_visit_status_has_a_hungarian_label(self):
        self.assertEqual(set(VISIT_STATUS_LABELS), VISIT_STATUSES)
        self.assertEqual(VISIT_STATUS_LABELS["not_contacted"], "Még nem kerestük")
        self.assertEqual(VISIT_STATUS_LABELS["no_show"], "Nem jelent meg")
        self.assertEqual(VISIT_STATUS_LABELS["unreachable"], "Nem elérhető")

    def test_upload_file_selection_ignores_macos_metadata_and_flattens_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            patient = Path(temporary_directory)
            nested = patient / "fogatlan"
            nested.mkdir()
            model = nested / "lower jaw.stl"
            model.touch()
            (nested / "._lower jaw.stl").touch()
            (patient / ".DS_Store").touch()

            files = uploadable_files(patient)

        self.assertEqual(files, [model])
        self.assertEqual(flattened_name(patient, model), "fogatlan__lower jaw.stl")

    def test_protected_addon_download_serves_repository_version(self):
        app = Flask(__name__, root_path=str(Path(__file__).parent))
        app.secret_key = "test-secret"
        baseline = Blueprint("baseline", __name__, url_prefix="/baseline")

        @baseline.get("")
        def dashboard():
            return "baseline"

        app.register_blueprint(baseline)
        app.register_blueprint(
            create_followup_blueprint(
                connection_factory=lambda: None,
                hue_calculator=lambda *_: None,
                nas_uploader=lambda *_: None,
                model_stl_inventory_loader=lambda: ({}, True),
                model_file_lister=lambda *_: ([], True),
                model_file_uploader=lambda *_: None,
                model_file_downloader=lambda *_: None,
                upload_folder=tempfile.gettempdir(),
                allowed_file=lambda *_: True,
            )
        )

        with patch.dict(os.environ, {"FOLLOWUP_ACCESS_CODE": "test-code"}):
            with app.test_client() as client:
                with client.session_transaction() as session:
                    session["followup_authenticated"] = True
                response = client.get("/followup/blender-addon/download")

        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment; filename=taf_addon.py", response.headers["Content-Disposition"])
        self.assertIn(b"taf.save_and_upload_blend", response.data)
        response.close()

        with patch.dict(os.environ, {"FOLLOWUP_ACCESS_CODE": "test-code"}):
            with app.test_client() as client:
                with client.session_transaction() as session:
                    session["followup_authenticated"] = True
                response = client.get("/followup/blender-addon")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Install from Disk", response.get_data(as_text=True))
        response.close()

if __name__ == "__main__":
    unittest.main()
