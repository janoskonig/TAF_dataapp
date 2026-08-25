import tempfile
import unittest
from pathlib import Path

from followup import normalise_taj
from upload_model_analysis_to_nas import flattened_name, uploadable_files


class ModelStlInventoryTests(unittest.TestCase):
    def test_normalise_taj_keeps_leading_zeroes(self):
        self.assertEqual(normalise_taj("009-907-384"), "009907384")
        self.assertEqual(normalise_taj(9907384), "009907384")

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

if __name__ == "__main__":
    unittest.main()
