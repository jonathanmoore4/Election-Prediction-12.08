import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import read_in_raw


class LocalSourceTests(unittest.TestCase):
    def test_relative_source_reads_offline_and_validates_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            contents = b" name ,votes\nExample,123\n"
            (Path(directory) / "input.csv").write_bytes(contents)
            source = {
                "path": "input.csv",
                "sha256": hashlib.sha256(contents).hexdigest(),
                "reader": "csv",
                "kwargs": {},
                "expected_columns": ["name", "votes"],
            }
            with patch.object(read_in_raw, "PROJECT_ROOT", Path(directory)), patch.object(
                read_in_raw, "urlopen", side_effect=AssertionError("Unexpected download")
            ):
                result = read_in_raw.read_raw_data({"local": source})["local"]
                self.assertEqual(result.to_dict("records"), [{"name": "Example", "votes": 123}])
                source["expected_columns"] = ["absent"]
                with self.assertRaisesRegex(RuntimeError, "Missing expected columns"):
                    read_in_raw.read_raw_data({"local": source})
                (Path(directory) / "input.csv").write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    read_in_raw.read_raw_data({"local": source})
                (Path(directory) / "input.csv").unlink()
                with self.assertRaisesRegex(RuntimeError, "Required local input is missing"):
                    read_in_raw.read_raw_data({"local": source})

    def test_supplied_workbook_is_repeatable_without_downloads(self):
        sources = {"results_1997_local": read_in_raw.RAW_SOURCES["results_1997_local"]}
        with patch.object(read_in_raw, "urlopen", side_effect=AssertionError("Unexpected download")):
            first = read_in_raw.read_raw_data(sources)["results_1997_local"]
            second = read_in_raw.read_raw_data(sources)["results_1997_local"]
        self.assertFalse(first.empty)
        pd.testing.assert_frame_equal(first, second)

    def test_empty_sources_does_not_download(self):
        with patch.object(read_in_raw, "urlopen", side_effect=AssertionError("Unexpected download")):
            self.assertEqual(read_in_raw.read_raw_data({}), {})

    def test_url_sources_still_use_download_reader(self):
        source = {"url": "https://example.invalid/results.csv", "reader": "csv", "kwargs": {}}
        with patch.object(read_in_raw, "_read_and_validate_dataframe") as reader:
            read_in_raw.read_raw_data({"remote": source})
        reader.assert_called_once_with(source["url"], source)


if __name__ == "__main__":
    unittest.main()
