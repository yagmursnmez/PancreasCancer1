"""Web PACS panel status is informational and never starts a listener."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH / "web"))
import app as web_app


class PacsWebStatusTests(unittest.TestCase):
    def test_missing_config_is_reported_without_starting_pacs(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            web_app, "PACS_CONFIG_PATH", Path(folder) / "pacs_config.json"
        ):
            response = web_app.app.test_client().get("/api/pacs/status")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "not_configured")
        self.assertFalse(payload["listener_enabled"])

    def test_index_has_a_pacs_source_tab(self):
        html = web_app.app.test_client().get("/").get_data(as_text=True)
        self.assertIn('id="tabPacsBtn"', html)
        self.assertIn('id="pacsPanel"', html)


if __name__ == "__main__":
    unittest.main()
