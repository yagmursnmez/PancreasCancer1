"""Unit tests for the memory-only PACS bridge; no PACS network is contacted."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pydicom.dataset import Dataset
from pydicom.uid import CTImageStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pacs_bridge import (
    InMemorySeriesCache, NOT_AUTHORIZED, PacsBridgeConfig,
    PacsConfigurationError, PacsStorageSCP, SUCCESS,
)


def _ct(study="1.2.3", series="1.2.3.4", pixels=128):
    dataset = Dataset()
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = f"{series}.{pixels}"
    dataset.StudyInstanceUID = study
    dataset.SeriesInstanceUID = series
    dataset.PixelData = b"x" * pixels
    return dataset


class PacsBridgeTests(unittest.TestCase):
    def test_enabled_config_requires_explicit_sender_allowlist(self):
        with self.assertRaises(PacsConfigurationError):
            PacsBridgeConfig.from_mapping({
                "pacs": {"enabled": True, "local": {"ae_title": "PANC_AI"},
                         "remote": {"ae_title": "PACS", "host": "10.0.0.1"}}
            })

    def test_cache_keeps_datasets_in_memory_and_take_frees_capacity(self):
        cache = InMemorySeriesCache(max_bytes=200_000, max_instances_per_series=2)
        self.assertTrue(cache.add(_ct(pixels=1024)))
        self.assertGreater(cache.used_bytes, 0)
        received = cache.take("1.2.3", "1.2.3.4")
        self.assertIsNotNone(received)
        self.assertEqual(len(received.instances), 1)
        self.assertEqual(cache.used_bytes, 0)

    def test_cache_rejects_series_instance_limit_without_disk_fallback(self):
        cache = InMemorySeriesCache(max_bytes=200_000, max_instances_per_series=1)
        self.assertTrue(cache.add(_ct(pixels=128)))
        self.assertFalse(cache.add(_ct(pixels=129)))
        self.assertEqual(cache.summary()[0]["instances"], 1)

    def test_cache_rejects_new_series_when_capacity_is_reached(self):
        cache = InMemorySeriesCache(max_bytes=130_000, max_instances_per_series=2)
        self.assertTrue(cache.add(_ct(study="1.2.3", series="1.2.3.1", pixels=60_000)))
        self.assertFalse(cache.add(_ct(study="1.2.4", series="1.2.4.1", pixels=60_000)))
        self.assertEqual([row["study_instance_uid"] for row in cache.summary()], ["1.2.3"])

    def test_store_handler_requires_registered_calling_ae(self):
        config = PacsBridgeConfig.from_mapping({
            "pacs": {
                "local": {"ae_title": "PANC_AI"},
                "remote": {"ae_title": "PACS", "host": "10.0.0.1"},
                "security": {"allowed_calling_aes": ["PACS"]},
            }
        })
        service = PacsStorageSCP(config)
        accepted_event = SimpleNamespace(
            assoc=SimpleNamespace(requestor=SimpleNamespace(ae_title=b"PACS ")),
            dataset=_ct(),
        )
        rejected_event = SimpleNamespace(
            assoc=SimpleNamespace(requestor=SimpleNamespace(ae_title="UNKNOWN")),
            dataset=_ct(),
        )
        self.assertEqual(service.handle_c_store(accepted_event), SUCCESS)
        self.assertEqual(service.handle_c_store(rejected_event), NOT_AUTHORIZED)

    def test_association_handler_rejects_unknown_ae_before_echo_or_store(self):
        config = PacsBridgeConfig.from_mapping({
            "pacs": {
                "local": {"ae_title": "PANC_AI"},
                "remote": {"ae_title": "PACS", "host": "10.0.0.1"},
                "security": {"allowed_calling_aes": ["PACS"]},
            }
        })
        reject = Mock()
        event = SimpleNamespace(
            assoc=SimpleNamespace(
                requestor=SimpleNamespace(ae_title="UNKNOWN"), reject=reject,
            )
        )
        PacsStorageSCP(config).handle_requested(event)
        reject.assert_called_once_with(0x01, 0x01, 0x03)


if __name__ == "__main__":
    unittest.main()
