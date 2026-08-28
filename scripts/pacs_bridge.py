"""Memory-only DICOM network bridge for the PancreasAI workstation.

This module is deliberately independent from ``web/app.py`` and the inference
models.  It accepts DICOM C-STORE requests into a bounded RAM cache and offers
a standards-based C-STORE client for already-created result objects.  It never
writes received DICOM instances to disk.

Run ``python scripts/pacs_bridge.py --config pacs_config.json --validate``
before enabling the listener.  A listener is never started unless the separate
PACS configuration explicitly has ``enabled: true``.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from pydicom.dataset import Dataset
    from pydicom.uid import CTImageStorage, UID
except ImportError as exc:  # pragma: no cover - caught at command start
    raise RuntimeError("pydicom is required for the PACS bridge") from exc


STORAGE_OUT_OF_RESOURCES = 0xA700
NOT_AUTHORIZED = 0x0124
SOP_CLASS_NOT_SUPPORTED = 0x0122
SUCCESS = 0x0000
VERIFICATION_SOP_CLASS = UID("1.2.840.10008.1.1")


class PacsConfigurationError(ValueError):
    """Raised when the separately managed PACS configuration is unsafe."""


def _ae_title(value: Any, field: str) -> str:
    title = str(value or "").strip().upper()
    if not title or len(title) > 16 or not title.isascii():
        raise PacsConfigurationError(
            f"{field} 1-16 karakterlik ASCII bir DICOM AE Title olmalıdır."
        )
    return title


def _positive_int(value: Any, field: str, minimum: int = 1) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PacsConfigurationError(f"{field} bir tam sayı olmalıdır.") from exc
    if result < minimum:
        raise PacsConfigurationError(f"{field} en az {minimum} olmalıdır.")
    return result


@dataclass(frozen=True)
class PacsBridgeConfig:
    enabled: bool
    local_ae_title: str
    bind_host: str
    listen_port: int
    remote_ae_title: str
    remote_host: str
    remote_port: int
    allowed_calling_aes: frozenset[str]
    max_cache_mb: int
    max_instances_per_series: int
    accept_ct_only: bool

    @property
    def max_cache_bytes(self) -> int:
        return self.max_cache_mb * 1024 * 1024

    @classmethod
    def from_mapping(cls, source: dict[str, Any]) -> "PacsBridgeConfig":
        pacs = source.get("pacs", source)
        local = pacs.get("local", {})
        remote = pacs.get("remote", {})
        security = pacs.get("security", {})
        cache = pacs.get("cache", {})
        allowed = frozenset(
            _ae_title(item, "security.allowed_calling_aes")
            for item in security.get("allowed_calling_aes", [])
        )
        enabled = bool(pacs.get("enabled", False))
        if enabled and not allowed:
            raise PacsConfigurationError(
                "Canlı dinleme için security.allowed_calling_aes boş bırakılamaz."
            )
        bind_host = str(local.get("bind_host", "0.0.0.0")).strip()
        remote_host = str(remote.get("host", "")).strip()
        if not bind_host:
            raise PacsConfigurationError("local.bind_host boş olamaz.")
        if not remote_host:
            raise PacsConfigurationError("remote.host boş olamaz.")
        return cls(
            enabled=enabled,
            local_ae_title=_ae_title(local.get("ae_title"), "local.ae_title"),
            bind_host=bind_host,
            listen_port=_positive_int(local.get("port", 11112), "local.port"),
            remote_ae_title=_ae_title(remote.get("ae_title"), "remote.ae_title"),
            remote_host=remote_host,
            remote_port=_positive_int(remote.get("port", 104), "remote.port"),
            allowed_calling_aes=allowed,
            max_cache_mb=_positive_int(cache.get("max_cache_mb", 2048), "cache.max_cache_mb"),
            max_instances_per_series=_positive_int(
                cache.get("max_instances_per_series", 2000),
                "cache.max_instances_per_series",
            ),
            accept_ct_only=bool(pacs.get("accept_ct_only", True)),
        )


@dataclass(frozen=True)
class CachedSeries:
    study_instance_uid: str
    series_instance_uid: str
    instances: tuple[Dataset, ...]


class InMemorySeriesCache:
    """Bounded, process-local DICOM cache.  It has no filesystem fallback."""

    def __init__(self, max_bytes: int, max_instances_per_series: int):
        self.max_bytes = _positive_int(max_bytes, "max_bytes")
        self.max_instances_per_series = _positive_int(
            max_instances_per_series, "max_instances_per_series"
        )
        self._items: OrderedDict[tuple[str, str], list[Dataset]] = OrderedDict()
        self._sizes: dict[tuple[str, str], int] = {}
        self._used_bytes = 0
        self._lock = threading.RLock()

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    def add(self, dataset: Dataset) -> bool:
        """Add one instance, returning False rather than spilling PHI to disk."""
        key = self._series_key(dataset)
        size = self._dataset_size(dataset)
        if size > self.max_bytes:
            return False
        with self._lock:
            instances = self._items.get(key)
            if instances is None:
                if self._used_bytes + size > self.max_bytes:
                    return False
                instances = []
                self._items[key] = instances
            if len(instances) >= self.max_instances_per_series:
                return False
            if self._used_bytes + size > self.max_bytes:
                return False
            instances.append(dataset.copy())
            self._sizes[key] = self._sizes.get(key, 0) + size
            self._used_bytes += size
            self._items.move_to_end(key)
            return True

    def take(self, study_instance_uid: str, series_instance_uid: str) -> CachedSeries | None:
        """Atomically remove a received series for processing, freeing its RAM."""
        key = (str(study_instance_uid), str(series_instance_uid))
        with self._lock:
            instances = self._items.pop(key, None)
            if instances is None:
                return None
            self._used_bytes -= self._sizes.pop(key, 0)
            return CachedSeries(key[0], key[1], tuple(instances))

    def summary(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"study_instance_uid": key[0], "series_instance_uid": key[1],
                 "instances": len(instances), "bytes": self._sizes[key]}
                for key, instances in self._items.items()
            ]

    @staticmethod
    def _series_key(dataset: Dataset) -> tuple[str, str]:
        study = str(getattr(dataset, "StudyInstanceUID", "")).strip()
        series = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
        if not study or not series:
            raise ValueError("C-STORE nesnesinde StudyInstanceUID ve SeriesInstanceUID zorunludur.")
        return study, series

    @staticmethod
    def _dataset_size(dataset: Dataset) -> int:
        # PixelData dominates a CT instance.  The fixed allowance accounts for
        # headers without serialising or creating a temporary DICOM file.
        pixel_data = getattr(dataset, "PixelData", b"")
        return len(pixel_data) + 64 * 1024


class PacsResultSender:
    """Send already-valid DICOM result objects back to the configured PACS."""

    def __init__(self, config: PacsBridgeConfig):
        self.config = config

    def send(self, datasets: Iterable[Dataset]) -> None:
        pynetdicom = _require_pynetdicom()
        datasets = tuple(datasets)
        if not datasets:
            raise ValueError("PACS'e gönderilecek en az bir DICOM sonuç nesnesi gerekli.")
        ae = pynetdicom.AE(ae_title=self.config.local_ae_title)
        for dataset in datasets:
            sop_class = getattr(dataset, "SOPClassUID", None)
            sop_instance = getattr(dataset, "SOPInstanceUID", None)
            if not sop_class or not sop_instance:
                raise ValueError("Sonuç nesnesinde SOPClassUID ve SOPInstanceUID zorunludur.")
            ae.add_requested_context(sop_class)
        association = ae.associate(
            self.config.remote_host,
            self.config.remote_port,
            ae_title=self.config.remote_ae_title,
        )
        if not association.is_established:
            raise ConnectionError("PACS ile sonuç C-STORE bağlantısı kurulamadı.")
        try:
            for dataset in datasets:
                status = association.send_c_store(dataset)
                if not status or int(status.Status) != SUCCESS:
                    code = getattr(status, "Status", None)
                    raise RuntimeError(f"PACS C-STORE sonucu kabul etmedi (status={code!r}).")
        finally:
            association.release()

    def send_segmentation(
        self, source_images: Iterable[Dataset], label_mask, *, software_version: str
    ) -> None:
        """Encode the model labels as DICOM SEG, then C-STORE them to PACS."""
        from pacs_seg import create_segmentation_results

        results = create_segmentation_results(
            tuple(source_images), label_mask, software_version=software_version
        )
        self.send(results)


class PacsStorageSCP:
    """C-ECHO + CT C-STORE SCP, backed only by :class:`InMemorySeriesCache`."""

    def __init__(self, config: PacsBridgeConfig, cache: InMemorySeriesCache | None = None):
        self.config = config
        self.cache = cache or InMemorySeriesCache(
            config.max_cache_bytes, config.max_instances_per_series
        )

    def _calling_ae_title(self, event: Any) -> str:
        raw_calling_ae = event.assoc.requestor.ae_title
        if isinstance(raw_calling_ae, bytes):
            raw_calling_ae = raw_calling_ae.decode("ascii", errors="ignore")
        return str(raw_calling_ae).strip().upper()

    def handle_requested(self, event: Any) -> None:
        """Reject unregistered peers before they can use C-ECHO or C-STORE."""
        if self._calling_ae_title(event) not in self.config.allowed_calling_aes:
            event.assoc.reject(0x01, 0x01, 0x03)

    def handle_c_store(self, event: Any) -> int:
        calling_ae = self._calling_ae_title(event)
        if calling_ae not in self.config.allowed_calling_aes:
            return NOT_AUTHORIZED
        dataset = event.dataset
        # highdicom needs the original transfer syntax metadata when it later
        # creates a SEG referencing this source instance. Keep it in RAM with
        # the received dataset; do not serialize the source to a .dcm file.
        if getattr(event, "file_meta", None) is not None:
            dataset.file_meta = event.file_meta
        if self.config.accept_ct_only and str(getattr(dataset, "SOPClassUID", "")) != str(CTImageStorage):
            return SOP_CLASS_NOT_SUPPORTED
        try:
            return SUCCESS if self.cache.add(dataset) else STORAGE_OUT_OF_RESOURCES
        except (TypeError, ValueError):
            return STORAGE_OUT_OF_RESOURCES

    def start(self, *, block: bool = False):
        if not self.config.enabled:
            raise PacsConfigurationError("pacs.enabled true değil; dinleyici başlatılmadı.")
        pynetdicom = _require_pynetdicom()
        ae = pynetdicom.AE(ae_title=self.config.local_ae_title)
        ae.add_supported_context(VERIFICATION_SOP_CLASS)
        ae.add_supported_context(CTImageStorage)
        return ae.start_server(
            (self.config.bind_host, self.config.listen_port),
            block=block,
            evt_handlers=[
                (pynetdicom.evt.EVT_REQUESTED, self.handle_requested),
                (pynetdicom.evt.EVT_C_STORE, self.handle_c_store),
            ],
        )

    def serve_forever(self) -> None:
        self.start(block=True)


def _require_pynetdicom():
    try:
        import pynetdicom
    except ImportError as exc:
        raise RuntimeError(
            "pynetdicom kurulu değil. requirements.txt güncellendikten sonra "
            "venv içinde `python -m pip install -r requirements.txt` çalıştırın."
        ) from exc
    return pynetdicom


def load_config(path: Path) -> PacsBridgeConfig:
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PacsConfigurationError(f"PACS yapılandırması okunamadı: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PacsConfigurationError(f"PACS yapılandırması geçerli JSON değil: {exc}") from exc
    return PacsBridgeConfig.from_mapping(source)


def registration_summary(config: PacsBridgeConfig) -> str:
    allowed = ", ".join(sorted(config.allowed_calling_aes)) or "(yok)"
    return (
        "PACS AE kaydı için:\n"
        f"  Cihaz / AE Title : {config.local_ae_title}\n"
        f"  IP / Host        : {config.bind_host} (sunucunun gerçek sabit IP'sini girin)\n"
        f"  Port             : {config.listen_port}\n"
        "  Rol              : Verification SCP + CT Storage SCP\n"
        f"  Kabul edilen gönderen AE'ler: {allowed}\n"
        f"  RAM sınırı       : {config.max_cache_mb} MB (kalıcı DICOM depolaması yok)\n"
        f"  Sonuç hedefi     : {config.remote_ae_title}@{config.remote_host}:{config.remote_port}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="PancreasAI bellek-içi PACS köprüsü")
    parser.add_argument("--config", type=Path, required=True, help="PACS JSON yapılandırması")
    parser.add_argument("--validate", action="store_true", help="Yalnız yapılandırmayı doğrula")
    parser.add_argument("--serve", action="store_true", help="C-ECHO/C-STORE SCP'yi başlat")
    args = parser.parse_args()
    if args.validate == args.serve:
        parser.error("Tam olarak biri gerekli: --validate veya --serve")
    try:
        config = load_config(args.config)
        print(registration_summary(config))
        if args.serve:
            PacsStorageSCP(config).serve_forever()
    except (PacsConfigurationError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
