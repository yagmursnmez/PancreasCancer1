"""Create standards-based DICOM SEG objects from PancreasAI label masks.

The source CT instances stay in memory.  The result is a DICOM Segmentation
Storage object referencing those exact SOP instances; no RGB CT overlay is
used as a substitute for a segmentation result.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sr.coding import Code


def create_segmentation_results(
    source_images: Sequence[Dataset],
    label_mask: np.ndarray,
    *,
    software_version: str,
    manufacturer: str = "PancreasAI",
    device_serial_number: str = "PANC-AI-01",
) -> tuple[Dataset, ...]:
    """Return one binary DICOM SEG instance for each non-empty model label.

    ``label_mask`` must be in the project's NIfTI axis order ``(row, column,
    slice)`` and spatially match the source CT images.  A separate binary SEG
    per label provides broad PACS compatibility while preserving the required
    frame-to-source-image linkage.
    """
    try:
        import highdicom as hd
        from highdicom.seg import SegmentAlgorithmTypeValues, Segmentation, SegmentationTypeValues
        from pydicom.sr.codedict import codes
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("highdicom kurulu değil; DICOM SEG üretilemez.") from exc

    source_images = tuple(source_images)
    if not source_images:
        raise ValueError("DICOM SEG için kaynak CT kesitleri gerekli.")
    mask = np.asarray(label_mask, dtype=np.uint8)
    rows, columns = int(source_images[0].Rows), int(source_images[0].Columns)
    if mask.shape != (rows, columns, len(source_images)):
        raise ValueError(
            "Maske ile kaynak CT geometrisi uyuşmuyor: "
            f"{mask.shape} / {(rows, columns, len(source_images))}"
        )
    if any(
        int(image.Rows) != rows or int(image.Columns) != columns
        for image in source_images
    ):
        raise ValueError("Kaynak CT serisindeki kesit boyutları tutarsız.")

    algorithm = hd.AlgorithmIdentificationSequence(
        name="PancreasAI ensemble segmentation",
        family=codes.cid7162.ArtificialIntelligence,
        version=software_version,
        source="PancreasAI research workflow",
    )
    label_definitions = (
        # (project label, display label, coded category, coded type)
        (1, "Pancreas", codes.SCT.AnatomicalStructure, codes.SCT.Pancreas),
        (
            2,
            "Pancreatic tumor candidate",
            Code("49755003", "SCT", "Morphologically Abnormal Structure"),
            Code("108369006", "SCT", "Neoplasm"),
        ),
    )
    results: list[Dataset] = []
    base_series_number = int(getattr(source_images[0], "SeriesNumber", 0) or 0)
    for result_index, (label, description, category, property_type) in enumerate(label_definitions, start=1):
        binary = mask == label
        if not np.any(binary):
            continue
        segment = hd.seg.SegmentDescription(
            segment_number=1,
            segment_label=description,
            segmented_property_category=category,
            segmented_property_type=property_type,
            algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm,
        )
        # highdicom's 3D binary array convention is (source frame, row, column).
        frames = np.transpose(binary, (2, 0, 1))
        segmentation = Segmentation(
            source_images=source_images,
            pixel_array=frames,
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=[segment],
            series_instance_uid=hd.UID(),
            series_number=base_series_number + 700 + result_index,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer=manufacturer,
            manufacturer_model_name="PancreasAI ensemble",
            software_versions=software_version,
            device_serial_number=device_serial_number,
            series_description=f"PancreasAI {description}",
            content_label=("PANCREAS" if label == 1 else "PANC_TUMOR"),
            omit_empty_frames=True,
        )
        results.append(segmentation)
    if not results:
        raise ValueError("DICOM SEG için gönderilecek pankreas veya tümör maskesi yok.")
    return tuple(results)
