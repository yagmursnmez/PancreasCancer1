"""
dicom_export.py — Modüler Maskeli DICOM (2D & 3D) Export Modülü
PancreasAI Projesi için CT Görüntüsü ve Segmentasyon Maskesini Maskeli DICOM ve ZIP olarak dışa aktarır.
"""

import os
import shutil
import uuid
import zipfile
from pathlib import Path
import numpy as np

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


def create_overlay(ct_slice: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
    """
    CT kesitini ve segmentasyon maskesini (1=Pankreas Yeşil, 2=Tümör Kırmızı) harmanlar.
    
    Args:
        ct_slice: 2D numpy dizisi (H, W)
        mask_slice: 2D numpy dizisi (H, W)
        
    Returns:
        rgb_overlay: uint8 numpy dizisi (H, W, 3)
    """
    ct_min, ct_max = -150.0, 250.0
    ct_norm = np.clip(ct_slice, ct_min, ct_max)
    ct_norm = (ct_norm - ct_min) / (ct_max - ct_min) * 255.0
    ct_gray = ct_norm.astype(np.uint8)

    # 3 kanallı RGB matris oluştur (H, W, 3)
    rgb = np.stack([ct_gray, ct_gray, ct_gray], axis=-1).astype(np.float32)

    # Label 1: Pankreas (Yeşil: #27ae60 -> RGB: [39, 174, 96])
    pan_mask = (mask_slice == 1)
    if np.any(pan_mask):
        rgb[pan_mask, 0] = rgb[pan_mask, 0] * 0.5 + 39 * 0.5
        rgb[pan_mask, 1] = rgb[pan_mask, 1] * 0.5 + 174 * 0.5
        rgb[pan_mask, 2] = rgb[pan_mask, 2] * 0.5 + 96 * 0.5

    # Label 2: Tümör (Kırmızı: #e74c3c -> RGB: [231, 76, 60])
    tum_mask = (mask_slice == 2)
    if np.any(tum_mask):
        rgb[tum_mask, 0] = rgb[tum_mask, 0] * 0.4 + 231 * 0.6
        rgb[tum_mask, 1] = rgb[tum_mask, 1] * 0.4 + 76 * 0.6
        rgb[tum_mask, 2] = rgb[tum_mask, 2] * 0.4 + 60 * 0.6

    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_dicom_slice(
    image_rgb: np.ndarray,
    metadata: dict,
    save_path: Path,
    instance_num: int = 1,
    study_uid: str = None,
    series_uid: str = None
) -> bool:
    """
    RGB renkli kesit görüntüsünü DICOM formatında (.dcm) kaydeder.
    """
    if not PYDICOM_AVAILABLE:
        print("[DICOM Export Error]: pydicom kütüphanesi kurulu değil.")
        return False

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        h, w, _ = image_rgb.shape

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(save_path), {}, file_meta=file_meta, preamble=b'\0'*128)
        ds.PatientName = metadata.get("PatientName", "Anonymous^Patient")
        ds.PatientID = metadata.get("PatientID", "PancreasAI_Case")
        ds.Modality = "CT"
        ds.StudyInstanceUID = study_uid or generate_uid()
        ds.SeriesInstanceUID = series_uid or generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

        ds.Rows = h
        ds.Columns = w
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.InstanceNumber = instance_num

        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.PixelData = image_rgb.tobytes()

        ds.save_as(str(save_path))
        return True
    except Exception as e:
        print(f"[DICOM Save Error]: {e}")
        return False


def export_2d_dicom(
    ct_volume: np.ndarray,
    mask_volume: np.ndarray,
    metadata: dict,
    out_dir: Path,
    progress_callback=None,
    progress_offset: int = 0,
    total_steps: int = None,
) -> list:
    """
    Her kesiti bağımsız 2D DICOM (.dcm) dosyaları olarak '2D/' klasörüne kaydeder.
    """
    out_2d_dir = out_dir / "2D"
    out_2d_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    depth = ct_volume.shape[2] if ct_volume.ndim == 3 else 1
    study_uid = metadata.get("StudyInstanceUID") or generate_uid()

    for z in range(depth):
        ct_slice = ct_volume[:, :, z] if ct_volume.ndim == 3 else ct_volume
        mask_slice = mask_volume[:, :, z] if mask_volume.ndim == 3 else mask_volume

        rgb_overlay = create_overlay(ct_slice, mask_slice)
        slice_filename = out_2d_dir / f"slice_{z+1:04d}.dcm"

        # 2D için her slice bağımsız seri UID alır
        ok = save_dicom_slice(
            rgb_overlay, metadata, slice_filename,
            instance_num=z+1, study_uid=study_uid, series_uid=generate_uid()
        )
        if ok:
            saved_files.append(slice_filename)
        if progress_callback and (z % 25 == 0 or z + 1 == depth):
            progress_callback(
                progress_offset + z + 1,
                total_steps or depth,
                f"2D DICOM {z + 1}/{depth}",
            )

    return saved_files


def export_3d_dicom(
    ct_volume: np.ndarray,
    mask_volume: np.ndarray,
    metadata: dict,
    out_dir: Path,
    progress_callback=None,
    progress_offset: int = 0,
    total_steps: int = None,
) -> list:
    """
    Tüm hacmi tek bir 3D DICOM serisi olarak '3D/' klasörüne kaydeder.
    """
    out_3d_dir = out_dir / "3D"
    out_3d_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    depth = ct_volume.shape[2] if ct_volume.ndim == 3 else 1
    study_uid = metadata.get("StudyInstanceUID") or generate_uid()
    series_uid = metadata.get("SeriesInstanceUID") or generate_uid()

    for z in range(depth):
        ct_slice = ct_volume[:, :, z] if ct_volume.ndim == 3 else ct_volume
        mask_slice = mask_volume[:, :, z] if mask_volume.ndim == 3 else mask_volume

        rgb_overlay = create_overlay(ct_slice, mask_slice)
        slice_filename = out_3d_dir / f"volume_slice_{z+1:04d}.dcm"

        # 3D seri için tüm slice'lar AYNI series_uid paylaşır
        ok = save_dicom_slice(
            rgb_overlay, metadata, slice_filename,
            instance_num=z+1, study_uid=study_uid, series_uid=series_uid
        )
        if ok:
            saved_files.append(slice_filename)
        if progress_callback and (z % 25 == 0 or z + 1 == depth):
            progress_callback(
                progress_offset + z + 1,
                total_steps or depth,
                f"3D DICOM {z + 1}/{depth}",
            )

    return saved_files


def create_zip(output_folder: Path, zip_path: Path) -> bool:
    """
    Klasördeki 2D ve 3D DICOM dosyalarını tek bir ZIP arşivi halinde paketler.
    """
    try:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_folder):
                for file in files:
                    file_p = Path(root) / file
                    rel_p = file_p.relative_to(output_folder)
                    zipf.write(file_p, arcname=str(rel_p))
        return zip_path.exists()
    except Exception as e:
        print(f"[ZIP Creation Error]: {e}")
        return False


def export_full_dicom_package(
    ct_data: np.ndarray,
    mask_data: np.ndarray,
    case_id: str,
    output_zip_path: Path,
    progress_callback=None,
) -> bool:
    """
    2D + 3D Maskeli DICOM dosyalarını oluşturup tek ZIP paketi haline getiren ana fonksiyon.
    """
    tmp_export_dir = output_zip_path.parent / f"export_tmp_{uuid.uuid4().hex[:8]}"
    try:
        tmp_export_dir.mkdir(parents=True, exist_ok=True)

        # Matris eksen normalizasyonu (H, W, Depth)
        ct_data = np.squeeze(ct_data)
        mask_data = np.squeeze(mask_data)

        if ct_data.ndim == 2:
            ct_data = np.expand_dims(ct_data, axis=-1)
        if mask_data.ndim == 2:
            mask_data = np.expand_dims(mask_data, axis=-1)

        s_ct = ct_data.shape
        if len(s_ct) == 3 and s_ct[0] < s_ct[1] and s_ct[0] < s_ct[2]:
            ct_data = np.transpose(ct_data, (1, 2, 0))
            mask_data = np.transpose(mask_data, (1, 2, 0))

        metadata = {
            "PatientID": case_id,
            "PatientName": f"Case^{case_id}",
            "StudyInstanceUID": generate_uid(),
            "SeriesInstanceUID": generate_uid()
        }

        # 2D ve 3D DICOM aktarımı yap
        depth = ct_data.shape[2] if ct_data.ndim == 3 else 1
        total_steps = depth * 2 + 1
        export_2d_dicom(
            ct_data, mask_data, metadata, tmp_export_dir,
            progress_callback=progress_callback,
            progress_offset=0,
            total_steps=total_steps,
        )
        export_3d_dicom(
            ct_data, mask_data, metadata, tmp_export_dir,
            progress_callback=progress_callback,
            progress_offset=depth,
            total_steps=total_steps,
        )

        # ZIP paketini oluştur
        success = create_zip(tmp_export_dir, output_zip_path)
        if progress_callback:
            progress_callback(total_steps, total_steps, "DICOM ZIP paketi tamamlandı")
        return success
    finally:
        shutil.rmtree(tmp_export_dir, ignore_errors=True)
