import sys
import tempfile
import unittest
import zipfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH / "scripts"))
sys.path.insert(0, str(BASE_PATH / "web"))

from segmentation_postprocess import (
    assess_pancreas_gate,
    extract_unverified_tumor_candidates,
    validate_and_fuse_segmentation,
)
from segmentation_measurements import measure_segmentation
from refine_with_pants import (
    component_supported_by,
    refine_with_consensus,
    screening_consensus_candidate,
)
from reconstruct_3d import extract_surface_mesh
import app as web_app
from app import (
    ANATOMICAL_GATE_CONFIG,
    _GPU_MODEL_EXPORT_ACTIVE,
    _GPU_MODEL_PROCESS_ACTIVE,
    _AnalysisProcessPriority,
    _build_dicom_nifti_affine,
    convert_dicom_to_nifti,
    _dicom_series_content_key,
    _gpu_environment,
    _model_gpu_prediction_finished,
    _nvidia_gpu_snapshot,
    _prepare_anatomical_roi,
    _prepare_dints_roi,
    _prepare_pants_roi,
    _probe_cuda_runtime,
    _run_anatomical_gate,
    _run_pants_refinement,
    _safe_extract_zip,
    _score_dicom_series,
    _validated_power_w,
)


class SegmentationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.shape = (48, 48, 48)
        self.spacing = (1.0, 1.0, 1.0)
        self.gate = np.zeros(self.shape, dtype=np.uint8)
        self.gate[12:34, 12:34, 12:34] = 1

    def test_empty_gate_fails_closed(self):
        raw = self.gate.copy()
        raw[18:28, 18:28, 18:22] = 2
        final, qc = validate_and_fuse_segmentation(
            raw, np.zeros_like(self.gate), self.spacing
        )
        self.assertFalse(np.any(final))
        self.assertIsNone(qc["has_tumor"])
        self.assertFalse(qc["pancreas_verified"])

    def test_corner_touching_islands_are_not_one_pancreas_surface(self):
        gate = np.zeros(self.shape, dtype=np.uint8)
        gate[8:24, 8:24, 8:24] = 1
        gate[24:32, 24:32, 24:32] = 1  # only one corner touches the main island
        largest, assessment = assess_pancreas_gate(
            gate, self.spacing, {"min_pancreas_ml": 0.1}
        )
        self.assertEqual(assessment["gate_components"], 2)
        self.assertEqual(int(np.count_nonzero(largest)), 16 * 16 * 16)

    def test_supported_multislice_tumor_is_kept(self):
        raw = self.gate.copy()
        raw[18:28, 18:28, 18:22] = 2  # 0.4 mL, dört kesit
        final, qc = validate_and_fuse_segmentation(raw, self.gate, self.spacing)
        self.assertTrue(qc["pancreas_verified"])
        self.assertTrue(qc["has_tumor"])
        self.assertEqual(qc["tumor_voxels"], 400)
        self.assertEqual(int(np.count_nonzero(final == 2)), 400)

    def test_two_model_tumor_consensus_is_kept(self):
        primary = self.gate.copy()
        secondary = self.gate.copy()
        primary[18:28, 18:28, 18:22] = 2
        secondary[20:30, 18:28, 18:22] = 2
        final, qc = validate_and_fuse_segmentation(
            primary, self.gate, self.spacing, secondary_mask=secondary
        )
        self.assertTrue(qc["has_tumor"])
        self.assertGreaterEqual(qc["tumor_cross_model_dice"], 0.1)
        self.assertGreaterEqual(qc["tumor_cross_model_overlap_ml"], 0.1)
        self.assertTrue(np.any(final == 2))

    def test_physically_adjacent_two_model_candidates_do_not_require_exact_voxels(self):
        primary = self.gate.copy()
        secondary = self.gate.copy()
        primary[14:24, 18:28, 18:22] = 2
        secondary[24:34, 18:28, 18:22] = 2
        final, qc = validate_and_fuse_segmentation(
            primary, self.gate, self.spacing, secondary_mask=secondary
        )
        self.assertEqual(qc["tumor_cross_model_overlap_voxels"], 0)
        self.assertGreaterEqual(qc["tumor_cross_model_proximity_dice"], 0.1)
        self.assertTrue(qc["has_tumor"])
        self.assertTrue(np.any(final == 2))

    def test_single_model_tumor_is_rejected_when_consensus_is_available(self):
        primary = self.gate.copy()
        secondary = self.gate.copy()
        primary[18:28, 18:28, 18:22] = 2
        _, qc = validate_and_fuse_segmentation(
            primary, self.gate, self.spacing, secondary_mask=secondary
        )
        self.assertIsNone(qc["has_tumor"])
        self.assertEqual(qc["status"], "indeterminate")
        self.assertGreater(qc["unverified_tumor_voxels"], 0)
        self.assertIn("tek_model_adayi", qc["tumor_components"][0]["rejection_reasons"])

    def test_meaningful_single_model_candidate_is_preserved_for_review(self):
        primary = self.gate.copy()
        secondary = self.gate.copy()
        primary[18:28, 18:28, 18:22] = 2
        candidate, records = extract_unverified_tumor_candidates(
            primary,
            self.gate,
            self.spacing,
            secondary_mask=secondary,
        )
        self.assertEqual(int(candidate.sum()), 400)
        self.assertTrue(any(
            record["meaningful_unverified_candidate"] for record in records
        ))

    def test_single_slice_tumor_is_rejected(self):
        raw = self.gate.copy()
        raw[13:33, 13:33, 20] = 2  # 0.4 mL fakat tek kesit
        final, qc = validate_and_fuse_segmentation(raw, self.gate, self.spacing)
        self.assertIsNone(qc["has_tumor"])
        self.assertEqual(qc["tumor_voxels"], 0)
        self.assertIn("kesit_esigi", qc["tumor_components"][0]["rejection_reasons"])
        self.assertFalse(np.any(final == 2))

    def test_tiny_single_slice_speck_can_still_be_negative(self):
        raw = self.gate.copy()
        raw[18:28, 18:28, 20] = 2  # 0.1 mL; belirsiz aday alt sınırının altında
        final, qc = validate_and_fuse_segmentation(raw, self.gate, self.spacing)
        self.assertFalse(qc["has_tumor"])
        self.assertEqual(qc["unverified_tumor_voxels"], 0)
        self.assertFalse(np.any(final == 2))

    def test_distant_tumor_is_rejected(self):
        raw = self.gate.copy()
        raw[0:10, 0:10, 0:5] = 2  # 0.5 mL, pankreastan uzak
        _, qc = validate_and_fuse_segmentation(
            raw, self.gate, self.spacing,
            {"tumor_support_radius_mm": 5.0, "min_tumor_support_fraction": 0.01},
        )
        self.assertFalse(qc["has_tumor"])
        self.assertEqual(qc["rejected_tumor_voxels"], 500)
        self.assertIn(
            "pankreas_komsulugu",
            qc["tumor_components"][0]["rejection_reasons"],
        )

    def test_low_cross_model_agreement_is_indeterminate(self):
        raw = np.zeros(self.shape, dtype=np.uint8)
        raw[0:8, 0:8, 0:8] = 1
        final, qc = validate_and_fuse_segmentation(raw, self.gate, self.spacing)
        self.assertIsNone(qc["has_tumor"])
        self.assertTrue(qc["pancreas_verified"])
        self.assertEqual(qc["status"], "pancreas_localized")
        self.assertTrue(np.any(final == 1))
        self.assertFalse(np.any(final == 2))

    def test_gate_can_be_assessed_before_tumor_models(self):
        largest, qc = assess_pancreas_gate(self.gate, self.spacing)
        self.assertTrue(qc["gate_plausible"])
        self.assertEqual(int(np.count_nonzero(largest)), 22 ** 3)

    def test_invalid_fast_gate_retries_at_full_resolution(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            executable = root / "TotalSegmentator.exe"
            executable.touch()
            ct_path = root / "ct.nii.gz"
            affine = np.eye(4)
            nib.save(nib.Nifti1Image(np.zeros((32, 32, 32), dtype=np.int16), affine), ct_path)
            config = {
                "enabled": True,
                "executable": str(executable),
                "home_dir": str(root / "home"),
                "device": "gpu",
                "fast": True,
                "retry_full_resolution_if_invalid": True,
                "timeout_seconds": 30,
                "expected_seconds": 1,
                "min_pancreas_ml": 3.0,
                "max_pancreas_ml": 250.0,
                "min_pancreas_slices": 3,
            }

            def write_gate(command, *_args, **_kwargs):
                output_dir = Path(command[command.index("-o") + 1])
                gate = np.zeros((32, 32, 32), dtype=np.uint8)
                if "--fast" in command:
                    gate[1, 1, 1] = 1
                    elapsed = 1.0
                else:
                    gate[4:20, 4:20, 4:20] = 1
                    elapsed = 2.0
                nib.save(nib.Nifti1Image(gate, affine), output_dir / "pancreas.nii.gz")
                return 0, elapsed

            with patch.object(web_app, "ANATOMICAL_GATE_CONFIG", config), \
                 patch.object(web_app, "_probe_cuda_runtime"), \
                 patch.object(web_app, "_run_logged_process", side_effect=write_gate):
                gate_path, error, elapsed = _run_anatomical_gate(
                    ct_path, root / "gate", root / "gate.log"
                )
            self.assertIsNone(error)
            self.assertEqual(gate_path.parent.name, "full_resolution")
            self.assertEqual(elapsed, 3.0)


class RoiAndReconstructionTests(unittest.TestCase):
    def test_pants_screening_candidate_requires_two_nearby_models_inside_pancreas(self):
        shape = (48, 48, 24)
        pancreas = np.zeros(shape, dtype=bool)
        pancreas[12:36, 12:36, 6:18] = True
        med = np.zeros(shape, dtype=np.float32)
        rsuper = np.zeros(shape, dtype=np.float32)
        med[20:25, 20:25, 10:14] = 0.9
        rsuper[23:28, 20:25, 10:14] = 0.9
        med[2:8, 2:8, 2:5] = 0.9
        rsuper[2:8, 2:8, 2:5] = 0.9
        candidate, metrics = screening_consensus_candidate(
            med, rsuper, pancreas, (1.0, 1.0, 2.0), match_radius_mm=4.0
        )
        self.assertTrue(np.any(candidate))
        self.assertFalse(np.any(candidate[2:8, 2:8, 2:5]))
        self.assertGreater(metrics["proximity_dice"], 0.0)
    def test_inference_cache_roundtrips_uncertainty_with_new_case_name(self):
        import nibabel as nib

        shape = (12, 13, 8)
        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            ct_path = root / "ct.nii.gz"
            source_mask = root / "source.nii.gz"
            source_uncertainty = root / "source_uncertainty.nii.gz"
            destination_mask = root / "renamed_case.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), ct_path)
            nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine), source_mask)
            uncertainty = np.zeros(shape, dtype=np.uint8)
            uncertainty[3:7, 4:8, 2:5] = 3
            nib.save(
                nib.Nifti1Image(uncertainty, affine), source_uncertainty
            )
            quality = {"uncertainty_mask_path": str(source_uncertainty)}
            with patch.object(web_app, "INFERENCE_CACHE_DIR", cache_dir):
                web_app._save_inference_cache("test-key", source_mask, quality)
                loaded, loaded_quality = web_app._load_inference_cache(
                    "test-key", ct_path, destination_mask
                )
            destination_uncertainty = root / "renamed_case_uncertainty.nii.gz"
            self.assertIsNotNone(loaded)
            self.assertTrue(destination_uncertainty.exists())
            self.assertEqual(
                loaded_quality["uncertainty_mask_path"],
                str(destination_uncertainty),
            )
            np.testing.assert_array_equal(
                np.asanyarray(nib.load(destination_uncertainty).dataobj),
                uncertainty,
            )

    def test_analysis_priority_is_restored(self):
        if sys.platform != "win32":
            self.skipTest("Windows process priority test")
        import psutil

        process = psutil.Process()
        original = process.nice()
        with _AnalysisProcessPriority():
            self.assertEqual(process.nice(), psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        self.assertEqual(process.nice(), original)

    def test_zip_streaming_extracts_with_path_safety(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            archive = base / "safe.zip"
            destination = base / "out"
            destination.mkdir()
            payload = b"dicom-data" * 10000
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("series/slice001.dcm", payload)
            with zipfile.ZipFile(archive, "r") as source:
                _safe_extract_zip(source, destination)
            self.assertEqual((destination / "series/slice001.dcm").read_bytes(), payload)

    def test_zip_streaming_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            archive = base / "unsafe.zip"
            destination = base / "out"
            destination.mkdir()
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.dcm", b"bad")
            with zipfile.ZipFile(archive, "r") as source:
                with self.assertRaises(ValueError):
                    _safe_extract_zip(source, destination)
            self.assertFalse((base / "escape.dcm").exists())

    def test_model_subprocesses_use_configured_nvidia_selector(self):
        environment = _gpu_environment({"CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(environment["CUDA_MODULE_LOADING"], "LAZY")
        self.assertEqual(
            environment["PYTORCH_CUDA_ALLOC_CONF"], "backend:cudaMallocAsync"
        )
        self.assertEqual(environment["SHIM_MCCOMPAT_ENABLE_GPU"], "1")

    def test_cuda_probe_accepts_selected_nvidia_cuda_device(self):
        info = _probe_cuda_runtime(sys.executable)
        self.assertEqual(info["device"], "cuda:0")
        self.assertTrue(info["name"])
        self.assertEqual(info["vendor"], "NVIDIA")
        self.assertEqual(info["physical_selector"], "0")

    def test_cuda_probe_does_not_require_a_specific_gpu_model_name(self):
        response = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "device": "cuda:0", "name": "NVIDIA L40S",
                "uuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "pci_bus_id": "", "memory_gb": 48.0,
                "torch": "2.7.1+cu118", "cuda": "11.8",
                "allocator": "native", "module_loading": "LAZY",
            }) + "\n",
            stderr="",
        )
        fake_python = BASE_PATH / "tmp" / "generic-nvidia-python.exe"
        with patch.object(web_app.subprocess, "run", return_value=response):
            info = _probe_cuda_runtime(fake_python)
        self.assertEqual(info["name"], "NVIDIA L40S")

    def test_nvidia_telemetry_is_visible_only_during_gpu_compute(self):
        _GPU_MODEL_PROCESS_ACTIVE.clear()
        _GPU_MODEL_EXPORT_ACTIVE.clear()
        self.assertIsNone(_nvidia_gpu_snapshot(force=True))
        try:
            _GPU_MODEL_PROCESS_ACTIVE.set()
            response = SimpleNamespace(
                returncode=0,
                stdout=(
                    "0, GPU-babda9de-6daa-e75d-3b9c-0100760dd034, "
                    "NVIDIA Test GPU, 87, 3345, 6144, "
                    "71, 108.4, P0\n"
                ),
            )
            with (
                patch.object(web_app, "_read_nvml_gpu_snapshot", return_value=None),
                patch.object(web_app.subprocess, "run", return_value=response),
            ):
                snapshot = _nvidia_gpu_snapshot(force=True)
            self.assertEqual(snapshot["gpu_index"], 0)
            self.assertEqual(snapshot["utilization_percent"], 87)
            self.assertEqual(snapshot["memory_used_mib"], 3345)
            self.assertEqual(snapshot["power_w"], 108.4)

            _GPU_MODEL_EXPORT_ACTIVE.set()
            self.assertIsNone(_nvidia_gpu_snapshot(force=True))
        finally:
            _GPU_MODEL_PROCESS_ACTIVE.clear()
            _GPU_MODEL_EXPORT_ACTIVE.clear()

    def test_nvml_telemetry_path_does_not_spawn_nvidia_smi(self):
        payload = {
            "gpu_index": 0,
            "gpu_uuid": "GPU-test",
            "cuda_device": "cuda:0",
            "name": "NVIDIA Test GPU",
            "utilization_percent": 73,
            "memory_used_mib": 4096,
            "memory_total_mib": 6144,
            "temperature_c": 66,
            "power_w": 101.0,
            "performance_state": "P0",
            "source": "NVIDIA NVML driver API",
        }
        with (
            patch.object(web_app, "_read_nvml_gpu_snapshot", return_value=payload),
            patch.object(
                web_app.subprocess, "run",
                side_effect=AssertionError("nvidia-smi should not be spawned"),
            ),
        ):
            snapshot = web_app._read_nvidia_gpu_snapshot()
        self.assertEqual(snapshot["source"], "NVIDIA NVML driver API")
        self.assertEqual(snapshot["utilization_percent"], 73)

    def test_gpu_audit_persists_pid_telemetry_and_stdout(self):
        with tempfile.TemporaryDirectory() as folder:
            audit_root = Path(folder) / "gpu_runs"
            source_log = Path(folder) / "model.log"
            source_log.write_text("model output\n", encoding="utf-8")
            with patch.object(web_app, "GPU_RUN_DIR", audit_root):
                audit_id = web_app._start_gpu_audit(
                    "test-job", "test-case", "a" * 64, True,
                    {"device": "cuda:0", "name": "NVIDIA Test GPU"},
                )
                stage_id = web_app._begin_gpu_stage(
                    audit_id, "test model", "CUDA test", ["python", "model.py"],
                    BASE_PATH, 1234, web_app._gpu_environment(),
                    {"utilization_percent": 0, "memory_used_mib": 10, "power_w": 12.0},
                    [{"pid": 99, "process_name": "existing.exe", "used_memory_mib": None}],
                )
                web_app._append_gpu_stage_sample(audit_id, stage_id, {
                    "sample": 1,
                    "gpu": {
                        "utilization_percent": 91, "memory_used_mib": 2048,
                        "power_w": 85.0,
                    },
                    "owned_cuda_processes": [
                        {"pid": 4321, "process_name": "python.exe", "used_memory_mib": None}
                    ],
                    "launcher_cuda_context": False,
                })
                summary = web_app._end_gpu_stage(
                    audit_id, stage_id, source_log, 0, 2.5, evidence_required=True
                )
                public = web_app._finish_gpu_audit(audit_id, "completed")

            self.assertTrue(summary["gpu_evidence"])
            self.assertEqual(summary["peak_utilization_percent"], 91)
            self.assertEqual(summary["observed_model_cuda_pids"], [4321])
            self.assertTrue(public["gpu_verified"])
            report = audit_root / audit_id / "gpu_run_report.json"
            self.assertTrue(report.exists())
            self.assertTrue((audit_root / audit_id / f"{stage_id}.stdout.log").exists())
            self.assertTrue((audit_root / audit_id / f"{stage_id}.telemetry.jsonl").exists())

    def test_gpu_audit_rejects_unowned_whole_device_activity(self):
        with tempfile.TemporaryDirectory() as folder:
            audit_root = Path(folder) / "gpu_runs"
            source_log = Path(folder) / "model.log"
            source_log.write_text("model output\n", encoding="utf-8")
            with patch.object(web_app, "GPU_RUN_DIR", audit_root):
                audit_id = web_app._start_gpu_audit(
                    "unowned-job", "test-case", "b" * 64, True,
                    {"device": "cuda:0", "name": "NVIDIA test GPU"},
                )
                stage_id = web_app._begin_gpu_stage(
                    audit_id, "test model", "CUDA test", ["python", "model.py"],
                    BASE_PATH, 1234, web_app._gpu_environment(),
                    {"utilization_percent": 0, "memory_used_mib": 10}, [],
                )
                web_app._append_gpu_stage_sample(audit_id, stage_id, {
                    "sample": 1,
                    "gpu": {"utilization_percent": 99, "memory_used_mib": 4096},
                    "owned_cuda_processes": [],
                    "unrelated_cuda_processes": [
                        {"pid": 9876, "process_name": "other.exe"}
                    ],
                    "launcher_cuda_context": False,
                })
                summary = web_app._end_gpu_stage(
                    audit_id, stage_id, source_log, 0, 2.5, evidence_required=True
                )
            self.assertFalse(summary["gpu_evidence"])
            self.assertEqual(summary["observed_model_cuda_pids"], [])

    def test_web_defaults_to_verified_cache_execution(self):
        response = web_app.app.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('name="execution_mode" value="cache_allowed" checked', html)
        self.assertIn('name="execution_mode" value="fresh_gpu"', html)
        self.assertIn('name="analysis_profile" value="balanced"', html)
        self.assertIn('name="analysis_profile" value="full_ensemble" checked', html)
        self.assertIn('SON AŞAMA ÖZETİ · CANLI DEĞİL', html)

    def test_analysis_profiles_are_bounded_and_cache_separated(self):
        name, settings = web_app._analysis_profile("balanced")
        self.assertEqual(name, "balanced")
        self.assertFalse(settings["nnunet_tta"])
        self.assertEqual(settings["dints_overlap"], 0.5)
        self.assertFalse(settings["pants_enabled"])
        with tempfile.TemporaryDirectory() as folder:
            ct_path = Path(folder) / "ct.nii.gz"
            checkpoint_path = Path(folder) / "checkpoint.pth"
            ct_path.write_bytes(b"ct")
            checkpoint_path.write_bytes(b"weights")
            balanced_key = web_app._inference_cache_key(
                ct_path, checkpoint_path, "balanced"
            )
            full_key = web_app._inference_cache_key(
                ct_path, checkpoint_path, "full_ensemble"
            )
        self.assertNotEqual(balanced_key, full_key)

    def test_implausible_idle_power_sample_is_discarded(self):
        self.assertIsNone(_validated_power_w(752.7, utilization_percent=0))
        self.assertEqual(_validated_power_w(108.4, utilization_percent=87), 108.4)

    def test_dicom_conversion_cache_key_uses_exact_file_content(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.dcm"
            second = Path(folder) / "second.dcm"
            first.write_bytes(b"same-dicom-bytes")
            second.write_bytes(b"same-dicom-bytes")
            key_a = _dicom_series_content_key([(first, None, 512, 512)], "uid")
            key_b = _dicom_series_content_key([(second, None, 512, 512)], "uid")
            second.write_bytes(b"different-dicom-bytes")
            key_c = _dicom_series_content_key([(second, None, 512, 512)], "uid")
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_nnUNet_gpu_and_cpu_export_phases_are_distinguished(self):
        with tempfile.TemporaryDirectory() as folder:
            log_path = Path(folder) / "nnunet.log"
            log_path.write_text("predicting 888/888\n", encoding="utf-8")
            self.assertFalse(_model_gpu_prediction_finished(log_path))
            log_path.write_text(
                "GPU prediction completed. Waiting for remaining "
                "segmentation exports to finish...\n",
                encoding="utf-8",
            )
            self.assertTrue(_model_gpu_prediction_finished(log_path))

    def test_pants_models_are_skipped_without_verified_tumor_seed(self):
        source = SimpleNamespace(shape=(12, 12, 12))
        seed = np.zeros(source.shape, dtype=np.uint8)
        seed[3:9, 3:9, 3:9] = 1
        # This isolates the explicitly configured fast-negative path. The
        # deployed full profile keeps screening_enabled=True and needs a real
        # NIfTI image (affine/header/data), not this minimal shape-only double.
        with patch.dict(web_app.PANTS_REFINEMENT_CONFIG, {"screening_enabled": False}):
            refined, uncertainty, quality, timings = _run_pants_refinement(
                source, seed, "negative_case", ""
            )
        np.testing.assert_array_equal(refined, seed)
        self.assertFalse(np.any(uncertainty))
        self.assertTrue(quality["pants_skipped"])
        self.assertEqual(quality["pants_models"], [])
        self.assertEqual(timings, {})

    def test_pants_roi_crops_all_axes_and_preserves_world_origin(self):
        import nibabel as nib

        data = np.zeros((100, 120, 80), dtype=np.int16)
        seed = np.zeros_like(data, dtype=np.uint8)
        seed[40:50, 55:65, 30:40] = 1
        affine = np.diag([2.0, 2.0, 3.0, 1.0])
        with tempfile.TemporaryDirectory() as folder:
            source_path = Path(folder) / "source.nii.gz"
            output_path = Path(folder) / "roi.nii.gz"
            nib.save(nib.Nifti1Image(data, affine), str(source_path))
            shape, slices = _prepare_pants_roi(
                nib.load(str(source_path)), seed, output_path, margin_mm=20.0,
                return_slices=True,
            )
            roi = nib.load(str(output_path))
            self.assertEqual(shape, (30, 30, 24))
            self.assertEqual(
                tuple((part.start, part.stop) for part in slices),
                ((30, 60), (45, 75), (23, 47)),
            )
            self.assertEqual(tuple(roi.shape), shape)
            np.testing.assert_allclose(roi.affine[:3, 3], [60.0, 90.0, 69.0])

    def test_pants_roi_consensus_matches_zero_padded_full_volume(self):
        shape = (60, 70, 50)
        base = np.zeros(shape, dtype=np.uint8)
        base[25:31, 30:36, 20:25] = 2
        med = np.zeros(shape, dtype=np.float32)
        rsuper = np.zeros(shape, dtype=np.float32)
        med[22:35, 27:40, 18:28] = 0.9
        rsuper[23:36, 28:41, 18:28] = 0.9
        pancreas = np.zeros(shape, dtype=bool)
        pancreas[18:42, 20:48, 14:34] = True
        full_mask, full_uncertainty, full_audit = refine_with_consensus(
            base, med, rsuper, pancreas, pancreas, (1.0, 1.0, 2.0),
            min_cross_model_dice=0.50, max_expansion_mm=5.0,
        )

        roi = (slice(12, 48), slice(14, 54), slice(8, 40))
        roi_mask, roi_uncertainty, roi_audit = refine_with_consensus(
            base[roi], med[roi], rsuper[roi], pancreas[roi], pancreas[roi],
            (1.0, 1.0, 2.0), min_cross_model_dice=0.50,
            max_expansion_mm=5.0,
        )
        reconstructed = base.copy()
        reconstructed[roi] = roi_mask
        reconstructed_uncertainty = np.zeros(shape, dtype=np.uint8)
        reconstructed_uncertainty[roi] = roi_uncertainty

        np.testing.assert_array_equal(reconstructed, full_mask)
        np.testing.assert_array_equal(reconstructed_uncertainty, full_uncertainty)
        self.assertEqual(roi_audit, full_audit)

    def test_pants_component_must_overlap_seed(self):
        mask = np.zeros((40, 40, 40), dtype=bool)
        seed = np.zeros_like(mask)
        mask[2:20, 2:20, 2:20] = True
        mask[30:35, 30:35, 30:35] = True
        seed[31:34, 31:34, 31:34] = True
        kept = component_supported_by(mask, seed)
        self.assertEqual(int(kept.sum()), 5 ** 3)
        self.assertTrue(np.all(kept[seed]))

    def test_pants_unilateral_prediction_cannot_expand_published_mask(self):
        shape = (40, 40, 20)
        base = np.zeros(shape, dtype=np.uint8)
        base[16:22, 16:22, 8:12] = 2
        med = np.zeros(shape, dtype=np.float32)
        rsuper = np.zeros(shape, dtype=np.float32)
        med[12:26, 12:26, 6:14] = 0.9
        pancreas = base > 0
        refined, uncertainty, audit = refine_with_consensus(
            base, med, rsuper, pancreas, pancreas, (1.0, 1.0, 2.0)
        )
        np.testing.assert_array_equal(refined == 2, base == 2)
        self.assertFalse(audit["expansion_allowed"])
        self.assertEqual(audit["added_tumor_voxels"], 0)
        self.assertTrue(np.any(uncertainty == 3))

    def test_pants_can_confirm_an_unverified_candidate_with_two_model_consensus(self):
        shape = (48, 48, 24)
        base = np.zeros(shape, dtype=np.uint8)
        base[12:36, 12:36, 6:18] = 1
        candidate = np.zeros(shape, dtype=bool)
        candidate[20:26, 20:26, 10:14] = True
        med = np.zeros(shape, dtype=np.float32)
        rsuper = np.zeros(shape, dtype=np.float32)
        med[18:29, 18:29, 9:15] = 0.9
        rsuper[19:30, 19:30, 9:15] = 0.9
        pancreas = base > 0
        refined, uncertainty, audit = refine_with_consensus(
            base,
            med,
            rsuper,
            pancreas,
            pancreas,
            (1.0, 1.0, 2.0),
            candidate_seed=candidate,
        )
        self.assertTrue(audit["candidate_arbitration"])
        self.assertTrue(audit["candidate_confirmed"])
        self.assertTrue(np.any(refined == 2))
        self.assertTrue(np.any(uncertainty > 0))

    def test_pants_keeps_unconfirmed_candidate_only_in_uncertainty(self):
        shape = (48, 48, 24)
        base = np.zeros(shape, dtype=np.uint8)
        base[12:36, 12:36, 6:18] = 1
        candidate = np.zeros(shape, dtype=bool)
        candidate[20:26, 20:26, 10:14] = True
        med = np.zeros(shape, dtype=np.float32)
        rsuper = np.zeros(shape, dtype=np.float32)
        med[18:29, 18:29, 9:15] = 0.9
        pancreas = base > 0
        refined, uncertainty, audit = refine_with_consensus(
            base,
            med,
            rsuper,
            pancreas,
            pancreas,
            (1.0, 1.0, 2.0),
            candidate_seed=candidate,
        )
        self.assertTrue(audit["candidate_arbitration"])
        self.assertFalse(audit["candidate_confirmed"])
        self.assertFalse(np.any(refined == 2))
        self.assertTrue(np.all(uncertainty[candidate] == 3))

    def test_pants_consensus_can_expand_only_near_verified_seed(self):
        shape = (50, 50, 30)
        base = np.zeros(shape, dtype=np.uint8)
        base[20:26, 20:26, 12:16] = 2
        med = np.zeros(shape, dtype=np.float32)
        rsuper = np.zeros(shape, dtype=np.float32)
        med[17:30, 17:30, 11:17] = 0.9
        rsuper[18:31, 18:31, 11:17] = 0.9
        med[2:8, 2:8, 2:5] = 0.9
        rsuper[2:8, 2:8, 2:5] = 0.9
        pancreas = base > 0
        refined, _, audit = refine_with_consensus(
            base, med, rsuper, pancreas, pancreas, (1.0, 1.0, 2.0),
            min_cross_model_dice=0.50, max_expansion_mm=3.0,
        )
        self.assertTrue(audit["expansion_allowed"])
        self.assertGreater(audit["added_tumor_voxels"], 0)
        self.assertFalse(np.any((refined == 2)[2:8, 2:8, 2:5]))

    def test_roi_keeps_full_axial_planes_and_updates_affine(self):
        import nibabel as nib

        data = np.zeros((32, 32, 300), dtype=np.int16)
        gate = np.zeros_like(data, dtype=np.uint8)
        gate[10:20, 10:20, 100:120] = 1
        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        old_margin = ANATOMICAL_GATE_CONFIG.get("roi_margin_mm")
        old_minimum = ANATOMICAL_GATE_CONFIG.get("roi_min_slices")
        ANATOMICAL_GATE_CONFIG["roi_margin_mm"] = 80.0
        ANATOMICAL_GATE_CONFIG["roi_min_slices"] = 96
        try:
            with tempfile.TemporaryDirectory() as folder:
                source_path = Path(folder) / "source.nii.gz"
                output_path = Path(folder) / "roi.nii.gz"
                nib.save(nib.Nifti1Image(data, affine), str(source_path))
                source_img = nib.load(str(source_path))
                crop, roi_shape = _prepare_anatomical_roi(source_img, gate, output_path)
                roi_img = nib.load(str(output_path))
                self.assertEqual(roi_shape, (32, 32, 100))
                self.assertEqual((crop[2].start, crop[2].stop), (60, 160))
                self.assertEqual(tuple(roi_img.shape), roi_shape)
                self.assertAlmostEqual(float(roi_img.affine[2, 3]), 120.0)
        finally:
            if old_margin is None:
                ANATOMICAL_GATE_CONFIG.pop("roi_margin_mm", None)
            else:
                ANATOMICAL_GATE_CONFIG["roi_margin_mm"] = old_margin
            if old_minimum is None:
                ANATOMICAL_GATE_CONFIG.pop("roi_min_slices", None)
            else:
                ANATOMICAL_GATE_CONFIG["roi_min_slices"] = old_minimum

    def test_dints_roi_crops_all_axes_and_preserves_world_origin(self):
        import nibabel as nib

        data = np.zeros((100, 120, 80), dtype=np.int16)
        gate = np.zeros_like(data, dtype=np.uint8)
        gate[40:50, 55:65, 30:40] = 1
        affine = np.diag([2.0, 2.0, 3.0, 1.0])
        old_enabled = web_app.TUMOR_MODEL_3D_CONFIG.get("roi_crop_enabled")
        old_margin = web_app.TUMOR_MODEL_3D_CONFIG.get("roi_margin_mm")
        web_app.TUMOR_MODEL_3D_CONFIG["roi_crop_enabled"] = True
        web_app.TUMOR_MODEL_3D_CONFIG["roi_margin_mm"] = 20.0
        try:
            with tempfile.TemporaryDirectory() as folder:
                source_path = Path(folder) / "source.nii.gz"
                output_path = Path(folder) / "dints_roi.nii.gz"
                nib.save(nib.Nifti1Image(data, affine), str(source_path))
                crop, roi_shape = _prepare_dints_roi(
                    nib.load(str(source_path)), gate, output_path
                )
                roi_img = nib.load(str(output_path))
                self.assertEqual(roi_shape, (30, 30, 24))
                self.assertEqual(
                    tuple((part.start, part.stop) for part in crop),
                    ((30, 60), (45, 75), (23, 47)),
                )
                self.assertEqual(tuple(roi_img.shape), roi_shape)
                np.testing.assert_allclose(roi_img.affine[:3, 3], [60.0, 90.0, 69.0])

                reconstructed = np.zeros(data.shape, dtype=np.uint8)
                reconstructed[crop] = np.ones(roi_shape, dtype=np.uint8)
                self.assertEqual(int(reconstructed.sum()), int(np.prod(roi_shape)))
                self.assertFalse(np.any(reconstructed[:30]))
        finally:
            if old_enabled is None:
                web_app.TUMOR_MODEL_3D_CONFIG.pop("roi_crop_enabled", None)
            else:
                web_app.TUMOR_MODEL_3D_CONFIG["roi_crop_enabled"] = old_enabled
            if old_margin is None:
                web_app.TUMOR_MODEL_3D_CONFIG.pop("roi_margin_mm", None)
            else:
                web_app.TUMOR_MODEL_3D_CONFIG["roi_margin_mm"] = old_margin

    def test_surface_mesh_preserves_full_volume_offset(self):
        mask = np.zeros((40, 40, 40), dtype=np.uint8)
        mask[20:26, 21:27, 22:28] = 1
        mesh = extract_surface_mesh(mask, 1, (2.0, 2.0, 2.0), smooth_iterations=0)
        self.assertIsNotNone(mesh)
        vertices, faces, _ = mesh
        self.assertGreater(vertices[:, 0].min(), 35.0)
        self.assertGreater(vertices[:, 1].min(), 35.0)
        self.assertGreater(vertices[:, 2].min(), 35.0)
        self.assertGreater(len(faces), 0)

    def test_surface_mesh_can_use_patient_affine(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[4:8, 5:9, 6:10] = 1
        affine = np.array([
            [0.0, -2.0, 0.0, 100.0],
            [-2.0, 0.0, 0.0, 200.0],
            [0.0, 0.0, 3.0, -50.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        mesh = extract_surface_mesh(
            mask, 1, (2.0, 2.0, 3.0), smooth_iterations=0, affine=affine
        )
        self.assertIsNotNone(mesh)
        vertices, _, _ = mesh
        self.assertLess(vertices[:, 0].max(), 100.0)
        self.assertLess(vertices[:, 1].max(), 200.0)
        self.assertGreater(vertices[:, 2].min(), -50.0)

    def test_measurements_use_ras_patient_coordinates(self):
        mask = np.zeros((20, 30, 10), dtype=np.uint8)
        mask[2:12, 4:24, 2:8] = 1
        mask[3:8, 5:15, 3:7] = 2
        affine = np.array([
            [0.0, -2.0, 0.0, 100.0],
            [-1.0, 0.0, 0.0, 200.0],
            [0.0, 0.0, 3.0, -50.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        measured = measure_segmentation(mask, affine)
        self.assertEqual(measured["tumor"]["dimensions_rl_ap_si_mm"], [20.0, 5.0, 12.0])
        self.assertIsNotNone(measured["estimated_location"])


class DicomGeometryTests(unittest.TestCase):
    def test_affine_maps_dicom_lps_to_nifti_ras(self):
        header = SimpleNamespace(
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[-211.685, -378.916, -1488.144],
            PixelSpacing=[0.807890625, 0.807890625],
        )
        affine = _build_dicom_nifti_affine(header, [0, 0, 1], 3.0)
        expected = np.array([
            [0, -0.807890625, 0, 211.685],
            [-0.807890625, 0, 0, 378.916],
            [0, 0, 3.0, -1488.144],
            [0, 0, 0, 1],
        ])
        np.testing.assert_allclose(affine, expected)

    def test_thorax_series_receives_negative_score(self):
        header = SimpleNamespace(
            Modality="CT",
            SOPClassUID="1.2.840.10008.5.1.4.1.1.2",
            SeriesDescription="TORAKS AKCIGER",
            BodyPartExamined="CHEST",
            ImageType=["ORIGINAL", "PRIMARY", "AXIAL"],
            ConvolutionKernel="B70",
            SliceThickness=3.0,
            ContrastBolusAgent="",
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[0, 0, 0],
        )
        headers = [
            (Path(f"slice_{index}.dcm"), SimpleNamespace(**{
                **header.__dict__, "ImagePositionPatient": [0, 0, index * 3]
            }), 512, 512)
            for index in range(50)
        ]
        score, _ = _score_dicom_series(headers)
        self.assertLess(score, 0)

    def test_identical_dicom_series_reuses_verified_nifti_conversion(self):
        import nibabel as nib
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            series_dir = root / "series"
            cache_dir = root / "cache"
            series_dir.mkdir()
            cache_dir.mkdir()
            series_uid = generate_uid()
            for index in range(16):
                sop_uid = generate_uid()
                meta = FileMetaDataset()
                meta.MediaStorageSOPClassUID = CTImageStorage
                meta.MediaStorageSOPInstanceUID = sop_uid
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                dataset = FileDataset(
                    str(series_dir / f"slice-{index:03d}.dcm"),
                    {}, file_meta=meta, preamble=b"\0" * 128,
                )
                dataset.SOPClassUID = CTImageStorage
                dataset.SOPInstanceUID = sop_uid
                dataset.SeriesInstanceUID = series_uid
                dataset.Modality = "CT"
                dataset.SeriesDescription = "abdomen venous"
                dataset.BodyPartExamined = "ABDOMEN"
                dataset.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
                dataset.Rows = 32
                dataset.Columns = 32
                dataset.SamplesPerPixel = 1
                dataset.PhotometricInterpretation = "MONOCHROME2"
                dataset.PixelRepresentation = 1
                dataset.BitsAllocated = 16
                dataset.BitsStored = 16
                dataset.HighBit = 15
                dataset.PixelSpacing = [1.0, 1.0]
                dataset.SliceThickness = 1.0
                dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                dataset.ImagePositionPatient = [0, 0, float(index)]
                dataset.InstanceNumber = index + 1
                dataset.RescaleSlope = 1.0
                dataset.RescaleIntercept = -1024.0
                pixels = np.full((32, 32), index + 1024, dtype=np.int16)
                dataset.PixelData = pixels.tobytes()
                dataset.save_as(dataset.filename, enforce_file_format=True)

            first_output = root / "first.nii.gz"
            second_output = root / "second.nii.gz"
            first_detail = {}
            second_detail = {}
            with patch.object(web_app, "DICOM_CONVERSION_CACHE_DIR", cache_dir):
                self.assertTrue(convert_dicom_to_nifti(
                    series_dir, first_output, error_detail=first_detail
                ))
                self.assertTrue(convert_dicom_to_nifti(
                    series_dir, second_output, error_detail=second_detail
                ))
            self.assertFalse(first_detail["cache_hit"])
            self.assertTrue(second_detail["cache_hit"])
            first = np.asanyarray(nib.load(str(first_output)).dataobj)
            second = np.asanyarray(nib.load(str(second_output)).dataobj)
            np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
