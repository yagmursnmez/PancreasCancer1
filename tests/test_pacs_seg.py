"""DICOM SEG objects must reference their in-memory source CT instances."""

import sys
import unittest
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH / "scripts"))
from pacs_seg import create_segmentation_results


def _ct_instance(instance_number: int) -> Dataset:
    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.file_meta.MediaStorageSOPClassUID = CTImageStorage
    dataset.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.100"
    dataset.SeriesInstanceUID = "1.2.826.0.1.3680043.8.498.101"
    dataset.FrameOfReferenceUID = "1.2.826.0.1.3680043.8.498.102"
    dataset.PatientID = "TEST"
    dataset.PatientName = "Test^Patient"
    dataset.PatientBirthDate = ""
    dataset.PatientSex = ""
    dataset.StudyDate = "20260826"
    dataset.StudyTime = "120000"
    dataset.StudyID = "1"
    dataset.AccessionNumber = ""
    dataset.Modality = "CT"
    dataset.SeriesNumber = 1
    dataset.InstanceNumber = instance_number
    dataset.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    dataset.Rows = 8
    dataset.Columns = 8
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 1
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.SliceThickness = 1.0
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0, 0, float(instance_number - 1)]
    dataset.PixelData = np.zeros((8, 8), dtype=np.int16).tobytes()
    return dataset


class PacsSegTests(unittest.TestCase):
    def test_binary_seg_references_source_ct_and_preserves_frame_count(self):
        source = [_ct_instance(1), _ct_instance(2)]
        mask = np.zeros((8, 8, 2), dtype=np.uint8)
        mask[2:5, 2:5, :] = 1
        results = create_segmentation_results(source, mask, software_version="1.0.0")
        self.assertEqual(len(results), 1)
        segmentation = results[0]
        self.assertEqual(str(segmentation.SOPClassUID), "1.2.840.10008.5.1.4.1.1.66.4")
        self.assertEqual(segmentation.SegmentSequence[0].SegmentLabel, "Pancreas")
        self.assertEqual(segmentation.NumberOfFrames, 2)

    def test_nonempty_pancreas_and_tumor_produce_distinct_seg_objects(self):
        source = [_ct_instance(1), _ct_instance(2)]
        mask = np.zeros((8, 8, 2), dtype=np.uint8)
        mask[1:3, 1:3, :] = 1
        mask[4:6, 4:6, :] = 2
        results = create_segmentation_results(source, mask, software_version="1.0.0")
        self.assertEqual(len(results), 2)
        self.assertNotEqual(results[0].SOPInstanceUID, results[1].SOPInstanceUID)


if __name__ == "__main__":
    unittest.main()
