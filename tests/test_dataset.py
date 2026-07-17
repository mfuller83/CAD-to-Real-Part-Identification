import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PointCloudDataset, load_xyz, normalize_point_cloud


class PointCloudDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "part-b.xyz").write_text("0 0 0\n1 0 0\n0 1 0\n", encoding="utf-8")
        (self.root / "part-a.XYZ").write_text("0 0 0\n0 0 2\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_load_xyz_rejects_invalid_column_count(self) -> None:
        invalid = self.root / "invalid.xyz"
        invalid.write_text("1 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "three columns"):
            load_xyz(invalid)

    def test_normalization_centres_and_scales_points(self) -> None:
        points = normalize_point_cloud(load_xyz(self.root / "part-b.xyz"))
        torch.testing.assert_close(points.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)
        self.assertAlmostEqual(torch.linalg.vector_norm(points, dim=1).max().item(), 1.0)

    def test_dataset_has_stable_labels_and_pointnet_shape(self) -> None:
        dataset = PointCloudDataset(self.root, number_of_points=5, random_sample=False)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.class_to_idx, {"part-a": 0, "part-b": 1})
        sample = dataset[0]
        self.assertEqual(sample["points"].shape, (3, 5))
        self.assertEqual(sample["points"].dtype, torch.float32)
        self.assertEqual(sample["part_id"], "part-a")
        self.assertEqual(sample["label"], 0)

    def test_default_collation_creates_a_model_ready_batch(self) -> None:
        dataset = PointCloudDataset(self.root, number_of_points=4, random_sample=False)
        batch = next(iter(DataLoader(dataset, batch_size=2)))

        self.assertEqual(batch["points"].shape, (2, 3, 4))
        self.assertEqual(batch["label"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
