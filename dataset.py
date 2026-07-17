"""PyTorch datasets and preprocessing for CAD-derived point clouds."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# These defaults retain the existing local workflow, while environment
# variables allow training and conversion code to run on another machine.
_DATA_ROOT = Path(
    os.environ.get(
        "CAD_PART_DATA_ROOT",
        "/Volumes/MyProjects/02_AI/02_Datasets/Manufactured_Component_Data",
    )
)
DATA_STEP = str(_DATA_ROOT / "Data STEP")
DATA_STL = str(_DATA_ROOT / "Data STL")
DATA_XYZ = str(_DATA_ROOT / "Data XYZ")
DATA_VIEWS = str(_DATA_ROOT / "Data Views")


class PointCloudSample(TypedDict):
    """One item returned by :class:`PointCloudDataset`."""

    points: Tensor
    label: int
    part_id: str
    path: str


def load_xyz(path: str | Path) -> Tensor:
    """Read an XYZ text file and return a finite ``(N, 3)`` float tensor.

    Blank lines and lines beginning with ``#`` are accepted. Each data row
    must contain exactly three whitespace-separated coordinates.
    """
    xyz_path = Path(path)
    if not xyz_path.is_file():
        raise FileNotFoundError(f"Point-cloud file does not exist: {xyz_path}")

    try:
        values = np.loadtxt(xyz_path, dtype=np.float32, comments="#", ndmin=2)
    except ValueError as exc:
        raise ValueError(f"Could not parse XYZ point cloud: {xyz_path}") from exc

    if values.shape[0] == 0:
        raise ValueError(f"XYZ point cloud is empty: {xyz_path}")
    if values.shape[1] != 3:
        raise ValueError(
            f"XYZ point cloud must have three columns, found {values.shape[1]}: {xyz_path}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"XYZ point cloud contains non-finite coordinates: {xyz_path}")

    # Copy so the tensor owns writable, contiguous storage regardless of how
    # NumPy represented a very small input file.
    return torch.from_numpy(np.ascontiguousarray(values).copy())


def normalize_point_cloud(points: Tensor) -> Tensor:
    """Centre a point cloud and scale it to fit inside the unit sphere."""
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points must have shape (N, 3) with at least one point")

    centred = points - points.mean(dim=0, keepdim=True)
    radius = torch.linalg.vector_norm(centred, dim=1).max()
    if radius > 0:
        centred = centred / radius
    return centred


def sample_point_cloud(
    points: Tensor,
    number: int,
    *,
    random: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return exactly ``number`` points, repeating points when necessary."""
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points must have shape (N, 3) with at least one point")
    if number <= 0:
        raise ValueError("number must be greater than zero")

    point_count = points.shape[0]
    if random:
        if point_count >= number:
            indices = torch.randperm(point_count, generator=generator)[:number]
        else:
            extra = torch.randint(
                point_count, (number - point_count,), generator=generator
            )
            indices = torch.cat((torch.randperm(point_count, generator=generator), extra))
    else:
        # Even spacing avoids biasing deterministic evaluation toward the
        # beginning of files produced by a non-random exporter.
        indices = torch.arange(number) * point_count // number
    return points[indices]


class PointCloudDataset(Dataset[PointCloudSample]):
    """Load fixed-size point clouds from a directory of ``.xyz`` files.

    Files are sorted by their relative path for stable labels. A file stem is
    used as its ``part_id``; files with the same stem therefore share a label.

    Args:
        root: Directory containing XYZ files.
        number_of_points: Point count returned for every sample.
        recursive: Search below nested directories when true.
        normalize: Centre and scale each cloud to the unit sphere.
        random_sample: Randomize point selection on every access. Set false for
            deterministic validation and test datasets.
        transform: Optional callable applied to the sampled ``(N, 3)`` tensor.
            This is the extension point for rotations, jitter, and other data
            augmentation.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        number_of_points: int = 2048,
        recursive: bool = True,
        normalize: bool = True,
        random_sample: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"Point-cloud directory does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Point-cloud path is not a directory: {self.root}")
        if number_of_points <= 0:
            raise ValueError("number_of_points must be greater than zero")

        pattern = "**/*" if recursive else "*"
        self.files = sorted(
            (
                path
                for path in self.root.glob(pattern)
                if path.is_file() and path.suffix.lower() == ".xyz"
            ),
            key=lambda path: path.relative_to(self.root).as_posix().casefold(),
        )
        if not self.files:
            raise FileNotFoundError(f"No .xyz files found in: {self.root}")

        self.number_of_points = number_of_points
        self.normalize = normalize
        self.random_sample = random_sample
        self.transform = transform

        part_ids = sorted({path.stem for path in self.files})
        self.class_to_idx = {part_id: index for index, part_id in enumerate(part_ids)}
        self.idx_to_class = {index: part_id for part_id, index in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> PointCloudSample:
        path = self.files[index]
        points = load_xyz(path)
        if self.normalize:
            points = normalize_point_cloud(points)
        points = sample_point_cloud(
            points, self.number_of_points, random=self.random_sample
        )
        if self.transform is not None:
            points = self.transform(points)
        if points.shape != (self.number_of_points, 3):
            raise ValueError(
                "transform must preserve point-cloud shape "
                f"({self.number_of_points}, 3), received {tuple(points.shape)}"
            )

        part_id = path.stem
        return {
            # PointNet++ in pointnet2.py expects (channels, points) samples.
            "points": points.transpose(0, 1).contiguous(),
            "label": self.class_to_idx[part_id],
            "part_id": part_id,
            "path": str(path),
        }
