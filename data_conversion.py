"""Utilities for converting CAD mesh files to point clouds."""

from __future__ import annotations

import math
import random
import struct
from pathlib import Path
from typing import Iterable, Sequence

Point = tuple[float, float, float]
Triangle = tuple[Point, Point, Point]


def convert_stl_to_xyz(
    input_folder: str | Path,
    output_folder: str | Path,
    *,
    points_per_cloud: int = 10_000,
    seed: int | None = None,
) -> list[Path]:
    """Convert every STL file in a folder into a sampled XYZ point cloud.

    STL files are read from ``input_folder`` (non-recursively). One ``.xyz``
    file with the same stem is written to ``output_folder`` for each mesh.
    Points are sampled uniformly across the mesh surface, so large triangles
    contribute proportionally more points than small triangles.

    Args:
        input_folder: Folder containing ASCII or binary ``.stl`` files.
        output_folder: Folder in which the ``.xyz`` files will be created.
        points_per_cloud: Number of surface points to write for each mesh.
        seed: Optional random seed for reproducible point clouds.

    Returns:
        Paths of the generated ``.xyz`` files.

    Raises:
        FileNotFoundError: If ``input_folder`` does not exist.
        NotADirectoryError: If ``input_folder`` is not a directory.
        ValueError: If the point count or an STL mesh is invalid.
    """
    source_dir = Path(input_folder)
    destination_dir = Path(output_folder)

    if not source_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {source_dir}")
    if points_per_cloud <= 0:
        raise ValueError("points_per_cloud must be greater than zero")

    stl_files = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".stl"
    )
    destination_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    converted_files: list[Path] = []
    for stl_path in stl_files:
        triangles = _read_stl(stl_path)
        points = _sample_mesh_surface(triangles, points_per_cloud, rng)
        xyz_path = destination_dir / f"{stl_path.stem}.xyz"
        _write_xyz(xyz_path, points)
        converted_files.append(xyz_path)

    return converted_files


# More explicit alias for callers that want the folder-based behavior in the
# function name.
convert_stl_folder_to_xyz = convert_stl_to_xyz


def _read_stl(path: Path) -> list[Triangle]:
    """Read either binary or ASCII STL data from ``path``."""
    data = path.read_bytes()

    # A binary STL is 84 bytes of header/count followed by 50 bytes per face.
    # Checking the expected size avoids misclassifying binary files whose
    # headers happen to begin with the word "solid".
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + triangle_count * 50:
            return _read_binary_stl(data, triangle_count, path)

    return _read_ascii_stl(data, path)


def _read_binary_stl(data: bytes, triangle_count: int, path: Path) -> list[Triangle]:
    triangles: list[Triangle] = []
    for index in range(triangle_count):
        offset = 84 + index * 50
        values = struct.unpack_from("<12f", data, offset)
        triangles.append(
            (
                (values[3], values[4], values[5]),
                (values[6], values[7], values[8]),
                (values[9], values[10], values[11]),
            )
        )

    if not triangles:
        raise ValueError(f"STL file contains no triangles: {path}")
    return triangles


def _read_ascii_stl(data: bytes, path: Path) -> list[Triangle]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid or unsupported STL file: {path}") from exc

    vertices: list[Point] = []
    for line in text.splitlines():
        fields = line.strip().split()
        if not fields or fields[0].lower() != "vertex":
            continue
        if len(fields) != 4:
            raise ValueError(f"Malformed vertex in ASCII STL file: {path}")
        try:
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        except ValueError as exc:
            raise ValueError(f"Malformed vertex in ASCII STL file: {path}") from exc

    if not vertices or len(vertices) % 3:
        raise ValueError(f"STL file does not contain complete triangles: {path}")

    return [tuple(vertices[index : index + 3]) for index in range(0, len(vertices), 3)]  # type: ignore[list-item]


def _triangle_area(triangle: Triangle) -> float:
    a, b, c = triangle
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(sum(component * component for component in cross))


def _sample_mesh_surface(
    triangles: Sequence[Triangle], point_count: int, rng: random.Random
) -> Iterable[Point]:
    areas = [_triangle_area(triangle) for triangle in triangles]
    total_area = sum(areas)
    if total_area <= 0:
        raise ValueError("STL mesh has no non-degenerate surface triangles")

    # random.choices uses the face areas as sampling weights. The square-root
    # barycentric transform then distributes each point uniformly on its face.
    sampled_triangles = rng.choices(triangles, weights=areas, k=point_count)
    for a, b, c in sampled_triangles:
        root_r1 = math.sqrt(rng.random())
        r2 = rng.random()
        weight_a = 1.0 - root_r1
        weight_b = root_r1 * (1.0 - r2)
        weight_c = root_r1 * r2
        yield (
            weight_a * a[0] + weight_b * b[0] + weight_c * c[0],
            weight_a * a[1] + weight_b * b[1] + weight_c * c[1],
            weight_a * a[2] + weight_b * b[2] + weight_c * c[2],
        )


def _write_xyz(path: Path, points: Iterable[Point]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as xyz_file:
        for x, y, z in points:
            xyz_file.write(f"{x:.9g} {y:.9g} {z:.9g}\n")


if __name__ == "__main__":
    #from data_conversion import convert_stl_to_xyz

    generated_files = convert_stl_to_xyz(
        input_folder=r"/Volumes/MyProjects/02_AI/02_Datasets/Manufactured_Component_Data/Data STL",
        output_folder=r"/Volumes/MyProjects/02_AI/02_Datasets/Manufactured_Component_Data/Data XYZ",
        points_per_cloud=10_000,
        seed=42,
    )