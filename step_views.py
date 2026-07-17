"""Render evenly spaced PNG views of a part stored in a STEP file.

The public entry point is :func:`create_step_views`.  Heavy CAD dependencies
are imported only when rendering starts, so applications can import this
module before the rendering extras are installed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


Resolution = int | tuple[int, int]
RGB = tuple[float, float, float]


@dataclass
class BatchResult:
    """Summary returned after processing a folder of STEP files."""

    created_images: list[Path] = field(default_factory=list)
    skipped_files: list[Path] = field(default_factory=list)
    failed_files: dict[Path, str] = field(default_factory=dict)


def _view_filename(source: Path, index: int, angle: float) -> str:
    angle_label = f"{angle:07.2f}".replace(".", "p")
    return f"{source.stem}_view_{index:03d}_azimuth_{angle_label}.png"


def _progress_line(completed: int, total: int, label: str, started_at: float) -> str:
    fraction = completed / total if total else 1.0
    bar_width = 30
    filled = min(bar_width, round(fraction * bar_width))
    bar = "█" * filled + "░" * (bar_width - filled)
    elapsed = max(time.monotonic() - started_at, 0.0)
    rate = completed / elapsed if completed and elapsed else 0.0
    remaining = (total - completed) / rate if rate else 0.0
    eta = f" ETA {remaining / 60:.1f}m" if completed < total and rate else ""
    return f"\r[{bar}] {completed:,}/{total:,} ({fraction:6.2%}){eta}  {label[:55]:<55}"


def _normalise_resolution(resolution: Resolution) -> tuple[int, int]:
    if isinstance(resolution, bool):
        raise ValueError("resolution must be a positive integer or (width, height)")
    if isinstance(resolution, int):
        width = height = resolution
    else:
        if len(resolution) != 2:
            raise ValueError("resolution must contain width and height")
        width, height = resolution
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("resolution values must be integers")
    if width <= 0 or height <= 0:
        raise ValueError("resolution values must be greater than zero")
    return width, height


def _view_angles(number_of_views: int, start_angle: float) -> list[float]:
    if isinstance(number_of_views, bool) or not isinstance(number_of_views, int):
        raise ValueError("number_of_views must be a positive integer")
    if number_of_views <= 0:
        raise ValueError("number_of_views must be greater than zero")
    if not math.isfinite(start_angle):
        raise ValueError("start_angle must be finite")
    interval = 360.0 / number_of_views
    return [(start_angle + index * interval) % 360.0 for index in range(number_of_views)]


def _normalise_rgb(value: Sequence[float], name: str) -> RGB:
    if len(value) != 3:
        raise ValueError(f"{name} must contain red, green, and blue values")
    result = tuple(float(channel) for channel in value)
    if any(not math.isfinite(channel) or channel < 0 or channel > 1 for channel in result):
        raise ValueError(f"{name} values must be between 0 and 1")
    return result  # type: ignore[return-value]


def _load_cad_dependencies():
    try:
        import cadquery as cq
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import PolyCollection
        from matplotlib.figure import Figure
    except ImportError as error:
        raise RuntimeError(
            "STEP rendering dependencies are missing. Install them with "
            "'python -m pip install -r requirements.txt'. CadQuery may require "
            "Python 3.10-3.12 on some platforms."
        ) from error
    return cq, np, Figure, FigureCanvasAgg, PolyCollection


def _tessellate_shape(shape, linear_tolerance: float, angular_tolerance: float):
    # CadQuery's STEP importer returns a Workplane/CQ container in current
    # releases, but older releases may return a Shape directly.  Tessellate
    # every contained shape so multi-body STEP files are rendered completely.
    if hasattr(shape, "vals") and callable(shape.vals):
        cad_shapes = shape.vals()
    else:
        cad_shapes = [shape]

    vertices = []
    triangles = []
    for cad_shape in cad_shapes:
        if not hasattr(cad_shape, "tessellate"):
            raise TypeError(
                f"unsupported object returned by STEP importer: {type(cad_shape).__name__}"
            )
        shape_vertices, shape_triangles = cad_shape.tessellate(
            linear_tolerance, angular_tolerance
        )
        vertex_offset = len(vertices)
        vertices.extend(shape_vertices)
        triangles.extend(
            (int(a) + vertex_offset, int(b) + vertex_offset, int(c) + vertex_offset)
            for a, b, c in shape_triangles
        )

    if not vertices or not triangles:
        raise ValueError("the STEP file did not contain a renderable solid or surface")

    return [vertex.toTuple() for vertex in vertices], triangles


def _render_mesh(
    vertices,
    triangles,
    path: Path,
    *,
    angle: float,
    elevation: float,
    resolution: tuple[int, int],
    background: RGB,
    part_colour: RGB,
    transparent_background: bool,
    np,
    Figure,
    FigureCanvasAgg,
    PolyCollection,
) -> None:
    """Render a triangulated shape without creating a GUI window."""
    vertex_array = np.asarray(vertices, dtype=float)
    triangle_array = np.asarray(triangles, dtype=int)
    centred = vertex_array - (vertex_array.min(axis=0) + vertex_array.max(axis=0)) / 2.0

    azimuth = math.radians(angle)
    elevation_radians = math.radians(elevation)
    view_out = np.asarray(
        [
            math.cos(elevation_radians) * math.cos(azimuth),
            math.cos(elevation_radians) * math.sin(azimuth),
            math.sin(elevation_radians),
        ]
    )
    screen_right = np.asarray([-math.sin(azimuth), math.cos(azimuth), 0.0])
    screen_up = np.cross(view_out, screen_right)

    projected = np.column_stack((centred @ screen_right, centred @ screen_up))
    depth = centred @ view_out
    faces = centred[triangle_array]
    face_normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normal_lengths = np.linalg.norm(face_normals, axis=1)
    valid = normal_lengths > 1e-12
    face_normals[valid] /= normal_lengths[valid, None]

    # STEP face winding can vary between exporters, so use two-sided diffuse
    # lighting and keep every face. Far faces are painted first.
    light_direction = view_out + np.asarray([0.15, -0.2, 0.55])
    light_direction /= np.linalg.norm(light_direction)
    brightness = 0.38 + 0.62 * np.abs(face_normals @ light_direction)
    base_colour = np.asarray(part_colour)
    face_colours = np.clip(brightness[:, None] * base_colour, 0.0, 1.0)
    face_depth = depth[triangle_array].mean(axis=1)
    order = np.argsort(face_depth)

    width, height = resolution
    dpi = 100
    alpha = 0.0 if transparent_background else 1.0
    figure = Figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        facecolor=(*background, alpha),
    )
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1), facecolor=(*background, alpha))
    collection = PolyCollection(
        projected[triangle_array][order],
        facecolors=face_colours[order],
        edgecolors="none",
        antialiaseds=True,
    )
    axes.add_collection(collection)

    x_min, y_min = projected.min(axis=0)
    x_max, y_max = projected.max(axis=0)
    x_span = max(x_max - x_min, 1e-9) * 1.10
    y_span = max(y_max - y_min, 1e-9) * 1.10
    image_aspect = width / height
    if x_span / y_span < image_aspect:
        x_span = y_span * image_aspect
    else:
        y_span = x_span / image_aspect
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    axes.set_xlim(x_mid - x_span / 2.0, x_mid + x_span / 2.0)
    axes.set_ylim(y_mid - y_span / 2.0, y_mid + y_span / 2.0)
    axes.set_aspect("equal", adjustable="box")
    axes.axis("off")
    figure.savefig(
        path,
        dpi=dpi,
        transparent=transparent_background,
        facecolor=figure.get_facecolor(),
        edgecolor="none",
    )


def create_step_views(
    step_file: str | Path,
    output_directory: str | Path,
    *,
    number_of_views: int = 8,
    resolution: Resolution = (1024, 1024),
    elevation: float = 20.0,
    start_angle: float = 0.0,
    background: Sequence[float] = (1.0, 1.0, 1.0),
    part_colour: Sequence[float] = (0.72, 0.75, 0.80),
    transparent_background: bool = False,
    linear_tolerance: float = 0.1,
    angular_tolerance: float = 0.1,
) -> list[Path]:
    """Create PNG views of a STEP part and return their paths.

    ``number_of_views`` cameras are placed at equal azimuth intervals around
    the model. For example, 8 views produces one image every 45 degrees.
    ``resolution`` can be a square size (such as ``1024``) or a
    ``(width, height)`` tuple.
    """
    source = Path(step_file).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"STEP file not found: {source}")
    if source.suffix.lower() not in {".step", ".stp"}:
        raise ValueError("step_file must have a .step or .stp extension")

    width, height = _normalise_resolution(resolution)
    angles = _view_angles(number_of_views, float(start_angle))
    if not math.isfinite(elevation) or not -89.0 <= elevation <= 89.0:
        raise ValueError("elevation must be between -89 and 89 degrees")
    if linear_tolerance <= 0 or angular_tolerance <= 0:
        raise ValueError("tessellation tolerances must be greater than zero")
    background_rgb = _normalise_rgb(background, "background")
    part_rgb = _normalise_rgb(part_colour, "part_colour")

    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cq, np, Figure, FigureCanvasAgg, PolyCollection = _load_cad_dependencies()

    try:
        shape = cq.importers.importStep(str(source))
    except Exception as error:
        raise ValueError(f"could not read STEP file '{source}': {error}") from error

    vertices, triangles = _tessellate_shape(shape, linear_tolerance, angular_tolerance)

    output_paths: list[Path] = []
    manifest_views = []
    for index, angle in enumerate(angles, start=1):
        path = destination / _view_filename(source, index, angle)
        _render_mesh(
            vertices,
            triangles,
            path,
            angle=angle,
            elevation=elevation,
            resolution=(width, height),
            background=background_rgb,
            part_colour=part_rgb,
            transparent_background=transparent_background,
            np=np,
            Figure=Figure,
            FigureCanvasAgg=FigureCanvasAgg,
            PolyCollection=PolyCollection,
        )
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"failed to write image: {path}")
        output_paths.append(path)
        manifest_views.append({"file": path.name, "azimuth": angle, "elevation": elevation})

    manifest = {
        "source": str(source),
        "resolution": [width, height],
        "number_of_views": number_of_views,
        "views": manifest_views,
    }
    (destination / f"{source.stem}_views.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output_paths


def create_step_views_for_folder(
    input_directory: str | Path,
    output_directory: str | Path,
    *,
    recursive: bool = False,
    skip_existing: bool = True,
    continue_on_error: bool = True,
    show_progress: bool = True,
    **view_options,
) -> BatchResult:
    """Render every STEP file in a folder with a terminal progress bar.

    Each part is written into ``output_directory/<STEP filename>/``. When
    ``skip_existing`` is true, a part is skipped only when every image expected
    for the requested view count already exists in its subfolder.
    """
    source_directory = Path(input_directory).expanduser().resolve()
    if not source_directory.is_dir():
        raise NotADirectoryError(f"STEP input directory not found: {source_directory}")
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    iterator = source_directory.rglob("*") if recursive else source_directory.iterdir()
    step_files = sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in {".step", ".stp"}
    )
    if not step_files:
        raise FileNotFoundError(f"no .step or .stp files found in: {source_directory}")

    number_of_views = view_options.get("number_of_views", 8)
    start_angle = float(view_options.get("start_angle", 0.0))
    angles = _view_angles(number_of_views, start_angle)
    result = BatchResult()
    started_at = time.monotonic()

    if show_progress:
        sys.stdout.write(_progress_line(0, len(step_files), "Starting...", started_at))
        sys.stdout.flush()

    for completed, step_file in enumerate(step_files, start=1):
        part_destination = destination / step_file.stem
        expected_paths = [
            part_destination / _view_filename(step_file, index, angle)
            for index, angle in enumerate(angles, start=1)
        ]
        status = step_file.name
        if skip_existing and all(path.is_file() and path.stat().st_size > 0 for path in expected_paths):
            result.skipped_files.append(step_file)
            status = f"Skipped: {step_file.name}"
        else:
            try:
                result.created_images.extend(
                    create_step_views(step_file, part_destination, **view_options)
                )
            except Exception as error:
                result.failed_files[step_file] = f"{type(error).__name__}: {error}"
                status = f"Failed: {step_file.name}"
                if not continue_on_error:
                    if show_progress:
                        sys.stdout.write("\n")
                    raise

        if show_progress:
            sys.stdout.write(_progress_line(completed, len(step_files), status, started_at))
            sys.stdout.flush()

    if show_progress:
        sys.stdout.write("\n")

    error_report = {
        "input_directory": str(source_directory),
        "output_directory": str(destination),
        "processed_files": len(step_files) - len(result.skipped_files),
        "skipped_files": len(result.skipped_files),
        "failed_files": [
            {"file": str(path), "error": error}
            for path, error in result.failed_files.items()
        ],
    }
    (destination / "batch_report.json").write_text(
        json.dumps(error_report, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render evenly spaced PNG views from a STEP file.")
    parser.add_argument("step_file", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--views", type=int, default=8, help="number of angle views (default: 8)")
    parser.add_argument(
        "--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(1024, 1024)
    )
    parser.add_argument("--elevation", type=float, default=20.0, help="camera elevation in degrees")
    parser.add_argument("--start-angle", type=float, default=0.0, help="first azimuth angle in degrees")
    parser.add_argument("--transparent", action="store_true", help="use a transparent background")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    paths = create_step_views(
        args.step_file,
        args.output_directory,
        number_of_views=args.views,
        resolution=tuple(args.resolution),
        elevation=args.elevation,
        start_angle=args.start_angle,
        transparent_background=args.transparent,
    )
    print(f"Created {len(paths)} views in {args.output_directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
