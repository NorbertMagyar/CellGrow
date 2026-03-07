"""Render a flyby MP4 around a saved CellGrow NPZ result."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def load_sim_npz(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    with np.load(data_path, allow_pickle=False) as npz:
        if "points" not in npz:
            raise KeyError("NPZ file does not contain 'points'.")
        points = np.asarray(npz["points"], dtype=float)
        meta = {k: npz[k] for k in npz.files if k != "points"}
    return points, meta


def maybe_sample(
    points: np.ndarray,
    scalar_values: Optional[np.ndarray],
    max_render_cells: Optional[int],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if max_render_cells is None or max_render_cells <= 0 or points.shape[0] <= max_render_cells:
        return points, scalar_values
    idx = np.linspace(0, points.shape[0] - 1, int(max_render_cells), dtype=np.int64)
    sampled_points = points[idx]
    sampled_scalars = None if scalar_values is None else scalar_values[idx]
    print(f"Rendering {sampled_points.shape[0]} sampled cells out of {points.shape[0]}.")
    return sampled_points, sampled_scalars


def resolve_scalar(meta: dict[str, np.ndarray], color_by: str, n_points: int) -> Optional[np.ndarray]:
    if color_by == "order":
        return np.arange(n_points, dtype=float)
    if color_by == "radius":
        return np.ones(n_points, dtype=float)
    if color_by == "age" and "birth_age" in meta:
        return np.asarray(meta["birth_age"], dtype=float).reshape(-1)
    if color_by == "u" and "u" in meta:
        return np.asarray(meta["u"], dtype=float).reshape(-1)
    if color_by == "v" and "v" in meta:
        return np.asarray(meta["v"], dtype=float).reshape(-1)
    if color_by == "prog":
        if "u" in meta:
            return np.asarray(meta["u"], dtype=float).reshape(-1)
        if "program_counter" in meta:
            return np.asarray(meta["program_counter"], dtype=float).reshape(-1)
    if color_by == "pz" and "p" in meta:
        p = np.asarray(meta["p"], dtype=float)
        if p.ndim == 2 and p.shape[1] == 3:
            return p[:, 2]
    if color_by == "program_step" and "program_counter" in meta:
        return np.asarray(meta["program_counter"], dtype=float).reshape(-1)
    return None


def add_mesh_with_style(
    plotter,
    mesh,
    *,
    color_by: str,
    cmap: str,
    opacity,
    clim: Optional[tuple[float, float]],
    show_edges: bool,
    edge_color: str,
    edge_width: float,
):
    mesh_kwargs = dict(
        opacity=opacity,
        lighting=True,
        smooth_shading=True,
        ambient=0.12,
        diffuse=0.78,
        specular=0.22,
        specular_power=18.0,
        show_edges=show_edges,
        edge_color=edge_color,
        line_width=edge_width,
    )
    if color_by in {"order", "radius", "scalar"}:
        mesh_kwargs["scalars"] = "val"
        mesh_kwargs["cmap"] = cmap
        if clim is not None:
            mesh_kwargs["clim"] = clim
    else:
        mesh_kwargs["color"] = "lightsteelblue"
    plotter.add_mesh(mesh, **mesh_kwargs)


def build_flyby_movie(
    *,
    points: np.ndarray,
    scalar_values: Optional[np.ndarray],
    output_path: str,
    cell_radius: float,
    color_by: str,
    cmap: str,
    opacity: float,
    sphere_theta: int,
    sphere_phi: int,
    show_edges: bool,
    edge_color: str,
    edge_width: float,
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    macro_block_size: int,
    max_render_cells: Optional[int],
    elevation_degrees: float,
    zoom_out: float,
    show_centers: bool,
) -> Path:
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyVista is required. Install with: pip install pyvista") from exc

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be shaped (N, 3)")
    if pts.shape[0] == 0:
        raise ValueError("No points to render")

    scalars = None if scalar_values is None else np.asarray(scalar_values, dtype=float).reshape(-1)
    if scalars is not None and scalars.shape[0] != pts.shape[0]:
        raise ValueError("scalar_values length must match points")

    pts, scalars = maybe_sample(pts, scalars, max_render_cells)

    out_w, out_h = int(width), int(height)
    if macro_block_size > 1:
        adj_w = ((out_w + macro_block_size - 1) // macro_block_size) * macro_block_size
        adj_h = ((out_h + macro_block_size - 1) // macro_block_size) * macro_block_size
        if (adj_w, adj_h) != (out_w, out_h):
            print(
                f"Adjusted movie size from {(out_w, out_h)} to {(adj_w, adj_h)} "
                f"for macro_block_size={macro_block_size}"
            )
        out_w, out_h = adj_w, adj_h

    cloud = pv.PolyData(pts)
    cloud["scale"] = np.full(pts.shape[0], float(cell_radius), dtype=float)
    base_sphere = pv.Sphere(
        radius=1.0,
        theta_resolution=int(sphere_theta),
        phi_resolution=int(sphere_phi),
    )
    spheres = cloud.glyph(geom=base_sphere, scale="scale", orient=False)

    clim: Optional[tuple[float, float]] = None
    if color_by == "radius":
        spheres["val"] = np.repeat(np.full(pts.shape[0], float(cell_radius), dtype=float), base_sphere.n_points)
    elif color_by == "order":
        order = np.arange(pts.shape[0], dtype=float)
        spheres["val"] = np.repeat(order, base_sphere.n_points)
        clim = (0.0, max(1.0, float(pts.shape[0] - 1)))
    elif color_by == "scalar":
        if scalars is None:
            raise ValueError("scalar_values required for scalar coloring")
        spheres["val"] = np.repeat(scalars, base_sphere.n_points)
        lo = float(np.min(scalars))
        hi = float(np.max(scalars))
        if hi <= lo:
            hi = lo + 1.0
        clim = (lo, hi)

    center = pts.mean(axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    fit_radius = max(float(cell_radius), float(d.max() + cell_radius)) * float(zoom_out)
    mins = pts.min(axis=0) - cell_radius
    maxs = pts.max(axis=0) + cell_radius
    fixed_bounds = (
        float(mins[0]), float(maxs[0]),
        float(mins[1]), float(maxs[1]),
        float(mins[2]), float(maxs[2]),
    )

    frames = max(1, int(round(float(fps) * float(duration_seconds))))
    radius_xy = fit_radius * np.cos(np.deg2rad(elevation_degrees))
    z_offset = fit_radius * np.sin(np.deg2rad(elevation_degrees))

    out = Path(output_path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)

    pl = pv.Plotter(off_screen=True, window_size=(out_w, out_h))
    pl.set_background("white")
    pl.open_movie(str(out), framerate=int(fps), macro_block_size=int(macro_block_size))
    try:
        pl.enable_anti_aliasing()
    except Exception:
        pass
    try:
        pl.enable_lightkit()
    except Exception:
        pass
    try:
        pl.disable_parallel_projection()
    except Exception:
        try:
            pl.camera.parallel_projection = False
        except Exception:
            pass
    pl.camera.clipping_range = (0.01, 40.0 * fit_radius)

    try:
        for frame in range(frames):
            angle = 2.0 * np.pi * (frame / float(frames))
            camera_position = (
                center[0] + radius_xy * np.cos(angle),
                center[1] + radius_xy * np.sin(angle),
                center[2] + z_offset,
            )
            try:
                pl.clear_actors()
            except Exception:
                pl.clear()
                try:
                    pl.enable_lightkit()
                except Exception:
                    pass

            add_mesh_with_style(
                pl,
                spheres,
                color_by=("scalar" if color_by in {"age", "u", "v", "prog", "pz", "program_step"} else color_by),
                cmap=cmap,
                opacity=float(opacity),
                clim=clim,
                show_edges=bool(show_edges),
                edge_color=edge_color,
                edge_width=float(edge_width),
            )
            if show_centers:
                pl.add_points(cloud, color="black", point_size=6.0, render_points_as_spheres=True)

            pl.add_axes()
            pl.show_grid(bounds=fixed_bounds)
            pl.camera_position = [tuple(camera_position), tuple(center), (0.0, 0.0, 1.0)]
            pl.camera.SetViewAngle(30.0)
            pl.write_frame()
    finally:
        pl.close()

    print(f"Flyby movie saved: {out}")
    return out


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a 360-degree flyby movie from a CellGrow NPZ file")
    parser.add_argument("--input", type=str, required=True, help="Input NPZ path")
    parser.add_argument("--output", type=str, default="cell_flyby.mp4", help="Output MP4 path")
    parser.add_argument(
        "--color-by",
        choices=["order", "radius", "solid", "age", "u", "v", "prog", "pz", "program_step"],
        default="order",
        help="Coloring mode",
    )
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap name")
    parser.add_argument("--opacity", type=float, default=1.0, help="Sphere opacity")
    parser.add_argument("--cell-radius", type=float, default=-1.0, help="Override radius (<=0 uses NPZ)")
    parser.add_argument("--sphere-theta", type=int, default=16, help="Sphere theta resolution")
    parser.add_argument("--sphere-phi", type=int, default=16, help="Sphere phi resolution")
    parser.add_argument("--show-centers", action="store_true", help="Render cell centers as points")
    parser.add_argument("--fps", type=int, default=24, help="Movie frames per second")
    parser.add_argument("--duration-seconds", type=float, default=10.0, help="Movie duration in seconds")
    parser.add_argument("--width", type=int, default=1024, help="Movie width in pixels")
    parser.add_argument("--height", type=int, default=860, help="Movie height in pixels")
    parser.add_argument("--macro-block-size", type=int, default=16, help="Macro block size for MP4 encoding")
    parser.add_argument("--movie-show-edges", action=argparse.BooleanOptionalAction, default=False, help="Render mesh edges")
    parser.add_argument("--movie-edge-color", type=str, default="#000000", help="Edge color")
    parser.add_argument("--movie-edge-width", type=float, default=0.6, help="Edge line width")
    parser.add_argument("--max-render-cells", type=int, default=50000, help="Maximum cells to render (0 disables cap)")
    parser.add_argument("--elevation-degrees", type=float, default=35.26438968, help="Camera elevation angle for the orbit")
    parser.add_argument("--zoom-out", type=float, default=4.8, help="Camera distance multiplier relative to fitted colony radius")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    points, meta = load_sim_npz(args.input)
    radius_from_file = float(meta["radius"]) if "radius" in meta else 1.0
    cell_radius = float(args.cell_radius) if args.cell_radius > 0 else radius_from_file
    color_by = args.color_by
    scalar_values = resolve_scalar(meta, color_by, points.shape[0])
    render_color_by = "solid" if color_by == "solid" else ("scalar" if scalar_values is not None and color_by not in {"order", "radius"} else color_by)
    max_render_cells = None if args.max_render_cells == 0 else int(args.max_render_cells)

    print(f"Loaded: {args.input}")
    print(f"Cells: {points.shape[0]}")
    print(f"Radius used for rendering: {cell_radius}")
    if "step_index" in meta:
        print(f"Step index: {int(meta['step_index'])}")
    if "count_history" in meta:
        hist = np.asarray(meta["count_history"])
        print(f"Count history length: {hist.size}")
    print(
        f"Flyby: duration={float(args.duration_seconds):.2f}s, fps={int(args.fps)}, "
        f"frames={max(1, int(round(float(args.fps) * float(args.duration_seconds))))}"
    )

    build_flyby_movie(
        points=points,
        scalar_values=scalar_values,
        output_path=args.output,
        cell_radius=cell_radius,
        color_by=render_color_by,
        cmap=args.cmap,
        opacity=float(args.opacity),
        sphere_theta=int(args.sphere_theta),
        sphere_phi=int(args.sphere_phi),
        show_edges=bool(args.movie_show_edges),
        edge_color=args.movie_edge_color,
        edge_width=float(args.movie_edge_width),
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        duration_seconds=float(args.duration_seconds),
        macro_block_size=int(args.macro_block_size),
        max_render_cells=max_render_cells,
        elevation_degrees=float(args.elevation_degrees),
        zoom_out=float(args.zoom_out),
        show_centers=bool(args.show_centers),
    )


if __name__ == "__main__":
    main()
