"""
Inspect saved cell-growth NPZ files with PyVista.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_sim_npz(path: str) -> tuple[np.ndarray, dict]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    with np.load(data_path, allow_pickle=False) as npz:
        if "points" not in npz:
            raise KeyError("NPZ file does not contain 'points'.")
        points = np.asarray(npz["points"], dtype=float)
        meta = {k: npz[k] for k in npz.files if k != "points"}
    return points, meta


def maybe_sample(points: np.ndarray, max_render_cells: int) -> np.ndarray:
    if max_render_cells <= 0 or points.shape[0] <= max_render_cells:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_render_cells, dtype=np.int64)
    sampled = points[idx]
    print(f"Rendering {sampled.shape[0]} sampled cells out of {points.shape[0]}.")
    return sampled


def glyph_spheres(
    points: np.ndarray,
    *,
    cell_radius: float,
    sphere_theta: int,
    sphere_phi: int,
):
    import pyvista as pv

    cloud = pv.PolyData(points)
    cloud["scale"] = np.full(points.shape[0], float(cell_radius), dtype=float)
    base_sphere = pv.Sphere(
        radius=1.0,
        theta_resolution=int(sphere_theta),
        phi_resolution=int(sphere_phi),
    )
    spheres = cloud.glyph(geom=base_sphere, scale="scale", orient=False)
    return cloud, base_sphere, spheres


def add_colored_mesh(plotter, mesh, *, color_by: str, cmap: str, opacity: float):
    if color_by in {"radius", "order"} and "val" in mesh.array_names:
        plotter.add_mesh(mesh, scalars="val", cmap=cmap, opacity=opacity, smooth_shading=True)
        return
    plotter.add_mesh(mesh, color="lightsteelblue", opacity=opacity, smooth_shading=True)


def show_cells(
    points: np.ndarray,
    *,
    cell_radius: float,
    sphere_theta: int,
    sphere_phi: int,
    opacity: float,
    color_by: str,
    cmap: str,
    show_centers: bool,
    mode: str,
    max_render_cells: int,
) -> None:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("PyVista is required. Install with: pip install pyvista") from exc

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be shaped (N, 3)")
    if pts.shape[0] == 0:
        raise ValueError("No points to visualize.")

    pts = maybe_sample(pts, max_render_cells)
    cloud, base_sphere, spheres = glyph_spheres(
        pts,
        cell_radius=cell_radius,
        sphere_theta=sphere_theta,
        sphere_phi=sphere_phi,
    )

    if color_by == "radius":
        spheres["val"] = np.ones(spheres.n_points, dtype=float)
    elif color_by == "order":
        sphere_point_order = np.repeat(np.arange(pts.shape[0], dtype=float), base_sphere.n_points)
        spheres["val"] = sphere_point_order

    render_mesh = spheres
    if mode == "xpos_cut":
        clipped = spheres.clip(normal=(1.0, 0.0, 0.0), origin=(0.0, 0.0, 0.0), invert=False)
        # Robust side selection across VTK/PyVista versions.
        if clipped.n_points > 0 and clipped.bounds[1] < 0.0:
            clipped = spheres.clip(normal=(1.0, 0.0, 0.0), origin=(0.0, 0.0, 0.0), invert=True)
        render_mesh = clipped

    pl = pv.Plotter()
    pl.set_background("white")
    add_colored_mesh(
        pl,
        render_mesh,
        color_by=color_by,
        cmap=cmap,
        opacity=opacity,
    )

    if show_centers:
        pl.add_points(cloud, color="black", point_size=6.0, render_points_as_spheres=True)

    pl.add_axes()
    pl.show_grid()
    pl.show()


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect saved CellGrow NPZ data")
    parser.add_argument("--input", type=str, default="cell_growth_final.npz", help="Input NPZ path")
    parser.add_argument(
        "--mode",
        choices=["full", "xpos_cut"],
        default="xpos_cut",
        help="Visualization mode: full colony or x>=0 cut-through",
    )
    parser.add_argument(
        "--color-by",
        choices=["order", "radius", "solid"],
        default="order",
        help="Coloring mode",
    )
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap name")
    parser.add_argument("--opacity", type=float, default=1.0, help="Mesh opacity")
    parser.add_argument("--cell-radius", type=float, default=-1.0, help="Override radius (<=0 uses NPZ)")
    parser.add_argument("--sphere-theta", type=int, default=16, help="Sphere theta resolution")
    parser.add_argument("--sphere-phi", type=int, default=16, help="Sphere phi resolution")
    parser.add_argument("--show-centers", action="store_true", help="Render cell centers as points")
    parser.add_argument(
        "--max-render-cells",
        type=int,
        default=20000,
        help="Maximum cells to render (0 disables sampling cap)",
    )
    return parser


def main() -> None:
    global points
    args = build_cli().parse_args()
    points, meta = load_sim_npz(args.input)
    radius_from_file = float(meta["radius"]) if "radius" in meta else 1.0
    cell_radius = float(args.cell_radius) if args.cell_radius > 0 else radius_from_file
    color_by = "solid" if args.color_by == "solid" else args.color_by

    print(f"Loaded: {args.input}")
    print(f"Cells: {points.shape[0]}")
    print(f"Radius used for rendering: {cell_radius}")
    if "step_index" in meta:
        print(f"Step index: {int(meta['step_index'])}")
    if "count_history" in meta:
        hist = np.asarray(meta["count_history"])
        print(f"Count history length: {hist.size}")

    show_cells(
        points,
        cell_radius=cell_radius,
        sphere_theta=args.sphere_theta,
        sphere_phi=args.sphere_phi,
        opacity=float(args.opacity),
        color_by=color_by,
        cmap=args.cmap,
        show_centers=bool(args.show_centers),
        mode=args.mode,
        max_render_cells=int(args.max_render_cells),
    )


if __name__ == "__main__":
    main()
