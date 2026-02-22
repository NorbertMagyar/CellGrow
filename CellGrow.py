"""
3D point-growth simulation with CuPy acceleration.

Model summary:
- Each cell is represented by a 3D point.
- Default behavior is density-regulated fission/homeostasis/death.
- Fission replaces one parent with two daughters.
- Division direction follows a "least resistance" heuristic:
  repulsive vector away from all other cells.
- Post-division relaxation keeps a minimum center spacing of 1 radius.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


try:
    import cupy as cp

    _GPU_ENABLED = True
except Exception:  # pragma: no cover - fallback only when CuPy is unavailable
    import numpy as cp  # type: ignore

    _GPU_ENABLED = False


ActionRule = Callable[[cp.ndarray, int], cp.ndarray]

DEFAULT_STEPS = 50
DEFAULT_SHOW = True
DEFAULT_COLOR_BY = "order"
DEFAULT_SAVE_MOVIE = True
DEFAULT_MOVIE_PATH = "cell_growth.mp4"
DEFAULT_MOVIE_FPS = 24
DEFAULT_INTERP_FRAMES = 10
DEFAULT_MOVIE_WIDTH = 1024
DEFAULT_MOVIE_HEIGHT = 860
DEFAULT_MOVIE_SPHERE_THETA = 16
DEFAULT_MOVIE_SPHERE_PHI = 16
DEFAULT_MOVIE_SHOW_EDGES = False
DEFAULT_MOVIE_EDGE_COLOR = "#202020"
DEFAULT_MOVIE_EDGE_WIDTH = 0.6
DEFAULT_MOVIE_MACRO_BLOCK_SIZE = 16
DEFAULT_MOVIE_DEATH_ANIMATION = "shrink"


def show_cells_pyvista(
    points,
    *,
    cell_radius: float = 1.0,
    sphere_theta: int = 16,
    sphere_phi: int = 16,
    opacity: float = 1.0,
    show_centers: bool = False,
    centers_size: float = 6.0,
    color_by: str = "order",
    cmap: str = "viridis",
    notebook: bool = False,
) -> None:
    """
    Visualize cells as sphere glyphs in PyVista.

    Adapted from the glyph-based sphere method used in ~/algogen.
    """
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - dependency/env specific
        raise ImportError(
            "PyVista is required for visualization. Install with: pip install pyvista"
        ) from exc

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be shaped (N, 3)")
    if pts.shape[0] == 0:
        raise ValueError("No points to visualize.")

    radii = np.full(pts.shape[0], float(cell_radius), dtype=float)

    cloud = pv.PolyData(pts)
    cloud["scale"] = radii

    base_sphere = pv.Sphere(
        radius=1.0,
        theta_resolution=sphere_theta,
        phi_resolution=sphere_phi,
    )
    spheres = cloud.glyph(geom=base_sphere, scale="scale", orient=False)

    pl = pv.Plotter(notebook=notebook)
    pl.set_background("white")

    if color_by == "radius":
        spheres["val"] = np.repeat(radii, base_sphere.n_points)
        pl.add_mesh(spheres, scalars="val", cmap=cmap, opacity=opacity, smooth_shading=True)
    elif color_by == "order":
        order = np.arange(pts.shape[0], dtype=float)
        spheres["val"] = np.repeat(order, base_sphere.n_points)
        pl.add_mesh(spheres, scalars="val", cmap=cmap, opacity=opacity, smooth_shading=True)
    else:
        pl.add_mesh(spheres, color="lightsteelblue", opacity=opacity, smooth_shading=True)

    if show_centers:
        pl.add_points(cloud, color="black", point_size=centers_size, render_points_as_spheres=True)

    pl.add_axes()
    pl.show_grid()
    pl.show()


def backend_summary() -> str:
    """Return a human-readable summary of the active compute backend."""
    if not _GPU_ENABLED:
        return "Backend: NumPy CPU fallback (CuPy import failed)."

    try:
        dev = cp.cuda.Device()  # current device
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode("utf-8")
        return f"Backend: CuPy on GPU device {dev.id} ({name})"
    except Exception:
        return "Backend: CuPy enabled (GPU details unavailable)."


@dataclass
class StepTransition:
    """One timestep transition used for interpolated animation."""

    source_points: cp.ndarray
    target_points: cp.ndarray
    source_ids: cp.ndarray
    target_ids: cp.ndarray
    source_size: cp.ndarray
    target_size: cp.ndarray
    source_alpha: cp.ndarray
    target_alpha: cp.ndarray


@dataclass
class CellGrowth3D:
    """GPU-accelerated 3D growth simulation for point-based cells."""

    radius: float = 1.0
    split_distance: float = 1.0
    neighborhood_radius_factor: float = 2.0
    crowding_stay_threshold: int = 4
    crowding_death_threshold: int = 6
    enforce_non_overlap: bool = True
    overlap_relax_iters: int = 32
    overlap_tol: float = 1e-4
    seed: int = 42
    max_cells: Optional[int] = None
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be > 0")
        if self.neighborhood_radius_factor <= 0:
            raise ValueError("neighborhood_radius_factor must be > 0")
        if self.crowding_stay_threshold < 0 or self.crowding_death_threshold < 0:
            raise ValueError("crowding thresholds must be >= 0")
        if self.crowding_death_threshold <= self.crowding_stay_threshold:
            raise ValueError("crowding_death_threshold must be > crowding_stay_threshold")

        self.points = cp.zeros((1, 3), dtype=self.dtype)
        self.cell_ids = cp.zeros((1,), dtype=cp.int64)
        self._next_cell_id = 1
        self.step_index = 0
        self.count_history = [1]
        self._rng = cp.random.RandomState(self.seed)

    @staticmethod
    def always_divide_rule(points: cp.ndarray, step: int) -> cp.ndarray:
        """Action rule: all cells divide (1 = divide, 0 = stay, -1 = die)."""
        return cp.ones(points.shape[0], dtype=cp.int8)

    def _neighbor_counts_within(self, points: cp.ndarray, radius: float) -> cp.ndarray:
        """Count neighbors within a fixed radius (self excluded)."""
        n = points.shape[0]
        if n == 0:
            return cp.zeros((0,), dtype=cp.int32)
        if n == 1:
            return cp.zeros((1,), dtype=cp.int32)

        deltas = points[:, None, :] - points[None, :, :]  # (n, n, 3)
        dist2 = cp.sum(deltas * deltas, axis=2)
        within = dist2 <= float(radius * radius)
        within = within & (~cp.eye(n, dtype=cp.bool_))
        return cp.sum(within, axis=1, dtype=cp.int32)

    def density_regulated_rule(self, points: cp.ndarray, step: int) -> cp.ndarray:
        """
        Default biological rule:
        - neighbors >= crowding_death_threshold: die
        - neighbors > crowding_stay_threshold: stay
        - otherwise: divide
        """
        del step  # Rule currently uses geometry only.
        neighbor_radius = self.neighborhood_radius_factor * self.radius
        counts = self._neighbor_counts_within(points, neighbor_radius)
        actions = cp.ones(points.shape[0], dtype=cp.int8)
        actions[counts > self.crowding_stay_threshold] = 0
        actions[counts >= self.crowding_death_threshold] = -1
        return actions

    def _random_unit_vectors(self, n: int) -> cp.ndarray:
        vec = self._rng.normal(0.0, 1.0, size=(n, 3)).astype(self.dtype, copy=False)
        norm = cp.linalg.norm(vec, axis=1, keepdims=True)
        return vec / cp.maximum(norm, 1e-8)

    def _least_resistance_directions(self) -> cp.ndarray:
        """
        Compute a direction per cell that points toward sparse space.

        The direction is the normalized repulsion vector from all other cells.
        """
        n = self.points.shape[0]
        if n == 1:
            return cp.asarray([[1.0, 0.0, 0.0]], dtype=self.dtype)

        deltas = self.points[:, None, :] - self.points[None, :, :]  # (n, n, 3)
        dist2 = cp.sum(deltas * deltas, axis=2)
        mask = ~cp.eye(n, dtype=cp.bool_)

        # Inverse-cube weighting of repulsion with diagonal masked out.
        inv_dist3 = cp.where(mask, 1.0 / cp.maximum(dist2 * cp.sqrt(dist2), 1e-8), 0.0)
        repulsion = cp.einsum("ijk,ij->ik", deltas, inv_dist3).astype(
            self.dtype, copy=False
        )

        norms = cp.linalg.norm(repulsion, axis=1, keepdims=True)
        dirs = repulsion / cp.maximum(norms, 1e-8)

        zero_mask = (norms[:, 0] <= 1e-8)
        if bool(cp.any(zero_mask)):
            fallback = self._random_unit_vectors(int(zero_mask.sum()))
            dirs[zero_mask] = fallback
        return dirs

    def _resolve_overlaps(self, pts: cp.ndarray) -> cp.ndarray:
        """
        Iteratively separate cells so center distances are >= 1*radius.
        """
        n = pts.shape[0]
        if n < 2:
            return pts

        min_dist = float(1 * self.radius)
        max_step = 0.25 * min_dist
        jitter_scale = 1e-3 * min_dist

        for _ in range(self.overlap_relax_iters):
            deltas = pts[:, None, :] - pts[None, :, :]  # (n, n, 3)
            dist2 = cp.sum(deltas * deltas, axis=2)
            dist = cp.sqrt(cp.maximum(dist2, 1e-12))
            mask = ~cp.eye(n, dtype=cp.bool_)

            overlap = cp.maximum(min_dist - dist, 0.0)
            overlap = cp.where(mask, overlap, 0.0)
            if not bool(cp.any(overlap > self.overlap_tol)):
                break

            # Gradient-style repulsion from neighbors that overlap.
            weights = overlap / cp.maximum(dist, 1e-8)
            disp = cp.einsum("ij,ijk->ik", weights, deltas).astype(self.dtype, copy=False)
            disp_norm = cp.linalg.norm(disp, axis=1)
            stuck = (cp.sum(overlap, axis=1) > 0.0) & (disp_norm < 1e-8)
            if bool(cp.any(stuck)):
                disp[stuck] = self._random_unit_vectors(int(stuck.sum()))
                disp_norm = cp.linalg.norm(disp, axis=1)

            disp_norm_col = disp_norm[:, None]
            scale = cp.minimum(1.0, max_step / cp.maximum(disp_norm_col, 1e-8))
            pts = pts + 0.5 * disp * scale

            # Small nudge helps resolve exact duplicates quickly.
            if bool(cp.any(stuck)):
                pts[stuck] += jitter_scale * self._random_unit_vectors(int(stuck.sum()))

        return pts

    def _step_internal(
        self,
        action_rule: Optional[ActionRule] = None,
        return_transition: bool = False,
        death_animation: str = "none",
    ) -> tuple[cp.ndarray, Optional[StepTransition]]:
        """Core step logic with optional source->target transition capture."""
        valid_death_modes = {"none", "fade", "shrink", "fade_shrink"}
        if death_animation not in valid_death_modes:
            raise ValueError(f"death_animation must be one of {sorted(valid_death_modes)}")

        rule = action_rule or self.density_regulated_rule
        actions = rule(self.points, self.step_index)
        if actions.shape != (self.points.shape[0],):
            raise ValueError("Action rule must return a vector shaped (n_cells,)")

        divide_mask = actions == 1
        stay_mask = actions == 0
        die_mask = actions == -1

        if bool(cp.any((~divide_mask) & (~stay_mask) & (~die_mask))):
            raise ValueError("Actions must only contain -1, 0, or 1.")

        next_blocks = []
        next_id_blocks = []
        source_blocks = []
        source_id_blocks = []
        target_blocks = []
        target_id_blocks = []
        source_size_blocks = []
        target_size_blocks = []
        source_alpha_blocks = []
        target_alpha_blocks = []

        one_f = 1.0
        zero_f = 0.0

        if bool(cp.any(stay_mask)):
            stay_points = self.points[stay_mask]
            stay_ids = self.cell_ids[stay_mask]
            next_blocks.append(stay_points)
            next_id_blocks.append(stay_ids)
            if return_transition:
                source_blocks.append(stay_points)
                source_id_blocks.append(stay_ids)
                target_blocks.append(stay_points)
                target_id_blocks.append(stay_ids)
                n_stay = stay_points.shape[0]
                source_size_blocks.append(cp.full(n_stay, one_f, dtype=self.dtype))
                target_size_blocks.append(cp.full(n_stay, one_f, dtype=self.dtype))
                source_alpha_blocks.append(cp.full(n_stay, one_f, dtype=self.dtype))
                target_alpha_blocks.append(cp.full(n_stay, one_f, dtype=self.dtype))

        if bool(cp.any(divide_mask)):
            dirs = self._least_resistance_directions()[divide_mask]
            parents = self.points[divide_mask]
            parent_ids = self.cell_ids[divide_mask]
            # Never place sister cells closer than one radius.
            effective_split_distance = max(self.split_distance, 1.0 * self.radius)
            offset = (effective_split_distance * 0.5) * dirs
            daughters_a = parents + offset
            daughters_b = parents - offset
            n_div = int(parents.shape[0])
            daughter_ids_a = cp.arange(
                self._next_cell_id,
                self._next_cell_id + n_div,
                dtype=cp.int64,
            )
            daughter_ids_b = cp.arange(
                self._next_cell_id + n_div,
                self._next_cell_id + 2 * n_div,
                dtype=cp.int64,
            )
            self._next_cell_id += 2 * n_div
            next_blocks.extend([daughters_a, daughters_b])
            next_id_blocks.extend([daughter_ids_a, daughter_ids_b])
            if return_transition:
                source_blocks.extend([parents, parents])
                source_id_blocks.extend([parent_ids, parent_ids])
                target_blocks.extend([daughters_a, daughters_b])
                target_id_blocks.extend([daughter_ids_a, daughter_ids_b])
                source_size_blocks.extend(
                    [cp.full(n_div, one_f, dtype=self.dtype), cp.full(n_div, one_f, dtype=self.dtype)]
                )
                target_size_blocks.extend(
                    [cp.full(n_div, one_f, dtype=self.dtype), cp.full(n_div, one_f, dtype=self.dtype)]
                )
                source_alpha_blocks.extend(
                    [cp.full(n_div, one_f, dtype=self.dtype), cp.full(n_div, one_f, dtype=self.dtype)]
                )
                target_alpha_blocks.extend(
                    [cp.full(n_div, one_f, dtype=self.dtype), cp.full(n_div, one_f, dtype=self.dtype)]
                )

        if return_transition and death_animation != "none" and bool(cp.any(die_mask)):
            dead_points = self.points[die_mask]
            dead_ids = self.cell_ids[die_mask]
            n_dead = dead_points.shape[0]
            source_blocks.append(dead_points)
            source_id_blocks.append(dead_ids)
            target_blocks.append(dead_points)
            target_id_blocks.append(dead_ids)

            do_fade = death_animation in {"fade", "fade_shrink"}
            do_shrink = death_animation in {"shrink", "fade_shrink"}
            target_alpha = zero_f if do_fade else one_f
            target_size = zero_f if do_shrink else one_f
            source_size_blocks.append(cp.full(n_dead, one_f, dtype=self.dtype))
            target_size_blocks.append(cp.full(n_dead, target_size, dtype=self.dtype))
            source_alpha_blocks.append(cp.full(n_dead, one_f, dtype=self.dtype))
            target_alpha_blocks.append(cp.full(n_dead, target_alpha, dtype=self.dtype))

        if next_blocks:
            next_points = cp.concatenate(next_blocks, axis=0)
            next_ids = cp.concatenate(next_id_blocks, axis=0)
            if self.enforce_non_overlap and next_points.shape[0] > 1:
                next_points = self._resolve_overlaps(next_points)
        else:
            next_points = cp.empty((0, 3), dtype=self.dtype)
            next_ids = cp.empty((0,), dtype=cp.int64)

        if self.max_cells is not None and next_points.shape[0] > self.max_cells:
            raise RuntimeError(f"Cell limit exceeded: {next_points.shape[0]} > {self.max_cells}")

        transition = None
        if return_transition:
            if source_blocks:
                source_points = cp.concatenate(source_blocks, axis=0)
                source_ids = cp.concatenate(source_id_blocks, axis=0)
                target_points = cp.concatenate(target_blocks, axis=0)
                target_ids = cp.concatenate(target_id_blocks, axis=0)
                source_size = cp.concatenate(source_size_blocks, axis=0)
                target_size = cp.concatenate(target_size_blocks, axis=0)
                source_alpha = cp.concatenate(source_alpha_blocks, axis=0)
                target_alpha = cp.concatenate(target_alpha_blocks, axis=0)
            else:
                source_points = cp.empty((0, 3), dtype=self.dtype)
                source_ids = cp.empty((0,), dtype=cp.int64)
                target_points = cp.empty((0, 3), dtype=self.dtype)
                target_ids = cp.empty((0,), dtype=cp.int64)
                source_size = cp.empty((0,), dtype=self.dtype)
                target_size = cp.empty((0,), dtype=self.dtype)
                source_alpha = cp.empty((0,), dtype=self.dtype)
                target_alpha = cp.empty((0,), dtype=self.dtype)
            transition = StepTransition(
                source_points=source_points,
                target_points=target_points,
                source_ids=source_ids,
                target_ids=target_ids,
                source_size=source_size,
                target_size=target_size,
                source_alpha=source_alpha,
                target_alpha=target_alpha,
            )

        self.points = next_points
        self.cell_ids = next_ids
        self.step_index += 1
        self.count_history.append(int(self.points.shape[0]))
        return self.points, transition

    def step(self, action_rule: Optional[ActionRule] = None) -> cp.ndarray:
        """
        Run one timestep.

        Actions:
        - 1: divide (parent is replaced by two daughters)
        - 0: stay
        - -1: die
        """
        next_points, _ = self._step_internal(
            action_rule=action_rule,
            return_transition=False,
            death_animation="none",
        )
        return next_points

    def step_with_transition(
        self,
        action_rule: Optional[ActionRule] = None,
        death_animation: str = "none",
    ) -> StepTransition:
        """Run one timestep and return source->target points for interpolation."""
        _, transition = self._step_internal(
            action_rule=action_rule,
            return_transition=True,
            death_animation=death_animation,
        )
        if transition is None:
            raise RuntimeError("Internal error: expected transition metadata.")
        return transition

    def run(
        self,
        n_steps: int,
        action_rule: Optional[ActionRule] = None,
        log_counts: bool = False,
    ) -> cp.ndarray:
        for _ in range(n_steps):
            if self.points.shape[0] == 0:
                break
            self.step(action_rule=action_rule)
            if log_counts:
                print(f"Step {self.step_index}: {self.points.shape[0]} cells")
        return self.points

    def run_and_save_movie(
        self,
        n_steps: int,
        output_path: str,
        *,
        action_rule: Optional[ActionRule] = None,
        log_counts: bool = False,
        interp_frames: int = 8,
        fps: int = 24,
        show_centers: bool = False,
        color_by: str = "order",
        cmap: str = "viridis",
        opacity: float = 1.0,
        sphere_theta: int = DEFAULT_MOVIE_SPHERE_THETA,
        sphere_phi: int = DEFAULT_MOVIE_SPHERE_PHI,
        show_edges: bool = DEFAULT_MOVIE_SHOW_EDGES,
        edge_color: str = DEFAULT_MOVIE_EDGE_COLOR,
        edge_width: float = DEFAULT_MOVIE_EDGE_WIDTH,
        window_size: tuple[int, int] = (DEFAULT_MOVIE_WIDTH, DEFAULT_MOVIE_HEIGHT),
        macro_block_size: int = DEFAULT_MOVIE_MACRO_BLOCK_SIZE,
        death_animation: str = DEFAULT_MOVIE_DEATH_ANIMATION,
    ) -> cp.ndarray:
        """
        Run simulation and save an MP4 with interpolated division motion.

        Interpolation uses source->target mapping from each timestep:
        - stay: source is previous position
        - divide: both daughters start at parent position
        - die: either removed abruptly or animated (fade/shrink), per death_animation
        Movie rendering uses a fixed camera/bounds from final colony size and
        static-plot style for sphere appearance (color/opacity/tessellation).
        """
        if interp_frames < 1:
            raise ValueError("interp_frames must be >= 1")
        if fps < 1:
            raise ValueError("fps must be >= 1")
        if sphere_theta < 8 or sphere_phi < 8:
            raise ValueError("sphere_theta and sphere_phi must be >= 8")
        if edge_width < 0:
            raise ValueError("edge_width must be >= 0")
        if len(window_size) != 2 or window_size[0] < 320 or window_size[1] < 240:
            raise ValueError("window_size must be (width, height) with reasonable size")
        if macro_block_size < 1:
            raise ValueError("macro_block_size must be >= 1")
        valid_death_modes = {"none", "fade", "shrink", "fade_shrink"}
        if death_animation not in valid_death_modes:
            raise ValueError(f"death_animation must be one of {sorted(valid_death_modes)}")

        out_w, out_h = int(window_size[0]), int(window_size[1])
        if macro_block_size > 1:
            out_w = ((out_w + macro_block_size - 1) // macro_block_size) * macro_block_size
            out_h = ((out_h + macro_block_size - 1) // macro_block_size) * macro_block_size
            if (out_w, out_h) != window_size:
                print(
                    f"Adjusted movie size from {window_size} to {(out_w, out_h)} "
                    f"for macro_block_size={macro_block_size}"
                )

        def to_numpy(arr):
            if _GPU_ENABLED:
                return cp.asnumpy(arr)
            return np.asarray(arr)

        # Pass 1: simulate and capture transitions.
        initial_points_np = to_numpy(self.points.copy())
        initial_ids_np = to_numpy(self.cell_ids.copy())
        transitions_np: list[
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
        ] = []
        for _ in range(n_steps):
            if self.points.shape[0] == 0:
                break
            transition = self.step_with_transition(
                action_rule=action_rule,
                death_animation=death_animation,
            )
            if log_counts:
                print(f"Step {self.step_index}: {self.points.shape[0]} cells")
            transitions_np.append(
                (
                    to_numpy(transition.source_points),
                    to_numpy(transition.target_points),
                    to_numpy(transition.source_ids),
                    to_numpy(transition.target_ids),
                    to_numpy(transition.source_size),
                    to_numpy(transition.target_size),
                    to_numpy(transition.source_alpha),
                    to_numpy(transition.target_alpha),
                )
            )

        final_points_np = to_numpy(self.points.copy())
        movie_max_order = max(1, int(self._next_cell_id - 1))

        # Fixed camera based on final colony extent (fallback to initial if empty).
        framing_points = final_points_np if final_points_np.shape[0] > 0 else initial_points_np
        if framing_points.shape[0] > 0:
            center = framing_points.mean(axis=0)
            d = np.linalg.norm(framing_points - center, axis=1)
            fit_radius = max(float(self.radius), float(d.max() + self.radius))
            mins = framing_points.min(axis=0) - self.radius
            maxs = framing_points.max(axis=0) + self.radius
        else:
            center = np.array([0.0, 0.0, 0.0], dtype=float)
            fit_radius = float(self.radius)
            mins = center - self.radius
            maxs = center + self.radius

        fit_radius *= 1.2  # breathing room in frame
        fixed_bounds = (
            float(mins[0]),
            float(maxs[0]),
            float(mins[1]),
            float(maxs[1]),
            float(mins[2]),
            float(maxs[2]),
        )
        view_dir = np.array([1.0, 1.0, 1.0], dtype=float)
        view_dir /= np.linalg.norm(view_dir)
        cam_distance = max(4.0 * fit_radius, 1.0)
        camera_position = [
            tuple(center + view_dir * cam_distance),
            tuple(center),
            (0.0, 0.0, 1.0),
        ]

        try:
            import pyvista as pv
        except ImportError as exc:  # pragma: no cover - dependency/env specific
            raise ImportError(
                "PyVista is required for movie export. Install with: pip install pyvista"
            ) from exc

        out = Path(output_path)
        if out.parent and not out.parent.exists():
            out.parent.mkdir(parents=True, exist_ok=True)

        base_sphere = pv.Sphere(
            radius=1.0,
            theta_resolution=sphere_theta,
            phi_resolution=sphere_phi,
        )
        pl = pv.Plotter(off_screen=True, window_size=(out_w, out_h))
        pl.set_background("white")
        pl.open_movie(str(out), framerate=fps, macro_block_size=macro_block_size)
        try:
            pl.enable_anti_aliasing()
        except Exception:
            pass
        try:
            pl.enable_lightkit()
        except Exception:
            pass
        pl.camera_position = camera_position
        # Use perspective projection (as in interactive static view) for better depth shading.
        try:
            pl.disable_parallel_projection()
        except Exception:
            try:
                pl.camera.parallel_projection = False
            except Exception:
                pass
        pl.camera.clipping_range = (0.01, cam_distance + 20.0 * fit_radius)

        def write_frame(
            points_np: np.ndarray,
            order_np: np.ndarray,
            size_np: np.ndarray,
            alpha_np: np.ndarray,
        ) -> None:
            pts = np.asarray(points_np, dtype=float)
            order = np.asarray(order_np, dtype=float).reshape(-1)
            size = np.asarray(size_np, dtype=float).reshape(-1)
            alpha = np.asarray(alpha_np, dtype=float).reshape(-1)
            if pts.shape[0] != order.shape[0]:
                raise ValueError("Frame order array length must match frame point count.")
            if pts.shape[0] != size.shape[0]:
                raise ValueError("Frame size array length must match frame point count.")
            if pts.shape[0] != alpha.shape[0]:
                raise ValueError("Frame alpha array length must match frame point count.")
            # Clear actors but preserve renderer/light setup so shading stays consistent.
            try:
                pl.clear_actors()
            except Exception:
                pl.clear()
                try:
                    pl.enable_lightkit()
                except Exception:
                    pass

            if pts.shape[0] > 0:
                size = np.clip(size, 0.0, 1.0)
                alpha = np.clip(alpha, 0.0, 1.0)
                radii = np.full(pts.shape[0], float(self.radius), dtype=float) * size
                cloud = pv.PolyData(pts)
                cloud["scale"] = radii
                spheres = cloud.glyph(geom=base_sphere, scale="scale", orient=False)
                point_opacity = np.repeat(alpha * float(opacity), base_sphere.n_points)
                if color_by == "radius":
                    spheres["val"] = np.repeat(radii, base_sphere.n_points)
                    pl.add_mesh(
                        spheres,
                        scalars="val",
                        cmap=cmap,
                        opacity=point_opacity,
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
                elif color_by == "order":
                    spheres["val"] = np.repeat(order, base_sphere.n_points)
                    pl.add_mesh(
                        spheres,
                        scalars="val",
                        cmap=cmap,
                        clim=(0.0, float(movie_max_order)),
                        opacity=point_opacity,
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
                else:
                    pl.add_mesh(
                        spheres,
                        color="lightsteelblue",
                        opacity=point_opacity,
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

                if show_centers:
                    pl.add_points(
                        cloud,
                        color="black",
                        point_size=6.0,
                        render_points_as_spheres=True,
                    )

            pl.add_axes()
            pl.show_grid(bounds=fixed_bounds)
            pl.camera_position = camera_position
            pl.camera.SetViewAngle(30.0)
            pl.write_frame()

        try:
            # Initial frame
            initial_size = np.ones(initial_points_np.shape[0], dtype=float)
            initial_alpha = np.ones(initial_points_np.shape[0], dtype=float)
            write_frame(initial_points_np, initial_ids_np, initial_size, initial_alpha)

            for src, tgt, src_ids, tgt_ids, src_size, tgt_size, src_alpha, tgt_alpha in transitions_np:
                if src.shape[0] == 0:
                    write_frame(tgt, tgt_ids, tgt_size, tgt_alpha)
                    continue

                for i in range(1, interp_frames + 1):
                    alpha = i / float(interp_frames)
                    frame_pts = (1.0 - alpha) * src + alpha * tgt
                    frame_order = (1.0 - alpha) * src_ids + alpha * tgt_ids
                    frame_size = (1.0 - alpha) * src_size + alpha * tgt_size
                    frame_alpha = (1.0 - alpha) * src_alpha + alpha * tgt_alpha
                    write_frame(frame_pts, frame_order, frame_size, frame_alpha)
        finally:
            pl.close()

        print(f"Movie saved: {out}")
        return self.points

    def points_numpy(self):
        """Return positions as NumPy array for plotting/export."""
        if _GPU_ENABLED:
            return cp.asnumpy(self.points)
        return self.points

    def visualize_pyvista(self, **kwargs) -> None:
        """Render final cells as spheres in PyVista."""
        show_cells_pyvista(self.points_numpy(), cell_radius=self.radius, **kwargs)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D point-growth simulator")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of timesteps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=200_000,
        help="Safety cap to prevent uncontrolled exponential growth",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW,
        help="Show final cells in an interactive PyVista window",
    )
    parser.add_argument(
        "--show-centers",
        action="store_true",
        help="Overlay cell center points in the PyVista window",
    )
    parser.add_argument(
        "--color-by",
        choices=["order", "radius", "solid"],
        default=DEFAULT_COLOR_BY,
        help="Sphere coloring mode in PyVista view (order = list rank oldest->newest)",
    )
    parser.add_argument(
        "--save-movie",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_MOVIE,
        help="Save MP4 movie of the simulation with interpolated division motion",
    )
    parser.add_argument(
        "--movie-path",
        type=str,
        default=DEFAULT_MOVIE_PATH,
        help="Output MP4 path for --save-movie",
    )
    parser.add_argument(
        "--movie-fps",
        type=int,
        default=DEFAULT_MOVIE_FPS,
        help="Frames per second for --save-movie",
    )
    parser.add_argument(
        "--interp-frames",
        type=int,
        default=DEFAULT_INTERP_FRAMES,
        help="Interpolated frames between consecutive timesteps in --save-movie mode",
    )
    parser.add_argument(
        "--movie-width",
        type=int,
        default=DEFAULT_MOVIE_WIDTH,
        help="Movie frame width in pixels",
    )
    parser.add_argument(
        "--movie-height",
        type=int,
        default=DEFAULT_MOVIE_HEIGHT,
        help="Movie frame height in pixels",
    )
    parser.add_argument(
        "--movie-sphere-theta",
        type=int,
        default=DEFAULT_MOVIE_SPHERE_THETA,
        help="Sphere theta resolution for movie rendering",
    )
    parser.add_argument(
        "--movie-sphere-phi",
        type=int,
        default=DEFAULT_MOVIE_SPHERE_PHI,
        help="Sphere phi resolution for movie rendering",
    )
    parser.add_argument(
        "--movie-show-edges",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MOVIE_SHOW_EDGES,
        help="Render mesh edges in movie frames",
    )
    parser.add_argument(
        "--movie-edge-color",
        type=str,
        default=DEFAULT_MOVIE_EDGE_COLOR,
        help="Edge color for movie sphere meshes",
    )
    parser.add_argument(
        "--movie-edge-width",
        type=float,
        default=DEFAULT_MOVIE_EDGE_WIDTH,
        help="Edge line width for movie sphere meshes",
    )
    parser.add_argument(
        "--movie-macro-block-size",
        type=int,
        default=DEFAULT_MOVIE_MACRO_BLOCK_SIZE,
        help="Macro block size for MP4 encoding (16 for compatibility, 1 for exact size)",
    )
    parser.add_argument(
        "--movie-death-animation",
        choices=["none", "fade", "shrink", "fade_shrink"],
        default=DEFAULT_MOVIE_DEATH_ANIMATION,
        help="Dying-cell animation in movies: none, fade, shrink, or fade_shrink",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    sim = CellGrowth3D(seed=args.seed, max_cells=args.max_cells)
    print(backend_summary())
    print("Note: PyVista rendering/movie encoding is separate from simulation kernels.")
    color_by = "none" if args.color_by == "solid" else args.color_by
    if args.save_movie:
        sim.run_and_save_movie(
            args.steps,
            output_path=args.movie_path,
            log_counts=True,
            interp_frames=args.interp_frames,
            fps=args.movie_fps,
            show_centers=args.show_centers,
            color_by=color_by,
            sphere_theta=args.movie_sphere_theta,
            sphere_phi=args.movie_sphere_phi,
            show_edges=args.movie_show_edges,
            edge_color=args.movie_edge_color,
            edge_width=args.movie_edge_width,
            window_size=(args.movie_width, args.movie_height),
            macro_block_size=args.movie_macro_block_size,
            death_animation=args.movie_death_animation,
        )
    else:
        sim.run(args.steps, log_counts=True)

    print(f"CuPy GPU enabled: {_GPU_ENABLED}")
    print(f"Steps executed: {sim.step_index}")
    print(f"Final cell count: {sim.points.shape[0]}")
    print(f"Count history: {sim.count_history}")
    preview = sim.points_numpy()[: min(10, sim.points.shape[0])]
    print("First points:")
    print(preview)
    if args.show:
        sim.visualize_pyvista(show_centers=args.show_centers, color_by=color_by)


if __name__ == "__main__":
    main()
