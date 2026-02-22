"""
3D point-growth simulation with CuPy acceleration.

Model summary:
- Each cell is represented by a 3D point.
- Default behavior is density-regulated fission/homeostasis/death.
- Fission replaces one parent with two daughters.
- Division direction follows a "least resistance" heuristic:
  repulsive vector away from all other cells.
- Post-division relaxation uses a local overlap solver tied to split_distance.
- Uniform-grid spatial hashing is used as an acceleration structure only.
"""

from __future__ import annotations

import argparse
import time
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
DEFAULT_CROWDING_STAY_THRESHOLD = 6
DEFAULT_CROWDING_DEATH_THRESHOLD = 10
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
DEFAULT_TIMING = False
DEFAULT_TIMING_SYNC_GPU = True
DEFAULT_MOVIE_ADAPTIVE_LARGE = True
DEFAULT_MOVIE_LARGE_CELLS_THRESHOLD = 8000
DEFAULT_MOVIE_LARGE_INTERP_FRAMES = 2
DEFAULT_MOVIE_MAX_RENDER_CELLS = 18000
DEFAULT_VIEW_MAX_RENDER_CELLS = 20000

GPU_RAW_KERNELS_SRC = r"""
__device__ inline float maxf_(const float a, const float b) {
    return a > b ? a : b;
}

__device__ inline void pseudo_unit_dir(const int i, const int j, float* ux, float* uy, float* uz) {
    unsigned int h = ((unsigned int)(i + 1) * 73856093u) ^ ((unsigned int)(j + 1) * 19349663u);
    float x = ((float)(h & 1023u) / 511.5f) - 1.0f;
    float y = ((float)((h >> 10) & 1023u) / 511.5f) - 1.0f;
    float z = ((float)((h >> 20) & 1023u) / 511.5f) - 1.0f;
    float n = sqrtf(x * x + y * y + z * z);
    if (n < 1e-8f) {
        x = 1.0f;
        y = 0.0f;
        z = 0.0f;
        n = 1.0f;
    }
    *ux = x / n;
    *uy = y / n;
    *uz = z / n;
}

extern "C" __global__
void neighbor_count_kernel(
    const float* pos,
    const long long* sort_idx,
    const long long* start_lut,
    const int* count_lut,
    const float* origin,
    const long long* min_cell,
    const long long* max_cell,
    const float cell_size,
    const long long stride_x,
    const long long stride_y,
    const int span,
    const float radius2,
    const int n,
    int* out_counts
) {
    const int i = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= n) return;

    const float xi = pos[3 * i + 0];
    const float yi = pos[3 * i + 1];
    const float zi = pos[3 * i + 2];

    const long long cix = (long long)floorf((xi - origin[0]) / cell_size);
    const long long ciy = (long long)floorf((yi - origin[1]) / cell_size);
    const long long ciz = (long long)floorf((zi - origin[2]) / cell_size);

    int count = 0;
    for (int dx = -span; dx <= span; ++dx) {
        const long long nx = cix + (long long)dx;
        if (nx < min_cell[0] || nx > max_cell[0]) continue;
        for (int dy = -span; dy <= span; ++dy) {
            const long long ny = ciy + (long long)dy;
            if (ny < min_cell[1] || ny > max_cell[1]) continue;
            for (int dz = -span; dz <= span; ++dz) {
                const long long nz = ciz + (long long)dz;
                if (nz < min_cell[2] || nz > max_cell[2]) continue;

                const long long key = (nx - min_cell[0]) * stride_x + (ny - min_cell[1]) * stride_y + (nz - min_cell[2]);
                const long long start = start_lut[key];
                if (start < 0) continue;
                const int c = count_lut[key];
                for (int t = 0; t < c; ++t) {
                    const int j = (int)sort_idx[start + (long long)t];
                    if (j == i) continue;
                    const float dx_ = pos[3 * j + 0] - xi;
                    const float dy_ = pos[3 * j + 1] - yi;
                    const float dz_ = pos[3 * j + 2] - zi;
                    const float d2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                    if (d2 <= radius2) {
                        count += 1;
                    }
                }
            }
        }
    }
    out_counts[i] = count;
}

extern "C" __global__
void local_resultant_kernel(
    const float* pos,
    const long long* sort_idx,
    const long long* start_lut,
    const int* count_lut,
    const float* origin,
    const long long* min_cell,
    const long long* max_cell,
    const float cell_size,
    const long long stride_x,
    const long long stride_y,
    const int span,
    const float radius,
    const float radius2,
    const float sigma,
    const int weight_mode,
    const int n,
    const float eps,
    float* out_vhat
) {
    const int i = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= n) return;

    const float xi = pos[3 * i + 0];
    const float yi = pos[3 * i + 1];
    const float zi = pos[3 * i + 2];

    const long long cix = (long long)floorf((xi - origin[0]) / cell_size);
    const long long ciy = (long long)floorf((yi - origin[1]) / cell_size);
    const long long ciz = (long long)floorf((zi - origin[2]) / cell_size);

    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;

    for (int dx = -span; dx <= span; ++dx) {
        const long long nx = cix + (long long)dx;
        if (nx < min_cell[0] || nx > max_cell[0]) continue;
        for (int dy = -span; dy <= span; ++dy) {
            const long long ny = ciy + (long long)dy;
            if (ny < min_cell[1] || ny > max_cell[1]) continue;
            for (int dz = -span; dz <= span; ++dz) {
                const long long nz = ciz + (long long)dz;
                if (nz < min_cell[2] || nz > max_cell[2]) continue;

                const long long key = (nx - min_cell[0]) * stride_x + (ny - min_cell[1]) * stride_y + (nz - min_cell[2]);
                const long long start = start_lut[key];
                if (start < 0) continue;
                const int c = count_lut[key];
                for (int t = 0; t < c; ++t) {
                    const int j = (int)sort_idx[start + (long long)t];
                    if (j == i) continue;
                    const float dx_ = pos[3 * j + 0] - xi;
                    const float dy_ = pos[3 * j + 1] - yi;
                    const float dz_ = pos[3 * j + 2] - zi;
                    const float d2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                    if (d2 > radius2) continue;

                    const float r = sqrtf(d2);
                    float w = 1.0f;
                    if (weight_mode == 1) {
                        w = maxf_(0.0f, 1.0f - r / radius);
                    } else if (weight_mode == 2) {
                        const float q = r / maxf_(sigma, eps);
                        w = expf(-(q * q));
                    }
                    vx += w * dx_;
                    vy += w * dy_;
                    vz += w * dz_;
                }
            }
        }
    }

    const float vn = sqrtf(vx * vx + vy * vy + vz * vz);
    if (vn > eps) {
        out_vhat[3 * i + 0] = vx / vn;
        out_vhat[3 * i + 1] = vy / vn;
        out_vhat[3 * i + 2] = vz / vn;
    } else {
        out_vhat[3 * i + 0] = 0.0f;
        out_vhat[3 * i + 1] = 0.0f;
        out_vhat[3 * i + 2] = 0.0f;
    }
}

extern "C" __global__
void overlap_displacement_kernel(
    const float* pos,
    const long long* sort_idx,
    const long long* start_lut,
    const int* count_lut,
    const float* origin,
    const long long* min_cell,
    const long long* max_cell,
    const float cell_size,
    const long long stride_x,
    const long long stride_y,
    const int span,
    const float candidate_radius2,
    const float target_min_dist,
    const float overlap_tol,
    const float eps,
    const int n,
    float* out_disp,
    int* out_hits
) {
    const int i = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= n) return;

    const float xi = pos[3 * i + 0];
    const float yi = pos[3 * i + 1];
    const float zi = pos[3 * i + 2];

    const long long cix = (long long)floorf((xi - origin[0]) / cell_size);
    const long long ciy = (long long)floorf((yi - origin[1]) / cell_size);
    const long long ciz = (long long)floorf((zi - origin[2]) / cell_size);

    float sx = 0.0f;
    float sy = 0.0f;
    float sz = 0.0f;
    int hits = 0;

    for (int dx = -span; dx <= span; ++dx) {
        const long long nx = cix + (long long)dx;
        if (nx < min_cell[0] || nx > max_cell[0]) continue;
        for (int dy = -span; dy <= span; ++dy) {
            const long long ny = ciy + (long long)dy;
            if (ny < min_cell[1] || ny > max_cell[1]) continue;
            for (int dz = -span; dz <= span; ++dz) {
                const long long nz = ciz + (long long)dz;
                if (nz < min_cell[2] || nz > max_cell[2]) continue;

                const long long key = (nx - min_cell[0]) * stride_x + (ny - min_cell[1]) * stride_y + (nz - min_cell[2]);
                const long long start = start_lut[key];
                if (start < 0) continue;
                const int c = count_lut[key];
                for (int t = 0; t < c; ++t) {
                    const int j = (int)sort_idx[start + (long long)t];
                    if (j == i) continue;

                    const float dx_ = xi - pos[3 * j + 0];
                    const float dy_ = yi - pos[3 * j + 1];
                    const float dz_ = zi - pos[3 * j + 2];
                    const float d2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                    if (d2 > candidate_radius2) continue;

                    const float d = sqrtf(d2);
                    if (d >= target_min_dist - overlap_tol) continue;

                    float ux, uy, uz;
                    if (d > eps) {
                        const float inv = 1.0f / d;
                        ux = dx_ * inv;
                        uy = dy_ * inv;
                        uz = dz_ * inv;
                    } else {
                        pseudo_unit_dir(i, j, &ux, &uy, &uz);
                    }

                    const float pen = target_min_dist - d;
                    const float scale = 0.5f * pen;
                    sx += scale * ux;
                    sy += scale * uy;
                    sz += scale * uz;
                    hits += 1;
                }
            }
        }
    }

    out_disp[3 * i + 0] = sx;
    out_disp[3 * i + 1] = sy;
    out_disp[3 * i + 2] = sz;
    out_hits[i] = hits;
}
"""


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
    max_render_cells: Optional[int] = DEFAULT_VIEW_MAX_RENDER_CELLS,
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

    original_n = int(pts.shape[0])
    if max_render_cells is not None and max_render_cells > 0 and pts.shape[0] > max_render_cells:
        order_idx = np.arange(pts.shape[0], dtype=np.int64)
        pick = np.linspace(0, order_idx.size - 1, int(max_render_cells), dtype=np.int64)
        keep = order_idx[pick]
        pts = pts[keep]
        print(f"Static view: rendering {pts.shape[0]} sampled cells out of {original_n}.")

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
class GridStruct:
    """
    Uniform-grid spatial hash index for candidate neighbor generation.

    Grid is acceleration only; positions remain continuous and exact distances
    are used for all physical interactions.
    """

    origin: np.ndarray
    keys_sorted: np.ndarray
    sort_idx: np.ndarray
    unique_keys: np.ndarray
    start_idx: np.ndarray
    counts: np.ndarray
    positions_sorted: np.ndarray
    min_cell: np.ndarray
    max_cell: np.ndarray
    stride_x: int
    stride_y: int
    cell_size: float
    positions: np.ndarray
    n_cells: int
    cell_start_lut: Optional[np.ndarray] = None
    cell_count_lut: Optional[np.ndarray] = None


@dataclass
class CellGrowth3D:
    """GPU-accelerated 3D growth simulation for point-based cells."""

    radius: float = 1.0
    split_distance: float = 1.5
    neighborhood_radius_factor: float = 2.5
    crowding_stay_threshold: int = DEFAULT_CROWDING_STAY_THRESHOLD
    crowding_death_threshold: int = DEFAULT_CROWDING_DEATH_THRESHOLD
    fast_neighbors: bool = True
    grid_cell_size: Optional[float] = None
    overlap_margin: float = 0.05
    density_radius: Optional[float] = None
    R_sense: Optional[float] = None
    division_direction_mode: str = "tangential"
    neighbor_weight: str = "linear"
    radial_sign: str = "outward"
    eps: float = 1e-12
    enforce_non_overlap: bool = True
    overlap_relax_iters: int = 8
    overlap_tol: float = 1e-4
    seed: int = 42
    max_cells: Optional[int] = None
    dtype: str = "float32"
    enable_step_timing: bool = False
    sync_timing_gpu: bool = True

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be > 0")
        if self.split_distance <= 0:
            raise ValueError("split_distance must be > 0")
        if self.neighborhood_radius_factor <= 0:
            raise ValueError("neighborhood_radius_factor must be > 0")
        if self.crowding_stay_threshold < 0 or self.crowding_death_threshold < 0:
            raise ValueError("crowding thresholds must be >= 0")
        if self.crowding_death_threshold <= self.crowding_stay_threshold:
            raise ValueError("crowding_death_threshold must be > crowding_stay_threshold")
        if self.overlap_margin < 0:
            raise ValueError("overlap_margin must be >= 0")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")

        valid_dir_modes = {"least_resistance", "tangential", "radial"}
        if self.division_direction_mode not in valid_dir_modes:
            raise ValueError(f"division_direction_mode must be one of {sorted(valid_dir_modes)}")
        valid_weight_modes = {"uniform", "linear", "gaussian"}
        if self.neighbor_weight not in valid_weight_modes:
            raise ValueError(f"neighbor_weight must be one of {sorted(valid_weight_modes)}")
        valid_radial = {"inward", "outward", "random"}
        if self.radial_sign not in valid_radial:
            raise ValueError(f"radial_sign must be one of {sorted(valid_radial)}")

        self.overlap_cutoff = float(2.0 * self.radius * (1.0 + self.overlap_margin))
        if self.density_radius is None:
            self.density_radius = float(self.neighborhood_radius_factor * self.radius)
        if self.density_radius <= 0:
            raise ValueError("density_radius must be > 0")
        if self.R_sense is None:
            self.R_sense = float(self.density_radius)
        if self.R_sense <= 0:
            raise ValueError("R_sense must be > 0")
        if self.grid_cell_size is None:
            self.grid_cell_size = float(max(self.overlap_cutoff, self.density_radius, self.R_sense))
        if self.grid_cell_size <= 0:
            raise ValueError("grid_cell_size must be > 0")

        self.points = cp.zeros((1, 3), dtype=self.dtype)
        self.cell_ids = cp.zeros((1,), dtype=cp.int64)
        self._next_cell_id = 1
        self.step_index = 0
        self.count_history = [1]
        self._rng = cp.random.RandomState(self.seed)
        self._rng_np = np.random.RandomState(self.seed)
        self._offset_cache_np: dict[int, np.ndarray] = {}
        self._offset_cache_cp: dict[int, cp.ndarray] = {}
        self.last_step_timing: Optional[dict[str, float]] = None
        self.step_timing_history: list[dict[str, float]] = []
        self._gpu_kernels_ready = False
        self._gpu_kernel_error: Optional[str] = None
        self._init_gpu_kernels()

    @staticmethod
    def always_divide_rule(points: cp.ndarray, step: int) -> cp.ndarray:
        """Action rule: all cells divide (1 = divide, 0 = stay, -1 = die)."""
        return cp.ones(points.shape[0], dtype=cp.int8)

    def _as_numpy(self, arr: cp.ndarray) -> np.ndarray:
        if _GPU_ENABLED:
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def _is_gpu_array(self, arr) -> bool:
        return bool(_GPU_ENABLED and isinstance(arr, cp.ndarray))

    def _xp_of(self, arr):
        return cp if self._is_gpu_array(arr) else np

    @staticmethod
    def _to_float(x) -> float:
        try:
            return float(x.item())
        except Exception:
            return float(x)

    @staticmethod
    def _to_int(x) -> int:
        try:
            return int(x.item())
        except Exception:
            return int(x)

    def _any_true(self, mask, xp_module) -> bool:
        if xp_module is cp:
            return bool(mask.any().item())
        return bool(np.any(mask))

    def _sync_for_timing(self) -> None:
        if not (_GPU_ENABLED and self.sync_timing_gpu):
            return
        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass

    def _format_step_timing(self, timing: dict[str, float]) -> str:
        parts = []
        order = [
            "rule",
            "masks",
            "stay",
            "divide",
            "death_anim",
            "assemble",
            "overlap",
            "transition",
            "commit",
        ]
        for key in order:
            if key in timing:
                parts.append(f"{key}={timing[key] * 1e3:.2f}ms")
        details = " | ".join(parts) if parts else "no stage data"
        step_num = int(timing.get("step", -1))
        n_before = int(timing.get("n_before", -1))
        n_after = int(timing.get("n_after", -1))
        total_ms = timing.get("total", 0.0) * 1e3
        return (
            f"Timing step {step_num}: total={total_ms:.2f}ms "
            f"(cells {n_before}->{n_after}) | {details}"
        )

    def _init_gpu_kernels(self) -> None:
        self._gpu_kernels_ready = False
        self._gpu_kernel_error = None
        self._kernel_neighbor_count = None
        self._kernel_local_resultant = None
        self._kernel_overlap = None
        if not _GPU_ENABLED:
            return
        try:
            self._kernel_neighbor_count = cp.RawKernel(
                GPU_RAW_KERNELS_SRC, "neighbor_count_kernel"
            )
            self._kernel_local_resultant = cp.RawKernel(
                GPU_RAW_KERNELS_SRC, "local_resultant_kernel"
            )
            self._kernel_overlap = cp.RawKernel(
                GPU_RAW_KERNELS_SRC, "overlap_displacement_kernel"
            )
            self._gpu_kernels_ready = True
        except Exception as exc:
            self._gpu_kernel_error = str(exc)
            self._gpu_kernels_ready = False

    def _gpu_fast_path_available(self, arr) -> bool:
        if not _GPU_ENABLED:
            return False
        if not self._is_gpu_array(arr):
            return False
        if not getattr(self, "_gpu_kernels_ready", False):
            return False
        return bool(arr.dtype == cp.float32)

    def _disable_gpu_kernels(self, exc: Exception) -> None:
        self._gpu_kernels_ready = False
        self._gpu_kernel_error = str(exc)
        self._kernel_neighbor_count = None
        self._kernel_local_resultant = None
        self._kernel_overlap = None
        print(f"GPU raw kernels disabled at runtime, falling back: {exc}")

    @staticmethod
    def _launch_cfg_1d(n: int, threads: int = 128) -> tuple[tuple[int], tuple[int]]:
        blocks = max(1, (int(n) + threads - 1) // threads)
        return (blocks,), (threads,)

    def _ensure_grid_lookup(self, grid: GridStruct, xp_module):
        if grid.cell_start_lut is not None and grid.cell_count_lut is not None:
            return grid.cell_start_lut, grid.cell_count_lut

        n_cells = int(grid.n_cells)
        start_lut = xp_module.full((n_cells,), -1, dtype=xp_module.int64)
        count_lut = xp_module.zeros((n_cells,), dtype=xp_module.int32)
        if grid.unique_keys.size > 0:
            start_lut[grid.unique_keys] = grid.start_idx.astype(xp_module.int64, copy=False)
            count_lut[grid.unique_keys] = grid.counts.astype(xp_module.int32, copy=False)

        grid.cell_start_lut = start_lut
        grid.cell_count_lut = count_lut
        return start_lut, count_lut

    def _neighbor_offsets(self, span: int, xp_module):
        if xp_module is cp:
            cached = self._offset_cache_cp.get(span)
            if cached is not None:
                return cached
            vals = np.arange(-span, span + 1, dtype=np.int64)
            off_np = np.stack(np.meshgrid(vals, vals, vals, indexing="ij"), axis=-1).reshape(-1, 3)
            off_cp = cp.asarray(off_np)
            self._offset_cache_cp[span] = off_cp
            return off_cp

        cached = self._offset_cache_np.get(span)
        if cached is not None:
            return cached
        vals = np.arange(-span, span + 1, dtype=np.int64)
        off = np.stack(np.meshgrid(vals, vals, vals, indexing="ij"), axis=-1).reshape(-1, 3)
        self._offset_cache_np[span] = off
        return off

    def _expand_cell_ranges(self, starts, counts, xp_module):
        total = self._to_int(counts.sum())
        if total <= 0:
            return xp_module.zeros((0,), dtype=xp_module.int64)

        try:
            prefix = xp_module.cumsum(counts) - counts
            return (
                xp_module.repeat(starts, counts)
                + xp_module.arange(total, dtype=xp_module.int64)
                - xp_module.repeat(prefix, counts)
            )
        except Exception:
            chunks = []
            n_ranges = self._to_int(starts.shape[0])
            for idx in range(n_ranges):
                start = self._to_int(starts[idx])
                count = self._to_int(counts[idx])
                if count > 0:
                    chunks.append(xp_module.arange(start, start + count, dtype=xp_module.int64))
            if not chunks:
                return xp_module.zeros((0,), dtype=xp_module.int64)
            return xp_module.concatenate(chunks, axis=0)

    def _random_unit_vectors_np(self, n: int) -> np.ndarray:
        vec = self._rng_np.normal(0.0, 1.0, size=(n, 3)).astype(float, copy=False)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.maximum(norm, self.eps)

    def _random_unit_vectors_backend(self, n: int, xp_module, dtype):
        if n <= 0:
            return xp_module.zeros((0, 3), dtype=dtype)
        if xp_module is cp:
            vec = self._rng.normal(0.0, 1.0, size=(n, 3)).astype(dtype, copy=False)
            norm = cp.linalg.norm(vec, axis=1, keepdims=True)
            return vec / cp.maximum(norm, self.eps)
        vec = self._rng_np.normal(0.0, 1.0, size=(n, 3)).astype(dtype, copy=False)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.maximum(norm, self.eps)

    def _point_to_cell_coord(self, point, origin, cell_size: Optional[float] = None):
        xp_module = self._xp_of(point)
        use_cell_size = float(self.grid_cell_size if cell_size is None else cell_size)
        rel = (point - origin) / use_cell_size
        return xp_module.floor(rel).astype(xp_module.int64, copy=False)

    def _cell_key(self, cell_coord, grid: GridStruct) -> Optional[int]:
        """
        Convert integer cell coordinate to packed key used by the sorted index.

        Grid is acceleration only; positions are continuous and no quantization
        is applied to geometry beyond candidate lookup.
        """
        xp_module = self._xp_of(cell_coord)
        if self._any_true(cell_coord < grid.min_cell, xp_module) or self._any_true(
            cell_coord > grid.max_cell, xp_module
        ):
            return None
        off = cell_coord - grid.min_cell
        key = self._to_int(off[0] * grid.stride_x + off[1] * grid.stride_y + off[2])
        return key

    def _build_grid(self, positions, cell_size: Optional[float] = None) -> GridStruct:
        """
        Build a uniform-grid spatial hash:
        (origin, keys_sorted, sort_idx, unique_keys, start_idx, counts, positions_sorted).
        """
        xp_module = self._xp_of(positions)
        use_cell_size = float(self.grid_cell_size if cell_size is None else cell_size)
        n = positions.shape[0]
        if n == 0:
            zeros_i = xp_module.zeros((0,), dtype=xp_module.int64)
            zeros_p = xp_module.zeros((0, 3), dtype=float)
            z3 = xp_module.zeros(3, dtype=xp_module.int64)
            origin = xp_module.zeros(3, dtype=float)
            return GridStruct(
                origin=origin,
                keys_sorted=zeros_i,
                sort_idx=zeros_i,
                unique_keys=zeros_i,
                start_idx=zeros_i,
                counts=zeros_i,
                positions_sorted=zeros_p,
                min_cell=z3.copy(),
                max_cell=z3.copy(),
                stride_x=1,
                stride_y=1,
                cell_size=use_cell_size,
                positions=zeros_p,
                n_cells=0,
            )

        pos = positions.astype(self.dtype, copy=False)
        origin = pos.min(axis=0)
        cell_coords = xp_module.floor((pos - origin) / use_cell_size).astype(
            xp_module.int64
        )
        min_cell = cell_coords.min(axis=0)
        max_cell = cell_coords.max(axis=0)
        off = cell_coords - min_cell
        ranges = (max_cell - min_cell + 1).astype(xp_module.int64)
        stride_y = self._to_int(ranges[2])
        stride_x = self._to_int(ranges[1] * ranges[2])
        n_cells = self._to_int(ranges[0] * ranges[1] * ranges[2])
        keys = off[:, 0] * stride_x + off[:, 1] * stride_y + off[:, 2]

        sort_idx = xp_module.argsort(keys)
        keys_sorted = keys[sort_idx]
        positions_sorted = pos[sort_idx]
        unique_keys, start_idx, counts = xp_module.unique(
            keys_sorted,
            return_index=True,
            return_counts=True,
        )

        return GridStruct(
            origin=origin,
            keys_sorted=keys_sorted.astype(xp_module.int64, copy=False),
            sort_idx=sort_idx.astype(xp_module.int64, copy=False),
            unique_keys=unique_keys.astype(xp_module.int64, copy=False),
            start_idx=start_idx.astype(xp_module.int64, copy=False),
            counts=counts.astype(xp_module.int64, copy=False),
            positions_sorted=positions_sorted,
            min_cell=min_cell,
            max_cell=max_cell,
            stride_x=stride_x,
            stride_y=stride_y,
            cell_size=use_cell_size,
            positions=pos,
            n_cells=n_cells,
        )

    def _neighbor_data_within_radius(
        self,
        i: int,
        positions,
        grid: GridStruct,
        radius: float,
    ):
        """
        Return neighbor ids, exact displacement vectors (x_j - x_i), and distances.
        Candidate cells come from neighboring bins around particle i;
        filtering is exact Euclidean.
        """
        xp_module = self._xp_of(positions)
        if positions.shape[0] <= 1:
            empty_i = xp_module.zeros((0,), dtype=xp_module.int64)
            empty_v = xp_module.zeros((0, 3), dtype=float)
            empty_d = xp_module.zeros((0,), dtype=float)
            return empty_i, empty_v, empty_d

        p_i = positions[i]
        cell_i = self._point_to_cell_coord(p_i, grid.origin, grid.cell_size)
        span = max(1, int(np.ceil(float(radius) / max(grid.cell_size, self.eps))))

        offsets = self._neighbor_offsets(span, xp_module)
        n_cells = cell_i[None, :] + offsets
        in_bounds = ((n_cells >= grid.min_cell[None, :]) & (n_cells <= grid.max_cell[None, :])).all(
            axis=1
        )
        if not self._any_true(in_bounds, xp_module):
            empty_i = xp_module.zeros((0,), dtype=xp_module.int64)
            empty_v = xp_module.zeros((0, 3), dtype=float)
            empty_d = xp_module.zeros((0,), dtype=float)
            return empty_i, empty_v, empty_d

        valid_cells = n_cells[in_bounds]
        off = valid_cells - grid.min_cell[None, :]
        keys = off[:, 0] * grid.stride_x + off[:, 1] * grid.stride_y + off[:, 2]
        key_pos = xp_module.searchsorted(grid.unique_keys, keys)
        valid_pos = key_pos < grid.unique_keys.shape[0]
        if not self._any_true(valid_pos, xp_module):
            empty_i = xp_module.zeros((0,), dtype=xp_module.int64)
            empty_v = xp_module.zeros((0, 3), dtype=float)
            empty_d = xp_module.zeros((0,), dtype=float)
            return empty_i, empty_v, empty_d

        keys = keys[valid_pos]
        key_pos = key_pos[valid_pos]
        hits = grid.unique_keys[key_pos] == keys
        if not self._any_true(hits, xp_module):
            empty_i = xp_module.zeros((0,), dtype=xp_module.int64)
            empty_v = xp_module.zeros((0, 3), dtype=float)
            empty_d = xp_module.zeros((0,), dtype=float)
            return empty_i, empty_v, empty_d

        key_pos = key_pos[hits]
        starts = grid.start_idx[key_pos].astype(xp_module.int64, copy=False)
        counts = grid.counts[key_pos].astype(xp_module.int64, copy=False)
        idx_sorted = self._expand_cell_ranges(starts, counts, xp_module)
        if idx_sorted.size == 0:
            empty_i = xp_module.zeros((0,), dtype=xp_module.int64)
            empty_v = xp_module.zeros((0, 3), dtype=float)
            empty_d = xp_module.zeros((0,), dtype=float)
            return empty_i, empty_v, empty_d
        cand = grid.sort_idx[idx_sorted]
        cand = cand[cand != i]
        if cand.size == 0:
            empty_i = xp_module.zeros((0,), dtype=xp_module.int64)
            empty_v = xp_module.zeros((0, 3), dtype=float)
            empty_d = xp_module.zeros((0,), dtype=float)
            return empty_i, empty_v, empty_d

        vecs = positions[cand] - p_i
        dist2 = xp_module.sum(vecs * vecs, axis=1)
        mask = dist2 <= float(radius * radius)
        if not self._any_true(mask, xp_module):
            empty_i = xp_module.zeros((0,), dtype=xp_module.int64)
            empty_v = xp_module.zeros((0, 3), dtype=float)
            empty_d = xp_module.zeros((0,), dtype=float)
            return empty_i, empty_v, empty_d

        cand = cand[mask]
        vecs = vecs[mask]
        dists = xp_module.sqrt(dist2[mask])
        return cand.astype(xp_module.int64, copy=False), vecs, dists

    def _neighbors_within_radius(
        self,
        i: int,
        positions,
        grid: GridStruct,
        radius: float,
    ):
        ids, _, _ = self._neighbor_data_within_radius(i, positions, grid, radius)
        return ids

    def _neighbor_counts_within_slow(self, points: cp.ndarray, radius: float) -> cp.ndarray:
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

    def _neighbor_counts_within(self, points: cp.ndarray, radius: float) -> cp.ndarray:
        """Count neighbors within fixed radius (self excluded), grid-accelerated."""
        xp_module = self._xp_of(points)
        n = points.shape[0]
        if n <= 1:
            return xp_module.zeros((n,), dtype=xp_module.int32)
        if not self.fast_neighbors:
            return self._neighbor_counts_within_slow(points, radius)

        if self._gpu_fast_path_available(points):
            try:
                grid = self._build_grid(points)
                start_lut, count_lut = self._ensure_grid_lookup(grid, cp)
                counts = cp.zeros((n,), dtype=cp.int32)
                span = max(1, int(np.ceil(float(radius) / max(grid.cell_size, self.eps))))
                blocks, threads = self._launch_cfg_1d(n)
                self._kernel_neighbor_count(
                    blocks,
                    threads,
                    (
                        points,
                        grid.sort_idx,
                        start_lut,
                        count_lut,
                        grid.origin,
                        grid.min_cell,
                        grid.max_cell,
                        np.float32(grid.cell_size),
                        np.int64(grid.stride_x),
                        np.int64(grid.stride_y),
                        np.int32(span),
                        np.float32(radius * radius),
                        np.int32(n),
                        counts,
                    ),
                )
                return counts
            except Exception as exc:
                self._disable_gpu_kernels(exc)

        grid = self._build_grid(points)
        counts = xp_module.zeros(n, dtype=xp_module.int32)
        for i in range(n):
            ids = self._neighbors_within_radius(i, points, grid, radius)
            counts[i] = ids.shape[0]
        return counts.astype(xp_module.int32, copy=False)

    def _local_neighbor_resultants(
        self,
        positions,
        grid: GridStruct,
        radius: float,
        weight_mode: str,
    ):
        """
        Compute v_hat(i) from local neighbors using exact distances.
        No direction quantization is introduced.
        """
        xp_module = self._xp_of(positions)
        n = positions.shape[0]
        v_hat = xp_module.zeros((n, 3), dtype=float)
        if n == 0:
            return v_hat

        if self._gpu_fast_path_available(positions):
            try:
                start_lut, count_lut = self._ensure_grid_lookup(grid, cp)
                out = cp.zeros((n, 3), dtype=positions.dtype)
                span = max(1, int(np.ceil(float(radius) / max(grid.cell_size, self.eps))))
                weight_mode_id = {"uniform": 0, "linear": 1, "gaussian": 2}[weight_mode]
                sigma = float(0.5 * radius)
                blocks, threads = self._launch_cfg_1d(n)
                self._kernel_local_resultant(
                    blocks,
                    threads,
                    (
                        positions,
                        grid.sort_idx,
                        start_lut,
                        count_lut,
                        grid.origin,
                        grid.min_cell,
                        grid.max_cell,
                        np.float32(grid.cell_size),
                        np.int64(grid.stride_x),
                        np.int64(grid.stride_y),
                        np.int32(span),
                        np.float32(radius),
                        np.float32(radius * radius),
                        np.float32(sigma),
                        np.int32(weight_mode_id),
                        np.int32(n),
                        np.float32(self.eps),
                        out,
                    ),
                )
                return out
            except Exception as exc:
                self._disable_gpu_kernels(exc)

        sigma = 0.5 * radius
        for i in range(n):
            _, vecs, dists = self._neighbor_data_within_radius(i, positions, grid, radius)
            if dists.size == 0:
                continue
            if weight_mode == "uniform":
                w = xp_module.ones_like(dists)
            elif weight_mode == "linear":
                w = xp_module.maximum(0.0, 1.0 - dists / radius)
            elif weight_mode == "gaussian":
                w = xp_module.exp(-((dists / max(sigma, self.eps)) ** 2))
            else:
                raise ValueError("Unknown neighbor weight mode.")

            v = (w[:, None] * vecs).sum(axis=0)
            vn = self._to_float(xp_module.sqrt(xp_module.sum(v * v)))
            if vn > self.eps:
                v_hat[i] = v / vn
        return v_hat

    def _division_dirs_from_vhat(self, v_hat):
        xp_module = self._xp_of(v_hat)
        n = v_hat.shape[0]
        dirs = xp_module.zeros((n, 3), dtype=float)
        norms = xp_module.sqrt(xp_module.sum(v_hat * v_hat, axis=1))
        tiny = norms <= self.eps

        if self.division_direction_mode == "radial":
            base = v_hat.copy()
            if self.radial_sign == "inward":
                base *= -1.0
            elif self.radial_sign == "random":
                if xp_module is cp:
                    s = xp_module.where(
                        self._rng.random_sample(n) < 0.5,
                        -1.0,
                        1.0,
                    )
                else:
                    s = self._rng_np.choice(np.asarray([-1.0, 1.0], dtype=float), size=n)
                base *= s[:, None]
            dirs = base
            if self._any_true(~tiny, xp_module):
                dirs[~tiny] /= xp_module.maximum(norms[~tiny, None], self.eps)
        elif self.division_direction_mode == "tangential":
            u = self._random_unit_vectors_backend(n, xp_module, v_hat.dtype)
            d = xp_module.cross(v_hat, u)
            dn = xp_module.sqrt(xp_module.sum(d * d, axis=1))
            good = dn > self.eps
            if self._any_true(good, xp_module):
                dirs[good] = d[good] / xp_module.maximum(dn[good, None], self.eps)

            unresolved = (~good) & (~tiny)
            if self._any_true(unresolved, xp_module):
                idx = xp_module.where(unresolved)[0]
                v = v_hat[idx]
                axis = xp_module.zeros_like(v)
                axis[:, 0] = 1.0
                vnorm = xp_module.sqrt(xp_module.sum(v * v, axis=1))
                use_y = xp_module.abs(v[:, 0]) > 0.9 * xp_module.maximum(vnorm, self.eps)
                axis[use_y, 0] = 0.0
                axis[use_y, 1] = 1.0
                d2 = xp_module.cross(v, axis)
                dn2 = xp_module.sqrt(xp_module.sum(d2 * d2, axis=1))
                good2 = dn2 > self.eps
                if self._any_true(good2, xp_module):
                    idx_good = idx[good2]
                    dirs[idx_good] = d2[good2] / xp_module.maximum(dn2[good2, None], self.eps)

        dn = xp_module.sqrt(xp_module.sum(dirs * dirs, axis=1, keepdims=True))
        need_fallback = dn[:, 0] <= self.eps
        if self._any_true(need_fallback, xp_module):
            n_fallback = self._to_int(need_fallback.sum())
            dirs[need_fallback] = self._random_unit_vectors_backend(n_fallback, xp_module, dirs.dtype)
            dn = xp_module.sqrt(xp_module.sum(dirs * dirs, axis=1, keepdims=True))
        dirs = dirs / xp_module.maximum(dn, self.eps)
        return dirs

    def density_regulated_rule(self, points: cp.ndarray, step: int) -> cp.ndarray:
        """
        Default biological rule:
        - neighbors >= crowding_death_threshold: die
        - neighbors > crowding_stay_threshold: stay
        - otherwise: divide
        """
        del step  # Rule currently uses geometry only.
        counts = self._neighbor_counts_within(points, float(self.density_radius))
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
        Iteratively separate locally-overlapping cells with exact distances.

        Grid is acceleration only; points are continuous and never snapped.
        """
        n = pts.shape[0]
        if n < 2:
            return pts

        # Keep relaxation target consistent with fission spacing.
        target_min_dist = float(max(self.split_distance, 1.0 * self.radius))
        candidate_radius = float(target_min_dist * (1.0 + self.overlap_margin))
        overlap_cell_size = float(max(self.eps, min(self.grid_cell_size, candidate_radius)))
        xp_module = self._xp_of(pts)
        pos = pts.astype(self.dtype, copy=True)

        if self._gpu_fast_path_available(pos):
            try:
                span = max(1, int(np.ceil(candidate_radius / max(overlap_cell_size, self.eps))))
                blocks, threads = self._launch_cfg_1d(n)
                for _ in range(max(1, self.overlap_relax_iters)):
                    grid = self._build_grid(pos, cell_size=overlap_cell_size)
                    start_lut, count_lut = self._ensure_grid_lookup(grid, cp)
                    disp = cp.zeros_like(pos)
                    hits = cp.zeros((n,), dtype=cp.int32)
                    self._kernel_overlap(
                        blocks,
                        threads,
                        (
                            pos,
                            grid.sort_idx,
                            start_lut,
                            count_lut,
                            grid.origin,
                            grid.min_cell,
                            grid.max_cell,
                            np.float32(grid.cell_size),
                            np.int64(grid.stride_x),
                            np.int64(grid.stride_y),
                            np.int32(span),
                            np.float32(candidate_radius * candidate_radius),
                            np.float32(target_min_dist),
                            np.float32(self.overlap_tol),
                            np.float32(self.eps),
                            np.int32(n),
                            disp,
                            hits,
                        ),
                    )
                    if not bool(cp.any(hits > 0).item()):
                        break
                    pos += disp
                return pos.astype(self.dtype, copy=False)
            except Exception as exc:
                self._disable_gpu_kernels(exc)

        for _ in range(max(1, self.overlap_relax_iters)):
            moved = False
            grid = self._build_grid(pos, cell_size=overlap_cell_size)
            disp = xp_module.zeros_like(pos)
            for i in range(n):
                nbr_ids, vecs, dists = self._neighbor_data_within_radius(
                    i,
                    pos,
                    grid,
                    candidate_radius,
                )
                if nbr_ids.size == 0:
                    continue

                # Process each pair once.
                mask = nbr_ids > i
                if not self._any_true(mask, xp_module):
                    continue
                nbr_ids = nbr_ids[mask]
                vecs = vecs[mask]
                dists = dists[mask]

                overlap_mask = dists < (target_min_dist - self.overlap_tol)
                if not self._any_true(overlap_mask, xp_module):
                    continue

                moved = True
                nbr_ids = nbr_ids[overlap_mask]
                vecs = vecs[overlap_mask]
                dists = dists[overlap_mask]

                dirs = xp_module.zeros_like(vecs)
                nz = dists > self.eps
                if self._any_true(nz, xp_module):
                    dirs[nz] = -vecs[nz] / dists[nz, None]
                z = ~nz
                if self._any_true(z, xp_module):
                    z_count = self._to_int(z.sum())
                    dirs[z] = self._random_unit_vectors_backend(z_count, xp_module, pos.dtype)

                penetration = target_min_dist - dists
                shifts = 0.5 * penetration[:, None] * dirs
                disp[i] += shifts.sum(axis=0)
                for axis in range(3):
                    xp_module.add.at(disp[:, axis], nbr_ids, -shifts[:, axis])

            if not moved:
                break
            pos += disp

        return pos.astype(self.dtype, copy=False)

    def _step_internal(
        self,
        action_rule: Optional[ActionRule] = None,
        return_transition: bool = False,
        death_animation: str = "none",
        profile_timing: bool = False,
    ) -> tuple[cp.ndarray, Optional[StepTransition]]:
        """Core step logic with optional source->target transition capture."""
        valid_death_modes = {"none", "fade", "shrink", "fade_shrink"}
        if death_animation not in valid_death_modes:
            raise ValueError(f"death_animation must be one of {sorted(valid_death_modes)}")

        timing_enabled = bool(profile_timing or self.enable_step_timing)
        timing: dict[str, float] = {}
        t_prev = 0.0
        t_total_start = 0.0
        step_number = int(self.step_index + 1)
        n_before = int(self.points.shape[0])

        def mark(label: str) -> None:
            nonlocal t_prev
            if not timing_enabled:
                return
            self._sync_for_timing()
            now = time.perf_counter()
            timing[label] = timing.get(label, 0.0) + (now - t_prev)
            t_prev = now

        if timing_enabled:
            self._sync_for_timing()
            t_total_start = time.perf_counter()
            t_prev = t_total_start

        rule = action_rule or self.density_regulated_rule
        actions = rule(self.points, self.step_index)
        if actions.shape != (self.points.shape[0],):
            raise ValueError("Action rule must return a vector shaped (n_cells,)")
        mark("rule")

        divide_mask = actions == 1
        stay_mask = actions == 0
        die_mask = actions == -1

        if bool(cp.any((~divide_mask) & (~stay_mask) & (~die_mask))):
            raise ValueError("Actions must only contain -1, 0, or 1.")
        mark("masks")

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
        mark("stay")

        if bool(cp.any(divide_mask)):
            if self.division_direction_mode == "least_resistance":
                # NOTE: kept for baseline behavior; this remains all-pairs and slow.
                dirs_all = self._least_resistance_directions()
            else:
                grid = self._build_grid(self.points)
                v_hat = self._local_neighbor_resultants(
                    self.points,
                    grid,
                    float(self.R_sense),
                    self.neighbor_weight,
                )
                dirs_all = self._division_dirs_from_vhat(v_hat).astype(self.dtype, copy=False)

            dirs = dirs_all[divide_mask]
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
        mark("divide")

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
        mark("death_anim")

        if next_blocks:
            next_points = cp.concatenate(next_blocks, axis=0)
            next_ids = cp.concatenate(next_id_blocks, axis=0)
            mark("assemble")
            if self.enforce_non_overlap and next_points.shape[0] > 1:
                if timing_enabled:
                    self._sync_for_timing()
                    t_overlap = time.perf_counter()
                next_points = self._resolve_overlaps(next_points)
                if timing_enabled:
                    self._sync_for_timing()
                    now = time.perf_counter()
                    timing["overlap"] = timing.get("overlap", 0.0) + (now - t_overlap)
                    t_prev = now
        else:
            next_points = cp.empty((0, 3), dtype=self.dtype)
            next_ids = cp.empty((0,), dtype=cp.int64)
            mark("assemble")

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

                # Keep movie transitions continuous with simulation state:
                # after overlap relaxation, live-cell targets should match next_points.
                # Dead-cell entries (if animated) remain appended at the end.
                live_count = int(next_points.shape[0])
                if live_count > 0:
                    target_points[:live_count] = next_points
                    target_ids[:live_count] = next_ids
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
        mark("transition")

        self.points = next_points
        self.cell_ids = next_ids
        self.step_index += 1
        self.count_history.append(int(self.points.shape[0]))
        mark("commit")

        if timing_enabled:
            self._sync_for_timing()
            timing["total"] = time.perf_counter() - t_total_start
            timing["step"] = float(step_number)
            timing["n_before"] = float(n_before)
            timing["n_after"] = float(self.points.shape[0])
            self.last_step_timing = timing
            self.step_timing_history.append(dict(timing))
        else:
            self.last_step_timing = None
        return self.points, transition

    def step(
        self,
        action_rule: Optional[ActionRule] = None,
        profile_timing: bool = False,
    ) -> cp.ndarray:
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
            profile_timing=profile_timing,
        )
        return next_points

    def step_with_transition(
        self,
        action_rule: Optional[ActionRule] = None,
        death_animation: str = "none",
        profile_timing: bool = False,
    ) -> StepTransition:
        """Run one timestep and return source->target points for interpolation."""
        _, transition = self._step_internal(
            action_rule=action_rule,
            return_transition=True,
            death_animation=death_animation,
            profile_timing=profile_timing,
        )
        if transition is None:
            raise RuntimeError("Internal error: expected transition metadata.")
        return transition

    def run(
        self,
        n_steps: int,
        action_rule: Optional[ActionRule] = None,
        log_counts: bool = False,
        log_timing: bool = False,
    ) -> cp.ndarray:
        for _ in range(n_steps):
            if self.points.shape[0] == 0:
                break
            self.step(action_rule=action_rule, profile_timing=log_timing)
            if log_counts:
                print(f"Step {self.step_index}: {self.points.shape[0]} cells")
            if log_timing and self.last_step_timing is not None:
                print(self._format_step_timing(self.last_step_timing))
        return self.points

    def run_and_save_movie(
        self,
        n_steps: int,
        output_path: str,
        *,
        action_rule: Optional[ActionRule] = None,
        log_counts: bool = False,
        log_timing: bool = False,
        adaptive_large_render: bool = DEFAULT_MOVIE_ADAPTIVE_LARGE,
        large_cells_threshold: int = DEFAULT_MOVIE_LARGE_CELLS_THRESHOLD,
        large_interp_frames: int = DEFAULT_MOVIE_LARGE_INTERP_FRAMES,
        max_render_cells: Optional[int] = DEFAULT_MOVIE_MAX_RENDER_CELLS,
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
        if large_interp_frames < 1:
            raise ValueError("large_interp_frames must be >= 1")
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
        if large_cells_threshold < 1:
            raise ValueError("large_cells_threshold must be >= 1")
        if max_render_cells is not None and max_render_cells < 1:
            raise ValueError("max_render_cells must be >= 1 when provided")
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
                profile_timing=log_timing,
            )
            if log_counts:
                print(f"Step {self.step_index}: {self.points.shape[0]} cells")
            if log_timing and self.last_step_timing is not None:
                print(self._format_step_timing(self.last_step_timing))
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
        max_frame_cells = int(initial_points_np.shape[0])
        for src, tgt, *_ in transitions_np:
            max_frame_cells = max(max_frame_cells, int(src.shape[0]), int(tgt.shape[0]))

        effective_interp_frames = int(interp_frames)
        if adaptive_large_render and max_frame_cells >= large_cells_threshold:
            effective_interp_frames = min(effective_interp_frames, int(large_interp_frames))
            if effective_interp_frames != interp_frames:
                print(
                    "Adaptive movie: reducing interpolation frames "
                    f"from {interp_frames} to {effective_interp_frames} "
                    f"(max cells/frame={max_frame_cells})."
                )

        effective_max_render_cells = max_render_cells
        if effective_max_render_cells is not None and max_frame_cells > effective_max_render_cells:
            print(
                "Adaptive movie: sampling each frame to "
                f"{int(effective_max_render_cells)} cells "
                f"(max cells/frame={max_frame_cells})."
            )

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

            if (
                effective_max_render_cells is not None
                and pts.shape[0] > int(effective_max_render_cells)
            ):
                cap = int(effective_max_render_cells)
                sorted_idx = np.argsort(order, kind="mergesort")
                pick = np.linspace(0, sorted_idx.size - 1, cap, dtype=np.int64)
                keep = sorted_idx[pick]
                pts = pts[keep]
                order = order[keep]
                size = size[keep]
                alpha = alpha[keep]

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

                for i in range(1, effective_interp_frames + 1):
                    alpha = i / float(effective_interp_frames)
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
    parser.add_argument(
        "--crowding-stay-threshold",
        type=int,
        default=DEFAULT_CROWDING_STAY_THRESHOLD,
        help="If neighbor count is above this value, cell stays (no divide)",
    )
    parser.add_argument(
        "--crowding-death-threshold",
        type=int,
        default=DEFAULT_CROWDING_DEATH_THRESHOLD,
        help="If neighbor count reaches this value, cell dies",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--timing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TIMING,
        help="Print per-step timing breakdown for simulation kernels",
    )
    parser.add_argument(
        "--timing-sync-gpu",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TIMING_SYNC_GPU,
        help="Synchronize GPU before timing marks for accurate timings",
    )
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
        "--view-max-render-cells",
        type=int,
        default=DEFAULT_VIEW_MAX_RENDER_CELLS,
        help=(
            "Maximum cells to render in final static view "
            "(0 disables sampling cap)"
        ),
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
        "--movie-adaptive-large-render",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MOVIE_ADAPTIVE_LARGE,
        help="Adapt movie rendering for large colonies (fewer interp frames/sampling)",
    )
    parser.add_argument(
        "--movie-large-cells-threshold",
        type=int,
        default=DEFAULT_MOVIE_LARGE_CELLS_THRESHOLD,
        help="Cell-count threshold where adaptive movie behavior starts",
    )
    parser.add_argument(
        "--movie-large-interp-frames",
        type=int,
        default=DEFAULT_MOVIE_LARGE_INTERP_FRAMES,
        help="Interp frames used after threshold when adaptive movie mode is on",
    )
    parser.add_argument(
        "--movie-max-render-cells",
        type=int,
        default=DEFAULT_MOVIE_MAX_RENDER_CELLS,
        help="Per-frame cell cap for movie rendering (0 disables cap)",
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
    movie_max_render_cells = None if args.movie_max_render_cells == 0 else args.movie_max_render_cells
    view_max_render_cells = None if args.view_max_render_cells == 0 else args.view_max_render_cells
    sim = CellGrowth3D(
        seed=args.seed,
        max_cells=args.max_cells,
        crowding_stay_threshold=args.crowding_stay_threshold,
        crowding_death_threshold=args.crowding_death_threshold,
        enable_step_timing=args.timing,
        sync_timing_gpu=args.timing_sync_gpu,
    )
    print(backend_summary())
    if _GPU_ENABLED:
        if getattr(sim, "_gpu_kernels_ready", False):
            print("GPU neighbor/overlap raw kernels: enabled")
        else:
            err = getattr(sim, "_gpu_kernel_error", None)
            print(f"GPU neighbor/overlap raw kernels: disabled ({err})")
    print("Note: PyVista rendering/movie encoding is separate from simulation kernels.")
    color_by = "none" if args.color_by == "solid" else args.color_by
    if args.save_movie:
        sim.run_and_save_movie(
            args.steps,
            output_path=args.movie_path,
            log_counts=True,
            log_timing=args.timing,
            adaptive_large_render=args.movie_adaptive_large_render,
            large_cells_threshold=args.movie_large_cells_threshold,
            large_interp_frames=args.movie_large_interp_frames,
            max_render_cells=movie_max_render_cells,
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
        sim.run(args.steps, log_counts=True, log_timing=args.timing)

    print(f"CuPy GPU enabled: {_GPU_ENABLED}")
    print(f"Steps executed: {sim.step_index}")
    print(f"Final cell count: {sim.points.shape[0]}")
    print(f"Count history: {sim.count_history}")
    preview = sim.points_numpy()[: min(10, sim.points.shape[0])]
    print("First points:")
    print(preview)
    if args.show:
        sim.visualize_pyvista(
            show_centers=args.show_centers,
            color_by=color_by,
            max_render_cells=view_max_render_cells,
        )


if __name__ == "__main__":
    main()
