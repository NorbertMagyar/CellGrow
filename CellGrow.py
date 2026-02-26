"""
3D point-growth simulation with CuPy acceleration.

Model summary:
- Each cell is represented by a 3D point.
- Default behavior is density-regulated fission/homeostasis/death.
- Fission replaces one parent with two daughters.
- Division direction follows a "least resistance" heuristic:
  repulsive vector away from all other cells.
- Post-division relaxation uses local spring relaxation around split_distance.
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

# Main simulation controls
#==========================================================================
DEFAULT_STEPS = 30  # Number of simulation timesteps.
DEFAULT_CROWDING_STAY_THRESHOLD = 12  # Neighbor count above which cells stop dividing.
DEFAULT_CROWDING_DEATH_THRESHOLD = 16  # Neighbor count at/above which cells die.
DEFAULT_NEIGHBORHOOD_RADIUS_FACTOR = 2.1  # Multiplier for density_radius relative to cell radius.
DEFAULT_OVERLAP_RELAX_ITERS = 10  # Iterations of local spring relaxer per simulation step.
DEFAULT_DIVISION_DIRECTION_MODE = "tangential"  # Division direction mode; options: least_resistance, tangential, radial.

# I/O knobs
#============================================================================
DEFAULT_SHOW = True  # Show interactive PyVista window at end of run.
DEFAULT_COLOR_BY = "order"  # Static color mode; options: order, radius, solid, u, v, prog, age, pz.
DEFAULT_OUTPUT_STEM = (  # Base filename stem used by default output paths.
    f"cell_growth_"
    f"cr_{DEFAULT_CROWDING_STAY_THRESHOLD}_"
    f"de_{DEFAULT_CROWDING_DEATH_THRESHOLD}_"
    f"dir_{DEFAULT_DIVISION_DIRECTION_MODE}"
)
DEFAULT_SAVE_DATA = True  # Save final state to NPZ by default.
DEFAULT_DATA_PATH = f"{DEFAULT_OUTPUT_STEM}.npz"  # Default NPZ output path.
DEFAULT_SAVE_SNAPSHOT = False  # Save final snapshot image by default.
DEFAULT_SNAPSHOT_PATH = f"{DEFAULT_OUTPUT_STEM}.png"  # Default snapshot output path.
DEFAULT_SAVE_MOVIE = False  # Render and save movie by default.
DEFAULT_MOVIE_PATH = f"{DEFAULT_OUTPUT_STEM}.mp4"  # Default movie output path.
DEFAULT_MOVIE_FPS = 24  # Default movie FPS when not auto-scaled.
DEFAULT_MOVIE_DURATION_SECONDS: Optional[float] = 10  # Target duration (seconds) for auto FPS scaling; set None to disable.
DEFAULT_INTERP_FRAMES = 20  # Interpolated frames between simulation steps in movie mode.
DEFAULT_MOVIE_WIDTH = 1024  # Movie frame width (px).
DEFAULT_MOVIE_HEIGHT = 860  # Movie frame height (px).
DEFAULT_MOVIE_SPHERE_THETA = 16  # Sphere theta tessellation in movie rendering.
DEFAULT_MOVIE_SPHERE_PHI = 16  # Sphere phi tessellation in movie rendering.
DEFAULT_MOVIE_SHOW_EDGES = False  # Render sphere mesh edges in movies.
DEFAULT_MOVIE_EDGE_COLOR = "#000000"  # Edge color when movie edges are enabled.
DEFAULT_MOVIE_EDGE_WIDTH = 0.6  # Edge line width when movie edges are enabled.
DEFAULT_MOVIE_MACRO_BLOCK_SIZE = 16  # MP4 encoder macro block size.
DEFAULT_MOVIE_DEATH_ANIMATION = "shrink"  # Death animation; options: none, fade, shrink, fade_shrink.
DEFAULT_TIMING = False  # Print per-step timing breakdown.
DEFAULT_TIMING_SYNC_GPU = True  # Synchronize GPU before timing marks for accurate timings.
DEFAULT_MOVIE_ADAPTIVE_LARGE = True  # Enable adaptive movie settings for large colonies.
DEFAULT_MOVIE_LARGE_CELLS_THRESHOLD = 8000  # Cell-count threshold for adaptive movie behavior.
DEFAULT_MOVIE_LARGE_INTERP_FRAMES = 2  # Interp frame count used after adaptive threshold.
DEFAULT_MOVIE_MAX_RENDER_CELLS = 18000  # Max cells rendered per movie frame (sampling above this).
DEFAULT_VIEW_MAX_RENDER_CELLS = 20000  # Max cells rendered in static final view.

# Reaction-Diffusion calculation across cells
#==============================================================================
DEFAULT_ENABLE_REACTION_DIFFUSION = True  # Enable reaction-diffusion subsystem.
DEFAULT_RD_START_STEP = 30  # Step index when RD starts affecting dynamics.
DEFAULT_RD_DT = 0.1  # RD integrator timestep. Forward Euler solver
DEFAULT_RD_SUBSTEPS = 30  # RD substeps per cell timestep.
DEFAULT_RD_DU = 0.1  # RD diffusion coefficient for u.
DEFAULT_RD_DV = 0.2  # RD diffusion coefficient for v.
# Values inspired by: https://visualpde.com/nonlinear-physics/gray-scott.html
DEFAULT_GS_F = 0.03  # Gray-Scott parameter a (feed-like term in this formulation).
DEFAULT_GS_K = 0.065  # Gray-Scott parameter b (kill-like term in this formulation).
DEFAULT_RD_MODEL = "gray_scott"  # RD kinetics model; options: gray_scott.
DEFAULT_RD_CLAMP = True  # Clamp RD fields to [0,1] each substep.
DEFAULT_RD_NOISE = 0.0  # Additive RD noise amplitude.
DEFAULT_RD_INIT_MODE = "seed_random_cells"  # RD seeding mode; options: uniform_noise, seed_center, seed_random_cells.
DEFAULT_RD_SEED_AMP = 1.0  # RD seed perturbation amplitude.
DEFAULT_RD_SEED_FRAC = 0.1  # Fraction of seeded cells in seed_random_cells mode.
DEFAULT_RD_PRINT_STATS_EVERY = 50  # RD diagnostics print interval (steps; 0 disables).
DEFAULT_RD_COUPLE_TO_PROG = True  # Map RD signal to division program scalar.
DEFAULT_RD_PROG_FROM = "u"  # Legacy (ignored): program scalar is now always sigmoid(u).
DEFAULT_RD_PROG_GAIN = 0.01  # Legacy (ignored): program scalar is now always sigmoid(u).
DEFAULT_RD_PROG_CENTER = 0.1  # Legacy (ignored): program scalar is now always sigmoid(u).
DEFAULT_DIVIDE_BASE_P = 0.0  # Baseline division probability before RD boost.
DEFAULT_RD_DIVIDE_BOOST = 1  # Strength of RD-driven division boost.
DEFAULT_RD_DIVIDE_CENTER = 0.3  # RD center for division boost response.
DEFAULT_RD_DIVIDE_MIN_P = 0.0  # Lower cap for RD-driven division probability.
DEFAULT_RD_DIVIDE_MAX_P = 1.0  # Upper cap for RD-driven division probability.
DEFAULT_RD_APOPTOSIS_BOOST = 0  # Strength of RD-driven apoptosis boost.
DEFAULT_RD_APOPTOSIS_BASE_P = 0.0  # Baseline RD apoptosis probability.
DEFAULT_RD_APOPTOSIS_CENTER = 0.1  # RD center for apoptosis boost response.
DEFAULT_RD_APOPTOSIS_MIN_P = 0.0  # Lower cap for RD-driven apoptosis probability.
DEFAULT_RD_APOPTOSIS_MAX_P = 0.5  # Upper cap for RD-driven apoptosis probability.
DEFAULT_RD_INTERIOR_PROTECTION = False  # Enable interior-protection modifiers for RD gating.
DEFAULT_RD_INTERIOR_APOPTOSIS_SHIELD = 1.0  # Strength of interior suppression on RD apoptosis.
DEFAULT_RD_INTERIOR_DIVIDE_DAMP = 1.0  # Strength of interior damping on RD division.
DEFAULT_RD_INTERIOR_CROWD_WEIGHT = 0.5  # Crowding weight in interior-score computation.
DEFAULT_ENABLE_APOPTOSIS = False  # Enable age-based apoptosis.
DEFAULT_APOPTOSIS_AGE = 10  # Birth-age threshold for apoptosis.

# Polarity - preferential division axis for cells. Options to align polarity between cells and for RD variables to influence polarity
#====================================================================================
DEFAULT_ENABLE_POLARITY = False  # Enable polarity vector dynamics.
DEFAULT_POLARITY_NOISE = 0.01  # Per-step polarity noise amplitude.
DEFAULT_POLARITY_ALIGN_ALPHA0 = 1.0  # Baseline polarity neighbor-alignment strength.
DEFAULT_POLARITY_ALIGN_ALPHA_U = 1.0  # Extra polarity alignment strength scaled by local u.
DEFAULT_POLARITY_RADIUS: Optional[float] = None  # Polarity neighbor radius (None -> use R_signal).
DEFAULT_POLARITY_PROJECT_TO_TANGENT = False  # Project polarity updates onto local tangent plane.
DEFAULT_POLARITY_USE_U_GRADIENT = True  # Blend local u-gradient into polarity update.
DEFAULT_POLARITY_GRAD_GAIN = 0.4  # Blend gain for u-gradient contribution in polarity update.
DEFAULT_POLARITY_MIX_PREV = 0.0  # Temporal inertia of polarity (0=new only, 1=keep previous).
DEFAULT_FORCE_DIVISION_DIRECTION: Optional[str] = None  # Fixed global division axis as 'x,y,z' string; None disables override.

# The relaxation step (pushing or pulling cells to split_distance) enforces alignment of cells
DEFAULT_RELAX_PROJECTION_MODE = "none"  # Relaxer displacement projection mode; options: none, force_dir, polarity.

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
void neighbor_mean2_kernel(
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
    const float* s0,
    const float* s1,
    float* out_mean0,
    float* out_mean1,
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

    float sum0 = 0.0f;
    float sum1 = 0.0f;
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
                        sum0 += s0[j];
                        sum1 += s1[j];
                        count += 1;
                    }
                }
            }
        }
    }

    out_counts[i] = count;
    if (count > 0) {
        const float inv = 1.0f / (float)count;
        out_mean0[i] = sum0 * inv;
        out_mean1[i] = sum1 * inv;
    } else {
        out_mean0[i] = s0[i];
        out_mean1[i] = s1[i];
    }
}

extern "C" __global__
void polarity_stats_kernel(
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
    const float eps,
    const float* p,
    const float* u,
    float* out_mean_p,
    float* out_grad_u,
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

    float sum_px = 0.0f;
    float sum_py = 0.0f;
    float sum_pz = 0.0f;
    float gx = 0.0f;
    float gy = 0.0f;
    float gz = 0.0f;
    int count = 0;
    const float ui = u[i];
    const float eps2 = eps * eps;

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

                    sum_px += p[3 * j + 0];
                    sum_py += p[3 * j + 1];
                    sum_pz += p[3 * j + 2];

                    const float du = u[j] - ui;
                    const float inv_d = rsqrtf(maxf_(d2, eps2));
                    gx += du * dx_ * inv_d;
                    gy += du * dy_ * inv_d;
                    gz += du * dz_ * inv_d;
                    count += 1;
                }
            }
        }
    }

    out_counts[i] = count;
    if (count > 0) {
        const float inv = 1.0f / (float)count;
        out_mean_p[3 * i + 0] = sum_px * inv;
        out_mean_p[3 * i + 1] = sum_py * inv;
        out_mean_p[3 * i + 2] = sum_pz * inv;
    } else {
        out_mean_p[3 * i + 0] = p[3 * i + 0];
        out_mean_p[3 * i + 1] = p[3 * i + 1];
        out_mean_p[3 * i + 2] = p[3 * i + 2];
    }
    out_grad_u[3 * i + 0] = gx;
    out_grad_u[3 * i + 1] = gy;
    out_grad_u[3 * i + 2] = gz;
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
    float* out_vhat,
    float* out_vmag
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
    out_vmag[i] = vn;
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
void least_resistance_kernel(
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
    const float eps,
    float* out_dirs
) {
    const int i = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= n) return;

    const float xi = pos[3 * i + 0];
    const float yi = pos[3 * i + 1];
    const float zi = pos[3 * i + 2];

    const long long cix = (long long)floorf((xi - origin[0]) / cell_size);
    const long long ciy = (long long)floorf((yi - origin[1]) / cell_size);
    const long long ciz = (long long)floorf((zi - origin[2]) / cell_size);

    float rx = 0.0f;
    float ry = 0.0f;
    float rz = 0.0f;

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

                    const float denom = maxf_(d2 * sqrtf(d2), eps);
                    const float inv = 1.0f / denom;
                    rx -= dx_ * inv;
                    ry -= dy_ * inv;
                    rz -= dz_ * inv;
                }
            }
        }
    }

    const float rn = sqrtf(rx * rx + ry * ry + rz * rz);
    if (rn > eps) {
        out_dirs[3 * i + 0] = rx / rn;
        out_dirs[3 * i + 1] = ry / rn;
        out_dirs[3 * i + 2] = rz / rn;
    } else {
        float ux, uy, uz;
        pseudo_unit_dir(i, i + 104729, &ux, &uy, &uz);
        out_dirs[3 * i + 0] = ux;
        out_dirs[3 * i + 1] = uy;
        out_dirs[3 * i + 2] = uz;
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
    const float adhesion_radius,
    const float adhesion_radius2,
    const float rest_dist,
    const float spring_k,
    const float spring_max_step,
    const float relax_tol,
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
                    if (d2 > adhesion_radius2) continue;

                    const float d = sqrtf(d2);

                    float ux, uy, uz;
                    if (d > eps) {
                        const float inv = 1.0f / d;
                        ux = dx_ * inv;
                        uy = dy_ * inv;
                        uz = dz_ * inv;
                    } else {
                        pseudo_unit_dir(i, j, &ux, &uy, &uz);
                    }

                    float scale = 0.0f;
                    if (d < rest_dist - relax_tol) {
                        // Strong local repulsion/projection when too close.
                        const float gap = rest_dist - d;
                        const float corr = fminf(gap, spring_max_step);
                        scale = 0.5f * corr;
                    } else if (d > rest_dist + relax_tol && adhesion_radius > rest_dist + relax_tol) {
                        // Softer adhesion when farther than rest.
                        // Attraction tapers to zero at the adhesion boundary.
                        const float attr_err = d - rest_dist;
                        const float span_attr = maxf_(adhesion_radius - rest_dist, eps);
                        const float frac = attr_err / span_attr;
                        if (frac >= 1.0f) continue;
                        float delta = spring_k * attr_err * (1.0f - frac);
                        if (delta > spring_max_step) delta = spring_max_step;
                        scale = -0.5f * delta;
                    } else {
                        continue;
                    }

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
    scalar_values: Optional[np.ndarray] = None,
    notebook: bool = False,
    max_render_cells: Optional[int] = DEFAULT_VIEW_MAX_RENDER_CELLS,
    snapshot_path: Optional[str] = None,
) -> Optional[Path]:
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

    vals = None if scalar_values is None else np.asarray(scalar_values, dtype=float).reshape(-1)
    if vals is not None and vals.shape[0] != pts.shape[0]:
        raise ValueError("scalar_values length must match points length")

    original_n = int(pts.shape[0])
    if max_render_cells is not None and max_render_cells > 0 and pts.shape[0] > max_render_cells:
        order_idx = np.arange(pts.shape[0], dtype=np.int64)
        pick = np.linspace(0, order_idx.size - 1, int(max_render_cells), dtype=np.int64)
        keep = order_idx[pick]
        pts = pts[keep]
        if vals is not None:
            vals = vals[keep]
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

    off_screen = snapshot_path is not None
    pl = pv.Plotter(notebook=notebook, off_screen=off_screen)
    pl.set_background("white")

    if color_by == "radius":
        spheres["val"] = np.repeat(radii, base_sphere.n_points)
        pl.add_mesh(spheres, scalars="val", cmap=cmap, opacity=opacity, smooth_shading=True)
    elif color_by == "order":
        order = np.arange(pts.shape[0], dtype=float)
        spheres["val"] = np.repeat(order, base_sphere.n_points)
        pl.add_mesh(spheres, scalars="val", cmap=cmap, opacity=opacity, smooth_shading=True)
    elif color_by == "scalar":
        if vals is None:
            raise ValueError("color_by='scalar' requires scalar_values")
        spheres["val"] = np.repeat(vals.astype(float, copy=False), base_sphere.n_points)
        pl.add_mesh(spheres, scalars="val", cmap=cmap, opacity=opacity, smooth_shading=True)
    else:
        pl.add_mesh(spheres, color="lightsteelblue", opacity=opacity, smooth_shading=True)

    if show_centers:
        pl.add_points(cloud, color="black", point_size=centers_size, render_points_as_spheres=True)

    pl.add_axes()
    pl.show_grid()
    if snapshot_path is not None:
        out = Path(snapshot_path)
        if out.parent and not out.parent.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(out))
        pl.close()
        print(f"Saved snapshot: {out}")
        return out

    pl.show()
    return None


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
    neighborhood_radius_factor: float = DEFAULT_NEIGHBORHOOD_RADIUS_FACTOR
    crowding_stay_threshold: int = DEFAULT_CROWDING_STAY_THRESHOLD
    crowding_death_threshold: int = DEFAULT_CROWDING_DEATH_THRESHOLD
    fast_neighbors: bool = True
    grid_cell_size: Optional[float] = None
    overlap_margin: float = 0.05
    density_radius: Optional[float] = None
    R_sense: Optional[float] = None
    division_direction_mode: str = DEFAULT_DIVISION_DIRECTION_MODE
    neighbor_weight: str = "linear"
    radial_sign: str = "outward"
    eps: float = 1e-12
    enforce_non_overlap: bool = True
    overlap_relax_iters: int = DEFAULT_OVERLAP_RELAX_ITERS
    overlap_tol: float = 1e-4
    rest_distance_factor: Optional[float] = None
    adhesion_radius_factor: Optional[float] = None
    spring_k: float = 0.05
    spring_max_step: Optional[float] = None
    seed: int = 42
    max_cells: Optional[int] = None
    dtype: str = "float32"
    enable_step_timing: bool = False
    sync_timing_gpu: bool = True
    program_surface_gain: float = DEFAULT_PROGRAM_SURFACE_GAIN
    program_crowd_gain: float = DEFAULT_PROGRAM_CROWD_GAIN
    program_noise: float = DEFAULT_RD_NOISE
    program_sigmoid_center: float = 0.0
    program_sigmoid_slope: float = 1.0
    program_radial_sign: str = "outward"
    program_divide_boost: float = DEFAULT_RD_DIVIDE_BOOST
    program_divide_min_p: float = DEFAULT_RD_DIVIDE_MIN_P
    program_divide_max_p: float = DEFAULT_RD_DIVIDE_MAX_P
    print_program_summary: bool = True
    enable_apoptosis: bool = DEFAULT_ENABLE_APOPTOSIS
    apoptosis_age: int = DEFAULT_APOPTOSIS_AGE
    enable_reaction_diffusion: bool = DEFAULT_ENABLE_REACTION_DIFFUSION
    R_signal: Optional[float] = None
    rd_dt: float = DEFAULT_RD_DT
    rd_substeps: int = DEFAULT_RD_SUBSTEPS
    Du: float = DEFAULT_RD_DU
    Dv: float = DEFAULT_RD_DV
    rd_model: str = DEFAULT_RD_MODEL
    gs_F: float = DEFAULT_GS_F
    gs_k: float = DEFAULT_GS_K
    rd_clamp: bool = DEFAULT_RD_CLAMP
    rd_noise: float = DEFAULT_RD_NOISE
    rd_init_mode: str = DEFAULT_RD_INIT_MODE
    rd_seed_amp: float = DEFAULT_RD_SEED_AMP
    rd_seed_frac: float = DEFAULT_RD_SEED_FRAC
    rd_print_stats_every: int = DEFAULT_RD_PRINT_STATS_EVERY
    rd_couple_to_prog: bool = DEFAULT_RD_COUPLE_TO_PROG
    rd_prog_from: str = DEFAULT_RD_PROG_FROM
    rd_prog_gain: float = DEFAULT_RD_PROG_GAIN
    rd_prog_center: float = DEFAULT_RD_PROG_CENTER
    rd_divide_base_p: float = DEFAULT_DIVIDE_BASE_P
    rd_divide_boost: float = DEFAULT_RD_DIVIDE_BOOST
    rd_divide_center: float = DEFAULT_RD_DIVIDE_CENTER
    rd_divide_min_p: float = DEFAULT_RD_DIVIDE_MIN_P
    rd_divide_max_p: float = DEFAULT_RD_DIVIDE_MAX_P
    rd_start_step: int = DEFAULT_RD_START_STEP
    rd_apoptosis_boost: float = DEFAULT_RD_APOPTOSIS_BOOST
    rd_apoptosis_base_p: float = DEFAULT_RD_APOPTOSIS_BASE_P
    rd_apoptosis_center: float = DEFAULT_RD_APOPTOSIS_CENTER
    rd_apoptosis_min_p: float = DEFAULT_RD_APOPTOSIS_MIN_P
    rd_apoptosis_max_p: float = DEFAULT_RD_APOPTOSIS_MAX_P
    rd_interior_protection: bool = DEFAULT_RD_INTERIOR_PROTECTION
    rd_interior_apoptosis_shield: float = DEFAULT_RD_INTERIOR_APOPTOSIS_SHIELD
    rd_interior_divide_damp: float = DEFAULT_RD_INTERIOR_DIVIDE_DAMP
    rd_interior_crowd_weight: float = DEFAULT_RD_INTERIOR_CROWD_WEIGHT
    enable_polarity: bool = DEFAULT_ENABLE_POLARITY
    polarity_noise: float = DEFAULT_POLARITY_NOISE
    polarity_align_alpha0: float = DEFAULT_POLARITY_ALIGN_ALPHA0
    polarity_align_alpha_u: float = DEFAULT_POLARITY_ALIGN_ALPHA_U
    polarity_radius: Optional[float] = DEFAULT_POLARITY_RADIUS
    polarity_project_to_tangent: bool = DEFAULT_POLARITY_PROJECT_TO_TANGENT
    polarity_use_u_gradient: bool = DEFAULT_POLARITY_USE_U_GRADIENT
    polarity_grad_gain: float = DEFAULT_POLARITY_GRAD_GAIN
    polarity_mix_prev: float = DEFAULT_POLARITY_MIX_PREV
    force_division_direction: Optional[str] = DEFAULT_FORCE_DIVISION_DIRECTION
    relax_projection_mode: str = DEFAULT_RELAX_PROJECTION_MODE

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
        if self.spring_k <= 0:
            raise ValueError("spring_k must be > 0")
        if self.apoptosis_age < 0:
            raise ValueError("apoptosis_age must be >= 0")
        if self.rd_dt <= 0:
            raise ValueError("rd_dt must be > 0")
        if self.rd_substeps < 1:
            raise ValueError("rd_substeps must be >= 1")
        if self.Du < 0 or self.Dv < 0:
            raise ValueError("Du and Dv must be >= 0")
        if self.rd_noise < 0:
            raise ValueError("rd_noise must be >= 0")
        if self.rd_seed_amp < 0:
            raise ValueError("rd_seed_amp must be >= 0")
        if not (0.0 <= self.rd_seed_frac <= 1.0):
            raise ValueError("rd_seed_frac must be in [0,1]")
        if self.rd_print_stats_every < 0:
            raise ValueError("rd_print_stats_every must be >= 0")
        if self.rd_prog_gain <= 0:
            raise ValueError("rd_prog_gain must be > 0")
        if not (0.0 <= self.rd_divide_min_p <= 1.0):
            raise ValueError("rd_divide_min_p must be in [0,1]")
        if not (0.0 <= self.rd_divide_max_p <= 1.0):
            raise ValueError("rd_divide_max_p must be in [0,1]")
        if self.rd_divide_max_p < self.rd_divide_min_p:
            raise ValueError("rd_divide_max_p must be >= rd_divide_min_p")
        if self.rd_start_step < 0:
            raise ValueError("rd_start_step must be >= 0")
        if not (0.0 <= self.rd_apoptosis_min_p <= 1.0):
            raise ValueError("rd_apoptosis_min_p must be in [0,1]")
        if not (0.0 <= self.rd_apoptosis_max_p <= 1.0):
            raise ValueError("rd_apoptosis_max_p must be in [0,1]")
        if self.rd_apoptosis_max_p < self.rd_apoptosis_min_p:
            raise ValueError("rd_apoptosis_max_p must be >= rd_apoptosis_min_p")
        if not (0.0 <= self.rd_interior_apoptosis_shield <= 1.0):
            raise ValueError("rd_interior_apoptosis_shield must be in [0,1]")
        if not (0.0 <= self.rd_interior_divide_damp <= 1.0):
            raise ValueError("rd_interior_divide_damp must be in [0,1]")
        if not (0.0 <= self.rd_interior_crowd_weight <= 1.0):
            raise ValueError("rd_interior_crowd_weight must be in [0,1]")
        if self.polarity_noise < 0:
            raise ValueError("polarity_noise must be >= 0")
        if not (0.0 <= self.polarity_align_alpha0 <= 1.0):
            raise ValueError("polarity_align_alpha0 must be in [0,1]")
        if not (0.0 <= self.polarity_align_alpha_u <= 1.0):
            raise ValueError("polarity_align_alpha_u must be in [0,1]")
        if self.polarity_radius is not None and self.polarity_radius <= 0:
            raise ValueError("polarity_radius must be > 0 when set")
        if not (0.0 <= self.polarity_grad_gain <= 1.0):
            raise ValueError("polarity_grad_gain must be in [0,1]")
        if not (0.0 <= self.polarity_mix_prev <= 1.0):
            raise ValueError("polarity_mix_prev must be in [0,1]")

        valid_dir_modes = {"least_resistance", "tangential", "radial"}
        if self.division_direction_mode not in valid_dir_modes:
            raise ValueError(f"division_direction_mode must be one of {sorted(valid_dir_modes)}")
        valid_weight_modes = {"uniform", "linear", "gaussian"}
        if self.neighbor_weight not in valid_weight_modes:
            raise ValueError(f"neighbor_weight must be one of {sorted(valid_weight_modes)}")
        valid_radial = {"inward", "outward", "random"}
        if self.radial_sign not in valid_radial:
            raise ValueError(f"radial_sign must be one of {sorted(valid_radial)}")
        valid_rd_models = {"gray_scott"}
        if self.rd_model not in valid_rd_models:
            raise ValueError(f"rd_model must be one of {sorted(valid_rd_models)}")
        valid_rd_init = {"uniform_noise", "seed_center", "seed_random_cells"}
        if self.rd_init_mode not in valid_rd_init:
            raise ValueError(f"rd_init_mode must be one of {sorted(valid_rd_init)}")
        valid_prog_from = {"u", "v", "u_minus_v", "v_minus_u"}
        if self.rd_prog_from not in valid_prog_from:
            raise ValueError(f"rd_prog_from must be one of {sorted(valid_prog_from)}")
        valid_relax_projection = {"none", "force_dir", "polarity"}
        if self.relax_projection_mode not in valid_relax_projection:
            raise ValueError(
                f"relax_projection_mode must be one of {sorted(valid_relax_projection)}"
            )

        # Legacy program knobs are now aliases to RD controls.
        self.program_noise = float(self.rd_noise)
        self.program_sigmoid_center = 0.0
        self.program_sigmoid_slope = 1.0
        self.program_radial_sign = "inward" if self.radial_sign == "inward" else "outward"
        self.program_divide_boost = float(self.rd_divide_boost)
        self.program_divide_min_p = float(self.rd_divide_min_p)
        self.program_divide_max_p = float(self.rd_divide_max_p)

        self.overlap_cutoff = float(2.0 * self.radius * (1.0 + self.overlap_margin))
        if self.density_radius is None:
            self.density_radius = float(self.neighborhood_radius_factor * self.radius)
        if self.density_radius <= 0:
            raise ValueError("density_radius must be > 0")
        if self.R_sense is None:
            self.R_sense = float(self.density_radius)
        if self.R_sense <= 0:
            raise ValueError("R_sense must be > 0")
        if self.R_signal is None:
            self.R_signal = float(self.density_radius)
        if self.R_signal <= 0:
            raise ValueError("R_signal must be > 0")
        if self.grid_cell_size is None:
            self.grid_cell_size = float(max(self.overlap_cutoff, self.density_radius, self.R_sense))
        if self.grid_cell_size <= 0:
            raise ValueError("grid_cell_size must be > 0")
        if self.rest_distance_factor is None:
            self.rest_distance_factor = float(self.split_distance / self.radius)
        if self.rest_distance_factor <= 0:
            raise ValueError("rest_distance_factor must be > 0")
        if self.adhesion_radius_factor is None:
            self.adhesion_radius_factor = float(self.neighborhood_radius_factor)
        if self.adhesion_radius_factor <= 0:
            raise ValueError("adhesion_radius_factor must be > 0")
        if self.spring_max_step is None:
            self.spring_max_step = float(0.25 * self.radius)
        if self.spring_max_step <= 0:
            raise ValueError("spring_max_step must be > 0")
        self._force_division_dir_np: Optional[np.ndarray] = None
        if self.force_division_direction is not None:
            text = str(self.force_division_direction).strip()
            if text:
                parts = [p.strip() for p in text.split(",")]
                if len(parts) != 3:
                    raise ValueError(
                        "force_division_direction must be formatted as 'x,y,z'"
                    )
                vec = np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
                nrm = float(np.linalg.norm(vec))
                if nrm <= self.eps:
                    raise ValueError("force_division_direction norm must be > 0")
                self._force_division_dir_np = (vec / nrm).astype(np.float32, copy=False)
        if self.relax_projection_mode == "force_dir" and self._force_division_dir_np is None:
            raise ValueError(
                "relax_projection_mode='force_dir' requires force_division_direction to be set"
            )

        self.points = cp.zeros((1, 3), dtype=self.dtype)
        self.cell_ids = cp.zeros((1,), dtype=cp.int64)
        self.birth_age = cp.zeros((1,), dtype=cp.int32)
        self.cycle_age = cp.zeros((1,), dtype=cp.int32)
        self.cell_prog = cp.zeros((1,), dtype=self.dtype)
        self.u = cp.ones((1,), dtype=self.dtype)
        self.v = cp.zeros((1,), dtype=self.dtype)
        self._rng = cp.random.RandomState(self.seed)
        self._rng_np = np.random.RandomState(self.seed)
        self.p = self._random_unit_vectors_backend(1, cp, self.dtype).astype(self.dtype, copy=False)
        self._next_cell_id = 1
        self.step_index = 0
        self.count_history = [1]
        self._offset_cache_np: dict[int, np.ndarray] = {}
        self._offset_cache_cp: dict[int, cp.ndarray] = {}
        self.last_step_timing: Optional[dict[str, float]] = None
        self.step_timing_history: list[dict[str, float]] = []
        self.last_program_summary: Optional[dict[str, float]] = None
        self.last_rd_summary: Optional[dict[str, float]] = None
        self.last_polarity_coherence: float = 0.0
        self._rd_reseed_done = False
        self._gpu_kernels_ready = False
        self._gpu_kernel_error: Optional[str] = None
        self._init_gpu_kernels()
        self._initialize_rd_state()
        self._assert_state_aligned()
        if self.enable_reaction_diffusion:
            print(
                "Reaction-diffusion enabled: "
                f"model={self.rd_model}, R_signal={self.R_signal:g}, dt={self.rd_dt:g}, "
                f"substeps={self.rd_substeps}, Du={self.Du:g}, Dv={self.Dv:g}, "
                f"F={self.gs_F:g}, k={self.gs_k:g}, "
                f"start_step={self.rd_start_step}"
            )
        if self._force_division_dir_np is not None:
            fx, fy, fz = self._force_division_dir_np.tolist()
            print(
                "Forced division direction enabled: "
                f"[{fx:.4f}, {fy:.4f}, {fz:.4f}]"
            )

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
            "program",
            "masks",
            "divide_mod",
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

    def _format_program_summary(self, summary: dict[str, float]) -> str:
        step_num = int(summary.get("step", -1))
        n_cells = int(summary.get("cells", -1))
        n_div = int(summary.get("divide", 0))
        n_div_eligible = int(summary.get("divide_eligible", n_div))
        n_stay = int(summary.get("stay", 0))
        n_die = int(summary.get("die", 0))
        n_apop = int(summary.get("apoptosis", 0))
        n_apop_rd = int(summary.get("apoptosis_rd", 0))
        interior_mean = float(summary.get("interior_mean", 0.0))
        return (
            f"Program step {step_num}: cells={n_cells} | "
            f"prog(mean/min/max)="
            f"{summary.get('prog_mean', 0.0):.3f}/"
            f"{summary.get('prog_min', 0.0):.3f}/"
            f"{summary.get('prog_max', 0.0):.3f} | "
            f"age_birth_mean={summary.get('birth_mean', 0.0):.2f} "
            f"age_cycle_mean={summary.get('cycle_mean', 0.0):.2f} | "
            f"interior_mean={interior_mean:.3f} | "
            f"actions divide={n_div} (eligible={n_div_eligible}) "
            f"stay={n_stay} die={n_die} apoptosis={n_apop} (rd={n_apop_rd})"
        )

    def _format_rd_summary(self, summary: dict[str, float]) -> str:
        step_num = int(summary.get("step", -1))
        return (
            f"RD step {step_num}: "
            f"u(min/mean/max)="
            f"{summary.get('u_min', 0.0):.3f}/"
            f"{summary.get('u_mean', 0.0):.3f}/"
            f"{summary.get('u_max', 0.0):.3f} | "
            f"v(min/mean/max)="
            f"{summary.get('v_min', 0.0):.3f}/"
            f"{summary.get('v_mean', 0.0):.3f}/"
            f"{summary.get('v_max', 0.0):.3f} | "
            f"prog(min/mean/max)="
            f"{summary.get('prog_min', 0.0):.3f}/"
            f"{summary.get('prog_mean', 0.0):.3f}/"
            f"{summary.get('prog_max', 0.0):.3f} | "
            f"polarity_coherence={summary.get('polarity_coherence', 0.0):.3f}"
        )

    def _assert_state_aligned(self) -> None:
        n = int(self.points.shape[0])
        if int(self.cell_ids.shape[0]) != n:
            raise RuntimeError("Internal state mismatch: cell_ids length != points length")
        if int(self.birth_age.shape[0]) != n:
            raise RuntimeError("Internal state mismatch: birth_age length != points length")
        if int(self.cycle_age.shape[0]) != n:
            raise RuntimeError("Internal state mismatch: cycle_age length != points length")
        if int(self.cell_prog.shape[0]) != n:
            raise RuntimeError("Internal state mismatch: cell_prog length != points length")
        if int(self.u.shape[0]) != n:
            raise RuntimeError("Internal state mismatch: u length != points length")
        if int(self.v.shape[0]) != n:
            raise RuntimeError("Internal state mismatch: v length != points length")
        if int(self.p.shape[0]) != n:
            raise RuntimeError("Internal state mismatch: p length != points length")

    def _init_gpu_kernels(self) -> None:
        self._gpu_kernels_ready = False
        self._gpu_kernel_error = None
        self._kernel_neighbor_count = None
        self._kernel_neighbor_mean2 = None
        self._kernel_polarity_stats = None
        self._kernel_local_resultant = None
        self._kernel_least_resistance = None
        self._kernel_overlap = None
        if not _GPU_ENABLED:
            return
        try:
            self._kernel_neighbor_count = cp.RawKernel(
                GPU_RAW_KERNELS_SRC, "neighbor_count_kernel"
            )
            self._kernel_neighbor_mean2 = cp.RawKernel(
                GPU_RAW_KERNELS_SRC, "neighbor_mean2_kernel"
            )
            self._kernel_polarity_stats = cp.RawKernel(
                GPU_RAW_KERNELS_SRC, "polarity_stats_kernel"
            )
            self._kernel_local_resultant = cp.RawKernel(
                GPU_RAW_KERNELS_SRC, "local_resultant_kernel"
            )
            self._kernel_least_resistance = cp.RawKernel(
                GPU_RAW_KERNELS_SRC, "least_resistance_kernel"
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
        self._kernel_neighbor_mean2 = None
        self._kernel_polarity_stats = None
        self._kernel_local_resultant = None
        self._kernel_least_resistance = None
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

    def _clip01_backend(self, x, xp_module):
        return xp_module.clip(x, 0.0, 1.0)

    def _normalize_vec_rows(self, vecs, xp_module):
        nrm = xp_module.sqrt(xp_module.sum(vecs * vecs, axis=1, keepdims=True))
        return vecs / xp_module.maximum(nrm, self.eps)

    def _neighbor_mean_vec3(
        self,
        points,
        vec3,
        radius: float,
        grid: Optional[GridStruct] = None,
    ):
        """
        Unweighted neighbor mean for a 3-vector field on the moving graph.

        Uses grid candidates + exact distances (O(N*k) average).
        """
        xp_module = self._xp_of(points)
        n = int(points.shape[0])
        mean_vec = xp_module.zeros((n, 3), dtype=self.dtype)
        counts = xp_module.zeros((n,), dtype=xp_module.int32)
        if n == 0:
            return mean_vec, counts

        use_grid = grid if grid is not None else self._build_grid(points)
        vals = vec3.astype(self.dtype, copy=False)
        for i in range(n):
            ids = self._neighbors_within_radius(i, points, use_grid, radius)
            c = int(ids.shape[0])
            counts[i] = c
            if c > 0:
                mean_vec[i] = xp_module.mean(vals[ids], axis=0)
            else:
                mean_vec[i] = vals[i]
        return mean_vec.astype(self.dtype, copy=False), counts

    def _neighbor_u_gradient(
        self,
        points,
        u_vals,
        radius: float,
        grid: Optional[GridStruct] = None,
    ):
        """
        Local graph gradient of u using exact neighbor vectors/distances.
        """
        xp_module = self._xp_of(points)
        n = int(points.shape[0])
        grad = xp_module.zeros((n, 3), dtype=self.dtype)
        if n == 0:
            return grad

        use_grid = grid if grid is not None else self._build_grid(points)
        u = u_vals.astype(self.dtype, copy=False)
        for i in range(n):
            _ids, vecs, dists = self._neighbor_data_within_radius(i, points, use_grid, radius)
            if dists.size == 0:
                continue
            du = u[_ids] - u[i]
            contrib = (du[:, None] * vecs) / xp_module.maximum(dists[:, None], self.eps)
            grad[i] = xp_module.sum(contrib, axis=0)
        return grad.astype(self.dtype, copy=False)

    def _neighbor_polarity_stats(
        self,
        points,
        p_vals,
        u_vals,
        radius: float,
        grid: Optional[GridStruct] = None,
    ):
        """
        Fused neighbor pass for polarity update:
        - mean neighbor polarity vector
        - local u-graph gradient
        - neighbor counts
        """
        xp_module = self._xp_of(points)
        n = int(points.shape[0])
        mean_p = xp_module.zeros((n, 3), dtype=self.dtype)
        grad_u = xp_module.zeros((n, 3), dtype=self.dtype)
        counts = xp_module.zeros((n,), dtype=xp_module.int32)
        if n == 0:
            return mean_p, grad_u, counts

        use_grid = grid if grid is not None else self._build_grid(points)
        p_loc = p_vals.astype(self.dtype, copy=False)
        u_loc = u_vals.astype(self.dtype, copy=False)

        if self.fast_neighbors and self._gpu_fast_path_available(points):
            try:
                start_lut, count_lut = self._ensure_grid_lookup(use_grid, cp)
                span = max(1, int(np.ceil(float(radius) / max(use_grid.cell_size, self.eps))))
                blocks, threads = self._launch_cfg_1d(n)
                self._kernel_polarity_stats(
                    blocks,
                    threads,
                    (
                        points,
                        use_grid.sort_idx,
                        start_lut,
                        count_lut,
                        use_grid.origin,
                        use_grid.min_cell,
                        use_grid.max_cell,
                        np.float32(use_grid.cell_size),
                        np.int64(use_grid.stride_x),
                        np.int64(use_grid.stride_y),
                        np.int32(span),
                        np.float32(radius * radius),
                        np.int32(n),
                        np.float32(self.eps),
                        p_loc,
                        u_loc,
                        mean_p,
                        grad_u,
                        counts,
                    ),
                )
                return (
                    mean_p.astype(self.dtype, copy=False),
                    grad_u.astype(self.dtype, copy=False),
                    counts.astype(cp.int32, copy=False),
                )
            except Exception as exc:
                self._disable_gpu_kernels(exc)

        for i in range(n):
            ids, vecs, dists = self._neighbor_data_within_radius(i, points, use_grid, radius)
            c = int(ids.shape[0])
            counts[i] = c
            if c > 0:
                mean_p[i] = xp_module.mean(p_loc[ids], axis=0)
                du = u_loc[ids] - u_loc[i]
                contrib = (du[:, None] * vecs) / xp_module.maximum(dists[:, None], self.eps)
                grad_u[i] = xp_module.sum(contrib, axis=0)
            else:
                mean_p[i] = p_loc[i]

        return (
            mean_p.astype(self.dtype, copy=False),
            grad_u.astype(self.dtype, copy=False),
            counts.astype(xp_module.int32, copy=False),
        )

    def _update_polarity(
        self,
        points,
        v_hat,
        *,
        grid: Optional[GridStruct] = None,
    ) -> float:
        """
        Update per-cell planar polarity vector with local alignment and optional u-gradient.
        """
        xp_module = self._xp_of(points)
        n = int(points.shape[0])
        if n == 0:
            self.p = xp_module.zeros((0, 3), dtype=self.dtype)
            return 0.0

        radius = float(self.polarity_radius if self.polarity_radius is not None else self.R_signal)
        use_grid = grid if grid is not None else self._build_grid(points)
        p_old = self.p.astype(self.dtype, copy=False)
        mean_p, grad, _ = self._neighbor_polarity_stats(
            points,
            p_old,
            self.u,
            radius,
            grid=use_grid,
        )
        mean_p_dir = self._normalize_vec_rows(mean_p, xp_module)

        u_clip = xp_module.clip(self.u.astype(self.dtype, copy=False), 0.0, 1.0)
        alpha = xp_module.clip(
            self.polarity_align_alpha0 + self.polarity_align_alpha_u * u_clip,
            0.0,
            1.0,
        ).reshape(-1, 1)

        p_prev_mix = self.polarity_mix_prev * p_old + (1.0 - self.polarity_mix_prev) * mean_p_dir
        p_prev_mix = self._normalize_vec_rows(p_prev_mix, xp_module)
        p_new = (1.0 - alpha) * p_prev_mix + alpha * mean_p
        p_new = self._normalize_vec_rows(p_new, xp_module)

        n_norm = xp_module.sqrt(xp_module.sum(v_hat * v_hat, axis=1, keepdims=True))
        n_unit = xp_module.zeros_like(v_hat)
        valid_n = n_norm[:, 0] > self.eps
        if self._any_true(valid_n, xp_module):
            n_unit[valid_n] = v_hat[valid_n] / xp_module.maximum(n_norm[valid_n], self.eps)

        if self.polarity_use_u_gradient:
            if self.polarity_project_to_tangent and self._any_true(valid_n, xp_module):
                dgn = xp_module.sum(grad * n_unit, axis=1, keepdims=True)
                grad = grad - dgn * n_unit
            grad_dir = self._normalize_vec_rows(grad, xp_module)
            g = float(self.polarity_grad_gain)
            p_new = self._normalize_vec_rows((1.0 - g) * p_new + g * grad_dir, xp_module)

        if self.polarity_noise > 0:
            noise = self._rng.normal(0.0, 1.0, size=(n, 3)).astype(self.dtype, copy=False)
            p_new = p_new + self.polarity_noise * noise

        if self.polarity_project_to_tangent and self._any_true(valid_n, xp_module):
            dpn = xp_module.sum(p_new * n_unit, axis=1, keepdims=True)
            p_new = p_new - dpn * n_unit

        p_norm = xp_module.sqrt(xp_module.sum(p_new * p_new, axis=1, keepdims=True))
        tiny = p_norm[:, 0] <= self.eps
        if self._any_true(tiny, xp_module):
            p_new[tiny] = p_old[tiny]
            p_norm = xp_module.sqrt(xp_module.sum(p_new * p_new, axis=1, keepdims=True))
            tiny2 = p_norm[:, 0] <= self.eps
            if self._any_true(tiny2, xp_module):
                n_fb = self._to_int(tiny2.sum())
                p_new[tiny2] = self._random_unit_vectors_backend(n_fb, xp_module, self.dtype)
                p_norm = xp_module.sqrt(xp_module.sum(p_new * p_new, axis=1, keepdims=True))

        p_new = p_new / xp_module.maximum(p_norm, self.eps)
        self.p = p_new.astype(self.dtype, copy=False)

        # Alignment coherence diagnostic in [0,1].
        dots = xp_module.sum(self.p * mean_p_dir, axis=1)
        coh = xp_module.mean(xp_module.clip(dots, -1.0, 1.0))
        return self._to_float(coh)

    def _initialize_rd_state(self) -> None:
        """Initialize per-cell reaction-diffusion fields u and v."""
        xp_module = self._xp_of(self.points)
        n = int(self.points.shape[0])
        self.u = xp_module.zeros((n,), dtype=self.dtype)
        self.v = xp_module.ones((n,), dtype=self.dtype)
        if n == 0:
            return

        if self.rd_init_mode == "uniform_noise":
            if self.rd_seed_amp > 0:
                if xp_module is cp:
                    self.u += (self.rd_seed_amp * self._rng.normal(0.0, 1.0, size=n)).astype(
                        self.dtype, copy=False
                    )
                    self.v += (self.rd_seed_amp * self._rng.normal(0.0, 1.0, size=n)).astype(
                        self.dtype, copy=False
                    )
                else:
                    self.u += (self.rd_seed_amp * self._rng_np.normal(0.0, 1.0, size=n)).astype(
                        self.dtype, copy=False
                    )
                    self.v += (self.rd_seed_amp * self._rng_np.normal(0.0, 1.0, size=n)).astype(
                        self.dtype, copy=False
                    )
        elif self.rd_init_mode == "seed_center":
            center = self.points.mean(axis=0)
            d2 = xp_module.sum((self.points - center[None, :]) ** 2, axis=1)
            idx = self._to_int(xp_module.argmin(d2))
            self.u[idx] = 1.0
        elif self.rd_init_mode == "seed_random_cells":
            if xp_module is cp:
                anchor_idx = self._to_int(self._rng.randint(0, n))
            else:
                anchor_idx = int(self._rng_np.randint(0, n))
            n_seed = max(1, int(round(self.rd_seed_frac * n)))
            idx = self._seed_indices_around_anchor(anchor_idx, n_seed=n_seed)
            self.u[idx] = 1.0
        else:  # pragma: no cover
            raise ValueError(f"Unsupported rd_init_mode: {self.rd_init_mode}")

        if self.rd_clamp:
            self.u = self._clip01_backend(self.u, xp_module).astype(self.dtype, copy=False)
            self.v = self._clip01_backend(self.v, xp_module).astype(self.dtype, copy=False)

    def _seed_indices_around_anchor(self, anchor_idx: int, n_seed: int = 1):
        """
        Build a contiguous seed around one anchor cell.

        Primary selection is all cells within R_signal of the anchor.
        If that set is too small, fill with nearest neighbors to reach n_seed.
        """
        xp_module = self._xp_of(self.points)
        n = int(self.points.shape[0])
        if n <= 0:
            return xp_module.zeros((0,), dtype=xp_module.int64)

        a = int(max(0, min(anchor_idx, n - 1)))
        target = int(max(1, min(n_seed, n)))

        anchor = self.points[a]
        dif = self.points - anchor[None, :]
        d2 = xp_module.sum(dif * dif, axis=1)

        seed_radius = float(self.R_signal if self.R_signal is not None else self.density_radius)
        if seed_radius > 0:
            mask = d2 <= (seed_radius * seed_radius + self.eps)
            idx = xp_module.where(mask)[0].astype(xp_module.int64, copy=False)
        else:
            idx = xp_module.asarray([a], dtype=xp_module.int64)

        if int(idx.shape[0]) < target:
            order = xp_module.argsort(d2)
            idx = order[:target].astype(xp_module.int64, copy=False)

        return idx

    def _reinitialize_rd_state_at_activation(self) -> None:
        """
        Reinitialize RD fields when delayed activation begins.

        For delayed starts, seed_center and seed_random_cells both seed a single
        cell on the current colony geometry, then diffusion/reaction takes over.
        """
        xp_module = self._xp_of(self.points)
        n = int(self.points.shape[0])
        self.u = xp_module.zeros((n,), dtype=self.dtype)
        self.v = xp_module.ones((n,), dtype=self.dtype)
        if n == 0:
            return

        if self.rd_init_mode == "seed_center":
            center = self.points.mean(axis=0)
            d2 = xp_module.sum((self.points - center[None, :]) ** 2, axis=1)
            idx = self._to_int(xp_module.argmin(d2))
            self.u[idx] = 1.0
        elif self.rd_init_mode == "seed_random_cells":
            idx = self._to_int(self._rng.randint(0, n))
            n_seed = max(1, int(round(self.rd_seed_frac * n)))
            idx = self._seed_indices_around_anchor(idx, n_seed=n_seed)
            self.u[idx] = 1.0
        elif self.rd_init_mode == "uniform_noise":
            if self.rd_seed_amp > 0:
                if xp_module is cp:
                    self.u += (self.rd_seed_amp * self._rng.normal(0.0, 1.0, size=n)).astype(
                        self.dtype, copy=False
                    )
                    self.v += (self.rd_seed_amp * self._rng.normal(0.0, 1.0, size=n)).astype(
                        self.dtype, copy=False
                    )
                else:
                    self.u += (self.rd_seed_amp * self._rng_np.normal(0.0, 1.0, size=n)).astype(
                        self.dtype, copy=False
                    )
                    self.v += (self.rd_seed_amp * self._rng_np.normal(0.0, 1.0, size=n)).astype(
                        self.dtype, copy=False
                    )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported rd_init_mode: {self.rd_init_mode}")

        if self.rd_clamp:
            self.u = self._clip01_backend(self.u, xp_module).astype(self.dtype, copy=False)
            self.v = self._clip01_backend(self.v, xp_module).astype(self.dtype, copy=False)

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

    def _neighbor_counts_within(
        self,
        points: cp.ndarray,
        radius: float,
        grid: Optional[GridStruct] = None,
    ) -> cp.ndarray:
        """Count neighbors within fixed radius (self excluded), grid-accelerated."""
        xp_module = self._xp_of(points)
        n = points.shape[0]
        if n <= 1:
            return xp_module.zeros((n,), dtype=xp_module.int32)
        if not self.fast_neighbors:
            return self._neighbor_counts_within_slow(points, radius)

        if self._gpu_fast_path_available(points):
            try:
                use_grid = grid if grid is not None else self._build_grid(points)
                start_lut, count_lut = self._ensure_grid_lookup(use_grid, cp)
                counts = cp.zeros((n,), dtype=cp.int32)
                span = max(1, int(np.ceil(float(radius) / max(use_grid.cell_size, self.eps))))
                blocks, threads = self._launch_cfg_1d(n)
                self._kernel_neighbor_count(
                    blocks,
                    threads,
                    (
                        points,
                        use_grid.sort_idx,
                        start_lut,
                        count_lut,
                        use_grid.origin,
                        use_grid.min_cell,
                        use_grid.max_cell,
                        np.float32(use_grid.cell_size),
                        np.int64(use_grid.stride_x),
                        np.int64(use_grid.stride_y),
                        np.int32(span),
                        np.float32(radius * radius),
                        np.int32(n),
                        counts,
                    ),
                )
                return counts
            except Exception as exc:
                self._disable_gpu_kernels(exc)

        use_grid = grid if grid is not None else self._build_grid(points)
        counts = xp_module.zeros(n, dtype=xp_module.int32)
        for i in range(n):
            ids = self._neighbors_within_radius(i, points, use_grid, radius)
            counts[i] = ids.shape[0]
        return counts.astype(xp_module.int32, copy=False)

    def _neighbor_mean_scalar(
        self,
        points,
        values,
        radius: float,
        grid: Optional[GridStruct] = None,
    ):
        """
        Unweighted neighbor-mean for a scalar field on the moving proximity graph.

        Uses grid candidates + exact distance filtering (O(N*k) average).
        """
        xp_module = self._xp_of(points)
        n = int(points.shape[0])
        mean_vals = xp_module.zeros((n,), dtype=self.dtype)
        counts = xp_module.zeros((n,), dtype=xp_module.int32)
        if n == 0:
            return mean_vals, counts

        use_grid = grid if grid is not None else self._build_grid(points)
        vals = values.astype(self.dtype, copy=False)
        for i in range(n):
            ids = self._neighbors_within_radius(i, points, use_grid, radius)
            c = int(ids.shape[0])
            counts[i] = c
            if c > 0:
                mean_vals[i] = xp_module.mean(vals[ids])
            else:
                mean_vals[i] = vals[i]
        return mean_vals.astype(self.dtype, copy=False), counts

    def _neighbor_means_two_scalars(
        self,
        points,
        values0,
        values1,
        radius: float,
        grid: Optional[GridStruct] = None,
    ):
        """
        Compute unweighted neighbor means for two scalar fields in one pass.

        GPU fast path uses a RawKernel to avoid Python-loop overhead.
        """
        xp_module = self._xp_of(points)
        n = int(points.shape[0])
        mean0 = xp_module.zeros((n,), dtype=self.dtype)
        mean1 = xp_module.zeros((n,), dtype=self.dtype)
        counts = xp_module.zeros((n,), dtype=xp_module.int32)
        if n == 0:
            return mean0, mean1, counts

        if self.fast_neighbors and self._gpu_fast_path_available(points):
            try:
                use_grid = grid if grid is not None else self._build_grid(points)
                start_lut, count_lut = self._ensure_grid_lookup(use_grid, cp)
                s0 = values0.astype(self.dtype, copy=False)
                s1 = values1.astype(self.dtype, copy=False)
                span = max(1, int(np.ceil(float(radius) / max(use_grid.cell_size, self.eps))))
                blocks, threads = self._launch_cfg_1d(n)
                self._kernel_neighbor_mean2(
                    blocks,
                    threads,
                    (
                        points,
                        use_grid.sort_idx,
                        start_lut,
                        count_lut,
                        use_grid.origin,
                        use_grid.min_cell,
                        use_grid.max_cell,
                        np.float32(use_grid.cell_size),
                        np.int64(use_grid.stride_x),
                        np.int64(use_grid.stride_y),
                        np.int32(span),
                        np.float32(radius * radius),
                        np.int32(n),
                        s0,
                        s1,
                        mean0,
                        mean1,
                        counts,
                    ),
                )
                return (
                    mean0.astype(self.dtype, copy=False),
                    mean1.astype(self.dtype, copy=False),
                    counts.astype(cp.int32, copy=False),
                )
            except Exception as exc:
                self._disable_gpu_kernels(exc)

        # Fallback: two scalar passes using existing grid neighbor iteration.
        use_grid = grid if grid is not None else self._build_grid(points)
        mean0, counts = self._neighbor_mean_scalar(points, values0, radius, grid=use_grid)
        mean1, _ = self._neighbor_mean_scalar(points, values1, radius, grid=use_grid)
        return mean0, mean1, counts

    def _local_neighbor_resultants(
        self,
        positions,
        grid: GridStruct,
        radius: float,
        weight_mode: str,
    ):
        """
        Compute local resultant direction and magnitude from exact neighbors.

        Returns:
        - v_hat: normalized resultant direction per cell
        - v_mag: raw resultant magnitude before normalization
        No direction quantization is introduced.
        """
        xp_module = self._xp_of(positions)
        n = positions.shape[0]
        v_hat = xp_module.zeros((n, 3), dtype=float)
        v_mag = xp_module.zeros((n,), dtype=float)
        if n == 0:
            return v_hat, v_mag

        if self._gpu_fast_path_available(positions):
            try:
                start_lut, count_lut = self._ensure_grid_lookup(grid, cp)
                out = cp.zeros((n, 3), dtype=positions.dtype)
                out_mag = cp.zeros((n,), dtype=positions.dtype)
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
                        out_mag,
                    ),
                )
                return out, out_mag
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
            v_mag[i] = vn
            if vn > self.eps:
                v_hat[i] = v / vn
        return v_hat, v_mag

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

    def _density_actions_from_counts(self, counts) -> cp.ndarray:
        """Map local crowding counts to divide/stay/die actions."""
        actions = cp.ones(counts.shape[0], dtype=cp.int8)
        actions[counts > self.crowding_stay_threshold] = 0
        actions[counts >= self.crowding_death_threshold] = -1
        return actions

    def _prog_from_reaction_diffusion(self, u, v):
        """Map RD state to scalar program: prog = sigmoid(u)."""
        xp_module = self._xp_of(u)
        del v  # intentionally unused: scalar program is now defined from u only.
        prog = 1.0 / (1.0 + xp_module.exp(-u.astype(self.dtype, copy=False)))
        return xp_module.clip(prog, 0.0, 1.0).astype(self.dtype, copy=False)

    def _rd_interior_score(self, counts, v_mag):
        """
        Estimate interior-ness in [0,1] from crowding and local resultant magnitude.

        Higher score means denser/more interior cells that should be protected from
        random RD-driven apoptosis to preserve cohesive tissue-like behavior.
        """
        xp_module = self._xp_of(counts)
        crowd_norm = float(max(1, self.crowding_stay_threshold))
        crowd_score = xp_module.clip(
            counts.astype(self.dtype, copy=False) / crowd_norm,
            0.0,
            1.0,
        )

        if v_mag is None:
            return crowd_score.astype(self.dtype, copy=False)

        v_mag = v_mag.astype(self.dtype, copy=False)
        if xp_module is cp:
            mag_scale = cp.mean(v_mag) + 2.0 * cp.std(v_mag) + self.eps
        else:
            mag_scale = np.percentile(v_mag, 90) + self.eps
        surface = xp_module.clip(v_mag / mag_scale, 0.0, 1.0)
        interior_geom = 1.0 - surface

        w_c = float(self.rd_interior_crowd_weight)
        w_g = float(1.0 - w_c)
        interior = w_c * crowd_score + w_g * interior_geom
        return xp_module.clip(interior, 0.0, 1.0).astype(self.dtype, copy=False)

    def _rd_step(self, points, grid: Optional[GridStruct] = None):
        """Advance per-cell reaction-diffusion state by one explicit Euler step."""
        if not self.enable_reaction_diffusion:
            return self.u, self.v

        xp_module = self._xp_of(points)
        n = int(points.shape[0])
        if n == 0:
            self.u = xp_module.zeros((0,), dtype=self.dtype)
            self.v = xp_module.zeros((0,), dtype=self.dtype)
            return self.u, self.v

        use_grid = grid if grid is not None else self._build_grid(points)
        radius = float(self.R_signal)
        u = self.u.astype(self.dtype, copy=False)
        v = self.v.astype(self.dtype, copy=False)
        dt = float(self.rd_dt)
        substeps = int(self.rd_substeps)

        for _ in range(substeps):
            u_bar, v_bar, _ = self._neighbor_means_two_scalars(points, u, v, radius, grid=use_grid)
            du_diff = self.Du * (u_bar - u)
            dv_diff = self.Dv * (v_bar - v)

            if self.rd_model == "gray_scott":
                # VisualPDE form:
                # du/dt = Du*L(u) + u^2*v - (a+b)*u
                # dv/dt = Dv*L(v) - u^2*v + a*(1-v)
                # with a=gs_F and b=gs_k.
                uuv = u * u * v
                du_react = uuv - (self.gs_F + self.gs_k) * u
                dv_react = -uuv + self.gs_F * (1.0 - v)
            else:  # pragma: no cover
                raise ValueError(f"Unsupported rd_model: {self.rd_model}")

            u = u + dt * (du_diff + du_react)
            v = v + dt * (dv_diff + dv_react)

            if self.rd_noise > 0:
                # Each RD substep is an explicit timestep with variance ~dt.
                noise_sigma = float(self.rd_noise * np.sqrt(dt))
                if xp_module is cp:
                    u += (noise_sigma * self._rng.normal(0.0, 1.0, size=n)).astype(
                        self.dtype,
                        copy=False,
                    )
                    v += (noise_sigma * self._rng.normal(0.0, 1.0, size=n)).astype(
                        self.dtype,
                        copy=False,
                    )
                else:
                    u += (noise_sigma * self._rng_np.normal(0.0, 1.0, size=n)).astype(
                        self.dtype,
                        copy=False,
                    )
                    v += (noise_sigma * self._rng_np.normal(0.0, 1.0, size=n)).astype(
                        self.dtype,
                        copy=False,
                    )

            if self.rd_clamp:
                u = self._clip01_backend(u, xp_module)
                v = self._clip01_backend(v, xp_module)

        self.u = u.astype(self.dtype, copy=False)
        self.v = v.astype(self.dtype, copy=False)
        return self.u, self.v

    def _update_cell_program(
        self,
        points,
        birth_age,
        cycle_age,
        *,
        grid: Optional[GridStruct] = None,
    ):
        """
        Legacy compatibility path: derive program from RD scalar (sigmoid(u)).

        Grid is still used for exact-neighbor geometry/crowding outputs needed by
        other parts of the step logic.
        Returns: (prog in [0,1], v_hat, neighbor_counts)
        """
        xp_module = self._xp_of(points)
        n = int(points.shape[0])
        if n == 0:
            empty_prog = xp_module.zeros((0,), dtype=self.dtype)
            empty_v = xp_module.zeros((0, 3), dtype=self.dtype)
            empty_counts = xp_module.zeros((0,), dtype=xp_module.int32)
            return empty_prog, empty_v, empty_counts
        del birth_age
        del cycle_age

        use_grid = grid if grid is not None else self._build_grid(points)
        v_hat, _ = self._local_neighbor_resultants(
            points,
            use_grid,
            float(self.R_sense),
            self.neighbor_weight,
        )
        v_hat = v_hat.astype(self.dtype, copy=False)
        counts = self._neighbor_counts_within(
            points,
            float(self.density_radius),
            grid=use_grid,
        ).astype(xp_module.int32, copy=False)
        prog = self._prog_from_reaction_diffusion(self.u, self.v)
        return prog, v_hat, counts

    def _division_dirs_programmed(self, points, v_hat, prog):
        """
        Programmed per-cell direction:
        tangential for low prog (young cycle), radial for high prog (older/mature).
        """
        xp_module = self._xp_of(points)
        n = int(points.shape[0])
        if n == 0:
            return xp_module.zeros((0, 3), dtype=self.dtype)

        v_norm = xp_module.sqrt(xp_module.sum(v_hat * v_hat, axis=1, keepdims=True))
        v_unit = xp_module.zeros_like(v_hat)
        good_v = v_norm[:, 0] > self.eps
        if self._any_true(good_v, xp_module):
            v_unit[good_v] = v_hat[good_v] / xp_module.maximum(v_norm[good_v], self.eps)

        if self.radial_sign == "inward":
            d_rad = v_unit.copy()
        elif self.radial_sign == "random":
            if xp_module is cp:
                s = xp_module.where(
                    self._rng.random_sample(n) < 0.5,
                    -1.0,
                    1.0,
                )
            else:
                s = self._rng_np.choice(np.asarray([-1.0, 1.0], dtype=float), size=n)
            d_rad = v_unit * s[:, None]
        else:
            # v_hat points toward local mass, so outward is -v_hat.
            d_rad = -v_unit

        u = self._random_unit_vectors_backend(n, xp_module, self.dtype)
        d_tan = xp_module.cross(v_unit, u)
        dtn = xp_module.sqrt(xp_module.sum(d_tan * d_tan, axis=1, keepdims=True))
        good_t = dtn[:, 0] > self.eps
        if self._any_true(good_t, xp_module):
            d_tan[good_t] = d_tan[good_t] / xp_module.maximum(dtn[good_t], self.eps)

        unresolved = (~good_t) & good_v
        if self._any_true(unresolved, xp_module):
            idx = xp_module.where(unresolved)[0]
            v = v_unit[idx]
            axis = xp_module.zeros_like(v)
            axis[:, 0] = 1.0
            use_y = xp_module.abs(v[:, 0]) > 0.9
            axis[use_y, 0] = 0.0
            axis[use_y, 1] = 1.0
            alt = xp_module.cross(v, axis)
            altn = xp_module.sqrt(xp_module.sum(alt * alt, axis=1, keepdims=True))
            good_alt = altn[:, 0] > self.eps
            if self._any_true(good_alt, xp_module):
                idx_good = idx[good_alt]
                d_tan[idx_good] = alt[good_alt] / xp_module.maximum(altn[good_alt], self.eps)

        prog_col = prog.reshape(-1, 1).astype(self.dtype, copy=False)
        blended = (1.0 - prog_col) * d_tan + prog_col * d_rad
        bn = xp_module.sqrt(xp_module.sum(blended * blended, axis=1, keepdims=True))
        need_fallback = bn[:, 0] <= self.eps
        if self._any_true(need_fallback, xp_module):
            n_fb = self._to_int(need_fallback.sum())
            blended[need_fallback] = self._random_unit_vectors_backend(n_fb, xp_module, self.dtype)
            bn = xp_module.sqrt(xp_module.sum(blended * blended, axis=1, keepdims=True))
        return blended / xp_module.maximum(bn, self.eps)

    def _division_dirs_from_polarity(self, v_hat, prog):
        """
        RD-active polarity-driven directions:
        - tangential axis from per-cell polarity p
        - radial axis from -v_hat (outward)
        - blended by prog in [0,1]
        """
        xp_module = self._xp_of(v_hat)
        n = int(v_hat.shape[0])
        if n == 0:
            return xp_module.zeros((0, 3), dtype=self.dtype)

        p = self.p.astype(self.dtype, copy=False)
        if int(p.shape[0]) != n:
            p = self._random_unit_vectors_backend(n, xp_module, self.dtype)
        d_tan = self._normalize_vec_rows(p, xp_module)

        v_norm = xp_module.sqrt(xp_module.sum(v_hat * v_hat, axis=1, keepdims=True))
        v_unit = xp_module.zeros_like(v_hat)
        good_v = v_norm[:, 0] > self.eps
        if self._any_true(good_v, xp_module):
            v_unit[good_v] = v_hat[good_v] / xp_module.maximum(v_norm[good_v], self.eps)
        d_rad = -v_unit

        prog_col = xp_module.clip(
            prog.astype(self.dtype, copy=False).reshape(-1, 1),
            0.0,
            1.0,
        )
        blended = (1.0 - prog_col) * d_tan + prog_col * d_rad
        bn = xp_module.sqrt(xp_module.sum(blended * blended, axis=1, keepdims=True))
        need_fallback = bn[:, 0] <= self.eps
        if self._any_true(need_fallback, xp_module):
            n_fb = self._to_int(need_fallback.sum())
            blended[need_fallback] = self._random_unit_vectors_backend(n_fb, xp_module, self.dtype)
            bn = xp_module.sqrt(xp_module.sum(blended * blended, axis=1, keepdims=True))
        return blended / xp_module.maximum(bn, self.eps)

    def density_regulated_rule(self, points: cp.ndarray, step: int) -> cp.ndarray:
        """
        Default biological rule:
        - neighbors >= crowding_death_threshold: die
        - neighbors > crowding_stay_threshold: stay
        - otherwise: divide
        """
        del step  # Rule currently uses geometry only.
        counts = self._neighbor_counts_within(points, float(self.density_radius))
        return self._density_actions_from_counts(counts)

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

        # Fast path: local, grid-accelerated repulsion with cutoff R_sense.
        # This lowers complexity from O(N^2) to roughly O(N*k) for stable density.
        if self.fast_neighbors and self._gpu_fast_path_available(self.points):
            try:
                sense_radius = float(self.R_sense)
                local_cell_size = float(max(self.eps, min(self.grid_cell_size, sense_radius)))
                grid = self._build_grid(self.points, cell_size=local_cell_size)
                start_lut, count_lut = self._ensure_grid_lookup(grid, cp)
                dirs = cp.zeros((n, 3), dtype=self.dtype)
                span = max(1, int(np.ceil(sense_radius / max(grid.cell_size, self.eps))))
                blocks, threads = self._launch_cfg_1d(n)
                self._kernel_least_resistance(
                    blocks,
                    threads,
                    (
                        self.points,
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
                        np.float32(sense_radius * sense_radius),
                        np.int32(n),
                        np.float32(self.eps),
                        dirs,
                    ),
                )
                return dirs
            except Exception as exc:
                self._disable_gpu_kernels(exc)

        # Fallback: exact all-pairs O(N^2) repulsion.
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

    def _project_relax_displacements(self, disp, axes=None):
        """
        Optionally constrain spring-relaxer displacement directions.

        - none: unchanged
        - force_dir: project to global forced division axis
        - polarity: project to per-cell polarity axis
        """
        mode = self.relax_projection_mode
        if mode == "none":
            return disp

        xp_module = self._xp_of(disp)
        if mode == "force_dir":
            if self._force_division_dir_np is None:
                return disp
            axis = xp_module.asarray(self._force_division_dir_np, dtype=self.dtype).reshape(1, 3)
            dotv = xp_module.sum(disp * axis, axis=1, keepdims=True)
            return dotv * axis

        if mode == "polarity":
            if axes is None:
                return disp
            a = axes.astype(self.dtype, copy=False)
            if a.shape != disp.shape:
                return disp
            an = xp_module.sqrt(xp_module.sum(a * a, axis=1, keepdims=True))
            valid = an[:, 0] > self.eps
            a_unit = xp_module.zeros_like(a)
            if self._any_true(valid, xp_module):
                a_unit[valid] = a[valid] / xp_module.maximum(an[valid], self.eps)
            dotv = xp_module.sum(disp * a_unit, axis=1, keepdims=True)
            projected = dotv * a_unit
            # Keep original displacement where axis is undefined.
            if self._any_true(~valid, xp_module):
                projected[~valid] = disp[~valid]
            return projected

        return disp

    def _resolve_overlaps(self, pts: cp.ndarray, relax_axes=None) -> cp.ndarray:
        """
        Iteratively relax local pairwise springs with exact distances.

        Grid is acceleration only; points are continuous and never snapped.
        """
        n = pts.shape[0]
        if n < 2:
            return pts

        rest_dist = float(self.rest_distance_factor * self.radius)
        adhesion_radius = float(max(rest_dist, self.adhesion_radius_factor * self.radius))
        overlap_cell_size = float(max(self.eps, min(self.grid_cell_size, adhesion_radius)))
        xp_module = self._xp_of(pts)
        pos = pts.astype(self.dtype, copy=True)

        if self._gpu_fast_path_available(pos):
            try:
                span = max(1, int(np.ceil(adhesion_radius / max(overlap_cell_size, self.eps))))
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
                            np.float32(adhesion_radius),
                            np.float32(adhesion_radius * adhesion_radius),
                            np.float32(rest_dist),
                            np.float32(self.spring_k),
                            np.float32(self.spring_max_step),
                            np.float32(self.overlap_tol),
                            np.float32(self.eps),
                            np.int32(n),
                            disp,
                            hits,
                        ),
                    )
                    if not bool(cp.any(hits > 0).item()):
                        break
                    disp = self._project_relax_displacements(disp, axes=relax_axes)
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
                    adhesion_radius,
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

                # Strong repulsion for d < rest_dist, softer adhesion for d > rest_dist.
                repulse_mask = dists < (rest_dist - self.overlap_tol)
                attract_mask = dists > (rest_dist + self.overlap_tol)
                if adhesion_radius <= (rest_dist + self.overlap_tol):
                    attract_mask = xp_module.zeros_like(attract_mask)
                active = repulse_mask | attract_mask
                if not self._any_true(active, xp_module):
                    continue

                moved = True
                nbr_ids = nbr_ids[active]
                vecs = vecs[active]
                dists = dists[active]
                repulse_mask = repulse_mask[active]
                attract_mask = attract_mask[active]

                scales = xp_module.zeros_like(dists)
                if self._any_true(repulse_mask, xp_module):
                    gaps = rest_dist - dists[repulse_mask]
                    corr = xp_module.minimum(gaps, self.spring_max_step)
                    scales[repulse_mask] = 0.5 * corr
                if self._any_true(attract_mask, xp_module):
                    attr_err = dists[attract_mask] - rest_dist
                    span_attr = max(adhesion_radius - rest_dist, self.eps)
                    frac = xp_module.clip(attr_err / span_attr, 0.0, 1.0)
                    delta_attr = self.spring_k * attr_err * (1.0 - frac)
                    delta_attr = xp_module.minimum(delta_attr, self.spring_max_step)
                    scales[attract_mask] = -0.5 * delta_attr

                dirs = xp_module.zeros_like(vecs)
                nz = dists > self.eps
                if self._any_true(nz, xp_module):
                    dirs[nz] = -vecs[nz] / dists[nz, None]
                z = ~nz
                if self._any_true(z, xp_module):
                    z_count = self._to_int(z.sum())
                    dirs[z] = self._random_unit_vectors_backend(z_count, xp_module, pos.dtype)

                # unit_dir points j->i. Positive scales repel, negative scales attract.
                shifts = scales[:, None] * dirs
                disp[i] += shifts.sum(axis=0)
                for axis in range(3):
                    xp_module.add.at(disp[:, axis], nbr_ids, -shifts[:, axis])

            if not moved:
                break
            disp = self._project_relax_displacements(disp, axes=relax_axes)
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

        self._assert_state_aligned()
        rule = action_rule or self.density_regulated_rule

        if (
            self.enable_reaction_diffusion
            and (not self._rd_reseed_done)
            and int(self.rd_start_step) > 0
            and self.step_index == int(self.rd_start_step)
        ):
            self._reinitialize_rd_state_at_activation()
            self._rd_reseed_done = True
            print(
                "RD reseed at activation: "
                f"step={self.step_index}, mode={self.rd_init_mode}, cells={int(self.points.shape[0])}"
            )

        prog_all = None
        v_hat_all = None
        v_mag_all = None
        counts_all = None
        program_grid = None
        interior_score = None
        polarity_coherence_this_step = 0.0
        self.last_polarity_coherence = 0.0
        rd_active = bool(
            self.enable_reaction_diffusion
            and self.points.shape[0] > 0
            and self.step_index >= int(self.rd_start_step)
        )
        if rd_active:
            program_grid = self._build_grid(self.points)
            self._rd_step(self.points, grid=program_grid)
            # Program scalar is now RD-derived directly (sigmoid(u)).
            prog_all = self._prog_from_reaction_diffusion(self.u, self.v)
            counts_all = self._neighbor_counts_within(
                self.points,
                float(self.density_radius),
                grid=program_grid,
            )
            v_hat_all, v_mag_all = self._local_neighbor_resultants(
                self.points,
                program_grid,
                float(self.R_sense),
                self.neighbor_weight,
            )
            if counts_all is not None:
                interior_score = self._rd_interior_score(counts_all, v_mag_all)
            if self.enable_polarity and v_hat_all is not None:
                polarity_coherence_this_step = self._update_polarity(
                    self.points,
                    v_hat_all,
                    grid=program_grid,
                )
                self.last_polarity_coherence = float(polarity_coherence_this_step)
        elif self.points.shape[0] > 0:
            program_grid = self._build_grid(self.points)
            # Keep legacy branch but derive prog from RD state variables only
            # (no separate age/cycle program dynamics).
            prog_all, v_hat_all, counts_all = self._update_cell_program(
                self.points,
                self.birth_age,
                self.cycle_age,
                grid=program_grid,
            )
        mark("program")

        use_default_density_rule = action_rule is None
        if action_rule is not None:
            bound_func = getattr(action_rule, "__func__", None)
            bound_self = getattr(action_rule, "__self__", None)
            if bound_func is CellGrowth3D.density_regulated_rule and bound_self is self:
                use_default_density_rule = True

        if use_default_density_rule and counts_all is not None:
            actions = self._density_actions_from_counts(counts_all)
        else:
            actions = rule(self.points, self.step_index)
        if actions.shape != (self.points.shape[0],):
            raise ValueError("Action rule must return a vector shaped (n_cells,)")
        mark("rule")

        divide_mask = actions == 1
        stay_mask = actions == 0
        die_mask = actions == -1

        if bool(cp.any((~divide_mask) & (~stay_mask) & (~die_mask))):
            raise ValueError("Actions must only contain -1, 0, or 1.")

        n_apoptosis_this_step = 0
        n_apoptosis_rd_this_step = 0
        interior_mean_this_step = 0.0
        if interior_score is not None:
            interior_mean_this_step = self._to_float(cp.mean(interior_score))
        if self.enable_apoptosis and self.points.shape[0] > 0:
            apoptosis_mask = self.birth_age >= int(self.apoptosis_age)
            if bool(cp.any(apoptosis_mask)):
                n_apoptosis_this_step = self._to_int(apoptosis_mask.sum())
                die_mask = die_mask | apoptosis_mask
                divide_mask = divide_mask & (~apoptosis_mask)
                stay_mask = stay_mask & (~apoptosis_mask)

        if rd_active:
            candidate_mask = ~die_mask
            if bool(cp.any(candidate_mask)):
                u_signal = cp.clip(self.u.astype(self.dtype, copy=False), 0.0, 1.0)
                p_apop = cp.clip(
                    self.rd_apoptosis_base_p
                    + self.rd_apoptosis_boost * (self.rd_apoptosis_center - u_signal),
                    self.rd_apoptosis_min_p,
                    self.rd_apoptosis_max_p,
                )
                if self.rd_interior_protection and interior_score is not None:
                    # Protect interior/dense cells from random RD apoptosis.
                    # interior_score near 1 strongly suppresses apoptosis.
                    protect = 1.0 - float(self.rd_interior_apoptosis_shield) * interior_score
                    p_apop = cp.clip(
                        p_apop * cp.clip(protect, 0.0, 1.0),
                        0.0,
                        self.rd_apoptosis_max_p,
                    )
                rand_u = self._rng.random_sample(self.points.shape[0]).astype(self.dtype, copy=False)
                rd_apop_mask = candidate_mask & (rand_u < p_apop)
                if bool(cp.any(rd_apop_mask)):
                    n_apoptosis_rd_this_step = self._to_int(rd_apop_mask.sum())
                    n_apoptosis_this_step += n_apoptosis_rd_this_step
                    die_mask = die_mask | rd_apop_mask
                    divide_mask = divide_mask & (~rd_apop_mask)
                    stay_mask = stay_mask & (~rd_apop_mask)
        mark("masks")

        n_divide_eligible = self._to_int(divide_mask.sum())
        if rd_active and bool(cp.any(divide_mask)):
            eligible_divide_mask = divide_mask.copy()
            u_signal = cp.clip(self.u.astype(self.dtype, copy=False), 0.0, 1.0)
            # High u -> higher divide probability (RD-only controls).
            p_min = float(self.rd_divide_min_p)
            p_max = float(self.rd_divide_max_p)
            p_divide = cp.clip(
                self.rd_divide_base_p + self.rd_divide_boost * (u_signal - self.rd_divide_center),
                p_min,
                p_max,
            )
            if self.rd_interior_protection and interior_score is not None and self.rd_interior_divide_damp > 0:
                damp = 1.0 - float(self.rd_interior_divide_damp) * interior_score
                p_divide = cp.clip(
                    p_divide * cp.clip(damp, 0.0, 1.0),
                    0.0,
                    p_max,
                )
            rand_u = self._rng.random_sample(self.points.shape[0]).astype(
                self.dtype,
                copy=False,
            )
            draw_divide = rand_u < p_divide
            divide_mask = eligible_divide_mask & draw_divide
            stay_mask = stay_mask | (eligible_divide_mask & (~draw_divide))
        elif prog_all is not None and bool(cp.any(divide_mask)):
            eligible_divide_mask = divide_mask.copy()
            p_divide = cp.clip(
                1.0 + self.rd_divide_boost * (prog_all - 0.5),
                self.rd_divide_min_p,
                self.rd_divide_max_p,
            )
            rand_u = self._rng.random_sample(self.points.shape[0]).astype(
                self.dtype,
                copy=False,
            )
            draw_divide = rand_u < p_divide
            divide_mask = eligible_divide_mask & draw_divide
            stay_mask = stay_mask | (eligible_divide_mask & (~draw_divide))
        mark("divide_mod")
        n_divide_final = self._to_int(divide_mask.sum())
        n_stay_final = self._to_int(stay_mask.sum())
        n_die_final = self._to_int(die_mask.sum())

        next_blocks = []
        next_id_blocks = []
        next_birth_age_blocks = []
        next_cycle_age_blocks = []
        next_prog_blocks = []
        next_u_blocks = []
        next_v_blocks = []
        next_p_blocks = []
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
            stay_birth_age = self.birth_age[stay_mask] + cp.int32(1)
            stay_cycle_age = self.cycle_age[stay_mask] + cp.int32(1)
            stay_u = self.u[stay_mask]
            stay_v = self.v[stay_mask]
            stay_p = self.p[stay_mask]
            if prog_all is not None:
                stay_prog = prog_all[stay_mask]
            else:
                stay_prog = self.cell_prog[stay_mask]
            next_blocks.append(stay_points)
            next_id_blocks.append(stay_ids)
            next_birth_age_blocks.append(stay_birth_age.astype(cp.int32, copy=False))
            next_cycle_age_blocks.append(stay_cycle_age.astype(cp.int32, copy=False))
            next_prog_blocks.append(stay_prog.astype(self.dtype, copy=False))
            next_u_blocks.append(stay_u.astype(self.dtype, copy=False))
            next_v_blocks.append(stay_v.astype(self.dtype, copy=False))
            next_p_blocks.append(stay_p.astype(self.dtype, copy=False))
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
            if rd_active and self.enable_polarity and v_hat_all is not None:
                if prog_all is not None:
                    prog_dirs = prog_all
                else:
                    prog_dirs = cp.clip(self.u.astype(self.dtype, copy=False), 0.0, 1.0)
                dirs_all = self._division_dirs_from_polarity(
                    v_hat_all,
                    prog_dirs,
                ).astype(self.dtype, copy=False)
            elif prog_all is not None and v_hat_all is not None:
                dirs_all = self._division_dirs_programmed(
                    self.points,
                    v_hat_all,
                    prog_all,
                ).astype(self.dtype, copy=False)
            elif self.division_direction_mode == "least_resistance":
                dirs_all = self._least_resistance_directions()
            else:
                grid = program_grid if program_grid is not None else self._build_grid(self.points)
                v_hat, _ = self._local_neighbor_resultants(
                    self.points,
                    grid,
                    float(self.R_sense),
                    self.neighbor_weight,
                )
                dirs_all = self._division_dirs_from_vhat(v_hat).astype(self.dtype, copy=False)

            if self._force_division_dir_np is not None:
                xp_module = self._xp_of(self.points)
                d_fix = xp_module.asarray(self._force_division_dir_np, dtype=self.dtype)
                dirs_all = xp_module.repeat(d_fix[None, :], int(self.points.shape[0]), axis=0)

            dirs = dirs_all[divide_mask]
            parents = self.points[divide_mask]
            parent_ids = self.cell_ids[divide_mask]
            parent_u = self.u[divide_mask]
            parent_v = self.v[divide_mask]
            parent_p = self.p[divide_mask]
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
            daughters_birth_age = cp.zeros((n_div,), dtype=cp.int32)
            daughters_cycle_age = cp.zeros((n_div,), dtype=cp.int32)
            daughters_prog = cp.zeros((n_div,), dtype=self.dtype)
            daughters_u_a = parent_u.astype(self.dtype, copy=True)
            daughters_u_b = parent_u.astype(self.dtype, copy=True)
            daughters_v_a = parent_v.astype(self.dtype, copy=True)
            daughters_v_b = parent_v.astype(self.dtype, copy=True)
            if rd_active and self.rd_noise > 0:
                noise_amp = float(0.5 * self.rd_noise * np.sqrt(self.rd_dt))
                daughters_u_a += (noise_amp * self._rng.normal(0.0, 1.0, size=n_div)).astype(
                    self.dtype,
                    copy=False,
                )
                daughters_u_b += (noise_amp * self._rng.normal(0.0, 1.0, size=n_div)).astype(
                    self.dtype,
                    copy=False,
                )
                daughters_v_a += (noise_amp * self._rng.normal(0.0, 1.0, size=n_div)).astype(
                    self.dtype,
                    copy=False,
                )
                daughters_v_b += (noise_amp * self._rng.normal(0.0, 1.0, size=n_div)).astype(
                    self.dtype,
                    copy=False,
                )
            if rd_active and self.rd_clamp:
                daughters_u_a = cp.clip(daughters_u_a, 0.0, 1.0)
                daughters_u_b = cp.clip(daughters_u_b, 0.0, 1.0)
                daughters_v_a = cp.clip(daughters_v_a, 0.0, 1.0)
                daughters_v_b = cp.clip(daughters_v_b, 0.0, 1.0)

            daughters_p_a = parent_p.astype(self.dtype, copy=True)
            daughters_p_b = parent_p.astype(self.dtype, copy=True)
            pol_noise = (
                0.1
                * self._rng.normal(0.0, 1.0, size=(n_div, 3)).astype(self.dtype, copy=False)
            )
            daughters_p_a += pol_noise
            daughters_p_b += (
                0.1
                * self._rng.normal(0.0, 1.0, size=(n_div, 3)).astype(self.dtype, copy=False)
            )
            daughters_p_a = self._normalize_vec_rows(daughters_p_a, cp).astype(self.dtype, copy=False)
            daughters_p_b = self._normalize_vec_rows(daughters_p_b, cp).astype(self.dtype, copy=False)
            next_birth_age_blocks.extend([daughters_birth_age, daughters_birth_age.copy()])
            next_cycle_age_blocks.extend([daughters_cycle_age, daughters_cycle_age.copy()])
            next_prog_blocks.extend([daughters_prog, daughters_prog.copy()])
            next_u_blocks.extend([daughters_u_a, daughters_u_b])
            next_v_blocks.extend([daughters_v_a, daughters_v_b])
            next_p_blocks.extend([daughters_p_a, daughters_p_b])
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
            next_birth_age = cp.concatenate(next_birth_age_blocks, axis=0)
            next_cycle_age = cp.concatenate(next_cycle_age_blocks, axis=0)
            next_prog = cp.concatenate(next_prog_blocks, axis=0)
            next_u = cp.concatenate(next_u_blocks, axis=0)
            next_v = cp.concatenate(next_v_blocks, axis=0)
            next_p = cp.concatenate(next_p_blocks, axis=0)
            mark("assemble")
            if self.enforce_non_overlap and next_points.shape[0] > 1:
                if timing_enabled:
                    self._sync_for_timing()
                    t_overlap = time.perf_counter()
                relax_axes = None
                if self.relax_projection_mode == "polarity":
                    relax_axes = next_p
                next_points = self._resolve_overlaps(next_points, relax_axes=relax_axes)
                if timing_enabled:
                    self._sync_for_timing()
                    now = time.perf_counter()
                    timing["overlap"] = timing.get("overlap", 0.0) + (now - t_overlap)
                    t_prev = now
        else:
            next_points = cp.empty((0, 3), dtype=self.dtype)
            next_ids = cp.empty((0,), dtype=cp.int64)
            next_birth_age = cp.empty((0,), dtype=cp.int32)
            next_cycle_age = cp.empty((0,), dtype=cp.int32)
            next_prog = cp.empty((0,), dtype=self.dtype)
            next_u = cp.empty((0,), dtype=self.dtype)
            next_v = cp.empty((0,), dtype=self.dtype)
            next_p = cp.empty((0, 3), dtype=self.dtype)
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
        self.birth_age = next_birth_age
        self.cycle_age = next_cycle_age
        self.cell_prog = next_prog
        self.u = next_u
        self.v = next_v
        self.p = next_p
        self.step_index += 1
        self.count_history.append(int(self.points.shape[0]))
        n_after = int(self.points.shape[0])
        if n_after > 0:
            prog = self.cell_prog.astype(self.dtype, copy=False)
            birth = self.birth_age.astype(self.dtype, copy=False)
            cycle = self.cycle_age.astype(self.dtype, copy=False)
            self.last_program_summary = {
                "step": float(self.step_index),
                "cells": float(n_after),
                "prog_mean": self._to_float(cp.mean(prog)),
                "prog_min": self._to_float(cp.min(prog)),
                "prog_max": self._to_float(cp.max(prog)),
                "birth_mean": self._to_float(cp.mean(birth)),
                "cycle_mean": self._to_float(cp.mean(cycle)),
                "divide": float(n_divide_final),
                "divide_eligible": float(n_divide_eligible),
                "stay": float(n_stay_final),
                "die": float(n_die_final),
                "apoptosis": float(n_apoptosis_this_step),
                "apoptosis_rd": float(n_apoptosis_rd_this_step),
                "interior_mean": float(interior_mean_this_step),
            }
        else:
            self.last_program_summary = {
                "step": float(self.step_index),
                "cells": 0.0,
                "prog_mean": 0.0,
                "prog_min": 0.0,
                "prog_max": 0.0,
                "birth_mean": 0.0,
                "cycle_mean": 0.0,
                "divide": float(n_divide_final),
                "divide_eligible": float(n_divide_eligible),
                "stay": float(n_stay_final),
                "die": float(n_die_final),
                "apoptosis": float(n_apoptosis_this_step),
                "apoptosis_rd": float(n_apoptosis_rd_this_step),
                "interior_mean": float(interior_mean_this_step),
            }
        if (
            self.enable_reaction_diffusion
            and self.step_index > int(self.rd_start_step)
            and self.rd_print_stats_every > 0
        ):
            if (self.step_index % int(self.rd_print_stats_every)) == 0:
                if int(self.points.shape[0]) > 0:
                    prog = self.cell_prog.astype(self.dtype, copy=False)
                    u = self.u.astype(self.dtype, copy=False)
                    v = self.v.astype(self.dtype, copy=False)
                    self.last_rd_summary = {
                        "step": float(self.step_index),
                        "u_min": self._to_float(cp.min(u)),
                        "u_mean": self._to_float(cp.mean(u)),
                        "u_max": self._to_float(cp.max(u)),
                        "v_min": self._to_float(cp.min(v)),
                        "v_mean": self._to_float(cp.mean(v)),
                        "v_max": self._to_float(cp.max(v)),
                        "prog_min": self._to_float(cp.min(prog)),
                        "prog_mean": self._to_float(cp.mean(prog)),
                        "prog_max": self._to_float(cp.max(prog)),
                        "polarity_coherence": float(self.last_polarity_coherence),
                    }
                else:
                    self.last_rd_summary = {
                        "step": float(self.step_index),
                        "u_min": 0.0,
                        "u_mean": 0.0,
                        "u_max": 0.0,
                        "v_min": 0.0,
                        "v_mean": 0.0,
                        "v_max": 0.0,
                        "prog_min": 0.0,
                        "prog_mean": 0.0,
                        "prog_max": 0.0,
                        "polarity_coherence": float(self.last_polarity_coherence),
                    }
            else:
                self.last_rd_summary = None
        else:
            self.last_rd_summary = None
        self._assert_state_aligned()
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
            if self.print_program_summary and self.last_program_summary:
                print(self._format_program_summary(self.last_program_summary))
            if self.last_rd_summary is not None:
                print(self._format_rd_summary(self.last_rd_summary))
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
        interp_frames: int = 2,
        fps: int = 24,
        movie_duration_seconds: Optional[float] = DEFAULT_MOVIE_DURATION_SECONDS,
        show_centers: bool = False,
        color_by: str = "u",
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
        If movie_duration_seconds is set, output FPS is auto-scaled to match it.
        Movie rendering uses a fixed camera/bounds from final colony size and
        static-plot style for sphere appearance (color/opacity/tessellation).
        """
        if interp_frames < 1:
            raise ValueError("interp_frames must be >= 1")
        if large_interp_frames < 1:
            raise ValueError("large_interp_frames must be >= 1")
        if fps < 1:
            raise ValueError("fps must be >= 1")
        if movie_duration_seconds is not None and movie_duration_seconds <= 0:
            raise ValueError("movie_duration_seconds must be > 0 when provided")
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
        valid_color_modes = {"order", "radius", "none", "u", "v", "prog", "age", "pz"}
        if color_by not in valid_color_modes:
            raise ValueError(f"color_by must be one of {sorted(valid_color_modes)}")
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

        scalar_mode = color_by in {"u", "v", "prog", "age", "pz"}

        def current_movie_scalar_state() -> np.ndarray:
            if color_by == "u":
                return to_numpy(self.u.copy()).astype(float, copy=False)
            if color_by == "v":
                return to_numpy(self.v.copy()).astype(float, copy=False)
            if color_by == "prog":
                return to_numpy(self.cell_prog.copy()).astype(float, copy=False)
            if color_by == "age":
                return to_numpy(self.birth_age.copy()).astype(float, copy=False)
            if color_by == "pz":
                return to_numpy(self.p[:, 2].copy()).astype(float, copy=False)
            return np.empty((0,), dtype=float)

        # Pass 1: simulate and capture transitions.
        initial_points_np = to_numpy(self.points.copy())
        initial_ids_np = to_numpy(self.cell_ids.copy())
        initial_scalar_state_np = current_movie_scalar_state() if scalar_mode else None
        scalar_range_min = np.inf
        scalar_range_max = -np.inf
        if scalar_mode and initial_scalar_state_np is not None and initial_scalar_state_np.size > 0:
            scalar_range_min = min(scalar_range_min, float(initial_scalar_state_np.min()))
            scalar_range_max = max(scalar_range_max, float(initial_scalar_state_np.max()))
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
        scalar_transitions_np: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for _ in range(n_steps):
            if self.points.shape[0] == 0:
                break
            if scalar_mode:
                src_state_ids_np = to_numpy(self.cell_ids.copy()).astype(np.int64, copy=False)
                src_scalar_state_np = current_movie_scalar_state()
            else:
                src_state_ids_np = np.empty((0,), dtype=np.int64)
                src_scalar_state_np = np.empty((0,), dtype=float)
            transition = self.step_with_transition(
                action_rule=action_rule,
                death_animation=death_animation,
                profile_timing=log_timing,
            )
            if scalar_mode:
                tgt_state_ids_np = to_numpy(self.cell_ids.copy()).astype(np.int64, copy=False)
                tgt_scalar_state_np = current_movie_scalar_state()
                if src_scalar_state_np.size > 0:
                    scalar_range_min = min(scalar_range_min, float(src_scalar_state_np.min()))
                    scalar_range_max = max(scalar_range_max, float(src_scalar_state_np.max()))
                if tgt_scalar_state_np.size > 0:
                    scalar_range_min = min(scalar_range_min, float(tgt_scalar_state_np.min()))
                    scalar_range_max = max(scalar_range_max, float(tgt_scalar_state_np.max()))
                scalar_transitions_np.append(
                    (src_state_ids_np, src_scalar_state_np, tgt_state_ids_np, tgt_scalar_state_np)
                )
            if log_counts:
                print(f"Step {self.step_index}: {self.points.shape[0]} cells")
            if self.print_program_summary and self.last_program_summary:
                print(self._format_program_summary(self.last_program_summary))
            if self.last_rd_summary is not None:
                print(self._format_rd_summary(self.last_rd_summary))
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
        movie_scalar_clim: Optional[tuple[float, float]] = None
        if scalar_mode and np.isfinite(scalar_range_min) and np.isfinite(scalar_range_max):
            if color_by == "age":
                lo = 0.0
                hi = max(1.0, float(scalar_range_max))
            else:
                lo = float(scalar_range_min)
                hi = float(scalar_range_max)
                if hi <= lo:
                    hi = lo + 1.0
            movie_scalar_clim = (lo, hi)
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

        planned_frame_count = 1
        for src, _tgt, *_ in transitions_np:
            if src.shape[0] == 0:
                planned_frame_count += 1
            else:
                planned_frame_count += int(effective_interp_frames)

        effective_fps = float(fps)
        if movie_duration_seconds is not None:
            target_seconds = float(movie_duration_seconds)
            auto_fps = float(planned_frame_count) / target_seconds
            if auto_fps < 1.0:
                print(
                    "Movie duration request implies FPS < 1.0; clamping to 1.0 "
                    f"(requested duration={target_seconds:.2f}s, frames={planned_frame_count})."
                )
                effective_fps = 1.0
            else:
                effective_fps = auto_fps
            expected_duration = float(planned_frame_count) / effective_fps
            print(
                "Movie timing: "
                f"target={target_seconds:.2f}s, frames={planned_frame_count}, "
                f"fps={effective_fps:.2f}, expected={expected_duration:.2f}s."
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
        view_dir = np.array([-1.0, -1.0, 1.0], dtype=float)
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
        pl.open_movie(str(out), framerate=effective_fps, macro_block_size=macro_block_size)
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

        def lookup_scalar_by_ids(
            query_ids: np.ndarray,
            state_ids: np.ndarray,
            state_scalars: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            q = np.asarray(query_ids, dtype=np.int64).reshape(-1)
            out = np.zeros(q.shape[0], dtype=float)
            valid = np.zeros(q.shape[0], dtype=bool)
            if q.size == 0 or state_ids.size == 0 or state_scalars.size == 0:
                return out, valid
            sid = np.asarray(state_ids, dtype=np.int64).reshape(-1)
            sval = np.asarray(state_scalars, dtype=float).reshape(-1)
            order = np.argsort(sid, kind="mergesort")
            sid_sorted = sid[order]
            sval_sorted = sval[order]
            pos = np.searchsorted(sid_sorted, q)
            in_bounds = pos < sid_sorted.size
            if sid_sorted.size == 0:
                return out, valid
            pos_safe = np.minimum(pos, sid_sorted.size - 1)
            matched = in_bounds & (sid_sorted[pos_safe] == q)
            if np.any(matched):
                out[matched] = sval_sorted[pos_safe[matched]]
                valid[matched] = True
            return out, valid

        def write_frame(
            points_np: np.ndarray,
            order_np: np.ndarray,
            size_np: np.ndarray,
            alpha_np: np.ndarray,
            scalar_np: Optional[np.ndarray] = None,
        ) -> None:
            pts = np.asarray(points_np, dtype=float)
            order = np.asarray(order_np, dtype=float).reshape(-1)
            size = np.asarray(size_np, dtype=float).reshape(-1)
            alpha = np.asarray(alpha_np, dtype=float).reshape(-1)
            scalar = None if scalar_np is None else np.asarray(scalar_np, dtype=float).reshape(-1)
            if pts.shape[0] != order.shape[0]:
                raise ValueError("Frame order array length must match frame point count.")
            if pts.shape[0] != size.shape[0]:
                raise ValueError("Frame size array length must match frame point count.")
            if pts.shape[0] != alpha.shape[0]:
                raise ValueError("Frame alpha array length must match frame point count.")
            if scalar is not None and pts.shape[0] != scalar.shape[0]:
                raise ValueError("Frame scalar array length must match frame point count.")

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
                if scalar is not None:
                    scalar = scalar[keep]

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
                elif color_by in {"u", "v", "prog", "age", "pz"}:
                    if scalar is None:
                        raise ValueError(f"color_by='{color_by}' requires scalar frame values.")
                    spheres["val"] = np.repeat(scalar, base_sphere.n_points)
                    mesh_kwargs = dict(
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
                    if movie_scalar_clim is not None:
                        mesh_kwargs["clim"] = movie_scalar_clim
                    pl.add_mesh(spheres, **mesh_kwargs)
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
            if scalar_mode and initial_scalar_state_np is not None:
                initial_scalar = initial_scalar_state_np.astype(float, copy=False)
            else:
                initial_scalar = None
            write_frame(initial_points_np, initial_ids_np, initial_size, initial_alpha, initial_scalar)

            for idx, (src, tgt, src_ids, tgt_ids, src_size, tgt_size, src_alpha, tgt_alpha) in enumerate(transitions_np):
                if scalar_mode and idx < len(scalar_transitions_np):
                    src_state_ids, src_state_vals, tgt_state_ids, tgt_state_vals = scalar_transitions_np[idx]
                else:
                    src_state_ids = np.empty((0,), dtype=np.int64)
                    src_state_vals = np.empty((0,), dtype=float)
                    tgt_state_ids = np.empty((0,), dtype=np.int64)
                    tgt_state_vals = np.empty((0,), dtype=float)
                if src.shape[0] == 0:
                    if scalar_mode:
                        tgt_scalar_vals, _ = lookup_scalar_by_ids(tgt_ids, tgt_state_ids, tgt_state_vals)
                    else:
                        tgt_scalar_vals = None
                    write_frame(tgt, tgt_ids, tgt_size, tgt_alpha, tgt_scalar_vals)
                    continue

                for i in range(1, effective_interp_frames + 1):
                    alpha = i / float(effective_interp_frames)
                    frame_pts = (1.0 - alpha) * src + alpha * tgt
                    frame_order = (1.0 - alpha) * src_ids + alpha * tgt_ids
                    frame_size = (1.0 - alpha) * src_size + alpha * tgt_size
                    frame_alpha = (1.0 - alpha) * src_alpha + alpha * tgt_alpha
                    if scalar_mode:
                        src_vals, _ = lookup_scalar_by_ids(src_ids, src_state_ids, src_state_vals)
                        tgt_vals, tgt_ok = lookup_scalar_by_ids(tgt_ids, tgt_state_ids, tgt_state_vals)
                        if not np.all(tgt_ok):
                            fallback_vals, _ = lookup_scalar_by_ids(tgt_ids, src_state_ids, src_state_vals)
                            tgt_vals = np.where(tgt_ok, tgt_vals, fallback_vals)
                        frame_scalar = (1.0 - alpha) * src_vals + alpha * tgt_vals
                    else:
                        frame_scalar = None
                    write_frame(frame_pts, frame_order, frame_size, frame_alpha, frame_scalar)
        finally:
            pl.close()

        print(f"Movie saved: {out}")
        return self.points

    def points_numpy(self):
        """Return positions as NumPy array for plotting/export."""
        if _GPU_ENABLED:
            return cp.asnumpy(self.points)
        return self.points

    def save_data_npz(self, output_path: str) -> Path:
        """Save final simulation state to a compressed NPZ file."""
        out = Path(output_path)
        if out.parent and not out.parent.exists():
            out.parent.mkdir(parents=True, exist_ok=True)

        pts = self.points_numpy().astype(np.float32, copy=False)
        ids = cp.asnumpy(self.cell_ids) if _GPU_ENABLED else np.asarray(self.cell_ids)
        birth_age = cp.asnumpy(self.birth_age) if _GPU_ENABLED else np.asarray(self.birth_age)
        cycle_age = cp.asnumpy(self.cycle_age) if _GPU_ENABLED else np.asarray(self.cycle_age)
        cell_prog = cp.asnumpy(self.cell_prog) if _GPU_ENABLED else np.asarray(self.cell_prog)
        u = cp.asnumpy(self.u) if _GPU_ENABLED else np.asarray(self.u)
        v = cp.asnumpy(self.v) if _GPU_ENABLED else np.asarray(self.v)
        p = cp.asnumpy(self.p) if _GPU_ENABLED else np.asarray(self.p)
        np.savez_compressed(
            out,
            points=pts,
            cell_ids=ids.astype(np.int64, copy=False),
            birth_age=birth_age.astype(np.int32, copy=False),
            cycle_age=cycle_age.astype(np.int32, copy=False),
            cell_prog=cell_prog.astype(np.float32, copy=False),
            u=u.astype(np.float32, copy=False),
            v=v.astype(np.float32, copy=False),
            p=p.astype(np.float32, copy=False),
            count_history=np.asarray(self.count_history, dtype=np.int32),
            step_index=np.int32(self.step_index),
            next_cell_id=np.int64(self._next_cell_id),
            radius=np.float32(self.radius),
            split_distance=np.float32(self.split_distance),
            density_radius=np.float32(self.density_radius),
            R_sense=np.float32(self.R_sense),
            grid_cell_size=np.float32(self.grid_cell_size),
            crowding_stay_threshold=np.int32(self.crowding_stay_threshold),
            crowding_death_threshold=np.int32(self.crowding_death_threshold),
            division_direction_mode=np.asarray(self.division_direction_mode),
            neighbor_weight=np.asarray(self.neighbor_weight),
            radial_sign=np.asarray(self.radial_sign),
            program_surface_gain=np.float32(self.program_surface_gain),
            program_crowd_gain=np.float32(self.program_crowd_gain),
            program_noise=np.float32(self.program_noise),
            program_sigmoid_center=np.float32(self.program_sigmoid_center),
            program_sigmoid_slope=np.float32(self.program_sigmoid_slope),
            program_radial_sign=np.asarray(self.program_radial_sign),
            program_divide_boost=np.float32(self.program_divide_boost),
            program_divide_min_p=np.float32(self.program_divide_min_p),
            program_divide_max_p=np.float32(self.program_divide_max_p),
            enable_apoptosis=np.int8(1 if self.enable_apoptosis else 0),
            apoptosis_age=np.int32(self.apoptosis_age),
            rest_distance_factor=np.float32(self.rest_distance_factor),
            adhesion_radius_factor=np.float32(self.adhesion_radius_factor),
            spring_k=np.float32(self.spring_k),
            spring_max_step=np.float32(self.spring_max_step),
            enable_reaction_diffusion=np.int8(1 if self.enable_reaction_diffusion else 0),
            R_signal=np.float32(self.R_signal),
            rd_dt=np.float32(self.rd_dt),
            rd_substeps=np.int32(self.rd_substeps),
            Du=np.float32(self.Du),
            Dv=np.float32(self.Dv),
            rd_model=np.asarray(self.rd_model),
            gs_F=np.float32(self.gs_F),
            gs_k=np.float32(self.gs_k),
            rd_clamp=np.int8(1 if self.rd_clamp else 0),
            rd_noise=np.float32(self.rd_noise),
            rd_init_mode=np.asarray(self.rd_init_mode),
            rd_seed_amp=np.float32(self.rd_seed_amp),
            rd_seed_frac=np.float32(self.rd_seed_frac),
            rd_print_stats_every=np.int32(self.rd_print_stats_every),
            rd_couple_to_prog=np.int8(1 if self.rd_couple_to_prog else 0),
            rd_prog_from=np.asarray(self.rd_prog_from),
            rd_prog_gain=np.float32(self.rd_prog_gain),
            rd_prog_center=np.float32(self.rd_prog_center),
            rd_divide_base_p=np.float32(self.rd_divide_base_p),
            rd_divide_boost=np.float32(self.rd_divide_boost),
            rd_divide_center=np.float32(self.rd_divide_center),
            rd_divide_min_p=np.float32(self.rd_divide_min_p),
            rd_divide_max_p=np.float32(self.rd_divide_max_p),
            rd_start_step=np.int32(self.rd_start_step),
            rd_apoptosis_boost=np.float32(self.rd_apoptosis_boost),
            rd_apoptosis_base_p=np.float32(self.rd_apoptosis_base_p),
            rd_apoptosis_center=np.float32(self.rd_apoptosis_center),
            rd_apoptosis_min_p=np.float32(self.rd_apoptosis_min_p),
            rd_apoptosis_max_p=np.float32(self.rd_apoptosis_max_p),
            rd_interior_protection=np.int8(1 if self.rd_interior_protection else 0),
            rd_interior_apoptosis_shield=np.float32(self.rd_interior_apoptosis_shield),
            rd_interior_divide_damp=np.float32(self.rd_interior_divide_damp),
            rd_interior_crowd_weight=np.float32(self.rd_interior_crowd_weight),
            enable_polarity=np.int8(1 if self.enable_polarity else 0),
            polarity_noise=np.float32(self.polarity_noise),
            polarity_align_alpha0=np.float32(self.polarity_align_alpha0),
            polarity_align_alpha_u=np.float32(self.polarity_align_alpha_u),
            polarity_radius=np.float32(self.polarity_radius if self.polarity_radius is not None else -1.0),
            polarity_project_to_tangent=np.int8(1 if self.polarity_project_to_tangent else 0),
            polarity_use_u_gradient=np.int8(1 if self.polarity_use_u_gradient else 0),
            polarity_grad_gain=np.float32(self.polarity_grad_gain),
            polarity_mix_prev=np.float32(self.polarity_mix_prev),
            force_division_direction=np.asarray(
                self.force_division_direction if self.force_division_direction is not None else ""
            ),
            relax_projection_mode=np.asarray(self.relax_projection_mode),
            dtype=np.asarray(self.dtype),
        )
        print(f"Saved data: {out}")
        return out

    def visualize_pyvista(self, **kwargs) -> None:
        """Render final cells as spheres in PyVista."""
        opts = dict(kwargs)
        color_by = opts.get("color_by", "order")
        scalar_values = None
        if color_by == "u":
            scalar_values = self._as_numpy(self.u).astype(float, copy=False)
            opts["color_by"] = "scalar"
        elif color_by == "v":
            scalar_values = self._as_numpy(self.v).astype(float, copy=False)
            opts["color_by"] = "scalar"
        elif color_by == "prog":
            scalar_values = self._as_numpy(self.cell_prog).astype(float, copy=False)
            opts["color_by"] = "scalar"
        elif color_by == "age":
            scalar_values = self._as_numpy(self.birth_age).astype(float, copy=False)
            opts["color_by"] = "scalar"
        elif color_by == "pz":
            scalar_values = self._as_numpy(self.p[:, 2]).astype(float, copy=False)
            opts["color_by"] = "scalar"
        if scalar_values is not None:
            opts["scalar_values"] = scalar_values
        show_cells_pyvista(self.points_numpy(), cell_radius=self.radius, **opts)

    def save_snapshot_pyvista(self, output_path: str, **kwargs) -> Path:
        """Save final cells as a static PyVista snapshot image."""
        opts = dict(kwargs)
        color_by = opts.get("color_by", "order")
        scalar_values = None
        if color_by == "u":
            scalar_values = self._as_numpy(self.u).astype(float, copy=False)
            opts["color_by"] = "scalar"
        elif color_by == "v":
            scalar_values = self._as_numpy(self.v).astype(float, copy=False)
            opts["color_by"] = "scalar"
        elif color_by == "prog":
            scalar_values = self._as_numpy(self.cell_prog).astype(float, copy=False)
            opts["color_by"] = "scalar"
        elif color_by == "age":
            scalar_values = self._as_numpy(self.birth_age).astype(float, copy=False)
            opts["color_by"] = "scalar"
        elif color_by == "pz":
            scalar_values = self._as_numpy(self.p[:, 2]).astype(float, copy=False)
            opts["color_by"] = "scalar"
        if scalar_values is not None:
            opts["scalar_values"] = scalar_values
        out = show_cells_pyvista(
            self.points_numpy(),
            cell_radius=self.radius,
            snapshot_path=output_path,
            **opts,
        )
        if out is None:
            raise RuntimeError("Snapshot export did not produce an output path.")
        return out


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
        default=500_000,
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
    parser.add_argument(
        "--apoptosis",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_APOPTOSIS,
        help="Enable age-based apoptosis (birth_age >= apoptosis_age)",
    )
    parser.add_argument(
        "--apoptosis-age",
        type=int,
        default=DEFAULT_APOPTOSIS_AGE,
        help=f"Chronological age threshold for apoptosis (default: {DEFAULT_APOPTOSIS_AGE})",
    )
    parser.add_argument(
        "--program-surface-gain",
        type=float,
        default=DEFAULT_PROGRAM_SURFACE_GAIN,
        help="Program gain from local surface-ness (|v_hat|)",
    )
    parser.add_argument(
        "--program-crowd-gain",
        type=float,
        default=DEFAULT_PROGRAM_CROWD_GAIN,
        help="Program suppression gain from local crowding",
    )
    parser.add_argument(
        "--radial-sign",
        choices=["outward", "inward", "random"],
        default="outward",
        help="Radial division sign when using radial/programmed directions",
    )
    parser.add_argument(
        "--program-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-step program/action summary",
    )
    parser.add_argument(
        "--reaction-diffusion",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_REACTION_DIFFUSION,
        help="Enable self-organized reaction-diffusion fields on the moving neighbor graph",
    )
    parser.add_argument("--Du", type=float, default=DEFAULT_RD_DU, help="RD diffusion coefficient for u")
    parser.add_argument("--Dv", type=float, default=DEFAULT_RD_DV, help="RD diffusion coefficient for v")
    parser.add_argument(
        "--rd-model",
        choices=["gray_scott"],
        default=DEFAULT_RD_MODEL,
        help="Reaction-diffusion kinetics model",
    )
    parser.add_argument(
        "--gs-F",
        type=float,
        default=DEFAULT_GS_F,
        help="RD parameter a in du/dt..dv/dt terms: +a*(1-v) and -(a+b)*u",
    )
    parser.add_argument(
        "--gs-k",
        type=float,
        default=DEFAULT_GS_K,
        help="RD parameter b in the -(a+b)*u term",
    )
    parser.add_argument(
        "--R-signal",
        type=float,
        default=None,
        help="Neighbor radius for RD graph diffusion (default: density radius)",
    )
    parser.add_argument("--rd-dt", type=float, default=DEFAULT_RD_DT, help="RD integration timestep")
    parser.add_argument(
        "--rd-substeps",
        type=int,
        default=DEFAULT_RD_SUBSTEPS,
        help="Number of RD timesteps per cell timestep",
    )
    parser.add_argument("--rd-noise", type=float, default=DEFAULT_RD_NOISE, help="RD additive noise amplitude")
    parser.add_argument(
        "--rd-init-mode",
        choices=["uniform_noise", "seed_center", "seed_random_cells"],
        default=DEFAULT_RD_INIT_MODE,
        help="Initial perturbation mode for RD fields",
    )
    parser.add_argument("--rd-seed-amp", type=float, default=DEFAULT_RD_SEED_AMP, help="RD seed perturbation amplitude")
    parser.add_argument("--rd-seed-frac", type=float, default=DEFAULT_RD_SEED_FRAC, help="Fraction of seeded cells for rd-init-mode=seed_random_cells")
    parser.add_argument("--rd-print-stats-every", type=int, default=DEFAULT_RD_PRINT_STATS_EVERY, help="Print RD stats every N steps (0 disables)")
    parser.add_argument(
        "--rd-clamp",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RD_CLAMP,
        help="Clamp RD fields u,v to [0,1] after each step",
    )
    parser.add_argument(
        "--rd-prog-from",
        choices=["u", "v", "u_minus_v", "v_minus_u"],
        default=DEFAULT_RD_PROG_FROM,
        help="Legacy (ignored): program scalar now uses sigmoid(u)",
    )
    parser.add_argument("--rd-prog-gain", type=float, default=DEFAULT_RD_PROG_GAIN, help="Legacy (ignored): program scalar now uses sigmoid(u)")
    parser.add_argument(
        "--rd-prog-center",
        type=float,
        default=DEFAULT_RD_PROG_CENTER,
        help="Legacy (ignored): program scalar now uses sigmoid(u)",
    )
    parser.add_argument(
        "--rd-divide-boost",
        type=float,
        default=DEFAULT_RD_DIVIDE_BOOST,
        help="Optional divide-probability coupling strength from RD program",
    )
    parser.add_argument(
        "--rd-divide-center",
        type=float,
        default=DEFAULT_RD_DIVIDE_CENTER,
        help="u center for RD divide gating (higher than center increases division)",
    )
    parser.add_argument(
        "--rd-divide-min-p",
        type=float,
        default=DEFAULT_RD_DIVIDE_MIN_P,
        help="Minimum per-cell divide probability in RD divide gating",
    )
    parser.add_argument(
        "--rd-divide-max-p",
        type=float,
        default=DEFAULT_RD_DIVIDE_MAX_P,
        help="Maximum per-cell divide probability in RD divide gating",
    )
    parser.add_argument(
        "--rd-start-step",
        type=int,
        default=DEFAULT_RD_START_STEP,
        help="Enable RD dynamics/effects only after this many completed steps",
    )
    parser.add_argument(
        "--rd-apoptosis-boost",
        type=float,
        default=DEFAULT_RD_APOPTOSIS_BOOST,
        help="RD apoptosis boost: lower u increases apoptosis probability",
    )
    parser.add_argument(
        "--rd-apoptosis-base-p",
        type=float,
        default=DEFAULT_RD_APOPTOSIS_BASE_P,
        help="Baseline RD apoptosis probability before u-dependent boost",
    )
    parser.add_argument(
        "--rd-apoptosis-center",
        type=float,
        default=DEFAULT_RD_APOPTOSIS_CENTER,
        help="u center around which RD apoptosis boost is applied",
    )
    parser.add_argument(
        "--rd-apoptosis-min-p",
        type=float,
        default=DEFAULT_RD_APOPTOSIS_MIN_P,
        help="Minimum per-cell apoptosis probability in RD apoptosis gating",
    )
    parser.add_argument(
        "--rd-apoptosis-max-p",
        type=float,
        default=DEFAULT_RD_APOPTOSIS_MAX_P,
        help="Maximum per-cell apoptosis probability in RD apoptosis gating",
    )
    parser.add_argument(
        "--rd-interior-protection",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RD_INTERIOR_PROTECTION,
        help="Protect dense/interior cells from RD-driven random apoptosis",
    )
    parser.add_argument(
        "--rd-interior-apoptosis-shield",
        type=float,
        default=DEFAULT_RD_INTERIOR_APOPTOSIS_SHIELD,
        help="Strength of interior apoptosis suppression (0..1)",
    )
    parser.add_argument(
        "--rd-interior-divide-damp",
        type=float,
        default=DEFAULT_RD_INTERIOR_DIVIDE_DAMP,
        help="Optional interior divide damping (0..1, 0 disables)",
    )
    parser.add_argument(
        "--rd-interior-crowd-weight",
        type=float,
        default=DEFAULT_RD_INTERIOR_CROWD_WEIGHT,
        help="Weight of crowding in interior score (remaining weight uses 1-surface)",
    )
    parser.add_argument(
        "--rd-couple-to-prog",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RD_COUPLE_TO_PROG,
        help="Use RD fields to drive tangential/radial division-direction blend",
    )
    parser.add_argument(
        "--polarity",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_POLARITY,
        help="Enable self-organized planar polarity alignment for RD-active division orientation",
    )
    parser.add_argument(
        "--polarity-noise",
        type=float,
        default=DEFAULT_POLARITY_NOISE,
        help="Per-step polarity noise amplitude",
    )
    parser.add_argument(
        "--polarity-align-alpha0",
        type=float,
        default=DEFAULT_POLARITY_ALIGN_ALPHA0,
        help="Baseline neighbor alignment strength for polarity",
    )
    parser.add_argument(
        "--polarity-align-alpha-u",
        type=float,
        default=DEFAULT_POLARITY_ALIGN_ALPHA_U,
        help="Additional polarity alignment strength proportional to local u",
    )
    parser.add_argument(
        "--polarity-radius",
        type=float,
        default=DEFAULT_POLARITY_RADIUS,
        help="Neighbor radius for polarity alignment (default: R_signal)",
    )
    parser.add_argument(
        "--polarity-project-to-tangent",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_POLARITY_PROJECT_TO_TANGENT,
        help="Project polarity and optional u-gradient to local tangent plane from v_hat",
    )
    parser.add_argument(
        "--polarity-use-u-gradient",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_POLARITY_USE_U_GRADIENT,
        help="Blend local u-gradient into polarity update",
    )
    parser.add_argument(
        "--polarity-grad-gain",
        type=float,
        default=DEFAULT_POLARITY_GRAD_GAIN,
        help="Blend gain for u-gradient contribution to polarity direction",
    )
    parser.add_argument(
        "--polarity-mix-prev",
        type=float,
        default=DEFAULT_POLARITY_MIX_PREV,
        help="Polarity inertia (0=no memory, 1=fully persistent)",
    )
    parser.add_argument(
        "--force-division-direction",
        type=str,
        default=DEFAULT_FORCE_DIVISION_DIRECTION,
        help="Override all division directions with a fixed unit vector, format: x,y,z",
    )
    parser.add_argument(
        "--relax-projection-mode",
        choices=["none", "force_dir", "polarity"],
        default=DEFAULT_RELAX_PROJECTION_MODE,
        help="Project spring-relaxer displacements: none, force_dir, or polarity",
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
        choices=["order", "radius", "solid", "u", "v", "prog", "age", "pz"],
        default=DEFAULT_COLOR_BY,
        help="Sphere coloring mode in PyVista view",
    )
    parser.add_argument(
        "--save-data",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_DATA,
        help="Save final cell state to a compressed NPZ file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Output NPZ path for --save-data",
    )
    parser.add_argument(
        "--save-snapshot",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_SNAPSHOT,
        help="Save final PyVista static snapshot as a PNG image",
    )
    parser.add_argument(
        "--snapshot-path",
        type=str,
        default=DEFAULT_SNAPSHOT_PATH,
        help="Output PNG path for --save-snapshot",
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
        "--movie-duration-seconds",
        type=float,
        default=DEFAULT_MOVIE_DURATION_SECONDS,
        help=(
            "Target movie duration in seconds; when set, FPS is auto-adjusted "
            "to match this duration."
        ),
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
        program_surface_gain=args.program_surface_gain,
        program_crowd_gain=args.program_crowd_gain,
        radial_sign=args.radial_sign,
        print_program_summary=args.program_summary,
        enable_apoptosis=args.apoptosis,
        apoptosis_age=args.apoptosis_age,
        enable_reaction_diffusion=args.reaction_diffusion,
        R_signal=args.R_signal,
        rd_dt=args.rd_dt,
        rd_substeps=args.rd_substeps,
        Du=args.Du,
        Dv=args.Dv,
        rd_model=args.rd_model,
        gs_F=args.gs_F,
        gs_k=args.gs_k,
        rd_clamp=args.rd_clamp,
        rd_noise=args.rd_noise,
        rd_init_mode=args.rd_init_mode,
        rd_seed_amp=args.rd_seed_amp,
        rd_seed_frac=args.rd_seed_frac,
        rd_print_stats_every=args.rd_print_stats_every,
        rd_prog_from=args.rd_prog_from,
        rd_prog_gain=args.rd_prog_gain,
        rd_prog_center=args.rd_prog_center,
        rd_divide_boost=args.rd_divide_boost,
        rd_divide_center=args.rd_divide_center,
        rd_divide_min_p=args.rd_divide_min_p,
        rd_divide_max_p=args.rd_divide_max_p,
        rd_start_step=args.rd_start_step,
        rd_apoptosis_boost=args.rd_apoptosis_boost,
        rd_apoptosis_base_p=args.rd_apoptosis_base_p,
        rd_apoptosis_center=args.rd_apoptosis_center,
        rd_apoptosis_min_p=args.rd_apoptosis_min_p,
        rd_apoptosis_max_p=args.rd_apoptosis_max_p,
        rd_interior_protection=args.rd_interior_protection,
        rd_interior_apoptosis_shield=args.rd_interior_apoptosis_shield,
        rd_interior_divide_damp=args.rd_interior_divide_damp,
        rd_interior_crowd_weight=args.rd_interior_crowd_weight,
        rd_couple_to_prog=args.rd_couple_to_prog,
        enable_polarity=args.polarity,
        polarity_noise=args.polarity_noise,
        polarity_align_alpha0=args.polarity_align_alpha0,
        polarity_align_alpha_u=args.polarity_align_alpha_u,
        polarity_radius=args.polarity_radius,
        polarity_project_to_tangent=args.polarity_project_to_tangent,
        polarity_use_u_gradient=args.polarity_use_u_gradient,
        polarity_grad_gain=args.polarity_grad_gain,
        polarity_mix_prev=args.polarity_mix_prev,
        force_division_direction=args.force_division_direction,
        relax_projection_mode=args.relax_projection_mode,
    )
    print(backend_summary())
    if _GPU_ENABLED:
        if getattr(sim, "_gpu_kernels_ready", False):
            print("GPU neighbor/overlap raw kernels: enabled")
        else:
            err = getattr(sim, "_gpu_kernel_error", None)
            print(f"GPU neighbor/overlap raw kernels: disabled ({err})")
    print(f"Apoptosis: enabled={sim.enable_apoptosis}, age_threshold={sim.apoptosis_age}")
    print(
        "Reaction-diffusion: "
        f"enabled={sim.enable_reaction_diffusion}, model={sim.rd_model}, "
        f"R_signal={sim.R_signal:g}, Du={sim.Du:g}, Dv={sim.Dv:g}, "
        f"dt={sim.rd_dt:g}, substeps={sim.rd_substeps}, "
        f"F={sim.gs_F:g}, k={sim.gs_k:g}, start_step={sim.rd_start_step}"
    )
    if sim.enable_reaction_diffusion:
        print(
            "RD coupling: "
            f"divide_boost={sim.rd_divide_boost:g} "
            f"(center={sim.rd_divide_center:g}, min/max={sim.rd_divide_min_p:g}/{sim.rd_divide_max_p:g}), "
            f"apoptosis_boost={sim.rd_apoptosis_boost:g} "
            f"(base={sim.rd_apoptosis_base_p:g}, center={sim.rd_apoptosis_center:g}, "
            f"min/max={sim.rd_apoptosis_min_p:g}/{sim.rd_apoptosis_max_p:g}), "
            f"interior_protection={sim.rd_interior_protection} "
            f"(shield={sim.rd_interior_apoptosis_shield:g}, divide_damp={sim.rd_interior_divide_damp:g}, "
            f"crowd_weight={sim.rd_interior_crowd_weight:g})"
        )
        print(
            "Polarity: "
            f"enabled={sim.enable_polarity}, noise={sim.polarity_noise:g}, "
            f"alpha0={sim.polarity_align_alpha0:g}, alpha_u={sim.polarity_align_alpha_u:g}, "
            f"radius={sim.polarity_radius if sim.polarity_radius is not None else sim.R_signal:g}, "
            f"proj_tangent={sim.polarity_project_to_tangent}, "
            f"use_u_grad={sim.polarity_use_u_gradient}, grad_gain={sim.polarity_grad_gain:g}, "
            f"mix_prev={sim.polarity_mix_prev:g}"
        )
    if sim._force_division_dir_np is not None:
        fdx, fdy, fdz = sim._force_division_dir_np.tolist()
        print(f"Division direction override: [{fdx:.4f}, {fdy:.4f}, {fdz:.4f}]")
    print(f"Relaxer projection mode: {sim.relax_projection_mode}")
    print("Note: PyVista rendering/movie encoding is separate from simulation kernels.")
    color_by = "none" if args.color_by == "solid" else args.color_by
    movie_color_by = color_by
    if movie_color_by not in {"order", "radius", "none", "u", "v", "prog", "age", "pz"}:
        movie_color_by = "order"
        if args.save_movie:
            print("Unknown movie color-by mode; using order for movie frames.")
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
            movie_duration_seconds=args.movie_duration_seconds,
            show_centers=args.show_centers,
            color_by=movie_color_by,
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

    if args.save_data:
        sim.save_data_npz(args.data_path)
    if args.save_snapshot:
        sim.save_snapshot_pyvista(
            args.snapshot_path,
            show_centers=args.show_centers,
            color_by=color_by,
            max_render_cells=view_max_render_cells,
        )

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
