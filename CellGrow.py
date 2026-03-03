"""
Program-driven 3D point-growth simulation.

This branch is a minimum working example for inherited per-cell programs:
- each cell carries the same instruction string,
- at each timestep the current instruction decides stay/divide/die,
- when dividing, the parent stays in place and one daughter is displaced,
- daughter cells can either restart the program or inherit the parent's next step.

The simulation remains continuous in 3D space. PyVista visualization, MP4 movie
export, and NPZ state export are preserved.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cupy as cp

    _GPU_ENABLED = True
except Exception:  # pragma: no cover - fallback only when CuPy is unavailable
    import numpy as cp  # type: ignore

    _GPU_ENABLED = False


# Main simulation controls
DEFAULT_STEPS = 15  # Number of simulation timesteps.
DEFAULT_PROGRAM = "D[0,1,0.5];S;D[1,0,0.2];S"  # Age-indexed instruction string.
DEFAULT_PROGRAM_LOOP = False  # Loop the instruction string after the last step.
DEFAULT_DAUGHTER_PROGRAM_MODE = "restart"  # Daughter program mode; options: restart, inherit.
DEFAULT_RADIUS = 1.0  # Cell radius used for PyVista sphere glyphs.
DEFAULT_SPLIT_DISTANCE = 1.5  # Default daughter displacement distance for divide instructions.
DEFAULT_MAX_CELLS = 250_000  # Safety cap against uncontrolled growth.
DEFAULT_SHOW = True  # Show interactive PyVista window at the end.
DEFAULT_COLOR_BY = "order"  # Static color mode; options: order, radius, solid, age, program_step, prog.
DEFAULT_OUTPUT_STEM = "cell_program"  # Base filename stem for default outputs.
DEFAULT_SAVE_DATA = False  # Save final state to NPZ by default.
DEFAULT_DATA_PATH = f"{DEFAULT_OUTPUT_STEM}.npz"  # Default NPZ output path.
DEFAULT_SAVE_SNAPSHOT = False  # Save final snapshot image by default.
DEFAULT_SNAPSHOT_PATH = f"{DEFAULT_OUTPUT_STEM}.png"  # Default snapshot output path.
DEFAULT_SAVE_MOVIE = False  # Render and save movie by default.
DEFAULT_MOVIE_PATH = f"{DEFAULT_OUTPUT_STEM}.mp4"  # Default MP4 output path.
DEFAULT_MOVIE_FPS = 24  # Default movie FPS when not auto-scaled.
DEFAULT_MOVIE_DURATION_SECONDS: Optional[float] = 10.0  # Target movie duration in seconds; None disables auto-FPS.
DEFAULT_INTERP_FRAMES = 5  # Interpolated frames between simulation steps in movie mode.
DEFAULT_MOVIE_WIDTH = 1024  # Movie frame width in pixels.
DEFAULT_MOVIE_HEIGHT = 860  # Movie frame height in pixels.
DEFAULT_MOVIE_SPHERE_THETA = 16  # Sphere theta tessellation in movie rendering.
DEFAULT_MOVIE_SPHERE_PHI = 16  # Sphere phi tessellation in movie rendering.
DEFAULT_MOVIE_SHOW_EDGES = False  # Render sphere mesh edges in movies.
DEFAULT_MOVIE_EDGE_COLOR = "#000000"  # Edge color when movie edges are enabled.
DEFAULT_MOVIE_EDGE_WIDTH = 0.6  # Edge line width when movie edges are enabled.
DEFAULT_MOVIE_MACRO_BLOCK_SIZE = 16  # MP4 encoder macro block size.
DEFAULT_MOVIE_DEATH_ANIMATION = "shrink"  # Death animation; options: none, fade, shrink, fade_shrink.
DEFAULT_MOVIE_ADAPTIVE_LARGE = True  # Enable adaptive movie settings for large colonies.
DEFAULT_MOVIE_LARGE_CELLS_THRESHOLD = 8000  # Cell-count threshold for adaptive movie behavior.
DEFAULT_MOVIE_LARGE_INTERP_FRAMES = 2  # Interp frame count used after adaptive threshold.
DEFAULT_MOVIE_MAX_RENDER_CELLS = 50000  # Max cells rendered per movie frame; 0 disables sampling.
DEFAULT_VIEW_MAX_RENDER_CELLS = 50000  # Max cells rendered in static final view; 0 disables sampling.
DEFAULT_MOVIE_OPACITY = 1.0  # Sphere opacity in movies.
DEFAULT_STATIC_OPACITY = 1.0  # Sphere opacity in the static view.
DEFAULT_NOTEBOOK = False  # Pass notebook=False to PyVista plotter.
DEFAULT_DTYPE = "float32"  # Numeric dtype for simulation arrays.

_ACTION_STAY = 0
_ACTION_DIVIDE = 1
_ACTION_DIE = 2

_AXIS_ALIASES = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "+x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "+y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "-y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "+z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "-z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
}


@dataclass(frozen=True)
class CompiledProgram:
    """Compiled representation of a semicolon-delimited instruction string."""

    actions: np.ndarray
    global_directions: np.ndarray
    direction_exprs: tuple[tuple["ScalarExpression", "ScalarExpression", "ScalarExpression"], ...]
    distance_exprs: tuple["ScalarExpression", ...]
    raw_tokens: tuple[str, ...]


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


@dataclass(frozen=True)
class ScalarExpression:
    """Restricted arithmetic expression over the variables `age` and `step`."""

    source: str
    ast_body: ast.AST
    constant_value: Optional[float] = None



def backend_summary() -> str:
    """Return a human-readable summary of the active compute backend."""
    if not _GPU_ENABLED:
        return "Backend: NumPy CPU fallback (CuPy import failed)."
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode("utf-8")
        return f"Backend: CuPy on GPU device {dev.id} ({name})"
    except Exception:
        return "Backend: CuPy enabled (GPU details unavailable)."



def _normalize_numpy_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("division direction must be non-zero")
    return vec / norm


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split text by a separator while respecting parentheses/brackets."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced brackets in expression '{text}'")
        elif ch == sep and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if depth != 0:
        raise ValueError(f"unbalanced brackets in expression '{text}'")
    parts.append("".join(buf).strip())
    return parts


def _split_top_level_once(text: str, sep: str) -> tuple[str, Optional[str]]:
    parts = _split_top_level(text, sep)
    if len(parts) == 1:
        return parts[0], None
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"too many top-level '{sep}' separators in '{text}'")


def _compile_scalar_expression(expr_text: str) -> ScalarExpression:
    """Compile a safe arithmetic expression involving only numbers, `age`, and `step`."""
    source = expr_text.strip()
    if not source:
        raise ValueError("empty numeric expression")
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid numeric expression '{source}'") from exc

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )
    for node in ast.walk(parsed):
        if not isinstance(node, allowed_nodes):
            raise ValueError(
                f"unsupported syntax in expression '{source}'. Only +, -, *, /, **, parentheses, age, and step are allowed."
            )
        if isinstance(node, ast.Name) and node.id not in {"age", "step"}:
            raise ValueError(
                f"unsupported variable '{node.id}' in expression '{source}'. Only 'age' and 'step' are allowed."
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant in expression '{source}'")

    if any(isinstance(node, ast.Name) and node.id in {"age", "step"} for node in ast.walk(parsed)):
        constant_value: Optional[float] = None
    else:
        constant_value = float(
            eval(compile(parsed, "<cell-program-expr>", "eval"), {"__builtins__": {}}, {})
        )
    return ScalarExpression(source=source, ast_body=parsed.body, constant_value=constant_value)


def _evaluate_scalar_ast(node: ast.AST, age_values, step_values):
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "age":
            return age_values
        if node.id == "step":
            return step_values
        if node.id not in {"age", "step"}:
            raise ValueError(f"unsupported variable '{node.id}'")
    if isinstance(node, ast.UnaryOp):
        val = _evaluate_scalar_ast(node.operand, age_values, step_values)
        if isinstance(node.op, ast.UAdd):
            return val
        if isinstance(node.op, ast.USub):
            return -val
    if isinstance(node, ast.BinOp):
        left = _evaluate_scalar_ast(node.left, age_values, step_values)
        right = _evaluate_scalar_ast(node.right, age_values, step_values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    raise ValueError("unsupported expression node")


def _evaluate_scalar_expression(expr: ScalarExpression, age_values, step_values, xp, dtype):
    if expr.constant_value is not None:
        return xp.full(age_values.shape, expr.constant_value, dtype=dtype)
    result = _evaluate_scalar_ast(expr.ast_body, age_values, step_values)
    return xp.asarray(result, dtype=dtype)


def _parse_direction(text: str) -> np.ndarray:
    raw = text.strip().lower()
    if raw in _AXIS_ALIASES:
        return _AXIS_ALIASES[raw].copy()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(
            "divide direction must be an axis alias (x,-x,y,-y,z,-z) or a 3-vector like 1,0,0"
        )
    vec = np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
    return _normalize_numpy_vector(vec)


def _parse_direction_expressions(
    text: str,
) -> tuple[ScalarExpression, ScalarExpression, ScalarExpression]:
    raw = text.strip()
    lower = raw.lower()
    if lower in _AXIS_ALIASES:
        alias = _AXIS_ALIASES[lower]
        return (
            _compile_scalar_expression(f"{float(alias[0]):g}"),
            _compile_scalar_expression(f"{float(alias[1]):g}"),
            _compile_scalar_expression(f"{float(alias[2]):g}"),
        )
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
    parts = _split_top_level(raw, ",")
    if len(parts) != 3:
        raise ValueError(
            "divide direction must be an axis alias or a 3-vector expression like 1,1,0.33+(age*0.01)+(step*0.01)"
        )
    return (
        _compile_scalar_expression(parts[0]),
        _compile_scalar_expression(parts[1]),
        _compile_scalar_expression(parts[2]),
    )


def _expand_program_tokens(program_text: str) -> list[str]:
    tokens: list[str] = []
    for part in program_text.replace("\n", ";").split(";"):
        token = part.strip()
        if not token:
            continue
        body = token
        repeat = 1
        if "*" in token:
            maybe_body, maybe_repeat = token.rsplit("*", 1)
            maybe_repeat = maybe_repeat.strip()
            if maybe_repeat.isdigit():
                body = maybe_body.strip()
                repeat = int(maybe_repeat)
        if repeat < 1:
            raise ValueError(f"invalid repeat count in token '{token}'")
        tokens.extend([body] * repeat)
    return tokens



def compile_program(program_text: str, default_split_distance: float) -> CompiledProgram:
    """
    Compile a semicolon-delimited program string.

    Supported instruction forms:
    - `S` or `stay`
    - `X` or `die`
    - `D[x]`
    - `D[x@2.0]`
    - `D[1,0.1,0.33+(age*0.01)]`
    - `D[1,0.1,0.33+(step*0.01)@1.5+age*0.05]`
    - `Dg[y]`
    - `Dg[0,0,1@2.0]`
    - `Dabs[-1,0,0]`
    - any instruction can be repeated with `*N`, e.g. `S*3`

    Semantics:
    - `D[...]` is interpreted in the cell's local moving frame.
      The local x-axis is the cell's previous division axis.
    - `Dg[...]` / `Dabs[...]` are interpreted in the fixed global Cartesian frame.
    - Numeric slots may use the variables `age` and `step` with +, -, *, /, **, and parentheses.
    """
    if default_split_distance <= 0.0:
        raise ValueError("default_split_distance must be > 0")

    raw_tokens = _expand_program_tokens(program_text)
    if not raw_tokens:
        raise ValueError("program must contain at least one instruction")

    actions = np.empty((len(raw_tokens),), dtype=np.int8)
    global_directions = np.zeros((len(raw_tokens),), dtype=np.int8)
    direction_exprs: list[tuple[ScalarExpression, ScalarExpression, ScalarExpression]] = []
    distance_exprs: list[ScalarExpression] = []

    for idx, token in enumerate(raw_tokens):
        upper = token.strip().upper()
        if upper in {"S", "STAY"}:
            actions[idx] = _ACTION_STAY
            direction_exprs.append(
                (
                    _compile_scalar_expression("1"),
                    _compile_scalar_expression("0"),
                    _compile_scalar_expression("0"),
                )
            )
            distance_exprs.append(_compile_scalar_expression(f"{float(default_split_distance):g}"))
            continue
        if upper in {"X", "DIE", "KILL"}:
            actions[idx] = _ACTION_DIE
            direction_exprs.append(
                (
                    _compile_scalar_expression("1"),
                    _compile_scalar_expression("0"),
                    _compile_scalar_expression("0"),
                )
            )
            distance_exprs.append(_compile_scalar_expression(f"{float(default_split_distance):g}"))
            continue
        global_mode = False
        if upper.startswith("DABS"):
            global_mode = True
            payload = token.strip()[4:].strip()
        elif upper.startswith("DG"):
            global_mode = True
            payload = token.strip()[2:].strip()
        elif upper.startswith("D"):
            payload = token.strip()[1:].strip()
        else:
            raise ValueError(
                f"unsupported token '{token}'. Use S, X, D[local_dir@distance], or Dg[global_dir@distance]."
            )
        if not payload:
            raise ValueError(
                f"divide token '{token}' is missing a direction. Example: D[1,0.1,0] or Dg[x@1.5]"
            )
        if payload[0] in "[(" and payload[-1] in "])":
            payload = payload[1:-1].strip()

        direction_part, distance_part = _split_top_level_once(payload, "@")
        if distance_part is None:
            distance_expr = _compile_scalar_expression(f"{float(default_split_distance):g}")
        else:
            distance_expr = _compile_scalar_expression(distance_part.strip())

        actions[idx] = _ACTION_DIVIDE
        direction_exprs.append(_parse_direction_expressions(direction_part))
        global_directions[idx] = 1 if global_mode else 0
        distance_exprs.append(distance_expr)

    return CompiledProgram(
        actions=actions,
        global_directions=global_directions,
        direction_exprs=tuple(direction_exprs),
        distance_exprs=tuple(distance_exprs),
        raw_tokens=tuple(raw_tokens),
    )



def show_cells_pyvista(
    points,
    *,
    cell_radius: float = DEFAULT_RADIUS,
    sphere_theta: int = 16,
    sphere_phi: int = 16,
    opacity: float = DEFAULT_STATIC_OPACITY,
    show_centers: bool = False,
    centers_size: float = 6.0,
    color_by: str = "order",
    cmap: str = "viridis",
    scalar_values: Optional[np.ndarray] = None,
    notebook: bool = DEFAULT_NOTEBOOK,
    max_render_cells: Optional[int] = DEFAULT_VIEW_MAX_RENDER_CELLS,
    snapshot_path: Optional[str] = None,
) -> Optional[Path]:
    """Visualize cells as sphere glyphs in PyVista."""
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

    cloud = pv.PolyData(pts)
    cloud["scale"] = np.full(pts.shape[0], float(cell_radius), dtype=float)

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
        spheres["val"] = np.repeat(np.full(pts.shape[0], float(cell_radius)), base_sphere.n_points)
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


@dataclass
class CellGrowth3D:
    """Program-driven 3D growth simulation for point-based cells."""

    radius: float = DEFAULT_RADIUS
    split_distance: float = DEFAULT_SPLIT_DISTANCE
    program_text: str = DEFAULT_PROGRAM
    daughter_program_mode: str = DEFAULT_DAUGHTER_PROGRAM_MODE
    program_loop: bool = DEFAULT_PROGRAM_LOOP
    seed: int = 42
    max_cells: Optional[int] = DEFAULT_MAX_CELLS
    dtype: str = DEFAULT_DTYPE

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")
        if self.split_distance <= 0.0:
            raise ValueError("split_distance must be > 0")
        if self.max_cells is not None and self.max_cells < 1:
            raise ValueError("max_cells must be >= 1 when provided")
        if self.daughter_program_mode not in {"restart", "inherit"}:
            raise ValueError("daughter_program_mode must be 'restart' or 'inherit'")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")

        self.xp = cp
        self.float_dtype = getattr(cp, self.dtype)
        self.program = compile_program(self.program_text, self.split_distance)
        self._program_actions = cp.asarray(self.program.actions, dtype=cp.int8)
        self._program_global_directions = cp.asarray(
            self.program.global_directions, dtype=cp.int8
        )

        self.points = cp.zeros((1, 3), dtype=self.float_dtype)
        self.cell_ids = cp.asarray([0], dtype=cp.int64)
        self.birth_age = cp.asarray([0], dtype=cp.int32)
        self.program_counter = cp.asarray([0], dtype=cp.int32)
        self.local_frame = cp.asarray(np.eye(3, dtype=np.float32)[None, :, :], dtype=self.float_dtype)
        self._next_cell_id = 1
        self.step_index = 0
        self.count_history = [1]
        self.last_step_summary: Optional[dict[str, int]] = None

    def _as_numpy(self, arr) -> np.ndarray:
        if _GPU_ENABLED:
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def _scalar_int(self, value) -> int:
        if _GPU_ENABLED:
            return int(cp.asnumpy(value))
        return int(value)

    def points_numpy(self) -> np.ndarray:
        return self._as_numpy(self.points)

    def _current_instruction_indices(self, counters: cp.ndarray) -> cp.ndarray:
        n_instr = int(self._program_actions.shape[0])
        if self.program_loop:
            return counters % max(1, n_instr)
        return cp.minimum(counters, n_instr - 1)

    def _current_program_state(
        self,
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        idx = self._current_instruction_indices(self.program_counter)
        return (
            idx,
            self._program_actions[idx],
            self._program_global_directions[idx],
        )

    def _evaluate_division_payload(
        self,
        instruction_indices: cp.ndarray,
        ages: cp.ndarray,
        step_index: int,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Evaluate age-dependent direction and distance expressions for active divide instructions."""
        n = int(instruction_indices.shape[0])
        directions = cp.zeros((n, 3), dtype=self.float_dtype)
        distances = cp.zeros((n,), dtype=self.float_dtype)
        if n == 0:
            return directions, distances

        ages_f = ages.astype(self.float_dtype, copy=False)
        step_values = cp.full(ages_f.shape, float(step_index), dtype=self.float_dtype)
        unique_idx = np.unique(self._as_numpy(instruction_indices).astype(np.int64, copy=False))
        eps = float(1e-12)

        for raw_idx in unique_idx.tolist():
            instr_idx = int(raw_idx)
            mask = instruction_indices == instr_idx
            age_subset = ages_f[mask]
            step_subset = step_values[mask]
            expr_x, expr_y, expr_z = self.program.direction_exprs[instr_idx]
            dx = _evaluate_scalar_expression(expr_x, age_subset, step_subset, cp, self.float_dtype)
            dy = _evaluate_scalar_expression(expr_y, age_subset, step_subset, cp, self.float_dtype)
            dz = _evaluate_scalar_expression(expr_z, age_subset, step_subset, cp, self.float_dtype)
            vec = cp.stack([dx, dy, dz], axis=1)
            norms = cp.linalg.norm(vec, axis=1)
            if self._scalar_int(cp.count_nonzero(norms <= eps)) > 0:
                token = self.program.raw_tokens[instr_idx]
                raise ValueError(
                    f"direction expression in token '{token}' evaluates to zero for some cells"
                )
            directions[mask] = vec / cp.maximum(norms[:, None], cp.asarray(eps, dtype=self.float_dtype))

            dist = _evaluate_scalar_expression(
                self.program.distance_exprs[instr_idx], age_subset, step_subset, cp, self.float_dtype
            )
            if self._scalar_int(cp.count_nonzero(dist <= 0.0)) > 0:
                token = self.program.raw_tokens[instr_idx]
                raise ValueError(
                    f"distance expression in token '{token}' must evaluate to > 0"
                )
            distances[mask] = dist

        return directions, distances

    def _frames_from_new_x(self, previous_frames: cp.ndarray, new_x: cp.ndarray) -> cp.ndarray:
        """Construct updated local frames with x aligned to the new division direction."""
        if int(new_x.shape[0]) == 0:
            return cp.zeros((0, 3, 3), dtype=self.float_dtype)

        eps = cp.asarray(1e-12, dtype=self.float_dtype)
        new_x = new_x / cp.maximum(cp.linalg.norm(new_x, axis=1, keepdims=True), eps)

        old_y = previous_frames[:, :, 1]
        old_z = previous_frames[:, :, 2]

        new_y = old_y - cp.sum(old_y * new_x, axis=1, keepdims=True) * new_x
        y_norm = cp.linalg.norm(new_y, axis=1, keepdims=True)
        bad_y = (y_norm[:, 0] <= eps)
        if self._scalar_int(cp.count_nonzero(bad_y)) > 0:
            fallback = old_z[bad_y] - cp.sum(old_z[bad_y] * new_x[bad_y], axis=1, keepdims=True) * new_x[bad_y]
            new_y[bad_y] = fallback
            y_norm = cp.linalg.norm(new_y, axis=1, keepdims=True)

        bad_y = (y_norm[:, 0] <= eps)
        if self._scalar_int(cp.count_nonzero(bad_y)) > 0:
            global_z = cp.asarray([0.0, 0.0, 1.0], dtype=self.float_dtype)
            fallback = global_z[None, :] - cp.sum(global_z[None, :] * new_x[bad_y], axis=1, keepdims=True) * new_x[bad_y]
            new_y[bad_y] = fallback
            y_norm = cp.linalg.norm(new_y, axis=1, keepdims=True)

        bad_y = (y_norm[:, 0] <= eps)
        if self._scalar_int(cp.count_nonzero(bad_y)) > 0:
            global_y = cp.asarray([0.0, 1.0, 0.0], dtype=self.float_dtype)
            fallback = global_y[None, :] - cp.sum(global_y[None, :] * new_x[bad_y], axis=1, keepdims=True) * new_x[bad_y]
            new_y[bad_y] = fallback
            y_norm = cp.linalg.norm(new_y, axis=1, keepdims=True)

        new_y = new_y / cp.maximum(y_norm, eps)
        new_z = cp.cross(new_x, new_y)
        new_z = new_z / cp.maximum(cp.linalg.norm(new_z, axis=1, keepdims=True), eps)
        new_y = cp.cross(new_z, new_x)
        new_y = new_y / cp.maximum(cp.linalg.norm(new_y, axis=1, keepdims=True), eps)
        return cp.stack([new_x, new_y, new_z], axis=2)

    def _resolve_division_directions(
        self,
        instruction_directions: cp.ndarray,
        global_flags: cp.ndarray,
        local_frames: cp.ndarray,
    ) -> cp.ndarray:
        """Resolve local/global instruction vectors into normalized world-space directions."""
        resolved = cp.einsum("nij,nj->ni", local_frames, instruction_directions)
        if int(resolved.shape[0]) == 0:
            return resolved

        global_mask = global_flags.astype(cp.bool_)
        if self._scalar_int(cp.count_nonzero(global_mask)) > 0:
            resolved[global_mask] = instruction_directions[global_mask]
        eps = cp.asarray(1e-12, dtype=self.float_dtype)
        return resolved / cp.maximum(cp.linalg.norm(resolved, axis=1, keepdims=True), eps)

    def _clip_divides_to_max_cells(self, divide_mask: cp.ndarray, die_mask: cp.ndarray) -> cp.ndarray:
        if self.max_cells is None:
            return divide_mask

        n_cells = int(self.points.shape[0])
        n_die = self._scalar_int(cp.count_nonzero(die_mask))
        n_divide = self._scalar_int(cp.count_nonzero(divide_mask))
        max_new_cells = int(self.max_cells) - (n_cells - n_die)
        if n_divide <= max_new_cells:
            return divide_mask
        if max_new_cells <= 0:
            return cp.zeros_like(divide_mask, dtype=cp.bool_)

        divide_idx = cp.flatnonzero(divide_mask)
        keep_idx = divide_idx[:max_new_cells]
        clipped = cp.zeros_like(divide_mask, dtype=cp.bool_)
        clipped[keep_idx] = True
        return clipped

    def _death_targets(self, n_dead: int, death_animation: str) -> tuple[cp.ndarray, cp.ndarray]:
        ones = cp.ones((n_dead,), dtype=self.float_dtype)
        zeros = cp.zeros((n_dead,), dtype=self.float_dtype)
        if death_animation == "fade":
            return ones, zeros
        if death_animation == "shrink":
            return zeros, ones
        if death_animation == "fade_shrink":
            return zeros, zeros
        return zeros, zeros

    def _step_internal(self, death_animation: str = DEFAULT_MOVIE_DEATH_ANIMATION) -> StepTransition:
        if self.points.shape[0] == 0:
            empty_pts = cp.zeros((0, 3), dtype=self.float_dtype)
            empty_i64 = cp.zeros((0,), dtype=cp.int64)
            empty_f = cp.zeros((0,), dtype=self.float_dtype)
            return StepTransition(
                source_points=empty_pts,
                target_points=empty_pts,
                source_ids=empty_i64,
                target_ids=empty_i64,
                source_size=empty_f,
                target_size=empty_f,
                source_alpha=empty_f,
                target_alpha=empty_f,
            )

        source_points_all = self.points.copy()
        source_ids_all = self.cell_ids.copy()
        source_age_all = self.birth_age.copy()
        source_pc_all = self.program_counter.copy()
        source_frame_all = self.local_frame.copy()

        instr_idx, actions, global_flags = self._current_program_state()
        divide_mask = actions == _ACTION_DIVIDE
        die_mask = actions == _ACTION_DIE
        divide_mask = self._clip_divides_to_max_cells(divide_mask, die_mask)
        stay_mask = ~(divide_mask | die_mask)
        survive_mask = ~die_mask

        parent_points = source_points_all[survive_mask]
        parent_ids = source_ids_all[survive_mask]
        parent_age = source_age_all[survive_mask] + 1
        parent_pc = source_pc_all[survive_mask] + 1
        parent_frame = source_frame_all[survive_mask].copy()

        divide_points = source_points_all[divide_mask]
        divide_instr_idx = instr_idx[divide_mask]
        divide_instruction_dirs, divide_dist = self._evaluate_division_payload(
            divide_instr_idx,
            source_age_all[divide_mask],
            self.step_index,
        )
        divide_dirs = self._resolve_division_directions(
            divide_instruction_dirs,
            global_flags[divide_mask],
            source_frame_all[divide_mask],
        )
        n_divide = int(divide_points.shape[0])
        daughter_points = divide_points + divide_dirs * divide_dist[:, None] if n_divide else cp.zeros((0, 3), dtype=self.float_dtype)
        daughter_ids = cp.arange(self._next_cell_id, self._next_cell_id + n_divide, dtype=cp.int64)
        self._next_cell_id += n_divide
        daughter_age = cp.zeros((n_divide,), dtype=cp.int32)
        daughter_frame = self._frames_from_new_x(source_frame_all[divide_mask], divide_dirs)
        if self.daughter_program_mode == "restart":
            daughter_pc = cp.zeros((n_divide,), dtype=cp.int32)
        else:
            daughter_pc = source_pc_all[divide_mask] + 1
        if n_divide > 0:
            parent_frame[divide_mask[survive_mask]] = daughter_frame

        if n_divide > 0:
            next_points = cp.concatenate([parent_points, daughter_points], axis=0)
            next_ids = cp.concatenate([parent_ids, daughter_ids], axis=0)
            next_age = cp.concatenate([parent_age, daughter_age], axis=0)
            next_pc = cp.concatenate([parent_pc, daughter_pc], axis=0)
            next_frame = cp.concatenate([parent_frame, daughter_frame], axis=0)
        else:
            next_points = parent_points
            next_ids = parent_ids
            next_age = parent_age
            next_pc = parent_pc
            next_frame = parent_frame

        self.points = next_points
        self.cell_ids = next_ids
        self.birth_age = next_age
        self.program_counter = next_pc
        self.local_frame = next_frame
        self.step_index += 1
        self.count_history.append(int(self.points.shape[0]))

        n_stay = self._scalar_int(cp.count_nonzero(stay_mask))
        n_die = self._scalar_int(cp.count_nonzero(die_mask))
        self.last_step_summary = {
            "step": int(self.step_index),
            "cells": int(self.points.shape[0]),
            "divide": int(n_divide),
            "stay": int(n_stay),
            "die": int(n_die),
        }

        keep_source_points = source_points_all[survive_mask]
        keep_source_ids = source_ids_all[survive_mask]
        keep_target_points = parent_points
        keep_target_ids = parent_ids
        keep_count = int(keep_source_points.shape[0])
        keep_size = cp.ones((keep_count,), dtype=self.float_dtype)
        keep_alpha = cp.ones((keep_count,), dtype=self.float_dtype)

        daughter_source_points = divide_points
        daughter_target_points = daughter_points
        daughter_source_ids = daughter_ids
        daughter_target_ids = daughter_ids
        daughter_size = cp.ones((n_divide,), dtype=self.float_dtype)
        daughter_alpha = cp.ones((n_divide,), dtype=self.float_dtype)

        dead_source_points = source_points_all[die_mask]
        dead_target_points = dead_source_points.copy()
        dead_ids = source_ids_all[die_mask]
        n_dead = int(dead_source_points.shape[0])
        dead_source_size = cp.ones((n_dead,), dtype=self.float_dtype)
        dead_source_alpha = cp.ones((n_dead,), dtype=self.float_dtype)
        dead_target_size, dead_target_alpha = self._death_targets(n_dead, death_animation)

        source_points = cp.concatenate(
            [keep_source_points, daughter_source_points, dead_source_points], axis=0
        )
        target_points = cp.concatenate(
            [keep_target_points, daughter_target_points, dead_target_points], axis=0
        )
        source_ids = cp.concatenate([keep_source_ids, daughter_source_ids, dead_ids], axis=0)
        target_ids = cp.concatenate([keep_target_ids, daughter_target_ids, dead_ids], axis=0)
        source_size = cp.concatenate([keep_size, daughter_size, dead_source_size], axis=0)
        target_size = cp.concatenate([keep_size, daughter_size, dead_target_size], axis=0)
        source_alpha = cp.concatenate([keep_alpha, daughter_alpha, dead_source_alpha], axis=0)
        target_alpha = cp.concatenate([keep_alpha, daughter_alpha, dead_target_alpha], axis=0)

        return StepTransition(
            source_points=source_points,
            target_points=target_points,
            source_ids=source_ids,
            target_ids=target_ids,
            source_size=source_size,
            target_size=target_size,
            source_alpha=source_alpha,
            target_alpha=target_alpha,
        )

    def step_with_transition(
        self,
        death_animation: str = DEFAULT_MOVIE_DEATH_ANIMATION,
    ) -> StepTransition:
        return self._step_internal(death_animation=death_animation)

    def step(self) -> cp.ndarray:
        self._step_internal(death_animation=DEFAULT_MOVIE_DEATH_ANIMATION)
        return self.points

    def run(self, n_steps: int, *, log_counts: bool = False) -> cp.ndarray:
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        for _ in range(n_steps):
            if self.points.shape[0] == 0:
                break
            self.step()
            if log_counts and self.last_step_summary is not None:
                summary = self.last_step_summary
                print(
                    f"Step {summary['step']}: {summary['cells']} cells "
                    f"| divide={summary['divide']} stay={summary['stay']} die={summary['die']}"
                )
        return self.points

    def _current_scalar_state(self, color_by: str) -> np.ndarray:
        if color_by == "age":
            return self._as_numpy(self.birth_age).astype(float, copy=False)
        if color_by in {"program_step", "prog"}:
            return self._as_numpy(self.program_counter).astype(float, copy=False)
        return np.empty((0,), dtype=float)

    def run_and_save_movie(
        self,
        n_steps: int,
        output_path: str,
        *,
        log_counts: bool = False,
        adaptive_large_render: bool = DEFAULT_MOVIE_ADAPTIVE_LARGE,
        large_cells_threshold: int = DEFAULT_MOVIE_LARGE_CELLS_THRESHOLD,
        large_interp_frames: int = DEFAULT_MOVIE_LARGE_INTERP_FRAMES,
        max_render_cells: Optional[int] = DEFAULT_MOVIE_MAX_RENDER_CELLS,
        interp_frames: int = DEFAULT_INTERP_FRAMES,
        fps: int = DEFAULT_MOVIE_FPS,
        movie_duration_seconds: Optional[float] = DEFAULT_MOVIE_DURATION_SECONDS,
        show_centers: bool = False,
        color_by: str = DEFAULT_COLOR_BY,
        cmap: str = "viridis",
        opacity: float = DEFAULT_MOVIE_OPACITY,
        sphere_theta: int = DEFAULT_MOVIE_SPHERE_THETA,
        sphere_phi: int = DEFAULT_MOVIE_SPHERE_PHI,
        show_edges: bool = DEFAULT_MOVIE_SHOW_EDGES,
        edge_color: str = DEFAULT_MOVIE_EDGE_COLOR,
        edge_width: float = DEFAULT_MOVIE_EDGE_WIDTH,
        window_size: tuple[int, int] = (DEFAULT_MOVIE_WIDTH, DEFAULT_MOVIE_HEIGHT),
        macro_block_size: int = DEFAULT_MOVIE_MACRO_BLOCK_SIZE,
        death_animation: str = DEFAULT_MOVIE_DEATH_ANIMATION,
    ) -> cp.ndarray:
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
        valid_color_modes = {"order", "radius", "none", "age", "program_step", "prog"}
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

        scalar_mode = color_by in {"age", "program_step", "prog"}
        initial_points_np = self.points_numpy().copy()
        initial_ids_np = self._as_numpy(self.cell_ids.copy()).astype(np.int64, copy=False)
        initial_scalar_state_np = self._current_scalar_state(color_by) if scalar_mode else None
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
        scalar_transitions_np: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        for _ in range(n_steps):
            if self.points.shape[0] == 0:
                break
            if scalar_mode:
                src_state_ids_np = self._as_numpy(self.cell_ids.copy()).astype(np.int64, copy=False)
                src_scalar_state_np = self._current_scalar_state(color_by)
            else:
                src_state_ids_np = np.empty((0,), dtype=np.int64)
                src_scalar_state_np = np.empty((0,), dtype=float)

            transition = self.step_with_transition(death_animation=death_animation)
            if scalar_mode:
                tgt_state_ids_np = self._as_numpy(self.cell_ids.copy()).astype(np.int64, copy=False)
                tgt_scalar_state_np = self._current_scalar_state(color_by)
                if src_scalar_state_np.size > 0:
                    scalar_range_min = min(scalar_range_min, float(src_scalar_state_np.min()))
                    scalar_range_max = max(scalar_range_max, float(src_scalar_state_np.max()))
                if tgt_scalar_state_np.size > 0:
                    scalar_range_min = min(scalar_range_min, float(tgt_scalar_state_np.min()))
                    scalar_range_max = max(scalar_range_max, float(tgt_scalar_state_np.max()))
                scalar_transitions_np.append(
                    (src_state_ids_np, src_scalar_state_np, tgt_state_ids_np, tgt_scalar_state_np)
                )
            if log_counts and self.last_step_summary is not None:
                summary = self.last_step_summary
                print(
                    f"Step {summary['step']}: {summary['cells']} cells "
                    f"| divide={summary['divide']} stay={summary['stay']} die={summary['die']}"
                )
            transitions_np.append(
                (
                    self._as_numpy(transition.source_points),
                    self._as_numpy(transition.target_points),
                    self._as_numpy(transition.source_ids),
                    self._as_numpy(transition.target_ids),
                    self._as_numpy(transition.source_size),
                    self._as_numpy(transition.target_size),
                    self._as_numpy(transition.source_alpha),
                    self._as_numpy(transition.target_alpha),
                )
            )

        final_points_np = self.points_numpy().copy()
        movie_max_order = max(1, int(self._next_cell_id - 1))
        movie_scalar_clim: Optional[tuple[float, float]] = None
        if scalar_mode and np.isfinite(scalar_range_min) and np.isfinite(scalar_range_max):
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
            planned_frame_count += 1 if src.shape[0] == 0 else int(effective_interp_frames)

        effective_fps = float(fps)
        if movie_duration_seconds is not None:
            target_seconds = float(movie_duration_seconds)
            auto_fps = float(planned_frame_count) / target_seconds
            effective_fps = max(1.0, auto_fps)
            expected_duration = float(planned_frame_count) / effective_fps
            print(
                "Movie timing: "
                f"target={target_seconds:.2f}s, frames={planned_frame_count}, "
                f"fps={effective_fps:.2f}, expected={expected_duration:.2f}s."
            )

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

        fit_radius *= 1.2
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
            out_vals = np.zeros(q.shape[0], dtype=float)
            valid = np.zeros(q.shape[0], dtype=bool)
            if q.size == 0 or state_ids.size == 0 or state_scalars.size == 0:
                return out_vals, valid
            sid = np.asarray(state_ids, dtype=np.int64).reshape(-1)
            sval = np.asarray(state_scalars, dtype=float).reshape(-1)
            order = np.argsort(sid, kind="mergesort")
            sid_sorted = sid[order]
            sval_sorted = sval[order]
            pos = np.searchsorted(sid_sorted, q)
            if sid_sorted.size == 0:
                return out_vals, valid
            in_bounds = pos < sid_sorted.size
            pos_safe = np.minimum(pos, sid_sorted.size - 1)
            matched = in_bounds & (sid_sorted[pos_safe] == q)
            if np.any(matched):
                out_vals[matched] = sval_sorted[pos_safe[matched]]
                valid[matched] = True
            return out_vals, valid

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
            if pts.shape[0] != order.shape[0] or pts.shape[0] != size.shape[0] or pts.shape[0] != alpha.shape[0]:
                raise ValueError("frame arrays must all have matching lengths")
            if scalar is not None and pts.shape[0] != scalar.shape[0]:
                raise ValueError("scalar array length must match frame point count")

            if effective_max_render_cells is not None and pts.shape[0] > int(effective_max_render_cells):
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
                    mesh_kwargs = dict(scalars="val", cmap=cmap)
                elif color_by == "order":
                    spheres["val"] = np.repeat(order, base_sphere.n_points)
                    mesh_kwargs = dict(scalars="val", cmap=cmap, clim=(0.0, float(movie_max_order)))
                elif color_by in {"age", "program_step", "prog"}:
                    if scalar is None:
                        raise ValueError(f"color_by='{color_by}' requires scalar frame values")
                    spheres["val"] = np.repeat(scalar, base_sphere.n_points)
                    mesh_kwargs = dict(scalars="val", cmap=cmap)
                    if movie_scalar_clim is not None:
                        mesh_kwargs["clim"] = movie_scalar_clim
                else:
                    mesh_kwargs = dict(color="lightsteelblue")

                pl.add_mesh(
                    spheres,
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
                    **mesh_kwargs,
                )

                if show_centers:
                    pl.add_points(cloud, color="black", point_size=6.0, render_points_as_spheres=True)

            pl.add_axes()
            pl.show_grid(bounds=fixed_bounds)
            pl.camera_position = camera_position
            pl.camera.SetViewAngle(30.0)
            pl.write_frame()

        try:
            initial_size = np.ones(initial_points_np.shape[0], dtype=float)
            initial_alpha = np.ones(initial_points_np.shape[0], dtype=float)
            initial_scalar = initial_scalar_state_np.astype(float, copy=False) if scalar_mode and initial_scalar_state_np is not None else None
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

    def save_data_npz(self, output_path: str) -> Path:
        """Save final simulation state to a compressed NPZ file."""
        out = Path(output_path)
        if out.parent and not out.parent.exists():
            out.parent.mkdir(parents=True, exist_ok=True)

        points = self.points_numpy().astype(np.float32, copy=False)
        cell_ids = self._as_numpy(self.cell_ids).astype(np.int64, copy=False)
        birth_age = self._as_numpy(self.birth_age).astype(np.int32, copy=False)
        program_counter = self._as_numpy(self.program_counter).astype(np.int32, copy=False)
        local_frame = self._as_numpy(self.local_frame).astype(
            np.float32, copy=False
        )
        np.savez_compressed(
            out,
            points=points,
            cell_ids=cell_ids,
            birth_age=birth_age,
            program_counter=program_counter,
            local_frame=local_frame,
            count_history=np.asarray(self.count_history, dtype=np.int32),
            step_index=np.int32(self.step_index),
            next_cell_id=np.int64(self._next_cell_id),
            radius=np.float32(self.radius),
            split_distance=np.float32(self.split_distance),
            program_text=np.asarray(self.program_text),
            compiled_program_tokens=np.asarray(self.program.raw_tokens),
            daughter_program_mode=np.asarray(self.daughter_program_mode),
            program_loop=np.int8(1 if self.program_loop else 0),
            dtype=np.asarray(self.dtype),
        )
        print(f"Saved data: {out}")
        return out

    def visualize_pyvista(self, **kwargs) -> None:
        """Render final cells as spheres in PyVista."""
        opts = dict(kwargs)
        color_by = opts.get("color_by", DEFAULT_COLOR_BY)
        scalar_values = None
        if color_by == "age":
            scalar_values = self._as_numpy(self.birth_age).astype(float, copy=False)
            opts["color_by"] = "scalar"
        elif color_by in {"program_step", "prog"}:
            scalar_values = self._as_numpy(self.program_counter).astype(float, copy=False)
            opts["color_by"] = "scalar"
        elif color_by == "solid":
            opts["color_by"] = "solid"
        if scalar_values is not None:
            opts["scalar_values"] = scalar_values
        show_cells_pyvista(self.points_numpy(), cell_radius=self.radius, **opts)

    def save_snapshot_pyvista(self, output_path: str, **kwargs) -> Path:
        """Save final cells as a static PyVista snapshot image."""
        opts = dict(kwargs)
        color_by = opts.get("color_by", DEFAULT_COLOR_BY)
        scalar_values = None
        if color_by == "age":
            scalar_values = self._as_numpy(self.birth_age).astype(float, copy=False)
            opts["color_by"] = "scalar"
        elif color_by in {"program_step", "prog"}:
            scalar_values = self._as_numpy(self.program_counter).astype(float, copy=False)
            opts["color_by"] = "scalar"
        elif color_by == "solid":
            opts["color_by"] = "solid"
        if scalar_values is not None:
            opts["scalar_values"] = scalar_values
        out = show_cells_pyvista(
            self.points_numpy(),
            cell_radius=self.radius,
            snapshot_path=output_path,
            **opts,
        )
        if out is None:
            raise RuntimeError("snapshot export did not produce an output path")
        return out



def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Program-driven 3D point-growth simulator")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"Number of timesteps (default: {DEFAULT_STEPS})")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS, help="Cell radius used in PyVista rendering")
    parser.add_argument("--split-distance", type=float, default=DEFAULT_SPLIT_DISTANCE, help="Default daughter displacement distance for divide instructions")
    parser.add_argument(
        "--program",
        type=str,
        default=DEFAULT_PROGRAM,
        help=(
            "Semicolon-delimited instruction string, e.g. "
            "'D[x];D[1,0.1,0.33+(age*0.01)+(step*0.01)];Dg[z@1.5+age*0.05];S'. "
            "D[...] uses the cell's carried local frame; Dg[...] or Dabs[...] use the fixed global frame; "
            "numeric slots may depend on age and step."
        ),
    )
    parser.add_argument("--program-loop", action=argparse.BooleanOptionalAction, default=DEFAULT_PROGRAM_LOOP, help="Loop the instruction string after the last instruction")
    parser.add_argument(
        "--daughter-program-mode",
        choices=["restart", "inherit"],
        default=DEFAULT_DAUGHTER_PROGRAM_MODE,
        help="Whether daughters restart the program or inherit the parent's next program step",
    )
    parser.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS, help="Safety cap to prevent uncontrolled growth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed placeholder for future stochastic program extensions")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=DEFAULT_SHOW, help="Show final cells in an interactive PyVista window")
    parser.add_argument("--show-centers", action="store_true", help="Overlay cell center points in PyVista views")
    parser.add_argument("--view-max-render-cells", type=int, default=DEFAULT_VIEW_MAX_RENDER_CELLS, help="Maximum cells to render in the final static view (0 disables cap)")
    parser.add_argument(
        "--color-by",
        choices=["order", "radius", "solid", "age", "program_step", "prog"],
        default=DEFAULT_COLOR_BY,
        help="Sphere coloring mode in PyVista views",
    )
    parser.add_argument("--save-data", action=argparse.BooleanOptionalAction, default=DEFAULT_SAVE_DATA, help="Save final cell state to a compressed NPZ file")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="Output NPZ path for --save-data")
    parser.add_argument("--save-snapshot", action=argparse.BooleanOptionalAction, default=DEFAULT_SAVE_SNAPSHOT, help="Save final PyVista snapshot as a PNG image")
    parser.add_argument("--snapshot-path", type=str, default=DEFAULT_SNAPSHOT_PATH, help="Output PNG path for --save-snapshot")
    parser.add_argument("--save-movie", action=argparse.BooleanOptionalAction, default=DEFAULT_SAVE_MOVIE, help="Save MP4 movie of the simulation with interpolated motion")
    parser.add_argument("--movie-path", type=str, default=DEFAULT_MOVIE_PATH, help="Output MP4 path for --save-movie")
    parser.add_argument("--movie-fps", type=int, default=DEFAULT_MOVIE_FPS, help="Frames per second for --save-movie")
    parser.add_argument("--movie-duration-seconds", type=float, default=DEFAULT_MOVIE_DURATION_SECONDS, help="Target movie duration in seconds; FPS is auto-adjusted when set")
    parser.add_argument("--interp-frames", type=int, default=DEFAULT_INTERP_FRAMES, help="Interpolated frames between consecutive timesteps in movie mode")
    parser.add_argument("--movie-adaptive-large-render", action=argparse.BooleanOptionalAction, default=DEFAULT_MOVIE_ADAPTIVE_LARGE, help="Adapt movie rendering for large colonies")
    parser.add_argument("--movie-large-cells-threshold", type=int, default=DEFAULT_MOVIE_LARGE_CELLS_THRESHOLD, help="Cell-count threshold where adaptive movie behavior starts")
    parser.add_argument("--movie-large-interp-frames", type=int, default=DEFAULT_MOVIE_LARGE_INTERP_FRAMES, help="Interp frames used after threshold in adaptive movie mode")
    parser.add_argument("--movie-max-render-cells", type=int, default=DEFAULT_MOVIE_MAX_RENDER_CELLS, help="Per-frame cell cap for movie rendering (0 disables cap)")
    parser.add_argument("--movie-width", type=int, default=DEFAULT_MOVIE_WIDTH, help="Movie frame width in pixels")
    parser.add_argument("--movie-height", type=int, default=DEFAULT_MOVIE_HEIGHT, help="Movie frame height in pixels")
    parser.add_argument("--movie-sphere-theta", type=int, default=DEFAULT_MOVIE_SPHERE_THETA, help="Sphere theta resolution for movie rendering")
    parser.add_argument("--movie-sphere-phi", type=int, default=DEFAULT_MOVIE_SPHERE_PHI, help="Sphere phi resolution for movie rendering")
    parser.add_argument("--movie-show-edges", action=argparse.BooleanOptionalAction, default=DEFAULT_MOVIE_SHOW_EDGES, help="Render mesh edges in movie frames")
    parser.add_argument("--movie-edge-color", type=str, default=DEFAULT_MOVIE_EDGE_COLOR, help="Edge color for movie sphere meshes")
    parser.add_argument("--movie-edge-width", type=float, default=DEFAULT_MOVIE_EDGE_WIDTH, help="Edge line width for movie sphere meshes")
    parser.add_argument("--movie-macro-block-size", type=int, default=DEFAULT_MOVIE_MACRO_BLOCK_SIZE, help="Macro block size for MP4 encoding")
    parser.add_argument(
        "--movie-death-animation",
        choices=["none", "fade", "shrink", "fade_shrink"],
        default=DEFAULT_MOVIE_DEATH_ANIMATION,
        help="Dying-cell animation in movies",
    )
    return parser



def main() -> None:
    args = _build_cli().parse_args()
    movie_max_render_cells = None if args.movie_max_render_cells == 0 else args.movie_max_render_cells
    view_max_render_cells = None if args.view_max_render_cells == 0 else args.view_max_render_cells

    sim = CellGrowth3D(
        radius=args.radius,
        split_distance=args.split_distance,
        program_text=args.program,
        daughter_program_mode=args.daughter_program_mode,
        program_loop=args.program_loop,
        seed=args.seed,
        max_cells=args.max_cells,
    )

    print(backend_summary())
    print(
        "Program model: "
        f"steps={args.steps}, split_distance={sim.split_distance:g}, program='{sim.program_text}', "
        f"daughter_mode={sim.daughter_program_mode}, loop={sim.program_loop}"
    )
    print(
        "Instruction legend: "
        "S=stay, X=die, D[dir@dist]=local-frame divide, "
        "Dg[dir@dist]=global-frame divide; "
        "dir/dist expressions may use age and step; "
        "axis aliases: x,-x,y,-y,z,-z"
    )
    print("Note: This branch is a reduced program-driven engine. Crowding, RD, and polarity are disabled.")

    color_by = "solid" if args.color_by == "solid" else args.color_by
    movie_color_by = "none" if color_by == "solid" else color_by

    if args.save_movie:
        sim.run_and_save_movie(
            args.steps,
            output_path=args.movie_path,
            log_counts=True,
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
        sim.run(args.steps, log_counts=True)

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
