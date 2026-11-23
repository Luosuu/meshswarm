"""Three-tier Modelica-to-JAX benchmark task definitions.

Modelica sources live as standalone `.mo` files in `models/`, and the starter
deliverable template lives in `skeleton_template.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
REFS_DIR = HERE / "references"
SKELETON_PATH = HERE / "skeleton_template.py"

DELIVERABLE = "solver_jax.py"
KIND = (
    "refactoring engineering Modelica (.mo) differential-algebraic models into "
    "conservation-preserving, differentiable JAX GPU ensemble solvers"
)

# Combined wall-time rungs (seconds) for 1,000-trajectory forward GPU solve + 64-trajectory jax.grad pass.
THRESHOLDS_S = [30.0, 15.0, 8.0, 4.0, 2.0, 1.0, 0.50, 0.25, 0.12, 0.06]

STAGE_HOLDUPS_16: list[float] = [4.0] + [1.0] * 14 + [6.0]
TIER3_X_H2_0: list[float] = [round(0.55 * math.exp(-0.14 * i), 10) for i in range(16)]
TIER3_X_HD_0: list[float] = [0.28] * 16
TIER3_Y0: list[float] = TIER3_X_H2_0 + TIER3_X_HD_0


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    title: str
    about: str
    model_file: str
    model_name: str
    needs_transform: bool
    t_end: float
    n_states: int
    param_names: list[str]
    obs_names: list[str]
    y0_default: list[float]
    p_default: list[float]
    dev_cases: list[list[float]]
    test_cases: list[list[float]]
    ensemble_bounds: tuple[list[float], list[float]]
    traj_rtol: float
    traj_atol: float
    conservation_kind: str  # "sum_last3" | "hdt_column16"

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / self.model_file

    def model_source(self) -> str:
        return self.model_path.read_text(encoding="utf-8")


TASKS: dict[str, BenchmarkTask] = {
    "tier1_blanket_dae": BenchmarkTask(
        name="tier1_blanket_dae",
        title="a stiff fusion blanket check-valve & implicit Sieverts permeation DAE loop",
        about=(
            "The Modelica model `BlanketValveDAE.mo` couples fast primary coolant hydraulics "
            "(`tau_hyd = 0.02 s`, piecewise forward/reverse check-valve resistance and orifice bypass), "
            "blanket and heat-exchanger thermal inertia, and a closed-loop 3-reservoir tritium inventory "
            "governed by an implicit cubic-Sieverts algebraic flux equation."
        ),
        model_file="BlanketValveDAE.mo",
        model_name="BlanketValveDAE",
        needs_transform=False,
        t_end=20.0,
        n_states=6,
        param_names=["u_pump", "u_ext", "P_pulse", "dp_back"],
        obs_names=["obs_q_flow", "obs_T_blanket", "obs_T_hx", "obs_N_blanket", "obs_N_extractor", "obs_N_storage"],
        y0_default=[1.6, 755.0, 635.0, 4.2, 0.4, 0.4],
        p_default=[1.0, 0.70, 1.0, 0.35],
        dev_cases=[
            [1.00, 0.70, 1.00, 0.35],
            [0.12, 0.95, 1.30, 0.55],
        ],
        test_cases=[
            [1.25, 0.85, 0.80, 0.25],
            [0.75, 0.50, 1.35, 0.40],
            [0.18, 1.10, 1.15, 0.52],
            [0.05, 0.90, 0.70, 0.65],
            [1.40, 0.35, 1.45, 0.20],
            [0.55, 1.25, 0.60, 0.45],
            [0.28, 0.75, 1.20, 0.62],
            [0.95, 1.05, 0.95, 0.30],
        ],
        ensemble_bounds=([0.04, 0.25, 0.50, 0.15], [1.45, 1.30, 1.50, 0.70]),
        traj_rtol=2.5e-4,
        traj_atol=1.0e-5,
        conservation_kind="sum_last3",
    ),
    "tier2_transform_hx": BenchmarkTask(
        name="tier2_transform_hx",
        title="an ORNL TRANSFORM counter-current PbLi/FLiBe heat exchanger with Arrhenius tritium extraction",
        about=(
            "The Modelica model `TransformPbLiFLiBeHX.mo` instantiates ORNL's `TRANSFORM.HeatExchangers.Simple_HX` "
            "(`nV=6`, `counterCurrent=true`) between liquid lead-lithium (`ConstantPropertyLiquidPbLi`) and "
            "molten salt (`ConstantPropertyLiquidFLiBe`), with piecewise low-flow limiters and a 6-zone "
            "Arrhenius wall-permeation + implicit nonlinear Sieverts tritium extraction loop."
        ),
        model_file="TransformPbLiFLiBeHX.mo",
        model_name="TransformPbLiFLiBeHX",
        needs_transform=True,
        t_end=30.0,
        n_states=15,
        param_names=["m_flow_pbli", "m_flow_flibe", "T_in_pbli", "u_vac"],
        obs_names=["obs_Q_MW", "obs_T1_out", "obs_T2_out", "obs_N_loop", "obs_N_perm", "obs_N_getter"],
        y0_default=[930.0] * 6 + [880.0] * 6 + [5.0, 0.5, 0.5],
        p_default=[90.0, 12.0, 955.0, 0.80],
        dev_cases=[
            [90.0, 12.0, 955.0, 0.80],
            [12.0, 2.5, 940.0, 1.10],
        ],
        test_cases=[
            [110.0, 15.0, 965.0, 0.95],
            [65.0, 9.5, 945.0, 0.60],
            [14.0, 11.0, 950.0, 1.15],
            [80.0, 2.6, 935.0, 0.85],
            [12.5, 2.4, 960.0, 1.05],
            [98.0, 13.5, 970.0, 0.50],
            [50.0, 7.5, 930.0, 1.20],
            [75.0, 10.5, 952.0, 0.75],
        ],
        ensemble_bounds=([11.0, 2.2, 925.0, 0.40], [115.0, 16.0, 972.0, 1.25]),
        traj_rtol=3.0e-4,
        traj_atol=1.0e-5,
        conservation_kind="sum_last3",
    ),
    "tier3_hdt_column": BenchmarkTask(
        name="tier3_hdt_column",
        title="a 16-stage cryogenic H2/HD/T2 isotope distillation MESH DAE column",
        about=(
            "The Modelica model `CryoHDTColumn16.mo` models a 16-stage closed-recycle cryogenic hydrogen "
            "isotope distillation column (`32` differential mole-fraction states + `16` implicit Antoine "
            "bubble-point algebraic equations + hydraulic flooding efficiency drop above `V_flood = 2.15 mol/s`). "
            "All three total column species inventories (H2, HD, T2) are strictly conserved."
        ),
        model_file="CryoHDTColumn16.mo",
        model_name="CryoHDTColumn16",
        needs_transform=False,
        t_end=25.0,
        n_states=32,
        param_names=["V_boil", "d_rec_frac", "dp_tray", "P_top"],
        obs_names=[
            "obs_xH2_top", "obs_xT2_bot", "obs_T_top", "obs_T_bot",
            "obs_tot_H2", "obs_tot_HD", "obs_tot_T2",
        ],
        y0_default=list(TIER3_Y0),
        p_default=[1.85, 0.14, 0.012, 1.15],
        dev_cases=[
            [1.85, 0.14, 0.012, 1.15],
            [2.30, 0.18, 0.015, 1.25],
        ],
        test_cases=[
            [1.60, 0.10, 0.010, 1.08],
            [1.95, 0.16, 0.014, 1.20],
            [2.12, 0.12, 0.012, 1.15],
            [2.40, 0.20, 0.016, 1.30],
            [1.45, 0.22, 0.009, 1.05],
            [2.05, 0.08, 0.013, 1.18],
            [2.28, 0.15, 0.011, 1.12],
            [1.75, 0.19, 0.015, 1.24],
        ],
        ensemble_bounds=([1.35, 0.06, 0.008, 1.02], [2.45, 0.24, 0.018, 1.35]),
        traj_rtol=3.5e-4,
        traj_atol=1.0e-5,
        conservation_kind="hdt_column16",
    ),
}


def skeleton_code(task: BenchmarkTask) -> str:
    tpl = SKELETON_PATH.read_text(encoding="utf-8")
    return (
        tpl.replace("__MODEL_NAME__", task.model_name)
        .replace("__N_STATES__", str(task.n_states))
        .replace("__N_PARAMS__", str(len(task.param_names)))
        .replace("__PARAM_NAMES__", str(task.param_names))
        .replace("__N_OBS__", str(len(task.obs_names)))
        .replace("__OBS_NAMES__", str(task.obs_names))
        .replace("__T_END__", str(task.t_end))
    )


def protect_patterns(name: str) -> list[str]:
    return [TASKS[name].model_file, DELIVERABLE]


def starting_files(name: str) -> dict[str, str]:
    t = TASKS[name]
    return {
        t.model_file: t.model_source(),
        DELIVERABLE: skeleton_code(t),
    }


def reference_path(name: str) -> Path:
    return REFS_DIR / f"{name}.py"


def task_text(name: str) -> str:
    t = TASKS[name]
    rungs = ", ".join(f"{s:g}s" for s in THRESHOLDS_S)
    omc_hint = (
        "Set `setModelicaPath(getEnvironmentVar(\"OPENMODELICALIBRARY\"))`, `loadModel(Modelica)`, and "
        "`loadFile(getEnvironmentVar(\"TRANSFORM_LIB\"))` in `.mos` scripts to inspect (`instantiateModel`) or simulate with `omc`."
        if t.needs_transform
        else "Use `omc` to inspect or simulate the `.mo` file (`method=\"dassl\", tolerance=1e-10`) and compare your JAX outputs."
    )
    return f"""Build the fastest verified, differentiable JAX GPU ensemble solver for {t.title}.

{t.about}

Your workspace holds:
  - `{t.model_file}`: the reference Modelica model (`t in [0, {t.t_end}]`). Direct `Rumoca.to_jax()` fails on this model; you must handle its package flattening, algebraic constraints, and non-smooth branches properly. Do not modify `{t.model_file}`.
  - `{DELIVERABLE}`: the deliverable module you must implement.

Required functions in `{DELIVERABLE}` (FP64 enabled):
  1. `forward_ensemble(y0_batch, p_batch)` -> `(B, {len(t.obs_names)})` final observables `{t.obs_names}` at `t = {t.t_end}` for initial states `(B, {t.n_states})` and parameters `(B, {len(t.param_names)})` = `{t.param_names}`.
  2. `differentiable_closure(y0_batch, p_ref_batch)` -> `Callable[[jax.Array], jax.Array]` returning a `jax.grad`-differentiable function `p_jax -> obs_final_jax` around `p_ref_batch`.

Note on stiff solvers & implicit equations: if you use a Rosenbrock solver such as `gradsolve` (`rodas5p_replay`), it computes per-step state Jacobians with `jax.jacfwd` and differentiates through them in reverse mode, so any custom implicit root-finder should use `@jax.custom_jvp` (Implicit Function Theorem) rather than `@jax.custom_vjp`.

Default initial state `y0`: `{t.y0_default}`
Example parameter sets `p`: `{t.dev_cases}`

How it is graded (strict multiplicative zero-multiplier gates on unseen parameter sets):
  1. Compilation & finiteness: runs on GPU in float64 with no NaN/Inf and no external subprocess/omc/stop_gradient calls.
  2. Trajectory parity: matches OpenModelica DASSL (`tolerance=1e-10`) on held-out parameter sets (including regime transitions) to relative error <= {t.traj_rtol:g}.
  3. Exact conservation: closed-loop molar inventory drift across 1,000 trajectories must be < 1e-11.
  4. Exact reverse-mode sensitivities: `jax.grad` through `differentiable_closure` must match `jax.jvp` to < 1e-10, closure central finite differences to < 5e-5, AND fresh macro finite differences to < 1.5e-2.
  5. Speedup / wall-time ladder: once all four gates pass, score (0 to 10) is the share of wall-time rungs ({rungs}) beaten by `(median of 5 runs of 1,000-trajectory forward_ensemble) + (64-trajectory jax.grad pass)`.

{omc_hint}"""
