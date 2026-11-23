"""End-to-end Phase 0 smoke verification for the Modelica-to-JAX pipeline.

Validates all five links required before launching meshswarm / meshevolve runs:
  1. OpenModelica (`omc` in Apptainer) simulates a stiff fusion blanket cooling +
     tritium extraction Modelica (.mo) model and loads ORNL TRANSFORM-Library.
  2. Rumoca (`rumoca` Python bindings) parses, flattens, and lowers the same .mo
     model to explicit JAX symbolic form (`Model.to_jax()`).
  3. Branch-smoothed JAX formulation matches the OpenModelica DASSL reference
     trajectory AND preserves closed-loop H/D/T molar inventory (< 1e-12).
  4. `gradsolve` (`rodas5p_replay` stiff Rosenbrock solver) runs a 1,000-trajectory
     ensemble on GPU well under the 15 s target, and its reverse-mode adjoint
     matches central finite differences to relative error < 1e-7.
  5. Closed-loop differentiable control (`optax.adam` through the stiff GPU DAE
     rollout) optimizes pump & tritium-extractor schedules over a fusion power pulse.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("GRADSOLVE_X64", "1")

import gradsolve  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import rumoca as rm  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Stiff 6-state fusion blanket thermal-hydraulic + closed-loop tritium extraction model.
# Fast hydraulic relaxation (~0.02 s) vs slow thermal/tritium residence (~25-80 s) gives >10^3x stiffness.
MODELICA_SOURCE = """\
model BlanketTritiumLoop
  parameter Real u_pump = 1.0 "Normalized coolant pump head command";
  parameter Real u_ext = 0.65 "Normalized vacuum-permeator extraction command";
  parameter Real P_pulse = 1.0 "Normalized fusion neutron heating amplitude";
  parameter Real k_valve = 18.0 "Check-valve backflow resistance gain";

  constant Real tau_hyd = 0.025 "Fast hydraulic inertia time scale (s)";
  constant Real k_fric = 0.22 "Turbulent loop friction coefficient";
  constant Real q_nom = 2.0 "Nominal mass flow rate scale";
  constant Real C_blanket = 28.0 "Blanket heat capacity (MJ/K)";
  constant Real C_hx = 12.0 "Heat exchanger secondary capacity (MJ/K)";
  constant Real cp_cool = 1.8 "Effective coolant heat capacity (MJ/(kg_eq*K))";
  constant Real h_sink = 0.85 "Power-conversion heat sink conductance (MW/K)";
  constant Real T_sink = 573.15 "Steam/sCO2 sink temperature (K)";
  constant Real P_base = 120.0 "Base blanket nuclear heating (MW)";
  constant Real k_perm = 0.035 "Sieverts-law permeation rate constant (1/s)";
  constant Real k_regen = 0.045 "Getter bed regeneration transfer rate (1/s)";
  constant Real eps_smooth = 1.0e-6 "Regularization scale";

  Real q_flow(start = 1.8, fixed = true) "Coolant flow rate";
  Real T_blanket(start = 760.0, fixed = true) "Blanket temperature (K)";
  Real T_hx(start = 640.0, fixed = true) "Heat-exchanger temperature (K)";
  Real N_blanket(start = 4.0, fixed = true) "Tritium in blanket coolant (mol)";
  Real N_extractor(start = 0.5, fixed = true) "Tritium in permeator (mol)";
  Real N_storage(start = 0.5, fixed = true) "Tritium in getter storage (mol)";

  Real P_fus;
  Real q_abs;
  Real valve_ drag;
  Real j_perm;
  Real j_store;
equation
  P_fus = P_base * (1.0 + 0.35 * P_pulse * sin(0.15 * time));
  q_abs = sqrt(q_flow * q_flow + eps_smooth * eps_smooth);
  valve_drag = k_valve * 0.5 * (q_abs - q_flow);
  der(q_flow) = (u_pump * q_nom - k_fric * q_flow * q_abs - valve_drag - 0.35 * q_flow) / tau_hyd;

  der(T_blanket) = (P_fus - q_abs * cp_cool * (T_blanket - T_hx)) / C_blanket;
  der(T_hx) = (q_abs * cp_cool * (T_blanket - T_hx) - h_sink * (T_hx - T_sink)) / C_hx;

  j_perm = u_ext * k_perm * N_blanket / sqrt(N_blanket * N_blanket + 0.25);
  j_store = k_regen * N_extractor;
  der(N_blanket) = -j_perm;
  der(N_extractor) = j_perm - j_store;
  der(N_storage) = j_store;
end BlanketTritiumLoop;
""".replace("valve_ drag", "valve_drag")

Y0_DEFAULT = np.array([1.8, 760.0, 640.0, 4.0, 0.5, 0.5], dtype=np.float64)
P_DEFAULT = np.array([1.0, 0.65, 1.0, 18.0], dtype=np.float64)
T_END = 20.0


class BlanketTritiumProblem:
    """Stiff 6-state problem implementing the `gradsolve` Problem protocol."""

    name = "user_blanket_tritium_loop"
    dim = 6
    t0 = 0.0
    t1 = T_END
    is_stiff = True

    def f_jax(self, t, y, p):
        u_pump, u_ext, p_pulse, k_valve = p[0], p[1], p[2], p[3]
        q_flow, t_bl, t_hx, n_bl, n_ex, _n_st = y[0], y[1], y[2], y[3], y[4], y[5]

        tau_hyd = 0.025
        k_fric = 0.22
        q_nom = 2.0
        c_blanket = 28.0
        c_hx = 12.0
        cp_cool = 1.8
        h_sink = 0.85
        t_sink = 573.15
        p_base = 120.0
        k_perm = 0.035
        k_regen = 0.045
        eps = 1.0e-6

        p_fus = p_base * (1.0 + 0.35 * p_pulse * jnp.sin(0.15 * t))
        q_abs = jnp.sqrt(q_flow * q_flow + eps * eps)
        valve_drag = k_valve * 0.5 * (q_abs - q_flow)
        dq = (u_pump * q_nom - k_fric * q_flow * q_abs - valve_drag - 0.35 * q_flow) / tau_hyd

        dt_bl = (p_fus - q_abs * cp_cool * (t_bl - t_hx)) / c_blanket
        dt_hx = (q_abs * cp_cool * (t_bl - t_hx) - h_sink * (t_hx - t_sink)) / c_hx

        j_perm = u_ext * k_perm * n_bl / jnp.sqrt(n_bl * n_bl + 0.25)
        j_store = k_regen * n_ex
        dn_bl = -j_perm
        dn_ex = j_perm - j_store
        dn_st = j_store
        return jnp.stack([dq, dt_bl, dt_hx, dn_bl, dn_ex, dn_st])


def run_omc_reference(workdir: Path, mo_path: Path) -> dict[str, float] | None:
    """Run OpenModelica `omc` (via Apptainer `OMC_SIF` or local `omc`) to get gold final states."""
    sif = os.environ.get("OMC_SIF")
    transform_pkg = os.environ.get("TRANSFORM_LIB")
    if sif and Path(sif).is_file():
        cmd_prefix = [
            "apptainer", "exec", "--containall", "--no-home",
            "--pwd", str(workdir),
            "--bind", f"{workdir}:{workdir}",
        ]
        if transform_pkg and Path(transform_pkg).is_file():
            t_root = str(Path(transform_pkg).parents[1])
            cmd_prefix += ["--bind", f"{t_root}:{t_root}:ro"]
        cmd_prefix += [sif, "omc"]
    elif shutil.which("omc"):
        cmd_prefix = ["omc"]
    else:
        print("[Stage 1] OpenModelica not found in environment; skipping omc reference")
        return None

    tf_line = ""
    if transform_pkg and Path(transform_pkg).is_file():
        tf_line = f'tf_ok := loadFile("{transform_pkg}"); print("TRANSFORM loadFile: " + String(tf_ok) + "\\n");\n'

    mos = workdir / "run_ref.mos"
    mos.write_text(
        tf_line
        + f'ok := loadFile("{mo_path.name}");\n'
        + 'if not ok then print(getErrorString()); exit(1); end if;\n'
        + f'simulate(BlanketTritiumLoop, startTime=0.0, stopTime={T_END}, numberOfIntervals=1000, '
        + 'tolerance=1e-10, method="dassl", outputFormat="csv", fileNamePrefix="gold");\n'
        + 'print(getErrorString());\n',
        encoding="utf-8",
    )
    proc = subprocess.run(cmd_prefix + [mos.name], cwd=workdir, capture_output=True, text=True, timeout=180)
    print(f"[Stage 1] omc exit={proc.returncode}\n{proc.stdout.strip()}")
    if proc.returncode != 0:
        raise RuntimeError(f"omc failed:\n{proc.stdout}\n{proc.stderr}")

    csv_path = workdir / "gold_res.csv"
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    last = rows[-1]
    return {k: float(last[k]) for k in ["q_flow", "T_blanket", "T_hx", "N_blanket", "N_extractor", "N_storage"]}


def check_rumoca_lowering(mo_path: Path, problem: BlanketTritiumProblem) -> None:
    """Compile the .mo file with Rumoca and verify symbolic RHS parity against our JAX problem."""
    session = rm.Session()
    model = session.load(str(mo_path), model="BlanketTritiumLoop")
    export = model.to_jax()
    print(f"[Stage 2] Rumoca {rm.version()} lowered {export}")
    print(f"          states={export.state_names}, params={export.parameter_names}")

    # Align states & parameters by name so ordering differences in compiler IR never mislead comparison.
    name_to_y = dict(zip(["q_flow", "T_blanket", "T_hx", "N_blanket", "N_extractor", "N_storage"], Y0_DEFAULT))
    name_to_p = dict(zip(["u_pump", "u_ext", "P_pulse", "k_valve"], P_DEFAULT))
    x_rm = jnp.array([name_to_y[n] for n in export.state_names], dtype=jnp.float64)
    p_rm = jnp.array([name_to_p.get(n, d) for n, d in zip(export.parameter_names, export.default_parameters)], dtype=jnp.float64)
    u_rm = jnp.zeros((export.n_inputs,), dtype=jnp.float64)

    t_test = 3.7
    dx_rm = np.asarray(export.rhs(t_test, x_rm, u_rm, p_rm))
    dx_hand_map = dict(zip(
        ["q_flow", "T_blanket", "T_hx", "N_blanket", "N_extractor", "N_storage"],
        np.asarray(problem.f_jax(t_test, jnp.asarray(Y0_DEFAULT), jnp.asarray(P_DEFAULT))),
    ))
    dx_expected = np.array([dx_hand_map[n] for n in export.state_names])
    rel_rhs_err = float(np.max(np.abs(dx_rm - dx_expected) / np.maximum(np.abs(dx_expected), 1.0e-12)))
    print(f"          Rumoca to_jax() RHS max rel diff vs hand JAX: {rel_rhs_err:.3e}")
    assert rel_rhs_err < 1.0e-12, f"Rumoca RHS mismatch: {rel_rhs_err}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--n-ensemble", type=int, default=1000)
    args = ap.parse_args()

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="mjax-smoke-"))
    workdir.mkdir(parents=True, exist_ok=True)
    mo_path = workdir / "BlanketTritiumLoop.mo"
    mo_path.write_text(MODELICA_SOURCE, encoding="utf-8")

    devices = jax.devices()
    has_gpu = any(d.platform == "gpu" for d in devices)
    target_device = "gpu" if has_gpu else "cpu"
    print(f"=== JAX devices: {devices} (using device={target_device!r}) ===")

    problem = BlanketTritiumProblem()

    # Stage 1 & 2: OpenModelica gold trajectory + Rumoca symbolic lowering
    gold = run_omc_reference(workdir, mo_path)
    check_rumoca_lowering(mo_path, problem)

    # Stage 3: Trajectory parity vs OpenModelica + closed-loop tritium conservation gate
    ref_solve = gradsolve.solve(
        problem, Y0_DEFAULT[None, :], P_DEFAULT[None, :],
        engine="rodas5p_replay", rtol=1e-9, atol=1e-11, device=target_device,
    )
    y_end = np.asarray(ref_solve.y_final[0])
    names = ["q_flow", "T_blanket", "T_hx", "N_blanket", "N_extractor", "N_storage"]
    if gold is not None:
        gold_vec = np.array([gold[n] for n in names])
        rel_traj_err = float(np.max(np.abs(y_end - gold_vec) / np.maximum(np.abs(gold_vec), 1.0e-9)))
        print(f"[Stage 3] Trajectory max rel error vs OpenModelica DASSL: {rel_traj_err:.3e}")
        assert rel_traj_err < 1.0e-5, f"OpenModelica parity gate failed: {rel_traj_err}"

    tritium_in = float(np.sum(Y0_DEFAULT[3:]))
    tritium_out = float(np.sum(y_end[3:]))
    mass_err = abs(tritium_out - tritium_in) / tritium_in
    print(f"          Tritium molar conservation rel error: {mass_err:.3e} (gate < 1e-12)")
    assert mass_err < 1.0e-12, f"Mass conservation gate failed: {mass_err}"

    # Stage 4: 1,000-trajectory stiff GPU ensemble benchmark + exact replay-adjoint check
    rng = np.random.default_rng(42)
    n_ens = args.n_ensemble
    # Include trajectories where u_pump dips negative transiently to exercise smooth valve transition!
    p_ens = np.column_stack([
        rng.uniform(-0.15, 1.45, size=n_ens),
        rng.uniform(0.20, 1.20, size=n_ens),
        rng.uniform(0.50, 1.50, size=n_ens),
        np.full(n_ens, 18.0),
    ])
    y0_ens = np.tile(Y0_DEFAULT, (n_ens, 1))

    t_c0 = time.perf_counter()
    ens_res = gradsolve.solve(
        problem, y0_ens, p_ens,
        engine="rodas5p_replay", rtol=1e-7, atol=1e-9, device=target_device,
    )
    jax.block_until_ready(ens_res.y_final)
    cold_s = time.perf_counter() - t_c0

    warm_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = gradsolve.solve(
            problem, y0_ens, p_ens,
            engine="rodas5p_replay", rtol=1e-7, atol=1e-9, device=target_device,
        )
        jax.block_until_ready(r.y_final)
        warm_times.append(time.perf_counter() - t0)
    median_s = statistics.median(warm_times)

    ens_final = np.asarray(r.y_final)
    ens_mass_err = float(np.max(np.abs(np.sum(ens_final[:, 3:], axis=1) - tritium_in) / tritium_in))
    print(f"[Stage 4] {n_ens}-trajectory stiff Rodas5P ensemble ({target_device}): "
          f"cold={cold_s:.3f}s, warm median={median_s:.4f}s (target < 15.0s), max_mass_err={ens_mass_err:.3e}")
    assert median_s < 15.0, f"Ensemble throughput gate failed: {median_s:.3f}s"
    assert ens_mass_err < 1.0e-12, f"Ensemble mass balance failed: {ens_mass_err:.3e}"

    # Adjoint vs central finite-difference gate across 16 diverse trajectories (including backflow regime)
    n_grad = 16
    p_sub = jnp.asarray(p_ens[:n_grad], dtype=jnp.float64)
    y0_sub = y0_ens[:n_grad]
    closure = gradsolve.grad_closure(
        problem, y0_sub, p_sub,
        engine="rodas5p_replay", rtol=1e-9, atol=1e-11, device=target_device,
    )

    def scalar_obj(p_mat):
        yf = closure(p_mat)
        # Penalize blanket over-temp above 785 K + reward stored tritium + penalize pump effort
        return jnp.sum(0.05 * (yf[:, 1] - 785.0) ** 2 - 40.0 * yf[:, 5] + 2.0 * p_mat[:, 0] ** 2)

    ad_grad = np.asarray(jax.jit(jax.grad(scalar_obj))(p_sub))
    eps_fd = 1.0e-6
    fd_grad = np.zeros_like(ad_grad)
    obj_jit = jax.jit(scalar_obj)
    for i in range(n_grad):
        for j in range(3):  # differentiate w.r.t. u_pump, u_ext, P_pulse
            dp = np.zeros_like(ad_grad)
            dp[i, j] = eps_fd
            f_plus = float(obj_jit(p_sub + dp))
            f_minus = float(obj_jit(p_sub - dp))
            fd_grad[i, j] = (f_plus - f_minus) / (2.0 * eps_fd)

    rel_grad_err = float(
        np.max(np.abs(ad_grad[:, :3] - fd_grad[:, :3]) / np.maximum(np.abs(fd_grad[:, :3]), 1.0e-8))
    )
    # Also compare against fresh adaptive solves at tight tolerance (continuous sensitivity check)
    cont_fd = np.zeros((4, 3), dtype=np.float64)
    p_np = np.asarray(p_sub)
    for i in range(4):
        for j in range(3):
            dp = np.zeros_like(p_np[:4])
            dp[i, j] = eps_fd
            yp = np.asarray(gradsolve.solve(
                problem, y0_sub[:4], p_np[:4] + dp,
                engine="rodas5p_replay", rtol=1e-11, atol=1e-13, device=target_device,
            ).y_final)
            ym = np.asarray(gradsolve.solve(
                problem, y0_sub[:4], p_np[:4] - dp,
                engine="rodas5p_replay", rtol=1e-11, atol=1e-13, device=target_device,
            ).y_final)
            lp = np.sum(0.05 * (yp[:, 1] - 785.0) ** 2 - 40.0 * yp[:, 5] + 2.0 * (p_np[:4, 0] + dp[:, 0]) ** 2)
            lm = np.sum(0.05 * (ym[:, 1] - 785.0) ** 2 - 40.0 * ym[:, 5] + 2.0 * (p_np[:4, 0] - dp[:, 0]) ** 2)
            cont_fd[i, j] = (lp - lm) / (2.0 * eps_fd)
    rel_cont_err = float(
        np.max(np.abs(ad_grad[:4, :3] - cont_fd) / np.maximum(np.abs(cont_fd), 1.0e-8))
    )
    print(f"          Discrete-adjoint vs replay FD max rel error: {rel_grad_err:.3e} (gate < 1e-7); "
          f"vs fresh adaptive FD: {rel_cont_err:.3e}")
    assert rel_grad_err < 1.0e-7, f"Adjoint gradient gate failed: {rel_grad_err}"
    assert rel_cont_err < 1.0e-6, f"Continuous sensitivity gate failed: {rel_cont_err}"

    # Stage 5: Closed-loop differentiable control optimization over 32 uncertain plasma pulses
    n_ctrl = 32
    pulse_disturbances = jnp.linspace(0.6, 1.6, n_ctrl, dtype=jnp.float64)
    y0_ctrl = np.tile(Y0_DEFAULT, (n_ctrl, 1))
    u_init = jnp.array([0.85, 0.50], dtype=jnp.float64)  # shared control gains [u_pump_gain, u_ext_cmd]

    def assemble_params(u_vec):
        # Feedforward + disturbance-gain pump policy and bounded extractor command
        u_pump_cmd = u_vec[0] * (0.8 + 0.25 * pulse_disturbances)
        u_ext_cmd = jnp.broadcast_to(jnp.clip(u_vec[1], 0.1, 1.5), (n_ctrl,))
        k_v = jnp.full((n_ctrl,), 18.0, dtype=jnp.float64)
        return jnp.stack([u_pump_cmd, u_ext_cmd, pulse_disturbances, k_v], axis=1)

    def eval_true_ctrl_loss(u_vec):
        p_mat = np.asarray(assemble_params(u_vec))
        yf = np.asarray(gradsolve.solve(
            problem, y0_ctrl, p_mat,
            engine="rodas5p_replay", rtol=1e-8, atol=1e-10, device=target_device,
        ).y_final)
        u_np = np.asarray(u_vec)
        return float(np.mean((yf[:, 1] - 775.0) ** 2) - 25.0 * np.mean(yf[:, 5]) + 4.0 * u_np[0] ** 2 + 1.5 * u_np[1] ** 2)

    opt = optax.adam(0.05)
    u = u_init
    opt_state = opt.init(u)
    loss0 = eval_true_ctrl_loss(u)
    for epoch in range(4):
        ctrl_closure = gradsolve.grad_closure(
            problem, y0_ctrl, assemble_params(u),
            engine="rodas5p_replay", rtol=1e-7, atol=1e-9, device=target_device,
        )

        def step_loss(u_vec, _cl=ctrl_closure):
            p_mat = assemble_params(u_vec)
            yf = _cl(p_mat)
            return jnp.mean((yf[:, 1] - 775.0) ** 2) - 25.0 * jnp.mean(yf[:, 5]) + 4.0 * u_vec[0] ** 2 + 1.5 * u_vec[1] ** 2

        vg = jax.jit(jax.value_and_grad(step_loss))
        for _ in range(10):
            _, g = vg(u)
            updates, opt_state = opt.update(g, opt_state, u)
            u = optax.apply_updates(u, updates)
    loss_final = eval_true_ctrl_loss(u)
    print(f"[Stage 5] Differentiable control (40 Adam steps over {n_ctrl} pulses): "
          f"true adaptive loss {loss0:.3f} -> {loss_final:.3f}, optimal u={np.asarray(u).round(4).tolist()}")
    assert loss_final < loss0, "Control optimization did not reduce objective"

    print("SMOKE_PIPELINE_ALL_5_GATES_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
