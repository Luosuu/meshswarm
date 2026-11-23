"""End-to-end Phase 0 smoke verification for the Modelica-to-JAX pipeline.

Uses the standalone Modelica model `models/BlanketTritiumLoop.mo`.
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

HERE = Path(__file__).resolve().parent
MODEL_SRC_PATH = HERE / "models" / "BlanketTritiumLoop.mo"

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

        p_fus = 120.0 * (1.0 + 0.35 * p_pulse * jnp.sin(0.15 * t))
        q_abs = jnp.sqrt(q_flow * q_flow + 1.0e-12)
        valve_drag = k_valve * 0.5 * (q_abs - q_flow)
        dq = (u_pump * 2.0 - 0.22 * q_flow * q_abs - valve_drag - 0.35 * q_flow) / 0.025

        dt_bl = (p_fus - q_abs * 1.8 * (t_bl - t_hx)) / 28.0
        dt_hx = (q_abs * 1.8 * (t_bl - t_hx) - 0.85 * (t_hx - 573.15)) / 12.0

        j_perm = u_ext * 0.035 * n_bl / jnp.sqrt(n_bl * n_bl + 0.25)
        j_store = 0.045 * n_ex
        return jnp.stack([dq, dt_bl, dt_hx, -j_perm, j_perm - j_store, j_store])


def run_omc_reference(workdir: Path, mo_path: Path) -> dict[str, float] | None:
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

    with open(workdir / "gold_res.csv", encoding="utf-8") as fh:
        last = list(csv.DictReader(fh))[-1]
    return {k: float(last[k]) for k in ["q_flow", "T_blanket", "T_hx", "N_blanket", "N_extractor", "N_storage"]}


def check_rumoca_lowering(mo_path: Path, problem: BlanketTritiumProblem) -> None:
    session = rm.Session()
    model = session.load(str(mo_path), model="BlanketTritiumLoop")
    export = model.to_jax()
    print(f"[Stage 2] Rumoca {rm.version()} lowered {export}")
    print(f"          states={export.state_names}, params={export.parameter_names}")

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
    shutil.copyfile(MODEL_SRC_PATH, mo_path)

    devices = jax.devices()
    target_device = "gpu" if any(d.platform == "gpu" for d in devices) else "cpu"
    print(f"=== JAX devices: {devices} (using device={target_device!r}) ===")

    problem = BlanketTritiumProblem()
    gold = run_omc_reference(workdir, mo_path)
    check_rumoca_lowering(mo_path, problem)

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
        assert rel_traj_err < 1.0e-5

    tritium_in = float(np.sum(Y0_DEFAULT[3:]))
    mass_err = abs(float(np.sum(y_end[3:])) - tritium_in) / tritium_in
    print(f"          Tritium molar conservation rel error: {mass_err:.3e} (gate < 1e-12)")
    assert mass_err < 1.0e-12

    rng = np.random.default_rng(42)
    n_ens = args.n_ensemble
    p_ens = np.column_stack([
        rng.uniform(-0.15, 1.45, size=n_ens),
        rng.uniform(0.20, 1.20, size=n_ens),
        rng.uniform(0.50, 1.50, size=n_ens),
        np.full(n_ens, 18.0),
    ])
    y0_ens = np.tile(Y0_DEFAULT, (n_ens, 1))

    t_c0 = time.perf_counter()
    ens_res = gradsolve.solve(problem, y0_ens, p_ens, engine="rodas5p_replay", rtol=1e-7, atol=1e-9, device=target_device)
    jax.block_until_ready(ens_res.y_final)
    cold_s = time.perf_counter() - t_c0

    warm_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = gradsolve.solve(problem, y0_ens, p_ens, engine="rodas5p_replay", rtol=1e-7, atol=1e-9, device=target_device)
        jax.block_until_ready(r.y_final)
        warm_times.append(time.perf_counter() - t0)
    median_s = statistics.median(warm_times)
    ens_mass_err = float(np.max(np.abs(np.sum(np.asarray(r.y_final)[:, 3:], axis=1) - tritium_in) / tritium_in))
    print(f"[Stage 4] {n_ens}-trajectory stiff Rodas5P ensemble ({target_device}): "
          f"cold={cold_s:.3f}s, warm median={median_s:.4f}s (target < 15.0s), max_mass_err={ens_mass_err:.3e}")
    assert median_s < 15.0 and ens_mass_err < 1.0e-12

    n_grad = 16
    p_sub = jnp.asarray(p_ens[:n_grad], dtype=jnp.float64)
    y0_sub = y0_ens[:n_grad]
    closure = gradsolve.grad_closure(problem, y0_sub, p_sub, engine="rodas5p_replay", rtol=1e-9, atol=1e-11, device=target_device)

    def scalar_obj(p_mat):
        yf = closure(p_mat)
        return jnp.sum(0.05 * (yf[:, 1] - 785.0) ** 2 - 40.0 * yf[:, 5] + 2.0 * p_mat[:, 0] ** 2)

    ad_grad = np.asarray(jax.jit(jax.grad(scalar_obj))(p_sub))
    eps_fd = 1.0e-6
    fd_grad = np.zeros_like(ad_grad)
    obj_jit = jax.jit(scalar_obj)
    for i in range(n_grad):
        for j in range(3):
            dp = np.zeros_like(ad_grad)
            dp[i, j] = eps_fd
            fd_grad[i, j] = (float(obj_jit(p_sub + dp)) - float(obj_jit(p_sub - dp))) / (2.0 * eps_fd)

    rel_grad_err = float(np.max(np.abs(ad_grad[:, :3] - fd_grad[:, :3]) / np.maximum(np.abs(fd_grad[:, :3]), 1.0e-8)))
    print(f"          Discrete-adjoint vs replay FD max rel error: {rel_grad_err:.3e} (gate < 1e-7)")
    assert rel_grad_err < 1.0e-7

    n_ctrl = 32
    pulse_disturbances = jnp.linspace(0.6, 1.6, n_ctrl, dtype=jnp.float64)
    y0_ctrl = np.tile(Y0_DEFAULT, (n_ctrl, 1))
    u_init = jnp.array([0.85, 0.50], dtype=jnp.float64)

    def assemble_params(u_vec):
        u_pump_cmd = u_vec[0] * (0.8 + 0.25 * pulse_disturbances)
        u_ext_cmd = jnp.broadcast_to(jnp.clip(u_vec[1], 0.1, 1.5), (n_ctrl,))
        return jnp.stack([u_pump_cmd, u_ext_cmd, pulse_disturbances, jnp.full((n_ctrl,), 18.0, dtype=jnp.float64)], axis=1)

    def eval_true_ctrl_loss(u_vec):
        yf = np.asarray(gradsolve.solve(
            problem, y0_ctrl, np.asarray(assemble_params(u_vec)),
            engine="rodas5p_replay", rtol=1e-8, atol=1e-10, device=target_device,
        ).y_final)
        u_np = np.asarray(u_vec)
        return float(np.mean((yf[:, 1] - 775.0) ** 2) - 25.0 * np.mean(yf[:, 5]) + 4.0 * u_np[0] ** 2 + 1.5 * u_np[1] ** 2)

    opt = optax.adam(0.05)
    u = u_init
    opt_state = opt.init(u)
    loss0 = eval_true_ctrl_loss(u)
    for _ in range(4):
        ctrl_closure = gradsolve.grad_closure(problem, y0_ctrl, assemble_params(u), engine="rodas5p_replay", rtol=1e-7, atol=1e-9, device=target_device)

        def step_loss(u_vec, _cl=ctrl_closure):
            yf = _cl(assemble_params(u_vec))
            return jnp.mean((yf[:, 1] - 775.0) ** 2) - 25.0 * jnp.mean(yf[:, 5]) + 4.0 * u_vec[0] ** 2 + 1.5 * u_vec[1] ** 2

        vg = jax.jit(jax.value_and_grad(step_loss))
        for _ in range(10):
            _, g = vg(u)
            updates, opt_state = opt.update(g, opt_state, u)
            u = optax.apply_updates(u, updates)
    loss_final = eval_true_ctrl_loss(u)
    print(f"[Stage 5] Differentiable control (40 Adam steps over {n_ctrl} pulses): "
          f"true adaptive loss {loss0:.3f} -> {loss_final:.3f}, optimal u={np.asarray(u).round(4).tolist()}")
    assert loss_final < loss0

    print("SMOKE_PIPELINE_ALL_5_GATES_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
