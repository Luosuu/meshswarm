"""Subprocess worker evaluating a candidate `solver_jax.py` against all four gates."""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("GRADSOLVE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cand_path = Path(payload["candidate"]).resolve()
    spec = importlib.util.spec_from_file_location("candidate_solver", cand_path)
    if spec is None or spec.loader is None:
        print(json.dumps({"ok": False, "stage": "compile", "reason": "could not load solver_jax.py"}))
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for fn in ("forward_ensemble", "differentiable_closure"):
        if not hasattr(mod, fn):
            print(json.dumps({"ok": False, "stage": "compile", "reason": f"missing `{fn}` in solver_jax.py"}))
            return

    y0_default = np.array(payload["y0_default"], dtype=np.float64)
    p_test = np.array(payload["test_cases"], dtype=np.float64)
    gold = np.array(payload["gold"], dtype=np.float64)
    obs_names = payload["obs_names"]
    param_names = payload["param_names"]
    n_test = len(p_test)
    y0_test = np.tile(y0_default, (n_test, 1))

    # Gate 1 & Gate 2: Compile & held-out trajectory parity against OpenModelica
    obs_got = np.asarray(mod.forward_ensemble(y0_test, p_test))
    if obs_got.shape != gold.shape:
        print(json.dumps({
            "ok": False, "stage": "compile",
            "reason": f"forward_ensemble returned shape {obs_got.shape}, expected {gold.shape}",
        }))
        return
    if obs_got.dtype != np.float64:
        print(json.dumps({
            "ok": False, "stage": "compile",
            "reason": f"forward_ensemble returned dtype {obs_got.dtype}, expected float64",
        }))
        return
    if not np.all(np.isfinite(obs_got)):
        print(json.dumps({"ok": False, "stage": "compile", "reason": "forward_ensemble produced NaN or Inf"}))
        return

    denom = np.maximum(np.abs(gold), payload["traj_atol"])
    rel_mat = np.abs(obs_got - gold) / denom
    worst_idx = np.unravel_index(int(np.argmax(rel_mat)), rel_mat.shape)
    max_traj_err = float(rel_mat[worst_idx])
    if max_traj_err > payload["traj_rtol"]:
        ci, oi = int(worst_idx[0]), int(worst_idx[1])
        print(json.dumps({
            "ok": False,
            "stage": "traj_gold",
            "reason": (f"held-out case {ci} `{obs_names[oi]}` got {obs_got[ci, oi]:.6g} vs "
                       f"OpenModelica {gold[ci, oi]:.6g} (rel err {max_traj_err:.3e} > {payload['traj_rtol']:.1e})"),
        }))
        return

    # Gate 3: 1,000-trajectory GPU ensemble conservation + warm timing
    n_ens = int(payload["n_ensemble"])
    rng = np.random.default_rng(20261123)
    lo = np.array(payload["bounds_lo"], dtype=np.float64)
    hi = np.array(payload["bounds_hi"], dtype=np.float64)
    p_ens = rng.uniform(lo, hi, size=(n_ens, len(lo)))
    y0_ens = np.tile(y0_default, (n_ens, 1))

    out_cold = mod.forward_ensemble(y0_ens, p_ens)
    jax.block_until_ready(out_cold)
    ens_np = np.asarray(out_cold)
    if not np.all(np.isfinite(ens_np)):
        print(json.dumps({"ok": False, "stage": "conservation", "reason": "1,000-ensemble produced NaN or Inf"}))
        return

    ckind = payload["conservation_kind"]
    if ckind == "sum_last3":
        init_tot = float(np.sum(y0_default[-3:]))
        got_tot = np.sum(ens_np[:, -3:], axis=1)
        cons_err = float(np.max(np.abs(got_tot - init_tot) / init_tot))
    elif ckind == "hdt_column16":
        m_hold = np.array(payload["stage_holdups"], dtype=np.float64)
        xh2_0 = y0_default[:16]
        xhd_0 = y0_default[16:]
        xt2_0 = 1.0 - xh2_0 - xhd_0
        want_tots = np.array([np.sum(m_hold * xh2_0), np.sum(m_hold * xhd_0), np.sum(m_hold * xt2_0)])
        got_tots = ens_np[:, -3:]
        cons_err = float(np.max(np.abs(got_tots - want_tots[None, :]) / want_tots[None, :]))
    else:
        raise ValueError(f"unknown conservation_kind {ckind}")

    if cons_err >= 1.0e-11:
        print(json.dumps({
            "ok": False,
            "stage": "conservation",
            "reason": f"molar conservation drift {cons_err:.3e} >= 1e-11 across {n_ens} trajectories",
        }))
        return

    fwd_times = []
    for _ in range(int(payload["repeats"])):
        t0 = time.perf_counter()
        o = mod.forward_ensemble(y0_ens, p_ens)
        jax.block_until_ready(o)
        fwd_times.append(time.perf_counter() - t0)
    t_fwd_med = float(statistics.median(fwd_times))

    # Gate 4: Reverse-mode sensitivity checks (JVP-VJP < 1e-10, closure FD < 5e-5, fresh FD < 1.5e-2)
    cl = mod.differentiable_closure(y0_test, p_test)
    p_test_j = jnp.asarray(p_test, dtype=jnp.float64)
    obs_cl = np.asarray(cl(p_test_j))
    cl_consistency = float(np.max(np.abs(obs_cl - obs_got) / denom))
    if not np.all(np.isfinite(obs_cl)) or cl_consistency > 1.0e-5:
        print(json.dumps({
            "ok": False,
            "stage": "adjoint",
            "reason": f"differentiable_closure output differs from forward_ensemble (rel diff {cl_consistency:.3e})",
        }))
        return

    obs_scale = jnp.asarray(np.maximum(np.mean(np.abs(gold), axis=0), 1.0e-3), dtype=jnp.float64)
    weights = jnp.linspace(1.1, 2.3, gold.shape[1], dtype=jnp.float64)

    def row_obj_fn(p_mat):
        pred = cl(p_mat) / obs_scale[None, :]
        return jnp.sum(weights[None, :] * (pred ** 2 + 0.35 * pred), axis=1)

    def obj_fn(p_mat):
        return jnp.sum(row_obj_fn(p_mat))

    grad_jit = jax.jit(jax.grad(obj_fn))
    row_obj_jit = jax.jit(row_obj_fn)
    ad_grad = np.asarray(grad_jit(p_test_j))
    if not np.all(np.isfinite(ad_grad)):
        print(json.dumps({"ok": False, "stage": "adjoint", "reason": "jax.grad returned NaN or Inf"}))
        return

    jvp_grad = np.zeros_like(ad_grad)
    jvp_fn = jax.jit(lambda pm, tang: jax.jvp(row_obj_fn, (pm,), (tang,))[1])
    for j in range(p_test.shape[1]):
        tang = np.zeros_like(p_test)
        tang[:, j] = 1.0
        jvp_grad[:, j] = np.asarray(jvp_fn(p_test_j, jnp.asarray(tang)))

    p_scales = np.maximum(np.abs(p_test), 0.1)
    obs_scale_np = np.asarray(obs_scale)
    weights_np = np.asarray(weights)
    cl_fd = np.zeros_like(ad_grad)
    h_cl_mat = 1.0e-5 * p_scales
    for j in range(p_test.shape[1]):
        dp = np.zeros_like(p_test)
        dp[:, j] = h_cl_mat[:, j]
        fp = np.asarray(row_obj_jit(jnp.asarray(p_test + dp)))
        fm = np.asarray(row_obj_jit(jnp.asarray(p_test - dp)))
        cl_fd[:, j] = (fp - fm) / (2.0 * h_cl_mat[:, j])

    plus_rows, minus_rows, h_fr_list = [], [], []
    for i in range(n_test):
        for j in range(p_test.shape[1]):
            h_fr = 1.0e-3 * p_scales[i, j]
            row_p = p_test[i].copy()
            row_m = p_test[i].copy()
            row_p[j] += h_fr
            row_m[j] -= h_fr
            plus_rows.append(row_p)
            minus_rows.append(row_m)
            h_fr_list.append(h_fr)

    p_pert = np.vstack([np.array(plus_rows), np.array(minus_rows)])
    y0_pert = np.tile(y0_default, (len(p_pert), 1))
    obs_pert = np.asarray(mod.forward_ensemble(y0_pert, p_pert)) / obs_scale_np[None, :]
    row_losses = np.sum(weights_np[None, :] * (obs_pert ** 2 + 0.35 * obs_pert), axis=1)
    half = len(plus_rows)
    fresh_fd = ((row_losses[:half] - row_losses[half:]) / (2.0 * np.array(h_fr_list))).reshape(ad_grad.shape)

    grad_floor = np.maximum(np.max(np.abs(fresh_fd), axis=0, keepdims=True) * 5.0e-3, 1.0e-5)
    rel_jvp_err = float(np.max(np.abs(ad_grad - jvp_grad) / np.maximum(np.abs(jvp_grad), grad_floor)))
    rel_cl_err = float(np.max(np.abs(ad_grad - cl_fd) / np.maximum(np.abs(cl_fd), grad_floor)))
    rel_fresh_err = float(np.max(np.abs(ad_grad - fresh_fd) / np.maximum(np.abs(fresh_fd), grad_floor)))

    if rel_jvp_err >= 1.0e-10:
        print(json.dumps({
            "ok": False,
            "stage": "adjoint",
            "reason": f"reverse-mode VJP vs forward-mode JVP rel error {rel_jvp_err:.3e} >= 1e-10",
        }))
        return
    if rel_cl_err >= 5.0e-5:
        print(json.dumps({
            "ok": False,
            "stage": "adjoint",
            "reason": f"reverse-mode grad vs closure FD rel error {rel_cl_err:.3e} >= 5e-5",
        }))
        return
    if rel_fresh_err >= 1.5e-2:
        wi, wj = np.unravel_index(
            int(np.argmax(np.abs(ad_grad - fresh_fd) / np.maximum(np.abs(fresh_fd), grad_floor))),
            ad_grad.shape,
        )
        print(json.dumps({
            "ok": False,
            "stage": "adjoint",
            "reason": (f"reverse-mode grad vs fresh forward FD rel error {rel_fresh_err:.3e} >= 1.5e-2 "
                       f"(case {int(wi)} param `{param_names[int(wj)]}`: AD={ad_grad[wi, wj]:.5g}, FD={fresh_fd[wi, wj]:.5g})"),
        }))
        return

    # Benchmark 64-trajectory reverse-mode gradient pass
    n_g = min(64, n_ens)
    y0_g = y0_ens[:n_g]
    p_g = p_ens[:n_g]
    cl_g = mod.differentiable_closure(y0_g, p_g)
    p_g_j = jnp.asarray(p_g, dtype=jnp.float64)

    def obj_g(pm):
        pred = cl_g(pm) / obs_scale[None, :]
        return jnp.sum(weights[None, :] * (pred ** 2 + 0.35 * pred))

    g_fn = jax.jit(jax.grad(obj_g))
    jax.block_until_ready(g_fn(p_g_j))
    grad_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        gg = g_fn(p_g_j)
        jax.block_until_ready(gg)
        grad_times.append(time.perf_counter() - t0)
    t_grad_med = float(statistics.median(grad_times))

    print(json.dumps({
        "ok": True,
        "device": jax.devices()[0].platform,
        "max_traj_err": max_traj_err,
        "cons_err": cons_err,
        "rel_cl_err": rel_cl_err,
        "rel_fresh_err": rel_fresh_err,
        "t_fwd_med": t_fwd_med,
        "t_grad_med": t_grad_med,
        "t_combined": t_fwd_med + t_grad_med,
    }))


if __name__ == "__main__":
    main()
