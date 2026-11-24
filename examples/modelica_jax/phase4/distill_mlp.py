"""Distill differentiable MPC / SHAC into an ultra-compact (<1,000-parameter) real-time
CPU MLP controller and benchmark deterministic single-sample inference latency (< 1 ms).
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from plant_env import (
    DT_CTRL,
    HORIZON,
    N_GETTER_TARGET,
    Scenario,
    T2_TARGET,
    Y0_PLANT,
    build_obs,
    count_params,
    init_policy_params,
    policy_forward,
    rollout_batch,
    sample_scenarios,
    step_plant_one,
)


def solve_differentiable_mpc_expert(scen_batch: Scenario, n_iters: int = 80, lr: float = 0.06):
    """Solve open-loop optimal action trajectories `u*(t)` across `B` scenarios via exact plant gradients."""
    b_size = scen_batch.t_in_base.shape[0]
    u_logits = jnp.zeros((b_size, HORIZON, 3), dtype=jnp.float64)

    def traj_loss_one(logits_seq, scen_one):
        u_seq = jax.nn.sigmoid(logits_seq)
        u0 = jnp.array([0.55, 0.45, 0.50], dtype=jnp.float64)

        def body(carry, step_idx):
            y, prev_u = carry
            idx_i = step_idx.astype(jnp.int32)
            u = u_seq[idx_i]
            t_sec = step_idx * DT_CTRL
            obs = build_obs(y, t_sec, scen_one, prev_u)
            y_next, cost, _, _ = step_plant_one(y, u, prev_u, step_idx, scen_one)
            return (y_next, u), (cost, obs, u)

        (y_end, _), (costs, obs_seq, act_seq) = jax.lax.scan(
            body, (Y0_PLANT, u0), jnp.arange(HORIZON, dtype=jnp.float64)
        )
        term = 0.8 * ((y_end[14] - N_GETTER_TARGET) / 0.25) ** 2
        return jnp.sum(costs) + term, (obs_seq, act_seq)

    batch_obj = lambda lg: jnp.mean(jax.vmap(lambda l1, s1: traj_loss_one(l1, s1)[0])(lg, scen_batch))
    opt = optax.adam(lr)
    opt_state = opt.init(u_logits)

    @jax.jit
    def mpc_step(lg, ost):
        lv, gr = jax.value_and_grad(batch_obj)(lg)
        upd, next_ost = opt.update(gr, ost, lg)
        return optax.apply_updates(lg, upd), next_ost, lv

    for _ in range(n_iters):
        u_logits, opt_state, _ = mpc_step(u_logits, opt_state)

    losses, (obs_all, act_all) = jax.jit(jax.vmap(traj_loss_one))(u_logits, scen_batch)
    return float(jnp.mean(losses)), obs_all.reshape(-1, 9), act_all.reshape(-1, 3)


def distill_compact_realtime_mlp(
    eval_scen: Scenario,
    *,
    hidden: int = 16,
    n_expert_scenarios: int = 1024,
    bc_steps: int = 400,
    finetune_steps: int = 100,
    seed: int = 303,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Distill a 483-parameter (`9 -> 16 -> 16 -> 3`) real-time MLP and verify < 1 ms CPU latency."""
    key = jax.random.PRNGKey(seed)
    k_exp, k_init, k_ft = jax.random.split(key, 3)

    t0 = time.perf_counter()
    expert_scen = sample_scenarios(k_exp, n_expert_scenarios)
    mpc_mean_cost, obs_dataset, act_dataset = solve_differentiable_mpc_expert(expert_scen)

    student = init_policy_params(k_init, hidden=hidden)
    opt_bc = optax.adam(5.0e-3)
    st_bc = opt_bc.init(student)

    @jax.jit
    def bc_step(p, ost):
        def mse(cp):
            pred = jax.vmap(policy_forward, in_axes=(None, 0))(cp, obs_dataset)
            return jnp.mean((pred - act_dataset) ** 2)

        lv, gr = jax.value_and_grad(mse)(p)
        upd, next_ost = opt_bc.update(gr, ost, p)
        return optax.apply_updates(p, upd), next_ost, lv

    for _ in range(bc_steps):
        student, st_bc, bc_mse = bc_step(student, st_bc)

    # Short closed-loop differentiable fine-tune to eliminate covariate shift (DAgger-by-backprop!)
    opt_ft = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1.5e-3))
    st_ft = opt_ft.init(student)

    @jax.jit
    def ft_step(p, ost, sk):
        sc = sample_scenarios(sk, 256)
        lv, gr = jax.value_and_grad(lambda cp: jnp.mean(rollout_batch(cp, sc)[0]))(p)
        upd, next_ost = opt_ft.update(gr, ost, p)
        return optax.apply_updates(p, upd), next_ost, lv

    for _ in range(finetune_steps):
        k_ft, sk = jax.random.split(k_ft)
        student, st_ft, _ = ft_step(student, st_ft, sk)

    distill_s = time.perf_counter() - t0
    ec, ermse, et1, edrift = rollout_batch(student, eval_scen)
    jax.block_until_ready(ec)

    # Export pure standalone NumPy weights & measure single-sample deterministic CPU latency
    np_weights = {k: np.asarray(v, dtype=np.float64) for k, v in student.items()}

    def cpu_infer_single(obs_vec: np.ndarray) -> np.ndarray:
        h1 = np.tanh(obs_vec @ np_weights["w1"] + np_weights["b1"])
        h2 = np.tanh(h1 @ np_weights["w2"] + np_weights["b2"])
        z = h2 @ np_weights["w3"] + np_weights["b3"]
        return 1.0 / (1.0 + np.exp(-z))

    # Verify numerical equivalence between JAX GPU policy and exported standalone CPU MLP
    sample_obs_np = np.asarray(obs_dataset[:32])
    jax_actions = np.asarray(jax.vmap(policy_forward, in_axes=(None, 0))(student, obs_dataset[:32]))
    cpu_actions = np.stack([cpu_infer_single(sample_obs_np[i]) for i in range(32)])
    export_max_diff = float(np.max(np.abs(jax_actions - cpu_actions)))

    # Warm up CPU caches, then measure 10,000 sequential real-time control ticks
    dummy = sample_obs_np[0].copy()
    for _ in range(200):
        _ = cpu_infer_single(dummy)

    latencies_us = np.empty(10000, dtype=np.float64)
    for i in range(10000):
        dummy[0] = 0.01 * (i % 17)
        t_tick = time.perf_counter_ns()
        _ = cpu_infer_single(dummy)
        latencies_us[i] = (time.perf_counter_ns() - t_tick) / 1000.0

    return np_weights, {
        "architecture": f"9 -> {hidden} -> {hidden} -> 3",
        "param_count": count_params(student),
        "param_limit": 10000,
        "mpc_teacher_mean_cost": round(mpc_mean_cost, 4),
        "bc_final_mse": float(bc_mse),
        "distill_total_seconds": round(distill_s, 3),
        "eval_mean_cost": round(float(jnp.mean(ec)), 4),
        "eval_rmse_t2_K": round(float(jnp.mean(ermse)), 4),
        "eval_max_t1_K": round(float(jnp.max(et1)), 3),
        "max_tritium_drift": float(jnp.max(edrift)),
        "jax_vs_numpy_max_diff": export_max_diff,
        "cpu_latency_us": {
            "mean": round(float(np.mean(latencies_us)), 2),
            "p50": round(float(np.percentile(latencies_us, 50)), 2),
            "p99": round(float(np.percentile(latencies_us, 99)), 2),
            "p99_9": round(float(np.percentile(latencies_us, 99.9)), 2),
            "target_max_us": 1000.0,
        },
    }
