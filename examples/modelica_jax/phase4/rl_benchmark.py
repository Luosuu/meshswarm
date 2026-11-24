"""Compare first-order differentiable control (SHAC / Short-Horizon Analytic Gradient)
against model-free PPO over 1,000,000+ GPU plant episodes on the Swarm-generated
ORNL TRANSFORM PbLi/FLiBe HX + Tritium Permeator operator.
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
    Y0_PLANT,
    build_obs,
    count_params,
    init_policy_params,
    policy_forward,
    rollout_batch,
    sample_scenarios,
    step_plant_one,
)


def evaluate_constant_open_loop(eval_scen: Scenario) -> dict[str, float]:
    """Fixed nominal command `u = [0.55, 0.45, 0.50]` with zero feedback."""
    zero_params = {
        "w1": jnp.zeros((9, 32), dtype=jnp.float64),
        "b1": jnp.zeros((32,), dtype=jnp.float64),
        "w2": jnp.zeros((32, 32), dtype=jnp.float64),
        "b2": jnp.zeros((32,), dtype=jnp.float64),
        "w3": jnp.zeros((32, 3), dtype=jnp.float64),
        "b3": jax.scipy.special.logit(jnp.array([0.55, 0.45, 0.50], dtype=jnp.float64)),
    }
    costs, rmse_t2, peak_t1, drift = rollout_batch(zero_params, eval_scen)
    return {
        "mean_cost": float(jnp.mean(costs)),
        "rmse_t2_K": float(jnp.mean(rmse_t2)),
        "max_t1_K": float(jnp.max(peak_t1)),
        "max_tritium_drift": float(jnp.max(drift)),
    }


def train_shac_analytic(
    eval_scen: Scenario,
    *,
    n_updates: int = 200,
    batch_envs: int = 256,
    window: int = 30,
    lr: float = 5.0e-3,
    seed: int = 101,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Differentiable Physics Policy Optimization (exact analytic policy gradient through plant rollouts)."""
    key = jax.random.PRNGKey(seed)
    key, init_k = jax.random.split(key)
    params = init_policy_params(init_k)
    sched = optax.cosine_decay_schedule(lr, n_updates, alpha=0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = opt.init(params)

    def windowed_loss_one(p, scen_one):
        u0 = jnp.array([0.55, 0.45, 0.50], dtype=jnp.float64)

        def body(carry, step_idx):
            y, prev_u = carry
            # SHAC-style short-horizon gradient detachment at window boundaries keeps Jacobians well-conditioned
            y_in = jnp.where((step_idx > 0) & (step_idx % window == 0), jax.lax.stop_gradient(y), y)
            t_sec = step_idx * DT_CTRL
            obs = build_obs(y_in, t_sec, scen_one, prev_u)
            u = policy_forward(p, obs)
            y_next, cost, _, _ = step_plant_one(y_in, u, prev_u, step_idx, scen_one)
            return (y_next, u), cost

        (y_end, _), costs = jax.lax.scan(body, (Y0_PLANT, u0), jnp.arange(HORIZON, dtype=jnp.float64))
        term = 0.8 * ((y_end[14] - N_GETTER_TARGET) / 0.25) ** 2
        return jnp.sum(costs) + term

    batch_loss_fn = lambda p, sc: jnp.mean(jax.vmap(windowed_loss_one, in_axes=(None, 0))(p, sc))

    @jax.jit
    def update_step(p, o_state, step_key):
        sc = sample_scenarios(step_key, batch_envs)
        loss_val, grads = jax.value_and_grad(batch_loss_fn)(p, sc)
        updates, next_o = opt.update(grads, o_state, p)
        return optax.apply_updates(p, updates), next_o, loss_val

    # Warm-up JIT compile, then time pure GPU training
    key, wk = jax.random.split(key)
    p_warm, _, l_warm = update_step(params, opt_state, wk)
    jax.block_until_ready(l_warm)
    del p_warm

    history = []
    t0 = time.perf_counter()
    for it in range(1, n_updates + 1):
        key, sk = jax.random.split(key)
        params, opt_state, l_val = update_step(params, opt_state, sk)
        if it in (1, 25, 50, 100, n_updates):
            jax.block_until_ready(l_val)
            ec, ermse, et1, edrift = rollout_batch(params, eval_scen)
            jax.block_until_ready(ec)
            history.append({
                "update": it,
                "episodes": it * batch_envs,
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "eval_cost": round(float(jnp.mean(ec)), 4),
                "eval_rmse_t2_K": round(float(jnp.mean(ermse)), 4),
            })
    total_s = time.perf_counter() - t0
    ec, ermse, et1, edrift = rollout_batch(params, eval_scen)
    return params, {
        "method": "SHAC / Windowed Differentiable Physics Policy",
        "updates": n_updates,
        "total_episodes": n_updates * batch_envs,
        "train_seconds": round(total_s, 3),
        "episodes_per_sec": round((n_updates * batch_envs) / max(total_s, 1e-6), 1),
        "final_mean_cost": round(float(jnp.mean(ec)), 4),
        "final_rmse_t2_K": round(float(jnp.mean(ermse)), 4),
        "final_max_t1_K": round(float(jnp.max(et1)), 3),
        "max_tritium_drift": float(jnp.max(edrift)),
        "param_count": count_params(params),
        "curve": history,
    }


def _init_ppo_params(key: jax.Array, hidden: int = 32) -> dict[str, jax.Array]:
    k_pol, kv1, kv2, kv3 = jax.random.split(key, 4)
    pol = init_policy_params(k_pol, hidden=hidden)
    return {
        **pol,
        "log_std": jnp.full((3,), -1.2, dtype=jnp.float64),
        "vw1": jax.random.normal(kv1, (9, hidden), dtype=jnp.float64) * jnp.sqrt(2.0 / 9.0),
        "vb1": jnp.zeros((hidden,), dtype=jnp.float64),
        "vw2": jax.random.normal(kv2, (hidden, hidden), dtype=jnp.float64) * jnp.sqrt(2.0 / hidden),
        "vb2": jnp.zeros((hidden,), dtype=jnp.float64),
        "vw3": jax.random.normal(kv3, (hidden, 1), dtype=jnp.float64) * 0.05,
        "vb3": jnp.zeros((1,), dtype=jnp.float64),
    }


def _value_forward(params: dict[str, jax.Array], obs: jax.Array) -> jax.Array:
    h1 = jnp.tanh(obs @ params["vw1"] + params["vb1"])
    h2 = jnp.tanh(h1 @ params["vw2"] + params["vb2"])
    return jnp.squeeze(h2 @ params["vw3"] + params["vb3"], axis=-1)


def train_ppo_1m_episodes(
    eval_scen: Scenario,
    *,
    n_updates: int = 500,
    batch_envs: int = 2048,
    lr: float = 1.5e-3,
    seed: int = 202,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Model-free PPO across `500 * 2048 = 1,024,000` full 30-second plant episodes on GPU."""
    key = jax.random.PRNGKey(seed)
    key, ik = jax.random.split(key)
    params = _init_ppo_params(ik)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = opt.init(params)

    gamma = 0.99
    gae_lam = 0.95
    clip_eps = 0.2

    def collect_one(p, scen_one, ep_key):
        u0 = jnp.array([0.55, 0.45, 0.50], dtype=jnp.float64)
        std = jnp.exp(p["log_std"])
        step_keys = jax.random.split(ep_key, HORIZON)

        def body(carry, inputs):
            y, prev_u = carry
            step_idx, sk = inputs
            t_sec = step_idx * DT_CTRL
            obs = build_obs(y, t_sec, scen_one, prev_u)
            mean_u = policy_forward(p, obs)
            noise = jax.random.normal(sk, (3,), dtype=jnp.float64)
            u_raw = mean_u + std * noise
            logp = jnp.sum(-0.5 * ((u_raw - mean_u) / std) ** 2 - p["log_std"] - 0.5 * jnp.log(2.0 * jnp.pi))
            val = _value_forward(p, obs)
            # Model-free RL treats plant transition as a black box (`stop_gradient`)
            y_next, cost, _, _ = jax.lax.stop_gradient(step_plant_one(y, u_raw, prev_u, step_idx, scen_one))
            u_applied = jnp.clip(u_raw, 0.0, 1.0)
            return (y_next, u_applied), (obs, u_raw, logp, -cost, val)

        (y_end, _), (obs_tr, act_tr, logp_tr, rew_tr, val_tr) = jax.lax.scan(
            body, (Y0_PLANT, u0), (jnp.arange(HORIZON, dtype=jnp.float64), step_keys)
        )
        term_rew = -0.8 * ((y_end[14] - N_GETTER_TARGET) / 0.25) ** 2
        rew_tr = rew_tr.at[-1].add(jax.lax.stop_gradient(term_rew))

        def gae_body(gae, idx_rev):
            r, v, v_next = idx_rev
            delta = r + gamma * v_next - v
            gae_new = delta + gamma * gae_lam * gae
            return gae_new, gae_new

        v_next_tr = jnp.concatenate([val_tr[1:], jnp.zeros((1,), dtype=jnp.float64)])
        _, adv_rev = jax.lax.scan(
            gae_body, jnp.array(0.0, dtype=jnp.float64), (rew_tr[::-1], val_tr[::-1], v_next_tr[::-1])
        )
        adv_tr = adv_rev[::-1]
        ret_tr = adv_tr + val_tr
        return obs_tr, act_tr, logp_tr, adv_tr, ret_tr

    @jax.jit
    def ppo_update(p, o_state, step_key):
        k_sc, k_ep = jax.random.split(step_key)
        sc = sample_scenarios(k_sc, batch_envs)
        ep_keys = jax.random.split(k_ep, batch_envs)
        obs_b, act_b, old_lp_b, adv_b, ret_b = jax.vmap(collect_one, in_axes=(None, 0, 0))(p, sc, ep_keys)

        adv_norm = (adv_b - jnp.mean(adv_b)) / (jnp.std(adv_b) + 1.0e-8)

        def surrogate_loss(curr_p):
            std = jnp.exp(curr_p["log_std"])
            means = jax.vmap(jax.vmap(policy_forward, in_axes=(None, 0)), in_axes=(None, 0))(curr_p, obs_b)
            vals = jax.vmap(jax.vmap(_value_forward, in_axes=(None, 0)), in_axes=(None, 0))(curr_p, obs_b)
            new_lp = jnp.sum(-0.5 * ((act_b - means) / std) ** 2 - curr_p["log_std"] - 0.5 * jnp.log(2.0 * jnp.pi), axis=-1)
            ratio = jnp.exp(new_lp - old_lp_b)
            pg1 = ratio * adv_norm
            pg2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
            pol_loss = -jnp.mean(jnp.minimum(pg1, pg2))
            v_loss = 0.5 * jnp.mean((vals - ret_b) ** 2)
            ent = jnp.sum(curr_p["log_std"] + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))
            return pol_loss + 0.25 * v_loss - 0.002 * ent

        def epoch_step(carry, _):
            cp, co = carry
            lv, gr = jax.value_and_grad(surrogate_loss)(cp)
            upd, next_co = opt.update(gr, co, cp)
            return (optax.apply_updates(cp, upd), next_co), lv

        (p_next, o_next), lvs = jax.lax.scan(epoch_step, (p, o_state), None, length=4)
        return p_next, o_next, lvs[-1]

    # Warm-up JIT compilation
    key, wk = jax.random.split(key)
    pw, _, lw = ppo_update(params, opt_state, wk)
    jax.block_until_ready(lw)
    del pw

    history = []
    t0 = time.perf_counter()
    for it in range(1, n_updates + 1):
        key, sk = jax.random.split(key)
        params, opt_state, lv = ppo_update(params, opt_state, sk)
        if it in (1, 50, 150, 300, n_updates):
            jax.block_until_ready(lv)
            pol_only = {k: params[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")}
            ec, ermse, _, _ = rollout_batch(pol_only, eval_scen)
            jax.block_until_ready(ec)
            history.append({
                "update": it,
                "episodes": it * batch_envs,
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "eval_cost": round(float(jnp.mean(ec)), 4),
                "eval_rmse_t2_K": round(float(jnp.mean(ermse)), 4),
            })
    total_s = time.perf_counter() - t0
    pol_final = {k: params[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")}
    ec, ermse, et1, edrift = rollout_batch(pol_final, eval_scen)
    return pol_final, {
        "method": "Model-Free PPO (Black-Box Rollouts + GAE)",
        "updates": n_updates,
        "total_episodes": n_updates * batch_envs,
        "train_seconds": round(total_s, 3),
        "episodes_per_sec": round((n_updates * batch_envs) / max(total_s, 1e-6), 1),
        "final_mean_cost": round(float(jnp.mean(ec)), 4),
        "final_rmse_t2_K": round(float(jnp.mean(ermse)), 4),
        "final_max_t1_K": round(float(jnp.max(et1)), 3),
        "max_tritium_drift": float(jnp.max(edrift)),
        "curve": history,
    }
