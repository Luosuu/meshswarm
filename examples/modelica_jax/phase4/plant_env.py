"""Vectorized differentiable Fusion Blanket + TRANSFORM PbLi/FLiBe HX control environment.

Wraps the Phase 3 Swarm-generated `10.0/10` solver (`results/modelica_jax/tier2_transform_hx/gen-2/run-1/solver_jax.py`)
into a 30-step (30 s, dt_ctrl = 1.0 s) closed-loop control benchmark subject to stochastic
plasma power ramps, ELM thermal bursts, and partial primary PbLi coolant pump trips.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[3]
SWARM_TIER2_SOLVER = (
    REPO_ROOT / "results" / "modelica_jax" / "tier2_transform_hx" / "gen-2" / "run-1" / "solver_jax.py"
)


def _load_swarm_tier2():
    spec = importlib.util.spec_from_file_location("swarm_tier2_winner", SWARM_TIER2_SOLVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SWARM_TIER2_SOLVER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_TIER2 = _load_swarm_tier2()

# Control horizon: 30 control steps of 1.0 s each (4 RK4 sub-steps of 0.25 s per control interval)
HORIZON = 30
DT_CTRL = 1.0
SUBSTEPS = 4
DT_SUB = DT_CTRL / SUBSTEPS

T2_TARGET = 892.0       # Target secondary FLiBe turbine-inlet temperature (K)
T1_SOFT_MAX = 948.0     # Structural creep soft ceiling on primary PbLi outlet (K)
N_GETTER_TARGET = 1.25  # Target recovered tritium in getter bed at t=30s (mol)

Y0_PLANT = jnp.array([930.0] * 6 + [880.0] * 6 + [5.0, 0.5, 0.5], dtype=jnp.float64)


class Scenario(NamedTuple):
    t_in_base: jax.Array     # (B,) baseline blanket PbLi inlet temp [935..970 K]
    ramp_amp: jax.Array      # (B,) plasma ramp delta-T [0..35 K]
    elm_amp: jax.Array       # (B,) ELM transient thermal spike [0..28 K]
    elm_time: jax.Array      # (B,) ELM center time [6..24 s]
    trip_time: jax.Array     # (B,) primary pump partial-trip time [8..20 s]
    trip_retain: jax.Array   # (B,) fraction of primary pump flow retained post-trip [0.45..0.90]


def sample_scenarios(key: jax.Array, batch_size: int) -> Scenario:
    keys = jax.random.split(key, 6)
    return Scenario(
        t_in_base=jax.random.uniform(keys[0], (batch_size,), minval=938.0, maxval=968.0),
        ramp_amp=jax.random.uniform(keys[1], (batch_size,), minval=5.0, maxval=32.0),
        elm_amp=jax.random.uniform(keys[2], (batch_size,), minval=0.0, maxval=25.0),
        elm_time=jax.random.uniform(keys[3], (batch_size,), minval=7.0, maxval=22.0),
        trip_time=jax.random.uniform(keys[4], (batch_size,), minval=8.0, maxval=19.0),
        trip_retain=jax.random.uniform(keys[5], (batch_size,), minval=0.45, maxval=0.90),
    )


def build_obs(y: jax.Array, t_sec: jax.Array, scen_one: Scenario, prev_u: jax.Array) -> jax.Array:
    """9-dimensional normalized sensor observation for real-time plant control."""
    t1_in_now = (
        scen_one.t_in_base
        + scen_one.ramp_amp * jnp.tanh(t_sec / 8.0)
        + scen_one.elm_amp * jnp.exp(-0.5 * ((t_sec - scen_one.elm_time) / 2.0) ** 2)
    )
    pump_health = 1.0 - (1.0 - scen_one.trip_retain) * 0.5 * (1.0 + jnp.tanh((t_sec - scen_one.trip_time) / 0.5))
    return jnp.array([
        (y[0] - 930.0) / 30.0,            # PbLi HX inlet zone temp
        (y[5] - 910.0) / 30.0,            # PbLi HX outlet zone temp
        (y[11] - T2_TARGET) / 20.0,       # FLiBe HX outlet temp error from setpoint!
        (t1_in_now - 955.0) / 30.0,       # Measured blanket outlet / HX supply temp
        pump_health - 0.75,               # Primary pump tachometer health indicator
        (y[12] - 4.0) / 2.0,              # Loop tritium inventory
        (y[14] - 1.0) / 1.0,              # Getter recovered tritium inventory
        t_sec / 30.0,                     # Normalized pulse clock
        prev_u[1] - 0.5,                  # Previous normalized FLiBe valve/pump state
    ], dtype=jnp.float64)


def actions_to_physical(u_norm: jax.Array, t_sec: jax.Array, scen_one: Scenario) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Map normalized policy actions `u_norm in [0, 1]^3` to physical commands under pump trip."""
    pump_health = 1.0 - (1.0 - scen_one.trip_retain) * 0.5 * (1.0 + jnp.tanh((t_sec - scen_one.trip_time) / 0.5))
    m_pbli_cmd = (25.0 + 105.0 * u_norm[0]) * pump_health
    m_flibe_cmd = 3.5 + 18.5 * u_norm[1]
    u_vac_cmd = 0.20 + 1.30 * u_norm[2]
    t_in_eff = (
        scen_one.t_in_base
        + scen_one.ramp_amp * jnp.tanh(t_sec / 8.0)
        + scen_one.elm_amp * jnp.exp(-0.5 * ((t_sec - scen_one.elm_time) / 2.0) ** 2)
        - 18.0 * jnp.sin(0.2 * t_sec)  # cancel built-in 18*sin(0.2*t) in _TIER2._rhs so t_in is exact
    )
    return m_pbli_cmd, m_flibe_cmd, t_in_eff, u_vac_cmd


def step_plant_one(y: jax.Array, u_norm: jax.Array, prev_u: jax.Array, step_idx: jax.Array, scen_one: Scenario):
    """Advance one 15-state plant trajectory by 1.0 s using the Swarm's verified `_rk4_step`."""
    t0 = step_idx * DT_CTRL
    u_clipped = jnp.clip(u_norm, 0.0, 1.0)

    def sub_body(state, k):
        tk = t0 + k * DT_SUB
        m_pbli, m_flibe, t_in_eff, u_vac = actions_to_physical(u_clipped, tk, scen_one)
        p_vec = jnp.stack([m_pbli, m_flibe, t_in_eff, u_vac])
        next_state = _TIER2._rk4_step(tk, state, DT_SUB, p_vec)
        return next_state, None

    y_next, _ = jax.lax.scan(sub_body, y, jnp.arange(SUBSTEPS, dtype=jnp.float64))
    obs_vec = _TIER2._observe(y_next, jnp.zeros(4, dtype=jnp.float64))
    t1_out, t2_out, n_getter = obs_vec[1], obs_vec[2], obs_vec[5]

    # Step control loss: tight FLiBe turbine-inlet tracking + PbLi creep-ceiling guard + smooth actuation
    track_err = ((t2_out - T2_TARGET) / 5.0) ** 2
    over_t1 = jax.nn.softplus(t1_out - T1_SOFT_MAX) ** 2 * 0.25
    pump_power = 0.06 * (u_clipped[0] ** 2 + 1.2 * u_clipped[1] ** 2) + 0.03 * u_clipped[2] ** 2
    slew_cost = 0.08 * jnp.sum((u_clipped - prev_u) ** 2)
    trit_bonus = -0.04 * n_getter
    step_cost = track_err + over_t1 + pump_power + slew_cost + trit_bonus
    return y_next, step_cost, t2_out, t1_out


# Policy MLP architecture: 9 -> 32 -> 32 -> 3 (1,443 total parameters < 10,000 real-time PLC/CPU limit!)
def init_policy_params(key: jax.Array, hidden: int = 32) -> dict[str, jax.Array]:
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (9, hidden), dtype=jnp.float64) * jnp.sqrt(2.0 / 9.0),
        "b1": jnp.zeros((hidden,), dtype=jnp.float64),
        "w2": jax.random.normal(k2, (hidden, hidden), dtype=jnp.float64) * jnp.sqrt(2.0 / hidden),
        "b2": jnp.zeros((hidden,), dtype=jnp.float64),
        "w3": jax.random.normal(k3, (hidden, 3), dtype=jnp.float64) * 0.05,
        "b3": jnp.zeros((3,), dtype=jnp.float64),
    }


def count_params(params: dict[str, jax.Array]) -> int:
    return int(sum(v.size for v in params.values()))


def policy_forward(params: dict[str, jax.Array], obs: jax.Array) -> jax.Array:
    h1 = jnp.tanh(obs @ params["w1"] + params["b1"])
    h2 = jnp.tanh(h1 @ params["w2"] + params["b2"])
    return jax.nn.sigmoid(h2 @ params["w3"] + params["b3"])


def rollout_deterministic_one(params: dict[str, jax.Array], scen_one: Scenario):
    u0 = jnp.array([0.55, 0.45, 0.50], dtype=jnp.float64)

    def body(carry, step_idx):
        y, prev_u = carry
        t_sec = step_idx * DT_CTRL
        obs = build_obs(y, t_sec, scen_one, prev_u)
        u = policy_forward(params, obs)
        y_next, cost, t2_out, t1_out = step_plant_one(y, u, prev_u, step_idx, scen_one)
        return (y_next, u), (cost, t2_out, t1_out)

    (y_final, _), (costs, t2_traj, t1_traj) = jax.lax.scan(
        body, (Y0_PLANT, u0), jnp.arange(HORIZON, dtype=jnp.float64)
    )
    # Terminal tritium extraction target penalty
    term_cost = 0.8 * ((y_final[14] - N_GETTER_TARGET) / 0.25) ** 2
    total_cost = jnp.sum(costs) + term_cost
    trit_drift = jnp.abs(jnp.sum(y_final[12:15]) - jnp.sum(Y0_PLANT[12:15])) / jnp.sum(Y0_PLANT[12:15])
    rmse_t2 = jnp.sqrt(jnp.mean((t2_traj - T2_TARGET) ** 2))
    peak_t1 = jnp.max(t1_traj)
    return total_cost, rmse_t2, peak_t1, trit_drift


rollout_batch = jax.jit(jax.vmap(rollout_deterministic_one, in_axes=(None, 0)))
