"""Generic adaptive-replay baseline solver for Tier 3 (`CryoHDTColumn16.mo`)."""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("GRADSOLVE_X64", "1")

import gradsolve  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

N = 16
M_HOLDUP = jnp.array([4.0] + [1.0] * 14 + [6.0], dtype=jnp.float64)
INV_M = 1.0 / M_HOLDUP
STAGE_IDX = jnp.arange(N, dtype=jnp.float64)
V_FLOOD = 2.15


@jax.custom_jvp
def solve_bubble_temp(x_h2: jax.Array, x_hd: jax.Array, p_stage: jax.Array) -> jax.Array:
    x_t2 = 1.0 - x_h2 - x_hd
    t = jnp.full_like(p_stage, 11.5)
    for _ in range(9):
        inv = 1.0 / (t + 1.8)
        inv2 = inv * inv
        p1 = jnp.exp(9.45 - 106.0 * inv)
        p2 = jnp.exp(9.62 - 129.0 * inv)
        p3 = jnp.exp(9.85 - 156.0 * inv)
        f = x_h2 * p1 + x_hd * p2 + x_t2 * p3 - p_stage
        df = (x_h2 * p1 * 106.0 + x_hd * p2 * 129.0 + x_t2 * p3 * 156.0) * inv2
        t = t - f / df
    return t


@solve_bubble_temp.defjvp
def _bubble_jvp(primals, tangents):
    x_h2, x_hd, p_stage = primals
    dx_h2, dx_hd, dp_stage = tangents
    t = solve_bubble_temp(x_h2, x_hd, p_stage)
    x_t2 = 1.0 - x_h2 - x_hd
    inv = 1.0 / (t + 1.8)
    inv2 = inv * inv
    p1 = jnp.exp(9.45 - 106.0 * inv)
    p2 = jnp.exp(9.62 - 129.0 * inv)
    p3 = jnp.exp(9.85 - 156.0 * inv)
    df_dt = (x_h2 * p1 * 106.0 + x_hd * p2 * 129.0 + x_t2 * p3 * 156.0) * inv2
    dt = (dp_stage - (p1 - p3) * dx_h2 - (p2 - p3) * dx_hd) / df_dt
    return t, dt


def _rhs_single(t: jax.Array, y: jax.Array, p: jax.Array) -> jax.Array:
    v_boil, d_rec_frac, dp_tray, p_top = p[0], p[1], p[2], p[3]
    x_h2 = y[:N]
    x_hd = y[N:]

    v_eff = v_boil * (1.0 + 0.08 * jnp.sin(0.25 * t))
    dv = v_eff - V_FLOOD
    dv_pos = 0.5 * (dv + jnp.sqrt(dv * dv + 1.0e-8))
    eta = 0.92 / (1.0 + 1.8 * dv_pos * dv_pos)
    d_rec = d_rec_frac * v_eff
    l_rect = v_eff - d_rec

    p_stage = p_top + STAGE_IDX * dp_tray
    t_stage = solve_bubble_temp(x_h2, x_hd, p_stage)
    inv = 1.0 / (t_stage + 1.8)
    psat_h2 = jnp.exp(9.45 - 106.0 * inv)
    psat_hd = jnp.exp(9.62 - 129.0 * inv)

    yh2 = (1.0 - eta) * x_h2 + eta * x_h2 * psat_h2 / p_stage
    yhd = (1.0 - eta) * x_hd + eta * x_hd * psat_hd / p_stage

    def species_flux(x, yv):
        d_top = v_eff * yv[1] - v_eff * x[0]
        d_rect = l_rect * x[0:6] + v_eff * yv[2:8] - l_rect * x[1:7] - v_eff * yv[1:7]
        d_feed = l_rect * x[6] + d_rec * x[0] + v_eff * yv[8] - v_eff * x[7] - v_eff * yv[7]
        d_strip = v_eff * x[7:14] + v_eff * yv[9:16] - v_eff * x[8:15] - v_eff * yv[8:15]
        d_bot = v_eff * x[14] - v_eff * yv[15]
        return jnp.concatenate([jnp.array([d_top]), d_rect, jnp.array([d_feed]), d_strip, jnp.array([d_bot])]) * INV_M

    return jnp.concatenate([species_flux(x_h2, yh2), species_flux(x_hd, yhd)])


def _obs_single(y: jax.Array, p: jax.Array) -> jax.Array:
    x_h2 = y[:N]
    x_hd = y[N:]
    x_t2 = 1.0 - x_h2 - x_hd
    p_stage = p[3] + STAGE_IDX * p[2]
    t_stage = solve_bubble_temp(x_h2, x_hd, p_stage)
    tot_h2 = jnp.sum(M_HOLDUP * x_h2)
    tot_hd = jnp.sum(M_HOLDUP * x_hd)
    tot_t2 = jnp.sum(M_HOLDUP * x_t2)
    return jnp.array([x_h2[0], x_t2[15], t_stage[0], t_stage[15], tot_h2, tot_hd, tot_t2])


class _Tier3GenericProblem:
    name = "user_tier3_cryo_hdt_generic"
    dim = 32
    t0 = 0.0
    t1 = 25.0
    is_stiff = False

    def f_jax(self, t, y, p):
        return _rhs_single(t, y, p)


_PROB = _Tier3GenericProblem()
_DEV = "gpu" if any(d.platform == "gpu" for d in jax.devices()) else "cpu"
_OBS_BATCH = jax.jit(jax.vmap(_obs_single, in_axes=(0, 0)))


def forward_ensemble(y0_batch: np.ndarray, p_batch: np.ndarray) -> jax.Array:
    p_arr = np.asarray(p_batch, dtype=np.float64)
    res = gradsolve.solve(
        _PROB, np.asarray(y0_batch, dtype=np.float64), p_arr,
        engine="tsit5_replay", rtol=1e-8, atol=1e-10, device=_DEV,
    )
    return _OBS_BATCH(jnp.asarray(res.y_final), jnp.asarray(p_arr))


def differentiable_closure(y0_batch: np.ndarray, p_ref_batch: np.ndarray):
    raw_cl = gradsolve.grad_closure(
        _PROB, np.asarray(y0_batch, dtype=np.float64), np.asarray(p_ref_batch, dtype=np.float64),
        engine="tsit5_replay", rtol=1e-8, atol=1e-10, device=_DEV,
    )
    return lambda p_jax: _OBS_BATCH(raw_cl(p_jax), p_jax)
