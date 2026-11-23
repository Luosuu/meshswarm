"""Verified reference JAX implementations for all three Phase 1 benchmark tiers.

Key numerical lesson illustrated here: Rosenbrock-Wanner stiff solvers (`rodas5p_replay`)
compute per-step state Jacobians via `jax.jacfwd` and differentiate through those Jacobians
during the reverse-mode replay pass. Implicit algebraic solvers must therefore use
`@jax.custom_jvp` (which supports JVP, transposed VJP, and higher-order JVP-in-VJP
automatically via the Implicit Function Theorem), rather than `@jax.custom_vjp`.
"""

from __future__ import annotations

TIER1_REFERENCE = '''\
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("GRADSOLVE_X64", "1")

import gradsolve
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ALPHA_NL = 0.65
BETA_NL = 0.30
EPS_Z = 1.0e-7


@jax.custom_jvp
def solve_sieverts_flux(rhs_drive: jax.Array) -> jax.Array:
    z = rhs_drive / (1.0 + jnp.cbrt(rhs_drive * rhs_drive + 1.0e-12))
    for _ in range(7):
        az = jnp.sqrt(z * z + EPS_Z * EPS_Z)
        f = z + ALPHA_NL * z * az + BETA_NL * z * z * z - rhs_drive
        df = 1.0 + ALPHA_NL * (2.0 * z * z + EPS_Z * EPS_Z) / az + 3.0 * BETA_NL * z * z
        z = z - f / df
    return z


@solve_sieverts_flux.defjvp
def _sieverts_jvp(primals, tangents):
    (r,), (dr,) = primals, tangents
    z = solve_sieverts_flux(r)
    az = jnp.sqrt(z * z + EPS_Z * EPS_Z)
    df = 1.0 + ALPHA_NL * (2.0 * z * z + EPS_Z * EPS_Z) / az + 3.0 * BETA_NL * z * z
    return z, dr / df


class _Tier1Problem:
    name = "user_tier1_blanket_valve_dae"
    dim = 6
    t0 = 0.0
    t1 = 20.0
    is_stiff = True

    def f_jax(self, t, y, p):
        u_pump, u_ext, p_pulse, dp_back = p[0], p[1], p[2], p[3]
        q_flow, t_bl, t_hx, n_bl, n_ex, _n_st = y[0], y[1], y[2], y[3], y[4], y[5]

        p_fus = 115.0 * (1.0 + 0.40 * p_pulse * jnp.sin(0.20 * t))
        dp_drive = 2.0 * u_pump - dp_back * (1.0 + 0.25 * jnp.cos(0.35 * t))

        eps_q = 1.0e-5
        q_abs = jnp.sqrt(q_flow * q_flow + eps_q * eps_q)
        w_pos = 0.5 * (1.0 + q_flow / q_abs)
        drag_valve = (w_pos * 0.24 - (1.0 - w_pos) * 14.0) * q_flow * q_flow

        dp_abs = jnp.sqrt(dp_drive * dp_drive + 1.0e-10)
        w_dp = 0.5 * (1.0 + dp_drive / dp_abs)
        dp_pos = 0.5 * (dp_drive + dp_abs)
        dp_neg = 0.5 * (-dp_drive + dp_abs)
        q_byp = w_dp * 0.08 * jnp.sqrt(dp_pos + 1.0e-8) - (1.0 - w_dp) * 0.15 * 0.08 * jnp.sqrt(dp_neg + 1.0e-8)

        dq = (dp_drive - drag_valve - 0.30 * q_flow) / 0.02

        q_pos_part = 0.55 * q_abs + 0.45 * q_flow
        q_byp_pos = 0.5 * (q_byp + jnp.sqrt(q_byp * q_byp + 1.0e-8))
        q_eff = q_pos_part + q_byp_pos

        dt_bl = (p_fus - q_eff * 1.85 * (t_bl - t_hx)) / 26.0
        dt_hx = (q_eff * 1.85 * (t_bl - t_hx) - 0.90 * (t_hx - 570.0)) / 11.0

        z_flux = solve_sieverts_flux(u_ext * 0.042 * n_bl)
        j_store = 0.050 * n_ex
        return jnp.stack([dq, dt_bl, dt_hx, -z_flux, z_flux - j_store, j_store])


_PROB = _Tier1Problem()
_DEV = "gpu" if any(d.platform == "gpu" for d in jax.devices()) else "cpu"


def forward_ensemble(y0_batch: np.ndarray, p_batch: np.ndarray) -> jax.Array:
    res = gradsolve.solve(
        _PROB, np.asarray(y0_batch, dtype=np.float64), np.asarray(p_batch, dtype=np.float64),
        engine="rodas5p_replay", rtol=1e-8, atol=1e-10, device=_DEV,
    )
    return jnp.asarray(res.y_final)


def differentiable_closure(y0_batch: np.ndarray, p_ref_batch: np.ndarray):
    return gradsolve.grad_closure(
        _PROB, np.asarray(y0_batch, dtype=np.float64), np.asarray(p_ref_batch, dtype=np.float64),
        engine="rodas5p_replay", rtol=1e-8, atol=1e-10, device=_DEV,
    )
'''


TIER2_REFERENCE = '''\
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("GRADSOLVE_X64", "1")

import gradsolve
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Exact constants from TRANSFORM ConstantPropertyLiquidPbLi & ConstantPropertyLiquidFLiBe:
# Modelica PartialSimpleMedium uses u = cv_const*(T - T0) and h = cp_const*(T - T0)!
NV = 6
D_PBLI, CP_PBLI, CV_PBLI = 9637.0, 188.0, 141.0
D_FLIBE, CP_FLIBE, CV_FLIBE = 1948.0, 2380.0, 1785.0
V1_CELL = 0.12 / NV
V2_CELL = 0.12 / NV
CAP1 = V1_CELL * D_PBLI * CV_PBLI
CAP2 = V2_CELL * D_FLIBE * CV_FLIBE
G_CELL = 48000.0 / NV
EPS_Z = 1.0e-8


@jax.custom_jvp
def _solve_trit_flux(rhs: jax.Array) -> jax.Array:
    j = rhs / (1.0 + jnp.cbrt(rhs * rhs + 1.0e-12))
    for _ in range(7):
        aj = jnp.sqrt(j * j + EPS_Z * EPS_Z)
        f = j + 0.45 * j * aj + 0.25 * j * j * j - rhs
        df = 1.0 + 0.45 * (2.0 * j * j + EPS_Z * EPS_Z) / aj + 0.75 * j * j
        j = j - f / df
    return j


@_solve_trit_flux.defjvp
def _trit_jvp(primals, tangents):
    (r,), (dr,) = primals, tangents
    j = _solve_trit_flux(r)
    aj = jnp.sqrt(j * j + EPS_Z * EPS_Z)
    df = 1.0 + 0.45 * (2.0 * j * j + EPS_Z * EPS_Z) / aj + 0.75 * j * j
    return j, dr / df


def _smooth_knee(x: jax.Array, knee: float, slope_low: float = 0.15, eps: float = 1.0e-4) -> jax.Array:
    dx = x - knee
    return knee + 0.5 * (1.0 + slope_low) * dx + 0.5 * (1.0 - slope_low) * jnp.sqrt(dx * dx + eps * eps)


class _Tier2Problem:
    name = "user_tier2_transform_hx"
    dim = 15
    t0 = 0.0
    t1 = 30.0
    is_stiff = True

    def f_jax(self, t, y, p):
        m_pbli_cmd, m_flibe_cmd, t_in_base, u_vac = p[0], p[1], p[2], p[3]
        t1 = y[0:6]
        t2 = y[6:12]
        n_loop, n_perm, _n_get = y[12], y[13], y[14]

        m1 = _smooth_knee(m_pbli_cmd, 15.0, 0.15)
        m2 = _smooth_knee(m_flibe_cmd, 3.0, 0.15)
        t1_in = t_in_base + 18.0 * jnp.sin(0.20 * t)
        t2_in = 865.0

        q_wall = G_CELL * (t1 - t2[::-1])

        t1_prev = jnp.concatenate([jnp.array([t1_in]), t1[:-1]])
        t2_prev = jnp.concatenate([jnp.array([t2_in]), t2[:-1]])

        dt1 = (m1 * CP_PBLI * (t1_prev - t1) - q_wall) / CAP1
        dt2 = (m2 * CP_FLIBE * (t2_prev - t2) + q_wall[::-1]) / CAP2

        k_bar = jnp.mean(0.22 * jnp.exp(-3400.0 / t1))
        j_trit = _solve_trit_flux(u_vac * k_bar * n_loop)
        dn_loop = -j_trit
        dn_perm = j_trit - 0.06 * n_perm
        dn_get = 0.06 * n_perm

        return jnp.concatenate([dt1, dt2, jnp.array([dn_loop, dn_perm, dn_get])])


_PROB = _Tier2Problem()
_DEV = "gpu" if any(d.platform == "gpu" for d in jax.devices()) else "cpu"


def _to_obs(y_final: jax.Array) -> jax.Array:
    t1 = y_final[:, 0:6]
    t2 = y_final[:, 6:12]
    q_mw = jnp.sum(G_CELL * (t1 - t2[:, ::-1]), axis=1) * 1.0e-6
    return jnp.column_stack([q_mw, t1[:, 5], t2[:, 5], y_final[:, 12], y_final[:, 13], y_final[:, 14]])


def forward_ensemble(y0_batch: np.ndarray, p_batch: np.ndarray) -> jax.Array:
    res = gradsolve.solve(
        _PROB, np.asarray(y0_batch, dtype=np.float64), np.asarray(p_batch, dtype=np.float64),
        engine="rodas5p_replay", rtol=1e-8, atol=1e-10, device=_DEV,
    )
    return _to_obs(res.y_final)


def differentiable_closure(y0_batch: np.ndarray, p_ref_batch: np.ndarray):
    raw_cl = gradsolve.grad_closure(
        _PROB, np.asarray(y0_batch, dtype=np.float64), np.asarray(p_ref_batch, dtype=np.float64),
        engine="rodas5p_replay", rtol=1e-8, atol=1e-10, device=_DEV,
    )
    return lambda p_jax: _to_obs(raw_cl(p_jax))
'''


_TIER3_SHARED = '''\
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
'''


TIER3_FAST_REFERENCE = '''\
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

''' + _TIER3_SHARED + '''

N_STEPS = 500
DT = 25.0 / N_STEPS


def _integrate_single(y0: jax.Array, p: jax.Array) -> jax.Array:
    def body(y, step_idx):
        t = step_idx * DT
        k1 = _rhs_single(t, y, p)
        k2 = _rhs_single(t + 0.5 * DT, y + 0.5 * DT * k1, p)
        k3 = _rhs_single(t + 0.5 * DT, y + 0.5 * DT * k2, p)
        k4 = _rhs_single(t + DT, y + DT * k3, p)
        return y + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    y_end, _ = jax.lax.scan(body, y0, jnp.arange(N_STEPS, dtype=jnp.float64))
    return _obs_single(y_end, p)


_SOLVE_BATCH_JIT = jax.jit(jax.vmap(_integrate_single, in_axes=(0, 0)))


def forward_ensemble(y0_batch: np.ndarray, p_batch: np.ndarray) -> jax.Array:
    return _SOLVE_BATCH_JIT(jnp.asarray(y0_batch, dtype=jnp.float64), jnp.asarray(p_batch, dtype=jnp.float64))


def differentiable_closure(y0_batch: np.ndarray, p_ref_batch: np.ndarray):
    y0_j = jnp.asarray(y0_batch, dtype=jnp.float64)
    return lambda p_jax: _SOLVE_BATCH_JIT(y0_j, p_jax)
'''


TIER3_DENSE_STIFF_REFERENCE = '''\
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("GRADSOLVE_X64", "1")

import gradsolve
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

''' + _TIER3_SHARED + '''

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
'''


REFERENCES: dict[str, str] = {
    "tier1_blanket_dae": TIER1_REFERENCE,
    "tier2_transform_hx": TIER2_REFERENCE,
    "tier3_hdt_column": TIER3_FAST_REFERENCE,
}
