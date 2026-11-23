"""Reference JAX solver for Tier 2 (`TransformPbLiFLiBeHX.mo`)."""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("GRADSOLVE_X64", "1")

import gradsolve  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Exact constants from TRANSFORM ConstantPropertyLiquidPbLi & ConstantPropertyLiquidFLiBe:
# Modelica PartialSimpleMedium uses u = cv_const*(T - T0) and h = cp_const*(T - T0).
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
    return _to_obs(jnp.asarray(res.y_final))


def differentiable_closure(y0_batch: np.ndarray, p_ref_batch: np.ndarray):
    raw_cl = gradsolve.grad_closure(
        _PROB, np.asarray(y0_batch, dtype=np.float64), np.asarray(p_ref_batch, dtype=np.float64),
        engine="rodas5p_replay", rtol=1e-8, atol=1e-10, device=_DEV,
    )
    return lambda p_jax: _to_obs(raw_cl(p_jax))
