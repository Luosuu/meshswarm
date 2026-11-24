"""Deliverable JAX GPU ensemble solver for CryoHDTColumn16.

DAE -> ODE reduction:
  * The 16 bubble-point algebraic equations are solved per RHS call with a Newton
    iteration (vectorised over stages). The solve uses ``@jax.custom_jvp`` with the
    Implicit Function Theorem so reverse-mode AD is exact and cheap.
  * The flooding efficiency branch is continuous at V_eff == V_flood, so a ``where``
    on the branch is sufficient.
  * Integration is a fixed-step RK4 under ``jax.lax.scan``; the whole ensemble is
    evaluated with ``jax.vmap`` and jitted.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_N = 16
_M = jnp.array([4.0] + [1.0] * 14 + [6.0], dtype=jnp.float64)
_IDX = jnp.arange(_N, dtype=jnp.float64)
_ARANGE = jnp.arange(_N)
_V_FLOOD = 2.15
_ETA_NOM = 0.92


def _psat(T):
    pH2 = jnp.exp(9.45 - 106.0 / (T + 1.8))
    pHD = jnp.exp(9.62 - 129.0 / (T + 1.8))
    pT2 = jnp.exp(9.85 - 156.0 / (T + 1.8))
    return pH2, pHD, pT2


@jax.custom_jvp
def _bubble_T(x_H2, x_HD, P):
    # Newton solve for the bubble-point temperature; initial guess T = 11.5 K.
    T = jnp.full_like(P, 11.5)
    for _ in range(7):
        pH2, pHD, pT2 = _psat(T)
        x_T2 = 1.0 - x_H2 - x_HD
        f = x_H2 * pH2 + x_HD * pHD + x_T2 * pT2 - P
        df = (x_H2 * pH2 * (106.0 / (T + 1.8) ** 2)
              + x_HD * pHD * (129.0 / (T + 1.8) ** 2)
              + x_T2 * pT2 * (156.0 / (T + 1.8) ** 2))
        T = T - f / df
    return T


@_bubble_T.defjvp
def _bubble_T_jvp(primals, tangents):
    x_H2, x_HD, P = primals
    dx_H2, dx_HD, dP = tangents
    T = _bubble_T(x_H2, x_HD, P)
    pH2, pHD, pT2 = _psat(T)
    x_T2 = 1.0 - x_H2 - x_HD
    dfT = (x_H2 * pH2 * (106.0 / (T + 1.8) ** 2)
           + x_HD * pHD * (129.0 / (T + 1.8) ** 2)
           + x_T2 * pT2 * (156.0 / (T + 1.8) ** 2))
    df_in = pH2 * dx_H2 + pHD * dx_HD + pT2 * (-dx_H2 - dx_HD) - dP
    dT = -df_in / dfT
    return T, dT


def _rhs(t, y, V_boil, d_rec_frac, dp_tray, P_top):
    x_H2 = y[:_N]
    x_HD = y[_N:]

    V_eff = V_boil * (1.0 + 0.08 * jnp.sin(0.25 * t))
    eta = jnp.where(V_eff <= _V_FLOOD,
                    _ETA_NOM,
                    _ETA_NOM / (1.0 + 1.8 * (V_eff - _V_FLOOD) ** 2))
    D_rec = d_rec_frac * V_eff
    L_rect = V_eff - D_rec

    P_stage = P_top + _IDX * dp_tray
    T = _bubble_T(x_H2, x_HD, P_stage)
    pH2, pHD, pT2 = _psat(T)

    y_H2 = (1.0 - eta) * x_H2 + eta * x_H2 * pH2 / P_stage
    y_HD = (1.0 - eta) * x_HD + eta * x_HD * pHD / P_stage

    def col(x, yfrac):
        # Liquid entering stage j from above:
        #   j == 0    : none
        #   1..7      : L_rect * x[j-1]   (j == 7 also gets D_rec * x[0])
        #   8..15     : V_eff  * x[j-1]
        in_x = jnp.concatenate([jnp.zeros(1, dtype=x.dtype), x[:-1]])
        in_x = in_x * jnp.where(_ARANGE <= 7, L_rect, V_eff)
        in_x = in_x.at[7].add(D_rec * x[0])

        # Vapour entering stage j from below:
        #   j == 15   : none
        #   0..14     : V_eff * y[j+1]
        in_y = jnp.concatenate([yfrac[1:], jnp.zeros(1, dtype=x.dtype)]) * V_eff

        # Liquid leaving stage j downwards:
        #   j == 0    : V_eff * x[0]
        #   1..6      : L_rect * x[j]
        #   7..14     : V_eff * x[j]
        #   j == 15   : none
        coef_out_x = jnp.where(_ARANGE == 0, V_eff,
                               jnp.where(_ARANGE <= 6, L_rect,
                                         jnp.where(_ARANGE <= 14, V_eff, 0.0)))
        out_x = x * coef_out_x

        # Vapour leaving stage j upwards:
        #   j == 0    : none (condenser / no vapour out term)
        #   1..15     : V_eff * y[j]
        out_y = jnp.where(_ARANGE == 0, 0.0, V_eff) * yfrac

        return (in_x + in_y - out_x - out_y) / _M

    dx_H2 = col(x_H2, y_H2)
    dx_HD = col(x_HD, y_HD)
    return jnp.concatenate([dx_H2, dx_HD])


def _observables(y_final, V_boil, d_rec_frac, dp_tray, P_top):
    x_H2 = y_final[:_N]
    x_HD = y_final[_N:]
    x_T2 = 1.0 - x_H2 - x_HD
    P_stage = P_top + _IDX * dp_tray
    T = _bubble_T(x_H2, x_HD, P_stage)
    tot_H2 = jnp.sum(_M * x_H2)
    tot_HD = jnp.sum(_M * x_HD)
    tot_T2 = jnp.sum(_M * x_T2)
    return jnp.array([x_H2[0], x_T2[_N - 1], T[0], T[_N - 1],
                      tot_H2, tot_HD, tot_T2], dtype=jnp.float64)


_T_END = 25.0
_NSTEPS = 500
_H = _T_END / _NSTEPS


def _integrate(y0, params):
    V_boil, d_rec_frac, dp_tray, P_top = params

    def step(y, t):
        k1 = _rhs(t, y, V_boil, d_rec_frac, dp_tray, P_top)
        k2 = _rhs(t + 0.5 * _H, y + 0.5 * _H * k1, V_boil, d_rec_frac, dp_tray, P_top)
        k3 = _rhs(t + 0.5 * _H, y + 0.5 * _H * k2, V_boil, d_rec_frac, dp_tray, P_top)
        k4 = _rhs(t + _H, y + _H * k3, V_boil, d_rec_frac, dp_tray, P_top)
        y_new = y + (_H / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y_new, None

    ts = jnp.arange(_NSTEPS, dtype=jnp.float64) * _H
    y_final, _ = jax.lax.scan(step, y0, ts)
    return y_final


def _forward_one(y0, params):
    y_final = _integrate(y0, params)
    return _observables(y_final, *params)


_forward_batch = jax.jit(jax.vmap(_forward_one, in_axes=(0, 0)))


def forward_ensemble(y0_batch, p_batch):
    y0 = jnp.asarray(y0_batch, dtype=jnp.float64)
    p = jnp.asarray(p_batch, dtype=jnp.float64)
    return _forward_batch(y0, p)


def differentiable_closure(y0_batch, p_ref_batch):
    y0 = jnp.asarray(y0_batch, dtype=jnp.float64)

    def f(p_jax):
        p = jnp.asarray(p_jax, dtype=jnp.float64)
        return _forward_batch(y0, p)

    return f
