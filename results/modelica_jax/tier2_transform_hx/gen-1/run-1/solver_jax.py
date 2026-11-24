"""Differentiable JAX GPU ensemble solver for TransformPbLiFLiBeHX.

Flattened model (verified against OpenModelica DASSL trajectories to ~1e-8):

  State y[15] = [T1_1..T1_6 (PbLi zones), T2_1..T2_6 (FLiBe zones),
                 N_loop, N_perm, N_getter]

  PbLi zone i:  dT1_i = ( m1*cp1*(T1_up,i - T1_i) - (T1_i - T2_{7-i})*UA6 ) / M1
  FLiBe zone i: dT2_i = ( m2*cp2*(T2_up,i - T2_i) + (T1_{7-i} - T2_i)*UA6 ) / M2

  Counter-current pairing (zone i side 1 <-> zone 7-i side 2), M_i = rho*cv*V_cell.
  m1 = piecewise low-flow limited pbli flow, m2 likewise for FLiBe.
  T_in_pbli(t) = T_in_pbli + 18*sin(0.2*t), FLiBe inlet fixed 865 K.

  Sieverts wall permeation: j solves  j + 0.45*j*|j| + 0.25*j**3 = u_vac*k_bar*N_loop
  with k_bar = mean_i(0.22*exp(-3400/T1_i)); the cubic is strictly monotone so the
  root is unique. Solved with Newton inside a @jax.custom_jvp wrapper (implicit
  function theorem) so reverse-mode gradients never see the iterations.

  Tritium inventory: dN_loop = -j ;  dN_perm = j - 0.06*N_perm ;
  dN_getter = -(dN_loop + dN_perm) so that N_loop+N_perm+N_getter is an exact
  invariant (linear invariant preserved to machine precision by RK methods).
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Constants from TRANSFORM.Media.Fluids (SimpleLiquidPbLi / SimpleLiquidFLiBe)
RHO1, CV1, CP1 = 9637.0, 141.0, 188.0
RHO2, CV2, CP2 = 1948.0, 1785.0, 2380.0
V_CELL = 0.12 / 6.0
M1 = RHO1 * CV1 * V_CELL   # 27176.34 J/K
M2 = RHO2 * CV2 * V_CELL   # 69543.6  J/K
UA6 = 48000.0 / 6.0        # 8000.0 W/K
T_FLIBE_IN = 865.0
SORB_K = 0.06
K_ARR_A = 0.22
K_ARR_E = 3400.0
T0_REF = 298.15
SIN_AMP = 18.0
SIN_OMEGA = 0.2

T_FINAL = 30.0
Y0_DEFAULT = np.array([930.0]*6 + [880.0]*6 + [5.0, 0.5, 0.5], dtype=np.float64)


@jax.custom_jvp
def _sieverts_j(s):
    """Unique root of j + 0.45*j*|j| + 0.25*j^3 = s. Monotone odd cubic."""
    sa = jnp.abs(s)

    def newton(j, _):
        g = j + 0.45 * j * j + 0.25 * j * j * j - sa
        gp = 1.0 + 0.9 * j + 0.75 * j * j
        return j - g / gp, None

    j0 = sa  # g convex & increasing for j>0; root <= sa
    j, _ = jax.lax.scan(newton, j0, None, length=60)
    return jnp.sign(s) * j


@_sieverts_j.defjvp
def _sieverts_j_jvp(primals, tangents):
    (s,), (ds,) = primals, tangents
    j = _sieverts_j(s)
    gj = 1.0 + 0.9 * jnp.abs(j) + 0.75 * j * j
    return j, ds / gj


def _flow_limit(m, threshold):
    return jnp.where(m >= threshold, m, threshold + 0.15 * (m - threshold))


def rhs_jax(t, y, p):
    T1 = y[:6]
    T2 = y[6:12]
    N_loop = y[12]
    N_perm = y[13]

    m_flow_pbli, m_flow_flibe, T_in_pbli, u_vac = p[0], p[1], p[2], p[3]
    m1 = _flow_limit(m_flow_pbli, 15.0)
    m2 = _flow_limit(m_flow_flibe, 3.0)

    Ti = T_in_pbli + SIN_AMP * jnp.sin(SIN_OMEGA * t)

    T1_up = jnp.concatenate([jnp.expand_dims(Ti, 0), T1[:-1]])
    T2_up = jnp.concatenate([jnp.expand_dims(jnp.float64(T_FLIBE_IN), 0), T2[:-1]])

    T1_pair = T2[::-1]   # counter-current: side-1 zone i <-> side-2 zone 7-i
    T2_pair = T1[::-1]

    dT1 = (m1 * CP1 * (T1_up - T1) - (T1 - T1_pair) * UA6) / M1
    dT2 = (m2 * CP2 * (T2_up - T2) + (T2_pair - T2) * UA6) / M2

    k_arr = K_ARR_A * jnp.exp(-K_ARR_E / T1)
    k_bar = jnp.mean(k_arr)
    s = u_vac * k_bar * N_loop
    j = _sieverts_j(s)

    dN_loop = -j
    dN_perm = j - SORB_K * N_perm
    dN_getter = -(dN_loop + dN_perm)  # exact invariant: sum of ders is exactly 0

    return jnp.concatenate([dT1, dT2, jnp.array([dN_loop, dN_perm, dN_getter])])


def obs_from_final(y):
    T1 = y[..., :6]
    T2 = y[..., 6:12]
    Q = jnp.sum((T1 - T2[..., ::-1]), axis=-1) * UA6 * 1e-6
    return jnp.stack([Q, T1[..., 5], T2[..., 5], y[..., 12], y[..., 13], y[..., 14]], axis=-1)


class _Problem:
    name = "TransformPbLiFLiBeHX"
    dim = 15
    t0 = 0.0
    t1 = T_FINAL
    is_stiff = True

    def f_jax(self, t, y, params):
        return rhs_jax(t, y, params)


_PROBLEM = _Problem()


def forward_ensemble(y0_batch, p_batch):
    import gradsolve

    y0 = np.asarray(y0_batch, dtype=np.float64)
    p = np.asarray(p_batch, dtype=np.float64)
    res = gradsolve.solve(
        _PROBLEM, y0, p,
        engine="rodas5p_replay",
        device="cuda",
        rtol=1e-11,
        atol=1e-12,
    )
    yf = res.y_final
    yf = jnp.asarray(yf, dtype=jnp.float64)
    return obs_from_final(yf)


def differentiable_closure(y0_batch, p_ref_batch):
    import gradsolve

    y0 = np.asarray(y0_batch, dtype=np.float64)
    p_ref = np.asarray(p_ref_batch, dtype=np.float64)

    cl = gradsolve.grad_closure(
        _PROBLEM, y0, p_ref,
        wrt="params",
        engine="rodas5p_replay",
        device="cuda",
        rtol=1e-11,
        atol=1e-12,
    )

    def p_jax_to_obs(p_jax):
        yf = cl(p_jax)
        return obs_from_final(yf)

    return p_jax_to_obs
