"""Deliverable JAX GPU ensemble solver for TransformPbLiFLiBeHX.

Reduced-order model derived from the flattened OpenModelica equations of
`TransformPbLiFLiBeHX` (TRANSFORM.HeatExchangers.Simple_HX, nV=6,
counterCurrent=true, two ConstantPropertyLiquid single-state media).

Both media are incompressible with constant properties (cp, cv, rho const),
so density and pressure decouple from the energy balance. The 6-zone
counter-current heat exchanger reduces to 12 coupled first-order ODEs for
the zone temperatures, plus 3 ODEs for the closed tritium loop inventories
(N_loop, N_perm, N_getter). The Sieverts cubic for the extraction current
j_trit is scalar and monotone:

    j + 0.45*j*|j| + 0.25*j^3 = u_vac * k_bar * N_loop,
    k_arr[i] = 0.22*exp(-3400/T1[i]),  k_bar = mean(k_arr)

which has a unique real root for any (u_vac, k_bar, N_loop); it is solved by
a damped Newton iteration wrapped in @jax.custom_jvp (implicit function
theorem) so reverse-mode AD is exact and cheap.

Integration: fixed-step classic RK4 with `lax.scan`, vectorized over the
ensemble with `vmap`. Linear invariants (N_loop+N_perm+N_getter) are
preserved exactly by any Runge-Kutta method, matching the closed-loop
molar-inventory drift requirement.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---- Physical constants (from TRANSFORM library media + model) ----------
_N_ZONE = 6
_V_ZONE = 0.12 / 6.0            # m^3 per zone (V_1 = V_2 = 0.12)
_UA_ZONE = 48000.0 / 6.0        # W/K per zone (UA/nV, CF = 1)
_CP1, _CV1, _RHO1 = 188.0, 141.0, 9637.0   # PbLi (J/kg/K, J/kg/K, kg/m^3)
_CP2, _CV2, _RHO2 = 2380.0, 1785.0, 1948.0  # FLiBe
_M1 = _V_ZONE * _RHO1           # kg per zone (side 1)
_M2 = _V_ZONE * _RHO2           # kg per zone (side 2)
_T2_IN = 865.0                  # FLiBe inlet temperature (K)
_T_END = 30.0

# Fixed RK4 step count (chosen from a measured accuracy sweep, see FINAL.md)
_N_STEPS = 600
_DT = _T_END / _N_STEPS


def _limit_flow_pbli(m):
    # piecewise low-flow limiter, smooth branch for AD
    return jnp.where(m >= 15.0, m, 15.0 + 0.15 * (m - 15.0))


def _limit_flow_flibe(m):
    return jnp.where(m >= 3.0, m, 3.0 + 0.15 * (m - 3.0))


@jax.custom_jvp
def _sieverts_j(c):
    """Solve j + 0.45*j*|j| + 0.25*j^3 = c  (monotone, unique real root)."""
    j = c
    def body(_, j):
        fj = j + 0.45 * j * jnp.abs(j) + 0.25 * j**3 - c
        dfj = 1.0 + 0.9 * jnp.abs(j) + 0.75 * j**2
        return j - fj / dfj
    j = jax.lax.fori_loop(0, 40, body, j)
    return j


@_sieverts_j.defjvp
def _sieverts_j_jvp(primals, tangents):
    (c,) = primals
    (dc,) = tangents
    j = _sieverts_j(c)
    dfj = 1.0 + 0.9 * jnp.abs(j) + 0.75 * j**2
    return j, dc / dfj


def _rhs(y, t, p):
    m_pbli, m_flibe, T_in_pbli, u_vac = p
    T1 = y[:6]
    T2 = y[6:12]
    N_loop, N_perm, N_getter = y[12], y[13], y[14]

    u = T_in_pbli + 18.0 * jnp.sin(0.2 * t)
    m1 = _limit_flow_pbli(m_pbli)
    m2 = _limit_flow_flibe(m_flibe)

    UA = _UA_ZONE
    # side 1 (PbLi), zones in flow order; zone capacity M*cv, flow enthalpy rate m*cp
    tin1 = jnp.concatenate([jnp.array([u]), T1[:5]])
    dT1 = (m1 * _CP1 * (tin1 - T1) + UA * (T2[::-1] - T1)) / (_M1 * _CV1)
    # side 2 (FLiBe), counter-current: zone j receives flow from j-1, heat from T1[5-j]
    tin2 = jnp.concatenate([jnp.array([_T2_IN]), T2[:5]])
    dT2 = (m2 * _CP2 * (tin2 - T2) + UA * (T1[::-1] - T2)) / (_M2 * _CV2)

    k_arr = 0.22 * jnp.exp(-3400.0 / T1)
    k_bar = jnp.mean(k_arr)
    c = u_vac * k_bar * N_loop
    j = _sieverts_j(c)

    dN_loop = -j
    dN_perm = j - 0.06 * N_perm
    dN_getter = 0.06 * N_perm
    return jnp.concatenate([dT1, dT2, jnp.array([dN_loop, dN_perm, dN_getter])])


def _rk4_step(y, t, p, dt):
    k1 = _rhs(y, t, p)
    k2 = _rhs(y + 0.5 * dt * k1, t + 0.5 * dt, p)
    k3 = _rhs(y + 0.5 * dt * k2, t + 0.5 * dt, p)
    k4 = _rhs(y + dt * k3, t + dt, p)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _integrate(y0, p):
    def step(y, i):
        t = i * _DT
        y = _rk4_step(y, t, p, _DT)
        return y, None
    yT, _ = jax.lax.scan(step, y0, jnp.arange(_N_STEPS))
    return yT


def _obs(yT):
    T1 = yT[:6]
    T2 = yT[6:12]
    Q_MW = _UA_ZONE * jnp.sum(T1 - T2[::-1]) * 1e-6
    return jnp.array([Q_MW, T1[5], T2[5], yT[12], yT[13], yT[14]])


def _single(y0, p):
    return _obs(_integrate(y0, p))


_forward_jit = jax.jit(jax.vmap(_single))


def _as_f64(a):
    return jnp.asarray(a, dtype=jnp.float64)


def forward_ensemble(y0_batch, p_batch):
    y0 = _as_f64(y0_batch)
    p = _as_f64(p_batch)
    return _forward_jit(y0, p)


def differentiable_closure(y0_batch, p_ref_batch):
    y0 = _as_f64(y0_batch)
    p_ref = _as_f64(p_ref_batch)

    @jax.jit
    def closure(p_jax):
        p = jnp.asarray(p_jax, dtype=jnp.float64)
        return jax.vmap(_single)(y0, p)

    return closure

# NOTE (design): the closure returned by `differentiable_closure` is jitted so
# that `jax.grad(closure)` reuses cached XLA executables instead of re-tracing
# the Python/scan body on every call (~11x faster for the 64-trajectory pass).
# `jax.jit` is transparent to AD, so forward/reverse-mode and jvp all work
# exactly as with the un-jitted closure.
