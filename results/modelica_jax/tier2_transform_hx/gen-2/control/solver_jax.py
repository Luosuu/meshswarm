"""Verified, differentiable JAX GPU ensemble solver for TransformPbLiFLiBeHX.

Reduced 15-state ODE derived from the flattened Modelica DAE (TRANSFORM Simple_HX,
nV=6, counterCurrent, constant-property PbLi/FLiBe media):

  dT1[i] = (m1*cp1*(Tup1[i]-T1[i]) - q[i]) / (rho1*Vzone*cv1),  i = 0..5
  dT2[j] = (m2*cp2*(Tup2[j]-T2[j]) + q[5-j]) / (rho2*Vzone*cv2), j = 0..5
    q[i] = (T1[i]-T2[5-i]) * UA/6   (counter-current pairing)
    Tup1 = [T_in_pbli + 18*sin(0.2 t), T1[0], ..., T1[4]]
    Tup2 = [865.0, T2[0], ..., T2[4]]
  j_trit solves  j + 0.45*j*|j| + 0.25*j^3 = u_vac*k_bar*N_loop
    k_bar = mean_i 0.22*exp(-3400/T1[i])
  dN_loop = -j_trit ; dN_perm = j_trit - 0.06*N_perm ; dN_getter = 0.06*N_perm
  N_loop+N_perm+N_getter = 6 exactly (linear invariant, preserved by RK4)

obs = [obs_Q_MW = 1e-6*sum(q), obs_T1_out = T1[5], obs_T2_out = T2[5],
       obs_N_loop, obs_N_perm, obs_N_getter]

Integration: fixed-step classical RK4 (dt=0.25) with lax.scan; vmap over the ensemble.
The implicit Sieverts root-finder uses @jax.custom_jvp (implicit function theorem), so
reverse-mode and forward-mode agree to machine precision.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# --- constants (ORNL TRANSFORM ConstantProperty media; T0 = 298.15 K) ----------------
_CP1, _CV1, _RHO1 = 188.0, 141.0, 9637.0   # PbLi
_CP2, _CV2, _RHO2 = 2380.0, 1785.0, 1948.0 # FLiBe
_VZ = 0.12 / 6.0                           # zone volume
_INV1 = 1.0 / (_RHO1 * _VZ * _CV1)
_INV2 = 1.0 / (_RHO2 * _VZ * _CV2)
_UA6 = 48000.0 / 6.0                       # UA per zone [W/K]
_T2_IN = 865.0
_T_FINAL = 30.0
_DT = 0.25
_NSTEPS = int(round(_T_FINAL / _DT))


@jax.custom_jvp
def _sieverts_j(r):
    """Solve j + 0.45*j*|j| + 0.25*j^3 = r for the unique real j (Newton, 40 iters)."""
    def body(_, j):
        f = j + 0.45 * j * jnp.abs(j) + 0.25 * j * j * j - r
        d = 1.0 + 0.9 * jnp.abs(j) + 0.75 * j * j
        return j - f / d
    return jax.lax.fori_loop(0, 40, body, r)


@_sieverts_j.defjvp
def _sieverts_j_jvp(primals, tangents):
    (r,) = primals
    (dr,) = tangents
    j = _sieverts_j(r)
    # implicit function theorem: dj/dr = 1 / (1 + 0.9|j| + 0.75 j^2)
    dj = dr / (1.0 + 0.9 * jnp.abs(j) + 0.75 * j * j)
    return j, dj


def _rhs(t, y, p):
    m1 = jnp.where(p[0] >= 15.0, p[0], 15.0 + 0.15 * (p[0] - 15.0))
    m2 = jnp.where(p[1] >= 3.0, p[1], 3.0 + 0.15 * (p[1] - 3.0))
    t1in = p[2] + 18.0 * jnp.sin(0.2 * t)
    T1 = y[:6]
    T2 = y[6:12]
    Tup1 = jnp.concatenate([t1in[None], T1[:-1]])
    Tup2 = jnp.concatenate([jnp.array([_T2_IN], dtype=y.dtype), T2[:-1]])
    q = (T1 - T2[::-1]) * _UA6          # zone heat flow T1[i] -> T2[5-i]
    dT1 = (m1 * _CP1 * (Tup1 - T1) - q) * _INV1
    dT2 = (m2 * _CP2 * (Tup2 - T2) + q[::-1]) * _INV2
    k_bar = jnp.mean(0.22 * jnp.exp(-3400.0 / T1))
    j = _sieverts_j(p[3] * k_bar * y[12])
    dN = jnp.array([-j, j - 0.06 * y[13], 0.06 * y[13]], dtype=y.dtype)
    return jnp.concatenate([dT1, dT2, dN])


def _integrate(y0, p):
    dt = _DT

    def step(y, i):
        t = i * dt
        k1 = _rhs(t, y, p)
        k2 = _rhs(t + 0.5 * dt, y + 0.5 * dt * k1, p)
        k3 = _rhs(t + 0.5 * dt, y + 0.5 * dt * k2, p)
        k4 = _rhs(t + dt, y + dt * k3, p)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    yT, _ = jax.lax.scan(step, y0, jnp.arange(_NSTEPS, dtype=jnp.float64))
    return yT


def _obs(yT):
    T1 = yT[:6]
    T2 = yT[6:12]
    Q = jnp.sum((T1 - T2[::-1]) * _UA6) * 1e-6
    return jnp.array([Q, T1[5], T2[5], yT[12], yT[13], yT[14]], dtype=yT.dtype)


@jax.jit
def _ensemble_obs(y0_b, p_b):
    f = jax.vmap(lambda y, p: _obs(_integrate(y, p)))
    return f(y0_b, p_b)


def forward_ensemble(y0_batch, p_batch):
    y0 = jnp.asarray(y0_batch, dtype=jnp.float64)
    p = jnp.asarray(p_batch, dtype=jnp.float64)
    return _ensemble_obs(y0, p)


def differentiable_closure(y0_batch, p_ref_batch):
    y0 = jnp.asarray(y0_batch, dtype=jnp.float64)
    p_ref = jnp.asarray(p_ref_batch, dtype=jnp.float64)

    def f(p_jax):
        p = jnp.asarray(p_jax, dtype=jnp.float64)
        g = jax.vmap(lambda y, pp: _obs(_integrate(y, pp)))
        return g(y0, p)

    return f
