"""Deliverable JAX GPU ensemble solver for BlanketValveDAE.

Contract required by the grader:
  1. `forward_ensemble(y0_batch, p_batch) -> jax.Array`
     - `y0_batch`: float64 array of shape `(B, 6)`
     - `p_batch`:  float64 array of shape `(B, 4)` with columns ['u_pump', 'u_ext', 'P_pulse', 'dp_back']
     - returns float64 array of shape `(B, 6)` at t = 20.0.
  2. `differentiable_closure(y0_batch, p_ref_batch) -> Callable[[jax.Array], jax.Array]`
     - returns a pure JAX-traceable function `f(p_jax) -> obs_final_jax` of shape `(B, 6)`.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---- Model constants (flattened from BlanketValveDAE.mo) ----
TAU_HYD = 0.02
K_FWD = 0.24
K_REV = 14.0
K_BYP = 0.08
C_BLANKET = 26.0
C_HX = 11.0
CP_COOL = 1.85
H_SINK = 0.90
T_SINK = 570.0
P_BASE = 115.0
K_PERM = 0.042
ALPHA_NL = 0.65
BETA_NL = 0.30
K_REGEN = 0.050

T_END = 20.0
DT = 0.005
N_STEPS = int(round(T_END / DT))


@jax.custom_jvp
def z_of_R(R):
    """Solve z + alpha*z*|z| + beta*z^3 = R  (monotone, unique root)."""
    z = R
    for _ in range(16):
        f = z + ALPHA_NL * z * jnp.abs(z) + BETA_NL * z**3 - R
        fp = 1.0 + 2.0 * ALPHA_NL * jnp.abs(z) + 3.0 * BETA_NL * z**2
        z = z - f / fp
    return z


@z_of_R.defjvp
def _z_of_R_jvp(primals, tangents):
    (R,) = primals
    (Rdot,) = tangents
    z = z_of_R(R)
    fp = 1.0 + 2.0 * ALPHA_NL * jnp.abs(z) + 3.0 * BETA_NL * z**2
    return z, Rdot / fp


def _rhs(y, t, p):
    u_pump, u_ext, P_pulse, dp_back = p[0], p[1], p[2], p[3]
    q = y[0]
    Tb = y[1]
    Th = y[2]
    Nb = y[3]
    Ne = y[4]
    Ns = y[5]

    P_fus = P_BASE * (1.0 + 0.40 * P_pulse * jnp.sin(0.20 * t))
    dp_drive = 2.0 * u_pump - dp_back * (1.0 + 0.25 * jnp.cos(0.35 * t))

    drag_valve = jnp.where(q >= 0.0, K_FWD * q * q, -K_REV * q * q)
    q_byp = jnp.where(
        dp_drive >= 0.0,
        K_BYP * jnp.sqrt(jnp.maximum(dp_drive, 0.0) + 1.0e-8),
        -0.15 * K_BYP * jnp.sqrt(jnp.maximum(-dp_drive, 0.0) + 1.0e-8),
    )

    dq = (dp_drive - drag_valve - 0.30 * q) / TAU_HYD

    q_eff = jnp.where(q >= 0.0, q, -0.10 * q) + jnp.where(q_byp >= 0.0, q_byp, 0.0)

    dTb = (P_fus - q_eff * CP_COOL * (Tb - Th)) / C_BLANKET
    dTh = (q_eff * CP_COOL * (Tb - Th) - H_SINK * (Th - T_SINK)) / C_HX

    R = u_ext * K_PERM * Nb
    z = z_of_R(R)
    j_store = K_REGEN * Ne

    dNb = -z
    dNe = z - j_store
    dNs = j_store

    return jnp.stack([dq, dTb, dTh, dNb, dNe, dNs])


def _integrate_one(y0, p):
    def body(carry, t):
        y = carry
        k1 = _rhs(y, t, p)
        k2 = _rhs(y + 0.5 * DT * k1, t + 0.5 * DT, p)
        k3 = _rhs(y + 0.5 * DT * k2, t + 0.5 * DT, p)
        k4 = _rhs(y + DT * k3, t + DT, p)
        y1 = y + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y1, None

    ts = DT * jnp.arange(N_STEPS)
    yT, _ = jax.lax.scan(body, y0, ts)
    return yT


_integrate_batch = jax.jit(jax.vmap(_integrate_one, in_axes=(0, 0)))


def forward_ensemble(y0_batch, p_batch):
    y0 = jnp.asarray(y0_batch, dtype=jnp.float64)
    p = jnp.asarray(p_batch, dtype=jnp.float64)
    return _integrate_batch(y0, p)


def differentiable_closure(y0_batch, p_ref_batch):
    y0 = jnp.asarray(y0_batch, dtype=jnp.float64)

    def f(p_jax):
        p = jnp.asarray(p_jax, dtype=jnp.float64)
        return _integrate_batch(y0, p)

    return f
