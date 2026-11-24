"""Deliverable JAX GPU ensemble solver for BlanketValveDAE."""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Physical constants (from BlanketValveDAE.mo)
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


@jax.custom_jvp
def _sieverts_z(R):
    """Solve  z + alpha*z*|z| + beta*z^3 = R  for z (monotone, unique)."""
    # Newton from z0 = R; g is convex for R>0 / concave for R<0, converges monotonically.
    def body(z, _):
        g = z + ALPHA_NL * z * jnp.abs(z) + BETA_NL * z * z * z - R
        gp = 1.0 + 2.0 * ALPHA_NL * jnp.abs(z) + 3.0 * BETA_NL * z * z
        return z - g / gp, None

    z0 = R
    z, _ = jax.lax.scan(body, z0, None, length=12)
    return z


@_sieverts_z.defjvp
def _sieverts_z_jvp(primals, tangents):
    (R,) = primals
    (dR,) = tangents
    z = _sieverts_z(R)
    gp = 1.0 + 2.0 * ALPHA_NL * jnp.abs(z) + 3.0 * BETA_NL * z * z
    return z, dR / gp


def rhs(t, y, p):
    u_pump, u_ext, P_pulse, dp_back = p[0], p[1], p[2], p[3]
    q, Tb, Th, Nb, Ne, Ns = y[0], y[1], y[2], y[3], y[4], y[5]

    P_fus = P_BASE * (1.0 + 0.40 * P_pulse * jnp.sin(0.20 * t))
    dp_drive = 2.0 * u_pump - dp_back * (1.0 + 0.25 * jnp.cos(0.35 * t))

    drag_valve = jnp.where(q >= 0.0, K_FWD * q * q, -K_REV * q * q)
    q_byp = jnp.where(
        dp_drive >= 0.0,
        K_BYP * jnp.sqrt(dp_drive + 1.0e-8),
        -0.15 * K_BYP * jnp.sqrt(-dp_drive + 1.0e-8),
    )
    dq = (dp_drive - drag_valve - 0.30 * q) / TAU_HYD

    q_eff = jnp.where(q >= 0.0, q, -0.10 * q) + jnp.where(q_byp >= 0.0, q_byp, 0.0)
    dTb = (P_fus - q_eff * CP_COOL * (Tb - Th)) / C_BLANKET
    dTh = (q_eff * CP_COOL * (Tb - Th) - H_SINK * (Th - T_SINK)) / C_HX

    R = u_ext * K_PERM * Nb
    z = _sieverts_z(R)
    j_store = K_REGEN * Ne

    dNb = -z
    dNe = z - j_store
    dNs = j_store

    return jnp.array([dq, dTb, dTh, dNb, dNe, dNs])


# ---------------- Rosenbrock23 (ode23s) fixed-step core ----------------
_D_COEF = 1.0 / (2.0 + np.sqrt(2.0))


def _ros2_step(f, t, y, dt, p):
    D = y.shape[-1]
    eye = jnp.eye(D, dtype=y.dtype)
    J = jax.jacfwd(f, argnums=1)(t, y, p)
    W = eye - dt * _D_COEF * J
    F0 = f(t, y, p)
    k1 = jnp.linalg.solve(W, F0)
    F1 = f(t, y + 0.5 * dt * k1, p)
    k2 = jnp.linalg.solve(W, F1 - k1) + k1
    return y + dt * k2


def _integrate(y0, p, t0, t1, dt, stepper):
    n_steps = int(round((t1 - t0) / dt))

    def body(carry, t):
        y = carry
        y = stepper(rhs, t, y, dt, p)
        return y, None

    ts = t0 + dt * jnp.arange(n_steps, dtype=y0.dtype)
    yT, _ = jax.lax.scan(body, y0, ts)
    return yT


def _solve_one(y0, p, dt):
    return _integrate(y0, p, 0.0, 20.0, dt, _ros2_step)


_solve_batch = jax.jit(
    jax.vmap(_solve_one, in_axes=(0, 0, None)),
    static_argnums=2,
)


def forward_ensemble(y0_batch, p_batch):
    y0_batch = jnp.asarray(y0_batch, dtype=jnp.float64)
    p_batch = jnp.asarray(p_batch, dtype=jnp.float64)
    return _solve_batch(y0_batch, p_batch, 0.05)


def differentiable_closure(y0_batch, p_ref_batch):
    y0_batch = jnp.asarray(y0_batch, dtype=jnp.float64)
    p_ref_batch = jnp.asarray(p_ref_batch, dtype=jnp.float64)

    def f(p_jax):
        return _solve_batch(y0_batch, p_jax, 0.05)

    return f
