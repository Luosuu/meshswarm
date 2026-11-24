"""Deliverable JAX GPU ensemble solver for BlanketValveDAE.

Approach
--------
The reference Modelica DAE is integrated as a semi-explicit ODE in the 6 differential
states ``y = [q_flow, T_blanket, T_hx, N_blanket, N_extractor, N_storage]``:

* The algebraic Sieverts permeation flux ``z_flux`` solves the stiff cubic
  ``z + a*z*|z| + b*z^3 = u_ext*k_perm*N_blanket`` with a Newton iteration implemented
  as ``jax.lax.fori_loop`` over fixed iterations (cubic with positive coefficients is
  strictly monotone, so Newton from ``z0 = rhs`` converges monotonically and is safe).
  The root finder carries a ``@jax.custom_jvp`` implementing the Implicit Function
  Theorem, so the Rosenbrock step's ``jax.jacfwd`` state Jacobian and all reverse-mode
  AD see the exact implicit derivative instead of differentiating through the Newton
  iterations.
* The non-smooth branches (check-valve ``q_flow >= 0``, orifice bypass sign) use
  ``jnp.where`` on the traceable values, matching the Modelica ``if`` equations.
* Time integration uses ``gradsolve``'s Rodas5P (Rosenbrock-W, order 5(4)) adaptive
  stiff solver: ``record_rodas5p`` records the accepted mesh on the device (vmapped
  ``lax.while_loop``) and returns the final states directly for ``forward_ensemble``;
  ``differentiable_closure`` replays the mesh frozen at the reference parameters via
  ``make_rodas5p_frozen_mesh_kernel`` so ``jax.grad`` / ``jax.jvp`` differentiate the
  same replay program (exact discrete adjoint), with the Sieverts solve handled by the
  custom JVP.

FP64 is enabled at import. No subprocess/omc/stop_gradient anywhere.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Constants from BlanketValveDAE.mo (unchanged).
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

T0 = 0.0
T1 = 20.0

# Newton iteration count for the cubic Sieverts flux. The residual is strictly
# monotone (derivative >= 1), z0 = rhs > 0 is an upper bound on the root when
# rhs > 0 and a lower bound when rhs < 0; 60 quadratic steps are far more than
# enough to reach float64 precision from the default initial state.
Z_NEWTON_ITERS = 60


@jax.custom_jvp
def _z_flux(u_ext, N_blanket):
    """Implicit cubic Sieverts flux z:  z + a*z*|z| + b*z^3 = u_ext*k_perm*N_blanket."""
    rhs = u_ext * K_PERM * N_blanket

    def body(i, z):
        f = z + ALPHA_NL * z * jnp.abs(z) + BETA_NL * z * z * z - rhs
        fp = 1.0 + 2.0 * ALPHA_NL * jnp.abs(z) + 3.0 * BETA_NL * z * z
        return z - f / fp

    # Bracket-free: Newton from rhs is monotone-convergent for this monotone cubic.
    z0 = rhs
    z = jax.lax.fori_loop(0, Z_NEWTON_ITERS, body, z0)
    return z


@_z_flux.defjvp
def _z_flux_jvp(primals, tangents):
    u_ext, N_blanket = primals
    du_ext, dN_blanket = tangents
    z = _z_flux(u_ext, N_blanket)
    denom = 1.0 + 2.0 * ALPHA_NL * jnp.abs(z) + 3.0 * BETA_NL * z * z
    dz = (K_PERM * N_blanket * du_ext + u_ext * K_PERM * dN_blanket) / denom
    return z, dz


def rhs_f(t, y, p):
    """Per-trajectory RHS (gradsolve contract: y (6,), p (4,))."""
    u_pump, u_ext, P_pulse, dp_back = p[0], p[1], p[2], p[3]
    q, Tb, Th, Nb, Ne, Ns = y[0], y[1], y[2], y[3], y[4], y[5]

    P_fus = P_BASE * (1.0 + 0.40 * P_pulse * jnp.sin(0.20 * t))
    dp_drive = 2.0 * u_pump - dp_back * (1.0 + 0.25 * jnp.cos(0.35 * t))

    drag_valve = jnp.where(q >= 0.0, K_FWD * q * q, -K_REV * q * q)
    q_byp = jnp.where(dp_drive >= 0.0,
                      K_BYP * jnp.sqrt(jnp.abs(dp_drive) + 1.0e-8),
                      -0.15 * K_BYP * jnp.sqrt(jnp.abs(dp_drive) + 1.0e-8))

    dq = (dp_drive - drag_valve - 0.30 * q) / TAU_HYD

    q_eff = (jnp.where(q >= 0.0, q, -0.10 * q)
             + jnp.where(q_byp >= 0.0, q_byp, 0.0))
    dTb = (P_fus - q_eff * CP_COOL * (Tb - Th)) / C_BLANKET
    dTh = (q_eff * CP_COOL * (Tb - Th) - H_SINK * (Th - T_SINK)) / C_HX

    z = _z_flux(u_ext, Nb)
    j_store = K_REGEN * Ne

    return jnp.stack([dq, dTb, dTh, -z, z - j_store, j_store])


class _Problem:
    """gradsolve-compatible problem descriptor (duck-typed)."""

    name = "BlanketValveDAE_jax"
    dim = 6
    t0 = T0
    t1 = T1
    is_stiff = True

    @staticmethod
    def f_jax(t, y, p):
        return rhs_f(t, y, p)


_PROBLEM = _Problem()

# Solver tolerances. Forward trajectory parity gate is 2.5e-4 relative to DASSL; we
# integrate at much tighter tolerance for margin. The closure record uses an even
# tighter tolerance so frozen-mesh gradients are accurate.
RTOL_FWD = 1e-8
ATOL_FWD = 1e-11
RTOL_GRAD = 1e-9
ATOL_GRAD = 1e-12
MAX_STEPS = 50000


def forward_ensemble(y0_batch, p_batch):
    """Integrate the ensemble to t = 20.0 and return (B, 6) observables."""
    from gradsolve.solvers.rodas5p_replay import record_rodas5p

    y0 = np.asarray(y0_batch, dtype=np.float64)
    p = np.asarray(p_batch, dtype=np.float64)
    if y0.ndim == 1:
        y0 = y0[None, :]
    if p.ndim == 1:
        p = p[None, :]
    yf, _dts, _n_acc, _n_rej = record_rodas5p(
        rhs_f, y0, p, T0, T1, rtol=RTOL_FWD, atol=ATOL_FWD, max_steps=MAX_STEPS)
    return jnp.asarray(yf)


def differentiable_closure(y0_batch, p_ref_batch):
    """Return f(p_jax) -> (B, 6) obs, jax.grad/jvp differentiable around p_ref_batch."""
    from gradsolve.solvers.rodas5p_replay import make_rodas5p_frozen_mesh_kernel

    y0 = np.asarray(y0_batch, dtype=np.float64)
    p_ref = np.asarray(p_ref_batch, dtype=np.float64)
    if y0.ndim == 1:
        y0 = y0[None, :]
    if p_ref.ndim == 1:
        p_ref = p_ref[None, :]
    kernel = make_rodas5p_frozen_mesh_kernel(
        _PROBLEM, y0, p_ref, rtol=RTOL_GRAD, atol=ATOL_GRAD, max_steps=MAX_STEPS)
    y0_j = jnp.asarray(y0)

    def closure(p_jax):
        p_jax = jnp.asarray(p_jax, dtype=jnp.float64)
        return kernel(y0_j, p_jax)

    return closure
