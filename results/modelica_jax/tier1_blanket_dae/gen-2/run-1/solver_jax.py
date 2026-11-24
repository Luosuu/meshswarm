"""Deliverable JAX GPU ensemble solver for BlanketValveDAE.

Strategy
--------
The Modelica model reduces to a 6-state ODE once the single algebraic
Sieverts-flux constraint is eliminated:

  0 = z + alpha_nl*z*|z| + beta_nl*z^3 - u_ext*k_perm*N_blanket

This scalar equation is solved with a Newton iteration wrapped in a
``@jax.custom_jvp`` rule implementing the Implicit Function Theorem, so reverse-mode
autodiff never unrolls the Newton loop.  The remaining dynamics are integrated with a
stiff SDIRK method (diffrax Kvaerno5, adaptive PID controller) batched with ``jax.vmap``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# --- Model constants (copied from BlanketValveDAE.mo; file is never modified) ---
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

_T0 = 0.0
_T1 = 20.0


def _sieverts_residual(z, S):
    return z + ALPHA_NL * z * jnp.abs(z) + BETA_NL * z * z * z - S


def _sieverts_residual_prime(z):
    return 1.0 + 2.0 * ALPHA_NL * jnp.abs(z) + 3.0 * BETA_NL * z * z


@jax.custom_jvp
def _solve_z(S):
    # Newton solve for z in  z + a|z|z + b z^3 = S.  S >= 0 in this model, but the
    # solver is sign-agnostic.  Initial guess: the linear approximation.
    z0 = S / (1.0 + ALPHA_NL * jnp.abs(S) + BETA_NL * S * S)

    def cond(carry):
        z, i = carry
        return (jnp.abs(_sieverts_residual(z, S)) > 1e-18) & (i < 60)

    def body(carry):
        z, i = carry
        z = z - _sieverts_residual(z, S) / _sieverts_residual_prime(z)
        return (z, i + 1)

    z, _ = jax.lax.while_loop(cond, body, (z0, 0))
    return z


@_solve_z.defjvp
def _solve_z_jvp(primals, tangents):
    (S,), (S_dot,) = primals, tangents
    z = _solve_z(S)
    # Implicit Function Theorem: dz/dS = 1 / f'(z)
    z_dot = S_dot / _sieverts_residual_prime(z)
    return z, z_dot


def _rhs(t, y, p):
    q, T_b, T_hx, N_b, N_e, N_s = y
    u_pump, u_ext, P_pulse, dp_back = p

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

    dT_b = (P_fus - q_eff * CP_COOL * (T_b - T_hx)) / C_BLANKET
    dT_hx = (q_eff * CP_COOL * (T_b - T_hx) - H_SINK * (T_hx - T_SINK)) / C_HX

    z_flux = _solve_z(u_ext * K_PERM * N_b)
    j_store = K_REGEN * N_e

    dN_b = -z_flux
    dN_e = z_flux - j_store
    dN_s = j_store

    return jnp.stack([dq, dT_b, dT_hx, dN_b, dN_e, dN_s])


def _solve_one(y0, p, rtol=1e-11, atol=1e-12):
    import diffrax

    term = diffrax.ODETerm(_rhs)
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=_T0,
        t1=_T1,
        dt0=0.02,
        y0=y0,
        args=p,
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=200000,
    )
    return sol.ys[-1]


_solve_one_vmap = jax.vmap(_solve_one, in_axes=(0, 0))


def forward_ensemble(y0_batch, p_batch) -> jax.Array:
    y0 = jnp.asarray(y0_batch, dtype=jnp.float64)
    p = jnp.asarray(p_batch, dtype=jnp.float64)
    return _solve_one_vmap(y0, p)


def differentiable_closure(y0_batch, p_ref_batch):
    y0 = jnp.asarray(y0_batch, dtype=jnp.float64)
    p_ref = jnp.asarray(p_ref_batch, dtype=jnp.float64)

    # Closure over constants; p_jax must stay a traced argument.
    def closure(p_jax):
        p = jnp.asarray(p_jax, dtype=jnp.float64)
        return jax.vmap(_solve_one, in_axes=(0, 0))(y0, p)

    return closure
