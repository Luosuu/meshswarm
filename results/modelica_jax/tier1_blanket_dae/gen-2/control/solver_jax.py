"""JAX GPU ensemble solver for the stiff fusion blanket check-valve & implicit Sieverts DAE.

Strategy
--------
The Modelica DAE is reduced analytically to a 5-state ODE by
  1. eliminating the implicit cubic Sieverts algebraic equation for the permeation flux
     `z_flux` with a Newton solve whose derivative is registered with @jax.custom_jvp
     (implicit function theorem), so reverse-mode AD never differentiates through the
     iteration, and
  2. dropping the `der(N_storage) = j_store` equation: total closed-loop tritium inventory
     N_total = N_blanket + N_extractor + N_storage is exactly conserved by the DAE
     (der(N_blanket) + der(N_extractor) + der(N_storage) = 0), so
     N_storage = N_total0 - N_blanket - N_extractor with N_total0 = sum(y0[3:6]).
     Conservation is then exact by construction rather than by solver tolerance.

Integration uses gradsolve's `rodas5p_replay` (Rodas5P Rosenbrock) engine on GPU; the
per-step `jax.jacfwd` Jacobians of the reduced RHS are cheap (dim=5). `differentiable_closure`
uses `gradsolve.grad_closure`, which records the adaptive Rodas5P mesh at `p_ref` once and
returns a jax.grad-differentiable frozen-mesh replay; the Sieverts solve differentiates
through its custom JVP (implicit function theorem).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import gradsolve

TAU_HYD = 0.02
K_FWD, K_REV, K_BYP = 0.24, 14.0, 0.08
C_BLANK, C_HX, CP_COOL = 26.0, 11.0, 1.85
H_SINK, T_SINK = 0.90, 570.0
P_BASE = 115.0
K_PERM, ALPHA_NL, BETA_NL = 0.042, 0.65, 0.30
K_REGEN = 0.050
T_END = 20.0
EPS = 1.0e-8


@jax.custom_jvp
def sieverts_flux(u_ext, n_blanket):
    """Solve z + alpha*z*|z| + beta*z^3 = u_ext * k_perm * n_blanket (cubic in z)."""
    c = u_ext * K_PERM * n_blanket
    ac = jnp.abs(c)
    # robust initial guess: linear vs cubic scaling
    z = jnp.sign(c) * jnp.minimum(ac, (ac / BETA_NL) ** (1.0 / 3.0))

    def body(_, z):
        fz = z + ALPHA_NL * z * jnp.abs(z) + BETA_NL * z**3 - c
        dfz = 1.0 + 2.0 * ALPHA_NL * jnp.abs(z) + 3.0 * BETA_NL * z**2
        return z - fz / dfz

    return jax.lax.fori_loop(0, 60, body, z)


@sieverts_flux.defjvp
def _sieverts_flux_jvp(primals, tangents):
    u_ext, n_blanket = primals
    du, dn = tangents
    z = sieverts_flux(u_ext, n_blanket)
    # implicit function theorem: dz = dc / (dF/dz)
    dfz = 1.0 + 2.0 * ALPHA_NL * jnp.abs(z) + 3.0 * BETA_NL * z**2
    dc = K_PERM * (n_blanket * du + u_ext * dn)
    return z, dc / dfz


def _rhs5(t, y, p):
    u_pump, u_ext, p_pulse, dp_back = p
    q, t_blanket, t_hx, n_blanket, n_extractor = y

    p_fus = P_BASE * (1.0 + 0.40 * p_pulse * jnp.sin(0.20 * t))
    dp_drive = 2.0 * u_pump - dp_back * (1.0 + 0.25 * jnp.cos(0.35 * t))

    drag_valve = jnp.where(q >= 0.0, K_FWD * q * q, -K_REV * q * q)
    q_byp = jnp.where(
        dp_drive >= 0.0,
        K_BYP * jnp.sqrt(jnp.maximum(dp_drive, 0.0) + EPS),
        -0.15 * K_BYP * jnp.sqrt(jnp.maximum(-dp_drive, 0.0) + EPS),
    )
    dq_flow = (dp_drive - drag_valve - 0.30 * q) / TAU_HYD

    q_eff = jnp.where(q >= 0.0, q, -0.10 * q) + jnp.where(q_byp >= 0.0, q_byp, 0.0)
    dt_blanket = (p_fus - q_eff * CP_COOL * (t_blanket - t_hx)) / C_BLANK
    dt_hx = (q_eff * CP_COOL * (t_blanket - t_hx) - H_SINK * (t_hx - T_SINK)) / C_HX

    z_flux = sieverts_flux(u_ext, n_blanket)
    j_store = K_REGEN * n_extractor

    dn_blanket = -z_flux
    dn_extractor = z_flux - j_store
    return jnp.array([dq_flow, dt_blanket, dt_hx, dn_blanket, dn_extractor])


class _BlanketValveProblem:
    name = "BlanketValveDAE"
    dim = 5
    t0 = 0.0
    t1 = T_END

    @property
    def is_stiff(self):
        return True

    def f_jax(self, t, y, params):
        return _rhs5(t, y, params)


_PROBLEM = _BlanketValveProblem()

_RTOL = 1e-10
_ATOL = 1e-12


def _final_observables(y5, n_total0):
    """`y5` may be (..., 5) = [q_flow, T_blanket, T_hx, N_blanket, N_extractor]."""
    n_blanket = y5[..., 3]
    n_extractor = y5[..., 4]
    n_storage = n_total0 - n_blanket - n_extractor
    return jnp.stack(
        [y5[..., 0], y5[..., 1], y5[..., 2], n_blanket, n_extractor, n_storage], axis=-1
    )


def _as_batch(a):
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def forward_ensemble(y0_batch, p_batch):
    y0 = _as_batch(y0_batch)
    p = _as_batch(p_batch)
    n_total0 = jnp.asarray(y0[:, 3] + y0[:, 4] + y0[:, 5], dtype=jnp.float64)
    y5 = jnp.asarray(y0[:, :5], dtype=jnp.float64)
    res = gradsolve.solve(
        _PROBLEM, y5, p, rtol=_RTOL, atol=_ATOL, engine="rodas5p_replay", device="cuda"
    )
    y_final = jnp.asarray(res.y_final)
    return _final_observables(y_final, n_total0)


def differentiable_closure(y0_batch, p_ref_batch):
    y0 = _as_batch(y0_batch)
    p_ref = _as_batch(p_ref_batch)
    n_total0 = jnp.asarray(y0[:, 3] + y0[:, 4] + y0[:, 5], dtype=jnp.float64)
    y5 = jnp.asarray(y0[:, :5], dtype=jnp.float64)

    base = gradsolve.grad_closure(
        _PROBLEM,
        y5,
        p_ref,
        wrt="params",
        rtol=_RTOL,
        atol=_ATOL,
        engine="rodas5p_replay",
        device="cuda",
    )

    def obs_final_jax(p_jax):
        p_jax = jnp.asarray(p_jax, dtype=jnp.float64)
        y_final = base(p_jax)
        return _final_observables(y_final, n_total0)

    return obs_final_jax
