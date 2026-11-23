"""Reference JAX solver for Tier 1 (`BlanketValveDAE.mo`)."""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("GRADSOLVE_X64", "1")

import gradsolve  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

ALPHA_NL = 0.65
BETA_NL = 0.30
EPS_Z = 1.0e-7


@jax.custom_jvp
def solve_sieverts_flux(rhs_drive: jax.Array) -> jax.Array:
    z = rhs_drive / (1.0 + jnp.cbrt(rhs_drive * rhs_drive + 1.0e-12))
    for _ in range(7):
        az = jnp.sqrt(z * z + EPS_Z * EPS_Z)
        f = z + ALPHA_NL * z * az + BETA_NL * z * z * z - rhs_drive
        df = 1.0 + ALPHA_NL * (2.0 * z * z + EPS_Z * EPS_Z) / az + 3.0 * BETA_NL * z * z
        z = z - f / df
    return z


@solve_sieverts_flux.defjvp
def _sieverts_jvp(primals, tangents):
    (r,), (dr,) = primals, tangents
    z = solve_sieverts_flux(r)
    az = jnp.sqrt(z * z + EPS_Z * EPS_Z)
    df = 1.0 + ALPHA_NL * (2.0 * z * z + EPS_Z * EPS_Z) / az + 3.0 * BETA_NL * z * z
    return z, dr / df


class _Tier1Problem:
    name = "user_tier1_blanket_valve_dae"
    dim = 6
    t0 = 0.0
    t1 = 20.0
    is_stiff = True

    def f_jax(self, t, y, p):
        u_pump, u_ext, p_pulse, dp_back = p[0], p[1], p[2], p[3]
        q_flow, t_bl, t_hx, n_bl, n_ex, _n_st = y[0], y[1], y[2], y[3], y[4], y[5]

        p_fus = 115.0 * (1.0 + 0.40 * p_pulse * jnp.sin(0.20 * t))
        dp_drive = 2.0 * u_pump - dp_back * (1.0 + 0.25 * jnp.cos(0.35 * t))

        eps_q = 1.0e-5
        q_abs = jnp.sqrt(q_flow * q_flow + eps_q * eps_q)
        w_pos = 0.5 * (1.0 + q_flow / q_abs)
        drag_valve = (w_pos * 0.24 - (1.0 - w_pos) * 14.0) * q_flow * q_flow

        dp_abs = jnp.sqrt(dp_drive * dp_drive + 1.0e-10)
        w_dp = 0.5 * (1.0 + dp_drive / dp_abs)
        dp_pos = 0.5 * (dp_drive + dp_abs)
        dp_neg = 0.5 * (-dp_drive + dp_abs)
        q_byp = w_dp * 0.08 * jnp.sqrt(dp_pos + 1.0e-8) - (1.0 - w_dp) * 0.15 * 0.08 * jnp.sqrt(dp_neg + 1.0e-8)

        dq = (dp_drive - drag_valve - 0.30 * q_flow) / 0.02

        q_pos_part = 0.55 * q_abs + 0.45 * q_flow
        q_byp_pos = 0.5 * (q_byp + jnp.sqrt(q_byp * q_byp + 1.0e-8))
        q_eff = q_pos_part + q_byp_pos

        dt_bl = (p_fus - q_eff * 1.85 * (t_bl - t_hx)) / 26.0
        dt_hx = (q_eff * 1.85 * (t_bl - t_hx) - 0.90 * (t_hx - 570.0)) / 11.0

        z_flux = solve_sieverts_flux(u_ext * 0.042 * n_bl)
        j_store = 0.050 * n_ex
        return jnp.stack([dq, dt_bl, dt_hx, -z_flux, z_flux - j_store, j_store])


_PROB = _Tier1Problem()
_DEV = "gpu" if any(d.platform == "gpu" for d in jax.devices()) else "cpu"


def forward_ensemble(y0_batch: np.ndarray, p_batch: np.ndarray) -> jax.Array:
    res = gradsolve.solve(
        _PROB, np.asarray(y0_batch, dtype=np.float64), np.asarray(p_batch, dtype=np.float64),
        engine="rodas5p_replay", rtol=1e-8, atol=1e-10, device=_DEV,
    )
    return jnp.asarray(res.y_final)


def differentiable_closure(y0_batch: np.ndarray, p_ref_batch: np.ndarray):
    return gradsolve.grad_closure(
        _PROB, np.asarray(y0_batch, dtype=np.float64), np.asarray(p_ref_batch, dtype=np.float64),
        engine="rodas5p_replay", rtol=1e-8, atol=1e-10, device=_DEV,
    )
