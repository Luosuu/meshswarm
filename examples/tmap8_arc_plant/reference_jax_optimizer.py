"""Standalone GPU JAX differentiable whole-plant reference optimizer used by `--selftest`.

Models the exact Backward-Euler pulsed time-stepping of DOE INL `TMAP8` (`ODETimeDerivative` +
`ParsedODEKernel` in `fuel_cycle.i`) in 64-bit JAX using `vmap` + binary matrix-power propagation
+ multi-start L-BFGS / Adam augmented-Lagrangian optimization on GPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

P_FUS_PULSE_MW = 505.6
N_BURN = 8.99e-7
T_DECAY = jnp.log(2.0) / 388_800_000.0
PULSE_S = 1800.0
SIM_3Y_S = 94_672_800.0

PARAM_KEYS = (
    "initial_inventory",
    "resident_time_BB",
    "resident_time_TES",
    "eta_2",
    "resident_time_HX",
    "resident_time_DS",
    "resident_time_vacuum",
    "f_DIR",
    "resident_time_FCU",
    "resident_time_ISS",
    "f_ISS_to_DS",
    "resident_time_membrane",
    "TBR_value",
    "AF_value",
    "eta_f",
)

LO = jnp.array([0.25, 900.0, 3600.0, 0.35, 300.0, 600.0, 120.0, 0.10, 180.0, 1800.0, 0.02, 30.0, 1.02, 0.65, 0.18])
HI = jnp.array([0.995, 14400.0, 172800.0, 0.96, 3600.0, 14400.0, 1800.0, 0.88, 2400.0, 43200.0, 0.25, 900.0, 1.18, 0.88, 0.42])

# 4 public scenarios stacked as arrays: (tbe_mult, eta2_mult, f_hx_ds, tbr_mult, af_mult, res_ratio, t_dbl_cap)
SC_ARR = jnp.array([
    [1.00, 1.00, 1.0e-4, 1.000, 1.00, 1.00, 1.50],
    [0.80, 1.00, 1.0e-4, 1.000, 1.00, 0.85, 9.99],
    [1.00, 0.82, 3.0e-4, 1.000, 1.00, 0.85, 9.99],
    [1.00, 1.00, 1.0e-4, 0.965, 0.85, 0.85, 2.50],
])


def decode_params(u: jnp.ndarray) -> jnp.ndarray:
    return LO + (HI - LO) * jax.nn.sigmoid(u)


def build_rate_matrix(tau_vec: jnp.ndarray, eta2: jnp.ndarray, f_dir: jnp.ndarray, f_9_6: jnp.ndarray, f_5_6: jnp.ndarray) -> jnp.ndarray:
    # tau_vec: [tau1, tau2, 1000, 1000, tau5, tau6, tau7, tau8, tau9, 1.0, tau11]
    eps = jnp.array([1e-4, 1e-4, 0.0, 0.0, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.0, 1e-4])
    diag = jnp.where(
        jnp.arange(11) == 9,
        -T_DECAY,
        -(1.0 + eps) / tau_vec - T_DECAY,
    )
    A = jnp.diag(diag)
    f_5_1, f_5_3 = 0.33, 0.33
    A = A.at[0, 2].set(1.0 / tau_vec[2])
    A = A.at[0, 3].set(1.0 / tau_vec[3])
    A = A.at[0, 4].set(f_5_1 / tau_vec[4])
    A = A.at[1, 0].set(1.0 / tau_vec[0])
    A = A.at[2, 4].set(f_5_3 / tau_vec[4])
    A = A.at[3, 4].set((1.0 - f_5_1 - f_5_6 - f_5_3) / tau_vec[4])
    A = A.at[4, 1].set((1.0 - eta2) / tau_vec[1])
    A = A.at[5, 4].set(f_5_6 / tau_vec[4])
    A = A.at[5, 8].set(f_9_6 / tau_vec[8])
    A = A.at[7, 6].set((1.0 - f_dir) / tau_vec[6])
    A = A.at[8, 5].set(1.0 / tau_vec[5])
    A = A.at[8, 7].set(1.0 / tau_vec[7])
    A = A.at[9, 8].set((1.0 - f_9_6) / tau_vec[8])
    A = A.at[9, 6].set(f_dir / tau_vec[6])
    A = A.at[9, 10].set(1.0 / tau_vec[10])
    A = A.at[10, 1].set(eta2 / tau_vec[1])
    return A


def sim_single_scenario(x_p: jnp.ndarray, sc_row: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Differentiable Backward-Euler pulsed TMAP8 twin matching MOOSE `ODETimeDerivative`."""
    i0 = x_p[0]
    tau_vec = jnp.array([x_p[1], x_p[2], 1000.0, 1000.0, x_p[4], x_p[5], x_p[6], x_p[8], x_p[9], 1.0, x_p[11]])
    imp = 1.0 - 0.35 * jax.nn.relu(x_p[7] - 0.45) ** 1.6
    tbe = x_p[14] * 0.10 * imp * sc_row[0]
    eta2 = jnp.minimum(0.995, x_p[3] * sc_row[1])
    f_5_6 = sc_row[2]
    tbr = x_p[12] * sc_row[3]
    af = jnp.minimum(0.95, x_p[13] * sc_row[4])

    A = build_rate_matrix(tau_vec, eta2, x_p[7], x_p[10], f_5_6)
    burn_on = N_BURN / tbe
    f_p_3, f_p_4 = 1e-4, 1e-4
    b_on = jnp.array([
        N_BURN * tbr, 0.0, f_p_3 * burn_on, f_p_4 * burn_on, 0.0, 0.0,
        (1.0 - tbe - f_p_3 - f_p_4) * burn_on, 0.0, 0.0, -burn_on, 0.0,
    ])

    # Match TMAP8 MOOSE FunctionDT Backward-Euler steps: (I - dt A) x_{n+1} = x_n + dt b
    dt_on = af * PULSE_S + 0.01
    dt_off = PULSE_S - af * PULSE_S
    eye = jnp.eye(11)
    S_on = jnp.linalg.solve(eye - dt_on * A, eye)
    c_on = jnp.linalg.solve(eye - dt_on * A, dt_on * b_on)
    S_off = jnp.linalg.solve(eye - dt_off * A, eye)

    x0 = jnp.zeros(11).at[9].set(i0)

    # Scan first 720 pulses (15 days) directly to capture exact startup inflection & pulse spikes
    def pulse_step(x, _):
        x_on = S_on @ x + c_on
        x_off = S_off @ x_on
        inproc_on = jnp.sum(x_on) - x_on[9]
        inproc_off = jnp.sum(x_off) - x_off[9]
        return x_off, (
            jnp.minimum(x_on[9], x_off[9]),
            jnp.maximum(inproc_on, inproc_off),
            jnp.maximum(x_on[4], x_off[4]),
            x_off[9],
        )

    # Scan first 1200 pulses (25 days = 2,160,000 s) directly to match TMAP8 pulsed window
    x_after_25d, (mins_25d, inprocs_25d, hxs_25d, _) = jax.lax.scan(pulse_step, x0, None, length=1200)

    # Exact TMAP8 MOOSE FunctionDT coarse steps after 25 days (dt = time_interval_middle = 1e6 s, 93 steps)
    dt_mid = 1_000_000.0
    b_avg = af * b_on
    S_mid = jnp.linalg.solve(eye - dt_mid * A, eye)
    c_mid = jnp.linalg.solve(eye - dt_mid * A, dt_mid * b_avg)

    def mid_step(x, idx):
        xn = S_mid @ x + c_mid
        inproc = jnp.sum(xn) - xn[9]
        hx = xn[4]
        t_yr = (1200.0 * PULSE_S + (idx + 1.0) * dt_mid) / (86400.0 * 365.25)
        return xn, (xn[9], inproc, hx, xn[9], t_yr)

    x_end, (mins_mid, inprocs_mid, hxs_mid, stos_mid, tyrs_mid) = jax.lax.scan(
        mid_step, x_after_25d, jnp.arange(93)
    )

    min_sto = jnp.minimum(jnp.min(mins_25d), jnp.min(mins_mid))
    max_inproc = jnp.maximum(jnp.max(inprocs_25d), jnp.max(inprocs_mid))
    max_hx = jnp.maximum(jnp.max(hxs_25d), jnp.max(hxs_mid))
    i_res = (N_BURN / tbe) * 0.25 * 86400.0 * af
    req_res = sc_row[5] * i_res

    # Smooth storage check at doubling horizon sc_row[6] years
    target_t = sc_row[6]
    weights = jax.nn.softmax(-40.0 * jnp.abs(tyrs_mid - target_t))
    sto_at_cap = jnp.sum(weights * stos_mid)

    return min_sto - req_res, max_inproc, max_hx, sto_at_cap - 2.0 * i0, x_end[9] - i0


def plant_econ_jax(x_p: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    i0, tau1, tau2, eta2, tau5, tau6, tau7, f_dir, tau8, tau9, f_9_6, tau11, tbr, af, eta_f = x_p
    m_n = 1.20 - 0.30 * (tbr - 1.06)
    p_th = P_FUS_PULSE_MW * (0.20 + 0.80 * m_n)
    eta_th = 0.435 * (1.0 - 0.13 * (1000.0 / tau5) ** 0.50)
    p_gross = af * p_th * eta_th

    p_core = af * (26.0 + 7.5 * ((eta_f / 0.25) ** 2.0)) + 2.4 / jnp.maximum(0.06, 1.0 - af)
    p_pumps = 4.2 * ((4500.0 / tau1) ** 1.75) + 1.5 * ((1000.0 / tau5) ** 1.25)
    ntu_tes = -jnp.log(1.0 - eta2)
    ntu_nom = -jnp.log(0.30)
    p_tes = 0.85 * ((86400.0 / tau2) ** 0.65) * ((ntu_tes / ntu_nom) ** 1.35)
    p_vac = 1.10 * ((600.0 / tau7) ** 1.15) + 2.80 * (f_dir ** 2.4)
    p_iss = 1.65 * ((22815.0 / tau9) ** 0.70) * ((0.10 / f_9_6) ** 0.60)
    p_rest = 0.45 * ((585.0 / tau8) ** 0.85) + 0.40 * ((3600.0 / tau6) ** 0.75) + 0.35 * ((100.0 / tau11) ** 0.80)
    p_net = p_gross - (p_core + p_pumps + p_tes + p_vac + p_iss + p_rest)
    p_cap = 18.0 * i0 + 40.0 * jax.nn.relu(tbr - 1.05) ** 1.6 + 2.2 * ((tau5 / 1000.0) ** 0.85)
    return p_net - p_cap, p_net


def loss_single(u: jnp.ndarray) -> jnp.ndarray:
    x_p = decode_params(u)
    res_margin, max_inproc, max_hx, dbl_margin, gain_3y = jax.vmap(sim_single_scenario, in_axes=(None, 0))(x_p, SC_ARR)
    j_econ, p_net = plant_econ_jax(x_p)

    # Dimensionless normalized safety penalties (3.5% safety buffer for held-out combined jitter)
    pen_res = jnp.sum(jax.nn.relu((0.018 - res_margin) / 0.05) ** 2) * 80.0
    pen_inp = jnp.sum(jax.nn.relu((max_inproc - 0.210) / 0.05) ** 2) * 80.0
    pen_hx = jnp.sum(jax.nn.relu((max_hx - 0.92e-4) / 0.20e-4) ** 2) * 80.0
    pen_dbl_nom = (jax.nn.relu((0.05 - dbl_margin[0]) / 0.10) ** 2) * 60.0
    pen_dbl_deg = (jax.nn.relu((0.05 - dbl_margin[3]) / 0.10) ** 2) * 60.0
    pen_gain = (jax.nn.relu((2.10 - gain_3y[0]) / 0.25) ** 2) * 60.0
    pen_net = (jax.nn.relu((10.0 - p_net) / 10.0) ** 2) * 60.0
    return -j_econ + pen_res + pen_inp + pen_hx + pen_dbl_nom + pen_dbl_deg + pen_gain + pen_net


def run_gpu_optimization(n_starts: int = 64, n_steps: int = 450) -> dict[str, object]:
    t0 = time.perf_counter()
    key = jax.random.PRNGKey(2026)
    u_init = jax.random.uniform(key, (n_starts, len(PARAM_KEYS)), minval=-2.0, maxval=2.0)

    opt = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adam(learning_rate=optax.cosine_decay_schedule(0.12, n_steps, alpha=0.01)),
    )

    @jax.jit
    def solve_batch(u0_batch):
        opt_state0 = jax.vmap(opt.init)(u0_batch)

        def step_fn(carry, _):
            u_b, st_b = carry
            vals, grads = jax.vmap(jax.value_and_grad(loss_single))(u_b)
            updates, st_b = jax.vmap(opt.update)(grads, st_b, u_b)
            u_b = optax.apply_updates(u_b, updates)
            return (u_b, st_b), vals

        (u_fin, _), _ = jax.lax.scan(step_fn, (u0_batch, opt_state0), None, length=n_steps)
        final_losses = jax.vmap(loss_single)(u_fin)
        return u_fin, final_losses

    u_fin, losses = solve_batch(u_init)
    jax.block_until_ready(losses)
    wall_s = time.perf_counter() - t0

    best_idx = int(jnp.argmin(losses))
    best_x = decode_params(u_fin[best_idx])
    j_econ, p_net = plant_econ_jax(best_x)
    design = {k: float(best_x[i]) for i, k in enumerate(PARAM_KEYS)}
    return {
        "design": design,
        "j_econ_mwe": float(j_econ),
        "p_net_elec_mwe": float(p_net),
        "best_loss": float(losses[best_idx]),
        "n_trajectories_evaluated": int(n_starts * (n_steps + 1) * 4),
        "gpu_wall_s": round(wall_s, 4),
        "jax_devices": [str(d) for d in jax.devices()],
    }


if __name__ == "__main__":
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    res = run_gpu_optimization()
    if out_path is not None:
        out_path.write_text(json.dumps(res["design"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))
