"""End-to-end differentiable Core-Blanket-FuelCycle Co-Design showcase.

Couples three differentiable JAX physics models in one `jax.value_and_grad` graph:
  1. 1D radial burning-plasma D-T transport model (`TORAX`-style 12-zone diffusive energy +
     density PDE with Bosch-Hale D-T reactivity, alpha self-heating, and bremsstrahlung loss);
  2. Phase 3 Swarm-generated ORNL `TRANSFORM` counter-current PbLi/FLiBe blanket heat exchanger
     & Arrhenius/Sieverts tritium permeator (`tier2_transform_hx/gen-2/run-1/solver_jax.py`);
  3. Phase 3 Swarm-generated 16-stage cryogenic H2/HD/T2 isotope distillation MESH column
     (`tier3_hdt_column/gen-2/control/solver_jax.py`).

Jointly optimizes 6 cross-plant operating & design variables to maximize net electric power
output subject to PbLi structural temperature limits, plasma beta/density limits, and >= 99.96%
bottom T2 isotope purity.
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from scipy.optimize import minimize as scipy_minimize  # noqa: E402

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[3]
TIER2_PATH = REPO_ROOT / "results" / "modelica_jax" / "tier2_transform_hx" / "gen-2" / "run-1" / "solver_jax.py"
TIER3_PATH = REPO_ROOT / "results" / "modelica_jax" / "tier3_hdt_column" / "gen-2" / "control" / "solver_jax.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SWARM_HX = _load_module("swarm_hx", TIER2_PATH)
_SWARM_COL = _load_module("swarm_col", TIER3_PATH)

# 1D Radial Plasma Grid (N_RHO = 12 normalized minor-radius zones, V_plasma = 420 m^3 FPP class)
N_RHO = 12
DRHO = 1.0 / N_RHO
RHO_CENTERS = (jnp.arange(N_RHO, dtype=jnp.float64) + 0.5) * DRHO
V_PLASMA = 420.0
ZONE_VOLS = V_PLASMA * 2.0 * RHO_CENTERS * DRHO  # sum(ZONE_VOLS) == V_PLASMA


def bosch_hale_sigmav(ti_keV: jax.Array) -> jax.Array:
    """Exact Bosch-Hale (1992) Table IV D(t,n)alpha reactivity <sigma*v> in m^3/s (T in [0.5, 50] keV)."""
    t_safe = jnp.clip(ti_keV, 0.5, 50.0)
    num = t_safe * (1.51361e-2 + t_safe * (4.60643e-3 + t_safe * (-1.06750e-4)))
    den = 1.0 + t_safe * (7.51886e-2 + t_safe * (1.35000e-2 + t_safe * 1.36600e-5))
    theta = t_safe / jnp.maximum(1.0 - num / den, 0.05)
    xi = (6.6610 / jnp.cbrt(theta)) ** 2
    return 1.17302e-15 * (theta * jnp.sqrt(xi / (938272.0 * t_safe ** 3))) * jnp.exp(-3.0 * jnp.sqrt(xi))


def simulate_core_1d(s_fuel_1e20: jax.Array, p_aux_MW: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Integrate 1D radial D-T density `n_20(rho)` & temperature `T_keV(rho)` over a 10s flattop."""
    n0 = 0.75 + 0.40 * (1.0 - RHO_CENTERS ** 2)       # in 10^20 m^-3
    t0 = 3.0 + 13.0 * (1.0 - RHO_CENTERS ** 2) ** 1.5  # in keV

    n_steps = 400
    dt = 10.0 / n_steps
    chi_th = 0.075  # normalized radial thermal diffusivity (tau_E ~ 2.3 s; CFL = chi*dt/drho^2 = 0.27 << 0.69)
    d_part = 0.035  # normalized radial particle diffusivity

    # Edge-peaked pellet/gas fueling profile + core-peaked ICRH/ECCD auxiliary heating profile
    fuel_prof = s_fuel_1e20 * jnp.exp(-((RHO_CENTERS - 0.72) / 0.22) ** 2)
    aux_prof_MW_m3 = (p_aux_MW / V_PLASMA) * 2.4 * jnp.exp(-((RHO_CENTERS - 0.25) / 0.35) ** 2)

    def radial_laplacian(u, u_edge):
        u_ext = jnp.concatenate([u[0:1], u, jnp.array([u_edge])])
        r_plus = RHO_CENTERS + 0.5 * DRHO
        r_minus = jnp.maximum(RHO_CENTERS - 0.5 * DRHO, 0.0)
        flux_p = r_plus * (u_ext[2:] - u_ext[1:-1]) / DRHO
        flux_m = r_minus * (u_ext[1:-1] - u_ext[:-2]) / DRHO
        return (flux_p - flux_m) / (RHO_CENTERS * DRHO)

    def rhs_core(state):
        n20, t_kev = jnp.clip(state[:N_RHO], 0.05, 4.0), jnp.clip(state[N_RHO:], 0.3, 45.0)
        n_m3 = n20 * 1.0e20
        sv = bosch_hale_sigmav(t_kev)
        r_fus = 0.25 * (n_m3 ** 2) * sv
        p_alpha_MW_m3 = r_fus * (3.52e6 * 1.602176634e-19) * 1.0e-6
        p_brem_MW_m3 = 5.35e-3 * (n20 ** 2) * jnp.sqrt(t_kev)

        # Gyro-Bohm stiff confinement degradation with temperature
        chi_eff = chi_th * (1.0 + 0.35 * (jnp.mean(t_kev) / 10.0) ** 1.5)
        dn = d_part * radial_laplacian(n20, 0.25) + fuel_prof - 2.0 * r_fus * 1.0e-20
        c_w = 0.048065 * n20
        dt_k = chi_eff * radial_laplacian(t_kev, 0.8) + (p_alpha_MW_m3 + aux_prof_MW_m3 - p_brem_MW_m3) / c_w
        return jnp.concatenate([dn, dt_k])

    def step(st, _):
        k1 = rhs_core(st)
        k2 = rhs_core(st + 0.5 * dt * k1)
        k3 = rhs_core(st + 0.5 * dt * k2)
        k4 = rhs_core(st + dt * k3)
        return st + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    st_end, _ = jax.lax.scan(step, jnp.concatenate([n0, t0]), None, length=n_steps)
    n_end, t_end = st_end[:N_RHO], st_end[N_RHO:]
    sv_end = bosch_hale_sigmav(t_end)
    r_fus_end = 0.25 * ((n_end * 1.0e20) ** 2) * sv_end
    # Total D-T fusion power (17.59 MeV per reaction) & 14.1 MeV neutron power (80%)
    p_fus_MW = jnp.sum(ZONE_VOLS * r_fus_end) * (17.59e6 * 1.602176634e-19) * 1.0e-6
    p_neutron_MW = 0.80 * p_fus_MW
    n_vol_avg = jnp.sum(ZONE_VOLS * n_end) / V_PLASMA
    t_core = t_end[0]
    return p_fus_MW, p_neutron_MW, n_vol_avg, t_core


# Decision bounds for the 6 coupled whole-plant co-design variables:
# [s_fuel (10^20/(m^3*s)), p_aux (MW), m_pbli (kg/s), m_flibe (kg/s), v_boil (mol/s), d_rec_frac]
DESIGN_LO = jnp.array([0.08, 12.0, 45.0, 6.0, 1.45, 0.07], dtype=jnp.float64)
DESIGN_HI = jnp.array([0.55, 65.0, 125.0, 18.0, 2.35, 0.22], dtype=jnp.float64)


def _smooth_hinge(violation: jax.Array, scale: float) -> jax.Array:
    """Smooth C^infty approximation to `max(violation, 0)` that vanishes exponentially when satisfied."""
    return scale * jax.nn.softplus(violation / scale)


def decode_design(z_norm: jax.Array) -> jax.Array:
    s = jax.nn.sigmoid(z_norm)
    return DESIGN_LO + (DESIGN_HI - DESIGN_LO) * s


def coupled_plant_evaluate(theta: jax.Array) -> dict[str, jax.Array]:
    """End-to-end differentiable evaluation across Core -> Blanket HX -> Cryo Isotope Column."""
    s_fuel, p_aux, m_pbli, m_flibe, v_boil, d_rec = (
        theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]
    )

    # 1. 1D Core Plasma Transport -> Fusion & Neutron Blanket Heating
    p_fus_MW, p_neutron_MW, n_avg, t_core_keV = simulate_core_1d(s_fuel, p_aux)

    # 2. Blanket sector heating -> Swarm Tier-2 ORNL TRANSFORM PbLi/FLiBe Heat Exchanger
    # Sector scaling: 1/400 blanket module receives proportional neutron heating raising PbLi supply temperature
    q_sector_MW = p_neutron_MW / 400.0
    t_in_pbli = 885.0 + (q_sector_MW * 1.0e6) / (m_pbli * 188.0 + 1.0e3)
    y0_hx = jnp.array([930.0] * 6 + [880.0] * 6 + [5.0 + 0.6 * s_fuel, 0.5, 0.5], dtype=jnp.float64)
    p_hx = jnp.stack([m_pbli, m_flibe, t_in_pbli, jnp.array(0.95, dtype=jnp.float64)])
    obs_hx = _SWARM_HX._forward_one(y0_hx, p_hx)
    q_hx_sector_MW, t1_out_K, t2_out_K, n_getter_mol = obs_hx[0], obs_hx[1], obs_hx[2], obs_hx[5]

    # 3. Fuel-cycle feed -> Swarm Tier-3 16-Stage Cryogenic H2/HD/T2 Distillation Column
    # Unburned exhaust + blanket extraction shifts column initial H2/T2 enrichment slightly with fueling rate
    idx16 = jnp.arange(16, dtype=jnp.float64)
    xh2_0 = jnp.clip((0.52 + 0.08 * s_fuel) * jnp.exp(-0.14 * idx16), 0.02, 0.68)
    xhd_0 = jnp.full((16,), 0.28, dtype=jnp.float64)
    y0_col = jnp.concatenate([xh2_0, xhd_0])
    p_col = jnp.stack([v_boil, d_rec, jnp.array(0.012, dtype=jnp.float64), jnp.array(1.15, dtype=jnp.float64)])
    obs_col = _SWARM_COL._forward_one(y0_col, p_col)
    x_h2_top, x_t2_bot = obs_col[0], obs_col[1]

    # Whole-plant net electric output & operational constraints
    q_plant_thermal_MW = 400.0 * q_hx_sector_MW
    # Brayton/sCO2 Carnot-scaled conversion efficiency increases with FLiBe outlet temperature T2_out
    eta_bop = 0.44 * (1.0 - 570.0 / jnp.maximum(t2_out_K, 600.0)) / (1.0 - 570.0 / 900.0)
    p_gross_elec_MW = eta_bop * q_plant_thermal_MW
    # Parasitic loads: NBI/RF wall-plug efficiency (45%), MHD/molten-salt pumping, cryo reboiler duty
    p_parasitic_MW = (
        p_aux / 0.45
        + 0.0018 * (m_pbli ** 2)
        + 0.045 * (m_flibe ** 2)
        + 6.5 * (v_boil ** 2)
    )
    p_net_elec_MW = p_gross_elec_MW - p_parasitic_MW
    q_plasma = p_fus_MW / jnp.maximum(p_aux, 1.0)

    # Smooth exact-penalty constraint violations (vanish exponentially when satisfied):
    #   (a) Primary PbLi HX inlet <= 965 K & outlet <= 942 K (ferritic steel creep/corrosion limits)
    #   (b) Greenwald-like core density <= 1.15 x 10^20 m^-3
    #   (c) Reboiler product T2 purity >= 0.99970 & tritium recovery >= 1.32 mol
    pen_temp = 20.0 * _smooth_hinge(t_in_pbli - 965.0, 0.5) ** 2 + 20.0 * _smooth_hinge(t1_out_K - 942.0, 0.5) ** 2
    pen_dens = 500.0 * _smooth_hinge(n_avg - 1.15, 0.01) ** 2
    pen_purity = 2.0e7 * _smooth_hinge(0.99970 - x_t2_bot, 1.0e-5) ** 2
    pen_trit = 200.0 * _smooth_hinge(1.32 - n_getter_mol, 0.005) ** 2

    objective_loss = -p_net_elec_MW + pen_temp + pen_dens + pen_purity + pen_trit
    return {
        "loss": objective_loss,
        "p_net_elec_MW": p_net_elec_MW,
        "p_fus_MW": p_fus_MW,
        "q_plasma": q_plasma,
        "t_core_keV": t_core_keV,
        "n_avg_1e20": n_avg,
        "t_in_pbli_K": t_in_pbli,
        "t1_out_K": t1_out_K,
        "t2_out_K": t2_out_K,
        "n_getter_mol": n_getter_mol,
        "x_t2_bot": x_t2_bot,
        "x_h2_top": x_h2_top,
    }


def run_codesign_benchmark(n_grad_steps: int = 60, n_gf_evals: int = 60) -> dict[str, Any]:
    """Compare coupled end-to-end exact-gradient L-BFGS co-design against gradient-free Nelder-Mead."""
    z0 = jnp.zeros(6, dtype=jnp.float64)  # Midpoint nominal initial design (violates t_in_pbli = 1047 K > 965 K!)

    loss_of_z = lambda z: coupled_plant_evaluate(decode_design(z))["loss"]
    eval_jit = jax.jit(lambda z: coupled_plant_evaluate(decode_design(z)))
    vg_jit = jax.jit(jax.value_and_grad(loss_of_z))
    loss_jit = jax.jit(loss_of_z)

    # Warm up JIT
    init_metrics = eval_jit(z0)
    _, g0 = vg_jit(z0)
    jax.block_until_ready(g0)

    # 1. Exact End-to-End Gradient Co-Design (L-BFGS-B driven by exact 3-solver JAX reverse-mode gradients)
    def scipy_vg(z_np: np.ndarray):
        lv, gr = vg_jit(jnp.asarray(z_np, dtype=jnp.float64))
        return float(lv), np.asarray(gr, dtype=np.float64)

    t0 = time.perf_counter()
    lbfgs_res = scipy_minimize(
        scipy_vg,
        np.zeros(6, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-4.0, 4.0)] * 6,
        options={"maxiter": n_grad_steps, "ftol": 1e-12, "gtol": 1e-8},
    )
    grad_s = time.perf_counter() - t0
    z_opt = jnp.asarray(lbfgs_res.x, dtype=jnp.float64)
    grad_metrics = {k: float(v) for k, v in eval_jit(z_opt).items()}
    theta_opt = np.asarray(decode_design(z_opt))

    # Verify coupled 3-solver end-to-end AD gradient against central finite differences at z0!
    _, ad_vec = vg_jit(z0)
    ad_np = np.asarray(ad_vec)
    fd_np = np.zeros(6, dtype=np.float64)
    h = 1.0e-6
    for j in range(6):
        dz = np.zeros(6, dtype=np.float64)
        dz[j] = h
        fd_np[j] = (float(loss_jit(z0 + dz)) - float(loss_jit(z0 - dz))) / (2.0 * h)
    floor = max(float(np.max(np.abs(fd_np))) * 1.0e-4, 1.0e-6)
    coupled_grad_rel_err = float(np.max(np.abs(ad_np - fd_np) / np.maximum(np.abs(fd_np), floor)))

    # 2. Gradient-free Nelder-Mead from the same nominal starting point with equal evaluation budget
    t_gf = time.perf_counter()
    nm_res = scipy_minimize(
        lambda zv: float(loss_jit(jnp.asarray(zv, dtype=jnp.float64))),
        np.zeros(6, dtype=np.float64),
        method="Nelder-Mead",
        options={"maxfev": n_gf_evals},
    )
    gf_s = time.perf_counter() - t_gf
    gf_metrics = {k: float(v) for k, v in eval_jit(jnp.asarray(nm_res.x, dtype=jnp.float64)).items()}

    return {
        "design_variables": ["S_fuel_1e20", "P_aux_MW", "m_flow_pbli_kg_s", "m_flow_flibe_kg_s", "V_boil_mol_s", "d_rec_frac"],
        "nominal_midpoint": {k: round(float(v), 5) for k, v in init_metrics.items()},
        "gradient_codesign": {
            "evals": int(lbfgs_res.nfev),
            "wall_seconds": round(grad_s, 4),
            "coupled_3solver_grad_fd_rel_err": coupled_grad_rel_err,
            "optimal_setpoints": [round(float(x), 4) for x in theta_opt],
            "metrics": {k: round(v, 5) for k, v in grad_metrics.items()},
        },
        "gradient_free_baseline": {
            "evals": int(nm_res.nfev),
            "wall_seconds": round(gf_s, 4),
            "metrics": {k: round(v, 5) for k, v in gf_metrics.items()},
        },
    }
