"""Read-only reference formulas for the ARC 505.6 MWf Whole-Plant Optimization Challenge.

Provides the exact parameter bounds, INL baseline values, public scenario multipliers,
reserve floor formula, and thermal-electric/capital economic equations used by the grader.
"""

from __future__ import annotations

import math

P_FUS_PULSE_MW = 505.6
N_BURN_KG_S = 8.99e-7
F_BURN_NOM = 0.10
PULSE_TIME_S = 1800.0
SIM_TIME_3Y_S = 94_672_800.0
PULSED_WINDOW_S = 2_160_000.0  # 25 days (1,200 pulses) explicit 1800s pulsing in TMAP8
RESERVE_FAIL_FRAC = 0.25
RESERVE_TIME_S = 86_400.0

MAX_STARTUP_TRITIUM_KG = 1.00
MAX_INPROCESS_NOM_KG = 0.220
MAX_HX_TRITIUM_NOM_KG = 1.0e-4
MAX_T_DOUBLE_NOM_YR = 1.50
MAX_T_DOUBLE_DEG_YR = 2.50
MIN_NET_GAIN_3Y_KG = 2.00

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "initial_inventory": (0.25, 1.50),
    "resident_time_BB": (900.0, 14400.0),
    "resident_time_TES": (3600.0, 172800.0),
    "eta_2": (0.35, 0.96),
    "resident_time_HX": (300.0, 3600.0),
    "resident_time_DS": (600.0, 14400.0),
    "resident_time_vacuum": (120.0, 1800.0),
    "f_DIR": (0.10, 0.88),
    "resident_time_FCU": (180.0, 2400.0),
    "resident_time_ISS": (1800.0, 43200.0),
    "f_ISS_to_DS": (0.02, 0.25),
    "resident_time_membrane": (30.0, 900.0),
    "TBR_value": (1.02, 1.18),
    "AF_value": (0.65, 0.88),
    "eta_f": (0.18, 0.42),
}

INL_BASELINE_DESIGN: dict[str, float] = {
    "initial_inventory": 1.14,
    "resident_time_BB": 4500.0,
    "resident_time_TES": 86400.0,
    "eta_2": 0.70,
    "resident_time_HX": 1000.0,
    "resident_time_DS": 3600.0,
    "resident_time_vacuum": 600.0,
    "f_DIR": 0.50,
    "resident_time_FCU": 585.0,
    "resident_time_ISS": 22815.0,
    "f_ISS_to_DS": 0.10,
    "resident_time_membrane": 100.0,
    "TBR_value": 1.067,
    "AF_value": 0.75,
    "eta_f": 0.25,
}

PUBLIC_SCENARIOS = {
    "nominal": {"tbe_mult": 1.0, "eta2_mult": 1.0, "f_hx_ds": 1.0e-4, "tbr_mult": 1.0, "af_mult": 1.0, "reserve_ratio": 1.0, "max_t_double_yr": 1.50},
    "tbe_drop": {"tbe_mult": 0.80, "eta2_mult": 1.0, "f_hx_ds": 1.0e-4, "tbr_mult": 1.0, "af_mult": 1.0, "reserve_ratio": 0.85, "max_t_double_yr": None},
    "tes_fouling": {"tbe_mult": 1.0, "eta2_mult": 0.82, "f_hx_ds": 3.0e-4, "tbr_mult": 1.0, "af_mult": 1.0, "reserve_ratio": 0.85, "max_t_double_yr": None},
    "blanket_degraded": {"tbe_mult": 1.0, "eta2_mult": 1.0, "f_hx_ds": 1.0e-4, "tbr_mult": 0.965, "af_mult": 0.85, "reserve_ratio": 0.85, "max_t_double_yr": 2.50},
}


def compute_effective_tbe(eta_f: float, f_dir: float) -> float:
    impurity_factor = 1.0 - 0.35 * max(0.0, f_dir - 0.45) ** 1.6
    return eta_f * F_BURN_NOM * impurity_factor


def compute_reserve_floor_kg(tbe_eff: float, af_eff: float) -> float:
    return (N_BURN_KG_S / tbe_eff) * RESERVE_FAIL_FRAC * RESERVE_TIME_S * af_eff


def compute_plant_economics(design: dict[str, float]) -> dict[str, float]:
    i0 = design["initial_inventory"]
    tau1 = design["resident_time_BB"]
    tau2 = design["resident_time_TES"]
    eta2 = design["eta_2"]
    tau5 = design["resident_time_HX"]
    tau6 = design["resident_time_DS"]
    tau7 = design["resident_time_vacuum"]
    f_dir = design["f_DIR"]
    tau8 = design["resident_time_FCU"]
    tau9 = design["resident_time_ISS"]
    f_9_6 = design["f_ISS_to_DS"]
    tau11 = design["resident_time_membrane"]
    tbr = design["TBR_value"]
    af = design["AF_value"]
    eta_f = design["eta_f"]

    m_neutron = 1.20 - 0.30 * (tbr - 1.06)
    p_th_pulse_mw = P_FUS_PULSE_MW * (0.20 + 0.80 * m_neutron)
    eta_th = 0.435 * (1.0 - 0.13 * (1000.0 / tau5) ** 0.50)
    p_gross_avg_mwe = af * p_th_pulse_mw * eta_th

    p_core_aux_mwe = af * (26.0 + 7.5 * ((eta_f / 0.25) ** 2.0)) + 2.4 / max(0.06, 1.0 - af)
    p_pumps_mwe = 4.2 * ((4500.0 / tau1) ** 1.75) + 1.5 * ((1000.0 / tau5) ** 1.25)

    ntu_tes = -math.log(1.0 - eta2)
    ntu_nom = -math.log(1.0 - 0.70)
    p_tes_mwe = 0.85 * ((86400.0 / tau2) ** 0.65) * ((ntu_tes / ntu_nom) ** 1.35)
    p_vac_dir_mwe = 1.10 * ((600.0 / tau7) ** 1.15) + 2.80 * (f_dir ** 2.4)
    p_iss_mwe = 1.65 * ((22815.0 / tau9) ** 0.70) * ((0.10 / f_9_6) ** 0.60)
    p_fcu_ds_tsm_mwe = (
        0.45 * ((585.0 / tau8) ** 0.85)
        + 0.40 * ((3600.0 / tau6) ** 0.75)
        + 0.35 * ((100.0 / tau11) ** 0.80)
    )
    p_parasitic_mwe = p_core_aux_mwe + p_pumps_mwe + p_tes_mwe + p_vac_dir_mwe + p_iss_mwe + p_fcu_ds_tsm_mwe
    p_net_elec_mwe = p_gross_avg_mwe - p_parasitic_mwe

    p_capital_equiv_mwe = (
        18.0 * i0
        + 40.0 * max(0.0, tbr - 1.05) ** 1.6
        + 2.2 * ((tau5 / 1000.0) ** 0.85)
    )
    return {
        "p_gross_avg_mwe": p_gross_avg_mwe,
        "p_parasitic_mwe": p_parasitic_mwe,
        "p_net_elec_mwe": p_net_elec_mwe,
        "p_capital_equiv_mwe": p_capital_equiv_mwe,
        "j_econ_mwe": p_net_elec_mwe - p_capital_equiv_mwe,
    }
