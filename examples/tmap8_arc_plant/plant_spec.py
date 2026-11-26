"""Specification, physics couplings, and CSV trajectory checks for the DOE INL TMAP8 ARC plant benchmark.

Grounded in Idaho National Laboratory's official `idaholab/tmap8` whole-plant tritium fuel cycle
benchmark (`test/tests/fuel_cycle_Meschini/fuel_cycle.i`, Meschini et al., Nuclear Fusion 63, 2023)
and MIT/CFS ARC power-balance scaling (Sorbom et al., Fusion Eng. Des. 100, 2015).
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
INL_FUEL_CYCLE_PATH = HERE / "inl_baseline" / "fuel_cycle.i"
FUEL_CYCLE_SHA256 = "b30426042d7c102397510c074f1c5454e18bd89d947c67ea0a3b1aaa31b2b0e6"
DELIVERABLE = "plant_design.json"

# Fixed ARC core & physical constants from INL fuel_cycle.i
P_FUS_PULSE_MW = 505.6          # 8.99e-7 kg/s DT tritium burn -> 505.6 MW fusion power during pulse
N_BURN_KG_S = 8.99e-7           # Plasma tritium burn rate during active pulse [kg/s]
F_BURN_NOM = 0.10               # Core plasma burn fraction [-]
PULSE_TIME_S = 1800.0           # Half-hour ARC pulse cycle [s]
SIM_TIME_3Y_S = 94_672_800.0    # 3.0 Julian years (3600 * 24 * 365.25 * 3) [s]
PULSED_WINDOW_S = 2_160_000.0   # 25 full days (1,200 pulses = 2,400 MOOSE steps) explicit pulsing [s]
RESERVE_FAIL_FRAC = 0.25        # Fraction of fuel cycle subject to outage q [-]
RESERVE_TIME_S = 86_400.0       # Required 24-hour reserve autonomy t_res [s]

# Hard DOE safety & self-sufficiency limits
MAX_STARTUP_TRITIUM_KG = 1.00   # Must strictly beat INL hand-tuned 1.14 kg startup inventory
MAX_INPROCESS_NOM_KG = 0.220    # DOE active non-storage vulnerable tritium limit (220 g)
MAX_HX_TRITIUM_NOM_KG = 1.0e-4  # Heat-exchanger steam-loop permeation barrier limit (100 mg)
MAX_T_DOUBLE_NOM_YR = 1.50      # Nominal tritium doubling horizon [years]
MAX_T_DOUBLE_DEG_YR = 2.50      # Degraded-blanket tritium doubling horizon [years]
MIN_NET_GAIN_3Y_KG = 2.00       # Minimum 3-year net bred storage gain T_10(3y) - I_0 [kg]

# Score normalization window [MWe] -> [0, 10]
ECON_FLOOR_MWE = 72.0
ECON_CEIL_MWE = 102.0

# 15 coupled whole-plant design & operating parameters
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "initial_inventory": (0.25, 1.50),         # Startup tritium in T_10_storage [kg]
    "resident_time_BB": (900.0, 14400.0),      # 1. Breeding Blanket FLiBe loop residence time [s]
    "resident_time_TES": (3600.0, 172800.0),   # 2. Tritium Extraction System residence time [s]
    "eta_2": (0.35, 0.96),                     # 2. Single-pass TES extraction efficiency [-]
    "resident_time_HX": (300.0, 3600.0),       # 5. Primary Heat Exchanger residence time [s]
    "resident_time_DS": (600.0, 14400.0),      # 6. Detritiation System residence time [s]
    "resident_time_vacuum": (120.0, 1800.0),   # 7. Divertor cryopump regeneration cycle time [s]
    "f_DIR": (0.10, 0.88),                     # 7->10. Direct Internal Recycling fraction [-]
    "resident_time_FCU": (180.0, 2400.0),      # 8. Fuel Clean-Up residence time [s]
    "resident_time_ISS": (1800.0, 43200.0),    # 9. Cryogenic Isotope Separation System residence time [s]
    "f_ISS_to_DS": (0.02, 0.25),               # 9->6. ISS overhead recycle slip fraction to DS [-]
    "resident_time_membrane": (30.0, 900.0),   # 11. Tritium Permeation Membrane residence time [s]
    "TBR_value": (1.02, 1.18),                 # Blanket Tritium Breeding Ratio [-]
    "AF_value": (0.65, 0.88),                  # Pulsed duty / plant availability factor [-]
    "eta_f": (0.18, 0.42),                     # Cryo-pellet fueling efficiency [-]
}

# Exact default values from INL's `fuel_cycle_Meschini/fuel_cycle.i`
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

# Hand-tuned conservative feasible reference (used by --selftest)
CONSERVATIVE_SAFE_DESIGN: dict[str, float] = {
    "initial_inventory": 0.86,
    "resident_time_BB": 7200.0,
    "resident_time_TES": 54000.0,
    "eta_2": 0.945,
    "resident_time_HX": 380.0,
    "resident_time_DS": 6500.0,
    "resident_time_vacuum": 620.0,
    "f_DIR": 0.62,
    "resident_time_FCU": 560.0,
    "resident_time_ISS": 8200.0,
    "f_ISS_to_DS": 0.11,
    "resident_time_membrane": 180.0,
    "TBR_value": 1.074,
    "AF_value": 0.79,
    "eta_f": 0.29,
}

# Naive corner-pushing design that blows up parasitic power (used by --selftest)
UNSAFE_CORNER_DESIGN: dict[str, float] = {
    "initial_inventory": 0.30,
    "resident_time_BB": 900.0,
    "resident_time_TES": 3600.0,
    "eta_2": 0.96,
    "resident_time_HX": 300.0,
    "resident_time_DS": 600.0,
    "resident_time_vacuum": 120.0,
    "f_DIR": 0.88,
    "resident_time_FCU": 180.0,
    "resident_time_ISS": 1800.0,
    "f_ISS_to_DS": 0.02,
    "resident_time_membrane": 30.0,
    "TBR_value": 1.18,
    "AF_value": 0.88,
    "eta_f": 0.42,
}


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    tbe_mult: float = 1.0
    eta2_mult: float = 1.0
    f_hx_ds: float = 1.0e-4
    tbr_mult: float = 1.0
    af_mult: float = 1.0
    reserve_ratio: float = 1.0
    max_inproc_kg: float = MAX_INPROCESS_NOM_KG
    max_hx_kg: float = MAX_HX_TRITIUM_NOM_KG
    max_t_double_yr: float | None = None
    public: bool = True


PUBLIC_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        name="nominal",
        description="Nominal pulsed 3-year ARC operation",
        reserve_ratio=1.00,
        max_inproc_kg=MAX_INPROCESS_NOM_KG,
        max_hx_kg=MAX_HX_TRITIUM_NOM_KG,
        max_t_double_yr=MAX_T_DOUBLE_NOM_YR,
        public=True,
    ),
    ScenarioSpec(
        name="tbe_drop",
        description="20% plasma burn efficiency degradation (impurity/pellet excursion)",
        tbe_mult=0.80,
        reserve_ratio=0.85,
        max_inproc_kg=MAX_INPROCESS_NOM_KG,
        max_hx_kg=MAX_HX_TRITIUM_NOM_KG,
        public=True,
    ),
    ScenarioSpec(
        name="tes_fouling",
        description="18% TES extraction fouling + 3x HX->DS tritium permeation surge",
        eta2_mult=0.82,
        f_hx_ds=3.0e-4,
        reserve_ratio=0.85,
        max_inproc_kg=MAX_INPROCESS_NOM_KG,
        max_hx_kg=MAX_HX_TRITIUM_NOM_KG,
        public=True,
    ),
    ScenarioSpec(
        name="blanket_degraded",
        description="3.5% blanket breeding degradation + 15% availability outage",
        tbr_mult=0.965,
        af_mult=0.85,
        reserve_ratio=0.85,
        max_inproc_kg=MAX_INPROCESS_NOM_KG,
        max_hx_kg=MAX_HX_TRITIUM_NOM_KG,
        max_t_double_yr=MAX_T_DOUBLE_DEG_YR,
        public=True,
    ),
)

HELDOUT_SCENARIO = ScenarioSpec(
    name="heldout_combined_jitter",
    description="Held-out combined multi-subsystem off-normal jitter check",
    tbe_mult=0.88,
    eta2_mult=0.90,
    f_hx_ds=2.0e-4,
    tbr_mult=0.985,
    af_mult=0.92,
    reserve_ratio=0.82,
    max_inproc_kg=0.230,
    max_hx_kg=1.10e-4,
    max_t_double_yr=2.20,
    public=False,
)

ALL_SCENARIOS: tuple[ScenarioSpec, ...] = (*PUBLIC_SCENARIOS, HELDOUT_SCENARIO)
SCENARIO_MAP: dict[str, ScenarioSpec] = {s.name: s for s in ALL_SCENARIOS}


def verify_fuel_cycle_integrity(path: Path = INL_FUEL_CYCLE_PATH) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FUEL_CYCLE_SHA256:
        raise RuntimeError(
            f"INL fuel_cycle.i SHA256 mismatch at {path}: got {digest}, expected {FUEL_CYCLE_SHA256}"
        )


def validate_design_bounds(raw: Any) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(raw, dict):
        return None, "plant_design.json must contain a JSON object mapping parameter names to numbers"
    missing = [k for k in PARAM_BOUNDS if k not in raw]
    if missing:
        return None, f"missing required design parameters: {missing}"
    extra = [k for k in raw if k not in PARAM_BOUNDS]
    if extra:
        return None, f"unknown extra keys in plant_design.json: {extra}"

    clean: dict[str, float] = {}
    for key, (lo, hi) in PARAM_BOUNDS.items():
        val = raw[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return None, f"parameter '{key}' must be a finite float, got {type(val).__name__}"
        fval = float(val)
        if not math.isfinite(fval):
            return None, f"parameter '{key}' is non-finite ({val})"
        if fval < lo - 1e-12 or fval > hi + 1e-12:
            return None, f"parameter '{key}'={fval:.6g} outside physical bounds [{lo}, {hi}]"
        clean[key] = min(hi, max(lo, fval))
    return clean, None


def compute_effective_tbe(eta_f: float, f_dir: float) -> float:
    """Coupled core Tritium Burn Efficiency accounting for DIR exhaust ash/impurity slip."""
    impurity_factor = 1.0 - 0.35 * max(0.0, f_dir - 0.45) ** 1.6
    return eta_f * F_BURN_NOM * impurity_factor


def compute_reserve_floor_kg(tbe_eff: float, af_eff: float) -> float:
    """INL Meschini 24-hour outage reserve inventory requirement [kg]."""
    return (N_BURN_KG_S / tbe_eff) * RESERVE_FAIL_FRAC * RESERVE_TIME_S * af_eff


def compute_plant_economics(design: dict[str, float]) -> dict[str, float]:
    """ARC 505.6 MWf whole-plant thermal-electric & annualized capital balance [MWe]."""
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

    # 1. Blanket neutron multiplication & Rankine/Brayton power conversion
    m_neutron = 1.20 - 0.30 * (tbr - 1.06)
    p_th_pulse_mw = P_FUS_PULSE_MW * (0.20 + 0.80 * m_neutron)
    eta_th = 0.435 * (1.0 - 0.13 * (1000.0 / tau5) ** 0.50)
    p_gross_avg_mwe = af * p_th_pulse_mw * eta_th

    # 2. Core plasma heating, cryo-magnets, dwell PF reset, and high-speed pellet fueling
    p_core_aux_mwe = af * (26.0 + 7.5 * ((eta_f / 0.25) ** 2.0)) + 2.4 / max(0.06, 1.0 - af)

    # 3. Primary FLiBe MHD + secondary heat-exchanger pumping
    p_pumps_mwe = 4.2 * ((4500.0 / tau1) ** 1.75) + 1.5 * ((1000.0 / tau5) ** 1.25)

    # 4. 11-subsystem tritium plant parasitic electric loads
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
    p_tritium_plant_mwe = p_tes_mwe + p_vac_dir_mwe + p_iss_mwe + p_fcu_ds_tsm_mwe
    p_parasitic_mwe = p_core_aux_mwe + p_pumps_mwe + p_tritium_plant_mwe
    p_net_elec_mwe = p_gross_avg_mwe - p_parasitic_mwe

    # 5. Annualized startup-tritium scarcity ($30M/kg + CANDU supply shadow cost), Li-6 & vessel capital
    p_capital_equiv_mwe = (
        18.0 * i0
        + 40.0 * max(0.0, tbr - 1.05) ** 1.6
        + 2.2 * ((tau5 / 1000.0) ** 0.85)
    )
    j_econ_mwe = p_net_elec_mwe - p_capital_equiv_mwe

    return {
        "p_th_pulse_mw": round(p_th_pulse_mw, 4),
        "eta_th": round(eta_th, 5),
        "p_gross_avg_mwe": round(p_gross_avg_mwe, 4),
        "p_core_aux_mwe": round(p_core_aux_mwe, 4),
        "p_pumps_mwe": round(p_pumps_mwe, 4),
        "p_tritium_plant_mwe": round(p_tritium_plant_mwe, 4),
        "p_parasitic_mwe": round(p_parasitic_mwe, 4),
        "p_net_elec_mwe": round(p_net_elec_mwe, 4),
        "p_capital_equiv_mwe": round(p_capital_equiv_mwe, 4),
        "j_econ_mwe": round(j_econ_mwe, 4),
    }


def build_tmap8_cli_args(
    design: dict[str, float],
    scenario: ScenarioSpec,
    out_base: str,
    pulsed_window_s: float = PULSED_WINDOW_S,
) -> tuple[list[str], dict[str, float]]:
    """Build exact CLI overrides for untouched `tmap8-opt -i fuel_cycle.i`."""
    tbe_nom = compute_effective_tbe(design["eta_f"], design["f_DIR"])
    tbe_eff = tbe_nom * scenario.tbe_mult
    eta2_eff = min(0.995, design["eta_2"] * scenario.eta2_mult)
    tbr_eff = design["TBR_value"] * scenario.tbr_mult
    af_eff = min(0.95, design["AF_value"] * scenario.af_mult)

    args = [
        f"initial_inventory=${{units {design['initial_inventory']:.10g} kg}}",
        f"initial_refinement_time=${{units {pulsed_window_s:.10g} s}}",
        f"simulation_time=${{units {SIM_TIME_3Y_S:.10g} s}}",
        f"resident_time_BB=${{units {design['resident_time_BB']:.10g} s}}",
        f"resident_time_TES=${{units {design['resident_time_TES']:.10g} s}}",
        f"resident_time_HX=${{units {design['resident_time_HX']:.10g} s}}",
        f"resident_time_DS=${{units {design['resident_time_DS']:.10g} s}}",
        f"resident_time_vacuum=${{units {design['resident_time_vacuum']:.10g} s}}",
        f"resident_time_FCU=${{units {design['resident_time_FCU']:.10g} s}}",
        f"resident_time_ISS=${{units {design['resident_time_ISS']:.10g} s}}",
        f"resident_time_membrane=${{units {design['resident_time_membrane']:.10g} s}}",
        f"eta_2={eta2_eff:.10g}",
        f"f_DIR={design['f_DIR']:.10g}",
        f"f_ISS_to_DS={design['f_ISS_to_DS']:.10g}",
        f"f_HX_to_DS={scenario.f_hx_ds:.10g}",
        f"TBR_value={tbr_eff:.10g}",
        f"AF_value={af_eff:.10g}",
        f"TBE_value={tbe_eff:.10g}",
        f"Outputs/file_base={out_base}",
    ]
    eff = {
        "tbe_eff": tbe_eff,
        "eta2_eff": eta2_eff,
        "tbr_eff": tbr_eff,
        "af_eff": af_eff,
        "i_reserve_kg": compute_reserve_floor_kg(tbe_eff, af_eff),
    }
    return args, eff


INPROC_COLS = (
    "T_01_BZ",
    "T_02_TES",
    "T_03_FW",
    "T_04_DIV",
    "T_05_HX",
    "T_06_DS",
    "T_07_vacuum",
    "T_08_FCU",
    "T_09_ISS",
    "T_11_membrane",
)


def evaluate_csv_trajectory(
    csv_path: Path,
    design: dict[str, float],
    scenario: ScenarioSpec,
    eff: dict[str, float],
) -> dict[str, Any]:
    """Parse `tmap8-opt` CSV output and check all scenario safety & self-sufficiency gates."""
    if not csv_path.is_file():
        return {"passed": False, "failures": [f"{scenario.name}: missing CSV output {csv_path.name}"]}

    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) < 10:
        return {"passed": False, "failures": [f"{scenario.name}: truncated CSV ({len(rows)} rows)"]}

    t_end = float(rows[-1]["time"])
    if t_end < 0.99 * SIM_TIME_3Y_S:
        return {
            "passed": False,
            "failures": [
                f"{scenario.name}: TMAP8 terminated early at t={t_end / 86400.0:.2f}d "
                "(storage tritium depleted)"
            ],
        }

    i0 = design["initial_inventory"]
    min_sto = float("inf")
    min_sto_day = 0.0
    max_inproc = 0.0
    max_hx = 0.0
    t_double_yr: float | None = None
    sec_per_yr = 86400.0 * 365.25

    prev_t = 0.0
    prev_sto = i0
    for idx, r in enumerate(rows):
        t_s = float(r["time"])
        sto = float(r["T_10_storage"])
        hx = float(r["T_05_HX"])
        inproc = sum(float(r[c]) for c in INPROC_COLS)
        if sto < min_sto:
            min_sto = sto
            min_sto_day = t_s / 86400.0
        if inproc > max_inproc:
            max_inproc = inproc
        if hx > max_hx:
            max_hx = hx
        if idx > 0 and t_double_yr is None and sto >= 2.0 * i0:
            if sto > prev_sto:
                frac = (2.0 * i0 - prev_sto) / (sto - prev_sto)
                t_cross = prev_t + max(0.0, min(1.0, frac)) * (t_s - prev_t)
            else:
                t_cross = t_s
            t_double_yr = t_cross / sec_per_yr
        prev_t, prev_sto = t_s, sto

    final_sto = float(rows[-1]["T_10_storage"])
    net_gain_3y = final_sto - i0
    req_reserve = scenario.reserve_ratio * eff["i_reserve_kg"]

    failures: list[str] = []
    if min_sto < req_reserve:
        failures.append(
            f"{scenario.name}: min T_10_storage={min_sto:.4f} kg < required reserve {req_reserve:.4f} kg"
        )
    if max_inproc > scenario.max_inproc_kg:
        failures.append(
            f"{scenario.name}: peak in-process tritium={max_inproc:.4f} kg > cap {scenario.max_inproc_kg:.4f} kg"
        )
    if max_hx > scenario.max_hx_kg:
        failures.append(
            f"{scenario.name}: peak HX tritium={max_hx * 1e6:.1f} mg > cap {scenario.max_hx_kg * 1e6:.1f} mg"
        )
    if scenario.max_t_double_yr is not None:
        if t_double_yr is None or t_double_yr > scenario.max_t_double_yr:
            td_str = "never" if t_double_yr is None else f"{t_double_yr:.3f} yr"
            failures.append(
                f"{scenario.name}: doubling time={td_str} > limit {scenario.max_t_double_yr:.2f} yr"
            )
    if scenario.name == "nominal" and net_gain_3y < MIN_NET_GAIN_3Y_KG:
        failures.append(
            f"nominal: 3-year net bred gain={net_gain_3y:.3f} kg < {MIN_NET_GAIN_3Y_KG:.2f} kg"
        )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "n_steps": len(rows),
        "min_sto_kg": round(min_sto, 6),
        "min_sto_day": round(min_sto_day, 4),
        "req_reserve_kg": round(req_reserve, 6),
        "max_inproc_kg": round(max_inproc, 6),
        "max_hx_kg": round(max_hx, 8),
        "final_sto_kg": round(final_sto, 6),
        "net_gain_3y_kg": round(net_gain_3y, 6),
        "t_double_yr": None if t_double_yr is None else round(t_double_yr, 5),
    }


def score_of_econ(j_econ_mwe: float) -> float:
    raw = (j_econ_mwe - ECON_FLOOR_MWE) / (ECON_CEIL_MWE - ECON_FLOOR_MWE)
    return round(10.0 * max(0.0, min(1.0, raw)), 3)
