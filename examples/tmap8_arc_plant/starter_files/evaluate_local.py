"""Local verification helper: runs DOE INL `tmap8-opt` concurrently across the 4 public scenarios.

Usage inside workspace:
    python evaluate_local.py
    python evaluate_local.py my_candidate.json
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from plant_economics_ref import (
    MAX_HX_TRITIUM_NOM_KG,
    MAX_INPROCESS_NOM_KG,
    MAX_STARTUP_TRITIUM_KG,
    MIN_NET_GAIN_3Y_KG,
    PARAM_BOUNDS,
    PUBLIC_SCENARIOS,
    PULSED_WINDOW_S,
    SIM_TIME_3Y_S,
    compute_effective_tbe,
    compute_plant_economics,
    compute_reserve_floor_kg,
)

INPROC_COLS = (
    "T_01_BZ", "T_02_TES", "T_03_FW", "T_04_DIV", "T_05_HX",
    "T_06_DS", "T_07_vacuum", "T_08_FCU", "T_09_ISS", "T_11_membrane",
)


def _eval_one(sc_name: str, sc: dict, design: dict[str, float], work: Path) -> tuple[str, dict | None, list[str]]:
    sc_dir = work / sc_name
    sc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile("fuel_cycle.i", sc_dir / "fuel_cycle.i")

    tbe_eff = compute_effective_tbe(design["eta_f"], design["f_DIR"]) * sc["tbe_mult"]
    eta2_eff = min(0.995, design["eta_2"] * sc["eta2_mult"])
    tbr_eff = design["TBR_value"] * sc["tbr_mult"]
    af_eff = min(0.95, design["AF_value"] * sc["af_mult"])
    out_base = f"out_{sc_name}"
    cmd = [
        "tmap8-opt", "-i", "fuel_cycle.i",
        f"initial_inventory=${{units {design['initial_inventory']:.10g} kg}}",
        f"initial_refinement_time=${{units {PULSED_WINDOW_S:.10g} s}}",
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
        f"f_HX_to_DS={sc['f_hx_ds']:.10g}",
        f"TBR_value={tbr_eff:.10g}",
        f"AF_value={af_eff:.10g}",
        f"TBE_value={tbe_eff:.10g}",
        f"Outputs/file_base={out_base}",
    ]
    try:
        proc = subprocess.run(cmd, cwd=sc_dir, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return sc_name, None, [f"{sc_name}: tmap8-opt timed out after 240s"]
    csv_path = sc_dir / f"{out_base}.csv"
    if proc.returncode != 0 or not csv_path.is_file():
        return sc_name, None, [f"{sc_name}: tmap8-opt failed (rc={proc.returncode})"]
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or float(rows[-1]["time"]) < 0.99 * SIM_TIME_3Y_S:
        return sc_name, None, [f"{sc_name}: storage depleted early"]

    i0 = design["initial_inventory"]
    min_sto = min(float(r["T_10_storage"]) for r in rows)
    max_inproc = max(sum(float(r[c]) for c in INPROC_COLS) for r in rows)
    max_hx = max(float(r["T_05_HX"]) for r in rows)
    final_sto = float(rows[-1]["T_10_storage"])
    t_double = None
    prev_t, prev_s = 0.0, i0
    for idx, r in enumerate(rows):
        t_s, s_v = float(r["time"]), float(r["T_10_storage"])
        if idx > 0 and t_double is None and s_v >= 2.0 * i0:
            frac = (2.0 * i0 - prev_s) / max(1e-15, s_v - prev_s)
            t_double = (prev_t + max(0.0, min(1.0, frac)) * (t_s - prev_t)) / (86400.0 * 365.25)
        prev_t, prev_s = t_s, s_v

    fails: list[str] = []
    req_res = sc["reserve_ratio"] * compute_reserve_floor_kg(tbe_eff, af_eff)
    if min_sto < req_res:
        fails.append(f"{sc_name}: min_sto={min_sto:.4f} < {req_res:.4f} kg")
    if max_inproc > MAX_INPROCESS_NOM_KG:
        fails.append(f"{sc_name}: max_inproc={max_inproc:.4f} > {MAX_INPROCESS_NOM_KG:.4f} kg")
    if max_hx > MAX_HX_TRITIUM_NOM_KG:
        fails.append(f"{sc_name}: max_hx={max_hx * 1e6:.1f} > {MAX_HX_TRITIUM_NOM_KG * 1e6:.1f} mg")
    if sc["max_t_double_yr"] is not None and (t_double is None or t_double > sc["max_t_double_yr"]):
        fails.append(f"{sc_name}: t_double={t_double} > {sc['max_t_double_yr']} yr")
    if sc_name == "nominal" and (final_sto - i0) < MIN_NET_GAIN_3Y_KG:
        fails.append(f"nominal: net_gain_3y={final_sto - i0:.3f} < {MIN_NET_GAIN_3Y_KG} kg")

    rep = {
        "min_sto_kg": round(min_sto, 5),
        "req_reserve_kg": round(req_res, 5),
        "max_inproc_kg": round(max_inproc, 5),
        "max_hx_mg": round(max_hx * 1e6, 2),
        "t_double_yr": None if t_double is None else round(t_double, 4),
        "final_sto_kg": round(final_sto, 4),
    }
    return sc_name, rep, fails


def main() -> int:
    design_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("plant_design.json")
    design = json.loads(design_file.read_text(encoding="utf-8"))

    for k, (lo, hi) in PARAM_BOUNDS.items():
        if k not in design:
            print(f"ERROR: missing key {k}")
            return 1
        v = float(design[k])
        if v < lo - 1e-12 or v > hi + 1e-12:
            print(f"ERROR: {k}={v} outside bounds [{lo}, {hi}]")
            return 1

    econ = compute_plant_economics(design)
    failures: list[str] = []
    if design["initial_inventory"] > MAX_STARTUP_TRITIUM_KG:
        failures.append(f"initial_inventory={design['initial_inventory']:.4f} > {MAX_STARTUP_TRITIUM_KG:.2f} kg")
    if econ["p_net_elec_mwe"] <= 0.0:
        failures.append(f"p_net_elec_mwe={econ['p_net_elec_mwe']:.2f} <= 0 MWe")

    t0 = time.perf_counter()
    work = Path(tempfile.mkdtemp(prefix="tmap8-local-"))
    scenario_reports = {}
    try:
        with ThreadPoolExecutor(max_workers=len(PUBLIC_SCENARIOS)) as pool:
            futs = [pool.submit(_eval_one, sc_name, sc, design, work) for sc_name, sc in PUBLIC_SCENARIOS.items()]
            for fut in futs:
                sc_name, rep, sc_fails = fut.result()
                if rep is not None:
                    scenario_reports[sc_name] = rep
                failures.extend(sc_fails)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    elapsed = time.perf_counter() - t0
    public_score = 0.0 if failures else round(10.0 * max(0.0, min(1.0, (econ["j_econ_mwe"] - 72.0) / 30.0)), 3)
    print(json.dumps({
        "public_passed": len(failures) == 0,
        "public_score_est": public_score,
        "j_econ_mwe": round(econ["j_econ_mwe"], 4),
        "p_net_elec_mwe": round(econ["p_net_elec_mwe"], 4),
        "p_gross_avg_mwe": round(econ["p_gross_avg_mwe"], 4),
        "p_parasitic_mwe": round(econ["p_parasitic_mwe"], 4),
        "p_capital_equiv_mwe": round(econ["p_capital_equiv_mwe"], 4),
        "tmap8_wall_s": round(elapsed, 2),
        "scenarios": scenario_reports,
        "failures": failures,
    }, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
