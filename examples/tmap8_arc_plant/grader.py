"""Untouched DOE INL `tmap8-opt` whole-plant oracle grader for `examples/tmap8_arc_plant`."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from meshevolve import Grade  # noqa: E402
from plant_spec import (  # noqa: E402
    ALL_SCENARIOS,
    DELIVERABLE,
    INL_FUEL_CYCLE_PATH,
    MAX_STARTUP_TRITIUM_KG,
    PULSED_WINDOW_S,
    ScenarioSpec,
    build_tmap8_cli_args,
    compute_plant_economics,
    evaluate_csv_trajectory,
    score_of_econ,
    validate_design_bounds,
    verify_fuel_cycle_integrity,
)

_CONTAINER_LD = (
    "/opt/rh/gcc-toolset-13/root/usr/lib64:/opt/openmpi/lib:"
    "/opt/petsc/lib:/opt/libmesh/lib:/opt/wasp/lib:/.singularity.d/libs"
)


def _tmap8_cmd_prefix(workdir: Path) -> list[str]:
    """Return Apptainer prefix invoking `tmap8-opt` inside `TMAP8_SIF` with native gcc-13/MPI FParser JIT."""
    sif = os.environ.get("TMAP8_SIF")
    bin_dir = os.environ.get("TMAP8_BIN_DIR", "/opt/tmap8/bin")
    c_path = (
        f"{bin_dir}:/opt/openmpi/bin:/opt/rh/gcc-toolset-13/root/usr/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    if sif and Path(sif).is_file():
        return [
            "apptainer", "exec", "--containall", "--no-home",
            "--pwd", str(workdir),
            "--bind", f"{workdir}:{workdir}",
            "--env", f"PATH={c_path}",
            "--env", f"LD_LIBRARY_PATH={_CONTAINER_LD}",
            sif,
            "tmap8-opt",
        ]
    if shutil.which("tmap8-opt"):
        return ["tmap8-opt"]
    raise RuntimeError("DOE INL TMAP8 (`TMAP8_SIF` or `tmap8-opt`) is not configured in this environment")


def _run_one_scenario(
    sc: ScenarioSpec,
    design: dict[str, float],
    work_root: Path,
    pulsed_window_s: float,
    timeout_s: float,
) -> tuple[str, dict[str, Any], list[str]]:
    sc_dir = work_root / sc.name
    sc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(INL_FUEL_CYCLE_PATH, sc_dir / "fuel_cycle.i")
    out_base = f"grade_{sc.name}"
    cli_args, eff = build_tmap8_cli_args(design, sc, out_base=out_base, pulsed_window_s=pulsed_window_s)
    prefix = _tmap8_cmd_prefix(sc_dir)
    try:
        proc = subprocess.run(
            [*prefix, "-i", "fuel_cycle.i", *cli_args],
            cwd=sc_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return sc.name, {"passed": False, "timed_out": True}, [f"{sc.name}: tmap8-opt timed out after {timeout_s:.0f}s"]
    csv_path = sc_dir / f"{out_base}.csv"
    if proc.returncode != 0 or not csv_path.is_file():
        tail = (proc.stdout + "\n" + proc.stderr).strip()[-300:]
        return sc.name, {"passed": False, "rc": proc.returncode}, [f"{sc.name}: tmap8-opt rc={proc.returncode} ({tail})"]
    traj = evaluate_csv_trajectory(csv_path, design, sc, eff)
    return sc.name, traj, list(traj["failures"])


def evaluate_design_with_tmap8(
    raw_design: Any,
    *,
    pulsed_window_s: float = PULSED_WINDOW_S,
    scratch_root: Path | None = None,
    timeout_per_scenario_s: float = 240.0,
) -> tuple[Grade, dict[str, Any]]:
    """Evaluate a candidate 15-parameter plant design using untouched `tmap8-opt` across all 5 scenarios."""
    verify_fuel_cycle_integrity(INL_FUEL_CYCLE_PATH)

    design, bound_err = validate_design_bounds(raw_design)
    if design is None:
        return Grade(0.0, f"invalid_design ({bound_err})"), {"passed": False, "error": bound_err}

    econ = compute_plant_economics(design)
    failures: list[str] = []
    if design["initial_inventory"] > MAX_STARTUP_TRITIUM_KG:
        failures.append(
            f"startup_tritium: initial_inventory={design['initial_inventory']:.4f} kg > {MAX_STARTUP_TRITIUM_KG:.2f} kg"
        )
    if econ["p_net_elec_mwe"] <= 0.0:
        failures.append(f"power_balance: p_net_elec_mwe={econ['p_net_elec_mwe']:.2f} <= 0 MWe")

    t0 = time.perf_counter()
    work = Path(tempfile.mkdtemp(prefix="tmap8-grade-", dir=scratch_root))
    scenario_metrics: dict[str, Any] = {}
    try:
        with ThreadPoolExecutor(max_workers=len(ALL_SCENARIOS)) as pool:
            futs = [
                pool.submit(_run_one_scenario, sc, design, work, pulsed_window_s, timeout_per_scenario_s)
                for sc in ALL_SCENARIOS
            ]
            for fut in futs:
                sc_name, traj, sc_fails = fut.result()
                scenario_metrics[sc_name] = traj
                failures.extend(sc_fails)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    wall_s = round(time.perf_counter() - t0, 3)
    details = {
        "passed": len(failures) == 0,
        **econ,
        "initial_inventory_kg": round(design["initial_inventory"], 5),
        "tmap8_grade_wall_s": wall_s,
        "scenarios": scenario_metrics,
        "failures": failures,
    }
    if failures:
        summary = f"safety_or_self_sufficiency_veto ({failures[0]}; total_violations={len(failures)})"
        return Grade(0.0, summary), details

    score = score_of_econ(econ["j_econ_mwe"])
    nom = scenario_metrics["nominal"]
    summary = (
        f"passed all 5 TMAP8 scenarios | J_econ={econ['j_econ_mwe']:.2f} MWe "
        f"(P_net={econ['p_net_elec_mwe']:.2f} MWe, I0={design['initial_inventory']:.3f} kg, "
        f"inproc_max={nom['max_inproc_kg'] * 1e3:.1f} g, HX_max={nom['max_hx_kg'] * 1e6:.1f} mg, "
        f"t_double={nom['t_double_yr']:.2f} yr, tmap8={wall_s:.1f}s)"
    )
    return Grade(score, summary), details


def make_grader(
    *,
    pulsed_window_s: float = PULSED_WINDOW_S,
    scratch_root: Path | None = None,
):
    async def grade_fn(result: Any) -> Grade:
        ws = Path(result.workspace)
        target = ws / DELIVERABLE
        if not target.is_file():
            return Grade(0.0, f"missing {DELIVERABLE}")
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            return Grade(0.0, f"malformed JSON in {DELIVERABLE}: {exc}")
        grade, report = await asyncio.to_thread(
            evaluate_design_with_tmap8,
            raw,
            pulsed_window_s=pulsed_window_s,
            scratch_root=scratch_root,
        )
        # Save structured oracle report inside `.mesh/` for post-run inspection & visualization
        mesh_dir = ws / ".mesh"
        if mesh_dir.is_dir():
            (mesh_dir / "tmap8_grade.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return grade

    return grade_fn
