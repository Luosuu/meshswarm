"""Task setup, starting workspace files, and protection rules for DOE INL TMAP8 ARC optimization."""

from __future__ import annotations

import json

from plant_spec import DELIVERABLE, HERE, INL_BASELINE_DESIGN, INL_FUEL_CYCLE_PATH, verify_fuel_cycle_integrity

KIND = "ARC-class 11-subsystem fusion power plant co-design and robust fuel-cycle optimization"
STARTER_DIR = HERE / "starter_files"

PROTECT_PATTERNS = [
    DELIVERABLE,
    "fuel_cycle.i",
    "plant_economics_ref.py",
    "evaluate_local.py",
]


def starting_files() -> dict[str, str]:
    verify_fuel_cycle_integrity(INL_FUEL_CYCLE_PATH)
    return {
        "fuel_cycle.i": INL_FUEL_CYCLE_PATH.read_text(encoding="utf-8"),
        "PROBLEM.md": (STARTER_DIR / "PROBLEM.md").read_text(encoding="utf-8"),
        "plant_economics_ref.py": (STARTER_DIR / "plant_economics_ref.py").read_text(encoding="utf-8"),
        "evaluate_local.py": (STARTER_DIR / "evaluate_local.py").read_text(encoding="utf-8"),
        DELIVERABLE: json.dumps(INL_BASELINE_DESIGN, indent=2) + "\n",
    }


def task_text() -> str:
    return (
        "Read `PROBLEM.md`, `fuel_cycle.i`, `plant_economics_ref.py`, and `evaluate_local.py`.\n\n"
        "Optimize the 15-parameter DOE INL `TMAP8` ARC-class 505.6 MWf whole-plant design in "
        f"`{DELIVERABLE}` to maximize Levelized Net Economic Electric Output (`j_econ_mwe`) while "
        "satisfying every hard radiological, storage-reserve, startup-tritium, and doubling-time gate "
        "across all pulsed operating scenarios (including held-out multi-subsystem off-normal jitter).\n\n"
        "Both `tmap8-opt` (DOE INL's compiled MOOSE C++ solver) and GPU `python` (`jax` with CUDA, "
        "`diffrax`, `optax`, `scipy`) are available inside your sandbox. Build and validate a fast "
        "GPU digital twin / optimizer to explore the coupled 15D constrained landscape, calibrate any "
        "time-discretization differences against `tmap8-opt`, verify your candidate with "
        "`python evaluate_local.py`, and ensure sufficient safety margin for unseen combined jitter."
    )
