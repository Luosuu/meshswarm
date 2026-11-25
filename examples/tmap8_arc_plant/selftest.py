"""End-to-end selftest for the DOE INL TMAP8 ARC whole-plant benchmark (`--selftest`).

Verifies four cases against untouched `tmap8-opt -i fuel_cycle.i` without calling any LLM API:
  1. INL Meschini baseline (`initial_inventory=1.14 kg`, etc.) -> 0.0 (fails safety/startup gates)
  2. Naive corner-pushing design -> 0.0 (fails net power balance / safety gates)
  3. Hand-tuned conservative feasible design -> passes all 5 scenarios with positive score
  4. Live GPU JAX 89,856-trajectory multi-start differentiable optimizer -> passes all 5 scenarios
     and beats the conservative baseline.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from grader import make_grader
from meshswarm.shell import ShellRunner
from plant_spec import (
    CONSERVATIVE_SAFE_DESIGN,
    DELIVERABLE,
    HERE,
    INL_BASELINE_DESIGN,
    PULSED_WINDOW_S,
    UNSAFE_CORNER_DESIGN,
)
from tasks import starting_files

REF_OPT_PY = HERE / "reference_jax_optimizer.py"


class _WS:
    def __init__(self, workspace: Path):
        self.workspace = workspace


async def run_selftest(
    *,
    runner: ShellRunner,
    pulsed_window_s: float = PULSED_WINDOW_S,
) -> None:
    scratch = os.environ.get("MESH_RUN_SCRATCH")
    scratch_path = Path(scratch) if scratch else None
    grade_fn = make_grader(pulsed_window_s=pulsed_window_s, scratch_root=scratch_path)

    ws = Path(tempfile.mkdtemp(prefix="tmap8-selftest-", dir=scratch_path))
    try:
        for rel, content in starting_files().items():
            (ws / rel).write_text(content, encoding="utf-8")

        # 1. INL Meschini baseline must be vetoed (0.0)
        (ws / DELIVERABLE).write_text(json.dumps(INL_BASELINE_DESIGN, indent=2), encoding="utf-8")
        g_base = await grade_fn(_WS(ws))
        print(f"[selftest 1/4] INL baseline      -> score={g_base.score} | {g_base.reasons}", flush=True)
        assert g_base.score == 0.0, f"Expected INL baseline to fail gates, got {g_base}"

        # 2. Naive corner design must be vetoed (0.0)
        (ws / DELIVERABLE).write_text(json.dumps(UNSAFE_CORNER_DESIGN, indent=2), encoding="utf-8")
        g_corner = await grade_fn(_WS(ws))
        print(f"[selftest 2/4] Naive corner      -> score={g_corner.score} | {g_corner.reasons}", flush=True)
        assert g_corner.score == 0.0, f"Expected corner design to fail, got {g_corner}"

        # 3. Conservative hand-tuned safe design must pass all 5 scenarios
        (ws / DELIVERABLE).write_text(json.dumps(CONSERVATIVE_SAFE_DESIGN, indent=2), encoding="utf-8")
        g_safe = await grade_fn(_WS(ws))
        print(f"[selftest 3/4] Conservative safe -> score={g_safe.score} | {g_safe.reasons}", flush=True)
        assert (g_safe.score or 0.0) > 0.5, f"Expected conservative safe design to pass, got {g_safe}"

        # 4. Run live GPU JAX differentiable optimizer inside the sandboxed container & grade with tmap8-opt
        shutil.copyfile(REF_OPT_PY, ws / "reference_jax_optimizer.py")
        r_opt = await runner.run(f"python reference_jax_optimizer.py {DELIVERABLE}", ws, 180)
        if r_opt.exit_code != 0:
            raise RuntimeError(f"reference_jax_optimizer.py failed (rc={r_opt.exit_code}):\n{r_opt.output}")
        opt_meta = json.loads(r_opt.output[r_opt.output.find("{"):])
        g_jax = await grade_fn(_WS(ws))
        print(
            f"[selftest 4/4] GPU JAX twin opt  -> score={g_jax.score} "
            f"(gpu_wall={opt_meta['gpu_wall_s']}s for {opt_meta['n_trajectories_evaluated']} trajectories on {opt_meta['jax_devices']}) "
            f"| {g_jax.reasons}",
            flush=True,
        )
        assert (g_jax.score or 0.0) > (g_safe.score or 0.0), (
            f"Expected GPU JAX optimizer to beat conservative baseline ({g_safe.score}), got {g_jax}"
        )
    finally:
        shutil.rmtree(ws, ignore_errors=True)
