"""Multiplicative zero-multiplier grader for Modelica-to-JAX benchmark tasks.

Orchestrates OpenModelica (`omc`) reference generation and invokes `eval_worker.py`
in an isolated subprocess (or through a sandboxed GPU `ShellRunner`).
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from meshswarm import RunResult  # noqa: E402
from meshswarm.shell import ShellRunner  # noqa: E402
from meshevolve import Grade  # noqa: E402
from tasks import DELIVERABLE, HERE, STAGE_HOLDUPS_16, TASKS, THRESHOLDS_S, BenchmarkTask  # noqa: E402

EVAL_WORKER_PY = HERE / "eval_worker.py"

FORBIDDEN = re.compile(
    r"\bsubprocess\b|\bos\.system\b|\bos\.popen\b|\bpty\b|\bshutil\.which\b|"
    r"\bomc\b|\.mesh\b|\bstop_gradient\b|jax_enable_x64[^\n]*False",
)


def score_of(combined_seconds: float) -> float:
    return round(10.0 * sum(1 for r in THRESHOLDS_S if combined_seconds <= r) / len(THRESHOLDS_S), 2)


def _omc_prefix(workdir: Path) -> list[str]:
    sif = os.environ.get("OMC_SIF")
    repos = os.environ.get("MODELICA_REPOS")
    omlib = os.environ.get("OPENMODELICALIBRARY", "")
    if sif and Path(sif).is_file():
        cmd = [
            "apptainer", "exec", "--containall", "--no-home",
            "--pwd", str(workdir),
            "--bind", f"{workdir}:{workdir}",
        ]
        if repos and Path(repos).is_dir():
            cmd += ["--bind", f"{repos}:{repos}:ro"]
        if omlib:
            cmd += ["--env", f"OPENMODELICALIBRARY={omlib}"]
        cmd += [sif]
        return cmd
    if shutil.which("omc"):
        return []
    raise RuntimeError("OpenModelica (`OMC_SIF` or `omc`) is not configured in this environment")


def compute_omc_gold(task: BenchmarkTask, scratch_root: Path | None = None) -> list[list[float]]:
    """Compile `task.model_path` once with OpenModelica and evaluate all held-out `test_cases`."""
    work = Path(tempfile.mkdtemp(prefix=f"omc-gold-{task.name}-", dir=scratch_root))
    try:
        shutil.copyfile(task.model_path, work / task.model_file)
        prefix = _omc_prefix(work)
        omlib = os.environ.get("OPENMODELICALIBRARY", "")
        tf_lib = os.environ.get("TRANSFORM_LIB", "")

        setup_lines = []
        if omlib:
            setup_lines.append(f'setModelicaPath("{omlib}");')
        if task.needs_transform:
            setup_lines.append("if not loadModel(Modelica) then print(getErrorString()); exit(1); end if;")
            if tf_lib:
                setup_lines.append(f'if not loadFile("{tf_lib}") then print(getErrorString()); exit(1); end if;')
        setup_lines += [
            f'if not loadFile("{task.model_file}") then print(getErrorString()); exit(1); end if;',
            f'res := buildModel({task.model_name}, startTime=0.0, stopTime={task.t_end}, '
            f'numberOfIntervals=500, tolerance=1e-10, method="dassl", outputFormat="csv");',
            'if res[1] == "" then print(getErrorString()); exit(1); end if;',
        ]
        (work / "build_gold.mos").write_text("\n".join(setup_lines) + "\n", encoding="utf-8")
        b = subprocess.run(prefix + ["omc", "build_gold.mos"], cwd=work, capture_output=True, text=True, timeout=300)
        if b.returncode != 0 or not (work / task.model_name).is_file():
            raise RuntimeError(f"OpenModelica buildModel failed for {task.model_name}:\n{b.stdout}\n{b.stderr}")

        out_rows = []
        for idx, p_vals in enumerate(task.test_cases):
            override = ",".join(f"{k}={v:.12g}" for k, v in zip(task.param_names, p_vals))
            csv_name = f"case_{idx}.csv"
            r = subprocess.run(
                prefix + [f"./{task.model_name}", f"-override={override}", f"-r={csv_name}"],
                cwd=work, capture_output=True, text=True, timeout=120,
            )
            if r.returncode != 0 or not (work / csv_name).is_file():
                raise RuntimeError(f"OpenModelica case {idx} ({override}) failed:\n{r.stdout}\n{r.stderr}")
            with open(work / csv_name, encoding="utf-8") as fh:
                last = list(csv.DictReader(fh))[-1]
            out_rows.append([float(last[col]) for col in task.obs_names])
        return out_rows
    finally:
        shutil.rmtree(work, ignore_errors=True)


def make_grader(
    task_name: str,
    *,
    runner: ShellRunner | None = None,
    n_ensemble: int = 1000,
    repeats: int = 5,
    timeout_s: float = 240.0,
    gold_override: list[list[float]] | None = None,
):
    task = TASKS[task_name]
    cached_gold: dict[str, list[list[float]]] = {}
    turn: asyncio.Lock | None = None

    async def grade(result: RunResult | Any) -> Grade:
        nonlocal turn
        if turn is None:
            turn = asyncio.Lock()
        async with turn:
            return await _grade_one(Path(result.workspace).resolve())

    async def _grade_one(ws: Path) -> Grade:
        cand = ws / DELIVERABLE
        if not cand.is_file():
            return Grade(0.0, f"no {DELIVERABLE} in the workspace")

        code = cand.read_text(encoding="utf-8", errors="replace")
        bad = FORBIDDEN.search(code)
        if bad:
            return Grade(0.0, f"forbidden construct `{bad.group(0)}` in {DELIVERABLE}")

        if "gold" not in cached_gold:
            if gold_override is not None:
                cached_gold["gold"] = [list(map(float, row)) for row in gold_override]
            else:
                scratch = Path(os.environ["MESH_RUN_SCRATCH"]) if os.environ.get("MESH_RUN_SCRATCH") else None
                cached_gold["gold"] = await asyncio.to_thread(compute_omc_gold, task, scratch)
        gold = cached_gold["gold"]

        eval_dir = ws / ".mesh" / "grade"
        eval_dir.mkdir(parents=True, exist_ok=True)
        worker_copy = eval_dir / "eval_worker.py"
        payload_json = eval_dir / "payload.json"
        shutil.copyfile(EVAL_WORKER_PY, worker_copy)
        payload_json.write_text(
            json.dumps({
                "candidate": str(cand),
                "y0_default": list(task.y0_default),
                "test_cases": task.test_cases,
                "gold": gold,
                "obs_names": task.obs_names,
                "param_names": task.param_names,
                "traj_rtol": task.traj_rtol,
                "traj_atol": task.traj_atol,
                "n_ensemble": n_ensemble,
                "repeats": repeats,
                "bounds_lo": task.ensemble_bounds[0],
                "bounds_hi": task.ensemble_bounds[1],
                "conservation_kind": task.conservation_kind,
                "stage_holdups": list(STAGE_HOLDUPS_16),
            }),
            encoding="utf-8",
        )
        try:
            if runner is not None:
                r = await runner.run(f"python {worker_copy} {payload_json}", ws, timeout_s)
                if r.timed_out:
                    return Grade(0.0, f"{DELIVERABLE} timed out after {timeout_s:.0f}s")
                exit_code, combined_out = r.exit_code, r.output.strip()
            else:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(worker_copy), str(payload_json),
                    cwd=ws, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                try:
                    out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    return Grade(0.0, f"{DELIVERABLE} timed out after {timeout_s:.0f}s")
                exit_code = proc.returncode
                combined_out = (out_b.decode("utf-8", "replace") + "\n" + err_b.decode("utf-8", "replace")).strip()

            lines = [ln for ln in combined_out.splitlines() if ln.strip().startswith("{")]
            if exit_code != 0 or not lines:
                return Grade(0.0, f"{DELIVERABLE} failed (exit {exit_code}): {combined_out[-400:]}")

            report = json.loads(lines[-1])
            if not report.get("ok"):
                return Grade(0.0, f"[{report.get('stage')}] {report.get('reason')}")

            t_comb = float(report["t_combined"])
            sc = score_of(t_comb)
            return Grade(
                sc,
                f"passed all 4 gates on {report['device']} | "
                f"1k-fwd={report['t_fwd_med']:.4f}s + 64-grad={report['t_grad_med']:.4f}s "
                f"(total {t_comb:.4f}s) | omc_err={report['max_traj_err']:.2e}, "
                f"mass_err={report['cons_err']:.2e}, adj_err={report['rel_cl_err']:.2e}/{report['rel_fresh_err']:.2e}",
            )
        finally:
            shutil.rmtree(eval_dir, ignore_errors=True)

    return grade
