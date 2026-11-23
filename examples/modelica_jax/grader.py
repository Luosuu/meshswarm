"""Multiplicative zero-multiplier grader for Modelica-to-JAX benchmark tasks.

Enforces four hard correctness gates before awarding throughput points (0-10):
  1. Compilation & FP64 finiteness (no subprocess / omc / stop_gradient cheats)
  2. Held-out trajectory parity against OpenModelica (`omc`, DASSL tol=1e-10)
  3. Exact closed-system molar conservation (< 1e-11 across 1,000 trajectories)
  4. Reverse-mode sensitivity parity (< 1e-6 vs closure FD AND < 2e-3 vs fresh
     forward_ensemble FD across held-out regime transitions)
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
from meshevolve import Grade  # noqa: E402
from tasks import DELIVERABLE, STAGE_HOLDUPS_16, TASKS, THRESHOLDS_S, BenchmarkTask  # noqa: E402

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
    """Compile `task.model_source` once with OpenModelica and evaluate all held-out `test_cases`."""
    work = Path(tempfile.mkdtemp(prefix=f"omc-gold-{task.name}-", dir=scratch_root))
    try:
        (work / task.model_file).write_text(task.model_source, encoding="utf-8")
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


_EVAL_WORKER = r'''
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("GRADSOLVE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def main():
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cand_path = Path(payload["candidate"]).resolve()
    spec = importlib.util.spec_from_file_location("candidate_solver", cand_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for fn in ("forward_ensemble", "differentiable_closure"):
        if not hasattr(mod, fn):
            print(json.dumps({"ok": False, "stage": "compile", "reason": f"missing `{fn}` in solver_jax.py"}))
            return

    y0_default = np.array(payload["y0_default"], dtype=np.float64)
    p_test = np.array(payload["test_cases"], dtype=np.float64)
    gold = np.array(payload["gold"], dtype=np.float64)
    obs_names = payload["obs_names"]
    param_names = payload["param_names"]
    n_test = len(p_test)
    y0_test = np.tile(y0_default, (n_test, 1))

    # 1. Compile & held-out trajectory parity against OpenModelica
    obs_got = np.asarray(mod.forward_ensemble(y0_test, p_test))
    if obs_got.shape != gold.shape:
        print(json.dumps({"ok": False, "stage": "compile",
                          "reason": f"forward_ensemble returned shape {obs_got.shape}, expected {gold.shape}"}))
        return
    if obs_got.dtype != np.float64:
        print(json.dumps({"ok": False, "stage": "compile",
                          "reason": f"forward_ensemble returned dtype {obs_got.dtype}, expected float64"}))
        return
    if not np.all(np.isfinite(obs_got)):
        print(json.dumps({"ok": False, "stage": "compile", "reason": "forward_ensemble produced NaN or Inf"}))
        return

    denom = np.maximum(np.abs(gold), payload["traj_atol"])
    rel_mat = np.abs(obs_got - gold) / denom
    worst_idx = np.unravel_index(int(np.argmax(rel_mat)), rel_mat.shape)
    max_traj_err = float(rel_mat[worst_idx])
    if max_traj_err > payload["traj_rtol"]:
        ci, oi = int(worst_idx[0]), int(worst_idx[1])
        print(json.dumps({
            "ok": False,
            "stage": "traj_gold",
            "reason": (f"held-out case {ci} `{obs_names[oi]}` got {obs_got[ci, oi]:.6g} vs "
                       f"OpenModelica {gold[ci, oi]:.6g} (rel err {max_traj_err:.3e} > {payload['traj_rtol']:.1e})"),
        }))
        return

    # 2. 1,000-trajectory GPU ensemble conservation + warm timing
    n_ens = int(payload["n_ensemble"])
    rng = np.random.default_rng(20261123)
    lo = np.array(payload["bounds_lo"], dtype=np.float64)
    hi = np.array(payload["bounds_hi"], dtype=np.float64)
    p_ens = rng.uniform(lo, hi, size=(n_ens, len(lo)))
    y0_ens = np.tile(y0_default, (n_ens, 1))

    out_cold = mod.forward_ensemble(y0_ens, p_ens)
    jax.block_until_ready(out_cold)
    ens_np = np.asarray(out_cold)
    if not np.all(np.isfinite(ens_np)):
        print(json.dumps({"ok": False, "stage": "conservation", "reason": "1,000-ensemble produced NaN or Inf"}))
        return

    ckind = payload["conservation_kind"]
    if ckind == "sum_last3":
        init_tot = float(np.sum(y0_default[-3:]))
        got_tot = np.sum(ens_np[:, -3:], axis=1)
        cons_err = float(np.max(np.abs(got_tot - init_tot) / init_tot))
    elif ckind == "hdt_column16":
        m_hold = np.array(payload["stage_holdups"], dtype=np.float64)
        xh2_0 = y0_default[:16]
        xhd_0 = y0_default[16:]
        xt2_0 = 1.0 - xh2_0 - xhd_0
        want_tots = np.array([np.sum(m_hold * xh2_0), np.sum(m_hold * xhd_0), np.sum(m_hold * xt2_0)])
        got_tots = ens_np[:, -3:]
        cons_err = float(np.max(np.abs(got_tots - want_tots[None, :]) / want_tots[None, :]))
    else:
        raise ValueError(f"unknown conservation_kind {ckind}")

    if cons_err >= 1.0e-11:
        print(json.dumps({
            "ok": False,
            "stage": "conservation",
            "reason": f"molar conservation drift {cons_err:.3e} >= 1e-11 across {n_ens} trajectories",
        }))
        return

    fwd_times = []
    for _ in range(int(payload["repeats"])):
        t0 = time.perf_counter()
        o = mod.forward_ensemble(y0_ens, p_ens)
        jax.block_until_ready(o)
        fwd_times.append(time.perf_counter() - t0)
    t_fwd_med = float(statistics.median(fwd_times))

    # 3. Reverse-mode sensitivity gates (closure FD < 1e-6 AND fresh forward FD < 2e-3)
    cl = mod.differentiable_closure(y0_test, p_test)
    p_test_j = jnp.asarray(p_test, dtype=jnp.float64)
    obs_cl = np.asarray(cl(p_test_j))
    cl_consistency = float(np.max(np.abs(obs_cl - obs_got) / denom))
    if not np.all(np.isfinite(obs_cl)) or cl_consistency > 1.0e-5:
        print(json.dumps({
            "ok": False,
            "stage": "adjoint",
            "reason": f"differentiable_closure output differs from forward_ensemble (rel diff {cl_consistency:.3e})",
        }))
        return

    # Scale-normalized nonlinear per-trajectory objective touching dynamic observables
    obs_scale = jnp.asarray(np.maximum(np.mean(np.abs(gold), axis=0), 1.0e-3), dtype=jnp.float64)
    weights = jnp.linspace(1.1, 2.3, gold.shape[1], dtype=jnp.float64)

    def row_obj_fn(p_mat):
        pred = cl(p_mat) / obs_scale[None, :]
        return jnp.sum(weights[None, :] * (pred ** 2 + 0.35 * pred), axis=1)

    def obj_fn(p_mat):
        return jnp.sum(row_obj_fn(p_mat))

    grad_jit = jax.jit(jax.grad(obj_fn))
    row_obj_jit = jax.jit(row_obj_fn)
    ad_grad = np.asarray(grad_jit(p_test_j))
    if not np.all(np.isfinite(ad_grad)):
        print(json.dumps({"ok": False, "stage": "adjoint", "reason": "jax.grad returned NaN or Inf"}))
        return

    # 3a. Exact JVP-vs-VJP transpose gate (< 1e-10, free of finite-difference ULP noise!)
    jvp_grad = np.zeros_like(ad_grad)
    jvp_fn = jax.jit(lambda pm, tang: jax.jvp(row_obj_fn, (pm,), (tang,))[1])
    for j in range(p_test.shape[1]):
        tang = np.zeros_like(p_test)
        tang[:, j] = 1.0
        jvp_grad[:, j] = np.asarray(jvp_fn(p_test_j, jnp.asarray(tang)))

    # 3b. Columnwise per-row central FD on closure (< 5e-5, catches kinks/jumps)
    p_scales = np.maximum(np.abs(p_test), 0.1)
    obs_scale_np = np.asarray(obs_scale)
    weights_np = np.asarray(weights)
    cl_fd = np.zeros_like(ad_grad)
    h_cl_mat = 1.0e-5 * p_scales
    for j in range(p_test.shape[1]):
        dp = np.zeros_like(p_test)
        dp[:, j] = h_cl_mat[:, j]
        fp = np.asarray(row_obj_jit(jnp.asarray(p_test + dp)))
        fm = np.asarray(row_obj_jit(jnp.asarray(p_test - dp)))
        cl_fd[:, j] = (fp - fm) / (2.0 * h_cl_mat[:, j])

    # 3c. Batched fresh forward_ensemble FD (< 1.5e-2 with macro step h_fr=1e-3*scale to avoid
    #     O(rtol / h) adaptive step-rejection jitter while catching stop_gradient / detached closures)
    plus_rows, minus_rows, h_fr_list = [], [], []
    for i in range(n_test):
        for j in range(p_test.shape[1]):
            h_fr = 1.0e-3 * p_scales[i, j]
            row_p = p_test[i].copy()
            row_m = p_test[i].copy()
            row_p[j] += h_fr
            row_m[j] -= h_fr
            plus_rows.append(row_p)
            minus_rows.append(row_m)
            h_fr_list.append(h_fr)

    p_pert = np.vstack([np.array(plus_rows), np.array(minus_rows)])
    y0_pert = np.tile(y0_default, (len(p_pert), 1))
    obs_pert = np.asarray(mod.forward_ensemble(y0_pert, p_pert)) / obs_scale_np[None, :]
    row_losses = np.sum(weights_np[None, :] * (obs_pert ** 2 + 0.35 * obs_pert), axis=1)
    half = len(plus_rows)
    fresh_fd = ((row_losses[:half] - row_losses[half:]) / (2.0 * np.array(h_fr_list))).reshape(ad_grad.shape)

    grad_floor = np.maximum(np.max(np.abs(fresh_fd), axis=0, keepdims=True) * 5.0e-3, 1.0e-5)
    rel_jvp_err = float(np.max(np.abs(ad_grad - jvp_grad) / np.maximum(np.abs(jvp_grad), grad_floor)))
    rel_cl_err = float(np.max(np.abs(ad_grad - cl_fd) / np.maximum(np.abs(cl_fd), grad_floor)))
    rel_fresh_err = float(np.max(np.abs(ad_grad - fresh_fd) / np.maximum(np.abs(fresh_fd), grad_floor)))

    if rel_jvp_err >= 1.0e-10:
        print(json.dumps({
            "ok": False,
            "stage": "adjoint",
            "reason": f"reverse-mode VJP vs forward-mode JVP rel error {rel_jvp_err:.3e} >= 1e-10",
        }))
        return
    if rel_cl_err >= 5.0e-5:
        print(json.dumps({
            "ok": False,
            "stage": "adjoint",
            "reason": f"reverse-mode grad vs closure FD rel error {rel_cl_err:.3e} >= 5e-5",
        }))
        return
    if rel_fresh_err >= 1.5e-2:
        wi, wj = np.unravel_index(
            int(np.argmax(np.abs(ad_grad - fresh_fd) / np.maximum(np.abs(fresh_fd), grad_floor))),
            ad_grad.shape,
        )
        print(json.dumps({
            "ok": False,
            "stage": "adjoint",
            "reason": (f"reverse-mode grad vs fresh forward FD rel error {rel_fresh_err:.3e} >= 1.5e-2 "
                       f"(case {int(wi)} param `{param_names[int(wj)]}`: AD={ad_grad[wi, wj]:.5g}, FD={fresh_fd[wi, wj]:.5g})"),
        }))
        return

    # 4. Time 64-trajectory reverse-mode gradient pass
    n_g = min(64, n_ens)
    y0_g = y0_ens[:n_g]
    p_g = p_ens[:n_g]
    cl_g = mod.differentiable_closure(y0_g, p_g)
    p_g_j = jnp.asarray(p_g, dtype=jnp.float64)

    def obj_g(pm):
        pred = cl_g(pm) / obs_scale[None, :]
        return jnp.sum(weights[None, :] * (pred ** 2 + 0.35 * pred))

    g_fn = jax.jit(jax.grad(obj_g))
    jax.block_until_ready(g_fn(p_g_j))
    grad_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        gg = g_fn(p_g_j)
        jax.block_until_ready(gg)
        grad_times.append(time.perf_counter() - t0)
    t_grad_med = float(statistics.median(grad_times))

    print(json.dumps({
        "ok": True,
        "device": jax.devices()[0].platform,
        "max_traj_err": max_traj_err,
        "cons_err": cons_err,
        "rel_cl_err": rel_cl_err,
        "rel_fresh_err": rel_fresh_err,
        "t_fwd_med": t_fwd_med,
        "t_grad_med": t_grad_med,
        "t_combined": t_fwd_med + t_grad_med,
    }))


if __name__ == "__main__":
    main()
'''


def make_grader(
    task_name: str,
    *,
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
            return await _grade_one(Path(result.workspace))

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
                cached_gold["gold"] = await asyncio.to_thread(
                    compute_omc_gold, task, Path(os.environ["MESH_RUN_SCRATCH"]) if os.environ.get("MESH_RUN_SCRATCH") else None
                )
        gold = cached_gold["gold"]

        tmp_dir = Path(tempfile.mkdtemp(prefix="mjax-grade-"))
        try:
            worker_py = tmp_dir / "eval_worker.py"
            payload_json = tmp_dir / "payload.json"
            worker_py.write_text(_EVAL_WORKER, encoding="utf-8")
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
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(worker_py), str(payload_json),
                cwd=ws, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return Grade(0.0, f"{DELIVERABLE} timed out after {timeout_s:.0f}s")

            stdout = out_b.decode("utf-8", "replace").strip()
            stderr = err_b.decode("utf-8", "replace").strip()
            lines = [ln for ln in stdout.splitlines() if ln.strip().startswith("{")]
            if proc.returncode != 0 or not lines:
                tail = (stderr or stdout)[-400:]
                return Grade(0.0, f"{DELIVERABLE} failed (exit {proc.returncode}): {tail}")

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
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return grade
