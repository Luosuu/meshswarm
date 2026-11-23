"""Phase 1/2 end-to-end self-test across all 3 Modelica-to-JAX benchmark tiers.

Reads standalone `.mo` models from `models/` and reference solvers from `references/`.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rumoca as rm  # noqa: E402

from grader import make_grader  # noqa: E402
from meshswarm.shell import ShellRunner  # noqa: E402
from tasks import DELIVERABLE, REFS_DIR, TASKS, reference_path, starting_files  # noqa: E402


class _WS:
    def __init__(self, workspace: Path):
        self.workspace = workspace


def verify_rumoca_fails_directly(ws: Path, model_file: str, model_name: str) -> str:
    try:
        sess = rm.Session()
        m = sess.load(str(ws / model_file), model=model_name)
        m.to_jax()
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
    raise AssertionError(f"Expected direct Rumoca.to_jax() to fail on {model_name}, but it succeeded!")


async def run_tier_selftest(name: str, n_ensemble: int = 1000, runner: ShellRunner | None = None) -> None:
    print(f"\n================ {name} ================", flush=True)
    scratch = Path(tempfile.gettempdir())
    if runner is not None:
        import os
        if os.environ.get("MESH_RUN_SCRATCH"):
            scratch = Path(os.environ["MESH_RUN_SCRATCH"])
    ws = Path(tempfile.mkdtemp(prefix=f"selftest-{name}-", dir=scratch))
    try:
        for rel, content in starting_files(name).items():
            (ws / rel).write_text(content, encoding="utf-8")

        task = TASKS[name]
        why_rumoca = verify_rumoca_fails_directly(ws, task.model_file, task.model_name)
        print(f"[1] Direct Rumoca.to_jax() blocked as expected -> {why_rumoca}", flush=True)

        grade = make_grader(name, runner=runner, n_ensemble=n_ensemble)

        g_skel = await grade(_WS(ws))
        print(f"[2] Starter skeleton -> score={g_skel.score} | {g_skel.reasons[:140]}", flush=True)
        assert g_skel.score == 0.0

        ref_code = reference_path(name).read_text(encoding="utf-8")
        (ws / DELIVERABLE).write_text(ref_code, encoding="utf-8")
        g_ref = await grade(_WS(ws))
        print(f"[3] Verified reference -> score={g_ref.score} | {g_ref.reasons}", flush=True)
        assert g_ref.score is not None and g_ref.score > 0.0, f"Reference failed on {name}: {g_ref}"

        cheat_detach = ref_code + (
            "\n_orig_dc = differentiable_closure\n"
            "def differentiable_closure(y0_b, p_b):\n"
            "    base = forward_ensemble(y0_b, p_b)\n"
            "    return lambda p_jax: base + 0.0 * jnp.sum(p_jax)\n"
        )
        (ws / DELIVERABLE).write_text(cheat_detach, encoding="utf-8")
        g_detach = await grade(_WS(ws))
        print(f"[4] Zero-grad cheat rejected -> score={g_detach.score} | {g_detach.reasons}", flush=True)
        assert g_detach.score == 0.0 and "[adjoint]" in g_detach.reasons

        cheat_leak = ref_code + (
            "\n_orig_fe = forward_ensemble\n"
            "def forward_ensemble(y0_b, p_b):\n"
            "    out = jnp.asarray(_orig_fe(y0_b, p_b))\n"
            "    return out.at[:, -1].add(1.0e-6)\n"
        )
        (ws / DELIVERABLE).write_text(cheat_leak, encoding="utf-8")
        g_leak = await grade(_WS(ws))
        print(f"[5] Mass-leak cheat rejected -> score={g_leak.score} | {g_leak.reasons}", flush=True)
        assert g_leak.score == 0.0 and "[conservation]" in g_leak.reasons

        if name == "tier2_transform_hx":
            wrong_cv = ref_code.replace("CV_PBLI = 9637.0, 188.0, 141.0", "CV_PBLI = 9637.0, 188.0, 188.0")
            (ws / DELIVERABLE).write_text(wrong_cv, encoding="utf-8")
            g_cv = await grade(_WS(ws))
            print(f"[6] Wrong cp-vs-cv physics rejected -> score={g_cv.score} | {g_cv.reasons}", flush=True)
            assert g_cv.score == 0.0 and "[traj_gold]" in g_cv.reasons

        if name == "tier3_hdt_column":
            generic_code = (REFS_DIR / "tier3_generic_baseline.py").read_text(encoding="utf-8")
            (ws / DELIVERABLE).write_text(generic_code, encoding="utf-8")
            g_gen = await grade(_WS(ws))
            print(f"[6] Tier 3 generic tsit5_replay baseline -> score={g_gen.score} | {g_gen.reasons}", flush=True)
            assert g_gen.score is not None and g_gen.score > 0.0
            assert g_ref.score >= g_gen.score
    finally:
        shutil.rmtree(ws, ignore_errors=True)


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["all", *TASKS.keys()], default="all")
    ap.add_argument("--n-ensemble", type=int, default=1000)
    args = ap.parse_args()

    chosen = list(TASKS.keys()) if args.tier == "all" else [args.tier]
    for name in chosen:
        await run_tier_selftest(name, args.n_ensemble)
    print("\nPHASE1_ALL_TIERS_SELFTEST_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
