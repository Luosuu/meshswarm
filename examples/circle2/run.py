"""Run meshevolve on the two-circle packing task with real Lean checking.

    python examples/circle2/run.py --lean-cmd '["lean"]' --generations 3 --population 2 --control

Needs MESH_BASE_URL, MESH_MODEL and an API key (--api-key-file, NAME=key lines are fine).
`--selftest` only checks the Lean setup and the grader, without calling any model.
"""

from __future__ import annotations

import argparse
import importlib
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from meshswarm import ChatEndpoint, Limits  # noqa: E402
from meshswarm.lean import LeanChecker, lean_tool  # noqa: E402
from meshswarm.pricing import lookup  # noqa: E402
from meshevolve import evolve  # noqa: E402

TINKER = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


def read_key(path: str, name: str) -> str:
    """The value of NAME=... in an env-style file, or the file's only line."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    for ln in lines:
        if ln.startswith(name + "="):
            return ln.split("=", 1)[1].strip().strip("'\"")
    if len(lines) == 1 and "=" not in lines[0]:
        return lines[0]
    raise ValueError(f"{name} not found in {path}")


def starting_lessons(spec: str | None) -> str:
    """`spec` is one file, or several separated by commas: the first that exists is used. That lets a
    queued run prefer the lessons of the run before it and fall back if that run left none."""
    for path in (spec or "").split(","):
        if path.strip() and Path(path.strip()).is_file():
            print(f"starting lessons: {path.strip()}", flush=True)
            return Path(path.strip()).read_text()
    if spec:
        print("starting lessons: none of the given files exist; starting without lessons", flush=True)
    return ""


async def warm_up(checker: LeanChecker) -> None:
    """Load Mathlib once with a generous limit: the first import on a node with a cold file cache
    can take minutes, later ones seconds."""
    limit, checker.timeout_s = checker.timeout_s, 1500
    try:
        for attempt in (1, 2):
            r = await checker.check("example : (1 : ℕ) + 1 = 2 := rfl\n", name="warmup.lean")
            print(f"warm-up {attempt}: ok={r.ok} in {r.seconds}s", flush=True)
            if not r.ok:
                raise SystemExit(f"Lean is not working:\n{r.shown[:2000]}")
    finally:
        checker.timeout_s = limit


async def selftest(checker: LeanChecker) -> int:
    """Check that Lean runs, the header compiles, the skeleton grades 0 and a cheat is rejected."""
    r = await checker.check("example : (2 : ℝ) + 2 = 4 := by norm_num\n", name="smoke.lean")
    print(f"header + smoke: ok={r.ok} in {r.seconds}s\n{r.shown.strip()[:6000]}")
    if not r.ok:
        return 1
    r = await checker.check(task.SKELETON, name="skeleton.lean", footer=task.footer(task.TARGETS))
    errors = "\n".join(ln for ln in r.shown.splitlines() if "error" in ln)[:4000]
    print(f"skeleton: compiles={r.ok} uses_sorry={r.uses_sorry} proved={task.proved(r.raw, task.TARGETS)}\n{errors}\n{r.shown.strip()[:3000] if not r.ok else ''}")
    if not r.ok or any(task.proved(r.raw, task.TARGETS).values()):
        return 1
    first = task.TARGETS[0][0]
    cheat = task.SKELETON.replace(f"theorem {first}", f"axiom cheat : False\ntheorem {first}").replace(
        "by\n  sorry", "by\n  exact cheat.elim", 1)
    r = await checker.check(cheat, name="cheat.lean", footer=task.footer(task.TARGETS))
    print(f"axiom cheat: proved={task.proved(r.raw, task.TARGETS)}")
    if any(task.proved(r.raw, task.TARGETS).values()):
        return 1
    ref = Path(__file__).with_name("reference.lean" if task.__name__ == "task" else f"reference_{task.__name__}.lean")
    if ref.is_file():
        r = await checker.check(ref.read_text(), name=ref.name, footer=task.footer(task.TARGETS))
        got = [n for n, ok in task.proved(r.raw, task.TARGETS).items() if ok]
        print(f"reference: compiles={r.ok} proved={got}\n{r.shown.strip()[:4000]}")
        if not got:
            return 1
    print("SELFTEST_OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="task", help="task module in this folder: task (n=2), task3 (n=3), task_multi (n=2..5), task_sens (Boolean sensitivity) or task_hard3 (one hard bound)")
    p.add_argument("--lean-cmd", default=os.environ.get("MESH_LEAN_CMD", '["lean"]'), help="JSON list: command that compiles a file")
    p.add_argument("--lean-scratch", default=os.environ.get("MESH_LEAN_SCRATCH"))
    p.add_argument("--lean-timeout", type=float, default=240)
    p.add_argument("--lean-concurrency", type=int, default=8)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--base-url", default=os.environ.get("MESH_BASE_URL", TINKER))
    p.add_argument("--model", default=os.environ.get("MESH_MODEL", "thinkingmachines/Inkling-2-Small"))
    p.add_argument("--api-key-file", default=".env")
    p.add_argument("--api-key-name", default="TINKER_API_KEY")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--population", type=int, default=2)
    p.add_argument("--controls", type=int, default=0, help="swarms per generation that get no lessons")
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--max-cost", type=float, default=0.25, help="dollar budget per swarm")
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=100_000_000,
                   help="token budget per swarm, cached included; set high so that --max-cost is the real limit")
    p.add_argument("--max-agents", type=int, default=8)
    p.add_argument("--max-subagents", type=int, default=16)
    p.add_argument("--timeout", type=float, default=2400, help="seconds per swarm")
    p.add_argument("--max-completion-tokens", type=int, default=8192)
    p.add_argument("--context-window", type=int, default=65536, help="tokens the model accepts for prompt + reply")
    p.add_argument("--compact-at", type=int, default=30000, help="estimated history size at which an agent summarizes itself")
    p.add_argument("--checkin-after", type=float, default=600,
                   help="seconds an agent may sleep, while teammates work and tasks are open, before a check-in wakes it (0 = never)")
    p.add_argument("--goal-grace", type=float, default=300, help="seconds a swarm may keep running after it completes the goal task")
    p.add_argument("--request-timeout", type=float, default=1800, help="seconds to wait for one model reply; long reasoning needs minutes")
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--out")
    p.add_argument("--lessons", help="file with starting lessons; several separated by commas means the first that exists")
    a = p.parse_args()
    global task
    task = importlib.import_module(a.task)

    checker = LeanChecker(json.loads(a.lean_cmd), header=task.HEADER, timeout_s=a.lean_timeout,
                          concurrency=a.lean_concurrency, scratch=Path(a.lean_scratch) if a.lean_scratch else None)
    asyncio.run(warm_up(checker))
    if a.selftest:
        return asyncio.run(selftest(checker))

    pricing = lookup(a.model)
    if pricing is None:
        p.error(f"no built-in prices for {a.model}")
    model = ChatEndpoint(a.base_url, a.model, api_key=read_key(a.api_key_file, a.api_key_name),
                         extra_body={"max_tokens": a.max_completion_tokens}, timeout_s=a.request_timeout,
                         context_window=a.context_window)
    limits = Limits(max_cost_usd=a.max_cost, max_steps_per_agent=a.max_steps, timeout_s=a.timeout,
                    max_total_tokens=a.max_tokens, compact_at_tokens=a.compact_at,
                    checkin_after_s=a.checkin_after or None, goal_grace_s=a.goal_grace,
                    max_agents=a.max_agents, max_subagents=a.max_subagents)
    out = a.out or os.path.join("runs", a.task + "-" + time.strftime("%Y%m%d-%H%M%S"))

    def progress(gen) -> None:
        for r in gen.runs:
            print(f"gen {gen.number} {r.label:8} score {r.grade.score}/10 cost ${r.shape['cost_usd']} "
                  f"checks {r.shape['lean_checks']} ({r.shape['lean_checks_ok']} complete) | {r.grade.reasons}", flush=True)
            when = {n: r.shape["progress"].get(n) for n, _ in task.TARGETS if n in r.shape["progress"]}
            print(f"    first in the deliverable (minutes): " + (", ".join(f"{n} {t // 60}" for n, t in when.items()) or "nothing"), flush=True)

    result = asyncio.run(evolve(
        task.TASK, model, generations=a.generations, population=a.population, control=a.controls, agents=a.agents,
        out=out, limits=limits, pricing=pricing, grade=task.make_grader(checker, task.TARGETS),
        tools=[lean_tool(checker, task.ABOUT_HEADER)], deliverable=task.DELIVERABLE, kind=task.KIND, parallel=a.parallel,
        lessons=starting_lessons(a.lessons), on_generation=progress))
    u = result.review_usage
    print(f"reviewer tokens: {u.prompt} in, {u.completion} out (about ${pricing.cost(u):.4f})")
    print(f"summary: {result.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
