"""Run meshevolve on a performance task (AMR or PIC) with a sandboxed compile-and-run tool.

    python examples/hpc/run.py --kernel amr --generations 3 --population 2

`--selftest` checks the sandbox, the compiler, OpenMP and the grader without calling a model.
MESH_RUN_WRAP (a JSON list) is the command prefix that runs a shell command in the sandbox; see
rivanna/env.sh. Without it commands run directly, which is only for local tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kernels  # noqa: E402
from meshswarm import ChatEndpoint, Limits  # noqa: E402
from meshswarm.pricing import lookup  # noqa: E402
from meshswarm.shell import ShellRunner, run_tool  # noqa: E402
from meshevolve import evolve  # noqa: E402

TINKER = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


def read_key(path: str, name: str) -> str:
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    for ln in lines:
        if ln.startswith(name + "="):
            return ln.split("=", 1)[1].strip().strip("'\"")
    if len(lines) == 1 and "=" not in lines[0]:
        return lines[0]
    raise ValueError(f"{name} not found in {path}")


def starting_lessons(spec: str | None) -> str:
    for path in (spec or "").split(","):
        if path.strip() and Path(path.strip()).is_file():
            print(f"starting lessons: {path.strip()}", flush=True)
            return Path(path.strip()).read_text()
    return ""


def core_sets(per_swarm: int, swarms: int) -> list[str]:
    """Disjoint sets of the cores this process may use, one per swarm. Empty where the OS cannot say."""
    if not hasattr(os, "sched_getaffinity"):
        return []
    cores = sorted(os.sched_getaffinity(0))
    sets = [cores[i * per_swarm:(i + 1) * per_swarm] for i in range(swarms)]
    if any(len(s) < per_swarm for s in sets):
        raise SystemExit(f"need {per_swarm * swarms} cores for {swarms} swarms, but only {len(cores)} are available")
    return [",".join(map(str, s)) for s in sets]


OMP_PROBE = """#include <omp.h>
#include <stdio.h>
int main(void) {
#pragma omp parallel
#pragma omp single
  printf("threads %d\\n", omp_get_num_threads());
  return 0;
}
"""


async def preflight(runner: ShellRunner, cores: int) -> None:
    """Stop before spending anything if the sandbox cannot build and run an OpenMP program on all its cores."""
    ws = Path(tempfile.mkdtemp(prefix="hpc-preflight-", dir=os.environ.get("MESH_RUN_SCRATCH")))
    (ws / "omp_probe.c").write_text(OMP_PROBE)
    r = await runner.run("${CC:-gcc} -fopenmp omp_probe.c -o omp_probe && ./omp_probe", ws, 120)
    print(f"preflight: {r.output.strip()[:200]}", flush=True)
    if r.exit_code != 0 or f"threads {cores}" not in r.output:
        raise SystemExit(f"sandbox preflight failed: expected an OpenMP program to report {cores} threads")


async def selftest(name: str, runner: ShellRunner, cores: int) -> int:
    ws = Path(tempfile.mkdtemp(prefix="hpc-selftest-", dir=os.environ.get("MESH_RUN_SCRATCH")))
    for rel, text in kernels.starting_files(name).items():
        (ws / rel).write_text(text)
    print(f"workspace {ws}; cores {runner.cores_for(ws)}")
    (ws / "omp_probe.c").write_text(OMP_PROBE)
    probe = ("${CC:-gcc} --version | head -1 && ${CC:-gcc} -fopenmp omp_probe.c -o omp_probe && ./omp_probe; nproc; "
             "(ping -c1 -W1 1.1.1.1 >/dev/null 2>&1 && echo network:reachable || echo network:none); ls ~ 2>&1 | head -3")
    r = await runner.run(probe, ws, 120)
    print(f"sandbox probe (exit {r.exit_code}, {r.seconds}s):\n{r.output.strip()}")
    r = await runner.run("bash build.sh && ./%s %s" % (name, kernels.KERNELS[name]["dev_args"]), ws, 300)
    print(f"build + example run (exit {r.exit_code}, {r.seconds}s):\n{r.output.strip()[-400:]}")
    if r.exit_code != 0 or "Solution validates" not in r.output:
        return 1

    class Result:
        workspace = ws

    grade = kernels.make_grader(name, runner)
    g = await grade(Result())
    print(f"untouched source -> {g}")
    if g.score != 0.0 or "speedup" not in g.reasons:
        return 1
    src = ws / f"{name}.c"
    src.write_text(src.read_text().replace('printf("Solution validates\\n");', 'printf("ok\\n");'))
    g = await grade(Result())
    print(f"verification removed -> {g}")
    if g.score != 0.0 or "verification" not in g.reasons:
        return 1
    print("SELFTEST_OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=sorted(kernels.KERNELS), default="amr")
    p.add_argument("--cores", type=int, default=8, help="cores each swarm gets, for its own runs and for grading")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--base-url", default=os.environ.get("MESH_BASE_URL", TINKER))
    p.add_argument("--model", default=os.environ.get("MESH_MODEL", "thinkingmachines/Inkling-2-Small:peft:262144"))
    p.add_argument("--api-key-file", default=".env")
    p.add_argument("--api-key-name", default="TINKER_API_KEY")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--population", type=int, default=2)
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--max-cost", type=float, default=5.0, help="dollar budget per swarm")
    p.add_argument("--max-tokens", type=int, default=400_000_000)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--max-agents", type=int, default=10)
    p.add_argument("--max-subagents", type=int, default=40)
    p.add_argument("--timeout", type=float, default=10800, help="seconds per swarm")
    p.add_argument("--max-completion-tokens", type=int, default=32768)
    p.add_argument("--context-window", type=int, default=262144)
    p.add_argument("--compact-at", type=int, default=120000)
    p.add_argument("--request-timeout", type=float, default=1800)
    p.add_argument("--checkin-after", type=float, default=600)
    p.add_argument("--goal-grace", type=float, default=300)
    p.add_argument("--command-timeout", type=float, default=120)
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--out")
    p.add_argument("--lessons", help="starting lessons file; several separated by commas means the first that exists")
    a = p.parse_args()

    swarms = a.population if a.parallel else 1
    runner = ShellRunner(json.loads(os.environ.get("MESH_RUN_WRAP", '["bash", "-c"]')), timeout_s=a.command_timeout,
                         core_sets=core_sets(a.cores, 1 if a.selftest else swarms))
    if a.selftest:
        return asyncio.run(selftest(a.kernel, runner, a.cores))

    if runner.core_sets:
        asyncio.run(preflight(runner, a.cores))
    pricing = lookup(a.model)
    if pricing is None:
        p.error(f"no built-in prices for {a.model}")
    model = ChatEndpoint(a.base_url, a.model, api_key=read_key(a.api_key_file, a.api_key_name),
                         extra_body={"max_tokens": a.max_completion_tokens}, timeout_s=a.request_timeout,
                         context_window=a.context_window)
    limits = Limits(max_cost_usd=a.max_cost, max_steps_per_agent=a.max_steps, timeout_s=a.timeout,
                    max_total_tokens=a.max_tokens, compact_at_tokens=a.compact_at, max_agents=a.max_agents,
                    max_subagents=a.max_subagents, checkin_after_s=a.checkin_after or None, goal_grace_s=a.goal_grace)
    out = a.out or os.path.join("runs", f"{a.kernel}-" + time.strftime("%Y%m%d-%H%M%S"))

    def progress(gen) -> None:
        for r in gen.runs:
            print(f"gen {gen.number} {r.label:8} score {r.grade.score}/10 cost ${r.shape['cost_usd']} "
                  f"ran {(r.shape['seconds'] or 0) // 60} min, stopped by {r.shape['stop']}, "
                  f"{r.shape['delegations']} delegations | {r.grade.reasons}", flush=True)

    result = asyncio.run(evolve(
        kernels.task_text(a.kernel, a.cores), model, generations=a.generations, population=a.population, agents=a.agents,
        out=out, limits=limits, pricing=pricing, grade=kernels.make_grader(a.kernel, runner),
        tools=[run_tool(runner, f"gcc with OpenMP is available; you have {a.cores} cores.")],
        deliverable="build.sh", files=kernels.starting_files(a.kernel), kind=kernels.KIND, parallel=a.parallel,
        lessons=starting_lessons(a.lessons), on_generation=progress))
    u = result.review_usage
    print(f"reviewer tokens: {u.prompt} in, {u.completion} out (about ${pricing.cost(u):.4f})")
    print(f"summary: {result.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
