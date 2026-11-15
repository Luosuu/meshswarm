"""Run meshevolve on a performance task (AMR, its stricter variant amrx, PIC or LULESH) with a sandboxed compile-and-run tool.

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
import lulesh  # noqa: E402
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


def _cpu_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def plan_core_sets(allowed: list[int], siblings: dict[int, list[int]], nodes: dict[int, list[int]],
                   per_swarm: int, swarms: int) -> list[list[int]]:
    """Choose `swarms` disjoint sets of `per_swarm` processors: one hardware thread per physical core,
    and each set inside one memory region (NUMA node) when the machine allows it. Timings on two
    threads of one core, or on cores spread over several regions, are slower and less repeatable."""
    seen: set[int] = set()
    cores = []
    for c in sorted(allowed):
        key = min(siblings.get(c, [c]))
        if key not in seen:
            seen.add(key)
            cores.append(c)
    region = {c: n for n, cpus in nodes.items() for c in cpus}
    groups: dict[int, list[int]] = {}
    for c in cores:
        groups.setdefault(region.get(c, -1), []).append(c)
    sets: list[list[int]] = []
    leftovers: list[int] = []
    for _, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        while len(group) >= per_swarm and len(sets) < swarms:
            sets.append(group[:per_swarm])
            group = group[per_swarm:]
        leftovers += group
    while len(sets) < swarms and len(leftovers) >= per_swarm:  # no single region is big enough: mix
        sets.append(leftovers[:per_swarm])
        leftovers = leftovers[per_swarm:]
    if len(sets) < swarms:
        raise SystemExit(f"need {per_swarm * swarms} physical cores for {swarms} swarms, but only {len(cores)} are available")
    return sets


def core_sets(per_swarm: int, swarms: int) -> list[str]:
    """Core sets for this machine, from the processors this process may use. Empty where the OS cannot say."""
    if not hasattr(os, "sched_getaffinity"):
        return []
    allowed = sorted(os.sched_getaffinity(0))
    siblings, nodes = {}, {}
    for c in allowed:
        f = Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        if f.is_file():
            siblings[c] = _cpu_list(f.read_text())
    for d in Path("/sys/devices/system/node").glob("node[0-9]*"):
        nodes[int(d.name[4:])] = _cpu_list((d / "cpulist").read_text())
    sets = plan_core_sets(allowed, siblings, nodes, per_swarm, swarms)
    region = {c: n for n, cpus in nodes.items() for c in cpus}
    for i, chosen in enumerate(sets):
        regions = sorted({region.get(c, -1) for c in chosen})
        print(f"core set {i + 1}: {chosen} (memory region{'s' if len(regions) > 1 else ''} {regions})", flush=True)
    return [",".join(map(str, chosen)) for chosen in sets]


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


async def selftest_lulesh(runner: ShellRunner) -> int:
    """Build and run the original, then check the grader: the original scores no speedup, a version that
    is deliberately slower still validates, and a version that computes something else is rejected."""
    ws = Path(tempfile.mkdtemp(prefix="hpc-selftest-", dir=os.environ.get("MESH_RUN_SCRATCH")))
    for rel, text in lulesh.starting_files().items():
        (ws / rel).write_text(text)
    print(f"workspace {ws}; cores {runner.cores_for(ws)}")
    r = await runner.run("${CXX:-g++} --version | head -1; nproc; bash build.sh && ./lulesh %s" % lulesh.DEV_ARGS, ws, 600)
    print(f"build + example run (exit {r.exit_code}, {r.seconds}s):\n{r.output.strip()[-700:]}")
    if r.exit_code != 0 or lulesh.answer(r.output) is None:
        return 1

    class Result:
        workspace = ws

    grade = lulesh.make_grader(runner)
    g = await grade(Result())
    print(f"untouched source -> {g}")
    if "speedup" not in g.reasons:
        return 1
    src = ws / "lulesh.cc"
    original = src.read_text()
    changed = original.replace("Real_t(0.01)", "Real_t(0.02)", 1)  # a different hourglass coefficient
    if changed == original:
        print("could not make the wrong-answer variant")
        return 1
    src.write_text(changed)
    g = await grade(Result())
    print(f"different physics -> {g}")
    if g.score != 0.0 or "the original gives" not in g.reasons:
        return 1
    print("SELFTEST_OK")
    return 0


async def selftest(name: str, runner: ShellRunner, cores: int) -> int:
    if name == "lulesh":
        return await selftest_lulesh(runner)
    ws = Path(tempfile.mkdtemp(prefix="hpc-selftest-", dir=os.environ.get("MESH_RUN_SCRATCH")))
    for rel, text in kernels.starting_files(name).items():
        (ws / rel).write_text(text)
    print(f"workspace {ws}; cores {runner.cores_for(ws)}")
    (ws / "omp_probe.c").write_text(OMP_PROBE)
    probe = ("${CC:-gcc} --version | head -1 && ${CC:-gcc} -fopenmp omp_probe.c -o omp_probe && ./omp_probe; nproc; "
             "(ping -c1 -W1 1.1.1.1 >/dev/null 2>&1 && echo network:reachable || echo network:none); ls ~ 2>&1 | head -3")
    r = await runner.run(probe, ws, 120)
    print(f"sandbox probe (exit {r.exit_code}, {r.seconds}s):\n{r.output.strip()}")
    exe, by_norms = kernels.binary(name), bool(kernels.KERNELS[name].get("norms"))
    r = await runner.run("bash build.sh && ./%s %s" % (exe, kernels.KERNELS[name]["dev_args"]), ws, 300)
    print(f"build + example run (exit {r.exit_code}, {r.seconds}s):\n{r.output.strip()[-400:]}")
    if r.exit_code != 0 or (kernels.norms(r.output) is None if by_norms else "Solution validates" not in r.output):
        return 1

    class Result:
        workspace = ws

    grade = kernels.make_grader(name, runner)
    g = await grade(Result())
    print(f"untouched source -> {g}")
    if g.score != 0.0 or "speedup" not in g.reasons:
        return 1
    src = ws / f"{exe}.c"
    if by_norms:  # the shortcut every swarm took on the published AMR: skip the sweeps that feed the input field
        src.write_text(src.read_text().replace("IN(i,j) += FEEDBACK*OUT(i,j);", ";"))
        g = await grade(Result())
        print(f"input field no longer updated -> {g}")
        if g.score != 0.0 or "the original gives" not in g.reasons:
            return 1
    else:
        src.write_text(src.read_text().replace('printf("Solution validates\\n");', 'printf("ok\\n");'))
        g = await grade(Result())
        print(f"verification removed -> {g}")
        if g.score != 0.0 or "verification" not in g.reasons:
            return 1
    print("SELFTEST_OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=sorted(kernels.KERNELS) + ["lulesh"], default="amr")
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

    if a.kernel == "lulesh":
        task, grader, files, protect = lulesh.task_text(a.cores), lulesh.make_grader(runner), lulesh.starting_files(), lulesh.PROTECT
    else:
        task, grader, files, protect = (kernels.task_text(a.kernel, a.cores), kernels.make_grader(a.kernel, runner),
                                        kernels.starting_files(a.kernel), ["build.sh", "*.c", "*.h"])
    result = asyncio.run(evolve(
        task, model, generations=a.generations, population=a.population, agents=a.agents,
        out=out, limits=limits, pricing=pricing, grade=grader,
        tools=[run_tool(runner, f"gcc and g++ with OpenMP are available; you have {a.cores} cores.")],
        deliverable="build.sh", protect=protect, files=files, kind=kernels.KIND, parallel=a.parallel,
        lessons=starting_lessons(a.lessons), on_generation=progress))
    u = result.review_usage
    print(f"reviewer tokens: {u.prompt} in, {u.completion} out (about ${pricing.cost(u):.4f})")
    print(f"summary: {result.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
