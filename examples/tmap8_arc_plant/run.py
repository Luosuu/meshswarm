"""Run meshevolve on the DOE INL TMAP8 ARC 505.6 MWf Whole-Plant Optimization Challenge.

    python examples/tmap8_arc_plant/run.py --selftest --require-gpu
    python examples/tmap8_arc_plant/run.py --generations 3 --population 2 --controls 1 --parallel
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grader import make_grader  # noqa: E402
from meshswarm import ChatEndpoint, Limits  # noqa: E402
from meshswarm.pricing import Pricing, lookup  # noqa: E402
from meshswarm.shell import ShellRunner, run_tool  # noqa: E402
from meshevolve import evolve  # noqa: E402
from plant_spec import PULSED_WINDOW_S  # noqa: E402
from tasks import DELIVERABLE, KIND, PROTECT_PATTERNS, starting_files, task_text  # noqa: E402

TINKER = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


def read_key(path: str, name: str) -> str:
    lines = [
        ln.strip()
        for ln in Path(path).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
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
            return Path(path.strip()).read_text(encoding="utf-8")
    return ""


def detect_gpu_sets(explicit: str | None) -> list[str]:
    raw = explicit if explicit is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [g.strip() for g in raw.split(",") if g.strip()]


def resolve_pricing(
    model_name: str,
    p_in: float | None,
    p_cached: float | None,
    p_out: float | None,
) -> Pricing | None:
    base = lookup(model_name)
    given = (p_in, p_cached, p_out)
    if all(v is None for v in given):
        return base
    if base is None and None in given:
        raise ValueError(f"no built-in prices for {model_name}; specify input, cached, and output prices")
    b = base or Pricing(0.0, 0.0, 0.0)
    return Pricing(
        input=p_in if p_in is not None else b.input,
        cached_input=p_cached if p_cached is not None else b.cached_input,
        output=p_out if p_out is not None else b.output,
    )


async def preflight(runner: ShellRunner, require_gpu: bool = False) -> None:
    """Verify the sandbox has both DOE INL `tmap8-opt` and CUDA GPU `jax` before spending LLM budget."""
    scratch = os.environ.get("MESH_RUN_SCRATCH")
    ws = Path(tempfile.mkdtemp(prefix="tmap8-preflight-", dir=scratch))
    try:
        probe = (
            "(tmap8-opt --version || which tmap8-opt) && "
            "python -c 'import jax, optax, scipy; "
            "devs = jax.devices(); "
            "print(\"JAX_DEVS:\", devs)' && "
            "(ping -c1 -W1 1.1.1.1 >/dev/null 2>&1 && echo network:reachable || echo network:none)"
        )
        r = await runner.run(probe, ws, 120)
        print(f"preflight (exit {r.exit_code}, {r.seconds}s):\n{r.output.strip()}", flush=True)
        if r.exit_code != 0 or "JAX_DEVS:" not in r.output:
            raise SystemExit("sandbox preflight failed: expected tmap8-opt + python(jax, optax, scipy) to succeed")
        if require_gpu and "CudaDevice" not in r.output and "gpu" not in r.output.lower():
            raise SystemExit("sandbox preflight failed: expected JAX inside container to see a CUDA GPU")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


async def selftest_cmd(runner: ShellRunner, pulsed_window_s: float, require_gpu: bool) -> int:
    await preflight(runner, require_gpu=require_gpu)
    import selftest as st

    await st.run_selftest(runner=runner, pulsed_window_s=pulsed_window_s)
    print("RUN_SELFTEST_OK", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", help="comma-separated GPU IDs to pin workspaces across (default: CUDA_VISIBLE_DEVICES)")
    p.add_argument("--require-gpu", action="store_true", help="fail preflight if JAX does not see a GPU")
    p.add_argument("--pulsed-window-s", type=float, default=PULSED_WINDOW_S)
    p.add_argument("--command-timeout", type=float, default=300)
    p.add_argument("--selftest", action="store_true")

    # Primary Architect model
    p.add_argument("--base-url", default=os.environ.get("MESH_BASE_URL", TINKER))
    p.add_argument("--model", default=os.environ.get("MESH_MODEL", "thinkingmachines/Inkling-2-Small:peft:262144"))
    p.add_argument("--api-key-file", default=".env")
    p.add_argument("--api-key-name", default="TINKER_API_KEY")
    p.add_argument("--price-input", type=float)
    p.add_argument("--price-cached", type=float)
    p.add_argument("--price-output", type=float)

    # Optional sub-agent Worker model
    p.add_argument("--sub-base-url", default=os.environ.get("MESH_SUB_BASE_URL"))
    p.add_argument("--sub-model", default=os.environ.get("MESH_SUB_MODEL"))
    p.add_argument("--sub-api-key-file")
    p.add_argument("--sub-api-key-name", default="TINKER_API_KEY")
    p.add_argument("--sub-max-completion-tokens", type=int, default=16384)
    p.add_argument("--sub-context-window", type=int, default=262144)
    p.add_argument("--sub-max-steps", type=int, default=35)
    p.add_argument("--sub-price-input", type=float)
    p.add_argument("--sub-price-cached", type=float)
    p.add_argument("--sub-price-output", type=float)

    # Evolution & swarm budgets
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--population", type=int, default=2)
    p.add_argument("--controls", type=int, default=1, help="swarms per generation that receive no lessons")
    p.add_argument("--parallel", action="store_true", help="run swarms within each generation concurrently")
    p.add_argument("--agents", default="3", help="starting peer count or comma-separated roles")
    p.add_argument("--max-cost", type=float, default=2.5, help="dollar budget per swarm")
    p.add_argument("--max-tokens", type=int, default=400_000_000)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-agents", type=int, default=8)
    p.add_argument("--max-subagents", type=int, default=32)
    p.add_argument("--timeout", type=float, default=5400, help="seconds per swarm")
    p.add_argument("--checkin-after", type=float, default=0)
    p.add_argument("--goal-grace", type=float, default=180)
    p.add_argument("--max-completion-tokens", type=int, default=32768)
    p.add_argument("--context-window", type=int, default=262144)
    p.add_argument("--compact-at", type=int, default=120000)
    p.add_argument("--request-timeout", type=float, default=1800)
    p.add_argument("--run-dir", default="runs")
    p.add_argument("--out")
    p.add_argument("--lessons", help="seed lessons file (comma-separated fallback list)")
    a = p.parse_args()

    wrap = json.loads(os.environ.get("MESH_RUN_WRAP", '["bash", "-c"]'))
    gpus = detect_gpu_sets(a.gpus)
    runner = ShellRunner(wrap, timeout_s=a.command_timeout, max_timeout_s=600, gpu_sets=gpus)
    print(
        f"runner: wrap={runner.wrap[:4]}..., network={os.environ.get('MESH_RUN_NETWORK', 'default')}, "
        f"gpu_sets={gpus or 'unpinned'}",
        flush=True,
    )

    if a.selftest:
        return asyncio.run(selftest_cmd(runner, pulsed_window_s=a.pulsed_window_s, require_gpu=a.require_gpu))

    asyncio.run(preflight(runner, require_gpu=a.require_gpu))

    pricing = resolve_pricing(a.model, a.price_input, a.price_cached, a.price_output)
    if pricing is None:
        p.error(f"no built-in prices for {a.model}; pass --price-input/--price-cached/--price-output")

    api_key = os.environ.get(a.api_key_name) or read_key(a.api_key_file, a.api_key_name)
    model = ChatEndpoint(
        a.base_url,
        a.model,
        api_key=api_key,
        extra_body={"max_tokens": a.max_completion_tokens},
        timeout_s=a.request_timeout,
        context_window=a.context_window,
    )

    sub_model: ChatEndpoint | None = None
    sub_pricing: Pricing | None = None
    if a.sub_model or a.sub_base_url:
        sub_url = a.sub_base_url or a.base_url
        sub_name = a.sub_model or a.model
        sub_key = read_key(a.sub_api_key_file, a.sub_api_key_name) if a.sub_api_key_file else api_key
        sub_model = ChatEndpoint(
            sub_url,
            sub_name,
            api_key=sub_key,
            extra_body={"max_tokens": a.sub_max_completion_tokens},
            timeout_s=a.request_timeout,
            context_window=a.sub_context_window,
        )
        sub_pricing = resolve_pricing(sub_name, a.sub_price_input, a.sub_price_cached, a.sub_price_output)
        if sub_pricing is None:
            p.error(f"no built-in prices for {sub_name}; pass --sub-price-input/--sub-price-cached/--sub-price-output")

    agents_spec: int | list[str] = (
        int(a.agents) if str(a.agents).isdigit() else [r.strip() for r in str(a.agents).split(",") if r.strip()]
    )
    limits = Limits(
        max_cost_usd=a.max_cost,
        max_steps_per_agent=a.max_steps,
        max_steps_per_subagent=a.sub_max_steps if sub_model is not None else None,
        max_concurrent_calls=a.concurrency,
        timeout_s=a.timeout,
        max_total_tokens=a.max_tokens,
        compact_at_tokens=a.compact_at,
        max_agents=a.max_agents,
        max_subagents=a.max_subagents,
        checkin_after_s=a.checkin_after or None,
        goal_grace_s=a.goal_grace,
    )
    out = a.out or os.path.join(a.run_dir, "tmap8_arc-" + time.strftime("%Y%m%d-%H%M%S"))

    def progress(gen) -> None:
        for r in gen.runs:
            print(
                f"gen {gen.number} {r.label:8} score {r.grade.score}/10 cost ${r.shape['cost_usd']} "
                f"ran {(r.shape['seconds'] or 0) // 60} min, stopped by {r.shape['stop']}, "
                f"{r.shape['delegations']} delegations | {r.grade.reasons}",
                flush=True,
            )

    scratch_env = os.environ.get("MESH_RUN_SCRATCH")
    scratch_root = Path(scratch_env) if scratch_env else None
    about_tool = (
        "DOE INL `tmap8-opt` (MOOSE C++ tritium solver) and Python 3.12 with CUDA GPU `jax` (`float64`), "
        "`diffrax`, `optax`, `equinox`, `scipy`, and `numpy` are available inside this isolated sandbox."
    )
    result = asyncio.run(evolve(
        task_text(),
        model,
        sub_model=sub_model,
        generations=a.generations,
        population=a.population,
        control=a.controls,
        agents=agents_spec,
        out=out,
        limits=limits,
        pricing=pricing,
        sub_pricing=sub_pricing,
        grade=make_grader(pulsed_window_s=a.pulsed_window_s, scratch_root=scratch_root),
        tools=[run_tool(runner, about_tool)],
        deliverable=DELIVERABLE,
        protect=PROTECT_PATTERNS,
        files=starting_files(),
        kind=KIND,
        parallel=a.parallel,
        lessons=starting_lessons(a.lessons),
        on_generation=progress,
    ))
    u = result.review_usage
    print(f"reviewer tokens: {u.prompt} in, {u.completion} out (about ${pricing.cost(u):.4f})", flush=True)
    print(f"summary: {result.out / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
