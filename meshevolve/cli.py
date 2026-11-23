"""Command line: python -m meshevolve "your task" --generations 3 --population 2"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from meshswarm import ChatEndpoint, Limits
from meshswarm.cli import _agents_arg, _env, _pricing, read_key_file

from .evolve import Generation, evolve


def _text(value: str | None) -> str:
    """A literal string, or @path to read it from a file."""
    if not value:
        return ""
    return Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="meshevolve", description="Run generations of swarms on one task; an outside "
                                "review after each generation writes lessons for the next.")
    p.add_argument("task", help="the task, or @path/to/file.md")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--population", type=int, default=2, help="swarms per generation")
    p.add_argument("--control", action="store_true", help="add one swarm per generation that gets no lessons")
    p.add_argument("--agents", default="3", help="starting agents per swarm: a number, or comma-separated roles")
    p.add_argument("--out", help="output folder (default: runs/evolve-<timestamp>)")
    p.add_argument("--lessons", help="starting lessons, or @file (e.g. lessons.md from an earlier run)")
    p.add_argument("--kind", help='the type of problem the lessons should serve, e.g. "literature surveys"; '
                   "lessons are kept general so they carry over to other problems of that kind")
    p.add_argument("--rubric", help="what the judge should look for, or @file")
    p.add_argument("--max-lesson-words", type=int, default=400)
    p.add_argument("--parallel", action="store_true", help="run a generation's swarms at the same time")
    p.add_argument("--base-url", default=_env("MESH_BASE_URL", "OPENAI_BASE_URL"))
    p.add_argument("--model", default=_env("MESH_MODEL"))
    p.add_argument("--sub-base-url", default=_env("MESH_SUB_BASE_URL"), help="optional separate endpoint for delegate() sub-agents")
    p.add_argument("--sub-model", default=_env("MESH_SUB_MODEL"), help="optional separate model for delegate() sub-agents")
    p.add_argument("--reviewer-model", help="model for the judge and the review (default: --model)")
    p.add_argument("--reviewer-base-url", help="endpoint for the reviewer model (default: --base-url)")
    p.add_argument("--api-key-env", default="MESH_API_KEY")
    p.add_argument("--api-key-file", default=_env("MESH_API_KEY_FILE"))
    p.add_argument("--max-completion-tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--extra-body", help="JSON merged into every request")
    p.add_argument("--ca-file")
    d = Limits()
    p.add_argument("--max-agents", type=int, default=d.max_agents)
    p.add_argument("--max-steps", type=int, default=d.max_steps_per_agent)
    p.add_argument("--sub-max-steps", type=int, default=d.max_steps_per_subagent)
    p.add_argument("--max-tokens", type=int, default=d.max_total_tokens, help="token budget per swarm")
    p.add_argument("--max-cost", type=float, help="dollar budget per swarm (needs known or given prices)")
    p.add_argument("--price-input", type=float)
    p.add_argument("--price-cached", type=float)
    p.add_argument("--price-output", type=float)
    p.add_argument("--sub-price-input", type=float)
    p.add_argument("--sub-price-cached", type=float)
    p.add_argument("--sub-price-output", type=float)
    p.add_argument("--concurrency", type=int, default=d.max_concurrent_calls)
    p.add_argument("--timeout", type=float, default=d.timeout_s, help="wall-clock limit per swarm, seconds")
    p.add_argument("--verbose", action="store_true", help="print every swarm event")
    a = p.parse_args(argv)

    if not a.base_url or not a.model:
        p.error("set --base-url and --model (or MESH_BASE_URL and MESH_MODEL)")
    try:
        key = read_key_file(a.api_key_file) if a.api_key_file else os.environ.get(a.api_key_env)
        extra = json.loads(a.extra_body) if a.extra_body else {}
        if a.max_completion_tokens:
            extra["max_tokens"] = a.max_completion_tokens

        def endpoint(base_url: str, model: str) -> ChatEndpoint:
            return ChatEndpoint(base_url, model, api_key=key, temperature=a.temperature, extra_body=extra, ca_file=a.ca_file)

        model = endpoint(a.base_url, a.model)
        sub_model = None
        if a.sub_model or a.sub_base_url:
            sub_model = endpoint(a.sub_base_url or a.base_url, a.sub_model or a.model)
        reviewer = None
        if a.reviewer_model or a.reviewer_base_url:
            reviewer = endpoint(a.reviewer_base_url or a.base_url, a.reviewer_model or a.model)
        pricing = _pricing(a.model, a.price_input, a.price_cached, a.price_output)
        sub_pricing = _pricing(sub_model.model, a.sub_price_input, a.sub_price_cached, a.sub_price_output) if sub_model else None
        if a.max_cost is not None and (pricing is None or (sub_model is not None and sub_pricing is None)):
            p.error("--max-cost needs prices for every active model")
        task, lessons, rubric = _text(a.task), _text(a.lessons), _text(a.rubric)
    except (ValueError, OSError) as e:
        p.error(str(e))

    limits = Limits(max_agents=a.max_agents, max_steps_per_agent=a.max_steps, max_steps_per_subagent=a.sub_max_steps,
                    max_total_tokens=a.max_tokens, max_cost_usd=a.max_cost, max_concurrent_calls=a.concurrency, timeout_s=a.timeout)
    out = a.out or os.path.join("runs", "evolve-" + time.strftime("%Y%m%d-%H%M%S"))

    def progress(gen: Generation) -> None:
        for r in gen.runs:
            cost = "?" if r.shape["cost_usd"] is None else f"${r.shape['cost_usd']:.4f}"
            score = "?" if r.grade.score is None else f"{r.grade.score:g}"
            print(f"gen {gen.number} {r.label:8} score {score}/10  cost {cost}  "
                  f"{r.shape['peer_messages']} msgs, {r.shape['delegations']} delegations  {r.grade.reasons}")
        print(f"gen {gen.number} lessons: {len(gen.lessons_out.split())} words\n")

    try:
        result = asyncio.run(evolve(
            task, model, reviewer=reviewer, sub_model=sub_model, generations=a.generations, population=a.population, control=a.control,
            agents=_agents_arg(a.agents), out=out, limits=limits, pricing=pricing, sub_pricing=sub_pricing, rubric=rubric, lessons=lessons, **({"kind": a.kind} if a.kind else {}),
            max_lesson_words=a.max_lesson_words, parallel=a.parallel, echo=a.verbose, on_generation=progress))
    except KeyboardInterrupt:
        print("\ninterrupted (finished generations are kept)", file=sys.stderr)
        return 130
    u = result.review_usage
    print(f"judge and reviewer tokens: {u.prompt} in, {u.completion} out")
    print(f"summary: {result.out / 'summary.md'}")
    print(f"lessons: {result.out / 'lessons.md'}")
    return 0
