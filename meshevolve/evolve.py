"""Generations of swarms on one task, with an outside review in between.

Each generation runs `population` swarms with meshswarm, unchanged. A judge
scores each deliverable, then a reviewer reads how every swarm organized itself
and rewrites a short set of lessons. The next generation receives those lessons
as part of its goal text, as notes it may follow, adapt or ignore.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from meshswarm import Limits, RunResult, run_mesh
from meshswarm.model import Model
from meshswarm.pricing import Pricing, Usage
from meshswarm.tools import ExtraTool

from .digest import first_seen, load_events, shape, timeline


@dataclass
class Grade:
    score: float | None  # 0-10, None when the judge's reply could not be read
    reasons: str = ""


Grader = Callable[[RunResult], "Grade | Awaitable[Grade]"]


@dataclass
class RunRecord:
    generation: int
    label: str  # "run-1", "run-2", ... or "control"
    had_lessons: bool
    workspace: Path
    grade: Grade
    shape: dict[str, Any]
    final_written: bool


@dataclass
class Generation:
    number: int
    lessons_in: str
    lessons_out: str
    runs: list[RunRecord]


@dataclass
class EvolveResult:
    generations: list[Generation]
    lessons: str
    out: Path
    review_usage: Usage = field(default_factory=Usage)  # judge and reviewer calls, not the swarms


JUDGE_SYSTEM = (
    "You grade the deliverable a team produced for a task. Judge only the deliverable against the task"
    "{rubric_hint}. You are not told how the team worked and must not guess. "
    'Reply with one JSON object and nothing else: {{"score": <number from 0 to 10>, "reasons": "<one or two sentences>"}}'
)

REVIEW_SYSTEM = """You review how swarms of LLM agents organized themselves on a task, and you keep a short set \
of lessons that the next generation of swarms will read before starting.

Each swarm is leaderless: agents message each other, share a task board and files, can create new peers \
and can delegate to private sub-agents. Nobody told them how to organize.

Write about HOW the agents worked together (who talked to whom, how work was split, claimed, checked and \
assembled, what was wasted), never about the subject matter of the task.

The lessons are for {kind}, not for this one task. The next swarm may get a different problem of that \
kind, so every lesson must still make sense there.

Rules:
- Keep the lessons general. Do not mention this task's particular objects, names, numbers, statements, \
file contents or solution steps, and do not give away any part of its answer. Where a run suggests \
something task-specific, state the general habit behind it or leave it out.
- Ground every lesson in what the runs below show. Runs are listed best first; weigh that, and say when \
the evidence is one run only.
- Keep what earlier lessons still hold, drop or correct what these runs contradict.
- Do not push the swarms toward familiar human structures for their own sake. If a swarm did something \
unusual and it worked, record it plainly.
- End with a short "Untested" list of one to three organizational ideas no run has tried yet, marked as \
experiments, so later generations keep exploring.
- Address the agents directly. Stay under {max_words} words. Reply with the lessons only, in Markdown."""

LESSONS_HEADER = (
    "\n\n---\nNotes from earlier swarms that worked on {kind} (generation {n}). They are about how to work "
    "together, not about this particular goal, and describe what happened to work or fail before. They are not rules: follow, adapt or ignore them, and try something different "
    "if you see a better way.\n\n"
)


DEFAULT_KIND = "tasks of the same kind as this one"


def with_lessons(task: str, lessons: str, generation: int, kind: str = DEFAULT_KIND) -> str:
    return task if not lessons.strip() else task + LESSONS_HEADER.format(n=generation, kind=kind) + lessons.strip()


def parse_grade(text: str) -> Grade:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
        score = max(0.0, min(10.0, float(data["score"])))
        return Grade(score, str(data.get("reasons", "")))
    except (TypeError, ValueError, KeyError):
        return Grade(None, f"judge reply could not be read: {text[:200]!r}")


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(sum(known) / len(known), 2) if known else None


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


class _Reviewer:
    """Judge and reviewer calls, with their token usage kept apart from the swarms'."""

    def __init__(self, model: Model, rubric: str, max_words: int, final_chars: int, kind: str = DEFAULT_KIND):
        self.model = model
        self.kind = kind
        self.rubric = rubric
        self.max_words = max_words
        self.final_chars = final_chars
        self.usage = Usage()

    async def _ask(self, system: str, user: str, agent: str) -> str:
        msg, usage = await self.model.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], None, agent=agent)
        self.usage = self.usage + usage
        return (msg.get("content") or "").strip()

    async def judge(self, task: str, result: RunResult) -> Grade:
        if not result.final or not result.final.strip():
            return Grade(0.0, "no FINAL.md was written")
        final = result.final
        if len(final) > self.final_chars:
            final = final[: self.final_chars] + f"\n\n[cut: {len(result.final) - self.final_chars} more characters]"
        system = JUDGE_SYSTEM.format(rubric_hint=" and the rubric" if self.rubric else "")
        user = f"# Task\n\n{task}\n\n" + (f"# Rubric\n\n{self.rubric}\n\n" if self.rubric else "") + f"# Deliverable\n\n{final}"
        return parse_grade(await self._ask(system, user, "judge"))

    async def review(self, task: str, lessons: str, runs: list[RunRecord], digests: dict[str, str]) -> str:
        ranked = sorted(runs, key=lambda r: (r.grade.score is None, -(r.grade.score or 0)))
        parts = [f"# Task\n\n{task}", f"# Lessons the swarms were given\n\n{lessons.strip() or '(none: first generation)'}"]
        for r in ranked:
            s = {k: v for k, v in r.shape.items() if k not in ("message_pairs", "progress")}
            given = "given the lessons" if r.had_lessons else "NOT given any lessons"
            parts.append(
                f"# {r.label} ({given})\n\n"
                f"Score: {_fmt(r.grade.score)}/10. Judge: {r.grade.reasons}\n"
                f"Numbers: {json.dumps(s)}\n"
                f"Messages between agents: {json.dumps(r.shape['message_pairs']) if r.shape['message_pairs'] else 'none'}\n\n"
                f"What happened:\n{digests[r.label] or '(no coordination events)'}"
            )
        reply = await self._ask(REVIEW_SYSTEM.format(max_words=self.max_words, kind=self.kind), "\n\n".join(parts), "reviewer")
        if not reply:
            return lessons  # a blank review must not wipe what earlier generations learned
        words = reply.split(" ")
        hard_cap = self.max_words * 2  # the prompt asks for max_words; this only stops a runaway reply
        return reply if len(words) <= hard_cap else " ".join(words[:hard_cap]) + " …"


async def evolve(
    task: str,
    model: Model,
    *,
    reviewer: Model | None = None,
    sub_model: Model | None = None,
    generations: int = 3,
    population: int = 2,
    control: bool | int = False,
    agents: int | list[str] = 3,
    out: str | Path = "evolve-run",
    limits: Limits | None = None,
    pricing: Pricing | None = None,
    sub_pricing: Pricing | None = None,
    grade: Grader | None = None,
    rubric: str = "",
    lessons: str = "",
    kind: str = DEFAULT_KIND,
    max_lesson_words: int = 400,
    parallel: bool = False,
    echo: bool = False,
    tools: list[ExtraTool] | None = None,
    deliverable: str = "FINAL.md",
    files: dict[str, str] | None = None,
    protect: list[str] | None = None,
    on_generation: Callable[[Generation], None] | None = None,
) -> EvolveResult:
    """Run `generations` rounds of `population` swarms on `task`, reviewing after each round.

    reviewer: model for the judge and the review (default: the swarm's model).
    control:  swarms per generation that get no lessons (True = 1), to tell learning from noise.
    grade:    your own scorer `(RunResult) -> Grade` (may be async), replacing the LLM judge.
    lessons:  starting lessons, e.g. from an earlier evolve run.
    kind:     the type of problem the lessons should serve, e.g. "formal proofs in Lean 4". Lessons are
              written to carry over to other problems of that kind, not to fit this one task.
    parallel: run a generation's swarms at the same time (each has its own concurrency limit).
    tools:    extra tools for every agent, passed to run_mesh (e.g. meshswarm.lean.lean_tool).
    deliverable: the workspace file that counts as the result (the judge reads FINAL.md).
    files:    starting files written into every swarm's workspace, as {relative path: text}.
    protect:  files (glob patterns) that can no longer be changed once a swarm completes its goal;
              by default the deliverable.
    """
    if generations < 1 or population < 1:
        raise ValueError("generations and population must be at least 1")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rev = _Reviewer(reviewer or model, rubric, max_lesson_words, final_chars=24_000, kind=kind)
    done: list[Generation] = []

    for g in range(1, generations + 1):
        gen_dir = out / f"gen-{g:02d}"
        gen_dir.mkdir(exist_ok=True)
        (gen_dir / "lessons_in.md").write_text(lessons, encoding="utf-8")
        plan = [(f"run-{i + 1}", True) for i in range(population)]
        if control and lessons.strip():
            n = int(control)
            plan += [("control" if n == 1 else f"control-{i + 1}", False) for i in range(n)]

        async def one(label: str, use_lessons: bool) -> tuple[RunRecord, str]:
            goal = with_lessons(task, lessons, g - 1, kind) if use_lessons else task
            for rel, text in (files or {}).items():
                target = gen_dir / label / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            result = await run_mesh(goal, model, agents=agents, workspace=gen_dir / label, echo=echo, pricing=pricing,
                                    sub_model=sub_model, sub_pricing=sub_pricing, tools=tools,
                                    protect=protect if protect is not None else [deliverable],
                                    limits=dataclasses.replace(limits) if limits else None)
            if grade is not None:
                got = grade(result)
                verdict = await got if inspect.isawaitable(got) else got
            else:
                verdict = await rev.judge(task, result)
            events = load_events(result.workspace)
            numbers = shape(events)
            numbers["progress"] = first_seen(events, deliverable)
            record = RunRecord(g, label, use_lessons and bool(lessons.strip()), result.workspace, verdict,
                               numbers, (result.workspace / deliverable).is_file())
            return record, timeline(events)

        if parallel:
            pairs = list(await asyncio.gather(*(one(label, use) for label, use in plan)))
        else:
            pairs = [await one(label, use) for label, use in plan]
        runs = [record for record, _ in pairs]
        new_lessons = await rev.review(task, lessons, runs, {record.label: text for record, text in pairs})

        generation = Generation(g, lessons, new_lessons, runs)
        done.append(generation)
        (gen_dir / "lessons_out.md").write_text(new_lessons, encoding="utf-8")
        (gen_dir / "scores.json").write_text(json.dumps([_row(r) for r in runs], indent=2), encoding="utf-8")
        (out / "lessons.md").write_text(new_lessons, encoding="utf-8")
        (out / "summary.md").write_text(summary(done), encoding="utf-8")
        if on_generation:
            on_generation(generation)
        lessons = new_lessons

    return EvolveResult(done, lessons, out, rev.usage)


def _row(r: RunRecord) -> dict[str, Any]:
    return {"generation": r.generation, "run": r.label, "had_lessons": r.had_lessons, "score": r.grade.score,
            "reasons": r.grade.reasons, "final_written": r.final_written, "workspace": str(r.workspace), **r.shape}


SUMMARY_COLUMNS = ["score", "cost_usd", "seconds", "agents", "spawned_peers", "subagents", "peer_messages",
                   "talking_pairs", "hub_share", "tasks_posted", "handovers", "delegations", "stop"]


def summary(generations: list[Generation]) -> str:
    """One table over all runs, then mean score per generation with and without lessons."""
    head = ["gen", "run", "lessons"] + SUMMARY_COLUMNS
    lines = ["# Evolution summary", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for gen in generations:
        for r in gen.runs:
            row = _row(r)
            cells = [str(gen.number), r.label, "yes" if r.had_lessons else "no"] + [_fmt(row[c]) for c in SUMMARY_COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| gen | mean score with lessons | mean score without | mean cost |", "|---|---|---|---|"]
    for gen in generations:
        with_l = _mean([r.grade.score for r in gen.runs if r.had_lessons])
        without = _mean([r.grade.score for r in gen.runs if not r.had_lessons])
        cost = _mean([r.shape["cost_usd"] for r in gen.runs])
        lines.append(f"| {gen.number} | {_fmt(with_l)} | {_fmt(without)} | {_fmt(cost)} |")
    lines += ["", "`hub_share` is the share of peer messages involving the busiest agent (1.0 = one hub).",
              "Each generation's lessons are in `gen-NN/lessons_out.md`; the latest are in `lessons.md`.", ""]
    return "\n".join(lines)
