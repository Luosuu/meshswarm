"""Extra tools and Lean checking, with a fake `lean` command (no Lean install needed)."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "circle2"))

import task  # noqa: E402
from meshswarm import ExtraTool, Limits, ScriptedModel, call, run_mesh, say  # noqa: E402
from meshswarm.lean import LeanChecker, lean_tool  # noqa: E402
from meshevolve import evolve  # noqa: E402

FAKE = [sys.executable, str(Path(__file__).with_name("fake_lean.py"))]
PROOF = "by\n  norm_num"


def step(messages: list[dict]) -> int:
    return sum(1 for m in messages if m["role"] == "assistant")


def tool_results(messages: list[dict]) -> list[dict]:
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def checker(**kw) -> LeanChecker:
    return LeanChecker(FAKE, header=task.HEADER, scratch=Path(tempfile.mkdtemp()), **kw)


def solution(*proved: str) -> str:
    return "\n".join(
        part.replace("by\n  sorry", PROOF) if any(f"theorem {n} " in part for n in proved) else part
        for part in task.SKELETON.split("\n\n"))


class ExtraTools(unittest.TestCase):
    def run_mesh(self, policy, tools, **kw):
        model = ScriptedModel(policy)
        r = asyncio.run(run_mesh("goal", model, workspace=tempfile.mkdtemp(), echo=False, tools=tools,
                                 limits=Limits(timeout_s=10), **kw))
        return r, model

    def test_extra_tool_is_offered_and_runs_for_agents_and_sub_agents(self):
        seen = []

        async def double(agent, n: int):
            seen.append(agent.name)
            return {"twice": 2 * n}

        tool = ExtraTool("double", "Double a number.", double, {"n": {"type": "integer"}}, ["n"])
        offered = {}

        def policy(agent, msgs):
            n = step(msgs)
            if ":" in agent:
                return [call("double", n=5), call("report", result="done")][min(n, 1)]
            return [call("double", n=4), call("delegate", task="double 5", wait=True), call("wait")][min(n, 2)]

        model = ScriptedModel(policy)
        original = model.complete

        async def complete(messages, tools, *, agent):
            offered[agent] = [t["function"]["name"] for t in tools or []]
            return await original(messages, tools, agent=agent)

        model.complete = complete
        asyncio.run(run_mesh("goal", model, agents=1, workspace=tempfile.mkdtemp(), echo=False, tools=[tool],
                             limits=Limits(timeout_s=10)))
        self.assertEqual(sorted(seen), ["agent-1", "agent-1:sub-1"])
        self.assertIn("double", offered["agent-1"])
        self.assertIn("double", offered["agent-1:sub-1"])
        self.assertIn({"twice": 8}, tool_results(next(m for a, m in reversed(model.calls) if a == "agent-1")))

    def test_no_extra_tools_by_default_and_names_must_not_clash(self):
        r, model = self.run_mesh(lambda a, m: [call("lean_check", path="x.lean"), call("wait")][min(step(m), 1)], None, agents=1)
        self.assertIn("unknown tool", tool_results(model.calls[-1][1])[0]["error"])

        async def noop(agent):
            return {}

        with self.assertRaises(ValueError):
            self.run_mesh(lambda a, m: say(""), [ExtraTool("read_file", "x", noop)])
        with self.assertRaises(ValueError):
            self.run_mesh(lambda a, m: say(""), [ExtraTool("x", "x", noop), ExtraTool("x", "x", noop)])


class Lean(unittest.TestCase):
    def test_checker_reports_authors_line_numbers(self):
        r = asyncio.run(checker().check("theorem a : True := by\n  ERR\n", name="work/a.lean"))
        self.assertFalse(r.ok)
        self.assertIn("work/a.lean:2:2: error: unsolved goals", r.shown)
        self.assertNotIn("check_", r.shown)
        ok = asyncio.run(checker().check("theorem a : True := by\n  trivial\n"))
        self.assertTrue(ok.ok and not ok.uses_sorry)
        sorry = asyncio.run(checker().check("theorem a : True := by\n  sorry\n"))
        self.assertTrue(sorry.ok and sorry.uses_sorry)

    def test_checker_times_out(self):
        r = asyncio.run(checker(timeout_s=0.3).check("-- SLOW\n"))
        self.assertTrue(r.timed_out)
        self.assertFalse(r.ok)
        self.assertIn("too long", r.shown)

    def test_lean_check_tool_in_a_run(self):
        def policy(agent, msgs):
            return [call("write_file", path="a.lean", content="theorem a : True := by\n  ERR\n"), call("lean_check", path="a.lean"),
                    call("write_file", path="b.lean", content="#eval 1\n"), call("lean_check", path="b.lean"),
                    call("lean_check", path="missing.lean"), call("wait")][min(step(msgs), 5)]

        model = ScriptedModel(policy)
        ws = tempfile.mkdtemp()
        asyncio.run(run_mesh("goal", model, agents=1, workspace=ws, echo=False, limits=Limits(timeout_s=10),
                             tools=[lean_tool(checker(), "Header is in front.")]))
        results = tool_results(model.calls[-1][1])
        self.assertEqual(results[1]["compiles"], False)
        self.assertEqual(results[1]["complete"], False)
        self.assertIn("a.lean:2:2: error", results[1]["messages"])
        self.assertIn("not allowed", results[3]["error"])
        self.assertIn("not a file", results[4]["error"])
        events = [json.loads(line) for line in (Path(ws) / ".mesh" / "events.jsonl").read_text().splitlines()]
        self.assertEqual([e["path"] for e in events if e["kind"] == "lean_check"], ["a.lean"])


class CircleTask(unittest.TestCase):
    def grade(self, body: str | None):
        ws = Path(tempfile.mkdtemp())
        if body is not None:
            (ws / task.DELIVERABLE).write_text(body)

        class R:
            workspace = ws

        return asyncio.run(task.make_grader(checker())(R()))

    def test_partial_credit_per_theorem(self):
        self.assertEqual(self.grade(None).score, 0.0)
        self.assertEqual(self.grade(task.SKELETON).score, 0.0)
        g = self.grade(solution("two_packable", "c_n_two"))
        self.assertEqual(g.score, 5.0)
        self.assertIn("proved: two_packable, c_n_two", g.reasons)
        self.assertEqual(self.grade(solution("two_packable", "two_upper", "r_two", "c_n_two")).score, 10.0)

    def test_missing_theorem_axioms_and_forbidden_commands_do_not_score(self):
        missing = solution("two_packable", "two_upper", "r_two", "c_n_two").replace("theorem c_n_two", "theorem c_n_twoo")
        self.assertEqual(self.grade(missing).score, 7.5)
        cheat = solution("two_packable").replace(PROOF, "by\n  exact cheat.elim")
        self.assertEqual(self.grade(cheat).score, 0.0)
        self.assertEqual(self.grade("#print axioms Grader.check_1\n" + solution("two_packable")).score, 0.0)

    def test_proved_parses_lean_output(self):
        raw = ("x.lean:40:0: info: 'Grader.check_1' depends on axioms: [propext,\n Classical.choice, Quot.sound]\n"
               "x.lean:42:0: info: 'Grader.check_2' depends on axioms: [propext, sorryAx]\n"
               "'CirclePackingConstants.Grader.check_3' does not depend on any axioms\n")
        self.assertEqual(task.proved(raw), {"two_packable": True, "two_upper": False, "r_two": True, "c_n_two": False})

    def test_evolve_passes_tools_and_grades_the_lean_deliverable(self):
        def policy(agent, msgs):
            if agent == "reviewer":
                return say("Check small pieces.")
            if agent != "agent-1":
                return call("wait")
            return [call("write_file", path="FINAL.lean", content=solution("two_packable")),
                    call("lean_check", path="FINAL.lean"), call("claim_task", task_id=1),
                    call("complete_task", task_id=1, result="FINAL.lean"), call("wait")][min(step(msgs), 4)]

        c = checker()
        r = asyncio.run(evolve(task.TASK, ScriptedModel(policy), generations=1, population=1, agents=2,
                               out=tempfile.mkdtemp(), limits=Limits(timeout_s=10), grade=task.make_grader(c),
                               tools=[lean_tool(c, task.ABOUT_HEADER)], deliverable=task.DELIVERABLE))
        run = r.generations[0].runs[0]
        self.assertEqual(run.grade.score, 2.5)
        self.assertTrue(run.final_written)
        self.assertEqual(run.shape["lean_checks"], 1)
        self.assertEqual(run.shape["lean_checks_ok"], 0)  # compiles, but three theorems are still sorry


if __name__ == "__main__":
    unittest.main()
