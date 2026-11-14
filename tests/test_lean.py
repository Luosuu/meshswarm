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
        self.assertEqual(list(run.shape["progress"]), ["two_packable"])
        self.assertIn("asleep_pct", run.shape)


class Shell(unittest.TestCase):
    def test_run_command_tool_runs_in_the_workspace_and_times_out(self):
        from meshswarm.shell import ShellRunner, run_tool

        def policy(agent, msgs):
            return [call("write_file", path="hello.c", content='#include <stdio.h>\nint main(){puts("hi");return 0;}\n'),
                    call("run_command", command="ls && echo here:$(basename $PWD) && false"),
                    call("run_command", command="sleep 5", timeout_s=0.3), call("wait")][min(step(msgs), 3)]

        model = ScriptedModel(policy)
        ws = Path(tempfile.mkdtemp()) / "space"
        asyncio.run(run_mesh("goal", model, agents=1, workspace=ws, echo=False, limits=Limits(timeout_s=20),
                             tools=[run_tool(ShellRunner())]))
        results = tool_results(model.calls[-1][1])
        self.assertEqual(results[1]["exit_code"], 1)
        self.assertIn("hello.c", results[1]["output"])
        self.assertIn("here:space", results[1]["output"])
        self.assertIsNone(results[2]["exit_code"])
        self.assertIn("timed_out", results[2])
        events = [json.loads(line) for line in (ws / ".mesh" / "events.jsonl").read_text().splitlines()]
        self.assertEqual(len([e for e in events if e["kind"] == "run"]), 2)

    def test_core_sets_are_assigned_per_workspace_and_output_is_clipped(self):
        from meshswarm.shell import ShellRunner, clip

        r = ShellRunner(core_sets=["0-7", "8-15"])
        a, b = Path("/tmp/a"), Path("/tmp/b")
        self.assertEqual([r.cores_for(a), r.cores_for(b), r.cores_for(a)], ["0-7", "8-15", "0-7"])
        self.assertEqual(r.threads_for(a), 8)
        self.assertEqual(ShellRunner(core_sets=["0,2,4-5"]).threads_for(a), 4)
        text = clip("A" * 3000 + "B" * 3000, 600)
        self.assertTrue(text.startswith("A") and text.endswith("B") and "cut" in text and len(text) < 700)

    def test_evolve_seeds_starting_files(self):
        def policy(agent, msgs):
            if agent == "reviewer":
                return say("ok")
            return [call("read_file", path="src/a.txt"), call("wait")][min(step(msgs), 1)]

        model = ScriptedModel(policy)

        async def grade(result):
            from meshevolve import Grade
            return Grade(1.0, "x")

        asyncio.run(evolve("t", model, generations=1, population=1, agents=1, out=tempfile.mkdtemp(),
                           limits=Limits(timeout_s=10, nudges=0), grade=grade, files={"src/a.txt": "seeded text"}))
        self.assertIn("seeded text", str(next(ms for a, ms in model.calls if a == "agent-1" and len(ms) > 3)))


class HpcKernels(unittest.TestCase):
    def test_task_files_scoring_and_tamper_check(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "hpc"))
        import kernels

        for name in ("amr", "pic"):
            files = kernels.starting_files(name)
            self.assertIn("build.sh", files)
            self.assertIn(f"-o {name}", files["build.sh"])
            ws = Path(tempfile.mkdtemp())
            for rel, text in files.items():
                (ws / rel).write_text(text)
            self.assertTrue(kernels.verification_intact(name, ws))
            src = ws / f"{name}.c"
            src.write_text(src.read_text().replace('printf("Solution validates\\n");', 'printf("fine\\n");'))
            self.assertFalse(kernels.verification_intact(name, ws))
            self.assertIn("run_command", kernels.task_text(name, 8))
        self.assertEqual([kernels.score_of(x) for x in (1.0, 2.0, 9.48, 6.3, 100.0)], [0.0, 1.0, 4.0, 3.0, 10.0])
        import run as hpc_run

        # 16 logical processors: 8 cores with two threads each, in two memory regions of 4 cores.
        siblings = {c: [c, c + 8] for c in range(8)} | {c + 8: [c, c + 8] for c in range(8)}
        nodes = {0: [0, 1, 2, 3, 8, 9, 10, 11], 1: [4, 5, 6, 7, 12, 13, 14, 15]}
        self.assertEqual(hpc_run.plan_core_sets(list(range(16)), siblings, nodes, 4, 2), [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(hpc_run.plan_core_sets(list(range(16)), siblings, nodes, 8, 1), [[0, 1, 2, 3, 4, 5, 6, 7]])
        with self.assertRaises(SystemExit):
            hpc_run.plan_core_sets(list(range(16)), siblings, nodes, 8, 2)
        self.assertEqual(kernels.TIMED.findall("MESHTIME 9.9\nSolution validates\nMESHTIME 1.234\n")[-1], "1.234")
        ws = Path(tempfile.mkdtemp())
        (ws / "build.sh").write_text("gcc -O3 -march=native -fopenmp pic.c -o pic\n")
        self.assertIsNone(kernels.unsafe_math_flag(ws))
        (ws / "build.sh").write_text("gcc -O3 -ffast-math pic.c -o pic\n")
        self.assertIn("-ffast-math in build.sh", kernels.unsafe_math_flag(ws))
        (ws / "build.sh").write_text("gcc -O3 pic.c -o pic\n")
        (ws / "pic.c").write_text('#pragma GCC optimize ("fast-math")\nint main(){return 0;}\n')
        self.assertIn("pic.c", kernels.unsafe_math_flag(ws))

    def test_lulesh_task_and_answer_check(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "hpc"))
        import lulesh

        files = lulesh.starting_files()
        self.assertIn("-fopenmp -DUSE_MPI=0", files["build.sh"])
        self.assertIn("lulesh.cc", files)
        self.assertIn("same iteration count", lulesh.task_text(8))
        out = "Run completed:\n   Iteration count     =  932\n   Final Origin Energy =  2.025075e+05\n"
        self.assertEqual(lulesh.answer(out), (932, 202507.5))
        self.assertIsNone(lulesh.answer("Segmentation fault"))
        want = (932, 202507.5)
        self.assertTrue(lulesh.same_answer((932, 202507.6), want))
        self.assertFalse(lulesh.same_answer((932, 202509.0), want))
        self.assertFalse(lulesh.same_answer((931, 202507.5), want))
        self.assertFalse(lulesh.same_answer((932, float("nan")), want))
        self.assertFalse(lulesh.same_answer(None, want))
        self.assertEqual([lulesh.score_of(x) for x in (1.0, 1.1, 1.6, 10.0)], [0.0, 1.0, 4.0, 10.0])


if __name__ == "__main__":
    unittest.main()
