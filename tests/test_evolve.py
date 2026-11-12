"""meshevolve tests with a scripted model (no endpoint needed)."""

from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from meshswarm import Limits, ScriptedModel, call, say
from meshevolve import Grade, evolve, load_events, shape, timeline, with_lessons
from meshevolve.cli import main
from meshevolve.evolve import parse_grade


def step(messages: list[dict]) -> int:
    return sum(1 for m in messages if m["role"] == "assistant")


def swarm_policy(agent: str, msgs: list[dict]) -> dict:
    """agent-1 asks agent-2 to do the work; agent-2 writes FINAL.md and completes the goal."""
    n = step(msgs)
    if agent == "agent-1":
        return [call("send_message", to="agent-2", content="please take the goal"), call("wait", reason="handed off")][min(n, 1)]
    return [call("claim_task", task_id=1), call("write_file", path="FINAL.md", content="the answer"),
            call("complete_task", task_id=1, result="see FINAL.md"), call("wait", reason="done")][min(n, 3)]


class Scripted:
    """Swarm agents follow `swarm_policy`; the judge and reviewer replies are scripted here."""

    def __init__(self, scores=(4, 7), review="Lesson {n}: split the work early."):
        self.scores = list(scores)
        self.review = review
        self.judged: list[str] = []
        self.reviewed: list[str] = []
        self.goals: list[str] = []
        self.model = ScriptedModel(self.policy)

    def policy(self, agent: str, msgs: list[dict]) -> dict:
        user = msgs[-1]["content"] if msgs[-1]["role"] == "user" else ""
        if agent == "judge":
            self.judged.append(user)
            score = self.scores[(len(self.judged) - 1) % len(self.scores)]
            return say(f'Here is my grade: {{"score": {score}, "reasons": "fine"}}')
        if agent == "reviewer":
            self.reviewed.append(user)
            return say(self.review.format(n=len(self.reviewed)))
        if step(msgs) == 0 and agent == "agent-1":
            self.goals.append(next(m["content"] for m in msgs if m["role"] == "user"))
        return swarm_policy(agent, msgs)


def run_evolve(s: Scripted, **kw):
    out = tempfile.mkdtemp()
    kw.setdefault("limits", Limits(timeout_s=10))
    kw.setdefault("agents", 2)
    return asyncio.run(evolve("write the answer", s.model, out=out, **kw))


class Evolve(unittest.TestCase):
    def test_lessons_flow_into_the_next_generation(self):
        s = Scripted()
        r = run_evolve(s, generations=3, population=2)
        self.assertEqual(len(r.generations), 3)
        self.assertEqual([len(g.runs) for g in r.generations], [2, 2, 2])
        # Generation 1 has no lessons; generation 2 sees the first review, generation 3 the second.
        self.assertEqual(len(s.goals), 6)
        self.assertNotIn("Notes from earlier swarms", s.goals[0])
        self.assertIn("Lesson 1: split the work early.", s.goals[2])
        self.assertIn("Lesson 2: split the work early.", s.goals[4])
        self.assertNotIn("Lesson 1:", s.goals[4])  # lessons are rewritten, not piled up
        self.assertIn("not rules", s.goals[2])
        self.assertEqual(r.lessons, "Lesson 3: split the work early.")
        self.assertFalse(r.generations[0].runs[0].had_lessons)
        self.assertTrue(r.generations[1].runs[0].had_lessons)

    def test_judge_sees_the_task_and_deliverable_but_not_the_lessons(self):
        s = Scripted()
        run_evolve(s, generations=2, population=1, rubric="Must be short.")
        self.assertEqual(len(s.judged), 2)
        for prompt in s.judged:
            self.assertIn("write the answer", prompt)
            self.assertIn("the answer", prompt)
            self.assertIn("Must be short.", prompt)
            self.assertNotIn("Lesson", prompt)

    def test_reviewer_sees_scores_shape_and_timeline_best_first(self):
        s = Scripted(scores=(4, 7))
        r = run_evolve(s, generations=1, population=2)
        prompt = s.reviewed[0]
        self.assertLess(prompt.index("# run-2"), prompt.index("# run-1"))  # 7 before 4
        self.assertIn("Score: 7.0/10", prompt)
        self.assertIn("agent-1 -> agent-2: please take the goal", prompt)
        self.assertIn("agent-2 claimed task #1", prompt)
        self.assertIn("(none: first generation)", prompt)
        self.assertEqual([x.grade.score for x in r.generations[0].runs], [4.0, 7.0])

    def test_shape_and_timeline_describe_the_run(self):
        s = Scripted()
        r = run_evolve(s, generations=1, population=1)
        run = r.generations[0].runs[0]
        self.assertEqual(run.shape["agents"], 2)
        self.assertEqual(run.shape["peer_messages"], 1)
        self.assertEqual(run.shape["talking_pairs"], 1)
        self.assertEqual(run.shape["hub_share"], 1.0)
        self.assertEqual(run.shape["claims"], 1)
        self.assertEqual(run.shape["message_pairs"], {"agent-1 -> agent-2": 1})
        self.assertEqual(run.shape["stop"], "quiescent")
        text = timeline(load_events(run.workspace))
        self.assertIn("agent-2 wrote FINAL.md", text)
        self.assertNotIn("mesh ->", text)
        self.assertEqual(shape(load_events(run.workspace)), run.shape)

    def test_control_run_gets_no_lessons(self):
        s = Scripted()
        r = run_evolve(s, generations=2, population=1, control=True)
        self.assertEqual([x.label for x in r.generations[0].runs], ["run-1"])  # nothing to control for yet
        self.assertEqual([x.label for x in r.generations[1].runs], ["run-1", "control"])
        self.assertIn("Lesson 1", s.goals[1])
        self.assertNotIn("Lesson 1", s.goals[2])
        self.assertFalse(r.generations[1].runs[1].had_lessons)
        self.assertIn("NOT given any lessons", s.reviewed[1])

    def test_custom_grader_replaces_the_judge(self):
        s = Scripted()

        async def grade(result):
            return Grade(10.0 if result.final == "the answer" else 0.0, "exact match")

        r = run_evolve(s, generations=1, population=1, grade=grade)
        self.assertEqual(s.judged, [])
        self.assertEqual(r.generations[0].runs[0].grade, Grade(10.0, "exact match"))

    def test_run_without_final_scores_zero_without_asking_the_judge(self):
        s = Scripted()
        s.model = ScriptedModel(lambda agent, msgs: s.policy(agent, msgs) if agent in ("judge", "reviewer")
                                else call("wait", reason="idle"))
        r = run_evolve(s, generations=1, population=1)
        self.assertEqual(s.judged, [])
        self.assertEqual(r.generations[0].runs[0].grade.score, 0.0)
        self.assertFalse(r.generations[0].runs[0].final_written)

    def test_blank_review_keeps_earlier_lessons_and_long_review_is_capped(self):
        s = Scripted(review="")
        r = run_evolve(s, generations=1, population=1, lessons="Start: talk less.")
        self.assertEqual(r.lessons, "Start: talk less.")
        self.assertIn("Start: talk less.", s.goals[0])

        s = Scripted(review="word " * 500)
        r = run_evolve(s, generations=1, population=1, max_lesson_words=50)
        self.assertEqual(len(r.lessons.split()), 101)  # 2x the asked length, plus the ellipsis

    def test_files_written(self):
        s = Scripted()
        r = run_evolve(s, generations=2, population=2, parallel=True)
        self.assertEqual((r.out / "lessons.md").read_text(), r.lessons)
        self.assertEqual((r.out / "gen-02" / "lessons_in.md").read_text(), "Lesson 1: split the work early.")
        self.assertEqual((r.out / "gen-02" / "lessons_out.md").read_text(), "Lesson 2: split the work early.")
        scores = json.loads((r.out / "gen-01" / "scores.json").read_text())
        self.assertEqual(sorted(x["score"] for x in scores), [4.0, 7.0])
        self.assertTrue(Path(scores[0]["workspace"], ".mesh", "events.jsonl").is_file())
        summary = (r.out / "summary.md").read_text()
        self.assertIn("| 1 | run-1 | no |", summary)
        self.assertIn("| 2 | run-2 | yes |", summary)
        self.assertIn("| 1 | - | 5.5 |", summary)
        self.assertGreater(r.review_usage.prompt, 0)

    def test_parse_grade(self):
        self.assertEqual(parse_grade('{"score": 8, "reasons": "good"}'), Grade(8.0, "good"))
        self.assertEqual(parse_grade('```json\n{"score": 42}\n```').score, 10.0)
        self.assertIsNone(parse_grade("I think it is good").score)
        self.assertIsNone(parse_grade('{"score": "high"}').score)

    def test_with_lessons(self):
        self.assertEqual(with_lessons("task", "  ", 1), "task")
        self.assertTrue(with_lessons("task", "be brief", 2).endswith("be brief"))

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            run_evolve(Scripted(), generations=0)


class FakeEndpoint(BaseHTTPRequestHandler):
    models: list[str] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        system = msgs[0]["content"]
        if system.startswith("You grade"):
            FakeEndpoint.models.append(body["model"])
            reply = say('{"score": 6, "reasons": "ok"}')
        elif system.startswith("You review"):
            FakeEndpoint.models.append(body["model"])
            reply = say("Claim before writing.")
        else:
            reply = swarm_policy(system.split(",")[0].removeprefix("You are "), msgs)
        data = json.dumps({"choices": [{"message": reply}], "usage": {"prompt_tokens": 40, "completion_tokens": 10}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class EvolveCli(unittest.TestCase):
    def test_cli_runs_generations_against_openai_compatible_server(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeEndpoint)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        out_dir = Path(tempfile.mkdtemp()) / "evo"
        os.environ["TEST_EVOLVE_KEY"] = "sk-test"
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main([
                    "write the answer", "--generations", "2", "--population", "1", "--agents", "2", "--out", str(out_dir),
                    "--base-url", f"http://127.0.0.1:{server.server_port}/v1", "--model", "my-model",
                    "--reviewer-model", "big-model", "--api-key-env", "TEST_EVOLVE_KEY", "--timeout", "20",
                ])
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("gen 2 run-1    score 6/10", out.getvalue())
        self.assertEqual((out_dir / "lessons.md").read_text(), "Claim before writing.")
        self.assertEqual((out_dir / "gen-02" / "run-1" / "FINAL.md").read_text(), "the answer")
        self.assertEqual(set(FakeEndpoint.models), {"big-model"})
        self.assertEqual(len(FakeEndpoint.models), 4)  # a judge and a review per generation


if __name__ == "__main__":
    unittest.main()
