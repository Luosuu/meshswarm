"""Offline unit tests for the Phase 1 Modelica-to-JAX benchmark tasks & grader."""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MJAX = ROOT / "examples" / "modelica_jax"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MJAX))

import grader  # noqa: E402
import tasks  # noqa: E402


class _FakeResult:
    def __init__(self, workspace: Path):
        self.workspace = workspace


class TestModelicaJaxTasks(unittest.TestCase):
    def test_three_tiers_well_formed(self) -> None:
        self.assertEqual(
            set(tasks.TASKS.keys()),
            {"tier1_blanket_dae", "tier2_transform_hx", "tier3_hdt_column"},
        )
        for name, t in tasks.TASKS.items():
            files = tasks.starting_files(name)
            self.assertIn(t.model_file, files)
            self.assertIn(tasks.DELIVERABLE, files)
            self.assertEqual(len(t.y0_default), t.n_states)
            self.assertEqual(len(t.p_default), len(t.param_names))
            self.assertEqual(len(t.test_cases), 8)
            for c in t.test_cases:
                self.assertEqual(len(c), len(t.param_names))
            prompt = tasks.task_text(name)
            self.assertIn(t.model_file, prompt)
            self.assertIn("forward_ensemble", prompt)
            self.assertIn("differentiable_closure", prompt)

    def test_score_ladder(self) -> None:
        self.assertEqual(grader.score_of(45.0), 0.0)
        self.assertEqual(grader.score_of(15.0), 2.0)
        self.assertEqual(grader.score_of(1.0), 6.0)
        self.assertEqual(grader.score_of(0.05), 10.0)

    def test_grader_blocks_forbidden_constructs_and_missing_file(self) -> None:
        ws = Path(tempfile.mkdtemp(prefix="test-mjax-"))
        try:
            g_fn = grader.make_grader("tier1_blanket_dae", gold_override=[[0.0] * 6 for _ in range(8)])
            g0 = asyncio.run(g_fn(_FakeResult(ws)))
            self.assertEqual(g0.score, 0.0)
            self.assertIn("no solver_jax.py", g0.reasons)

            for snippet in [
                "import subprocess\n",
                "x = jax.lax.stop_gradient(p)\n",
                'open(".mesh/baseline/case_0.csv")\n',
                "jax.config.update('jax_enable_x64', False)\n",
            ]:
                (ws / tasks.DELIVERABLE).write_text(snippet, encoding="utf-8")
                g = asyncio.run(g_fn(_FakeResult(ws)))
                self.assertEqual(g.score, 0.0)
                self.assertIn("forbidden construct", g.reasons)
        finally:
            shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
