"""Offline unit tests for Modelica-to-JAX standalone files, GPU runner, and grader."""

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
from meshswarm.shell import ShellRunner  # noqa: E402
import tasks  # noqa: E402


class _FakeResult:
    def __init__(self, workspace: Path):
        self.workspace = workspace


class TestModelicaJaxTasks(unittest.TestCase):
    def test_standalone_files_and_three_tiers_well_formed(self) -> None:
        self.assertTrue((MJAX / "eval_worker.py").is_file())
        self.assertTrue((MJAX / "skeleton_template.py").is_file())
        self.assertTrue((MJAX / "models" / "BlanketTritiumLoop.mo").is_file())
        self.assertTrue((MJAX / "references" / "tier3_generic_baseline.py").is_file())

        self.assertEqual(
            set(tasks.TASKS.keys()),
            {"tier1_blanket_dae", "tier2_transform_hx", "tier3_hdt_column"},
        )
        for name, t in tasks.TASKS.items():
            self.assertTrue(t.model_path.is_file(), f"missing {t.model_path}")
            self.assertTrue(tasks.reference_path(name).is_file(), f"missing {tasks.reference_path(name)}")
            files = tasks.starting_files(name)
            self.assertIn(t.model_file, files)
            self.assertIn(tasks.DELIVERABLE, files)
            self.assertNotIn("__MODEL_NAME__", files[tasks.DELIVERABLE])
            self.assertEqual(len(t.y0_default), t.n_states)
            self.assertEqual(len(t.p_default), len(t.param_names))
            self.assertEqual(len(t.test_cases), 8)
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

            # Comments and docstrings mentioning omc or subprocess must NOT be falsely flagged
            benign = '"""Verified offline against omc DASSL without subprocess."""\n# no omc or stop_gradient used\nx = 1\n'
            self.assertIsNone(grader.forbidden_construct(benign))
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def test_shell_runner_gpu_pinning_and_shared_gpu_lock(self) -> None:
        async def go() -> None:
            ws1 = Path(tempfile.mkdtemp(prefix="gpu-ws1-"))
            ws2 = Path(tempfile.mkdtemp(prefix="gpu-ws2-"))
            ws3 = Path(tempfile.mkdtemp(prefix="gpu-ws3-"))
            try:
                runner = ShellRunner(gpu_sets=["0", "1"])
                self.assertEqual(runner.gpus_for(ws1), "0")
                self.assertEqual(runner.gpus_for(ws2), "1")
                self.assertEqual(runner.gpus_for(ws3), "0")  # wraps around to GPU 0
                self.assertIs(runner.lock_for(ws1), runner.lock_for(ws3))
                self.assertIsNot(runner.lock_for(ws1), runner.lock_for(ws2))

                r1 = await runner.run('echo "$CUDA_VISIBLE_DEVICES|$APPTAINERENV_CUDA_VISIBLE_DEVICES"', ws1)
                r2 = await runner.run('echo "$CUDA_VISIBLE_DEVICES|$APPTAINERENV_CUDA_VISIBLE_DEVICES"', ws2)
                self.assertEqual(r1.output.strip(), "0|0")
                self.assertEqual(r2.output.strip(), "1|1")
            finally:
                for d in (ws1, ws2, ws3):
                    shutil.rmtree(d, ignore_errors=True)

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
