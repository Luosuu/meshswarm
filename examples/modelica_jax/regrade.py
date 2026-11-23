"""Re-grade saved Modelica-to-JAX workspaces sequentially on one GPU against one shared OpenModelica baseline.

    python examples/modelica_jax/regrade.py --tier tier1_blanket_dae WORKSPACE [WORKSPACE ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grader import make_grader  # noqa: E402
from meshswarm.shell import ShellRunner  # noqa: E402
from run import detect_gpu_sets  # noqa: E402
from tasks import TASKS  # noqa: E402


class _Result:
    def __init__(self, workspace: Path):
        self.workspace = workspace


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=sorted(TASKS), required=True)
    p.add_argument("--gpus", default=None)
    p.add_argument("--n-ensemble", type=int, default=1000)
    p.add_argument("workspaces", nargs="+")
    a = p.parse_args()

    wrap = json.loads(os.environ.get("MESH_RUN_WRAP", '["bash", "-c"]'))
    gpus = detect_gpu_sets(a.gpus)
    runner = ShellRunner(wrap, gpu_sets=gpus[:1] if gpus else [])
    grade = make_grader(a.tier, runner=runner, n_ensemble=a.n_ensemble)

    scratch = os.environ.get("MESH_RUN_SCRATCH")
    for src in a.workspaces:
        tmp_root = Path(tempfile.mkdtemp(prefix="regrade-mjax-", dir=scratch))
        ws = tmp_root / "ws"
        try:
            shutil.copytree(src, ws, ignore=shutil.ignore_patterns(".mesh"))
            (ws / ".mesh").mkdir(exist_ok=True)
            g = await grade(_Result(ws))
            print(f"{src} -> score {g.score} | {g.reasons}", flush=True)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
