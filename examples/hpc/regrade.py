"""Grade saved swarm workspaces again, optionally with fast-math flags removed from build.sh.

    python examples/hpc/regrade.py --kernel amr [--strip-fast-math] WORKSPACE [WORKSPACE ...]

Each workspace is copied first, so the saved run is never changed. Uses MESH_RUN_WRAP like run.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kernels  # noqa: E402
from meshswarm.shell import ShellRunner  # noqa: E402
from run import core_sets  # noqa: E402

STRIP = re.compile(r"\s(-ffast-math|-Ofast|-ffinite-math-only|-funsafe-math-optimizations|-fno-signed-nans)\b")


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=sorted(kernels.KERNELS), required=True)
    p.add_argument("--cores", type=int, default=8)
    p.add_argument("--strip-fast-math", action="store_true")
    p.add_argument("workspaces", nargs="+")
    a = p.parse_args()
    runner = ShellRunner(json.loads(os.environ.get("MESH_RUN_WRAP", '["bash", "-c"]')), core_sets=core_sets(a.cores, 1))
    grade = kernels.make_grader(a.kernel, runner)  # one grader: the serial baseline is timed once and shared
    for src in a.workspaces:
        ws = Path(tempfile.mkdtemp(prefix="regrade-", dir=os.environ.get("MESH_RUN_SCRATCH"))) / "ws"
        shutil.copytree(src, ws, ignore=shutil.ignore_patterns(".mesh"))
        (ws / ".mesh").mkdir()
        note = "as delivered"
        if a.strip_fast_math:
            build = ws / "build.sh"
            text = build.read_text()
            new = STRIP.sub(lambda m: " -O3" if m.group(1) == "-Ofast" else "", text)
            build.write_text(new)
            note = "fast-math flags removed" if new != text else "no fast-math flags found"

        class Result:
            workspace = ws

        g = await grade(Result())
        print(f"{src} [{note}] -> score {g.score} | {g.reasons}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
