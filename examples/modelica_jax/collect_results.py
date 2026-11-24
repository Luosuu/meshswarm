"""Collect compact, git-friendly Phase 3 artifacts from raw meshevolve run folders.

Mirrors the structure of `results/{amrx,comd,lulesh}/`: copies `lessons_in.md`,
`lessons_out.md`, `scores.json`, `summary.md`, and per-swarm `solver_jax.py`,
`changes.diff` (against starter skeleton), `FINAL.md` (if written), and
`.mesh/report.md` -> `swarm-report.md` without copying bulky `.mesh/events.jsonl`.

    python examples/modelica_jax/collect_results.py --runs-root /scratch/$USER/meshswarm/modelica_jax/runs --tag p3 --out results/modelica_jax
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasks import DELIVERABLE, TASKS, skeleton_code  # noqa: E402

CAMPAIGN = {
    "tier1_blanket_dae": ["tier1-g1", "tier1-g2", "tier1-g3"],
    "tier2_transform_hx": ["tier2-g1", "tier2-g2"],
    "tier3_hdt_column": ["tier3-g1", "tier3-g2"],
}


def collect_one_tier(tier: str, suffixes: list[str], runs_root: Path, tag: str, out_root: Path) -> None:
    task = TASKS[tier]
    starter_lines = skeleton_code(task).splitlines(keepends=True)
    tier_out = out_root / tier
    tier_out.mkdir(parents=True, exist_ok=True)

    for gen_idx, suffix in enumerate(suffixes, start=1):
        src_evolve = runs_root / f"{tag}-{suffix}"
        src_gen = src_evolve / "gen-01"
        if not src_gen.is_dir():
            print(f"skip missing {src_gen}")
            continue

        dst_gen = tier_out / f"gen-{gen_idx}"
        dst_gen.mkdir(parents=True, exist_ok=True)
        for fname in ("lessons_in.md", "lessons_out.md", "scores.json"):
            if (src_gen / fname).is_file():
                shutil.copyfile(src_gen / fname, dst_gen / fname)
        if (src_evolve / "summary.md").is_file():
            shutil.copyfile(src_evolve / "summary.md", dst_gen / "summary.md")

        for run_dir in sorted(p for p in src_gen.iterdir() if p.is_dir()):
            dst_run = dst_gen / run_dir.name
            dst_run.mkdir(parents=True, exist_ok=True)
            cand = run_dir / DELIVERABLE
            if cand.is_file():
                shutil.copyfile(cand, dst_run / DELIVERABLE)
                cand_lines = cand.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                diff = "".join(
                    difflib.unified_diff(starter_lines, cand_lines, fromfile="starter/solver_jax.py", tofile=f"{run_dir.name}/solver_jax.py")
                )
                (dst_run / "changes.diff").write_text(diff, encoding="utf-8")
            if (run_dir / "FINAL.md").is_file():
                shutil.copyfile(run_dir / "FINAL.md", dst_run / "FINAL.md")
            rep = run_dir / ".mesh" / "report.md"
            if rep.is_file():
                shutil.copyfile(rep, dst_run / "swarm-report.md")
        print(f"collected {tier} gen-{gen_idx} from {src_evolve}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--tag", default="p3")
    ap.add_argument("--out", default="results/modelica_jax")
    a = ap.parse_args()

    runs_root = Path(a.runs_root)
    out_root = Path(a.out)
    for tier, suffixes in CAMPAIGN.items():
        collect_one_tier(tier, suffixes, runs_root, a.tag, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
