"""Inspect live or finished Modelica-to-JAX meshevolve runs on Rivanna."""

import glob
import json
import sys
from pathlib import Path


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "/scratch/*/meshswarm/modelica_jax/runs/p3-*/gen-*/run-*"
    dirs = sorted(Path(p) for p in glob.glob(pattern) if (Path(p) / ".mesh" / "events.jsonl").is_file())
    if not dirs:
        print("no runs matched:", pattern)
        return 0
    for d in dirs:
        evs = [json.loads(ln) for ln in (d / ".mesh" / "events.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not evs:
            continue
        cost = sum(e.get("cost") or 0.0 for e in evs if e["kind"] == "usage")
        runs = [e for e in evs if e["kind"] == "run"]
        errs = [e for e in evs if e["kind"] == "error"]
        stop = next((e["reason"] for e in evs if e["kind"] == "stop"), "RUNNING")
        label = f"{d.parents[1].name}/{d.name}"
        print(
            f"{label:24} t={evs[-1]['t']:5.0f}s  cost=${cost:.3f}  events={len(evs):4d}  "
            f"cmds={len(runs):3d}  errs={len(errs)}  [{stop}]"
        )
        for r in runs[-2:]:
            cmd_preview = " ".join(str(r.get("command", "")).split())[:95]
            print(f"    {r['agent']} (exit {r['exit_code']}, {r['seconds']}s): {cmd_preview}")
        for er in errs:
            print(f"    ERROR {er['agent']}: {er.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
