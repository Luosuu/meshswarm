"""Print a short status of an evolve output folder: python status.py OUT_DIR"""
import collections
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
for gen in sorted(out.glob("gen-*")):
    for run in sorted(p for p in gen.iterdir() if (p / ".mesh" / "events.jsonl").is_file()):
        ev = [json.loads(line) for line in open(run / ".mesh" / "events.jsonl") if line.strip()]
        kinds = collections.Counter(e["kind"] for e in ev)
        checks = [e for e in ev if e["kind"] == "lean_check"]
        done = sum(1 for e in checks if e["ok"] and not e["uses_sorry"])
        cost = sum(e.get("cost") or 0 for e in ev if e["kind"] == "usage")
        stop = next((e["reason"][:40] for e in ev if e["kind"] == "stop"), "running")
        print(f"{gen.name}/{run.name}: t={ev[-1]['t']:.0f}s cost=${cost:.3f} msgs={kinds['message']} writes={kinds['file_write']} "
              f"checks={len(checks)} (errors {sum(1 for e in checks if not e['ok'])}, complete {done}) "
              f"delegations={kinds['delegate']} errors={kinds['error']} truncated={kinds['truncated']} | {stop}")
    scores = gen / "scores.json"
    if scores.is_file():
        for r in json.loads(scores.read_text()):
            print(f"  SCORE {gen.name}/{r['run']}: {r['score']} lessons={r['had_lessons']} | {r['reasons']}")
