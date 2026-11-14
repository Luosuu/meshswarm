"""Quick look at a swarm on the one-hard-problem task: ladder progress over time, delegation briefs,
check-ins, and what each agent is doing. python hardlook.py RUN_DIR"""
import collections
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
ev = [json.loads(line) for line in open(run / ".mesh" / "events.jsonl") if line.strip()]
k = collections.Counter(e["kind"] for e in ev)
cost = sum(e.get("cost") or 0 for e in ev if e["kind"] == "usage")
print(f"== {run} t={ev[-1]['t']:.0f}s cost=${cost:.2f} checks={k['lean_check']} cut_off={k['truncated']} "
      f"checkins={k['checkin']} delegations={k['delegate']} summaries={k['compact']}")
seen = {}
for e in ev:
    if e["kind"] == "lean_check" and e.get("ok"):
        for n in e.get("sorry_free") or []:
            n = n.split(".")[-1]
            if n.startswith("three_upper"):
                where = "FINAL" if e["path"] == "FINAL.lean" else "scratch"
                seen.setdefault((n, where), (round(e["t"] / 60), e["agent"], e["path"]))
for (n, where), (t, who, path) in sorted(seen.items(), key=lambda x: x[1][0]):
    print(f"  rung {n:14} first sorry-free in {where:7} at {t:3} min by {who} ({path})")
spawns = [(e["agent"], (e.get("role") or "")[:60]) for e in ev if e["kind"] == "spawn" and e.get("spawned_by") and not e.get("parent")]
print("  new peers:", spawns or "none")
for e in ev:
    if e["kind"] == "delegate":
        print(f"  [{e['t'] / 60:.0f}m] {e['agent']} -> {e['child'].split(':')[-1]}: " + " ".join(e["task"].split())[:300])
    if e["kind"] == "checkin":
        print(f"  [{e['t'] / 60:.0f}m] check-in woke {e['agent']} after {e['minutes']} min asleep")
for e in ev:
    if e["kind"] == "message" and e["agent"] != "mesh" and ":" not in e["agent"] and e["t"] > ev[-1]["t"] - 1500:
        print(f"  [{e['t'] / 60:.0f}m] {e['agent']} -> {e['to']}: " + " ".join(e["content"].split())[:230])
print("  files:", sorted(p.name for p in run.iterdir() if p.is_file())[:30])
