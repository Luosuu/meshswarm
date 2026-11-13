"""Show how a running swarm is collaborating: python peek.py RUN_DIR"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from meshevolve.digest import shape, timeline  # noqa: E402

run = Path(sys.argv[1])
ev = [json.loads(line) for line in open(run / ".mesh" / "events.jsonl") if line.strip()]
s = shape(ev)
print({k: s[k] for k in ("peer_messages", "hub_share", "tasks_posted", "claims", "handovers", "delegations", "lean_checks", "lean_checks_ok", "tokens")})
print("pairs:", s["message_pairs"])
board = run / ".mesh" / "board.json"
tasks = {}
for e in ev:
    if e["kind"] == "task_posted":
        tasks[e["task"]] = [e["title"][:70], "open", "-"]
    elif e["kind"] == "task_claimed":
        tasks[e["task"]][1:] = ["claimed", e["agent"]]
    elif e["kind"] == "task_done":
        tasks[e["task"]][1] = "done"
for k, v in tasks.items():
    print(f"task #{k}: {v[1]:8} {v[2]:8} {v[0]}")
print(timeline(ev, max_lines=int(sys.argv[2]) if len(sys.argv) > 2 else 45, clip=230))
print("files:", sorted(p.name for p in run.iterdir() if p.is_file()))
