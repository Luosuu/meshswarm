"""How is a swarm collaborating? Reads a run folder (finished or still running) and prints, per target
theorem, who worked on it and where it stands, plus stranded work, the deliverable's history and
what sub-agents were used for.

    python patterns.py RUN_DIR --task task_sens
"""
import argparse
import collections
import importlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[3]), str(HERE.parents[1])]

ROLES = [  # first match wins
    ("merge", r"\bmerge|replace (exactly|the)|\bsplice|insert into|edit (the file )?`?FINAL|in FINAL\.lean"),
    ("search", r"\bgrep|\bsearch|\blocate|\blist the|find (the )?(directory|occurrences)|which files"),
    ("paper proof", r"no lean|do not write any lean|pure math|rigorous mathematic|derive a|by hand|on paper|proof plan"),
    ("verify", r"\bverify|\bcheck (these|whether|that)|refute|double-check|count occurrences"),
    ("prove in Lean", r"lean_check|prove|formaliz|\.lean"),
]


def role_of(text: str) -> str:
    for name, pattern in ROLES:
        if re.search(pattern, text, re.I):
            return name
    return "other"


def blocks(text: str) -> dict[str, str]:
    out = {}
    for part in re.split(r"(?m)^(?=(?:private |protected )?(?:theorem|lemma) )", text):
        m = re.match(r"(?:private |protected )?(?:theorem|lemma) (\S+)", part)
        if m:
            out[m.group(1).split(".")[-1]] = part
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--task", default="task_sens")
    a = ap.parse_args()
    task = importlib.import_module(a.task)
    names = [n for n, _ in task.TARGETS]
    run = Path(a.run)
    ev = [json.loads(line) for line in open(run / ".mesh" / "events.jsonl") if line.strip()]
    now = ev[-1]["t"]
    top = lambda agent: agent.split(":")[0]  # noqa: E731
    stop = next((e["reason"][:50] for e in ev if e["kind"] == "stop"), "running")
    cost = sum(e.get("cost") or 0 for e in ev if e["kind"] == "usage")
    print(f"== {run}  t={now:.0f}s  cost=${cost:.2f}  {stop}")

    # Who touched which target: files they wrote (final contents) and briefs they sent or received.
    writers = collections.defaultdict(set)
    for e in ev:
        if e["kind"] == "file_write":
            writers[e["path"]].add(top(e["agent"]))
    last_check = {}
    for e in ev:
        if e["kind"] == "lean_check":
            last_check[e["path"]] = e
    touched = collections.defaultdict(set)
    proved_in = collections.defaultdict(list)
    for f in run.glob("*.lean"):
        text = f.read_text(errors="replace")
        b = blocks(text)
        for n in names:
            if n in b:
                real = "sorry" not in b[n]
                if f.name != task.DELIVERABLE and real:
                    touched[n] |= writers.get(f.name, set())
                chk = last_check.get(f.name)
                if real and chk and chk["ok"]:
                    proved_in[n].append(f.name)
    for e in ev:
        if e["kind"] == "delegate":
            for n in names:
                if re.search(rf"\b{re.escape(n)}\b", e["task"]):
                    touched[n].add(top(e["agent"]))
    final = blocks((run / task.DELIVERABLE).read_text(errors="replace")) if (run / task.DELIVERABLE).is_file() else {}
    print("targets:")
    stranded = []
    for n in names:
        in_final = n in final and "sorry" not in final[n]
        elsewhere = [f for f in proved_in[n] if f != task.DELIVERABLE]
        state = "IN FINAL" if in_final else ("proved in " + ",".join(elsewhere[:2]) + " (not merged)" if elsewhere else "open")
        if not in_final and elsewhere:
            stranded.append(n)
        print(f"  {n:42} {state:38} worked on by {sorted(touched[n]) or '-'}")
    overlap = [n for n in names if len(touched[n]) > 1]
    untouched = [n for n in names if not touched[n] and not (n in final and 'sorry' not in final[n])]
    print(f"overlap (2+ agents on one target): {overlap or 'none'}")
    print(f"untouched and unproved: {untouched or 'none'}")
    print(f"stranded (proved in a checked scratch file, missing from the deliverable): {stranded or 'none'}")

    # The deliverable's history.
    fw = [e for e in ev if e["kind"] == "file_write" and e["path"] == task.DELIVERABLE]
    fc = [e for e in ev if e["kind"] == "lean_check" and e["path"] == task.DELIVERABLE]
    broken = sum(1 for e in fc if not e["ok"])
    print(f"deliverable: {len(fw)} writes by {dict(collections.Counter(e['agent'] for e in fw))}; "
          f"{len(fc)} checks, {broken} failed; last check {'none' if not fc else ('FAILED' if not fc[-1]['ok'] else 'ok')} "
          f"at {fc[-1]['t']:.0f}s" if fc else f"deliverable: {len(fw)} writes, never checked")

    # Sub-agents and what they were for.
    dels = [e for e in ev if e["kind"] == "delegate"]
    roles = collections.Counter(role_of(e["task"]) for e in dels)
    peers = [e["agent"] for e in ev if e["kind"] == "spawn" and e.get("spawned_by") and not e.get("parent")]
    depth = max((e["agent"].count(":") for e in ev if e["kind"] == "spawn"), default=0)
    print(f"delegations: {len(dels)} {dict(roles)}; deepest nesting {depth}; new peers {peers or 'none'}")
    same = collections.Counter()
    for n in names:
        kids = {e["child"] for e in dels if re.search(rf"\b{re.escape(n)}\b", e["task"]) and role_of(e["task"]) in ("prove in Lean", "paper proof")}
        if len(kids) > 1:
            same[n] = len(kids)
    print(f"targets given to several sub-agents: {dict(same) or 'none'}")

    # Per agent: effort and waste.
    per = collections.defaultdict(lambda: collections.Counter())
    for e in ev:
        a_ = top(e["agent"])
        if e["kind"] == "usage":
            per[a_]["calls"] += 1
            per[a_]["cents"] += round(100 * (e.get("cost") or 0), 3)
            per[a_]["sub_cents"] += round(100 * (e.get("cost") or 0), 3) if ":" in e["agent"] else 0
        elif e["kind"] in ("truncated", "compact", "lean_check", "wait", "context_trim"):
            per[a_][e["kind"]] += 1
        elif e["kind"] == "message" and e["agent"] != "mesh":
            per[a_]["msgs"] += 1
    for a_ in sorted(k for k in per if k != "mesh"):
        c = per[a_]
        print(f"  {a_}: ${c['cents'] / 100:.2f} ({c['sub_cents'] / max(c['cents'], 1e-9):.0%} via sub-agents), {c['calls']} calls, "
              f"{c['lean_check']} checks, {c['msgs']} msgs, {c['truncated']} cut off, {c['compact']} summaries, {c['wait']} waits")
    notes = sorted(p.name for p in run.glob("*.md"))
    print(f"notes on disk: {notes[:12]}")


if __name__ == "__main__":
    main()
