"""Describe how a finished meshswarm run organized itself, from its event log.

Nothing here prescribes a shape. `shape` gives numbers for tracking across
generations; `timeline` gives the reviewer a condensed play-by-play.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

MESH = "mesh"


def load_events(workspace: str | Path) -> list[dict[str, Any]]:
    path = Path(workspace) / ".mesh" / "events.jsonl"
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _is_sub(name: str) -> bool:
    return ":" in name  # sub-agents are named parent:child


def shape(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Numbers describing the organization that emerged in one run."""
    kinds = Counter(e["kind"] for e in events)
    spawns = [e for e in events if e["kind"] == "spawn"]
    peers = [e for e in spawns if not e.get("parent")]
    peer_msgs = [e for e in events if e["kind"] == "message" and e["agent"] != MESH
                 and not _is_sub(e["agent"]) and not _is_sub(e["to"])]
    pairs = Counter((e["agent"], e["to"]) for e in peer_msgs)
    involved: Counter[str] = Counter()
    for (sender, to), n in pairs.items():
        involved[sender] += n
        involved[to] += n
    usage = [e for e in events if e["kind"] == "usage"]
    priced = bool(usage) and all(e.get("cost") is not None for e in usage)
    stop = next((e for e in events if e["kind"] == "stop"), None)
    return {
        "agents": len(peers),
        "spawned_peers": sum(1 for e in peers if e.get("spawned_by")),
        "subagents": len(spawns) - len(peers),
        "max_depth": max((e.get("depth") or 0 for e in spawns), default=0),
        "peer_messages": len(peer_msgs),
        "talking_pairs": len({frozenset(p) for p in pairs}),
        # Share of peer messages that involve the busiest agent: 1.0 means everything went through one hub.
        "hub_share": round(max(involved.values()) / len(peer_msgs), 2) if peer_msgs else None,
        "tasks_posted": sum(1 for e in events if e["kind"] == "task_posted" and e["agent"] != MESH),
        "claims": kinds.get("task_claimed", 0),
        "handovers": kinds.get("task_handover", 0),
        "delegations": kinds.get("delegate", 0),
        "waits": kinds.get("wait", 0),
        "errors": kinds.get("error", 0),
        "seconds": round(stop["t"]) if stop else None,
        "stop": stop["reason"].split(":")[0].split(" (")[0] if stop else "?",
        "tokens": sum(e["prompt"] + e["completion"] for e in usage),
        "cost_usd": round(sum(e["cost"] for e in usage), 4) if priced else None,
        "message_pairs": {f"{a} -> {b}": n for (a, b), n in sorted(pairs.items())},
    }


def _clip(text: Any, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _line(e: dict[str, Any], clip: int) -> str | None:
    kind, who = e["kind"], e["agent"]
    if kind == "message":
        if who == MESH:
            return None  # seeds and nudges, the same in every run
        return f"{who} -> {e['to']}: {_clip(e['content'], clip)}"
    if kind == "task_posted":
        return None if who == MESH else f"{who} posted task #{e['task']}: {_clip(e['title'], clip)}"
    if kind == "task_claimed":
        return f"{who} claimed task #{e['task']}"
    if kind == "task_released":
        return f"{who} released task #{e['task']}"
    if kind == "task_handover":
        return f"{who} handed task #{e['task']} to {e['to']}: {_clip(e.get('note', ''), clip)}"
    if kind == "task_done":
        return f"{who} completed task #{e['task']}: {_clip(e.get('result', ''), clip)}"
    if kind == "spawn" and (e.get("spawned_by") or e.get("parent")):
        by = e.get("parent") or e.get("spawned_by")
        return f"{by} created {'sub-agent' if e.get('parent') else 'peer'} {who} ({_clip(e.get('role') or 'no role', 60)})"
    if kind == "delegate":
        return f"{who} delegated to {e['child']}: {_clip(e['task'], clip)}"
    if kind == "report":
        return f"{who} reported to {e.get('parent')}: {_clip(e.get('result', ''), clip)}"
    if kind == "file_write":
        return f"{who} wrote {e['path']} ({e['chars']} chars{', append' if e.get('append') else ''})"
    if kind in ("error", "budget", "wrap_up", "team_halt", "stop", "compact", "poll_stop"):
        detail = e.get("error") or e.get("reason") or e.get("budget") or ""
        return f"{who}: {kind} {_clip(detail, clip)}".rstrip()
    return None


def timeline(events: list[dict[str, Any]], max_lines: int = 80, clip: int = 160) -> str:
    """Coordination events in order. Long runs keep the opening and the ending."""
    lines = []
    for e in events:
        text = _line(e, clip)
        if text:
            lines.append(f"{e['t']:7.1f}s  {text}")
    if len(lines) > max_lines:
        head = max_lines // 2
        tail = max_lines - head
        lines = lines[:head] + [f"          … {len(lines) - max_lines} events omitted …"] + lines[-tail:]
    return "\n".join(lines)
