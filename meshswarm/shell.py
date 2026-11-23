"""Optional command execution: a `run_command` tool for agents, and the runner a grader can reuse.

Nothing here is used unless you pass `tools=[run_tool(runner)]` to `run_mesh`. A command runs in the
swarm's workspace folder. It is arbitrary code, so `wrap` should start it inside a container that
sees only that folder (see examples/hpc); the default runs it directly and is meant for tests.

Timing matters for performance work, so a runner executes one command at a time per workspace and
can pin each workspace to its own set of cores.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tools import ExtraTool

if TYPE_CHECKING:
    from .agent import MeshAgent

OUTPUT_LIMIT = 4000


@dataclass
class RunResult:
    exit_code: int | None  # None when the command was stopped at the time limit
    output: str  # stdout and stderr together
    seconds: float
    timed_out: bool = False


@dataclass
class ShellRunner:
    # Command prefix; "{cwd}" in any part is replaced by the workspace folder. The command string is
    # appended as the last argument, so the prefix normally ends with a shell and "-c".
    wrap: list[str] = field(default_factory=lambda: ["bash", "-c"])
    timeout_s: float = 120  # default limit per command
    max_timeout_s: float = 600  # the most an agent may ask for
    core_sets: list[str] = field(default_factory=list)  # e.g. ["0-7", "8-15"]: one set per workspace
    gpu_sets: list[str] = field(default_factory=list)  # e.g. ["0", "1"]: one CUDA_VISIBLE_DEVICES set per workspace
    env: dict[str, str] | None = None

    def __post_init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._cores: dict[str, str] = {}
        self._gpus: dict[str, str] = {}

    def cores_for(self, workspace: Path) -> str | None:
        """The core set this workspace is pinned to (assigned on first use, in order)."""
        key = str(Path(workspace).resolve())
        if key not in self._cores and self.core_sets:
            self._cores[key] = self.core_sets[len(self._cores) % len(self.core_sets)]
        return self._cores.get(key)

    def gpus_for(self, workspace: Path) -> str | None:
        """The GPU set (`CUDA_VISIBLE_DEVICES`) this workspace is pinned to (assigned on first use, in order)."""
        key = str(Path(workspace).resolve())
        if key not in self._gpus and self.gpu_sets:
            self._gpus[key] = self.gpu_sets[len(self._gpus) % len(self.gpu_sets)]
        return self._gpus.get(key)

    def lock_for(self, workspace: Path) -> asyncio.Lock:
        """Lock serializing commands in `workspace` (and across workspaces that share the same pinned GPU)."""
        ws = Path(workspace).resolve()
        gpus = self.gpus_for(ws)
        lock_key = f"gpu:{gpus}" if gpus is not None else str(ws)
        return self._locks.setdefault(lock_key, asyncio.Lock())

    def threads_for(self, workspace: Path) -> int | None:
        cores = self.cores_for(workspace)
        if not cores:
            return None
        count = 0
        for part in cores.split(","):
            lo, _, hi = part.partition("-")
            count += int(hi or lo) - int(lo) + 1
        return count

    async def run(self, command: str, workspace: Path, timeout_s: float | None = None) -> RunResult:
        workspace = Path(workspace).resolve()
        limit = min(timeout_s or self.timeout_s, self.max_timeout_s)
        argv = [part.replace("{cwd}", str(workspace)) for part in self.wrap] + [command]
        cores = self.cores_for(workspace)
        gpus = self.gpus_for(workspace)
        env = {**os.environ, **(self.env or {})}
        if cores:
            argv = ["taskset", "-c", cores] + argv
            env.setdefault("OMP_NUM_THREADS", str(self.threads_for(workspace)))
        if gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpus
            env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = gpus
        lock = self.lock_for(workspace)
        async with lock:  # one command at a time per workspace (or shared GPU) keeps timings clean
            start = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=workspace, env=env, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
            timed_out = False
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), limit)
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)  # the whole group: a build may have children
                except ProcessLookupError:
                    pass
                out, _ = await proc.communicate()
            seconds = round(time.monotonic() - start, 2)
        return RunResult(None if timed_out else proc.returncode, out.decode("utf-8", "replace"), seconds, timed_out)


def clip(text: str, limit: int = OUTPUT_LIMIT) -> str:
    """Keep the start and the end: compiler errors come first, results last."""
    if len(text) <= limit:
        return text
    head = limit // 3
    return text[:head] + f"\n… {len(text) - limit} characters cut …\n" + text[-(limit - head):]


def run_tool(runner: ShellRunner, about: str = "") -> ExtraTool:
    """The `run_command(command, timeout_s?)` tool."""

    async def handler(a: "MeshAgent", command: str, timeout_s: float | None = None) -> Any:
        workspace = a.world.teams[a.team].workspace
        r = await runner.run(command, workspace, timeout_s)
        a.world.log.emit("run", a.name, command=command[:300], exit_code=r.exit_code, seconds=r.seconds,
                         timed_out=r.timed_out or None)
        out: dict[str, Any] = {"exit_code": r.exit_code, "seconds": r.seconds, "output": clip(r.output.strip()) or "(no output)"}
        if r.timed_out:
            out["timed_out"] = f"stopped after {r.seconds:.0f}s; pass a larger timeout_s (at most {runner.max_timeout_s:.0f}) or run something smaller"
        return out

    cores = " Each command is pinned to this team's own cores." if runner.core_sets else ""
    gpus = " Each command is pinned to this team's own GPU." if runner.gpu_sets else ""
    description = (
        "Run a shell command in the workspace folder and return its exit code, output and wall time in seconds. "
        "Use it to compile, run, test and time programs. Commands run one at a time, so timings are not disturbed "
        f"by your teammates; the default time limit is {runner.timeout_s:.0f}s.{cores}{gpus} " + about
    ).strip()
    return ExtraTool("run_command", description, handler,
                     {"command": {"type": "string"}, "timeout_s": {"type": "number"}}, ["command"])
