"""Optional Lean 4 checking: a `lean_check` tool for agents, and the checker a grader can reuse.

Nothing here is used unless you pass `tools=[lean_tool(checker)]` to `run_mesh`.

A checked file is the agent's text with a fixed `header` put in front (imports and shared
definitions), so agents write no imports and cannot change the definitions. Compiling a Lean
file can run code, so files containing commands that do so are refused, and `lean` should be
a command that runs inside a container (see examples/circle2).
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tools import ExtraTool

if TYPE_CHECKING:
    from .agent import MeshAgent

# Commands that run code or reach outside the file at compile time.
FORBIDDEN = re.compile(
    r"#eval|#exit|#print\s+axioms|\brun_cmd\b|\brun_elab\b|\brun_meta\b|\bIO\b|\bunsafe\b|\binitialize\b|"
    r"\bimplemented_by\b|\bextern\b|\binclude_str\b|\belab\b|\belab_rules\b|^\s*import\b", re.MULTILINE)
MESSAGE = re.compile(r"^(?P<path>[^\s:][^:\n]*):(?P<line>\d+):(?P<col>\d+):", re.MULTILINE)
OUTPUT_LIMIT = 3000


@dataclass
class LeanResult:
    ok: bool  # compiled with no errors (a proof with `sorry` still compiles)
    uses_sorry: bool
    raw: str  # Lean's output, untouched
    shown: str  # the same with the file name and line numbers as the author sees them
    seconds: float
    timed_out: bool = False


@dataclass
class LeanChecker:
    lean: list[str]  # command that compiles the file whose path is appended, e.g. ["lean"]
    header: str = "import Mathlib\n"
    env: dict[str, str] | None = None  # extra environment, e.g. {"LEAN_PATH": "..."}
    scratch: Path | None = None  # where combined files are written; must be visible to `lean`
    timeout_s: float = 240
    concurrency: int = 2

    def __post_init__(self) -> None:
        if not self.header.endswith("\n"):
            self.header += "\n"
        self._slots: asyncio.Semaphore | None = None

    @property
    def header_lines(self) -> int:
        return self.header.count("\n")

    async def check(self, body: str, name: str = "file.lean", footer: str = "") -> LeanResult:
        """Compile header + body + footer. Messages in `shown` use `name` and the body's own line numbers."""
        if self._slots is None:
            self._slots = asyncio.Semaphore(self.concurrency)
        scratch = Path(self.scratch) if self.scratch else Path(tempfile.gettempdir())
        scratch.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".lean", prefix="check_", dir=scratch)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(self.header + body + ("" if body.endswith("\n") else "\n") + footer)
        start = time.monotonic()
        timed_out = False
        async with self._slots:
            proc = await asyncio.create_subprocess_exec(
                *self.lean, tmp, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, **(self.env or {})})
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                out, _ = await proc.communicate()
                timed_out = True
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raw = out.decode("utf-8", "replace")
        shift = self.header_lines

        def rename(m: re.Match) -> str:
            return f"{name}:{max(int(m['line']) - shift, 1)}:{m['col']}:" if m["path"] == tmp else m[0]

        shown = MESSAGE.sub(rename, raw)
        if timed_out:
            shown += f"\n(stopped after {self.timeout_s:.0f}s: the file takes too long to compile)"
        ok = proc.returncode == 0 and not timed_out and ": error" not in raw
        return LeanResult(ok, "sorry" in raw, raw, shown, round(time.monotonic() - start, 1), timed_out)


def lean_tool(checker: LeanChecker, about_header: str = "") -> ExtraTool:
    """The `lean_check(path)` tool. `about_header` tells agents what is already in front of their file."""

    async def handler(a: "MeshAgent", path: str) -> Any:
        p = a.world.resolve(a.name, path)
        if not p.is_file():
            return {"error": f"{path!r} is not a file"}
        body = p.read_text(encoding="utf-8", errors="replace")
        bad = FORBIDDEN.search(body)
        if bad:
            return {"error": f"{bad.group(0).strip()!r} is not allowed in a checked file; remove it"}
        r = await checker.check(body, name=path)
        a.world.log.emit("lean_check", a.name, path=path, ok=r.ok, uses_sorry=r.uses_sorry, seconds=r.seconds)
        messages = r.shown.strip()
        if len(messages) > OUTPUT_LIMIT:
            messages = messages[:OUTPUT_LIMIT] + f"\n… {len(messages) - OUTPUT_LIMIT} more characters; fix the first errors first"
        return {"compiles": r.ok, "uses_sorry": r.uses_sorry, "complete": r.ok and not r.uses_sorry,
                "seconds": r.seconds, "messages": messages or "no messages"}

    description = (
        "Compile a Lean 4 file from the workspace and return the compiler's messages with line numbers. "
        "'complete' is true only when it compiles with no errors and no sorry. " + about_header
    ).strip()
    return ExtraTool("lean_check", description, handler, {"path": {"type": "string"}}, ["path"])
