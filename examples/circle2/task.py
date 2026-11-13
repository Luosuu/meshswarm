"""The n = 2 case of prove2.me's mission "Circle packing in a square: exact constants".

DEFS is the mission's own definition file, copied from prove2.me (read 2026-11-12), and the last
target, `c_n_two`, is the mission's Milestone 1 theorem with its official name and statement.
The three theorems before it are our own stepping stones, so that partial progress is scored.
"""

from __future__ import annotations

import re

from meshswarm import RunResult
from meshswarm.lean import FORBIDDEN, LeanChecker
from meshevolve import Grade

DEFS = """noncomputable section

namespace CirclePackingConstants

abbrev Point := ℝ × ℝ

/-- Squared Euclidean distance in Cartesian coordinates. -/
def sqDist (p q : Point) : ℝ :=
  (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2

/-- Coordinate containment conditions for a disk center in the unit square. -/
def InInnerSquare (r : ℝ) (p : Point) : Prop :=
  r ≤ p.1 ∧ p.1 ≤ 1 - r ∧ r ≤ p.2 ∧ p.2 ≤ 1 - r

/-- `n` equal disks of radius `r` fit in the unit square with disjoint interiors. -/
def Packable (n : ℕ) (r : ℝ) : Prop :=
  0 ≤ r ∧ r ≤ 1 / 2 ∧
    ∃ p : Fin n → Point,
      (∀ i, InInnerSquare r (p i)) ∧
      ∀ i j, i ≠ j → (2 * r) ^ 2 ≤ sqDist (p i) (p j)

/-- The supremum of radii admitted by `Packable`. -/
def r_n (n : ℕ) : ℝ := sSup {r : ℝ | Packable n r}

/-- The optimal covered-area fraction for `n` equal disks in the unit square. -/
def c_n (n : ℕ) : ℝ := (n : ℝ) * Real.pi * (r_n n) ^ 2

end CirclePackingConstants
"""

HEADER = "import Mathlib\n\n" + DEFS + "\n"

# (name, statement). Each is worth the same.
TARGETS = [
    ("two_packable", "CirclePackingConstants.Packable 2 ((2 - Real.sqrt 2) / 2)"),
    ("two_upper", "∀ r : ℝ, CirclePackingConstants.Packable 2 r → r ≤ (2 - Real.sqrt 2) / 2"),
    ("r_two", "CirclePackingConstants.r_n 2 = (2 - Real.sqrt 2) / 2"),
    ("c_n_two", "CirclePackingConstants.c_n 2 = Real.pi * (3 - 2 * Real.sqrt 2)"),
]
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
DELIVERABLE = "FINAL.lean"
KIND = "proving stated theorems in Lean 4 with Mathlib, as a team with a compiler-check tool"

SKELETON = "namespace CirclePackingConstants\n\n" + "\n\n".join(
    f"theorem {name} : {stmt.replace('CirclePackingConstants.', '')} := by\n  sorry" for name, stmt in TARGETS
) + "\n\nend CirclePackingConstants\n"

TASK = f"""Prove, in Lean 4 with Mathlib, the exact optimal packing of two equal circles in the unit square.

Two disks of radius r fit in the unit square exactly when their centers lie in [r, 1-r] x [r, 1-r] and are
at distance at least 2r. The largest such r is (2 - sqrt 2) / 2, reached by centers on a diagonal.

These definitions are fixed and are ALREADY in front of every file you check. Do not write `import`
lines and do not restate them:

```lean
import Mathlib

{DEFS}```

Deliverable: the file `{DELIVERABLE}` in the workspace, containing these four theorems with exactly these
names and statements, inside `namespace CirclePackingConstants`:

```lean
{SKELETON}```

How it is graded: each of the four theorems scores separately, so a file with some theorems proved and
others left as `sorry` still earns credit. A theorem counts only if its statement is unchanged and its
proof uses no `sorry` and no new axioms, directly or through anything it relies on: a theorem that uses
an earlier theorem counts only once that earlier theorem is fully proved too. You may add helper lemmas.

Use the `lean_check(path)` tool to compile any `.lean` file in the workspace. It reports errors with line
numbers. Check often and in small pieces: it is the only way to know whether a proof is right.
Mathematical hints: for the upper bound, each coordinate difference of two centers is at most 1 - 2r in
absolute value; `nlinarith` with `Real.sq_sqrt` and `Real.sqrt_nonneg` handles sqrt 2; for the supremum,
see `csSup_le` and `le_csSup`."""

ABOUT_HEADER = ("`import Mathlib` and the CirclePackingConstants definitions from the goal are already in front of the "
                "file, so do not import or restate them.")


def footer(targets=None) -> str:
    targets = TARGETS if targets is None else targets
    lines = ["", "-- grader"]
    for i, (name, stmt) in enumerate(targets, 1):
        lines.append(f"theorem Grader.check_{i} : {stmt} := _root_.CirclePackingConstants.{name}")
        lines.append(f"#print axioms Grader.check_{i}")
    return "\n".join(lines) + "\n"


def proved(raw: str, targets=None) -> dict[str, bool]:
    """Which targets Lean accepted with only the standard axioms, from the output of body + footer."""
    targets = TARGETS if targets is None else targets
    out = {}
    for i, (name, _) in enumerate(targets, 1):
        m = re.search(rf"'(?:[\w.]+\.)?Grader\.check_{i}' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])", raw)
        axioms = {a.strip() for a in (m.group(2) or "").split(",") if a.strip()} if m else {"missing"}
        out[name] = m is not None and axioms <= ALLOWED_AXIOMS
    return out


def make_grader(checker: LeanChecker, targets=None):
    targets = TARGETS if targets is None else targets

    async def grade(result: RunResult) -> Grade:
        path = result.workspace / DELIVERABLE
        if not path.is_file():
            return Grade(0.0, f"no {DELIVERABLE} was written")
        body = path.read_text(encoding="utf-8", errors="replace")
        bad = FORBIDDEN.search(body)
        if bad:
            return Grade(0.0, f"{DELIVERABLE} contains {bad.group(0).strip()!r}, which is not allowed")
        r = await checker.check(body, name=DELIVERABLE, footer=footer(targets))
        if r.timed_out:
            return Grade(0.0, f"{DELIVERABLE} did not compile within {checker.timeout_s:.0f}s")
        done = proved(r.raw, targets)
        yes = [n for n, ok in done.items() if ok]
        no = [n for n, ok in done.items() if not ok]
        score = round(10 * len(yes) / len(targets), 2)
        return Grade(score, f"proved: {', '.join(yes) or 'none'}; not proved: {', '.join(no) or 'none'}")

    return grade
