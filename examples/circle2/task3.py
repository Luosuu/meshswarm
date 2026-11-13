"""The n = 3 case of prove2.me's mission "Circle packing in a square: exact constants". Harder than
n = 2: the upper bound needs a real geometric argument, and the task text gives no hints.

`c_n_three` is the mission's Milestone 2 theorem with its official name and statement (read from
prove2.me on 2026-11-12). The four theorems before it are our own stepping stones for partial
credit; `three_upper_weak` is a deliberately easier bound so that scores are not all-or-nothing.
"""

from __future__ import annotations

from task import ABOUT_HEADER, DEFS, DELIVERABLE, HEADER, KIND, footer, make_grader, proved  # noqa: F401

R3 = "(Real.sqrt 6 - Real.sqrt 2) / (2 * (1 + (Real.sqrt 6 - Real.sqrt 2)))"
NS = "CirclePackingConstants"

TARGETS = [
    ("three_packable", f"{NS}.Packable 3 ({R3})"),
    ("three_upper_weak", f"∀ r : ℝ, {NS}.Packable 3 r → r ≤ (2 - Real.sqrt 2) / 2"),
    ("three_upper", f"∀ r : ℝ, {NS}.Packable 3 r → r ≤ {R3}"),
    ("r_three", f"{NS}.r_n 3 = {R3}"),
    ("c_n_three", f"{NS}.c_n 3 = 3 * Real.pi * (({R3})) ^ 2"),
]

SKELETON = f"namespace {NS}\n\n" + "\n\n".join(
    f"theorem {name} : {stmt.replace(NS + '.', '')} := by\n  sorry" for name, stmt in TARGETS
) + f"\n\nend {NS}\n"

TASK = f"""Prove, in Lean 4 with Mathlib, the exact optimal packing of three equal circles in the unit square.

Three disks of radius r fit in the unit square exactly when their centers lie in [r, 1-r] x [r, 1-r] and
are pairwise at distance at least 2r. The claim is that the largest such r is d / (2 (1 + d)) with
d = sqrt 6 - sqrt 2.

These definitions are fixed and are ALREADY in front of every file you check. Do not write `import`
lines and do not restate them:

```lean
import Mathlib

{DEFS}```

Deliverable: the file `{DELIVERABLE}` in the workspace, containing these five theorems with exactly these
names and statements, inside `namespace {NS}`:

```lean
{SKELETON}```

How it is graded: each of the five theorems scores separately, so a file with some theorems proved and
others left as `sorry` still earns credit. A theorem counts only if its statement is unchanged and its
proof uses no `sorry` and no new axioms, directly or through anything it relies on: a theorem that uses
an earlier theorem counts only once that earlier theorem is fully proved too. You may add helper lemmas.
`{DELIVERABLE}` must always compile: keep `sorry` in whatever is not proved yet.

Use the `lean_check(path)` tool to compile any `.lean` file in the workspace. It reports errors with line
numbers. It is the only way to know whether a proof is right."""
