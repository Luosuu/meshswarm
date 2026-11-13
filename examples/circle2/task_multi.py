"""Four cases of prove2.me's mission "Circle packing in a square: exact constants" at once:
n = 2, 3, 4 and 5. Each case is an independent block of four theorems (a packing that reaches the
radius, the matching upper bound, the supremum, and the mission's official `c_n_*` milestone), so
there are sixteen graded theorems and the blocks can be worked on in parallel.

The cases differ a lot in difficulty: n = 2 is short, n = 5 is a pigeonhole argument, n = 3 and
n = 4 need real geometric arguments. The task text gives no hints.

The `c_n_*` statements are the mission's own (read from prove2.me on 2026-11-12; `c_n_five` is taken
from the mission's root goal). The other twelve theorems are our stepping stones for partial credit.
"""

from __future__ import annotations

from task import ABOUT_HEADER, DEFS, DELIVERABLE, HEADER, KIND, footer, make_grader, proved  # noqa: F401

NS = "CirclePackingConstants"

# n -> (name, optimal radius, right-hand side of the official c_n statement)
CASES = {
    2: ("two", "(2 - Real.sqrt 2) / 2", "Real.pi * (3 - 2 * Real.sqrt 2)"),
    3: ("three", "(Real.sqrt 6 - Real.sqrt 2) / (2 * (1 + (Real.sqrt 6 - Real.sqrt 2)))",
        "3 * Real.pi * ((Real.sqrt 6 - Real.sqrt 2) / (2 * (1 + (Real.sqrt 6 - Real.sqrt 2)))) ^ 2"),
    4: ("four", "1 / 4", "Real.pi / 4"),
    5: ("five", "(1 / Real.sqrt 2) / (2 * (1 + 1 / Real.sqrt 2))",
        "5 * Real.pi * (((1 / Real.sqrt 2) / (2 * (1 + 1 / Real.sqrt 2)))) ^ 2"),
}

TARGETS = []
for n, (word, radius, area) in CASES.items():
    TARGETS += [
        (f"{word}_packable", f"{NS}.Packable {n} ({radius})"),
        (f"{word}_upper", f"∀ r : ℝ, {NS}.Packable {n} r → r ≤ {radius}"),
        (f"r_{word}", f"{NS}.r_n {n} = {radius}"),
        (f"c_n_{word}", f"{NS}.c_n {n} = {area}"),
    ]

SKELETON = f"namespace {NS}\n\n" + "\n\n".join(
    f"theorem {name} : {stmt.replace(NS + '.', '')} := by\n  sorry" for name, stmt in TARGETS
) + f"\n\nend {NS}\n"

TASK = f"""Prove, in Lean 4 with Mathlib, the exact optimal packings of 2, 3, 4 and 5 equal circles in the unit square.

n disks of radius r fit in the unit square exactly when their centers lie in [r, 1-r] x [r, 1-r] and are
pairwise at distance at least 2r. For each n the claim is an exact value for the largest such r.

These definitions are fixed and are ALREADY in front of every file you check. Do not write `import`
lines and do not restate them:

```lean
import Mathlib

{DEFS}```

Deliverable: the file `{DELIVERABLE}` in the workspace, containing these sixteen theorems with exactly
these names and statements, inside `namespace {NS}`. There are four theorems for each n: a packing that
reaches the radius, the matching upper bound, the supremum, and the area formula.

```lean
{SKELETON}```

How it is graded: each of the sixteen theorems scores separately, so a file with some theorems proved
and others left as `sorry` still earns credit. A theorem counts only if its statement is unchanged and
its proof uses no `sorry` and no new axioms, directly or through anything it relies on: a theorem that
uses an earlier theorem counts only once that earlier theorem is fully proved too. You may add helper
lemmas. The four cases are independent of each other and are not equally hard.
`{DELIVERABLE}` must always compile: keep `sorry` in whatever is not proved yet, and never let an
unfinished proof break the file, because a file that does not compile scores nothing.

Use the `lean_check(path)` tool to compile any `.lean` file in the workspace. It reports errors with line
numbers. It is the only way to know whether a proof is right."""
