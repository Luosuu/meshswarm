"""One hard problem with nothing to split by part: the upper bound for three circles in the unit
square, the step no swarm proved in the n=3 and n=2..5 runs.

The definitions are the circle-packing mission's own. The exact bound `three_upper` is the core of
the mission's Milestone 2 (`c_n_three`). To give partial credit without favouring any proof
strategy, the targets are the same claim at increasing strength: any argument yields some bound,
and a stronger bound is worth more. The first is the trivial two-point bound; a pigeonhole on two
halves of the square reaches 27/100; the last is exact (about 0.25433).
"""

from __future__ import annotations

from task import ABOUT_HEADER, DEFS, DELIVERABLE, HEADER, KIND, footer, make_grader, proved  # noqa: F401

NS = "CirclePackingConstants"
EXACT = "(Real.sqrt 6 - Real.sqrt 2) / (2 * (1 + (Real.sqrt 6 - Real.sqrt 2)))"

LADDER = [
    ("three_upper_a", "(2 - Real.sqrt 2) / 2"),
    ("three_upper_b", "27 / 100"),
    ("three_upper_c", "26 / 100"),
    ("three_upper_d", "255 / 1000"),
    ("three_upper", EXACT),
]
TARGETS = [(name, f"∀ r : ℝ, {NS}.Packable 3 r → r ≤ {bound}") for name, bound in LADDER]

SKELETON = f"namespace {NS}\n\n" + "\n\n".join(
    f"theorem {name} : {stmt.replace(NS + '.', '')} := by\n  sorry" for name, stmt in TARGETS
) + f"\n\nend {NS}\n"

TASK = f"""Prove, in Lean 4 with Mathlib, the sharp upper bound on the radius of three equal circles packed in the unit square.

Three disks of radius r fit in the unit square exactly when their centers lie in [r, 1-r] x [r, 1-r] and
are pairwise at distance at least 2r. The claim is that this forces r ≤ d / (2 (1 + d)) with
d = sqrt 6 - sqrt 2, which is about 0.25433. Nobody on this team has a proof yet; finding one and
getting Lean to accept it is the whole job.

These definitions are fixed and are ALREADY in front of every file you check. Do not write `import`
lines and do not restate them:

```lean
import Mathlib

{DEFS}```

Deliverable: the file `{DELIVERABLE}` in the workspace, containing these five theorems with exactly these
names and statements, inside `namespace {NS}`. They are one claim at increasing strength: the first
bound is weak, the last is the exact one.

```lean
{SKELETON}```

How it is graded: each of the five theorems scores separately, so a proof of a weaker bound earns
credit even if the exact bound is never reached, and a proof of a stronger bound makes the weaker ones
easy consequences. A theorem counts only if its statement is unchanged and its proof uses no `sorry`
and no new axioms, directly or through anything it relies on. You may add helper lemmas.
`{DELIVERABLE}` must always compile: keep `sorry` in whatever is not proved yet, because a file that does
not compile scores nothing.

Use the `lean_check(path)` tool to compile any `.lean` file in the workspace. It reports errors with line
numbers. It is the only way to know whether a proof is right."""
