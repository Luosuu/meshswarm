"""Lemmas from prove2.me's mission "Block Sensitivity vs. Sensitivity: OpenAI Baseline": sensitivity
and block sensitivity of Boolean functions. A different area from the circle-packing tasks, used to
test whether lessons learned there carry over.

DEFS is the part of the mission's definition files that these lemmas need (the campaign definitions,
Hamming balls, nested families, the initial family, composition and OR), read from prove2.me on
2026-11-13 and re-indented. Eight targets are the mission's own theorems with their official names
and statements. The four `*_le_*` warm-ups at the start are ours, added so that scores are not
all-or-nothing.
"""

from __future__ import annotations

from meshswarm import RunResult
from meshswarm.lean import FORBIDDEN, LeanChecker
from meshevolve import Grade

from task import ABOUT_HEADER as _ABOUT, DELIVERABLE, KIND, proved  # noqa: F401

DEFS = """namespace OAI

noncomputable section
open scoped Classical BigOperators

namespace Paper320

def flip {I : Type} (x : I → Bool) (B : Finset I) : I → Bool :=
  fun i => if i ∈ B then !(x i) else x i

def sensitivityAt {I : Type} [Fintype I]
    (f : (I → Bool) → Bool) (x : I → Bool) : ℕ :=
  (Finset.univ.filter fun i => f (flip x {i}) ≠ f x).card

def sensitivity {I : Type} [Fintype I] (f : (I → Bool) → Bool) : ℕ :=
  Finset.univ.sup (sensitivityAt f)

def blockSensitivityAt {I : Type} [Fintype I]
    (f : (I → Bool) → Bool) (x : I → Bool) : ℕ :=
  Finset.univ.sup fun blocks : Finset (Finset I) =>
    if (∀ A ∈ blocks, ∀ B ∈ blocks, A ≠ B → Disjoint A B) ∧
        (∀ B ∈ blocks, B.Nonempty ∧ f (flip x B) ≠ f x)
    then blocks.card else 0

def blockSensitivity {I : Type} [Fintype I] (f : (I → Bool) → Bool) : ℕ :=
  Finset.univ.sup (blockSensitivityAt f)

/-- The union of equal-radius Hamming balls. -/
def ballPredicate {I C : Type} [Fintype I] [Fintype C] (centers : C → I → Bool) (R : ℕ)
    (x : I → Bool) : Bool :=
  decide (∃ c, hammingDist x (centers c) ≤ R)

/-- Acceptance at a stronger index implies acceptance at every weaker index. -/
structure NestedFamily {h : ℕ} {I : Type}
    (F : Fin h → (I → Bool) → Bool) : Prop where
  accepts_of_le : ∀ {q q' : Fin h} {x}, q ≤ q' → F q' x = true → F q x = true

/-- The number of coordinates whose one-bit flip changes both predicates. -/
def jointSensitivityAt {I : Type} [Fintype I]
    (f g : (I → Bool) → Bool) (x : I → Bool) : ℕ :=
  (Finset.univ.filter fun i =>
    f (flip x {i}) ≠ f x ∧ g (flip x {i}) ≠ g x).card

/-- Maximum sensitivity over all indices and inputs with the indicated output. -/
def sideSensitivity {h : ℕ} {I : Type} [Fintype I]
    (F : Fin h → (I → Bool) → Bool) (b : Bool) : ℕ :=
  Finset.univ.sup fun q => Finset.univ.sup fun x =>
    if F q x = b then sensitivityAt (F q) x else 0

/-- Maximum same-input, same-bit joint sensitivity over two distinct ordered indices. -/
def jointSensitivity {h : ℕ} {I : Type} [Fintype I]
    (F : Fin h → (I → Bool) → Bool) (b : Bool) : ℕ :=
  Finset.univ.sup fun q => Finset.univ.sup fun q' => Finset.univ.sup fun x =>
    if q < q' ∧ F q x = b ∧ F q' x = b then
      jointSensitivityAt (F q) (F q') x else 0

@[simp] theorem flip_empty {I : Type} (x : I → Bool) : flip x ∅ = x := by
  funext i
  simp [flip]

@[simp] theorem flip_singleton_self {I : Type} (x : I → Bool) (i : I) :
    flip x {i} i = !(x i) := by
  simp [flip]

@[simp] theorem flip_flip {I : Type} (x : I → Bool) (B : Finset I) : flip (flip x B) B = x := by
  funext i
  by_cases h : i ∈ B <;> simp [flip, h]

@[simp] theorem flip_false {I : Type} (B : Finset I) (i : I) :
    flip (fun _ => false) B i = decide (i ∈ B) := by
  by_cases h : i ∈ B <;> simp [flip, h]

def blockCenter {M L : ℕ} (j : Fin M) : Fin M × Fin L → Bool :=
  fun p => decide (p.1 = j)

def baseBlock {M L : ℕ} (j : Fin M) : Finset (Fin M × Fin L) :=
  Finset.univ.filter fun p => p.1 = j

@[simp] theorem mem_baseBlock {M L : ℕ} (j : Fin M) (p : Fin M × Fin L) :
    p ∈ baseBlock j ↔ p.1 = j := by
  simp [baseBlock]

def initialFamily (d M : ℕ) : Fin (d + 1) → (Fin M × Fin (d + 2) → Bool) → Bool :=
  fun q => ballPredicate blockCenter (d - q.val)

def compose {I J : Type} (f : (I → Bool) → Bool) (g : (J → Bool) → Bool) :
    (I × J → Bool) → Bool :=
  fun x => f (fun i => g (fun j => x (i, j)))

def orBits {K : Type} [Fintype K] (x : K → Bool) : Bool := decide (∃ k, x k = true)

@[simp] theorem orBits_eq_true {K : Type} [Fintype K] (x : K → Bool) :
    orBits x = true ↔ ∃ k, x k = true := by
  simp [orBits]

@[simp] theorem orBits_eq_false {K : Type} [Fintype K] (x : K → Bool) :
    orBits x = false ↔ ∀ k, x k = false := by
  simp [orBits]

@[simp] theorem orBits_zero {K : Type} [Fintype K] : orBits (fun _ : K => false) = false := by
  simp [orBits]

def orCopies {I K : Type} [Fintype K] (f : (I → Bool) → Bool) : (K × I → Bool) → Bool :=
  compose orBits f

end Paper320

end

end OAI
"""

HEADER = "import Mathlib\n\n" + DEFS + "\n"
NS = "OAI.Paper320"

# (name inside the namespace, statement written inside the namespace, full name). Equal weight each.
_T = [
    # Warm-ups (ours).
    ("sensitivityAt_le_sensitivity",
     "∀ {I : Type} [Fintype I] (f : (I → Bool) → Bool) (x : I → Bool), sensitivityAt f x ≤ sensitivity f"),
    ("blockSensitivityAt_le_blockSensitivity",
     "∀ {I : Type} [Fintype I] (f : (I → Bool) → Bool) (x : I → Bool), blockSensitivityAt f x ≤ blockSensitivity f"),
    ("sensitivityAt_le_blockSensitivityAt",
     "∀ {I : Type} [Fintype I] (f : (I → Bool) → Bool) (x : I → Bool), sensitivityAt f x ≤ blockSensitivityAt f x"),
    ("sensitivity_le_blockSensitivity",
     "∀ {I : Type} [Fintype I] (f : (I → Bool) → Bool), sensitivity f ≤ blockSensitivity f"),
    # The mission's theorems.
    ("le_blockSensitivityAt_of_family",
     "∀ {I K : Type} [Fintype I] [Fintype K] (f : (I → Bool) → Bool) (x : I → Bool) (blocks : K → Finset I)\n"
     "    (_hne : ∀ k, (blocks k).Nonempty)\n"
     "    (_hd : Pairwise fun k l => Disjoint (blocks k) (blocks l))\n"
     "    (_hb : ∀ k, f (flip x (blocks k)) ≠ f x),\n"
     "    Fintype.card K ≤ blockSensitivityAt f x"),
    ("exists_blockSensitivityAt_family",
     "∀ {I : Type} [Fintype I] (f : (I → Bool) → Bool) (x : I → Bool),\n"
     "    ∃ blocks : Finset (Finset I), blocks.card = blockSensitivityAt f x ∧\n"
     "      (∀ A ∈ blocks, ∀ B ∈ blocks, A ≠ B → Disjoint A B) ∧\n"
     "      (∀ B ∈ blocks, B.Nonempty ∧ f (flip x B) ≠ f x)"),
    ("ballPredicate_sensitivityAt_zero_le",
     "∀ {I C : Type} [Fintype I] [Fintype C] (centers : C → I → Bool) (R : ℕ)\n"
     "    (_hsep : ∀ a b, a ≠ b → 2 * (R + 1) < hammingDist (centers a) (centers b))\n"
     "    (x : I → Bool) (_hx : ballPredicate centers R x = false),\n"
     "    sensitivityAt (ballPredicate centers R) x ≤ R + 1"),
    ("initialFamily_joint",
     "∀ (d M : ℕ) (b : Bool), jointSensitivity (initialFamily d M) b = 0"),
    ("sensitivity_orCopies_le",
     "∀ {I K : Type} [Fintype K] [Fintype I] (f : (I → Bool) → Bool) (S₀ S₁ : ℕ)\n"
     "    (_h₀ : ∀ x, f x = false → sensitivityAt f x ≤ S₀)\n"
     "    (_h₁ : ∀ x, f x = true → sensitivityAt f x ≤ S₁),\n"
     "    sensitivity (orCopies (K := K) f) ≤ max (Fintype.card K * S₀) S₁"),
    ("blockSensitivityAt_compose_zero_le",
     "∀ {I J : Type} [Fintype I] [Fintype J] (f : (I → Bool) → Bool)\n"
     "    (g : (J → Bool) → Bool) (_hg : g (fun _ => false) = false),\n"
     "    blockSensitivityAt f (fun _ => false) * blockSensitivityAt g (fun _ => false) ≤\n"
     "      blockSensitivityAt (compose f g) (fun _ => false)"),
    ("quantitative_profile_bounds",
     "∀ (d L M r : ℕ) (_hM : 0 < M)\n"
     "    (_hbudget : ((max r 16 : ℕ) : ℝ) / M * 3 ^ (d + 1) ≤ 2)\n"
     "    (S J : ℕ → Bool → ℕ)\n"
     "    (_hS₀ : S 0 false ≤ L) (_hS₁ : S 0 true ≤ L * M)\n"
     "    (_hJ₀ : J 0 false = 0) (_hJ₁ : J 0 true = 0)\n"
     "    (_hstepS₀ : ∀ n < d, S (n + 1) false ≤ 16 * S n false + S n true + 3 * J n true)\n"
     "    (_hstepS₁ : ∀ n < d, S (n + 1) true ≤ M ^ 2 * S n false + r * S n true)\n"
     "    (_hstepJ₀ : ∀ n < d, J (n + 1) false ≤ 16 * S n false + 3 * J n true)\n"
     "    (_hstepJ₁ : ∀ n < d, J (n + 1) true ≤ M ^ 2 * J n false + r * S n true),\n"
     "    ∀ n ≤ d, S n false ≤ 2 * L * M ^ n ∧ S n true ≤ 2 * L * M ^ (n + 1)"),
    ("baseline_arithmetic",
     "let M : ℕ := 9 ^ 10\n"
     "    ((2 * (9 + 2) * M ^ 10 : ℕ) : ℝ) ^ (2.000252407 : ℝ) ≤\n"
     "      ((M ^ 2 * (2 * M ^ 2 + 1) ^ 9 : ℕ) : ℝ)"),
]
TARGETS = [(name, stmt) for name, stmt in _T]
WARMUPS = 4

SKELETON = (f"namespace {NS}\nopen scoped Classical\n\n" + "\n\n".join(
    f"theorem {name} :\n    {stmt} := by\n  sorry" for name, stmt in TARGETS
) + f"\n\nend {NS}\n")

TASK = f"""Prove, in Lean 4 with Mathlib, twelve lemmas about the sensitivity and block sensitivity of Boolean functions.

For a Boolean function f on inputs x : I → Bool, the sensitivity of f at x counts the coordinates whose
flip changes f x, and the block sensitivity at x is the largest number of pairwise disjoint nonempty
blocks of coordinates such that flipping any one block changes f x. The lemmas also involve unions of
Hamming balls, an OR of disjoint copies of a function, and block composition.

These definitions are fixed and are ALREADY in front of every file you check. Do not write `import`
lines and do not restate them:

```lean
import Mathlib

{DEFS}```

Deliverable: the file `{DELIVERABLE}` in the workspace, containing these twelve theorems with exactly
these names and statements, inside `namespace {NS}`:

```lean
{SKELETON}```

How it is graded: each of the twelve theorems scores separately, so a file with some theorems proved
and others left as `sorry` still earns credit. A theorem counts only if its statement is unchanged and
its proof uses no `sorry` and no new axioms, directly or through anything it relies on: a theorem that
uses another theorem counts only once that one is fully proved too. You may add helper lemmas. The
theorems are not equally hard, and most do not depend on each other.
`{DELIVERABLE}` must always compile and must keep the `namespace {NS}` ... `end {NS}` structure: keep
`sorry` in whatever is not proved yet, because a file that does not compile scores nothing.

Use the `lean_check(path)` tool to compile any `.lean` file in the workspace. It reports errors with line
numbers. It is the only way to know whether a proof is right."""

ABOUT_HEADER = ("`import Mathlib` and the OAI.Paper320 definitions from the goal are already in front of the "
                "file, so do not import or restate them.")


def footer(targets=None) -> str:
    targets = TARGETS if targets is None else targets
    lines = ["", "-- grader", f"namespace {NS}", "open scoped Classical"]
    for i, (name, stmt) in enumerate(targets, 1):
        lines.append(f"theorem Grader.check_{i} :\n    {stmt} := _root_.{NS}.{name}")
        lines.append(f"#print axioms Grader.check_{i}")
    lines.append(f"end {NS}")
    return "\n".join(lines) + "\n"


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
