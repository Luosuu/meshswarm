# Modelica-to-JAX: Multi-Generation Swarm Evolution & Cross-Task Transfer

Twenty swarms of three `Inkling-2-Small` agents each were asked to turn three engineering Modelica
models (`.mo`)—all of which fail direct `Rumoca.to_jax()` compilation—into verified,
conservation-preserving, differentiable JAX GPU ensemble solvers. On a single-GPU re-grade
(NVIDIA A100-SXM4-40GB), **ten of twenty swarms passed all four multiplicative correctness gates**
(including **6/6** on ORNL's `TRANSFORM` counter-current `PbLi`/`FLiBe` heat exchanger and **3/3**
in Generation 2 of the 16-stage cryogenic $\text{H}_2/\text{HD}/\text{T}_2$ distillation column),
reaching **`0.0078 s` (`7.8 ms`) for 1,000 forward trajectories + `0.0175 s` for 64 reverse-mode
gradients (`10.0/10`)** on Tier 2 and **`0.0428 s` (`42.8 ms`) for 1,000 forward trajectories
(`8.0/10`)** on Tier 3.

---

## 1. Setup

- **Benchmark tiers (`examples/modelica_jax/tasks.py`):**
  1. **`tier1_blanket_dae` (`BlanketValveDAE.mo`, 6 states + 1 implicit algebraic equation):**
     fast primary hydraulics ($\tau_{\text{hyd}} = 0.02\text{ s}$) with piecewise check-valve drag
     and orifice bypass (`if/else` events) coupled to blanket/HX thermal inertia and an implicit
     cubic-Sieverts tritium permeation equation.
  2. **`tier2_transform_hx` (`TransformPbLiFLiBeHX.mo`, 15 states + 1 implicit algebraic equation):**
     instantiates ORNL's `TRANSFORM.HeatExchangers.Simple_HX` (`nV=6, counterCurrent=true`) between
     liquid `PbLi` (`cv=141 != cp=188`) and molten `FLiBe` (`cv=1785 != cp=2380`), with low-flow
     limiters and 6-zone Arrhenius wall tritium permeation.
  3. **`tier3_hdt_column` (`CryoHDTColumn16.mo`, 32 states + 16 implicit algebraic equations):**
     16-stage closed-recycle cryogenic $\text{H}_2/\text{HD}/\text{T}_2$ distillation MESH column
     with per-stage nonlinear Antoine bubble-point equations and hydraulic flooding above
     `V_flood = 2.15 mol/s`.
- **Multiplicative zero-multiplier gates (`examples/modelica_jax/grader.py`):**
  1. Pure FP64 GPU execution with AST-verified absence of `subprocess`/`omc`/`stop_gradient`;
  2. Trajectory parity on 8 unseen parameter sets against OpenModelica 1.27.1 DASSL (`tol=1e-10`);
  3. Closed-loop molar inventory drift `< 1e-11` across 1,000 random trajectories;
  4. Reverse-mode `jax.grad` matching `jax.jvp` (`< 1e-10`), closure central FD (`< 5e-5`), and
     fresh forward macro FD (`< 1.5e-2`);
  5. Wall-time rungs (`30, 15, 8, 4, 2, 1, 0.5, 0.25, 0.12, 0.06` seconds) on `(median of 5 × 1,000-trajectory forward) + (median of 3 × 64-trajectory jax.grad)`.
- **Swarm & campaign DAG:** 3 peer agents (`thinkingmachines/Inkling-2-Small:peft:262144`), $3.00
  cap and 60-minute wall limit per swarm (`--parallel` on 1 GPU per Slurm job, network-cut
  Apptainer container). Generation 1 of Tier 1 (`2` unguided swarms) produced the seed lessons
  `tier1-g1/lessons.md`, which fanned out simultaneously into:
  - Tier 1 Generations 2 & 3 (`2` lesson swarms + `1` unguided `--controls 1` swarm each);
  - Tier 2 Generations 1 & 2 (cross-task transfer to ORNL `TRANSFORM`, `2` lesson + `1` control each);
  - Tier 3 Generations 1 & 2 (cross-domain transfer to cryogenic distillation, `2` lesson + `1` control each).

---

## 2. Result of Record (Single-Node NVIDIA A100 Re-Grade)

Because in-job runs landed across heterogeneous GPUs (`A40`, `RTX A6000`, `A100-80GB`) whose FP64
throughput differs by over an order of magnitude, all 20 workspaces were re-graded sequentially in
one job on a single **NVIDIA A100-SXM4-40GB** (`udc-an28-1`, job `23388961`; raw output in
`regrade-one-a100.txt`). Total LLM spend across all 20 swarms was **$9.69**.

### Tier 1: `BlanketValveDAE` (3 Generations)

| Gen | `run-1` (lessons) | `run-2` (lessons) | `control` (no lessons) | Lesson Mean | Cost (total) | Notes (A100 re-grade) |
|---|---|---|---|---|---|---|
| 1 | 0.0 *(no lessons)* | 0.0 *(no lessons)* | — | — (0.0 unguided) | $0.90 | `run-1`: check-valve kink broke closure FD (`9.1e-4`); `run-2`: case-3 sign error (`2.6e-2`) |
| 2 | 0.0 | 0.0 | 0.0 | 0.0 | $0.56 | `run-1` & `control` timed out (>240s adaptive solver); `run-2` left stub unmerged |
| 3 | **6.0** (`0.843 s`) | 0.0 | 0.0 | **3.0** (vs 0.0 ctrl) | $0.61 | `run-1`: `1k-fwd=0.216s + 64-grad=0.627s`, `omc_err=3.62e-8`, `mass_err=7.99e-15` |

### Tier 2: `TransformPbLiFLiBeHX` (Cross-Task Transfer from Tier 1 + Refinement)

| Gen | `run-1` (lessons) | `run-2` (lessons) | `control` (no lessons) | Lesson Mean | Cost (total) | Notes (A100 re-grade) |
|---|---|---|---|---|---|---|
| 1 | **4.0** (`3.434 s`) | **9.0** (`0.107 s`) | **5.0** (`1.443 s`) | **6.50** (vs 5.00 ctrl) | $2.32 | **3/3 passed all 4 gates**; `run-2` hit `41.3 ms` 1k-fwd via 150-step RK4 + `@jax.custom_jvp` |
| 2 | **10.0** (`0.025 s`) | **5.0** (`1.329 s`) | **7.0** (`0.263 s`) | **7.50** (vs 7.00 ctrl) | $2.92 | **3/3 passed all 4 gates**; `run-1` hit **`7.8 ms` 1k-fwd + `17.5 ms` 64-grad = `25.3 ms`** |

### Tier 3: `CryoHDTColumn16` (Cross-Domain Transfer to 16-Stage Distillation + Refinement)

| Gen | `run-1` (lessons) | `run-2` (lessons) | `control` (no lessons) | Lesson Mean | Cost (total) | Notes (A100 re-grade) |
|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.0 | 0.0 | 0.0 | $0.79 | `run-1`/`run-2` left prototypes in `scratch/`; `control` used `@jax.custom_vjp` (broke `jvp`) |
| 2 | **2.0** (`8.045 s`) | **5.0** (`1.863 s`) | **8.0** (`0.175 s`) | **3.50** (vs 8.00 ctrl) | $1.00 | **3/3 passed all 4 gates** (`omc_err = 1.5e-10..5.2e-10`, `mass_err <= 3.9e-15`) |

---

## 3. What the Swarms Discovered

1. **Exact Modelica package flattening & subtle media thermodynamics (`Tier 2`):**
   Every one of the six Tier-2 swarms used `omc instantiateModel` to unpack
   `TRANSFORM.HeatExchangers.Simple_HX`, correctly distinguishing `PartialSimpleMedium`'s internal
   energy capacity (`u = cv_const * (T - T0)`, `cv = 141` for PbLi and `1785` for FLiBe) from its
   advective enthalpy (`h = cp_const * (T - T0)`, `cp = 188` and `2380`), as well as extracting
   `Modelica.Fluid.Utilities.regStep` and counter-current zone indexing (`i <-> 7 - i`).
2. **Eliminating algebraic loops with `@jax.custom_jvp` Implicit Function Theorem:**
   Across all three tiers, passing swarms solved the monotone cubic-Sieverts permeation root (Tiers
   1 & 2) and the 16-stage cryogenic Antoine bubble-point temperatures (Tier 3) inside RHS
   evaluations via fixed-iteration Newton (`lax.fori_loop` / `lax.scan`) wrapped in
   `@jax.custom_jvp`—avoiding both unrolled Newton AD tape bloat and the `@jax.custom_vjp`
   forward-mode `jvp` failure that felled `tier3/gen-1/control`.
3. **Quantified step-size reduction on GPU (`165×` faster than our hand-written Tier-2 reference):**
   Our hand-written Tier-2 reference used adaptive `rodas5p_replay` (`4.18 s` on A100, `3.0/10`).
   `tier2/gen-2/run-1` measured the actual thermal eigenvalue margin against OpenModelica DASSL on
   a 12+ case regime grid, replaced adaptive Rosenbrock with a shared module-level
   `jax.jit(jax.vmap(lax.scan))` 100-step RK4 kernel (`omc_err = 1.63e-7` vs `3e-4` gate), and cut
   combined runtime to **`25.3 ms` (`10.0/10`)**. Swarms on Tier 3 also independently identified
   warm-starting stage bubble-point Newton iterations from the previous time step.

---

## 4. How Collaboration & Lessons Evolved

- **Eliminating the "unmerged scratch prototype" failure mode:** In `tier1/gen-2` and `tier3/gen-1`,
  swarms built thorough OpenModelica reference grids (12–28 cases) and independent SciPy Radau/LSODA
  cross-checks, but ran out of the 60-minute clock before porting prototypes from `scratch/` into
  `solver_jax.py`. In both tracks, the Reviewer diagnosed *"Deliverable first; ugly and green within
  10–15 minutes beats elegant and absent"*, after which `tier3/gen-2` jumped from `0/3` to **`3/3`
  complete, gate-passing deliverables** and `tier1/gen-3` produced its `6.0/10` solver.
- **Cross-task transfer worked immediately on Tier 2:** Seeded only with failure-derived lessons
  from `tier1/gen-1` (build a regime-spanning `omc` reference grid first, settle sign/pairing
  disputes from CSVs rather than debate, check gates in order), both lesson swarms in `tier2/gen-1`
  passed every gate on their first try (`6.50` mean vs `5.00` control), and `tier2/gen-2` rose to
  `7.50` with a `10.0/10` peak.
- **When chatter hurt speed (`tier3/gen-2`):** While lessons lifted Tier-3 gate completion to 100%,
  the two lesson-equipped swarms in `tier3/gen-2` exchanged 27–37 peer messages and left conservative
  step/JIT choices (`2000` steps or per-call `jax.jit` wrapper, scoring `2.0` and `5.0`), whereas
  the single `control` swarm sent 15 messages (`hub_share = 0.93`), picked `500` steps with a
  module-level jitted kernel, and scored `8.0/10` (`0.175 s`).

---

## 5. Caveats

- **Three swarms shared one GPU lock per job under a 60-minute cap.** In `tier1` and `tier3`, slow
  200–500 s XLA compilations by one swarm delayed `run_command` calls in the other two swarms
  sharing the GPU, contributing to unmerged prototypes at the 3600 s cutoff (swarms spent only
  `$0.18–$0.35` of their `$3.00` cap on Tiers 1 & 3). Allocating one GPU per parallel swarm
  (`--gres=gpu:3`) or a 90–120 minute wall limit removes this contention.
- **Small per-generation sample (`2` lesson + `1` control).** Trends in gate reliability (`0/3 -> 3/3`
  on Tier 3, `6/6` on Tier 2) and peak speed (`10.0/10`) are clear, while single-control comparisons
  in any one generation carry run-to-run variance.

---

## 6. Files

| Path | Contents |
|---|---|
| `regrade-one-a100.txt` | Sequential single-A100 re-grade of all 20 workspaces (job `23388961`) |
| `<tier>/gen-N/scores.json` | In-job scores and collaboration metrics (`hub_share`, `asleep_pct`, `peer_messages`) |
| `<tier>/gen-N/lessons_in.md`, `lessons_out.md` | Input lessons and the outside Reviewer's distilled rewrite |
| `<tier>/gen-N/summary.md` | Per-generation summary table |
| `<tier>/gen-N/<run>/` | `solver_jax.py`, `changes.diff` against starter skeleton, `swarm-report.md` |
