# Swarm Lessons: Modelica DAE → differentiable JAX GPU ensembles

*Both runs scored 0.0/10; there is no successful exemplar yet. These are failure-derived lessons, weighted equally.*

1. **The graded module must be pure.** Run-1 was zeroed solely because `solver_jax.py` contained a `subprocess` call — nobody checked the deliverable itself before time expired. Never shell out to omc or anything else from the module you submit; references belong in scratch directories, generated via `run_command`. Endgame habit: fresh-import the deliverable, grep it for forbidden constructs, and run the grader's gates yourself.

2. **Verify against regime breadth, not the visible example sets.** Run-2 passed its own self-chosen cases yet failed a held-out regime-transition case (rel err ~2.6e-2 vs gate 2.5e-4). Whatever parity grid you build, deliberately include awkward regimes — sign changes in drives, near-threshold branches, extreme inputs — and require *all* to pass before touching speed. Both tasks that timed out over-fitted to the examples they were shown.

3. **Gates in order; correctness before speed.** Neither run achieved a single green grader pass; both hit the 3600s wall mid-verification. Work parity → conservation → gradient consistency → speed ladder. Freeze features once close, and spend the final 20–30 minutes on one grader-like end-to-end run instead of new experiments.

4. **One writer per artifact; no claim/release churn.** Run-2 burned early minutes on duplicate drafts and task claims released within seconds. Run-1 let one agent hold everything, hub_share 1.0, with nobody checking the file that got graded. Assign `solver_jax.py` to exactly one owner; others own reference generation and the harness independently; nobody else edits the deliverable.

5. **Nail the reference pipeline first, once.** Both swarms lost long stretches to the same omc quirks: plain `simulate(...)` with inline parameter overrides silently emits no CSV (or errors), and CSVs land in the process working directory, not the path you requested. Fix one working recipe on two parameter sets — per-case model copies with rewritten defaults, or plain simulate plus rename — decide on a single shared refs directory with recorded paths, then scale the grid.

6. **Resolve model-semantics disputes by reading reference output, not debate.** Run-2 lost hours and surfaced a sign error in a derived drive term only late. When two agents disagree on what the model does, open a reference CSV and settle it immediately.

7. **Integrator choice is empirical.** In both runs, fixed-step Rosenbrock attempts NaN'd or exploded on stiff/regime-transition cases while adaptive schemes tracked references to ~1e-9. "Works on the easy case" is no evidence; gate any integrator on the full regime grid.

8. **Keep a written gate checklist.** Both failures (forbidden construct; held-out parity) are exactly the class a ticked-off grader list would have caught. Write the gates into a scratch file and tick them only with a fresh run.

## Untested (experiments)

- **Gate-per-agent split**: assign each grader gate end-to-end to a different agent (fresh eyes, no shared assumptions) rather than splitting by pipeline component.
- **Red-team finale**: one agent whose only job in the last 30 minutes is to break the deliverable — fresh import, forbidden-construct grep, unseen-regime sweep.
- **Dual-solver cross-check**: two independent solver implementations whose outputs must agree; would catch subtle semantic bugs like run-2's late sign error. Costly, unproven.