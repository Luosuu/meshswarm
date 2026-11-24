# Swarm Lessons: Modelica DAE → differentiable JAX GPU ensembles

*No successful exemplar yet. Two lesson-equipped runs failed on an untouched deliverable stub; the control run reached a module that hit the grader's wall-time. All three built references and harnesses nobody could grade against. Weigh that accordingly.*

1. **The deliverable must exist and run before anything is polished.** In two runs the graded module was never implemented — the grader died on the stub — and the third was too slow to pass. First milestone, within ~10–15 minutes: a complete, naive, correctly shaped module (any sane integrator, FP64, known-good structure) that fresh-imports and runs end to end. Ugly and green beats elegant and absent.

2. **Deliverable-owner silence is the top failure mode.** Both lesson-equipped runs: the owner went quiet after minute-one assignment; the other two agents finished refs and harness by mid-run and then idled (one support agent asleep ~35% of the time) with nothing to check. Set explicit milestone pings for the owner (skeleton → forward path → differentiable closure → gates), and once support work is done, those agents converge on the deliverable — reviewing, optimizing, running gates — not on more scratch.

3. **Gate zero is "the graded file runs."** Harnesses were written in both runs but never ticked against a real candidate. Fresh-import + forbidden-construct grep + one end-to-end graded-style call must happen at every check-in, not once at the end.

4. **Settle ownership at minute zero, once.** One run reassigned the deliverable twice in three minutes with duplicate task postings. If a reassignment is unavoidable, do it in a single message everyone acknowledges; never leave two owners even briefly ambiguous.

5. **Single writer plus independent verifiers works — keep it.** All three runs got 12–28 reference cases (including off-example regimes) generated early and cleanly, and an independent scipy re-implementation matched them to ~1e-10/1e-9 before any JAX existed. That prototype de-risks DAE semantics more cheaply than debate; settle disagreements by opening a reference CSV. Don't let reference grids grow past what the solver consumes — ~10–15 regime-spanning cases steer implementation fine.

6. **Verify artifacts after writing them.** The control lost time to a reference CSV that vanished. Re-read every file another agent depends on and record its path in a manifest before building on it.

7. **The graded module must stay pure.** Never shell out to omc or anything else from it; references live in scratch, generated via run_command. Unrefuted by these runs.

8. **Gates in order: parity → conservation → gradient consistency → speed ladder.** Tick the checklist only on fresh runs. Integrator choice stays empirical — adaptive stiff schemes have outperformed fixed-step ones — but no run completed a solver, so treat that as provisional, and remember the grader itself has wall-time expectations at every stage.

## Untested (experiments)

- **Gate-per-agent split**: assign each gate end-to-end to a different agent — still untried.
- **Red-team finale**: last 30 minutes, one agent only breaks the deliverable — still untried.
- **Support-agent convergence**: after refs/harness land, both support agents pair with the deliverable owner (review + optimization + gate runs) instead of opening any new thread. The runs show they otherwise idle.