# Swarm Lessons: Modelica DAE → differentiable JAX GPU ensembles

*Three more runs, all 0.0/10. Still no successful exemplar. This round's failure mode: parallel infrastructure finished; the deliverable stalled or was never executed before grading.*

1. **Deliverable first; infrastructure around it.** In run-2, references landed early, the harness at ~25 min, the cross-check at ~47 min — but `solver_jax.py` shipped as an unimplemented stub and the grader died on `NotImplementedError`. In run-1 the module existed but nobody ran it; the grader hit a 240 s timeout. Phase the run: a minimal complete module (fresh import, gate 1 green on the default case) before any grid-building, then iterate.

2. **Heartbeat the actual deliverable.** All runs logged zero lean checks; none executed `solver_jax.py` fresh until grading. Assign someone to, on a fixed interval, fresh-import the module and run the smallest forward + grad step, and post the output — not intentions or claims.

3. **One writer holds, but the critical path needs watching.** Runs with these lessons had zero handovers (control churned a claim and handover) — assignment discipline worked. Yet the deliverable owner stalled (run-2) or produced an unmeasured, too-slow module (run-1) while peers finished everything else. If the writer is blocked >10 min, attack the blocker: derive the hard sub-step in scratch, hand over ready-to-paste code — keep sole write access to the file.

4. **Idle agents aim at the smallest failing gate.** Two runs had agents asleep 46–51% while the deliverable was unfinished. More reference cases don't move a stalled critical path; take the owner's last error and prepare the fix.

5. **Time-box prototyping.** Run-2's scratch prototype and control's solver explorations never reached the module. Every prototype gets port-or-kill by a stated time; the winner enters the deliverable or dies.

6. **Cross-checks must land by mid-run.** Run-2's independent LSODA/Radau-vs-DASSL check passed 24 cases — at ~78% of budget, with nothing left to fix. Dual-solver semantics verification is worth doing; finish it by the halfway mark or run it against the skeleton.

7. **Measure the graded workload, not proxies.** No run timed the grader's actual cost: median of 5 × 1000-trajectory `forward_ensemble` plus the 64-trajectory grad pass. Run-1 lost exactly there. The harness owner should own a stopwatch script mirroring the grader verbatim, first run at ~50% of budget.

8. **Keep from before:** pure module (no subprocess/omc; fresh-import + grep); regime-breadth parity grids, not the shown examples; gates in order, correctness before speed; sole writer per artifact; references generated early (now routine — treat as solved); settle semantics disputes by reading a reference CSV; integrator choice empirical; gate checklist ticked only by a fresh run.

9. **Don't gold-plate verification.** Harnesses were rewritten/extended repeatedly while solvers waited. A rough gate run every 15 minutes beats a perfect harness built once.

## Untested (experiments)

- **Vertical-slice hour**: first 60 minutes end with one case green through parity → conservation → grad → speed, before any grid construction.
- **Gate-per-agent split**: one owner per grader gate end-to-end, fresh eyes, no shared assumptions (retry).
- **Red-team finale**: last 30 minutes, one agent may only break the deliverable — fresh import, forbidden-construct grep, unseen-regime sweep, grader-metric stopwatch.