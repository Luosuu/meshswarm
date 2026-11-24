# Swarm Lessons: Modelica DAE → differentiable JAX GPU ensembles

*Run-1 (5.0/10) is the only gates-passing run, so correctness evidence rests on one run; it stalled at the wall-time ladder (1.6s total → only rungs ≥2s beaten). The two 0.0 runs failed identically: a stub deliverable while everything else finished. Refs-early is now solved and routine.*

1. **Deliverable first, with hard checkpoints.** Run-2's owner reported a prototype "passes DASSL parity on all 26 refs" at ~1975s; at grading, `solver_jax.py` still raised `NotImplementedError`. Meanwhile refs landed by ~190s and an independent cross-check by ~356s — *only* the deliverable failed. New rule beyond prior lessons: fresh-import `solver_jax.py` and run the graded call at fixed checkpoints (25/50/75% of budget). If it doesn't execute, sole write access transfers automatically to the peer holding the best prototype or paste-ready derivation.

2. **Port on pass, not at the end.** The moment a scratch prototype passes parity, copy it into the deliverable and re-run the gate there. No prototype survives more than ~15 min post-pass. Better still, prototype directly in `solver_jax.py` so "porting" never exists. Control shows the failure's endpoint: drafts in `solver_jax_draft.py`, a winning backend benchmark at 3481s — nothing ever became the module.

3. **Heartbeat the deliverable — voluntarily it never happens (0 lean checks in all three runs).** Automate it: a watcher that every 10–15 min fresh-imports the module, runs `forward_ensemble` on the default case plus one `jax.grad` call, and posts the raw output.

4. **Cross-checks by non-writers are cheap and proven — if early.** Run-2's independent scipy Radau/LSODA check (26 refs, <2.2e-9) landed by ~350s without touching the writer; it was the most useful verification produced. Control delegated a prototype at ~3485s — late delegation yields nothing. Delegate early or not at all; one owner per check.

5. **After gates pass, the rest of the budget goes to the ladder.** Run-1 stopped at correctness and scored 5/10. Parity margin was ~1000× (1.8e-7 vs 2.5e-4 gate), so tolerance/step tuning and backend benchmarking buy rungs. Own a grader-verbatim stopwatch (median of 5×1000-trajectory `forward_ensemble` + 64-trajectory grad) from ~50% of budget and iterate against it, not against proxies. Control's late benchmark: a Rosenbrock backend hit ~1.5s at rtol=1e-8; run-1's fixed-step RK4 also passed (~1.6s) — integrator choice is empirical; measure both.

6. **Keep from before:** pure module (no subprocess/omc/`stop_gradient`; fresh import + grep); regime-breadth parity grids, not just shown examples; gates in order, correctness before speed; sole writer per artifact (now with checkpoint transfer); settle semantics by reading a reference CSV; don't gold-plate the harness; conservation is free for linear invariants (~5e-15 measured) — verify, don't project; idle agents, when blocked >10 min, prepare the owner's last error as ready-to-paste fix code rather than sleep (run-2's agent-1 slept 83% yet its early delegation helped — re-aim, don't just delegate).

## Untested (experiments)

- **Vertical-slice hour**: first 60 min end with one case green through parity → conservation → grad → speed, before any grid construction (run-2 attempted this obliquely; not cleanly tested — retry with checkpoint transfer).
- **Checkpoint ownership transfer**: at 25/50/75% of budget the deliverable is fresh-imported; if broken, write access passes automatically to the most-prepared peer.
- **Red-team finale**: last 30 min, one agent may only break the deliverable — fresh import, forbidden-construct grep, unseen-regime sweep, grader-metric stopwatch.