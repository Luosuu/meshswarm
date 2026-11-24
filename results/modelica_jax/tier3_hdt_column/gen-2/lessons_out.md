# Swarm Lessons: Modelica DAE → differentiable JAX GPU ensembles

*All three new runs passed all four correctness gates — the old "stub deliverable" failure is gone. Scores were decided by wall-time: control 0.80s (6/10), run-2 4.63s (3/10), run-1 9.58s (2/10). Caveat: n=1 control, but the two lesson-equipped runs talked much more (27/37 messages vs 15) and still lost on speed. Weigh accordingly.*

1. **Deliverable exists early — keep.** All three runs had a complete `solver_jax.py` within ~10 minutes and passed gates. This lesson worked; retain it.

2. **Gates are solved; the speed ladder is the score.** Once gates pass (early in all three runs), stop re-verifying and spend the rest cutting wall-time: fewer integrator steps, warm-started implicit solves, lean vmap+scan+jit shapes. Timebox the gate checklist. Speed is roughly one point per rung — verification past "gates pass" earns nothing.

3. **Solver design beats late polish.** Control used ~500 RK4 steps and hit ~0.34s forward; run-2 sat on 2000 steps with 8-iteration Newton at the 1-hour mark and never recovered. Decide step count and implicit-solve iterations from prototype stability measurements early, not in the last hour. Measured warm-start Newton (carry stage temperatures in the scan state, 1–2 iterations) gave ~2x speedup at ~1e-9 max error — far inside the 3.5e-4 parity tolerance; use it instead of full re-solves.

4. **omc fails silently and burns budgets.** "simulate succeeded" with no CSV produced cost run-1 ~25 minutes and recurred in run-2. After every omc call, immediately confirm the output CSV exists and is non-trivial; use explicit `fileNamePrefix` and absolute paths; keep a manifest; re-read any reference another agent depends on before building on it. Refs should land by minute ~20.

5. **Single writer + refs + harness split held — keep.** Duplicate task postings (run-2 had two overlapping reference/harness tasks) cost minutes of dedup messaging. Settle the split in one message, claim explicitly, close duplicates at once.

6. **Prototype findings must reach the deliverable same-hour.** run-2's fast prototype measured ~1.09s forward, but the deliverable kept its slow settings and the gap never closed. Once a prototype clears DAE semantics, pair its author with the deliverable owner immediately and land its measurements as concrete changes.

7. **Keep coordination lean.** Control won with 15 messages and a 0.93 hub share; the lesson runs spent extra messages on alignment and dedup that never became speed. Agree roles once, then go quiet until a gate result or a blocker.

8. **Gate order and integrator choice — unchanged, provisional.** Parity → conservation → gradient consistency → speed, ticked on fresh runs. Fixed-step RK4 under scan, vmapped, outperformed adaptive stiff schemes for ensemble wall-time in all three runs; adaptive Rosenbrock differentiates through per-step Jacobians. Still provisional — no run tried Newton-Krylov or stage-parallel integrators.

## Untested (experiments)

- **Ladder sprint milestone**: once gates pass, all agents spend the final ~45 minutes only on the 1000-trajectory + 64-grad benchmark loop, cutting steps/iterations — untried as an explicit phase.
- **Prototype-owner pairing from minute zero**: hard-wire the prototype author to shadow the deliverable owner from the first benchmark — untried.
- **Red-team finale**: last 30 minutes, one agent only breaks the deliverable — still untried.