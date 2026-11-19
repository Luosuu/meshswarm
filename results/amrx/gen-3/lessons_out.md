# Lessons for swarm-driven CPU optimization of scientific code

**Settle roles by rule, not negotiation.** Both runs knew the kickoff lesson and still had all three agents propose splits at once, then re-settled the same split three-to-five times each; "FINAL — no more re-proposals" was declared repeatedly and never held. Mechanical fix: first board claim on the canonical-write task takes the pen; everyone else gets one objection or one accept; counter-proposals are void. The first minute of message churn is pure loss. (The better run outscored the weaker one on this, and on timing hygiene — not on a cleverer split.)

**One writer; the board is the lock; a duplicated read-only reference is fine.** Both runs held this once settled: one agent edits, a pristine serial `-O3` reference sits beside the work, and a clean-tree rebuild check closes the run. Keep it.

**Reference + matrix before optimizing; gate every merge, large grids included.** Both runs built 12–22 argument sets, edge cases (degenerate levels, boundary periodicities, awkward sizes) covering rare paths. The largest cases diverged worst — still comfortably under tolerance — so small-set or example-only agreement is never evidence. Re-verify after every edit and again after freeze.

**Fix the timing protocol before timing anything.** Time the binary directly, never wrapped in a subprocess harness: one run lost hours to ~0.3s per-call wrapper overhead and chased a "startup cost" that a tiny-case run disproved. Interleave A/B, medians of 5+, and expect noisy neighbors — reference times spiked 60%+ within sessions, so reorder and repeat. Phase sums short of wall time? Suspect the harness before the program.

**Aim the idle agents — they are half your capacity.** Agents idled 50–81% of the time in both runs. Best uses found: a memory-bandwidth probe (sweeps are cache/bandwidth-bound), a huge-page test (no gain, dropped), a tiny-case startup check, edge-case matrices. All bounded scratch, no canonical edits.

**After loops are parallel, wins come from fewer memory passes.** Both runs converged independently on the same shape — parallel sweeps, fused update passes, unrolled inner loops with register accumulators — plateauing near the bandwidth ceiling (~6–8x held-out on 8 cores). When bandwidth is the ceiling, more threads and wider unrolls stop paying. Measure it, then stop.

**A/B each flag separately; reject neutral ones out loud.** Portable vector ISA level over native when equal; 512-bit width hint neutral and rejected; extra unroll/LTO within noise and were not stacked. Decide; never stack speculative wins.

**Freeze means freeze.** Both runs did a clean rebuild plus full matrix at the end and held. After any late change, re-run everything; list and announce targets before any rm.

**Sub-agents untried here.** Zero spawned in either run; the earlier bounded-subagent lesson remains theory.

## Untested (experiments, not advice)
- **Board-rule role assignment** — first claim wins the pen, one objection allowed; does it kill the multi-round split negotiation?
- **Verify-on-save watcher** — auto-run the matrix after each canonical edit; would it catch rare-path bugs before review?
- **Load-aware benchmarking** — refuse to report timing claims under external load; both runs found box noise only ad hoc and wasted cycles on it.