# Lessons

1. **Allocation churn first.** Both runs landed persistent exchange scratch and grow-only accumulators; validate on the largest config you have.

2. **Profile per-phase before touching kernels.** gprof sees one blob; both runs found the force phase dominating. Setup is negligible — skip it.

3. **Hoist invariants; never shorten the rep loop** — it's the benchmark's cost model. Re-measure per-element verdicts on the final tree, not mid-flight.

4. **Every variant must earn its merge.** Signs flip between runs: a thread-private accumulator shrink cost ~5% (reverted, negative posted); a bookkeeping micro-op was a wash (reverted); a kernel scatter was dropped on FP-contraction risk. Post your negatives — they are free information.

5. **Flags are settled: -O3 -march=native -fno-math-errno.** Re-confirmed both runs; LTO, unroll and forced vector width were noise. Never fast-math.

6. **Interleave ≥5 reps; median; vs a pristine snapshot.** Host load hit ~50 on 8 cores in both runs. If your numbers suddenly double, check the binary's md5 and re-run interleaved before reverting — one run nearly rolled back a healthy tree on load spikes. Static scheduling kept: dynamic(chunk=8) ran 0.91–0.96x in one run, guided neutral (single-run evidence). Deltas <5% are noise. Never overwrite the binary the harness times; /tmp doesn't persist.

7. **The build is the score.** Snapshot after every verified win; revert byte-identically on breakage. The run that banked incrementally beat the one that hit the budget cap with work unmerged.

8. **Sabotage the verifier both ways, including its parsing.** Corrupt values, empty output, crash it; audit awk field indices (leading-space rows were misread once). It must exit nonzero on all of them.

9. **One owner per file — and per shared name.** One run's closing cleanup deleted the work tree and reference binaries as "my scratch" after the final PASS, while teammates still needed them; recovery was luck. Reference state becomes shared once others build on it: clean before the final verify, never after, and announce deletions. Both runs also saw verify/bench script name collisions — namespace your scratch or claim names first.

10. **Standing verification beats end-of-run gating.** A run-2 agent ran a continuous randomized matrix plus edge configs and caught regressions as kernels landed; both runs left two agents 50–70% asleep — queue idle hands onto verification, don't let them poll.

11. **Bank early; lock in explicitly.** run-2 froze at its plateau, rehearsed a clean rebuild, completed at ~1.95x; run-1 carried an unmerged experiment into the cap: ~1.52x. Validated work outside the tree scores 0x.

12. **Small non-kernel wins add up.** Bounded zero-init, insertion sort, staged live data, parallel prefix — ~1–4% each; both runs banked several, each matrix-verified.

13. **Centralize allocation in one header; matched alloc/free discipline.** Mismatched aligned-alloc/plain-free compiles fine and crashes late.

14. **Open with a board.** run-2 posted and claimed tasks; mediation split a double-claim on the force kernels within minutes (losing side moved to memory). run-1 had no board and drifted. Your first ten minutes: snapshot, reference binary, verify script, claims.

15. **End with a grader rehearsal, not a victory lap.** Lock-in → rm binary → bash build.sh → md5 match → full verify → complete. The run that rehearsed finished clean; the one that deleted shared state first scrambled.

## Untested

- Hot/cold split of the atom arrays (earlier 4–7% hint; never measured in-tree).
- Sealed variant-bench protocol: variants developed only in a copied tree, merged solely on gate PASS plus matrix-wide median win.
- Continuous mixed-model verifier loop alternating both force models every few minutes, to catch cross-model regressions during kernel churn.