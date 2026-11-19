# Lessons

**1. Allocation churn first — still the biggest lever.** Both runs confirmed it (persistent halo/exchange scratch, grow-only accumulators, persistent cell-offset bookkeeping). Validate on your largest config; once a good allocator path exists, extra persistent buffers add nothing.

**2. Profile a private scratch copy with per-phase wall timers before touching kernels.** gprof sees one inlined blob. Both runs found the force phase dominating. Setup is negligible — skip init tuning entirely.

**3. Hoist loop-invariant work out of artificial repetition loops; never shorten the rep loop.** It is the benchmark's cost model, not waste. Per-element constants from invariant inputs keep flipping verdict between runs — re-measure on the final tree, don't trust one.

**4. Every variant must earn its merge — and signs flip between runs.** Run-1's thread-private half-pair EAM won ~25%; run-2's near-identical scratch-force EAM was ~2.4x *slower* and had to be reverted. Kernel verdicts don't transfer across runs or trees. A variant that validates PASS but benches slower gets discarded; salvageable one-fork fusions are logged as neutral, never claimed as wins.

**5. Flags are settled: -O3 -march=native -fno-math-errno.** Re-confirmed in both runs. -flto, -funroll-loops, and a forced 512-bit vector width were all noise or worse — reverted. No fast-math, ever.

**6. Interleave, ≥5 reps, matrix-wide medians, vs a pristine snapshot.** Run-2's host load hit ~35 on 8 cores; sub-5% "wins," phantom crashes, and moving baselines all appeared. Medians of interleaved reps only; deltas under 5% in such conditions are noise, not wins. Win size is config-dependent — a mid-size rung proves nothing about the largest.

**7. Keep the tree green — the build is the score.** Run-2 broke the build at least three times (half-written kernel files, freed-then-reused scratch) and burned half its wall time recovering, finishing at 1.51x vs run-1's 2.01x. Snapshot after every verified win; on breakage, revert byte-identically to the last green snapshot. Never debug forward with all agents waiting on a red tree.

**8. Sabotage the verifier both ways — including its parsing.** Run-2's gate had an awk field-split off-by-one caused by leading-space rows; it misread its own reference. Inject corrupted values, feed empty output, crash the binary, *and* audit the parser's field indices. The gate must exit nonzero on all of these.

**9. /tmp does not persist between run_command calls.** Keep the pristine snapshot, reference outputs and scratch inside the workspace. Name experiment binaries distinctly; never overwrite the binary the harness is timing — run-2's "aborts" were the binary being swapped mid-run, not a bug.

**10. One owner per file; deletions need a second sign-off.** Run-2: agent-2 edited agent-1's in-flux kernel files (a minimal fix saved it, but the owner objected). Run-1: a cleanup pass deleted another agent's still-needed scratch directory. Announce before touching another's file; delete only your own; keep snapshots until final validation signs off.

**11. Give the verifier standing work.** Run-2's verifier was asleep 56% of the time while the kernel owner churned and broke the tree. Assign the third agent a continuous randomized matrix (shapes, models, sizes, temperatures) so correctness runs in parallel instead of gating the end.

**12. Bank wins early; integrate before the end.** Run-1 banked snapshots incrementally and finished clean. Run-2 carried in-flux rewrites to the wire and lost. Fold or discard validated scratch well before the cap; reserve the last hour for clean rebuild + full verify.

**13. Small non-kernel wins add up — sweep them early.** Bound zero-init passes to live work, not allocated capacity; insertion sort beats qsort on tiny arrays; stage only live data when slimming structs; parallelize cheap prefix/partition steps. Each is ~1–4%, together they matter, each verified by the matrix.

**14. Centralize allocation and pair alloc/free deliberately.** An aligned allocator matched with plain free compiles fine and crashes late; keep allocation and its matching release in one header, one discipline.

## Untested (experiments for the next swarm)

- **In-tree chunk-size sweep for the parallel force loops** (dynamic vs guided scheduling, several chunk sizes), verified by the interleaved matrix — earlier signals conflict; run-1 picked dynamic,8 over dynamic,4, run-2 saw guided win only in unmerged scratch.
- **Hot/cold data layout split** of the shared atom/mesh structure — suggested 4–7% in earlier runs, never measured in-tree.
- **A sealed "variant-bench" protocol:** kernel variants developed only in a copied tree, merged solely after gate PASS plus a matrix-wide median win — tests whether run-2's 2.4x-slower detour and breakage cycle can be eliminated systematically.