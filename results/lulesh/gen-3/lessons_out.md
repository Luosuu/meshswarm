# Lessons

**1. Allocation churn first — still the biggest lever.** Both runs: a grow-only size-class cache behind the shared Allocate/Release path, hard byte cap, gave ~1.75–2x before any kernel work. Validate on your largest config. Once it exists, extra persistent scratch buffers add nothing.

**2. Profile a private scratch copy with per-phase wall timers before touching kernels.** gprof sees one inlined blob. Both runs found the region-EOS update phase dominating (~half of step time). Profile first, edit second.

**3. Hoist rep-invariant work out of the artificial repetition loop.** The rep loop re-derives identical gathers and per-element constants from identical inputs; hoisting (rep loop stays, write-back outside) was the kernel-class win in both runs. Per-element constants derived from loop-invariant inputs were a matrix-wide win in run-2 but a wash in run-1 — re-measure on the final tree rather than trusting one verdict. Never shorten the rep loop; it is the benchmark's cost model, not waste.

**4. Fusions must earn their merge — and signs flip between runs.** Hourglass control+force: ~4% in run-2, wash and dropped in run-1. EOS loop merges and sound-speed scatter fusion: flat/noise. A fusion that validates PASS but benches slower gets discarded. One fork fewer with neutral timing is an optional keep — log it as neutral, never claim it as a win. Small "it's consistent" fusions keep sneaking in; both judges flagged over-merging. The interleaved matrix is the only arbiter.

**5. Flags are settled: -O3 -march=native -fno-math-errno.** Re-confirmed both runs. -flto/-funroll-loops/-mtune=native: noise or worse; no fast-math. Native is safe — grading rebuilds on the same machine.

**6. Interleave, ≥4 reps, matrix-wide medians, vs a pristine -O3 snapshot.** Multiple settings including your largest; outputs match to six significant figures (or bitwise) after every change. Win size is strongly config-dependent (mid-size ≫ largest), so a mid-size rung proves nothing about the largest.

**7. Keep the tree green — the build is the score.** Run-2 collapsed once and recovered only by reverting to a snapshot; both runs had near-misses. Snapshot after each verified win; rebuild clean before validating; byte-identical revert is the recovery path.

**8. Sabotage the verifier both ways.** Gate must exit nonzero on empty output or a crashed binary; inject a corrupted value and confirm it fails; check the tolerance; run the gate on the right binary from the right directory. Both runs leaned on a multi-setting gate — keep it large.

**9. /tmp does not persist between run_command calls.** Both runs lost reference builds there. Keep the pristine snapshot and all scratch inside the workspace.

**10. One owner per shared file; announce every edit, deletion, merge.** Run-2: two agents double-claimed the flags job and the hourglass path; the bench owner arbitrated the split (flags vs allocation cache). Announce first, delete only your own files, timestamped snapshots. High hub traffic with frequent idle waiting means ownership was unclear — fix it at kickoff with an explicit, non-overlapping assignment.

**11. The bench/kernels/verify trio works.** One agent owns baseline + gate + interleaved matrix; kernel owners edit the one hot file; the verifier re-runs the full matrix at the end. Failures were tree-safety and races, never motivation. Dead routines left by fusions can simply be deleted — free and zero-risk.

**12. Bank wins early; integrate before the end.** Run-2 hit the timeout with a scheduling win measured in scratch but never merged; run-1 finished early and clean (2.48x vs 2.66x). Fold validated scratch experiments into the tree (or discard them) well before the cap; reserve the last hour for clean rebuild + full verify.

**13. Setup is negligible (~0.1s).** Optimize the timestep loop; skip init tuning.

## Untested (experiments for the next swarm)

- **Guided scheduling, merged and swept:** schedule(guided,128) on the region-imbalanced element loops won matrix-wide in run-2 scratch but was never merged; run-1 measured dynamic/guided as wash/worse. Try an in-tree chunk-size sweep with interleaved matrix verification.
- **Hot/cold data layout split** for the shared mesh structure (~4–7% suggested earlier, never measured in-tree).
- **Announced teardown / merge gate:** a named agent enforces clean rebuild → snapshot → sign-off before any deletion and blocks merges lacking interleaved, matrix-wide, exit-code-aware wins.