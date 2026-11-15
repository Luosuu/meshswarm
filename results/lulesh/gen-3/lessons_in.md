# Lessons

**1. Kill per-step allocation churn first — biggest lever in both runs.** A grow-only, size-class cache behind the shared Allocate/Release path took large configs from ~19s to ~13–14s before any kernel work. Keep a hard byte cap; validate on your largest config. Once it exists, extra "persistent scratch buffer" ideas measured no gain in either run.

**2. Profile inside the binary.** gprof is useless — everything inlines into one blob. Instrument a private scratch copy with per-phase wall timers; both runs found one EOS/region-update phase dominating. Profile before touching kernels.

**3. Hoist rep-invariant work out of artificial repetition loops — biggest kernel-class win.** The region EOS is recomputed rep times on identical inputs; hoisting the invariant gathers (keeping the rep loop, write-back outside it) gave ~8% at large sizes (one run) and drove a clear step-time win in the other. Never skip or shorten the rep loop — it is the benchmark's cost model, not waste.

**4. Fuse shared-temporary element kernels, but only on interleaved wins.** Fusing hourglass control+force into one element loop (dropping per-element temp arrays) won in both runs (~4%). A stress/hourglass pre-pass fusion validated 5/5 yet ran consistently slower — discarded; EOS loop-merges were noise. Fusions must earn their merge.

**5. Flags are settled: -O3 -march=native -fno-math-errno.** Re-confirmed in both runs. -O3 > -O2; -flto, -funroll-loops, -mtune=native: noise or worse; no fast-math. Grading rebuilds on the same machine, so native is safe.

**6. Interleave, repeat ≥4, trust only matrix-wide wins.** Win size varies by config (one run: 2.46x mid-size, 1.94x largest). Confirm iteration count and origin energy to six significant digits vs a pristine -O3 build, on multiple settings, after every change.

**7. Keep the tree green — the build is the score.** Both runs lost real time to broken builds (one exited 255 with wrong energies after a partial edit; recovery was a byte-identical revert + clean rebuild). Announce before editing, never leave shared sources mid-edit, rebuild clean before validating.

**8. Sabotage the verifier both ways; beware vacuous PASS.** Gate must exit nonzero on empty output or a crashed binary; inject a corrupted value and confirm it fails; check the tolerance; run the gate on the right binary from the right cwd (a near-miss wrong-directory gate run appeared in one run).

**9. One owner per shared file; private scratch; snapshots outside the deliverable.** One run had a teammate's patch deleted during unannounced "cleanup", forcing re-derivation. Timestamped known-good snapshots; announce every edit and deletion; delete only your own files.

**10. Bench/kernel/verify trio works.** One agent owns baseline + gate + interleaved bench, others take kernels; the verifier also runs the final matrix. Failures were tree-safety and validator races, not motivation. Hot/cold layout split untested this round (~4–7% earlier) — ablate before believing.

**11. Setup is negligible (~0.1s).** Optimize the timestep loop; skip init.

## Untested (experiments for the next swarm)

- **Announced teardown:** a named agent enforces clean rebuild → snapshot → explicit sign-off before any deletion.
- **Merge-gate role:** one agent blocks any merge lacking interleaved, matrix-wide, exit-code-aware wins — both runs over-merged on dev numbers.
- **Imbalance-aware scheduling:** dynamic/guided scheduling for region-imbalanced timestep loops vs the default static used every run.