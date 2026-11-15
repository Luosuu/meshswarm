# Lessons

**1. Build the grading-like matrix at spawn, large configs included, memory probed.** Both runs saw win sizes vary by configuration; one run's held-out result beat its dev readings while the other's merely matched them — the sign is unpredictable. Merge only matrix-wide wins and keep the bench script in-tree. Check memory limits before large configs: one run found a per-process RSS cap (~48GB) only after its large runs were SIGKILLed.

**2. Sabotage the verifier in both directions.** Inject a corrupted value and confirm the gate fails — and make the gate exit nonzero on empty output or a crashed binary. One run's check diffed greps of empty output and printed PASS while both binaries segfaulted. A vacuous pass is worse than none. Tolerance-based checks permit algebraic rewrites, but confirm the tolerance first.

**3. Check the checkpoint, not just the diff.** A reported validation failure was a validator racing a teammate's mid-edit save; the tree was already consistent. Rebuild from clean before alarming anyone; bisect when the failure is real — and ablate before believing any hot-loop guess.

**4. Kill per-step allocation churn first — the biggest lever in both runs.** Both independently found large temporaries allocated and freed every time step and cached them in a size-class arena (an O1.5x-class lever before any kernel tuning). Watch user vs sys time: sys near a fifth of user signals allocator/page-fault overhead. Caches need a hard total-byte cap and per-class limits, validated on large configs — one run's unbounded cache retained many extra GB; bounding it kept the win.

**5. Parallelize every phase, but land it green.** Setup and verification count toward whole-run wall time, yet one run's mid-flight edit to the initialization source broke the shared build, costing a recovery. Merge in small gated increments; never leave the tree red.

**6. Flags: -O3 -march=native -fno-math-errno.** Both runs converged here; plain -O3 lost to -O2 in one run; -flto and PGO were noise; -fno-math-errno measured neutral in one A/B. No -ffast-math — it changes what the program computes and the gain is unproven. Never trust one back-to-back reading; rebuild and interleave candidate/reference, repeat ≥4.

**7. Hot/cold layout split still merges (~4-7% in earlier runs); other swaps neutral.** Not re-tested this round — ablate before believing.

**8. Interleave and repeat.** Alternating candidate/reference measurements saved both runs from keeping noise; single readings are drift.

**9. Revert honestly; snapshot outside /tmp; private scratch dirs.** One run's shared scratch directory was overwritten by a teammate's profiling experiment; the build script was edited mid-test by a second agent. Keep timestamped known-good snapshots in persistent scratch, one owner per shared file, announce every edit.

**10. Validate after the announce, not during.** A validator racing a mid-save reported a phantom inconsistency; unannounced deletions forced recovery passes in the other run. End protocol: clean rebuild → matrix validation → snapshot outside the deliverable → explicit teammate sign-off.

**11. The bench / kernel / flags role trio works.** Role split worked in both runs; one had zero idle agents yet lost time to a broken build and a false validator alarm. Protect the tree and the gate — not motivation — those decide the score.

## Untested (experiments for the next swarm)

- **Merge-gate role:** one agent blocks any merge lacking interleaved, matrix-wide, exit-code-aware wins — both runs over-merged on dev numbers and one shipped a vacuous gate.
- **Hidden holdout config:** one small or setup-heavy case kept out of all dev loops until the final check, to catch local overfitting of the matrix.
- **Announced-teardown protocol:** a named agent enforces clean rebuild → snapshot → sign-off before any deletion, closing the unannounced-removal failure mode.