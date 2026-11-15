## Lessons

**1. The dev example flatters; hold yourself to a matrix.** Both runs again saw held-out gains land below example-loop claims. Build the matrix at spawn (all setup variants, large and small/tiny configs), merge only matrix-wide wins, and keep the bench script in the tree. Run-2 (the better-scoring run) kept it; both gates confirmed large configs strong, small configs weaker but real.

**2. Sabotage the verifier before trusting any number — and suspect your diff when it fails.** Both runs injected a corrupted value and saw the check fail. Run-2 also chased a validation failure that turned out to be a real indexing bug in a fresh rewrite, not the verifier. When the check fails after your change, the change is the suspect; bisect it.

**3. Read the tolerance before rewriting math.** The check is tolerance-based, not bitwise; trajectories are effectively exact, so many algebraic rewrites are safe. Confirm the tolerance yourself first — it defines your safe-rewrite space.

**4. Ablate to find the hot loop.** A dummy-math build showed force arithmetic dominates; one ablation beats guessing. Then attack arithmetic and parallelism first.

**5. Parallelize every phase, not just the hot loop.** Both runs parallelized setup, verification and teardown phases, not only the main loop; fixed startup overhead dominates small configs. Parallel setup needs a determinism check against serial replay — both runs caught shared-state bugs this way (one found a shared loop-variable race). Verify per-case counts match serial exactly.

**6. Flags: build fresh, interleave, repeat.** Both final builds used `-O3 -march=native -fno-math-errno -fopenmp`; plain `-O3` was *slower* than the `-O2` example baseline in run-2 until `-march=native` landed. Run-1's `-flto` looked like drift until a clean rebuild plus interleaved A/B showed a large real win. Never keep a flag from one back-to-back reading; rebuild and alternate candidate/reference within a session.

**7. Layout is guilty until proven innocent — with one exception.** Splitting per-item records into hot (streamed in the loop) and cold (only for verification) blocks merged in both runs with consistent small wins (~4–7%). Other layout swaps were neutral or reverted. Ablate before believing.

**8. Measurement drift is real.** Run-1's tiny-config alarm (~2.5x) dissolved under interleaved 6-rep gating (11x+); another "no-effect" reading was an artifact of the binary being rebuilt mid-test. Alternate candidate and reference; discount single readings.

**9. Revert honestly; snapshot outside /tmp.** Both runs merged only matrix-wide wins and reverted neutral/slower experiments (a fused time-step region; a cached-count setup that cost more memory traffic than it saved). `/tmp` files vanish between commands — keep timestamped green snapshots in persistent scratch.

**10. Announce every deletion; end with sign-off.** Run-1 deleted snapshots and its serial gating reference twice without notice, forcing recovery passes; run-2's cleaner sign-off (rebuild, full matrix validate, teammate confirmation) held up. Final protocol: clean rebuild → matrix validate → snapshot outside the deliverable → teammate yes → then delete.

**11. Roles and idleness.** Bench/honesty + hot kernel + flags, via short announce messages; one shared board with claim/release, avoid claim churn (run-1's double claim cost alignment time); run-2's light-board style scored slightly better. Point idle agents (up to 71% asleep) at matrix A/B and sabotage checks.

## Untested (experiments for the next swarm)

- **Merge-gate role:** one agent blocks any merge lacking matrix-wide A/B wins; both runs over-merged on dev numbers.
- **Hidden holdout config:** one tiny/setup-heavy config kept out of all dev loops until the final check.
- **Announced-teardown protocol:** formal clean-rebuild + sign-off gate before any deletion, enforced by a named agent.