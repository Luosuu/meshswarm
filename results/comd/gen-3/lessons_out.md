# Lessons

Both runs (1.93x / 1.87x) agree closely; where they concur, treat it as settled.

**Board first; claim before editing.** Both posted tasks in minute one; mediation split a kernel double-claim within minutes. Your first ten minutes: snapshot, reference binary, verifier, claims.

**Freeze one reference binary; label its flags.** run-1 silently rebuilt its reference mid-run (-O2→-O3); every bench read 0.92x and nearly rolled back healthy work. run-2 ended with duplicate reference dirs. Never rebuild the baseline silently — announce and re-baseline aloud if you must.

**Verify continuously, not at the end.** Your merge gate is the matrix: both force models, non-cubic shapes, off-example temperatures, run per change. Audit the verifier both ways — corruption, empty output, leading-space field misparse must exit nonzero. Agents left idle (44–75% asleep in both runs, waiting on kernels) belong on randomized verification, not polling.

**Profile per-phase first; the hot phase is ~85% of the step.** Setup is noise — skip it. Re-check the split after the first structural win; it shifts.

**One structural change beats a dozen micro-opts.** Both runs got their biggest gain from a work-halving restructure of the hot loops; 1–4% trims compounded second (occupied-slot zeroing, persistent scratch, dead-array removal, an inline that wasn't inlining, mass precompute, insertion sort + early-out). Hunt symmetry first. Cut work inside the measured loop; never shorten it.

**Every variant earns its merge; post the negatives.** Register-accumulated force loops defeated vectorization; a per-pair cache arena lost twice; PGO, LTO, unroll, forced vector width, dynamic(8) scheduling: noise or worse — two-run agreement. Under load (~50 seen on 8 cores) deltas <5% are noise: interleave ≥5 reps vs the frozen reference, median, re-run spikes before reverting. Static scheduling held. Never overwrite the timed binary.

**Flags: -O3 -march=native -fno-math-errno; -ffp-contract=fast helped one run** (both models, identical energies; not banned). Single-run evidence — confirm energies before banking. Never -ffast-math/-Ofast/-ffinite-math-only.

**Namespace your scratch; the glob and filesystem are shared.** A stray .c broke the build via double definition; a copied in-flight file surprised its author; a stray file was deleted as junk; root littered with .yaml. Snapshots outside the source glob, claim names first, delete only what you created, announce deletions and copies of others' work.

**Snapshot the verified tree; restore, don't reconstruct.** run-2 truncated a source via scripted edit and recovered from backup; run-1 lost time to misattributed snapshots. Label snapshots with state and date.

**Bank early; rehearse last; re-rehearse after late merges.** Both runs froze at ~1.9x. run-1 nearly merged a flag change after the md5 rehearsal — even one flag invalidates a rebuild check. Endgame: lock-in → rm binary → build.sh → md5 → full verify → done. Validated work outside the tree scores 0.

# Untested

- Hot/cold split of the atom arrays (hint 4–7%; still never measured in-tree). Experiment.
- Sealed variant protocol: variants in a copied tree only; merge solely on gate PASS + matrix-wide median win. Experiment.
- Auto-verify hook on each merge (both models × shapes × temps), replacing hand-run verify. Experiment.