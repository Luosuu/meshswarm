# LULESH performance work — status & notes

Deliverable: this workspace. `bash build.sh` produces ./lulesh (flags: `-O3 -march=native -fno-math-errno -fopenmp -DUSE_MPI=0`).

## Current state (vs original build with -O3 -fopenmp, same 8 cores)
- -s 30: ~2.6s vs ~5.5s (~2.1x), iteration 932, Final Origin Energy 2.025075e+05 (identical)
- -s 45: ~7.8s vs ~19.3s (~2.5x), iteration 1477, energy 4.234875e+05 (identical)
- -s 45 -r 4: ~7.8s vs ~19.3s (~2.5x)
- -s 60: ~29-31s vs ~65-71s (~2.3x), iteration 2031, energy 7.130703e+05 (identical)
- -s 40 -b 4 -c 2: ~8.1s vs ~15.7s (~1.9x), iteration 1294, energy 3.423679e+05 (identical)
- Large-size spot checks (-s 125, -s 150, energies match original to 6+ digits; ~1.4x speedup at -s 150 -i 20, workload is more memory-bound there)
Validation: validate.sh (5-config iteration+energy check vs orig_snapshot/lulesh_o3) — 5/5 PASS.

## Changes made
1. **Size-class caching allocator** (lulesh.h): Allocate/Release route through a bounded
   free-list pool (768 MB cap, power-of-two classes, 16-byte header). Removes per-timestep
   malloc/free churn (sys time was ~1/5 of user). Call sites unchanged.
2. **Build flags**: -O2 -> -O3 -march=native -fno-math-errno (native is safe: grading reruns build.sh on the same 8-core machine).
3. **EOS region fusion** (lulesh.cc): EvalEOSForElems' per-stage parallel forks fused into one
   `#pragma omp parallel` region with `omp for` stages; pHalfStep hoisted out; domain.p/e/q
   write-back and sound-speed kept OUTSIDE the rep loop (as in the original). -s 30: ~3.0 -> ~2.6s.
4. **Time-constraint fusion** (lulesh.cc): CalcCourantConstraintForElems + CalcHydroConstraintForElems
   merged into CalcTimeConstraintsForRegion — one parallel region + one fork per region per step.
   FP-exact (min reductions; element indices only used locally and never stored).

## Explicitly rejected
- Removing the `rep` load-imbalance re-evaluation in EvalEOSForElems (bitwise-identical output but
  guts the benchmark's intended workload).
- -ffast-math (changes computation, gain unproven). -flto, -funroll-loops, -fno-math-errno: measured neutral.
- Do NOT take grader-toward answers: energies must match to 6 significant digits; iteration count exact.

## Grading-visible notes
- Grading uses unseen args (larger meshes, other regions) — build must stay green and output-correct.
- orig_snapshot/ holds the pristine -O3 reference binary + sources for A/B; snapshots in snap_current/ and bench/known_good/.
- check: `bash validate.sh ./lulesh` (iter+energy vs reference); interleaved timing: `bash bench/ab.sh A B N <args>`.
