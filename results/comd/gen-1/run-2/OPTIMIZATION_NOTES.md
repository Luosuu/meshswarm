# CoMD optimization — final state

Goal: fastest correct `./CoMD` on 8 cores (LJ + EAM), without changing what it computes.

## Changes vs original
- build.sh: -O2 -> `-O3 -march=native -fno-math-errno -fopenmp -DDOUBLE`
- ljForce.c: 4-wide unrolled pair loop (precomputed r2 lanes, fi0/fi1/fi2/uLocal accumulators, single force writeback), restrict rArr alias, zero-forces pass bounded to local boxes
- eam.c: same 4-wide unroll in pass 1 (two-body + rho) and pass 3 (embedding force), restrict aliases, zero-pass bounded to local boxes, `static inline interpolate`
- haloExchange.c: persistent halo buffers (allocated on first use), emptyHaloCells via memset
- linkCells.c: insertion sort in sortAtomsInCell (replaces VLA+qsort), containment fast-path in updateLinkCells (atoms strictly inside cell inset bounds skip floor-based lookup)
- memUtils.h: all allocation via _mm_malloc(64)/_mm_free; cmdLineParser.c frees use comdFree consistently
- timestep.c, decomposition.c, CoMD.c, initAtoms.c: unchanged

## Verification (all on this tree)
- gate.sh (7 configs vs bench_out/verify_o3.txt -O3 reference, rel-tol 1e-7): GATE PASS
- verify.sh rows identical in step/atoms/energies on all 7 configs (only rate column differs)
- From-scratch: rm CoMD; bash build.sh; gate PASS; RUN_OK
- Interleaved vs native -O3 reference (o3_snap/CoMD_native_o3 / CoMD_ref_o3):
  lj30 ~2.8-2.9 vs ~3.9-4.1s (~28-30%), eam24 ~2.3-2.4 vs ~3.6-3.7s (~34-35%),
  lj40 -N30 ~2.06-2.17 vs ~2.9s (~29%), eam32 -N30 ~1.69-1.78 vs ~2.6s (~33%)
- Energies bitwise-identical to -O3 reference on lj34^3/eam28^3 and large configs

## Rejected experiments
- schedule(dynamic/guided) variants: worse or neutral
- Half-pair "j-once" schemes w/ thread scratch: LJ bimodal no better, EAM much worse (reverted)
- -mprefer-vector-width=512: no gain (reverted)
- updateLinkCells parallelization: unsafe without ordering changes (agent-2 call, declined)

## Keep
- pristine/, orig_snap/, o3_snap/, snapshots/ contain untouched originals and references — do not delete
- bench.sh / bench_large.sh / gate.sh / interleave.sh / verify.sh / finalcheck.sh are the validation harness
