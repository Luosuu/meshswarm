# CoMD speedup — changes and results

Goal: fastest possible CoMD on 8 CPU cores, unchanged computation. Deliverable is the workspace; `bash build.sh` builds `./CoMD`.

## Build
`build.sh`: `-std=c99 -O3 -march=native -fno-math-errno -fopenmp -DDOUBLE -I. *.c -lm -o CoMD`
(baseline was `-O2 -fopenmp`; no fast-math flags).

## Source changes (energies bitwise-identical to the -O3 reference on the gate configs)
- ljForce.c: 4-wide unrolled pair (neighbor) loop with precomputed r² and `restrict` on the atom arrays.
- eam.c: same 4-wide unroll of the pair loop + static inline `interpolate` + restrict rArr.
- haloExchange.c / haloExchange.h: persistent send/receive buffers allocated on first use (was malloc/free every call); sortAtomsInCell uses an in-place insertion sort (was VLA + qsort); nAtoms<2 early-out; single-rank fast path in exchangeData that unloads directly from the send buffers (skips a memcpy when there is no real MPI).
- linkCells.c: `emptyHaloCells` via memset; conservative containment fast-path in `updateLinkCells` (atoms strictly inside their cell + 0.25 A inset skip the floor-based box lookup).
- ljForce.c / eam.c: zero-forces pass bounded to local boxes (nLocalBoxes) instead of nTotalBoxes.
- memUtils.h: allocation wrappers route through `_mm_malloc(size, 64)` / `_mm_free` (64-byte alignment; fixed mixed malloc/free mismatch).

## Verification
- gate.sh ./CoMD: 7/7 PASS (lj30, lj_small, lj_rect, lj_T, eam24, eam_small, eam_rect) — step, atom count, and total/potential/kinetic energies match the untouched -O3 reference to rel 1e-7 (in practice bitwise).
- verify.sh rows identical to the reference (only the rate column differs).
- Also checked bitwise-identical energies on off-axis boxes (44x36x28 LJ, 20x28x36 EAM), -N 20/30/50, -x36^3, -x48x16x16, eam30^3.
- From-scratch validation (rm CoMD; bash build.sh): build clean, gate PASS, verify match.

## Speedups (interleaved medians vs pristine -O3 / -O3 -march=native reference builds, 8 cores)
- lj30 (LJ, 30^3):   ~2.8-2.9 s vs ~3.95-4.05 s  (~28-30% faster, i.e. ~1.4x)
- eam24 (EAM, 24^3): ~2.3-2.4 s vs ~3.6-3.7 s    (~34-35% faster, i.e. ~1.5x)
- lj40 (-N30):       ~2.05-2.15 s vs ~2.87-2.92 s (~27-29%)
- eam32 (-N30):      ~1.68-1.78 s vs ~2.56-2.59 s (~32-35%)

Visible in table runs too (e.g. 36^3 LJ ~1.4x, eam30^3 ~1.7x vs the -O3 orig binary; grading baseline is -O3 -fopenmp). Host load was high/variable during measurement, so interleaving + medians were used.
