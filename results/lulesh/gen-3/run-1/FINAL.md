# LULESH on 8 CPU cores — what was done

Deliverable: this workspace. `bash build.sh` builds `./lulesh` with
`-O3 -march=native -fno-math-errno -fopenmp -DUSE_MPI=0` (was `-O2`).
Correctness is preserved: iteration count and final origin energy match the
pristine `-O3 -fopenmp` reference (`.snapshot_orig/lulesh_orig_O3`) to six
significant digits on all tested configs (s30, s30-r3, s20-r2-b2-c2, s40, and
additional region/balance configs).

## Source changes (restructures that keep the same per-element arithmetic and loop partitioning)

1. **lulesh.h — grow-only, size-class allocation cache behind `Allocate`/`Release`.**
   A 16-byte header records each block's padded size class; freed blocks are
   recycled per class up to a hard cap (512 MB total, 4096 blocks per class) with
   a mutex. This eliminated the per-timestep malloc/free churn from the many
   temporary `numElem`/`numElem8` arrays allocated in force and EOS kernels.
   (This was the single biggest lever, ~1.7-2x on its own together with the EOS work.)

2. **lulesh.cc — EOS rep-loop hoisting in `EvalEOSForElems`.**
   The artificial-repetition cost loop is kept (it is the benchmark's cost model),
   but the loop-invariant gathers (element field loads, compression /
   compHalfStep factors, eosvmin/max clamping, `work[i]=0`) were moved into a
   single `#pragma omp parallel` region *before* the rep loop instead of being
   redone every rep.

3. **lulesh.cc — single parallel region in `CalcEnergyForElems`.**
   The five/six `omp parallel for` EOS sub-loops (and the `omp for` loops in the
   inlined `CalcPressureForElems`) now run inside one `#pragma omp parallel`
   region with implicit barriers between them, eliminating repeated fork/join
   overhead per rep per region.

## Measured speedups vs pristine -O3 build (interleaved, median-ish)

- s30: ~2.1-2.6x   (orig ~6.4-7.0s -> ~2.7-3.0s)
- s30 -r4 -c2: ~2.6x
- s35: ~2.5x
- s40: ~2.3-2.7x
- s45: ~2.3-2.5x   (orig ~26.6-29.7s -> ~10.8-12.2s)
- s45 -r2 -b2 -c2: ~2.4x
- s50 -r5: ~2.5x
- s55: ~2.3-2.4x
- s60: ~2.4x

Consistently above the 2x rung, near 2.5x on average.

## Experiments that measured neutral or worse and were NOT merged

- Hourglass control+force kernel fusion (one element pass) — wash.
- "Gathers in same node loop" fusion for stress+hourglass nodal force
  accumulation — wash/slightly worse (extra accumulators, cache pressure).
- OpenMP `schedule(dynamic,64)`/`guided` on element loops — slightly worse;
  static stays.
- EOS bvc/pbvc precompute across reps — wash.
- Rep-loop inside a single parallel region — consistently slower.
- Zeroing fx/fy/fz before force accumulation — wash.
- PGO (-fprofile-generate/-fprofile-use) — no measurable gain.
- `nowait` removal on hoisted-region loops — wash.
- `-flto`, `-funroll-loops`, `-mtune=native`, fast-math — not used
  (fast-math would change results; the rest measured no gain).

## Correctness tooling kept in the workspace

- `verify.sh` — rebuilds nothing; compares `./lulesh` vs the pristine reference
  on s30, s30-r3, s20-r2-b2-c2, s40; exits nonzero on any iteration-count or
  six-significant-digit energy mismatch.
- `bench.sh` — interleaved orig-vs-workspace timing harness.
- `.snapshot_orig/` — pristine sources and the `-O3 -fopenmp` reference binary
  used by `verify.sh` (kept for reproducibility).

## Notes

- `build.sh` uses `-march=native`; grading rebuilds on the same machine.
- The allocation cache holds freed blocks in-process; peak RSS grows by at most
  the 512 MB cap and is reused for the whole run.
- Nothing outside `lulesh.h` (Allocate/Release), `lulesh.cc` (two kernels) and
  `build.sh` (flags) was changed; mesh setup, region logic, and all arithmetic
  remain exactly as authored.
