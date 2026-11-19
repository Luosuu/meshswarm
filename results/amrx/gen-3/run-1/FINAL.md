# AMR stencil optimization — final state

Deliverable: this workspace. `bash build.sh` produces `./amr` (same args, same output format as the original).

## What changed

`amr.c` (canonical optimizations, computation unchanged):
1. OpenMP parallel for on all grid sweeps: background stencil, refinement stencils, initialization, full-grid "IN += 1.0" and feedback loops, interpolation phases (x-phase parallel over rows; y-phase parallel over columns to preserve the row read-after-write order), and the final norm reductions.
2. STAR stencil body (RADIUS=2) fully unrolled with compile-time weights so the inner i-loop vectorizes (64-byte vectors on Zen 4); per-point addition order preserved.
3. Fused the two full-grid passes per grid per iteration: interior gets `(IN + 1.0) + FEEDBACK*OUT`, edges get `IN + 1.0` — identical operations in the same order as the original two passes, so results are bitwise identical.
4. `RESTRICT_KEYWORD=1` for restrict pointers.

`build.sh`: `-O3 -march=native -fopenmp` (no `-ffast-math`/`-Ofast`/`-ffinite-math-only`).

## Verification

Against pristine `amr_orig` (the original source built with `-O3`): 22 argument sets (normal + edge cases: level=0, period=duration, duration=1, odd sizes, 3000^2/4000^2 grids). All `Norms:` lines match with worst relative difference 4.6e-11, far below the 1e-09 tolerance.

## Speedup

Example (`./amr 40 2000 400 3 10 5 4`): serial -O3 original ~2.5s → optimized ~0.35s, about 7x on 8 cores. Larger grids ~5-6x.

## Rejected experiments (measured, not adopted)

- Double-buffered IN with feedback computed inside the stencil sweep: bitwise identical but ~19% slower.
- `-mprefer-vector-width=512`: neutral on Zen 4.
- `-funroll-loops`, `-flto`: within noise.
- Stride padding of refinement rows: ~1%, not adopted (worse on large grid).
- Persistent parallel region around the time loop: not pursued (est. 1-3% gain, race risk).
