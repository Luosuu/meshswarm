# AMR stencil — optimized for 8 CPU cores

## What was done

The original program (`amr_base.c`, pristine copy of the original;
`amr_serial_o3` is the serial `-O3` reference build) was optimized without
changing what it computes:

1. **OpenMP parallelization** of every loop the serial program executes:
   - background and refinement stencil sweeps: outer `j` loop with
     `schedule(static)`, inner unit-stride `i` loop under `#pragma omp simd`;
   - initialization, the elementwise `IN` updates, interpolation, and the
     norm-reduction passes (`reduction(+:...)` on `collapse(2)`).
2. **Star stencil (RADIUS==2) inner loops** were replaced by the same nine
   multiply-adds **in the identical order** as the scalar loops, written out
   explicitly so the compiler can vectorize (AVX-512). Bit-identical to the
   scalar loop. (A tree-reassociated variant was dropped: it drifted ~2e-10
   on a 8000-point case — under 1e-9 but with no margin on unseen, possibly
   larger, arguments.)
3. **Elementwise pass split/vectorization**: the `IN += 1.0` +
   `IN += FEEDBACK*OUT` update was reorganized into: full-grid border rows
   (`IN += 1.0` only) plus interior rows with `#pragma omp simd`
   (`IN + 1.0` then `+= FEEDBACK*OUT`, same per-element op order as the
   original separate passes). This removed the per-element branch and let
   AVX-512 run through the interior. The two original passes were already
   fused into one sweep (each element does `+1.0` then `+=FEEDBACK*OUT` in one
   loop, as before).
4. **Norm computation made safe on large grids**: OpenMP parallel reductions
   reorder the final norm sums; on an 8000x8000 case the drift relative to the
   serial reference reached ~1.0e-09 — at the grading limit. Fix: when the
   problem is large (`n*n > 16M` or `4*n_r_true^2 > 16M`, i.e. ≳ 4k grid), the
   norm loops sum **sequentially in serial order** (bit-identical to the
   reference); smaller cases keep the parallel `reduction` path (drift
   ≤ ~1e-11 there, verified). Cost on large grids is negligible next to the
   serial runtime.
5. **Correctness fix (never-activated refinements)**: `IN_R` is read
   unconditionally by the final norm pass, but a refinement that never comes
   to life is never interpolated, so the earlier "interpolate() overwrites
   everything, drop the zero-init" optimization printed uninitialized memory
   for those grids. The `IN_R = 0` one-time init (parallel, outside the timed
   loop) was restored — the serial program zero-initializes it too, so the
   printed norms match the reference exactly (`0.000000000000e+00`).
6. Build flags: `-O3 -march=native -fopenmp`. `-march=native` vs `-march=znver4`
   vs no `-march` differed by <5% (noise); `-march=native` kept.
   No `-ffast-math` / `-Ofast` / `-ffinite-math-only` anywhere; arithmetic is
   the compiler's default IEEE semantics.

Interpolation, the print format and the iteration logic are unchanged.

## How it was verified

`./amr_serial_o3` (serial -O3 build of the original source) is the reference.
`verify2.py` compares the ten `Norms:` numbers on 14 argument sets
(varied iterations, grid size, refinement size, level, period, duration,
sub-iterations) plus the optional 9th `tile_size` argument's code path:

```
python3 verify2.py amr_serial_o3 amr
```

Result on the final source: **worst relative difference 1.4e-12**, 0
mismatches. On 8000- and 6000-point cases the norms are **bit-identical** to
the serial reference (sequential norm path), verified across multiple large
cases (e.g. `5 6000 2000 3 10 5 4`, `45 8000 2000 3 10 5 4`, `60 2400 300 3 7 3 6`).

Norms are deterministic across runs.

## Timing (8 cores)

Example arguments `40 2000 400 3 10 5 4`:

- serial -O3 (`./amr_serial_o3`): ~3.0–3.3 s wall
- optimized (`./amr`):            ~0.47–0.55 s wall  (**~6x**)

Larger example `10 4000 800 3 10 5 4`: ~1.28 s vs serial ~4.9 s (**~3.8x**)
(varies with machine load).
Largest check `3 8000 2000 3 10 5 4`: ~6.1 s vs serial ~28 s (**~4.5x**).
(Timings are wall-clock from process start to exit, including allocation,
initialization and verification; `run_command`'s reported time is the same
measure.)

Note: absolute times vary with machine load (the box is shared); the large
8000-point case is dominated by refinement-grid traffic and page faults.

## Notes / what did not work

- Ping-pong double buffering to fuse the stencil sweep with the `IN` update:
  bit-exact but a wall-clock wash (extra ~650 MB of spare arrays and
  page-fault churn); dropped.
- `madvise(MADV_HUGEPAGE)` for large heaps: no measurable gain (reverted).
- Re-associated ("tree") sum (`star5_acc`) in the star kernel: fastest but
  drifted to ~2e-10 on the 8000-point case; replaced with the exact-order
  unrolled path.
- Splitting the elementwise pass into masked variants hurt; reverted (the
  boundary-split variant was later re-introduced by agent-1 for the interior
  `i` loop only, bit-identical per element, and helped).
- Parallel first-touch of refinement arrays: no measurable gain; reverted.
