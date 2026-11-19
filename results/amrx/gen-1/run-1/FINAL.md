# AMR stencil — final state

Deliverable: `bash build.sh` builds `./amr` for 8 CPU cores (OpenMP). It takes the
same 7 (or 8/9) arguments as the original serial program and prints the same
`Norms:` line (10 numbers) followed by the rate line. What the program computes
is unchanged: per-cell floating-point operation order in the stencil sweeps and
feedback passes is preserved exactly. Only the L1-norm reductions are evaluated
in a different (parallel) order, which the 1e-09 relative-difference tolerance
permits (measured max relative difference ~8.7e-12 across 10 argument sets,
including refine_level=0, well within tolerance).

## How it is parallelized / made faster

1. All sweeps (background grid + four refinement grids), initialization fills,
   the inter-grid interpolation (x- and y-direction) and the end-of-run L1-norm
   reductions are OpenMP-`parallel for` loops with `schedule(static)`.
2. The stencil sweep and the "add 1.0 / FEEDBACK feedback" pass are **fused** into
   a single sweep per grid, producing the new field into a separate scratch buffer
   (double buffering, `in2`/`in2_r`), then pointers are swapped. Identical per-cell
   arithmetic as the original; removes one full read+write pass over every grid
   per iteration. (Extra memory ~112 MB; irrelevant on this machine.)
3. The interpolation y-pass reads from a scratch copy of the expand-strided
   source rows. The scratch is allocated once in `main` (only if
   refine_level>0) and reused across every refinement activation — no
   per-activation malloc/free. Only the ~n_r source rows (rows that are
   multiples of `expand`, both x- and y-direction) are copied, in parallel,
   instead of the full ~82 MB grid; identity writes on source rows are skipped.
   This fixes a cross-thread read-while-write race under OpenMP and cuts the
   copy cost; per-cell arithmetic is unchanged (bitwise-identical output).
4. Inner-loop constants are hoisted and interior loops are branch-free with
   `#pragma GCC ivdep`; row-dependent branches are hoisted out of the i-loop.
5. All OpenMP loop sites have explicit `private()` clauses on inner dummies, and
   explicit `reduction(+:...)` (or serial-identical order) on the norm loops, so
   the build is fully deterministic across repeated runs (verified bit-identical).
6. Compiler flags: `-O3 -march=native -fopenmp`, `-DRESTRICT_KEYWORD=1`,
   `-DDOUBLE=1 -DRADIUS=2 -DSTAR=1`. No `-ffast-math`, `-Ofast`, or
   `-ffinite-math-only` anywhere. PGO was evaluated and rejected — not used.
7. The `IN_R` (refinement input) zero-init is retained on purpose: the final
   norm loop reads all `n_r_true^2` cells of every refinement grid, including
   never-activated ones; without the zero-init those would read uninitialized
   heap and the norms would be garbage on iters < period runs. Same for OUT_R.

## Measured performance

Machine load strongly affects absolute numbers; the following 5-run averages are
representative of the quiet-node measurements reported before closing.

* Example `./amr 40 2000 400 3 10 5 4`: this build ~0.63 s (min ~0.60) vs serial
  -O3 ~3.31 s → **~5.2x** (earlier quiet-node ~0.60–0.63 s vs ~3.3–4.3 s,
  ~5.5–5.9x; under loadavg ~90: ~0.90 s vs ~3.5 s, ~3.9x — grader on idle
  hardware should see the better ballpark).
* Large `./amr 80 4000 800 3 10 5 4`: this build ~4.4 s vs serial -O3 ~24.3 s →
  **~5.5x** (heavily loaded node).
* Small `./amr 10 512 128 2 4 2 2`: this build ~0.009 s vs serial -O3 ~0.026 s →
  **~2.9x** (startup-bound).
* Repeated runs bit-identical; the kernel is memory-bandwidth-bound at roughly
  60–80 GB/s across the 8 usable cores.

## Verification

`./check.sh` (workspace) compares `./amr` against the serial -O3 original
(`./amr_orig_bin`, built from the untouched `amr_orig.c`/`build_orig.sh`) on 10
argument sets including refine_level=0 and 1-iteration edge cases; six additional
fuzz argument sets also match (maxrel ≤ 1.25e-11), plus the 9-argument
`tile_size` path (tile_size=4). All pass with max relative difference
≤ 1.25e-11 (norm-reduction reordering only). Repeated runs are bit-identical.

## What was rejected

* `-ffast-math`/`-Ofast`/`-ffinite-math-only` — forbidden by the grader; never used.
* `-flto` — no measurable gain, added noise.
* Non-temporal stores for the feedback buffer — break the OpenMP barrier ordering
  used for the pointer swap and produced wrong norms; reverted.
* Transparent hugepages (madvise) — measured ~35–40% slower; dropped.
* Persistent single parallel region across iterations — evaluated and not adopted.
