# AMR stencil — optimization summary

Build: `bash build.sh` → `./amr` (gcc -O3 -march=native -fopenmp, same arguments/output as original).

Changes vs original serial code (all norms-verified to 1e-9 rel tol against the pristine serial -O3 build):
- OpenMP `parallel for schedule(static)` on all sweeps: background & refinement stencil loops,
  add-constant+feedback (fused into one pass with hoisted interior test), input/output init & zeroing,
  L1-norm reductions (reduction over rows), and the (rarely used) tiled loop nests — per-element
  arithmetic operation order is preserved everywhere.
- Stencil hot loops: explicit unrolled STAR (RADIUS==2) adds with register accumulation `o`
  (bitwise identical operation order to the original nested loops).
- Fused add-constant + FEEDBACK·OUT pass (single sweep, same per-element op order as the two original passes).
- madvise(MADV_HUGEPAGE) on the four big buffers (THP enabled; small consistent win, no downside).
- `-march=native` and OpenMP enabled in build.sh. No `-ffast-math`/`-Ofast`/`-ffinite-math-only`.

Verification: `./verify_rel.sh ./amr` — 13 arg sets (incl. tiled path & tiny/edge cases) vs `./amr_ref_o3`.
Large sets checked: 80 4000 800 3 10 5 4 → max rel diff 9.9e-11; 20 4000 800 → 6.3e-11; 30 3500 700 → 3.4e-11; 4 5000 1200 2 8 6 2 → 1.4e-10. All well under the 1e-9 gate.
Tiling path with 9th `tile_size` arg checked for 5 tile values (1,7,32,100,2000) — all < 4.5e-12.

Performance (median-of-N, noisy shared box; serial-O3 ref ~3.8s on example):
- Example `40 2000 400 3 10 5 4`: ~0.39 s vs ~3.8 s (~9–10x). (wall-clock floor ≈ 0.15 s from alloc+first-touch of ~400MB)
- Large `80 4000 800 3 10 5 4`: ~4.0–4.3 s vs ~33.8 s (~8x).

Rejected (measured, no gain or slower): collapse(2) on stencil loops, lagged/pipelined stencil+feedback fusion
(looked like ~3–4% on >3000-row refinements but consistently ~12% slower on the 2000-class grid; reverted to keep
the common case fast), register-accumulated feedback merge, j-by-2 unroll (boundary-row OOB risk),
`#pragma omp simd` inner loop, RESTRICT build, znver4 flags (== native), refinement row padding, flat/long-idx init,
guided/dynamic schedules, OMP_PROC_BIND spread/close (no measurable effect here; OMP_PROC_BIND=master was catastrophic),
-fno-tree-vectorize, single parallel region init, -flto, -funroll-loops.

Workspace hygiene: pristine originals kept (`amr_orig.c`, `amr_orig_o2`, `amr_ref_o3`); `verify.sh` / `verify_rel.sh`
comparison scripts; `bench.sh` helper; `PROGRESS.md` (detailed notes); `par-res-kern_general.h` untouched.
