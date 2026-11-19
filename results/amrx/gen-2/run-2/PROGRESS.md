# AMR optimization — progress and rejected ideas

## Verifier
`verify_rel.sh <binary>` — relative-tolerance (1e-9) Norms comparison vs `./amr_ref_o3`
(serial -O3 from pristine `amr_orig.c`), 13 arg sets incl. tiled path (`... 100`).
`verify.sh` (agent-2) does exact string compare — stricter than grader; both pass.

## Status
- Canonical `amr.c`: OpenMP (init/zero/stencil/fused-feedback/norms), unrolled
  STAR RADIUS==2 star stencil with explicit register accumulation, fused
  add-constant+feedback pass, `-O3 -march=native -fopenmp`. Norms 13/13 OK.
- Example (40 2000 400 3 10 5 4): ~0.39-0.46s vs serial -O3 ref ~3.6-3.9s (~8x).
- Large (80 4000 800 3 10 5 4): ./amr ~4.0-4.3s vs serial ref ~33.8s (~8x).
  Grading uses larger args, so this regime matters most.

## Keep
- OpenMP parallel regions on all sweep/init/norm loops (static schedule).
- Unrolled register-accumulated star stencil.
- Fused add-constant+feedback (one pass, hoisted interior test).

## Tried / rejected (noise not distinguished from signal on shared box; load avg ~100 outside container)
- madvise(MADV_HUGEPAGE): canonical has it; A/B removing it ~equal (0.426-0.450 vs 0.407-0.465). Keep (no harm).
- RESTRICT_KEYWORD=1: 13/13 OK but no clear win. Dropped.
- -march=znver4 vs native: identical (native IS znver4). Kept native.
- -fno-tree-vectorize: much worse. Dropped.
- collapse(2) / omp simd on inner loop: no consistent gain (.401-.472). Dropped.
- Flat/zero-fill of refinement buffers via long idx loops: SLOWER (0.45-0.46). Dropped.
- Zeroing full OUT including borders (spec6): slower-looking/no gain. Dropped. NOTE: borders of OUT are never written by stencil; norms only read interior — but OUT borders ARE never read either (only interior + norm reads interior). Still, do not remove zeroing: in_r/out_r borders are read? IN_R border cells are read by stencil (halo). OUT_R interior written. Keep zeros.
- Padding refinement rows to 64B (v5/v7 pad): agent-2 reports no gain. Dropped.
- First-touch single-parallel-region init: slower. Dropped.

## Freeze checklist
- bash build.sh from clean tree; ./verify_rel.sh ./amr; median-of-5 on example and a large set; remove scratch binaries; FINAL.md summary.
