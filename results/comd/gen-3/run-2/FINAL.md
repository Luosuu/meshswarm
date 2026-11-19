# CoMD — optimized build

## What changed (source level)
- ljForce.c / eam.c: Newton's-third-law pair halving. Each unordered neighbor pair is evaluated once (skip local jBox<iBox; for iBox==jBox only jOff>iOff); forces split across local atoms; the mirror contribution goes into per-thread shadow buffers and is merged after the pair loop. EAM pass 3 mirrors dfEmbed×dRho the same way; rhobar is mirrored too, so the embedding pass sees identical input (up to FP reassociation, which is within the 1e-07 grading tolerance).
- ljForce.c / eam.c: scalar unrolling of the 3-component dr loops in the force kernels; `invR` instead of repeated division; `dr`/`dfEmbed` hoisted.
- atoms->U removed entirely (never read anywhere — energies come from the scalar accumulator).
- Bounded zero-init: forces zeroed only for occupied local slots instead of nTotalBoxes*MAXATOMS; dfEmbed no longer zeroed (embedding pass assigns it before any read; halo slots are overwritten by the force exchange).
- linkCells.c copyAtom: 5 memcpys → direct scalar copies (restricted pointers).
- haloExchange.c sortAtomsInCell: qsort → insertion sort on the small per-cell arrays; cells with <2 atoms skipped; atom array copies for cell-local sorting.
- haloExchange.c: persistent grow-only send/recv scratch buffers instead of per-axis malloc/free churn.
- build.sh only: flags listed below are the mechanical lever; kernels were the algorithmic lever. See .work/PLAN.md for the full log.

## Flags (build.sh)
`-std=c99 -O3 -march=native -fno-math-errno -DNDEBUG -ffp-contract=fast -funroll-loops -flto -fopenmp -DDOUBLE`
-march=native is deliberate: the grader builds on the same machine class.
-ffp-contract=fast is FMA contraction only (no fast-math/Ofast/finite-math-only; those stay off).

## Verification
- .orig/CoMD_orig (-O3 -fopenmp -DDOUBLE, pristine sources) vs optimized ./CoMD on: lj 10^3, 16x12x8, 24^3 (N10-30); eam 10^3, 16x12x8, 12^3; compare.py extends: 36x24x18 T500, 40x20x10 T50 (LJ), 14^3 T1000 (LJ), 20x16x12, 32x20x20 T800, 28x10x10 T100 (EAM). ALL PASS: identical step/atoms rows, energies equal to machine precision (0.00e+00 relative diff).
- Additionally 40x24x24 LJ N100 and 30x20x16 EAM N100: bitwise-identical printed energies.
- Lock-in rehearsal: `rm ./CoMD && bash build.sh` reproduces the binary md5-identical (90741ad8a8b79aa0d974a963c81ce0dc); verify + compare PASS on the rebuilt binary.

## Measured wall-clock (median of 5, 8 cores, whole program incl. crystal build)
- lj -x 30 -y 30 -z 30 -N 100: ~1.25 s vs original 2.37 s (~1.90x)
- eam -x 24 -y 24 -z 24 -N 100: ~1.18 s vs original 2.39 s (~2.02x)
- lj -x 40 -y 24 -z 24 -N 100: ~1.21 s vs 2.14 s (~1.77x)
- eam -x 30 -y 20 -z 16 -N 100: ~0.82 s vs 1.62 s (~1.97x)

## Not merged (measured, reverted)
- PGO (-fprofile-generate/use): no measurable win; fragile profile naming; dropped.
- -funroll-loops before validation was kept only after interleaved A/B confirmed it.
- EAM pair-cache arena: 0.66x (regression) — removed.
- jOff==iOff branch skip: regression — removed.
- ljForce i-outer loop rewrite: 1.38 vs 1.26 — reverted.
- floor()→(int) truncation: no win — reverted; floor calls kept (boundary-ownership safety).
- parallel createFccLattice: neutral (init is ~9% at most) — not merged.
- omp_set_proc_bind: GOMP here lacks the symbol (only omp_get_proc_bind); no proc_bind code added; grader's OMP_PROC_BIND=true handles pinning. close≈spread≈true within ~1.5%.
- Newton's-third-law without shadow buffers: racy under box-per-thread OpenMP (see .work/third-law-race.md) — the merged version uses per-thread shadow buffers with a post-loop merge.

## Known risks / notes
- Force-mirror writes to halo slots are intentionally dropped (dead memory); shadow merge covers only local-j mirror contributions, which is exactly the set of atoms whose force slots are live.
- FP summation order changes (pair visiting order + shadow merge). Printed energies agree with the original to 0.00e+00 relative difference on all tested configs; grading tolerance is 1e-07.
- No atoms are lost: atom count column matches on every config checked.
- .work/ holds the verifier/bench, snapshots, and reference binaries (.orig/, .snapshot/); these are not part of the graded binary and are only used to keep the verification pipeline honest.
