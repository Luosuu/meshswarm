# CoMD optimization — final state

**Deliverable:** this workspace. `bash build.sh` produces `./CoMD` (gcc
`-std=c99 -O3 -march=native -fno-math-errno -fopenmp -DDOUBLE`). No
`-ffast-math`/`-Ofast`/`-ffinite-math-only` anywhere. Same arguments, same
table format as the original.

## Result (both force models, vs pristine `-O3 -fopenmp` build in `snapshot_orig/CoMD_O3`)

Interleaved medians, 6 reps each:

| case          | new (s) | orig (s) | speedup |
|---------------|---------|----------|---------|
| LJ 30³, N=100 | 1.227   | 2.479    | 2.02x   |
| EAM 24³, N=100| 1.194   | 2.475    | 2.07x   |
| LJ 40×20×30   | 1.074   | 2.225    | 2.07x   |
| EAM 24×30×20  | 1.230   | 2.561    | 2.08x   |

Larger/other shapes verified: LJ 48³ (N=100): 5.74s vs 10.55s (1.84x);
LJ 48³ N=30: 1.75s vs 3.04s (1.74x); EAM 36³ N=50: 2.00s vs ~4.0s.
Correctness gate (last table row step/atoms/total/potential/kinetic vs the
pristine binary, rel diff ≤ 1e-07): **6/6 verify, 16/16 matrix, 13/13 edge
configs PASS** (thin-z, odd shapes, tiny, T=10/1000, larger crystals).
Energies match the original to ~1e-12 (accumulation-order rounding only).

## What was changed

1. **Half-pair (Newton's-third-law) force kernels** in `ljForce.c` and
   `eam.c`: unordered pairs computed once per i-box pass instead of twice;
   both endpoints updated via thread-private accumulators, merged by a
   sparse reduction pass. Energy accounting kept identical to the original
   (local-local pair = 1.0×eLocal, local-halo pair = 0.5×eLocal).
2. **Thread-private accumulation with sparse reduce**: per-thread `pf`
   (real3 forces) and `prh`/`pU` (EAM only now), persistent across calls,
   zero-preserved by the reduction, which only touches live local slots.
3. **Dead work removed**: per-atom potential `U` accumulation removed
   entirely from both kernels and the pre-zero sweep for `f`/`rhobar`/
   `dfEmbed` removed — the reduction pass overwrites every live local slot
   with an assignment (halo slots are never read: halo forces are discarded
   and halo `dfEmbed` is rewritten by the force exchange).
4. **`schedule(dynamic,16)`** on the force outer loops (interleaved matrix
   beat dynamic,4/8 and guided).
5. **EAM interpolation** inlined (`static inline`); pair-site constants
   hoisted.
6. **linkCells/halo** (agent-1): persistent halo-exchange send/recv
   buffers, insertion sort in `sortAtomsInCell`, parallel zeroing of empty
   halo cells/init, 64-byte aligned allocator, `updateLinkCells` fast path
   (skip bookkeeping when no atom moves).
7. **timestep**: cached per-species inverse mass in `advancePosition` /
   `kineticEnergy`.
8. **build.sh**: `-O3 -march=native -fno-math-errno` (flags alone ≈ 1.05–
   1.15x).

Discarded as slower (interleaved, reverted byte-identically): accumulator
slot shrink to local boxes (LJ −5%); branch-guarded j-side halo stores
(−5%); scalar-ternary energy-accounting variants (noise/loss); simd
pragmas; guided scheduling; `kernel A` LJ restructure (−22%).

## Verification

`./CoMD` last-row energies vs `snapshot_orig/CoMD_O3` match exactly
(1.166049844180 / −1.206866877764 / 0.040817033584 for LJ 30³;
−3.460530041323 / −3.495899676824 / 0.035369635501 for EAM 24³). No atoms
lost. `bash build.sh` reruns clean from the delivered tree.
