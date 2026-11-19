# CoMD optimization summary

Deliverable: this workspace. `bash build.sh` produces `./CoMD` (flags: `-O3 -march=native -fno-math-errno -DNDEBUG -fopenmp -DDOUBLE`; no fast-math/Ofast/finite-math).

Reference for verification: `pristine/CoMD` (untouched copy of the original sources built with `-O3 -fopenmp`). Verification harness: `check.sh` (both force models, sizes 8^3..36^3, non-cubic shapes, temperatures 100..1500; checks last table row: step, atom count, total/potential/kinetic energy to 1e-07 rel diff) and `compare.sh`.

Main changes:
- Force kernels (ljForce.c, eam.c): Newton's-third-law half-pair loops (each pair once); thread-private grow-only force/energy/density accumulator buffers merged over occupied slots only; `restrict`-qualified position pointers; branchless same-box inner-loop start; `interpolateFast()` table lookup in eam.c (clamped floor indices, same formula); `schedule(dynamic,8)` on force-reduction loops; reduced-traversal of link cells.
- linkCells.c (agent-2): updateLinkCells parallelized (per-box parallel compaction + serial append; original placement semantics preserved); slimmed staged MovedAtom struct.
- haloExchange.c (agent-2): persistent/grow-only scratch buffers for send/recv and unloadAtomsBuffer (no per-step malloc); insertion sort replacing qsort in sortAtomsInCell; parallel loadAtomsBuffer (prefix-sum per cell); parallel unloadAtomsBuffer (parallel destination-box computation + box-disjoint writes).
- build.sh (agent-2): `-O3 -march=native -fno-math-errno -DNDEBUG`.

Measured (8 cores, 5 reps median): LJ 30^3 N=100 ~2.27s vs ~4.7s original (~2.1x); EAM 24^3 N=100 ~2.11s vs ~4.5s (~2.1x); 40x20x20 N=50 ~0.69s vs ~1.43s (~2.1x). All energy checks pass on both force models; no atoms lost.
