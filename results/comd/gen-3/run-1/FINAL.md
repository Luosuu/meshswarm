# CoMD optimization — summary of changes

Deliverable: this workspace. `bash build.sh` (gcc -std=c99 -O3 -march=native
-fno-math-errno -ffp-contract=fast -fopenmp -DDOUBLE -I. *.c -lm) produces ./CoMD with the same
CLI and printed output format as the original. Energies match the original
`-O3 -fopenmp` build to relerr well within 1e-07 on 27 checked
configurations (verify.sh: 14 configs incl. large/hot/cold/rod/tiny;
verify_extra.sh: 13 more incl. T=1200/T=50/T=2000, 32^3, 64x8x8, 8x8x48, 14^3/5-step). No -ffast-math / -Ofast / -ffinite-math-only anywhere.

Measured speedup vs ref/CoMD (original, -O3 -fopenmp), median of 5 interleaved
reps, 8 cores: LJ 30^3 ~1.99x, EAM 24^3 ~2.02x.

What changed (nothing touched under ref/):

1. Compiler flags: -O3 -march=native -fno-math-errno -ffp-contract=fast added
   to baseline -O3 -fopenmp. FP contraction only fuses existing mul/add pairs
   into FMA (no reassociation, no -ffast-math); verify 27/27 configs PASS with
   the change. LTO / unroll / forced vector width measured as noise.
2. Newton's third law in ljForce.c and eam.c: every unordered pair is
   computed exactly once (same-box j>i; both-local jBox>iBox handled from one
   side only; halo partners use a gid tiebreak so cross-rank pairs are
   consistent). Reverse updates go through thread-private accumulators
   (nThreads x maxTotalAtoms), merged once per pair loop into atoms->f/U/
   rhobar. No per-thread pair re-evaluation.
3. No f/U zero passes over the padded arrays: accumulators are zeroed only
   for occupied local slots; atoms->f/U are overwritten (not accumulated) by
   the merge. The merge also re-zeros force accumulators for EAM pass 3.
4. gid -> local-slot map (g2l, grow-only) so reverse (halo) updates find the
   owner slot without a search. memset reset bounded by nGlobal.
5. EAM: the F'(rhobar) halo exchange was eliminated entirely (dfEmbed no
   longer needs exporting: pass 1 accumulates rhobar correctly under N3L,
   pass 3 uses each partner's dfEmbed directly). Pass 1 computes the pair
   phi/rhobar (unused derivatives dropped via interpolateValue); pass 3 uses
   interpolateDeriv; pass 1/3 rho interpolation reduced to value-only /
   derivative-only variants with identical FP op sequences (bit-identical).
   All table lookups remain interpolation-identical.
6. EAM interpolate()/interpolateValue/interpolateDeriv are `static inline`
   (the call was previously out-of-line).
7. updateLinkCells (linkCells.c): classify-into-box pass is parallel (was
   serial floor+branch per atom); moves stay serial but are rare. Persistent
   grow-only jBoxScratch instead of per-step alloc.
8. haloExchange.c: single-rank path skips the memcpy of send/recv buffers
   when getNRanks()==1 (buffers fed straight into unloadBuffer; op order
   unchanged); persistent grow-only exchange scratch.
9. sortAtomsInCell call removed from redistributeAtoms (not needed).
10. EAM pass-3 force accumulation merged once at the end (same N3L scheme).
11. MAXATOMS left at 64 (48-slot experiment failed verification; observed
    max occupancy ~32). haloExchangeSingleRank fast path reverted after
    verify failures; EAM pair-caching probe rejected on cost.

See snap/current-good for a clean source snapshot of the final tree.
