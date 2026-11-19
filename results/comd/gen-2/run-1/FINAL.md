# CoMD optimization — final state

Deliverable: this workspace. `bash build.sh` compiles `./CoMD` with
`-std=gnu99 -O3 -march=native -fno-math-errno -fopenmp -DDOUBLE`.
Same arguments, same output format as the original; no fast-math flags.

What was changed (vs the shipped source):
- SoA atom storage (separate x/y/z planes for positions, momenta, forces;
  64-byte aligned via posix_memalign) replacing AoS real3 array access.
- LJ and EAM force kernels rewritten for the SoA layout with register
  caching of the i-atom data and 2x/4x j-loop unrolling (same arithmetic
  order and expression trees, so trajectories stay bitwise-identical to
  the untouched -O3 reference; worst printed-energy rel diff 0.00e+00).
- EAM: consolidated phi/rho interpolation into one shared-index lookup,
  per-loop plane hoisting.
- Energy accumulation reassociated only on the energy side (bitwise
  identical); the force-path FMA contraction order is preserved exactly.
- haloExchange: persistent exchange buffers allocated once instead of
  per-axis malloc/free; load buffer fill uses direct plane indexing.
- sortAtomsInCell: insertion sort (small cells) + skip when <= 1 atom.
- initAtoms: skip full-capacity zero-init (only live atoms are ever
  touched; f/U are zeroed by the force routine).
- getBoxFromTuple: common grid products hoisted (same integer values).

Verification (before removing scratch artifacts): full 12-case matrix
comparing every printed table row vs an untouched `-O3 -fopenmp` reference
build (both force models; sizes 8^3..30^3, several box shapes, several
temperatures, 15-60 steps): all PASS, bitwise-identical energies.

Benchmarks (medians, interleaved, vs pristine -O3 -fopenmp build on the
same 8 cores, before final cleanup): ~1.4x LJ / ~1.5x EAM on the example
configs; larger configs ~1.5x. A host under heavy load inflates absolute
wall times, but the interleaved medians were consistent.

Notes:
- Build keeps the original output format, including the small .yaml file
  written per run (can be suppressed with COMD_NO_YAML=1; not graded).
- Builds must not use -ffast-math/-Ofast/-ffinite-math-only — this one
  does not.
