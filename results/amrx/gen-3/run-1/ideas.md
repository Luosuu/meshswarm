True speedups with direct timing (/usr/bin/time):
- Example 40 2000 400 3 10 5 4: orig 2.50s -> v3 0.355s = ~7.0x
- 40 4000 800 3 10 5 4: orig 14.2s -> v3 2.47s = ~5.7x

4000^2 case drops toward the 4x rung territory; larger grids are memory/latency heavy. LOOP phase = 2.05s of the 2.47s wall (4000^2), so stencil loop still dominates.

Ideas worth a quick A/B test (you own canonical):
A. -mprefer-vector-width=512 (Zen4 double-pumps AVX-512; likely neutral but cheap to test).
B. If A neutral/positive, keep the better one in build.sh.

Not worth it: more fork/join reduction (negligible), THP (notes say rejected), binding (rejected), tiling (rejected), collapse (rejected).

Verification still required for any flag change: shared/verify.py matrix, worst rel must stay << 1e-9 (currently ~4.6e-11).
