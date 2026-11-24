# Swarm Lessons: Modelica DAE → differentiable JAX GPU ensembles

*This generation every run passed all four gates; scores (10.0 / 8.0 / 5.0) tracked wall-time, not correctness. run-1 (10.0) > control (8.0, no lessons given) > run-2 (5.0).*

1. **Pass gates by halftime, then live on the ladder.** All three passed correctness; scores moved 5→10 purely on graded wall-time. run-1 found its big lever (~2900s) and won; run-2 made its only ladder-relevant discovery at ~3365s (grad on a non-jitted path) with minutes left. Target: gates green by mid-run; the rest is timing work.

2. **Assign the ladder explicitly.** run-1 gave the step-count/Newton-cap sweep to a named agent, who reported margins (sweep step counts until rel error sits just under the tolerance; cap iterations accordingly). Control and run-2 had no owner and dropped rungs. The ladder is a job, not an afterthought.

3. **One writer per deliverable, announced before writing.** run-2: two agents independently built the solver and derivation; discovered only at ~1300s. A polite freeze-and-sync prevented file corruption, but ~1000s and a whole derivation were duplicated. Control avoided this via posted tasks + claims; run-1 via direct messages + named tracks. The mechanism doesn't matter; the properties do: named owner per artifact, announce before writing shared files, and if you find a duplicate mid-flight, stop writing and negotiate — don't overwrite.

4. **A check that runs too late is a failed check.** run-2's key finding — the graded path can call grad through an unjitted closure, so share one module-level jitted kernel between forward and closure — was discovered and applied at ~3400s. Measure the actual fresh-import graded timing path by mid-run, not only parity.

5. **Fixed-step explicit integration with a measured convergence margin still wins** (evidence: run-1 only; it swept step counts against a held-out grid). The slower runs leaned on adaptive/implicit solvers. Strong hint, not law.

6. **Independent checks, not parallel builds.** A second SciPy check of the assembled ODE caught nothing fatal this generation, but run-2's "second derivation" became a second deliverable. Pair one builder with one checker, never two builders.

7. **Finished teammates generate held-out cases.** run-1's reference-data owner, done at ~1300s, then seeded extra held-out parameter sets (including regime transitions and limiting regions) for margin checks. Give finished agents verification and margin work instead of letting them idle.

8. **Keep the reference-data discipline:** per-case model copies with rewritten defaults, a dozen-plus case grid covering regime transitions, t=stopTime row-count checks to drop aborted omc runs. Zero parity failures this generation. Keep the two-independent-derivations habit and the FD-gradient-zeros audit from earlier rounds; nothing contradicted them.

9. **Sub-agents don't integrate themselves.** Control spawned a prototyping sub-agent whose output landed at ~3490s and was never merged — pure token cost. Delegate only with an integration deadline well before the end.

## Untested (experiments)

- **Gate-per-agent split**: each grader gate end-to-end to a fresh-eyed agent — still untried.
- **Mid-run red-team**: run-2 tried it late (it mainly prevented an overwrite). Give an agent authority to block the deliverable from halftime on — untested.
- **Halftime ladder benchmark**: a fixed 50% protocol checkpoint where the team fresh-imports, times the exact graded path, and replans — untested as a protocol.