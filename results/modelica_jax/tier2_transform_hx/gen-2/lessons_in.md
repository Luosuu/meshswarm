# Swarm Lessons: Modelica DAE → differentiable JAX GPU ensembles

*This generation all three runs passed all four gates; scores (9.0/5.0/4.0) tracked wall-time ladder performance, not correctness. Keep the discipline, add speed iteration.*

1. **The ladder, not the gates, decides your score once correctness holds.** All three runs passed every gate with omc_err ≈ 1–3e-8 and mass_err ≈ 1e-15. The 9.0 run won by shrinking forward+grad time after gates were green. Deliver a passing solver by mid-run, then spend the back half on the speed ladder — don't stop at "parity passes."

2. **Speed comes from fixed-step explicit integration plus a measured convergence margin.** The winning run swept fixed-step RK4 step counts and capped the implicit root-solve's Newton iterations once the grid showed e.g. ~1e-6 rel error vs a 3e-4 gate. The two slower runs leaned on adaptive implicit solvers and landed at 1.5–3.7s. Quantify the largest error you can afford, then shrink the integrator to fit it.

3. **One writer per deliverable, independent refs and harness owners.** Every passing run converged on this without tasks/claims — direct messages sufficed, hub_share ~0.7–0.8 was harmless. The graded module's owner never builds the grader harness.

4. **Validate the reference set itself; trust nothing truncated.** One run's grid contained a case where omc aborts mid-run on an asserted bound, yet a final-obs JSON existed; caught by cross-checking CSV row count against stopTime. Drop aborted cases. The working omc recipe remains: per-case model copies with rewritten parameter defaults, grid ~7–15 sets including limiter kinks and regime transitions — no parity failures this generation. Keep the grid; it is cheap insurance.

5. **Two independent derivations beat one.** Each run paired the JAX solver with an independent Python/SciPy prototype checked against reference CSVs (≤2e-8). Disagreements (signs, pairing) were settled by opening a reference CSV, never by debate. This caught semantic bugs before GPU time was burned.

6. **Audit gradient-gate zeros before panicking.** One harness author found outputs insensitive to one parameter, making its sum-of-outputs gradient exactly zero and central FD a 0/0 noise source. Fix the FD denominator floor/harness, not the solver. Know which parameters move which outputs.

7. **A finished teammate goes idle.** In the control, one agent slept 70% of the run. If your piece is done, explicitly claim the speed sweep or a red-team pass — don't wait to be asigned.

8. **Fresh-import and re-run the full grader flow at the end.** Keep a checklist harness (parity vs DASSL CSVs, conservation over the ensemble, grad vs jvp/FD, warm-timed median of 5) and tick it only with fresh runs; re-check the deliverable for forbidden constructs (subprocess/omc/stop_gradient) as the last act.

## Untested (experiments)

- **Gate-per-agent split**: assign each grader gate end-to-end to a different agent with fresh eyes (still untried).
- **Red-team finale**: one agent's only job in the last 30 minutes is breaking the deliverable — fresh import, forbidden-construct grep, unseen-regime sweep.
- **Ladder-attack pair**: while the solver owner freezes features, a second agent owns step-count/Newton-cap/dtype sweeps and reports ladder timings every 10 minutes — untested, and the obvious lever given this generation's results.