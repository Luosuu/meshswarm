# CryoHDTColumn16 JAX GPU ensemble solver — approach & measurements

Deliverable: `solver_jax.py` (owned by agent-3). Reference pipeline & verification harness: agent-2.

## Model summary
- 16-stage closed-recycle cryogenic H2/HD/T2 distillation column.
- 32 differential states: x_H2[1..16], x_HD[1..16]; x_T2 = 1 - x_H2 - x_HD.
- 16 algebraic bubble-point equations (Antoine-like, T in Kelvin-ish units): sum_i x_i*Psat_i(T) = P_stage.
- V_eff(t) = V_boil*(1 + 0.08*sin(0.25*t)); flooding efficiency eta = 0.92 if V_eff <= 2.15 else 0.92/(1+1.8*(V_eff-2.15)^2).
- Recycle: stage-8 feed D_rec = d_rec_frac*V_eff from stage 1 (overhead), L_rect = V_eff - D_rec.
- M = [4, 1*14, 6] (stage 1 condenser mass, stage 16 reboiler mass).

## Reference data
- Generated with `omc` (dassl, tolerance=1e-10, stopTime=25.0) per-case model copies with rewritten defaults; stored in `shared/refs/<case>/CryoHDTColumn16_res.csv` plus `shared/refs/<case>.json` (final observables + final states) and `shared/refs/index.json`.
- Cases: ex1, ex2, near_thresh, deep_flood, low_boil, high_press, low_press, hi_boil_lo_rec, lo_boil_hi_rec, thresh_boil, extreme_flood, low_dp (12 regimes incl. flooding transitions).

## Verification harness
- `scratch/verify_gates.py`: fresh-imports solver_jax, scans for forbidden constructs, checks shape/finiteness, conservation drift over 1000 perturbed-parameter trajectories, per-output parity vs refs, grad-vs-jvp / grad-vs-jacrev / closure FD / macro FD, and times median-of-5 1000-trajectory forward passes + 64-trajectory grad pass against the wall-time ladder.
- `scratch/gates.md`: checklist, ticked only on fresh runs.

## Results
(pending solver delivery)
