# DOE INL `TMAP8` ARC-Class Fusion Power Plant Whole-Plant Optimization

## Background

`fuel_cycle.i` is the untouched **Idaho National Laboratory (`idaholab/tmap8`)** MOOSE input deck modeling the 11-subsystem pulsed deuterium-tritium (D-T) fuel cycle of an **ARC-class 505.6 MWf fusion power plant** over 3 years of 1800-second pulses (*Meschini et al., Nuclear Fusion 63, 2023*).

In INL's unoptimized baseline (`plant_design.json`), parameters were tuned manually under a 20-day pulse-cutoff approximation:
- Active non-storage tritium inventory reaches **463 g** (violating the **220 g** vulnerable in-process safety cap),
- Primary Heat Exchanger (`T_05_HX`) tritium reaches **310 mg** nominally and **537 mg** under TES fouling (violating the **100 mg** steam-loop permeation limit),
- Startup tritium requires **1.14 kg** (exceeding the **1.00 kg** scarce-supply budget), and
- Under a 20% plasma burn efficiency drop (`tbe_drop`), storage `T_10_storage` breaches the 24-hour outage reserve floor.

## Objective & Deliverable

Write your optimized 15-parameter plant design to **`plant_design.json`** (a JSON object with the exact 15 keys defined in `plant_economics_ref.py`).

Do **not** modify `fuel_cycle.i`, `plant_economics_ref.py`, or `evaluate_local.py`.

## Sandbox Environment & Suggested Strategy

1. **Ground-Truth Industrial Solver (`tmap8-opt`)**:
   - Available directly on `PATH` (`tmap8-opt -i fuel_cycle.i ...`).
   - Run `python evaluate_local.py` at any time to verify `plant_design.json` with real `tmap8-opt` across the 4 public pulsed scenarios (`25` days = `1,200` explicit 1800-second pulses + 3-year trajectory per scenario, ~50s wall-clock).
   - Because each `evaluate_local.py` call invokes MOOSE `PJFNK` across thousands of pulsed time steps on CPU, brute-force black-box optimization directly on `tmap8-opt` is slow.

2. **GPU Differentiable Digital Twin (`jax` + `optax` + `scipy`)**:
   - `python` has CUDA-enabled `jax` (`float64` enabled), `diffrax`, `optax`, `equinox`, and `scipy` on GPU.
   - Read `fuel_cycle.i` and `plant_economics_ref.py`: the 11 coupled `[ScalarKernels]` ODEs are piecewise linear time-invariant within each pulse (`AF * 1800 s` active burn phase + `(1 - AF) * 1800 s` dwell phase), or can be integrated via vectorized/differentiable GPU solvers + exact `jax.grad`.
   - Build a fast GPU twin to explore thousands of candidate designs and gradients across all scenarios, **calibrate/verify discrepancies** against `tmap8-opt` (note MOOSE `ODETimeDerivative` uses implicit Backward Euler steps controlled by `dt_function`!), and validate your final design with `python evaluate_local.py`.

## Hard Safety & Self-Sufficiency Gates (Zero-Multiplier — Any Breach = Score `0.0`)

 Evaluated by the external grader using untouched `tmap8-opt -i fuel_cycle.i`:
1. **Startup Tritium Cap**: `initial_inventory <= 1.00 kg` and `p_net_elec_mwe > 0`.
2. **Storage Reserve Autonomy**: Across `nominal` (`>= 1.00 * I_reserve`) and off-normal stress scenarios (`tbe_drop`, `tes_fouling`, `blanket_degraded`: `>= 0.85 * I_reserve`), `min_t T_10_storage(t)` must never breach the reserve floor, and `tmap8-opt` must complete all 3 years without triggering `Terminator`.
3. **In-Process Vulnerable Tritium Cap**: Peak active inventory $\max_t \sum_{i \neq 10} T_i(t) \le 0.220\text{ kg}$ (`220 g`).
4. **Heat-Exchanger Radiological Cap**: Peak `T_05_HX(t)` $\le 1.0 \times 10^{-4}\text{ kg}$ (`100 mg`).
5. **Tritium Doubling & 3-Year Surplus**: Doubling time $t_{\text{double}} \le 1.50\text{ yr}$ in `nominal`, $\le 2.50\text{ yr}$ in `blanket_degraded`, and 3-year storage gain $T_{10}(3\text{ yr}) - I_0 \ge 2.00\text{ kg}$ in `nominal`.
6. **Held-Out Combined Jitter Robustness**: The external grader also runs a 5th held-out scenario combining moderate simultaneous perturbations across subsystems (`tbe`, `eta_2`, `f_HX_to_DS`, `TBR`, `AF`). Do **not** park active constraints within `1e-6` of a cliff — maintain sensible engineering safety margins (~2–5% buffer on active limits) and verify sensitivity around your optimum!

## Whole-Plant Economic Score (`0.0` to `10.0`)

Designs passing all hard gates are scored on **Levelized Net Economic Electric Output (`j_econ_mwe`)** from `plant_economics_ref.py`:
$$\text{score} = 10 \times \text{clip}\!\left(\frac{J_{\text{econ}} - 72.0}{102.0 - 72.0},\; 0,\; 1\right)$$
where $J_{\text{econ}} = P_{\text{gross,avg}} - P_{\text{parasitic}} - P_{\text{capital,equiv}}$ rewards higher time-averaged net electricity and lower scarce startup-tritium & equipment capital.
