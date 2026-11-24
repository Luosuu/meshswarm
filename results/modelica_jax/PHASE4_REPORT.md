# Phase 4 Report: Closed-Loop Differentiable Control, 1M-Episode GPU RL, `< 1 ms` Real-Time Distillation & Whole-Plant Co-Design

Building directly on the **Swarm-generated winning JAX operators from Phase 3**—specifically
`tier2_transform_hx/gen-2/run-1/solver_jax.py` (`10.0/10` ORNL `TRANSFORM` counter-current
`PbLi`/`FLiBe` heat exchanger + Arrhenius/Sieverts tritium permeator) and
`tier3_hdt_column/gen-2/control/solver_jax.py` (`8.0/10` 16-stage cryogenic
$\text{H}_2/\text{HD}/\text{T}_2$ distillation MESH column)—Phase 4 closes the end-to-end
Fusion Pilot Plant (FPP) digital-twin loop across three benchmarks on a single **NVIDIA
A100-SXM4-40GB** (`udc-an28-7`, AMD EPYC 7742 host, Slurm job `23390494`; full JSON metrics in
`phase4_metrics.json`).

---

## 1. Closed-Loop Transient Plant Control: Differentiable Policy Optimization vs. `1,024,000`-Episode GPU PPO

- **Environment (`examples/modelica_jax/phase4/plant_env.py`):** 30-step (`30 s`, $\Delta t_{\text{ctrl}} = 1.0\text{ s}$)
  closed-loop control of the 15-state `TRANSFORM` `PbLi`/`FLiBe` heat exchanger & tritium
  extraction loop under simultaneous stochastic disturbances:
  1. Plasma fusion power ramp ($\Delta T \in [5, 32]\text{ K}$);
  2. Transient ELM heat pulse ($A_{\text{ELM}} \in [0, 25]\text{ K}$ at $t_{\text{ELM}} \in [7, 22]\text{ s}$);
  3. Partial primary `PbLi` coolant pump trip retaining $45\%\text{–}90\%$ flow at $t_{\text{trip}} \in [8, 19]\text{ s}$.
- **Objective:** Hold secondary `FLiBe` turbine-inlet temperature at $T_{2,\text{target}} = 892.0\text{ K}$,
  enforce primary `PbLi` structural ceiling $T_{1,\text{out}} \le 948.0\text{ K}$, track getter
  tritium recovery ($1.25\text{ mol}$), and minimize pump/vacuum power & valve slew across a
  held-out 1,024-scenario test suite.

| Controller / Training Method | Training Episodes | A100 Wall Time | Throughput (ep/s) | Held-Out Mean Cost | `FLiBe` Outlet RMSE | Max Tritium Drift |
|---|---|---|---|---|---|---|
| Open-Loop Nominal Constant | — | — | — | `280.83` | `14.93 K` | `1.63e-15` |
| **Differentiable Physics Policy (`SHAC`/BPTT, 1,475p)** | **`51,200`** (`6,400` to `< 7.0`) | **`5.09 s`** (`0.67 s` to `< 7.0`) | `10,051 ep/s` | **`2.05`** | **`1.37 K`** | **`1.78e-15`** |
| **Model-Free `PPO` + GAE (Black-Box)** | **`1,024,000`** | **`6.75 s`** | **`151,690 ep/s`** | **`1.99`** | **`1.34 K`** | **`1.78e-15`** |

**Key findings:**
- **Million-episode RL in under 7 seconds:** Because the Swarm's `TRANSFORM` operator compiles into
  a branch-free `jax.lax.scan` kernel, **1,024,000 full 30-second plant episodes (`122.9M` RK4 +
  Newton sub-steps) finish in `6.751 s` (`151,690 episodes/s`)** on one A100—compared to ~74 hours
  for 1.024M sequential OpenModelica CPU runs (**`~39,000×` speedup**).
- **20× sample-efficiency gain from exact physics gradients:** Differentiating directly through the
  plant physics reaches `RMSE = 2.24 K` (`cost = 6.77`) in `6,400` episodes (`0.67 s`) and
  `RMSE = 1.37 K` (`cost = 2.05`) in `51,200` episodes, while preserving closed-loop tritium
  inventory to `1.78e-15`.

---

## 2. Real-Time Edge Controller Distillation (`< 1 ms` Deterministic CPU Gate)

To satisfy real-time plant PLC/CPU feedback latency guardrails (`< 10,000` parameters, `< 1 ms`
deterministic execution per tick), `examples/modelica_jax/phase4/distill_mlp.py` solves exact
open-loop differentiable MPC trajectories across 1,024 disturbance scenarios (`30,720` state-action
pairs) and distills + closed-loop fine-tunes an ultra-compact **`9 -> 16 -> 16 -> 3` MLP (`483`
parameters, `< 5%` of the 10K ceiling)** exported to standalone NumPy weights (zero JAX dispatch):

- **JAX GPU vs. standalone CPU NumPy equivalence:** `2.22e-16` max action difference;
- **Closed-loop plant performance (1,024 unseen scenarios):** `mean_cost = 2.69`, `FLiBe` outlet
  `RMSE = 1.51 K` (down from `14.93 K` open-loop), `max_tritium_drift = 1.48e-15`;
- **Single-thread deterministic CPU latency (10,000 sequential ticks, `OMP_NUM_THREADS=1`, AMD EPYC 7742):**
  - **Mean:** `11.69 us` (`0.0117 ms`)
  - **p50:** `11.60 us`
  - **p99:** `15.45 us`
  - **p99.9:** **`17.47 us` (`0.0175 ms` — `57.2×` faster than the `1,000 us` = `1.0 ms` real-time gate)**.

---

## 3. End-to-End 3-Solver Coupled Core–Blanket–Distillation Co-Design

`examples/modelica_jax/phase4/codesign.py` couples three differentiable physics solvers inside a
single `jax.value_and_grad` graph:
1. **1D radial burning-plasma transport (`TORAX`-style 12-zone diffusive D-T density & temperature PDE)**
   with Bosch–Hale reactivity, alpha heating, and bremsstrahlung radiation;
2. **Swarm Tier-2 `TRANSFORM` `PbLi`/`FLiBe` blanket heat exchanger & tritium permeator** driven by
   14.1 MeV fusion neutron power;
3. **Swarm Tier-3 16-stage cryogenic $\text{H}_2/\text{HD}/\text{T}_2$ distillation column** driven
   by fueling/exhaust isotope feed.

Starting from an infeasible midpoint design (`t_in_pbli = 1047.3 K > 965.0 K` ferritic-steel limit),
coupled `L-BFGS-B` using exact 3-solver reverse-mode gradients (`coupled_3solver_grad_fd_rel_err =
2.51e-7` vs. central finite differences) is compared against gradient-free Nelder–Mead at an equal
60-evaluation budget:

| Method | Evals | Net Electric $P_{\text{net,e}}$ | Fusion Power $P_{\text{fus}}$ | Plasma $Q$ | `PbLi` Inlet ($\le 965\text{ K}$) | Bottom $\text{T}_2$ Purity ($\ge 99.97\%$) | Penalized Objective |
|---|---|---|---|---|---|---|---|
| Nominal Midpoint | 1 | 276.24 MWe *(infeasible)* | 1377.98 MW | 35.79 | 1047.31 K *(violated)* | 99.978% | `+135,209.9` |
| Gradient-Free Nelder–Mead | 60 | 250.85 MWe *(infeasible)* | 1303.74 MW | 33.69 | 1037.01 K *(violated)* | 99.978% | `+103,459.7` |
| **Coupled 3-Solver Exact-Gradient L-BFGS** | **59** | **`172.66 MWe` *(feasible)*** | **`965.52 MW`** | **`74.54`** | **`964.70 K` *(satisfied)*** | **`99.979%` *(satisfied)*** | **`-171.71`** |

Exact end-to-end gradients simultaneously throttle auxiliary heating to `12.95 MW` (raising plasma
$Q$ to `74.54`), raise `PbLi`/`FLiBe` coolant flows (`123.56` and `17.78 kg/s`) to sit cleanly
against the `965.0 K` structural ceiling (`964.70 K`), and hold cryogenic reboiler boilup at the
minimum energy needed (`1.466 mol/s`) to maintain `99.979%` $\text{T}_2$ product purity.
