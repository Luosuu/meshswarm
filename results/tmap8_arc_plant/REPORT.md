# DOE INL `TMAP8` ARC Whole-Plant Optimization Report (`examples/tmap8_arc_plant`)

## 1. Authentic DOE National Laboratory Target

Instead of a synthetic plant model, this benchmark directly targets the **official Idaho National Laboratory (`idaholab/tmap8`)** whole-plant fusion tritium fuel-cycle model:
- **Upstream file**: `test/tests/fuel_cycle_Meschini/fuel_cycle.i` (521 lines, SHA256 `b30426042d7c102397510c074f1c5454e18bd89d947c67ea0a3b1aaa31b2b0e6`), modeling the **11 interconnected subsystems** of an MIT/CFS **ARC-class 505.6 MWf D-T fusion power plant** over 3 years of 1800-second pulses (*Meschini et al., Nuclear Fusion 63, 2023*).
- **Untouched DOE Oracle**: Every candidate 15-parameter plant design is graded by invoking INL's compiled C++/MOOSE solver (`tmap8-opt`, `idaholab/tmap8:2026.05.10-0cdef71`, `snapshot-20-10-27-53529-g9fb74b436d`) inside an air-gapped Apptainer container (`--nv --containall --no-home --net --network none`) across **5 pulsed 3-year scenarios** (`12,470` MOOSE `PJFNK` time steps total, including an unseen held-out combined multi-subsystem off-normal jitter check).

## 2. Why INL's Hand-Tuned Baseline Fails Whole-Plant Safety & Robustness Gates

In INL's official documentation (`fuel_cycle_Abdou/index.md` & `fuel_cycle_Meschini/index.md`), parameters were tuned manually (*"iterating by hand"* / `fuel_cycle_gui.py`) under a 20-day pulse-cutoff approximation. Evaluated across all 5 pulsed scenarios on `tmap8-opt` (Rivanna Slurm Job `23418030` on `NVIDIA A40`), the baseline incurs **14 safety & self-sufficiency violations (`0.0 / 10`)**:
1. **In-Process Vulnerable Tritium Overload**: Active non-storage tritium reaches **`462.7 g`** (`557.2 g` under `tbe_drop`), **2.1× above** the `220 g` vulnerable active-inventory cap.
2. **Steam-Loop Heat Exchanger Permeation Breach**: `T_05_HX` reaches **`310.4 mg`** nominally and **`537.0 mg`** under `tes_fouling` (**5.4× above** the `100.0 mg` radiological barrier limit).
3. **Off-Normal Storage Depletion Breach**: Under a 20% plasma burn efficiency excursion (`tbe_drop`), `T_10_storage` drops to **`0.6012 kg`**, breaching the 24-hour outage reserve requirement.
4. **Excessive Scarce Startup Tritium**: Requires **`1.14 kg`** of scarce CANDU tritium (`> 1.00 kg` budget).

## 3. Verified Results on UVA Rivanna (17 Swarms Across 6 Generations, `$3.92` Total Cost)

### 3.1 GPU Differentiable Digital Twin vs CPU `tmap8-opt` (`--selftest`, Job `23418030`)
- Single-scenario CPU `tmap8-opt` runtime (with native `gcc-toolset-13` + `OpenMPI` FParser C++ JIT): **`~33–48 s`** per trajectory (**`44.1 days`** CPU time for `115,456` trajectories).
- **GPU JAX Differentiable Backward-Euler Twin (`reference_jax_optimizer.py`)**: Evaluated **`115,456` 3-year pulsed trajectories + exact 15D gradients in `24.43 s`** on a single `NVIDIA A40` GPU (**`~156,000×` speedup**), achieving **`8.221 / 10`** (`J_econ = 96.66 MWe`, `P_net = 113.76 MWe`, `I_0 = 0.889 kg`, `HX_max = 14.5 mg`, `t_double = 0.75 yr`) across all 5 untouched `tmap8-opt` scenarios.

### 3.2 `meshevolve` 6-Generation Lineage Progression (`Jobs 23418230, 23429870, 23430931`)
- **All 7 uninstructed runs (`Gen 1` × 2 + `Gen 2–6` no-lesson `control` × 5) scored `0.0 / 10`**: Without evolutionary organizational lessons, swarms repeatedly fell into analysis paralysis, built 3 competing duplicate twins, clobbered shared files, or stalled in population-based `differential_evolution` without ever committing a passing design to `plant_design.json`.
- **Generation 2 Breakthrough (`gen-02/run-1`, `$0.23`, `2.705 / 10`)**: Guided by Gen 1's Reviewer lesson (*"write an early gated candidate in the first third of the budget"*), `run-1` passed all 5 `tmap8-opt` scenarios (`J_econ = 80.11 MWe`, `P_net = 99.50 MWe`, `I_0 = 1.000 kg`, `inproc_max = 163.2 g`, `tes_fouling HX = 97.05 mg <= 100.0 mg`).
- **Generation 3–5 Organizational Diagnosis (*"Break the Write Monopoly"*)**:
  - Fixing `ShellRunner` from shared `gpu:0` locking to per-workspace locking (`--gpus ""`) boosted swarm throughput **>5×**, revealing a second organizational bottleneck: Gen 4's *"single designated file owner"* rule caused passing candidates (`agent2_best.json` at `I_0 = 0.800 kg`) to stall at handoff.
  - Gen 5's Reviewer autonomously coined the fix: **"Break the write monopoly: whoever holds a gate-passing candidate writes the deliverable immediately."**
- **Generation 6 Champion (`arc_gen6_showdown/gen-01/run-1`, `$0.47`, `4.507 / 10` — Surpassing Human Expert Baseline!)**:
  - Combining penalized `L-BFGS-B` local gradient search with the *"finder-writes"* protocol, `run-1` wrote a verified design mid-run (`t ~ 950s`) that **passed all 5 untouched `tmap8-opt` scenarios (`12,470` MOOSE steps, zero violations)** and **beat our human hand-tuned conservative baseline (`4.453 / 10`, `85.36 MWe`)**:
    - **Levelized Net Economic Output (`J_econ`)**: **`85.52 MWe`** (`P_net_elec = 101.45 MWe`, `AF = 85.83%`)
    - **Scarce Startup Tritium (`I_0`)**: **`0.825 kg`** (**`-27.6%`** vs INL `1.140 kg` — lower than both the human baseline `0.860 kg` and GPU reference `0.889 kg`!)
    - **Peak HX Boiler Tritium**: **`12.63 mg`** nominal (**`-95.9%`** vs INL `310.4 mg`; `77.07 mg` under `tes_fouling`; `41.05 mg` under unseen `heldout_combined_jitter`)
    - **Peak Active Tritium**: **`163.3 g`** (**`-64.7%`** vs INL `462.7 g`)
    - **Tritium Doubling Time**: **`0.516 yr`** (`6.2 months` — **`2.73×` faster** than INL `1.41 yr`)
  - Non-intuitive co-design trade-off discovered autonomously by `run-1`: **slowing** `FLiBe` blanket circulation (`resident_time_BB` `4,500 s -> 12,633 s`) saved **`2.02 MWe`** of MHD pumping power at negligible active-inventory cost, funding high-efficiency `TES` (`eta_2 = 95.7%`), fast `ISS` (`2,918 s`), and `77.1%` Direct Internal Recycling (`f_DIR`).
