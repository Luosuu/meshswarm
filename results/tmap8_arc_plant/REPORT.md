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

## 3. Verified Results on UVA Rivanna (23 Swarms Across 8 Generations, `$5.33` Total Cost)

### 3.1 GPU Differentiable Digital Twin vs CPU `tmap8-opt` (`--selftest`, Job `23418030`)
- Single-scenario CPU `tmap8-opt` runtime (with native `gcc-toolset-13` + `OpenMPI` FParser C++ JIT): **`~33–48 s`** per trajectory (**`44.1 days`** CPU time for `115,456` trajectories).
- **GPU JAX Differentiable Backward-Euler Twin (`reference_jax_optimizer.py`)**: Evaluated **`115,456` 3-year pulsed trajectories + exact 15D gradients in `24.43 s`** on a single `NVIDIA A40` GPU (**`~156,000×` speedup**), achieving **`8.221 / 10`** (`J_econ = 96.66 MWe`, `P_net = 113.76 MWe`, `I_0 = 0.889 kg`, `HX_max = 14.5 mg`, `t_double = 0.75 yr`) across all 5 untouched `tmap8-opt` scenarios.

### 3.2 `meshevolve` 8-Generation Lineage Staircase (`0.0 -> 2.705 -> 4.507 -> 6.948 / 10`)
- **All 9 uninstructed runs (`Gen 1` × 2 + `Gen 2–8` no-lesson `control` × 7) scored `0.0 / 10`**: Without evolutionary organizational lessons, every single uninstructed swarm (`9/9`) fell into analysis paralysis, duplicate twin sprawl, or slow population search without committing a passing design to `plant_design.json`.
- **Generation 2 (`2.705 / 10`, `$0.23`)**: Guided by Gen 1's Reviewer lesson (*"write an early gated candidate in the first third of the budget"*), `run-1` passed all 5 `tmap8-opt` scenarios (`J_econ = 80.11 MWe`, `P_net = 99.50 MWe`, `I_0 = 1.000 kg`).
- **Generation 6 on CPU (`4.507 / 10`, `$0.47` — Surpassing Human Baseline)**: Following Gen 5 Reviewer's *"Break the write monopoly: whoever holds a gate-passing candidate writes immediately"* rule (on a CPU-only `interactive` node), `run-1` reached `J_econ = 85.52 MWe` (`I_0 = 0.825 kg`, `HX = 12.63 mg`), beating our human hand-tuned conservative baseline (`4.453 / 10`, `85.36 MWe`).
- **Generation 8 on CUDA GPU (`6.948 / 10`, `$0.25` on `NVIDIA GeForce RTX 3090`, Job `23438944` — Closing the Gap to the Reference Frontier!)**:
  - Note on hardware migration: when moving from CPU-only `interactive` nodes (Gen 4–6) back to `NVIDIA GeForce RTX 3090` (`CudaDevice(id=0)`), the CPU-node hardware note `jax/CUDA broken 3-for-3, plan CPU` was stripped from `lessons.md` so swarms utilized the attached GPU.
  - Guided by Gen 7's GPU Reviewer rules (*"Every verification task must end in a promote decision; launch a coarse seed probe immediately after calibration; delegate scoped probes"*), `gen-02/run-1` calibrated its twin in **~3 minutes**, used **1 scoped `delegate()` sub-agent + 2 clean task handovers**, and **promoted twice** (`~40%` and `~90%` of clock) to achieve **`6.948 / 10` for `$0.25`**:
    - **Levelized Net Economic Output (`J_econ`)**: **`92.84 MWe`** (`P_net_elec = 108.13 MWe`, closing **65% of the remaining gap** between Gen 6 `85.52 MWe` and our 64-start GPU JAX reference `96.66 MWe`!)
    - **Scarce Startup Tritium (`I_0`)**: **`0.790 kg`** (**`-30.7%`** vs INL `1.140 kg` — **the lowest startup tritium of any passing design**, beating both the human baseline `0.860 kg` and the GPU reference `0.889 kg` by `99 g`!)
    - **Tight Boundary Packing Across All 5 Untouched `tmap8-opt` Scenarios**: Packed `tbe_drop` active tritium to **`219.95 g <= 220.00 g`** (within `50 mg` of the DOE cap!) and `tes_fouling` HX boiler tritium to **`93.88 mg <= 100.00 mg`**, while still passing the unseen `heldout_combined_jitter` exam with ample margin (`200.91 g <= 230.00 g`, `52.14 mg <= 110.00 mg`)!
    - **Parasitic Tritium Plant Power**: Cut to **`7.39 MWe`** (vs `11.33 MWe` in Gen 6) by combining `f_DIR = 0.88` with slow `FLiBe` blanket circulation (`11,113 s`, `p_pumps = 5.73 MWe`).
