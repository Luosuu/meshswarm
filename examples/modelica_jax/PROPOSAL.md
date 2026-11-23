# Proposal: Modelica-to-JAX Differentiable Simulation & Control Benchmark

## 1. Objective

Extend `meshswarm` and `meshevolve` to a third scientific domain alongside Lean 4 theorem proving (`examples/circle2`) and multi-core C/C++ HPC optimization (`examples/hpc`): **automatically refactoring acausal engineering Modelica models (`.mo`) into conservation-preserving, differentiable GPU ensemble solvers and closed-loop controllers in JAX**.

Motivated by Fusion Pilot Plant (FPP) digital-twin requirements (balance-of-plant thermo-fluid loops in ORNL's `TRANSFORM-Library` and cryogenic H/D/T isotope separation cascades), the goal is to turn serial, non-differentiable engineering differential-algebraic equations (DAEs) into branch-free GPU operators that:
- execute **1,000 stiff trajectories in `< 15 s`** on a single GPU in FP64;
- preserve exact physical invariants (e.g., closed-loop H/D/T molar balance drift **`< 1e-12`**);
- expose **exact reverse-mode sensitivities** matching central finite differences to **`< 1e-7`**;
- plug directly into gradient-based optimal control (differentiable MPC / SHAC / $10^6$-episode GPU RL) and core–plant co-design.

---

## 2. Why Compiler Export Alone (`Rumoca.to_jax()`) Is Not Enough

While `Rumoca` (`rumoca 0.10.4`) provides a fast Rust frontend and a `Model.to_jax()` export for already-clean explicit ODEs, empirical testing on UVA Rivanna reveals four structural walls on real engineering Modelica models:

1. **Package hierarchies, `stream` connectors, and tabulated `Medium` properties (`0 / 9` direct pass rate on `TRANSFORM-UnitTests`)**:
   Real thermo-fluid components inherit from the Modelica Standard Library (MSL) and `TRANSFORM` package trees, use bidirectional `stream` connectors for backflow enthalpy, and call piecewise/external fluid property functions (IF97 water/steam, helium, FLiBe, cryogenic hydrogen isotopes) that `Rumoca` cannot load or differentiate directly.
2. **Implicit algebraic loops / semi-explicit DAEs (`unsupported-feature:residual_equations`)**:
   Any nonlinear algebraic constraint $0 = g(x, z, u, p)$ (such as pipe-network pressure/flow closure or vapor–liquid equilibrium in distillation stages) is rejected outright by `Rumoca`'s `jax-ode` backend.
3. **Events, check valves, and flow-reversal singularities (`unsupported-feature:events`)**:
   Even a one-line check valve or orifice equation (`der(q) = if dp > 0 then sqrt(dp) else 0`) is rejected by `Rumoca`'s `jax-ode` backend. Naive `jnp.where` translations produce `inf`/`NaN` reverse-mode gradients at $\Delta p = 0$, stall adaptive stiff steppers ($\Delta t \to 0$), and cause SIMT warp divergence across GPU ensembles.
4. **High-dimensional stiff Jacobian scaling on GPUs**:
   When state dimension $N$ grows from $O(10)$ to $50\text{–}150$ (multi-node heat exchangers or 20-stage H/D/T distillation columns), generic per-thread dense $O(N^3)$ Rosenbrock LU factorization spills GPU registers. Achieving high throughput requires state reduction via conservation invariants, quasi-steady fast-mode elimination, or custom tridiagonal/block-sparse linear solves.

---

## 3. What `meshswarm` Agents Do

Given a starting workspace containing an engineering `.mo` model, an untouched OpenModelica (`omc`) reference trajectory generator, and a sandboxed GPU `run_command` tool (`omc` + `rumoca` + `jax` + `gradsolve`), a leaderless swarm must collaborate to deliver a verified JAX solver module:

1. **Flatten & decouple**: Use `omc` (`instantiateModel` / reference simulation) alongside `Rumoca` to extract flat governing equations and replace tabulated `Medium` calls with smooth, high-accuracy JAX rational/spline approximations.
2. **Eliminate or differentiate algebraic loops**: Either solve algebraic constraints symbolically into explicit form or implement vectorized damped Newton solvers wrapped in implicit-function-theorem custom VJPs (`jax.custom_vjp` with $\partial z / \partial x = -g_z^{-1} g_x$).
3. **Calibrate branch smoothing**: Regularize check valves, $|q|q$ friction, $\sqrt{|\Delta p|}$ orifice laws, and Sieverts-law $\sqrt{p_{\text{T}}}$ permeation so that smoothing parameter $\epsilon$ simultaneously satisfies `omc` trajectory accuracy, exact mass conservation, and well-conditioned reverse-mode adjoints.
4. **Optimize stiff GPU ensemble throughput**: Tune solver tolerances/formulations, eliminate redundant conserved states, or exploit tridiagonal/sparse Jacobian structure to climb the 1,000-trajectory GPU speedup ladder.

---

## 4. Multiplicative Verification & Grading Contract

To prevent reward hacking (mirroring lessons from `amrx`, `pic`, and `lulesh`), every swarm deliverable is graded under hard zero-multiplier gates on held-out initial conditions and transient control inputs:

$$
\text{Score} = I_{\text{compile}} \cdot I_{\text{traj\_gold}} \cdot I_{\text{mass\_conservation}} \cdot I_{\text{adjoint\_grad}} \cdot S_{\text{throughput}}
$$

- **$I_{\text{compile}} \in \{0, 1\}$**: Runs under `jax.jit` and `jax.vmap` on GPU in FP64 with zero `NaN`/`Inf` and no host callbacks.
- **$I_{\text{traj\_gold}} \in \{0, 1\}$**: Held-out transient states match OpenModelica DASSL/IDA reference outputs within task-specified relative tolerance.
- **$I_{\text{mass\_conservation}} \in \{0, 1\}$**: Closed-loop H/D/T molar or total mass balance drift remains **`< 1e-12`** across all trajectories.
- **$I_{\text{adjoint\_grad}} \in \{0, 1\}$**: Reverse-mode sensitivities (`jax.grad` through the stiff trajectory rollout, including across valve/flow-reversal transitions) match central finite differences to relative error **`< 1e-7`**.
- **$S_{\text{throughput}} \in [0, 10]$**: Rung-based score measuring 1,000-trajectory GPU median wall time (and/or speedup over serial OpenModelica baseline), targeting **`< 15 s`** per 1,000 trajectories on a single GPU.

---

## 5. Phase 0 Status (Verified on UVA Rivanna GPU)

All core dependencies and a 5-stage end-to-end smoke test (`examples/modelica_jax/smoke_pipeline.py`) have been deployed under `/scratch/$USER/meshswarm/modelica_jax` (zero `$HOME` footprint) and verified on a Rivanna **NVIDIA A40** node (Slurm job `23373343`):

- **Toolchain**: Python 3.12 `.venv` (`jax 0.12.1` CUDA 12, `rumoca 0.10.4`, `gradsolve 0.2.1` `rodas5p_replay`, `diffrax 0.7.2`, `optax 0.2.9`), Apptainer `openmodelica.sif` (`OpenModelica 1.27.1`), and cloned `TRANSFORM-Library` + `TRANSFORM-UnitTests`.
- **Stage 1–3 (`omc` + `Rumoca` + Conservation)**: `TRANSFORM-Library` loads cleanly in `omc`; on the 6-state stiff `BlanketTritiumLoop` smoke test ($\tau_{\text{hyd}} = 0.025\text{ s}$ vs $\tau_{\text{th/T}} \approx 28\text{–}80\text{ s}$), `rodas5p_replay` matches OpenModelica DASSL to **`2.65e-10`** max relative error with **`0.00e+00`** single-run and **`2.49e-15`** 1,000-ensemble tritium conservation drift.
- **Stage 4 (GPU Ensemble & Adjoint)**: 1,000 stiff trajectories run in **`1.34 s`** warm median (`2.45 s` cold JIT+run) on 1× NVIDIA A40; discrete replay-adjoint matches central finite differences to **`5.88e-8`** (`3.09e-7` vs fresh adaptive FD).
- **Stage 5 (Closed-Loop Differentiable Control)**: 40 `optax.adam` steps backpropagating through 32 uncertain fusion power-pulse rollouts reduce true adaptive control loss from **`492.81` to `-17.63`**.

---

## 6. Phased Roadmap

- [x] **Phase 0 — Rivanna Environment & End-to-End GPU Smoke Pipeline**: Containerized `OpenModelica 1.27.1`, `Rumoca`, `JAX` CUDA 12, `gradsolve` (`rodas5p_replay`), and `Optax` verified on Rivanna GPU (`smoke_pipeline.py`).
- [x] **Phase 1 — 3-Tier Non-Smooth / DAE Benchmark Suite** (`tasks.py`, `grader.py`, `references.py`, `selftest.py` — verified on Rivanna RTX A6000 GPU in Slurm job `23379646`):
  - *Tier 1 (`tier1_blanket_dae`)*: Unsmoothed check-valve & orifice blanket cooling + implicit cubic-Sieverts tritium loop (`omc_err = 3.47e-7`, `mass_err = 5.86e-15`, `adj_err = 9.10e-6`, `7.56 s` combined on GPU).
  - *Tier 2 (`tier2_transform_hx`)*: ORNL `TRANSFORM.HeatExchangers.Simple_HX` (`PbLi` + `FLiBe` molten salt, `cv != cp` internal energy semantics) + Arrhenius wall tritium permeation (`omc_err = 9.05e-8`, `mass_err = 2.37e-15`, `adj_err = 8.57e-8`, `5.29 s` combined on GPU).
  - *Tier 3 (`tier3_hdt_column`)*: 16-stage closed-recycle cryogenic H2/HD/T2 distillation MESH DAE column (`32` states + `16` implicit Antoine bubble-point equations; structure-aware scan + `@jax.custom_jvp` reaches **`0.143 s`** for 1,000 trajectories / `0.397 s` combined -> `7.0/10`, 18.7× faster forward than generic adaptive replay at `2.68 s` -> `4.0/10`, `omc_err = 1.90e-9`, `mass_err = 1.85e-15`).
- [x] **Phase 2 — `meshswarm` GPU-DAE Runner, Standalone File Layout & Two-Tier Agent Routing** (verified on Rivanna NVIDIA A100-80GB in Slurm job `23380990`, `network:none`):
  - Refactored all Modelica sources (`models/*.mo`), reference JAX solvers (`references/*.py`), starter template (`skeleton_template.py`), and subprocess evaluation worker (`eval_worker.py`) into standalone files with zero embedded code strings.
  - Extended `meshswarm/shell.py` (`ShellRunner.gpu_sets`) with round-robin `CUDA_VISIBLE_DEVICES` / `APPTAINERENV_CUDA_VISIBLE_DEVICES` pinning and shared-GPU `asyncio.Lock` serialization.
  - Added two-tier model routing (`sub_model`, `sub_pricing`, `Limits.max_steps_per_subagent`) across `meshswarm` and `meshevolve` so mesh peer Architects (`Inkling-2-Small`) can delegate bounded tasks to high-throughput sub-agent Workers (`Inkling-2-Mini`).
  - Built `examples/modelica_jax/run.py`, `regrade.py`, and unified `--nv --containall --no-home --net --network none` Apptainer sandbox scripts (`rivanna/{env.sh,job.sbatch,regrade.sbatch}`), achieving `RUN_SELFTEST_OK` across all 3 tiers (`0.0434 s` for 1,000 Tier-3 trajectories on A100-80GB, `56.7×` faster forward than generic adaptive replay).
- [ ] **Phase 3 — `meshevolve` Multi-Generation Experiments**: Run 3-generation swarm evolution with `--control` baselines on Rivanna GPUs; measure gate pass rates, GPU throughput gains, dollar cost, and cross-task lesson transfer from thermal loops to isotope distillation.
- [ ] **Phase 4 — Closed-Loop Control & Core–Plant Co-Design**: Benchmark differentiable MPC / SHAC vs PPO over $10^5\text{–}10^6$ GPU episodes under plasma power ramps and pump trips, distill `< 1 ms` compact MLP controllers, and demonstrate coupled gradient co-design.
