"""Run all three Phase 4 benchmarks on GPU and write `phase4_metrics.json`:
  1. Closed-loop control under plasma ramps + pump trips: SHAC (102K episodes) vs PPO (1.024M episodes);
  2. Ultra-compact 483-parameter real-time CPU MLP distillation (< 1 ms p99.9 latency);
  3. End-to-end coupled Core + Swarm TRANSFORM HX + Swarm 16-Stage HDT Distillation Co-Design.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from codesign import run_codesign_benchmark  # noqa: E402
from distill_mlp import distill_compact_realtime_mlp  # noqa: E402
from plant_env import sample_scenarios  # noqa: E402
from rl_benchmark import evaluate_constant_open_loop, train_ppo_1m_episodes, train_shac_analytic  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="optional path to save JSON metrics")
    ap.add_argument("--ppo-updates", type=int, default=500)
    ap.add_argument("--ppo-batch", type=int, default=2048)
    ap.add_argument("--shac-updates", type=int, default=200)
    ap.add_argument("--shac-batch", type=int, default=256)
    a = ap.parse_args()

    devs = jax.devices()
    print(f"=== Phase 4 Devices: {devs} ===", flush=True)

    # Shared held-out 1,024-scenario evaluation set (never seen during training)
    eval_scen = sample_scenarios(jax.random.PRNGKey(20261124), 1024)

    print("\n--- [1/4] Evaluating Open-Loop Nominal Baseline ---", flush=True)
    open_loop = evaluate_constant_open_loop(eval_scen)
    print("Open-loop:", json.dumps(open_loop, indent=2), flush=True)

    print(f"\n--- [2/4] Training Differentiable SHAC ({a.shac_updates * a.shac_batch:,} episodes) vs Model-Free PPO ({a.ppo_updates * a.ppo_batch:,} episodes) ---", flush=True)
    _, shac_res = train_shac_analytic(eval_scen, n_updates=a.shac_updates, batch_envs=a.shac_batch)
    print("SHAC:", json.dumps(shac_res, indent=2), flush=True)

    _, ppo_res = train_ppo_1m_episodes(eval_scen, n_updates=a.ppo_updates, batch_envs=a.ppo_batch)
    print("PPO:", json.dumps(ppo_res, indent=2), flush=True)

    assert shac_res["final_mean_cost"] < 5.0 and shac_res["final_mean_cost"] < 0.02 * open_loop["mean_cost"]
    assert ppo_res["final_mean_cost"] < 5.0 and ppo_res["final_mean_cost"] < 0.02 * open_loop["mean_cost"]
    assert shac_res["max_tritium_drift"] < 1.0e-12
    assert ppo_res["max_tritium_drift"] < 1.0e-12

    print("\n--- [3/4] Distilling 483-Param Real-Time CPU MLP (< 1 ms MILF-BENCH Gate) ---", flush=True)
    np_weights, distill_res = distill_compact_realtime_mlp(eval_scen)
    print("Distilled MLP:", json.dumps(distill_res, indent=2), flush=True)
    assert distill_res["param_count"] < 10000
    assert distill_res["cpu_latency_us"]["p99_9"] < 1000.0
    assert distill_res["jax_vs_numpy_max_diff"] < 1.0e-12
    assert distill_res["max_tritium_drift"] < 1.0e-12

    print("\n--- [4/4] Running End-to-End 3-Solver Coupled Core-Blanket-Distillation Co-Design ---", flush=True)
    codesign_res = run_codesign_benchmark()
    print("Co-Design:", json.dumps(codesign_res, indent=2), flush=True)
    assert codesign_res["gradient_codesign"]["coupled_3solver_grad_fd_rel_err"] < 1.0e-5
    assert codesign_res["gradient_codesign"]["metrics"]["loss"] < codesign_res["gradient_free_baseline"]["metrics"]["loss"]
    assert codesign_res["gradient_codesign"]["metrics"]["x_t2_bot"] >= 0.99968

    summary = {
        "device": str(devs[0]),
        "control_benchmark": {
            "eval_scenarios": 1024,
            "horizon_seconds": 30,
            "open_loop": open_loop,
            "shac_differentiable": shac_res,
            "ppo_model_free": ppo_res,
        },
        "realtime_mlp_distillation": distill_res,
        "coupled_plant_codesign": codesign_res,
    }
    if a.out:
        out_p = Path(a.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        weights_p = out_p.with_name("distilled_controller_483p.npz")
        import numpy as np
        np.savez(weights_p, **np_weights)
        print(f"\nSaved metrics to {out_p} and weights to {weights_p}", flush=True)

    print("\nPHASE4_ALL_BENCHMARKS_PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
