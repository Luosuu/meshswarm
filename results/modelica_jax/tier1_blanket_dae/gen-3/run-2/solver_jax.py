"""Deliverable JAX GPU ensemble solver for BlanketValveDAE.

Contract required by the grader:
  1. `forward_ensemble(y0_batch, p_batch) -> jax.Array`
     - `y0_batch`: float64 array of shape `(B, 6)`
     - `p_batch`:  float64 array of shape `(B, 4)` with columns ['u_pump', 'u_ext', 'P_pulse', 'dp_back']
     - returns float64 array of shape `(B, 6)` matching ['obs_q_flow', 'obs_T_blanket', 'obs_T_hx', 'obs_N_blanket', 'obs_N_extractor', 'obs_N_storage'] at t = 20.0.
  2. `differentiable_closure(y0_batch, p_ref_batch) -> Callable[[jax.Array], jax.Array]`
     - returns a pure JAX-traceable function `f(p_jax) -> obs_final_jax` of shape `(B, 6)`
       suitable for `jax.grad` around `p_ref_batch`.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def forward_ensemble(y0_batch: np.ndarray, p_batch: np.ndarray) -> jax.Array:
    raise NotImplementedError("Implement GPU-vectorized stiff forward ensemble solver")


def differentiable_closure(y0_batch: np.ndarray, p_ref_batch: np.ndarray):
    raise NotImplementedError("Return a jax.grad-differentiable function mapping p_jax -> obs_final_jax")
