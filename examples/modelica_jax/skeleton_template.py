"""Deliverable JAX GPU ensemble solver for __MODEL_NAME__.

Contract required by the grader:
  1. `forward_ensemble(y0_batch, p_batch) -> jax.Array`
     - `y0_batch`: float64 array of shape `(B, __N_STATES__)`
     - `p_batch`:  float64 array of shape `(B, __N_PARAMS__)` with columns __PARAM_NAMES__
     - returns float64 array of shape `(B, __N_OBS__)` matching __OBS_NAMES__ at t = __T_END__.
  2. `differentiable_closure(y0_batch, p_ref_batch) -> Callable[[jax.Array], jax.Array]`
     - returns a pure JAX-traceable function `f(p_jax) -> obs_final_jax` of shape `(B, __N_OBS__)`
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
