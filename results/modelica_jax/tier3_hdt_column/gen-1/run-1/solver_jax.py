"""Deliverable JAX GPU ensemble solver for CryoHDTColumn16.

Contract required by the grader:
  1. `forward_ensemble(y0_batch, p_batch) -> jax.Array`
     - `y0_batch`: float64 array of shape `(B, 32)`
     - `p_batch`:  float64 array of shape `(B, 4)` with columns ['V_boil', 'd_rec_frac', 'dp_tray', 'P_top']
     - returns float64 array of shape `(B, 7)` matching ['obs_xH2_top', 'obs_xT2_bot', 'obs_T_top', 'obs_T_bot', 'obs_tot_H2', 'obs_tot_HD', 'obs_tot_T2'] at t = 25.0.
  2. `differentiable_closure(y0_batch, p_ref_batch) -> Callable[[jax.Array], jax.Array]`
     - returns a pure JAX-traceable function `f(p_jax) -> obs_final_jax` of shape `(B, 7)`
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
