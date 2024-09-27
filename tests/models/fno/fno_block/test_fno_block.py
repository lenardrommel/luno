import jax
import jax.numpy as jnp
import numpy as np
import torch

import pytest

from . import _pdebench


def test_fno_block(
    grid_shape_out: tuple[int, ...] | None,
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    v_in_torch: torch.Tensor,
    v_out: jax.Array,
):
    if grid_shape_out is not None:
        pytest.skip("The PDEBench FNO block does not support output interpolation.")

    # Compute reference output
    pdebench_fno_block = _pdebench.FNOBlock(R, W, b)

    v_out_ref_torch = pdebench_fno_block(v_in_torch)

    v_out_ref = jnp.asarray(
        v_out_ref_torch.detach().numpy()  # shape = (1, C_out, N_1, N_2, ..., N_D)
    )
    v_out_ref = v_out_ref[0, ...]  # shape = (C_out, N_1, N_2, ..., N_D)
    v_out_ref = jnp.moveaxis(v_out_ref, 0, -1)  # shape = (N_1, N_2, ..., N_D, C_out)

    np.testing.assert_allclose(
        v_out,
        v_out_ref,
        atol=1e-6,
        rtol=1e-6,
    )
