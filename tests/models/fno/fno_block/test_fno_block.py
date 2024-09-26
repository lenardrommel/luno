import jax
import jax.numpy as jnp
import neuralop.layers.fno_block
import numpy as np
import torch

import pytest
from pytest_cases import fixture


@fixture(scope="session")
def v_out_ref(
    v_in_torch: torch.Tensor,
    grid_shape_out: tuple[int, ...],
    _neuralop_fno_block: neuralop.layers.fno_block.FNOBlocks,
) -> jax.Array:
    v_out_ref_torch = _neuralop_fno_block(v_in_torch, output_shape=grid_shape_out)

    v_out_ref = jnp.asarray(
        v_out_ref_torch.detach().numpy()  # shape = (1, C_out, N_1, N_2, ..., N_D)
    )
    v_out_ref = v_out_ref[0, ...]  # shape = (C_out, N_1, N_2, ..., N_D)
    v_out_ref = jnp.moveaxis(v_out_ref, 0, -1)  # shape = (N_1, N_2, ..., N_D, C_out)

    return v_out_ref


def test_fno_block(
    grid_shape_in: tuple[int, ...],
    grid_shape_out: tuple[int, ...],
    v_out: jax.Array,
    v_out_ref: jax.Array,
):
    if any(
        n_out != n_in for n_in, n_out in zip(grid_shape_in[:-1], grid_shape_out[:-1])
    ):
        pytest.skip(
            "There is a bug in the `neuraloperator` library."
            "Interpolation only works along the last axis."
        )

    np.testing.assert_allclose(
        v_out,
        v_out_ref,
        atol=1e-6,
        rtol=1e-6,
    )
