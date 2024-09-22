import jax
import jax.numpy as jnp
import neuralop.layers.fno_block
import numpy as np
import torch

from pytest_cases import fixture

from nola.models.fno import fno_block


@fixture(scope="session")
def v_out_ref(
    v_in_torch: torch.Tensor,
    _neuralop_fno_block: neuralop.layers.fno_block.FNOBlocks,
) -> jax.Array:
    v_out_ref_torch = _neuralop_fno_block(v_in_torch)

    v_out_ref = jnp.asarray(
        v_out_ref_torch.detach().numpy()  # shape = (1, C_out, N_1, N_2, ..., N_D)
    )
    v_out_ref = v_out_ref[0, ...]  # shape = (C_out, N_1, N_2, ..., N_D)
    v_out_ref = jnp.moveaxis(v_out_ref, 0, -1)  # shape = (N_1, N_2, ..., N_D, C_out)

    return v_out_ref


def test_fno_block(
    v_in: jax.Array,
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    grid_shape_out: tuple[int, ...],
    v_out_ref: jax.Array,
):
    v_out = fno_block(v_in, R, W, b, output_grid_shape=grid_shape_out)

    np.testing.assert_allclose(
        v_out,
        v_out_ref,
        atol=1e-6,
        rtol=1e-6,
    )
