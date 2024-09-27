import jax
import jax.numpy as jnp
import numpy as np
import torch

from . import pdebench


def test_fno_block(
    output_grid_shape: tuple[int, ...] | None,
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    v_in_torch: torch.Tensor,
    v_out: jax.Array,
):
    pdebench.skip_if_case_unsupported(R.shape[:-2], output_grid_shape)

    # Compute reference output
    pdebench_fno_block = pdebench.FNOBlock(R, W, b)

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
