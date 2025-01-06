import jax
from jax import numpy as jnp
import numpy as np
import torch

from . import pdebench


def test_spectral_convolution(
    output_grid_shape: tuple[int, ...],
    R: jax.Array,
    v_in_torch: torch.Tensor,
    v_out_sconv: jax.Array,
):
    pdebench.skip_if_case_unsupported(R.shape[:-2], output_grid_shape)

    # Compute reference output
    pdebench_spectral_conv = pdebench.spectal_convolution_from_lugano_weights(R)

    v_out_sconv_ref_torch = pdebench_spectral_conv(v_in_torch)

    v_out_sconv_ref = jnp.asarray(
        v_out_sconv_ref_torch.detach().numpy()  # shape = (1, C_out, N_1, N_2, ..., N_D)
    )
    v_out_sconv_ref = v_out_sconv_ref[0, ...]  # shape = (C_out, N_1, N_2, ..., N_D)
    v_out_sconv_ref = jnp.moveaxis(
        v_out_sconv_ref, 0, -1
    )  # shape = (N_1, N_2, ..., N_D, C_out)

    np.testing.assert_allclose(
        v_out_sconv,
        v_out_sconv_ref,
        atol=1e-6,
        rtol=1e-6,
    )
