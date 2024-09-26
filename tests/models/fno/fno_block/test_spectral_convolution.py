import jax
from jax import numpy as jnp
import neuralop.layers.spectral_convolution
import numpy as np
import torch

import pytest
from pytest_cases import fixture


@fixture(scope="session")
def v_out_sconv_ref(
    v_in_torch: torch.Tensor,
    b: jax.Array,
    _neuralop_spectral_conv: neuralop.layers.spectral_convolution.SpectralConv,
    grid_shape_out: tuple[int, ...],
) -> jax.Array:
    v_out_ref_torch = _neuralop_spectral_conv(v_in_torch, output_shape=grid_shape_out)

    v_out_ref = jnp.asarray(
        v_out_ref_torch.detach().numpy()  # shape = (1, C_out, N_1, N_2, ..., N_D)
    )
    v_out_ref = v_out_ref[0, ...]  # shape = (C_out, N_1, N_2, ..., N_D)
    v_out_ref = jnp.moveaxis(v_out_ref, 0, -1)  # shape = (N_1, N_2, ..., N_D, C_out)

    return v_out_ref - b


def test_spectral_convolution(
    grid_shape_in: tuple[int, ...],
    grid_shape_out: tuple[int, ...],
    v_out_sconv: jax.Array,
    v_out_sconv_ref: jax.Array,
):
    if any(
        n_out != n_in for n_in, n_out in zip(grid_shape_in[:-1], grid_shape_out[:-1])
    ):
        pytest.skip(
            "There is a bug in the `neuraloperator` library."
            "Interpolation only works along the last axis."
        )

    np.testing.assert_allclose(
        v_out_sconv,
        v_out_sconv_ref,
        atol=1e-6,
        rtol=1e-6,
    )
