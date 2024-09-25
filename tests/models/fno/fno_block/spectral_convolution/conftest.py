import jax
from jax import numpy as jnp
import neuralop.layers.spectral_convolution
import torch

from pytest_cases import fixture


@fixture(scope="session")
def z_in(_fno_block_out) -> jax.Array:
    return _fno_block_out[1]["spectral_convolution"]["z_in"]


@fixture(scope="session")
def v_out_sconv(_fno_block_out) -> jax.Array:
    return _fno_block_out[1]["v_out_sconv"]


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
