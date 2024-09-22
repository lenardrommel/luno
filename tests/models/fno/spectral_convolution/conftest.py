import jax
from jax import numpy as jnp
import neuralop.layers.spectral_convolution
import numpy as np
import torch

from pytest_cases import fixture

import nola


@fixture(scope="session")
def modes_shape(R: jax.Array) -> tuple[int, ...]:
    return R.shape[:-2]


@fixture(scope="session")
def z_in(v_in: jax.Array, modes_shape: tuple[int, ...]) -> jax.Array:
    return nola.models.fno.dft.rfftn(
        v_in,
        modes_shape=modes_shape,
        axes=tuple(range(len(modes_shape))),
        norm="forward",
    )


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
