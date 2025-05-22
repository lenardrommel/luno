from collections.abc import Mapping

import jax
from jax import numpy as jnp
from jax.typing import ArrayLike

from .._pointwise_linear import pointwise_linear
from ._periodic_interpolation import gridded_fourier_interpolation
from ._spectral_convolution import spectral_convolution


def fno_block(
    v_in: ArrayLike,
    R: ArrayLike,
    W: ArrayLike,
    b: ArrayLike | None = None,
    output_grid_shape: tuple[int, ...] | None = None,
) -> tuple[jax.Array, Mapping[str, jax.Array]]:
    v_in = jnp.asarray(v_in)

    input_grid_shape = v_in.shape[:-1]

    v_out_sconv, intermediates_sconv = spectral_convolution(
        v_in,
        R,
        output_grid_shape=output_grid_shape,
    )

    # Skip connection
    v_out_pointwise = pointwise_linear(v_in, W)

    if b is not None:
        v_out_pointwise += jnp.asarray(b)

    if output_grid_shape is not None and output_grid_shape != input_grid_shape:
        v_out_pointwise = gridded_fourier_interpolation(
            v_out_pointwise,
            axes=tuple(range(len(output_grid_shape))),
            output_grid_shape=output_grid_shape,
        )

    return v_out_sconv + v_out_pointwise, {
        "v_out_sconv": v_out_sconv,
        "v_out_pointwise": v_out_pointwise,
        "spectral_convolution": intermediates_sconv,
    }
