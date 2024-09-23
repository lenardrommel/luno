import jax
from jax import numpy as jnp

from collections.abc import Mapping
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
    v_out, intermediates_sconv = spectral_convolution(
        v_in,
        R,
        output_grid_shape=output_grid_shape,
    )

    # Skip connection
    v_out_pointwise = pointwise_linear(v_in, W)

    if b is not None:
        v_out_pointwise += jnp.asarray(b)

    if output_grid_shape is not None:
        v_out_pointwise = gridded_fourier_interpolation(
            v_out_pointwise,
            axes=tuple(range(len(output_grid_shape))),
            output_grid_shape=output_grid_shape,
        )

    v_out += v_out_pointwise

    return v_out, {"spectral_convolution": intermediates_sconv}
