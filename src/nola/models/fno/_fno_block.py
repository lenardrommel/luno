import jax

from jax.typing import ArrayLike

from .._pointwise_affine import pointwise_affine
from ._periodic_interpolation import (
    GriddedFourierInterpolationOperator,
    SciPyRegularGridInterpolator,
)
from ._spectral_convolution import spectral_convolution


def fno_block(
    v_in: ArrayLike,
    R: ArrayLike,
    W: ArrayLike,
    b: ArrayLike,
    grid_shape_out: tuple[int, ...] | None = None,
    interpolation_method: str = "fourier",
) -> jax.Array:
    v_out_sconv, (z_in, Rz_in) = spectral_convolution(
        v_in,
        R,
        grid_shape_out=grid_shape_out,
    )

    v_out_pointwise = pointwise_affine(v_in, W, b)

    # Spatial interpolation of skip connection
    v_out_pointwise_interp = v_out_pointwise

    if grid_shape_out is not None:
        if interpolation_method == "fourier":
            interpolator = GriddedFourierInterpolationOperator(
                input_grid_shape=v_out_pointwise.shape[:-1],
                output_grid_shape=grid_shape_out,
                dtype=v_out_pointwise.dtype,
            )
        else:
            interpolator = SciPyRegularGridInterpolator.from_regular_output_grid(
                input_grid_shape=v_out_pointwise.shape[:-1],
                output_grid_shape=grid_shape_out,
                mode=interpolation_method,
                dtype=v_out_pointwise.dtype,
            )

        v_out_pointwise_interp = v_out_pointwise_interp.reshape(
            -1, v_out_pointwise.shape[-1]
        )
        v_out_pointwise_interp = interpolator @ v_out_pointwise_interp
        v_out_pointwise_interp = v_out_pointwise_interp.reshape(
            *grid_shape_out, v_out_pointwise_interp.shape[-1]
        )

    v_out = v_out_sconv + v_out_pointwise_interp

    return v_out, (z_in, Rz_in, v_out_pointwise, v_out_pointwise_interp)
