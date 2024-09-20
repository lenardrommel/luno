import jax

from jax.typing import ArrayLike

from .._pointwise_affine import pointwise_affine
from ._spectral_convolution import spectral_convolution


def fno_block(
    v_in: ArrayLike,
    R: ArrayLike,
    W: ArrayLike,
    b: ArrayLike,
    grid_shape_out: tuple[int, ...] | None = None,
) -> jax.Array:
    v_out_sconv, (z_in, Rz_in) = spectral_convolution(
        v_in,
        R,
        grid_shape_out=grid_shape_out,
    )
    v_out_pointwise = pointwise_affine(v_in, W, b)

    # TODO: Spatial interpolation of `v_out_pointwise`

    v_out = v_out_sconv + v_out_pointwise

    return v_out, (z_in, Rz_in, v_out_pointwise)
