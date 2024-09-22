import jax
from jax import numpy as jnp

from jax.typing import ArrayLike

from . import dft


def spectral_convolution(
    v_in: ArrayLike,
    R: ArrayLike,
    output_grid_shape: tuple[int, ...] | None = None,
) -> jax.Array:
    """Computes the spectral convolution of a real input signal :math:`v_{in}` with the
    complex spectral weight tensor :math:`R`.

    Parameters
    ----------
    v_in :
        Real input of shape `(N_1, N_2, ..., N_D, C_in)`.
    R :
        Complex spectral weight tensor of shape `(M_1, ..., M_D, C_out, C_in)`.
    output_grid_shape :
        Shape of the output grid. By default, the output grid will match the input grid.

    Returns
    -------
    v_out :
        Real output of shape `output_grid_shape + (C_out,)`.
    """

    v_in = jnp.asarray(v_in)
    R = jnp.asarray(R)

    grid_shape_in = v_in.shape[:-1]
    D = len(grid_shape_in)

    # Truncated real Fourier transform of the input
    z_in = dft.rfftn(
        v_in,
        modes_shape=R.shape[:-2],
        axes=tuple(range(-D - 1, -1)),
        norm="forward",
    )

    # Pointwise multiplication by `R` in frequency domain
    Rz_in = (R @ z_in[..., None])[..., 0]

    # (Interpolated) inverse real Fourier transform
    if output_grid_shape is None:
        output_grid_shape = grid_shape_in

    v_out = dft.irfftn(
        Rz_in,
        grid_shape=output_grid_shape,
        axes=tuple(range(-D - 1, -1)),
        norm="forward",
    )

    return v_out
