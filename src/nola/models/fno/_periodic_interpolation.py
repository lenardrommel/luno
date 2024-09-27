from jax import numpy as jnp

from jax.typing import ArrayLike

from . import dft


def gridded_fourier_interpolation(
    v: ArrayLike,
    axes: tuple[int, ...],
    output_grid_shape: tuple[int, ...],
):
    v = jnp.asarray(v)

    input_grid_shape = tuple(v.shape[axis] for axis in axes)

    if output_grid_shape == input_grid_shape:
        return v

    v_rdft = dft.rfftn(
        v,
        axes=axes,
        norm="forward",
    )

    v_interp = dft.irfftn(
        v_rdft,
        grid_shape=output_grid_shape,
        axes=axes,
        last_axis_from_even=input_grid_shape[-1] % 2 == 0,
        norm="forward",
    )

    return v_interp
