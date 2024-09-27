"""Custom padding and truncation functions for discrete Fourier transforms."""

import jax
from jax import numpy as jnp

from jax.typing import ArrayLike


def dft_resize(
    z: ArrayLike,
    num_modes: int | None = None,
    axis: int | None = -1,
) -> jax.Array:
    if num_modes is None or num_modes == z.shape[axis]:
        return z

    # Truncation (num_modes < z.shape[axis]) / padding (num_modes > z.shape[axis])
    z_m = jnp.zeros_like(z, shape=(z.shape[:axis] + (num_modes,) + z.shape[axis + 1 :]))

    num_modes_nnz = min(num_modes, z.shape[axis])

    z_m = jax.lax.dynamic_update_slice_in_dim(
        z_m,
        jax.lax.slice_in_dim(z, 0, 1 + (num_modes_nnz - 1) // 2, axis=axis),
        0,
        axis=axis,
    )

    if num_modes_nnz > 1:
        z_m = jax.lax.dynamic_update_slice_in_dim(
            z_m,
            jax.lax.slice_in_dim(
                z,
                -(num_modes_nnz - 1) // 2,
                z.shape[axis],
                axis=axis,
            ),
            -(num_modes_nnz - 1) // 2,
            axis=axis,
        )

    return z_m


def rdft_resize(
    z: ArrayLike,
    num_modes: int | None = None,
    axis: int | None = -1,
) -> jax.Array:
    if num_modes is None or num_modes == z.shape[axis]:
        return z

    if num_modes < z.shape[axis]:
        # Truncation
        return jax.lax.slice_in_dim(z, 0, num_modes, axis=axis)

    # Padding
    z_m = jnp.zeros_like(z, shape=(z.shape[:axis] + (num_modes,) + z.shape[axis + 1 :]))

    z_m = jax.lax.dynamic_update_slice_in_dim(z_m, z, 0, axis=axis)

    return z_m


def rfftn(
    a: ArrayLike,
    modes_shape: tuple[int] | None = None,
    axes: tuple[int, ...] | None = None,
    norm: str = "forward",
) -> ArrayLike:
    if modes_shape is None and axes is None:
        modes_shape = (None,) * a.ndim
        axes = tuple(range(-len(a.ndim), 0))
    elif modes_shape is None:
        modes_shape = (None,) * len(axes)
    elif axes is None:
        axes = tuple(range(-len(modes_shape), 0))

    z = jnp.fft.rfft(a, axis=axes[-1], norm=norm)
    z_pad_trunc = rdft_resize(z, num_modes=modes_shape[-1], axis=axes[-1])

    for num_modes, axis in zip(modes_shape[:-1], axes[:-1], strict=True):
        z = jnp.fft.fft(z_pad_trunc, axis=axis, norm=norm)
        z_pad_trunc = dft_resize(z, num_modes=num_modes, axis=axis)

    return z_pad_trunc


def irfft(
    z: ArrayLike,
    n: int | None = None,
    axis: int | None = -1,
    norm: str = "forward",
    from_even: bool = False,
) -> jax.Array:
    m = z.shape[axis]

    if from_even and n > 2 * (m - 1):
        z = jax.lax.dynamic_update_index_in_dim(
            z,
            jax.lax.index_in_dim(z, -1, axis=axis) / 2,
            -1,
            axis=axis,
        )

    n_even = n % 2 == 0

    if n_even and n // 2 + 1 < m:
        z = jax.lax.dynamic_update_index_in_dim(
            z,
            jax.lax.index_in_dim(z, -1, axis=axis) * 2,
            -1,
            axis=axis,
        )

    return jnp.fft.irfft(z, n=n, axis=axis, norm=norm)


def irfftn(
    z: ArrayLike,
    grid_shape: tuple[int] | None = None,
    axes: tuple[int, ...] | None = None,
    last_axis_from_even: bool = False,
    norm: str = "forward",
) -> ArrayLike:
    if grid_shape is None and axes is None:
        grid_shape = (None,) * z.ndim
        axes = tuple(range(-len(z.ndim), 0))
    elif grid_shape is None:
        grid_shape = (None,) * len(axes)
    elif axes is None:
        axes = tuple(range(-len(grid_shape), 0))

    a = z

    for num_points, axis in zip(grid_shape[:-1], axes[:-1], strict=True):
        a_trunc_pad = dft_resize(a, num_modes=num_points, axis=axis)
        a = jnp.fft.ifft(a_trunc_pad, axis=axis, norm=norm)

    a = irfft(
        a,
        n=grid_shape[-1],
        axis=axes[-1],
        from_even=last_axis_from_even,
        norm=norm,
    )

    return a
