from __future__ import annotations

import functools
import operator

import jax
from jax import numpy as jnp
import linox

from jax.typing import ArrayLike, DTypeLike

from . import dft


class PeriodicGeneralizedLinearInterpolationOperator(linox.LinearOperator):
    def __init__(
        self,
        input_grid_shape: tuple[int, ...],
        output_size: int,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        self._input_grid_shape = input_grid_shape
        self._input_grid_size = functools.reduce(
            operator.mul, self._input_grid_shape, 1
        )

        self._output_size = output_size

        super().__init__(
            shape=(self._output_size, self._input_grid_size),
            dtype=dtype,
        )


class SciPyRegularGridInterpolator(PeriodicGeneralizedLinearInterpolationOperator):
    def __init__(
        self,
        input_grid_shape: tuple[int, ...],
        output_points: ArrayLike,
        mode: str = "linear",
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        self._output_points = jnp.asarray(output_points)

        self._mode = mode

        super().__init__(
            input_grid_shape=input_grid_shape,
            output_size=functools.reduce(
                operator.mul, self._output_points.shape[:-1], 1
            ),
            dtype=dtype,
        )

    @classmethod
    def from_regular_output_grid(
        self,
        input_grid_shape: tuple[int, ...],
        output_grid_shape: tuple[int, ...],
        mode: str = "linear",
        dtype: DTypeLike = jnp.float32,
    ) -> SciPyRegularGridInterpolator:
        return SciPyRegularGridInterpolator(
            input_grid_shape=input_grid_shape,
            output_points=jnp.stack(
                jnp.meshgrid(
                    *(jnp.linspace(0, 1, n + 1)[:-1] for n in output_grid_shape),
                    indexing="ij",
                ),
                axis=-1,
            ),
            mode=mode,
            dtype=dtype,
        )

    @functools.cached_property
    def _periodic_input_grids(self) -> tuple[jnp.Array, ...]:
        return tuple(jnp.linspace(0, 1, n + 1) for n in self._input_grid_shape)

    @functools.partial(jnp.vectorize, excluded={0}, signature="(n,k)->(m,k)")
    def _matmul(self, v: jax.Array) -> jax.Array:
        ncols = v.shape[-1]

        v = v.reshape(*self._input_grid_shape, ncols)

        v_periodic = jnp.pad(
            v,
            ((0, 1),) * len(self._input_grid_shape) + ((0, 0),),
            mode="wrap",
        )

        interpolator = jax.scipy.interpolate.RegularGridInterpolator(
            points=self._periodic_input_grids,
            values=v_periodic,
            method=self._mode,
        )

        v_interp = interpolator(self._output_points)

        return v_interp.reshape(self.shape[0], ncols)


class GriddedFourierInterpolationOperator(
    PeriodicGeneralizedLinearInterpolationOperator
):
    def __init__(
        self,
        input_grid_shape: tuple[int, ...],
        output_grid_shape: tuple[int, ...],
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        self._output_grid_shape = output_grid_shape

        super().__init__(
            input_grid_shape=input_grid_shape,
            output_size=functools.reduce(operator.mul, self._output_grid_shape, 1),
            dtype=dtype,
        )

    def _matmul(self, v: jax.Array) -> jax.Array:
        ncols = v.shape[-1]

        v = v.reshape(*self._input_grid_shape, ncols)

        v_rdft = dft.rfftn(
            v,
            axes=tuple(range(len(self._input_grid_shape))),
            norm="forward",
        )

        v_interp = dft.irfftn(
            v_rdft,
            grid_shape=self._output_grid_shape,
            axes=tuple(range(len(self._output_grid_shape))),
            norm="forward",
        )

        return v_interp.reshape(self.shape[0], ncols)
