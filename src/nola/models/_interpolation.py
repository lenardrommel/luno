import functools
import operator

import jax
from jax import numpy as jnp
import linox

from jax.typing import DTypeLike


class GeneralizedLinearRegression(linox.LinearOperator):
    def __init__(
        self,
        input_grid_shape: tuple[int, ...],
        output_grid_shape: tuple[int, ...],
        mode: str = "linear",
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        self._input_grid_shape = input_grid_shape
        self._input_grid_size = functools.reduce(
            operator.mul, self._input_grid_shape, 1
        )

        self._input_grids = tuple(
            jnp.linspace(0, 1, n + 1) for n in self._input_grid_shape
        )

        self._output_grid_shape = output_grid_shape
        self._output_grid_size = functools.reduce(
            operator.mul, self._output_grid_shape, 1
        )

        self._output_grid = jnp.stack(
            jnp.meshgrid(
                *(jnp.linspace(0, 1, n + 1)[:-1] for n in self._output_grid_shape),
                indexing="ij",
            ),
            axis=-1,
        )

        self._mode = mode

        super().__init__(
            shape=(self._output_grid_size, self._input_grid_size),
            dtype=dtype,
        )

    @functools.partial(jnp.vectorize, excluded={0}, signature="(n,k)->(m,k)")
    def _matmul(self, v: jax.Array) -> jax.Array:
        ncols = v.shape[-1]

        v = v.reshape(*self._input_grid_shape, ncols)

        v = jnp.pad(
            v,
            ((0, 1),) * len(self._input_grid_shape) + ((0, 0),),
            mode="wrap",
        )

        interpolator = jax.scipy.interpolate.RegularGridInterpolator(
            points=self._input_grids,
            values=v,
            method=self._mode,
        )

        return interpolator(self._output_grid).reshape(self.shape[0], ncols)
