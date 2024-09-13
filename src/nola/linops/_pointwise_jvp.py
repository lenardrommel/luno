import jax
from jax import numpy as jnp
import linox

from collections.abc import Callable
from jax.typing import ArrayLike, DTypeLike
from linox.typing import ShapeLike


class PointwiseJVP(linox.LinearOperator):
    def __init__(
        self,
        f: Callable[[jax.Array], jax.Array],
        Df_shape: ShapeLike,
        x: ArrayLike,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        self._f = f

        self._Df_shape = linox.utils.as_shape(Df_shape)

        if jnp.ndim(x) != 2 or jnp.shape(x)[1] != self._Df_shape[1]:
            raise ValueError(f"`x` must have shape (n, {self._Df_shape[1]})")

        self._x = jnp.asarray(x)
        self._N = self._x.shape[0]

        self._Dfx_v = jnp.vectorize(
            lambda x, v: jax.jvp(self._f, (x,), (v,))[1],
            signature="(din,k),(din,k)->(dout,k)",
        )

        super().__init__(
            shape=(self._N * self._Df_shape[0], self._N * self._Df_shape[1]),
            dtype=jnp.dtype(dtype),
        )

    def _matmul(self, v: jax.Array) -> jax.Array:
        batch_shape = v.shape[:-2]
        ncols = v.shape[-1]

        v = v.reshape(batch_shape + (self._N, self._Df_shape[1], ncols))
        Dfx_v = self._Dfx_v(self._x, v)
        return Dfx_v.reshape(batch_shape + (self._N, self._Df_shape[0], ncols))
