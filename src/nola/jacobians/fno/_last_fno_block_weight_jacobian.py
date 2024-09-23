import functools
import operator

import jax
from jax import numpy as jnp
import linox

from nola.models.fno import fno_block

from collections.abc import Callable
from jax.typing import ArrayLike, DTypeLike


class LastFNOBlockWeightJacobian(linox.LinearOperator):
    def __init__(
        self,
        v_in: ArrayLike,
        R_0: ArrayLike,
        W_0: ArrayLike,
        b_0: ArrayLike,
        output_grid_shape: tuple[int, ...],
        projection: Callable[[jax.Array], jax.Array],
        num_output_channels: int,
        dtype: DTypeLike = jnp.single,
    ) -> None:
        self._v_in = jnp.asarray(v_in)

        self._R_0 = jnp.asarray(R_0)
        self._W_0 = jnp.asarray(W_0)
        self._b_0 = jnp.asarray(b_0)

        self._output_grid_shape = output_grid_shape

        self._projection = projection
        self._num_output_channels = num_output_channels

        self._vectorized_jvp = jnp.vectorize(
            jax.vmap(
                lambda weights: jax.jvp(
                    self._last_fno_block_forward,
                    (self._weights_0,),
                    (weights,),
                )[1],
                in_axes=-1,
                out_axes=-1,
            ),
            signature=f"(n,k)->(m,k)",
        )

        super().__init__(
            shape=(
                self.output_grid_size * self._num_output_channels,
                2 * self._R_0.size + self._W_0.size + self._b_0.size,
            ),
            dtype=dtype,
        )

    @property
    def num_input_channels(self) -> int:
        return self._v_in.shape[-1]

    @property
    def num_hidden_channels(self) -> int:
        return self._R_0.shape[-2]

    @property
    def num_output_channels(self) -> int:
        return self.num_output_channels

    @property
    def num_modes(self) -> int:
        return self._R_0[..., 0, 0].size

    @property
    def output_grid_size(self) -> int:
        return functools.reduce(operator.mul, self._output_grid_shape)

    @functools.cached_property
    def _weights_0(self) -> jax.Array:
        return jnp.concatenate(
            (
                self._R_0.real.reshape(-1),
                self._R_0.imag.reshape(-1),
                self._W_0.reshape(-1),
                self._b_0.reshape(-1),
            )
        )

    def _matmul(self, weights: jax.Array) -> jax.Array:
        # Compute the Jacobian
        J = self._vectorized_jvp(weights)

        return J

    def _last_fno_block_forward(self, weights: jax.Array) -> jax.Array:
        # Extract R, W, and b from weights
        R, W, b = jnp.split(
            weights,
            (2 * self._R_0.size, 2 * self._R_0.size + self._W_0.size),
        )

        R = R.reshape(
            2, self.num_modes, self.num_hidden_channels, self.num_input_channels
        )
        R = R[0, :, :, :] + 1j * R[1, :, :, :]
        R = R.reshape(self._R_0.shape)

        W = W.reshape(self._W_0.shape)

        b = b.reshape(self._b_0.shape)

        v_out = fno_block(
            self._v_in,
            R,
            W,
            b,
            output_grid_shape=self._output_grid_shape,
        )

        y = self._projection(v_out)

        return y.reshape(-1)
