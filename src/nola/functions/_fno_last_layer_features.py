import functools

import jax
from jax import numpy as jnp
import linox

from nola import linops

from collections.abc import Callable
from jax.typing import ArrayLike


class FNOLastLayerJacobianFeatures:
    def __init__(
        self,
        hidden_signal: ArrayLike,
        weight_mean: tuple[jax.Array, jax.Array],
        projection: Callable[[jax.Array], jax.Array],
        num_output_channels: int,
    ) -> None:
        self._hidden_signal = jnp.asarray(hidden_signal)

        self._hidden_signal_interp = jax.scipy.interpolate.RegularGridInterpolator(
            points=tuple(
                jnp.linspace(0, 1, n + 1)[:-1] for n in self._hidden_signal.shape[:-1]
            ),
            values=self._hidden_signal,  # TODO: Periodic padding
            method="linear",
        )

        self._mean_R, self._mean_W = weight_mean

        self._projection = projection
        self._num_output_channels = num_output_channels

    @property
    def num_input_dimensions(self) -> int:
        return self._hidden_signal.ndim - 1

    @property
    def num_hidden_channels(self) -> int:
        return self._hidden_signal.shape[-1]

    @property
    def num_output_channels(self) -> int:
        return self._num_output_channels

    @functools.cached_property
    def _z(self) -> jax.Array:
        return jnp.fft.rfftn(
            self._hidden_signal,
            axes=tuple(range(self._hidden_signal.ndim - 1)),
            norm="forward",
        )

    @functools.cached_property
    def _mean_Rz(self) -> jax.Array:
        return jnp.sum(self._mean_R * self._z, axis=-1)

    def _fno_block_mean(self, x: jax.Array) -> jax.Array:
        # TODO: Support irregular grids
        v_out_sconv = jnp.fft.irfftn(
            self._mean_Rz,
            s=x.shape[:-1],
            axes=tuple(range(-(x.ndim - 1), self._mean_Rz.ndim)),
            norm="forward",
        )

        v_out_pointwise = (self._mean_W @ self._hidden_signal_interp(x)[..., None])[
            ..., 0
        ]

        return v_out_sconv + v_out_pointwise

    def __call__(self, x: jax.Array) -> jax.Array:
        # TODO: Support irregular grids
        sconv = linops.fno.FixedInputSpectralConvolution(
            input_signal=self._hidden_signal,
            output_grid_shape=x.shape[:-1],
            input_signal_rfft=self._z,
        )
        affine_skip = linops.FixedInputPointwiseAffineTransform(
            self._hidden_signal_interp(x.reshape(-1, self.num_input_dimensions))
        )

        fno_block = linox.BlockMatrix([[sconv, affine_skip]])

        embedding_jacobian = linops.PointwiseJVP(
            f=self._projection,
            Df_shape=(self.num_output_channels, self.num_hidden_channels),
            x=self._fno_block_mean(x).reshape(-1, self.num_hidden_channels),
        )

        return embedding_jacobian @ fno_block
