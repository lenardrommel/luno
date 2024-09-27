import functools
import operator

import jax
from jax import numpy as jnp
import linox
from linox._arithmetic import CongruenceTransform

from nola.models.fno import fno_block
from nola.models.fno._periodic_interpolation import gridded_fourier_interpolation

from collections.abc import Callable
from jax.typing import ArrayLike, DTypeLike


class LastFNOBlockWeightJacobian(linox.LinearOperator):
    def __init__(
        self,
        v_in: ArrayLike,
        R_0: ArrayLike,
        W_0: ArrayLike,
        b_0: ArrayLike,
        output_grid_shape: tuple[int, ...] | None,
        projection: Callable[[jax.Array], jax.Array],
        num_output_channels: int,
        z_in: ArrayLike | None = None,
        v_out_0: ArrayLike | None = None,
        dtype: DTypeLike = jnp.single,
    ) -> None:
        self._v_in = jnp.asarray(v_in)

        self._R_0 = jnp.asarray(R_0)
        self._W_0 = jnp.asarray(W_0)
        self._b_0 = jnp.asarray(b_0)

        self._output_grid_shape = (
            output_grid_shape
            if output_grid_shape is not None
            else self._v_in.shape[:-1]
        )

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

        # Caches for intermediate values
        self._z_in = z_in
        self._v_out_0 = v_out_0

        super().__init__(
            shape=(
                self.output_grid_size * self._num_output_channels,
                2 * self._R_0.size + self._W_0.size + self._b_0.size,
            ),
            dtype=dtype,
        )

    @property
    def v_in(self) -> jax.Array:
        return self._v_in

    @property
    def num_input_channels(self) -> int:
        return self._v_in.shape[-1]

    @property
    def R_0(self) -> jax.Array:
        return self._R_0

    @property
    def num_modes(self) -> int:
        return self._R_0[..., 0, 0].size

    @property
    def num_hidden_channels(self) -> int:
        return self._R_0.shape[-2]

    @property
    def W_0(self) -> jax.Array:
        return self._W_0

    @property
    def b_0(self) -> jax.Array:
        return self._b_0

    @property
    def input_grid_shape(self) -> tuple[int, ...]:
        return self._v_in.shape[:-1]

    @property
    def input_grid_size(self) -> int:
        return functools.reduce(operator.mul, self.input_grid_shape)

    @property
    def output_grid_shape(self) -> tuple[int, ...]:
        return self._output_grid_shape

    @property
    def output_grid_size(self) -> int:
        return functools.reduce(operator.mul, self._output_grid_shape)

    @property
    def projection(self) -> Callable[[jax.Array], jax.Array]:
        return self._projection

    @property
    def num_output_channels(self) -> int:
        return self.num_output_channels

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

    @property
    def z_in(self) -> jax.Array:
        if self._z_in is None:
            self._compute_intermediates()

        return self._z_in

    @property
    def v_out_0(self) -> jax.Array:
        if self._v_out_0 is None:
            self._compute_intermediates()

        return self._v_out_0

    def _matmul(self, weights: jax.Array) -> jax.Array:
        return self._vectorized_jvp(weights)

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

        v_out, _ = fno_block(
            self._v_in,
            R,
            W,
            b,
            output_grid_shape=self._output_grid_shape,
        )

        y = self._projection(v_out)

        return y.reshape(-1)

    def _compute_intermediates(self) -> None:
        self._v_out_0, intermediates = fno_block(
            self._v_in,
            self._R_0,
            self._W_0,
            self._b_0,
            self._output_grid_shape,
        )

        self._z_in = intermediates["spectral_convolution"]["z_in"]


########################################################################################
# Congruence Transforms ################################################################
########################################################################################


#############################################
# (LastFNOBlockWeightJacobian, Identity) #
#############################################


class CongruenceTransform_LastFNOBlockWeightJacobian_Identity(CongruenceTransform):
    @functools.cached_property
    def projection_jacobian_outer_product_diagonal(self) -> jax.Array:
        v_out_0 = self._A.v_out_0.reshape(-1, self._A.num_hidden_channels)

        J = jax.vmap(
            jax.jacobian(self._A.projection),
            in_axes=0,
            out_axes=0,
        )(v_out_0)

        return jnp.sum(J**2, axis=-1)


@linox.congruence_transform.dispatch
def _(
    A: LastFNOBlockWeightJacobian, B: linox.Identity
) -> CongruenceTransform_LastFNOBlockWeightJacobian_Identity:
    return CongruenceTransform_LastFNOBlockWeightJacobian_Identity(A, B)


@linox.diagonal.dispatch
def _(JJT: CongruenceTransform_LastFNOBlockWeightJacobian_Identity) -> jax.Array:
    J: LastFNOBlockWeightJacobian = JJT._A

    # Spectral convolution
    z = J.z_in
    z_abs_sq = z.real**2 + z.imag**2

    diag_sconv = jnp.sum(z_abs_sq[..., 0, :])

    if any(m_out < m_in for m_out, m_in in zip(J.output_grid_shape[:-1], z.shape[:-2])):
        raise NotImplementedError()

    m = J.output_grid_shape[-1] // 2 + 1

    if J.output_grid_shape[-1] % 2 == 0 and m <= z.shape[-2]:
        diag_sconv += 4 * jnp.sum(z_abs_sq[..., 1 : min(m, z.shape[-2]) - 1, :])
        diag_sconv += jnp.sum(z_abs_sq[..., m - 1, :])
    else:
        diag_sconv += 4 * jnp.sum(z_abs_sq[..., 1 : min(m, z.shape[-2]), :])

    # Skip connection
    v = J.v_in
    v_interp = gridded_fourier_interpolation(
        v,
        axes=tuple(range(len(J.output_grid_shape))),
        output_grid_shape=J.output_grid_shape,
    )

    diag_skip = jnp.sum(v_interp**2, axis=-1)
    diag_skip = diag_skip.reshape(-1, 1)

    # Bias
    diag_bias = 1

    # Projection
    diag_proj = JJT.projection_jacobian_outer_product_diagonal

    diag = diag_proj * (diag_sconv + diag_skip + diag_bias)

    return diag.reshape(-1)
