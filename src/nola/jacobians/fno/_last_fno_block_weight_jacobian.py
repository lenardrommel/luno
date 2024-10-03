import functools
import operator

import jax
from jax import numpy as jnp
import linox
from linox._arithmetic import CongruenceTransform

from nola.covariances.fno import CircularlySymmetricDiagonal
from nola.models.fno import dft, fno_block
from nola.models.fno._periodic_interpolation import gridded_fourier_interpolation

from collections.abc import Callable
from jax.typing import ArrayLike, DTypeLike


class LastFNOBlockWeightJacobian(linox.LinearOperator):
    def __init__(
        self,
        v_in: ArrayLike,
        R: ArrayLike,
        W: ArrayLike,
        b: ArrayLike,
        output_grid_shape: tuple[int, ...] | None,
        projection: Callable[[jax.Array], jax.Array],
        num_output_channels: int,
        z_in: ArrayLike | None = None,
        v_out: ArrayLike | None = None,
        dtype: DTypeLike = jnp.single,
    ) -> None:
        self._v_in = jnp.asarray(v_in)

        self._R = jnp.asarray(R)
        self._W = jnp.asarray(W)
        self._b = jnp.asarray(b)

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
        self._v_out = v_out

        super().__init__(
            shape=(
                self.output_grid_size * self._num_output_channels,
                2 * self._R.size + self._W.size + self._b.size,
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
    def R(self) -> jax.Array:
        return self._R

    @property
    def num_modes(self) -> int:
        return self._R[..., 0, 0].size

    @property
    def num_hidden_channels(self) -> int:
        return self._R.shape[-2]

    @property
    def W(self) -> jax.Array:
        return self._W

    @property
    def b(self) -> jax.Array:
        return self._b

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
                self._R.real.reshape(-1),
                self._R.imag.reshape(-1),
                self._W.reshape(-1),
                self._b.reshape(-1),
            )
        )

    @property
    def z_in(self) -> jax.Array:
        if self._z_in is None:
            self._compute_intermediates()

        return self._z_in

    @property
    def v_out(self) -> jax.Array:
        if self._v_out is None:
            self._compute_intermediates()

        return self._v_out

    def _matmul(self, weights: jax.Array) -> jax.Array:
        return self._vectorized_jvp(weights)

    def _last_fno_block_forward(self, weights: jax.Array) -> jax.Array:
        # Extract R, W, and b from weights
        R, W, b = jnp.split(
            weights,
            (2 * self._R.size, 2 * self._R.size + self._W.size),
        )

        R = R.reshape(
            2, self.num_modes, self.num_hidden_channels, self.num_input_channels
        )
        R = R[0, :, :, :] + 1j * R[1, :, :, :]
        R = R.reshape(self._R.shape)

        W = W.reshape(self._W.shape)

        b = b.reshape(self._b.shape)

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
        self._v_out, intermediates = fno_block(
            self._v_in,
            self._R,
            self._W,
            self._b,
            self._output_grid_shape,
        )

        self._z_in = intermediates["spectral_convolution"]["z_in"]


########################################################################################
# Congruence Transforms ################################################################
########################################################################################


#############################################################
# (LastFNOBlockWeightJacobian, CircularlySymmetricDiagonal) #
#############################################################


class CongruenceTransform_LastFNOBlockWeightJacobian_CircularlySymmetricDiagonal(
    CongruenceTransform
):
    @functools.cached_property
    def projection_pointwise_jacobian(self) -> jax.Array:
        return jax.vmap(
            jax.jacobian(self._A.projection),
            in_axes=0,
            out_axes=0,
        )(self._A.v_out.reshape(-1, self._A.num_hidden_channels))


@linox.congruence_transform.dispatch
def _(
    A: LastFNOBlockWeightJacobian, B: CircularlySymmetricDiagonal
) -> CongruenceTransform_LastFNOBlockWeightJacobian_CircularlySymmetricDiagonal:
    return CongruenceTransform_LastFNOBlockWeightJacobian_CircularlySymmetricDiagonal(
        A, B
    )


@linox.diagonal.dispatch
def _(
    JSigmaJT: CongruenceTransform_LastFNOBlockWeightJacobian_CircularlySymmetricDiagonal,
) -> jax.Array:
    J: LastFNOBlockWeightJacobian = JSigmaJT._A
    Sigma: CircularlySymmetricDiagonal = JSigmaJT._B

    D = len(J.input_grid_shape)

    ########################
    # Spectral convolution #
    ########################

    z = J.z_in  # shape: (M_in_1, ..., M_in_D, C_in)
    Sigma_R_real = Sigma.R_real  # shape: (M_in_1, ..., M_in_D, C_hidden, C_in)

    # Truncation
    input_modes_shape = z.shape[:-1]
    hidden_modes_shape = J.output_grid_shape[:-1] + (J.output_grid_shape[-1] // 2 + 1,)

    nnz_modes_shape = tuple(
        min(M_in, M_hidden)
        for M_in, M_hidden in zip(input_modes_shape, hidden_modes_shape, strict=True)
    )

    z = dft.rdftn_trunc(
        z,
        modes_shape=nnz_modes_shape,
        axes=tuple(range(D)),
    )
    Sigma_R_real = dft.rdftn_trunc(
        Sigma_R_real,
        modes_shape=nnz_modes_shape,
        axes=tuple(range(D)),
    )

    z_abs_sq = z.real**2 + z.imag**2

    # Rescaling along last mode axis
    alpha = jnp.full_like(
        z_abs_sq,
        4,
        dtype=jnp.uint8,
        shape=nnz_modes_shape[-1],
    )
    alpha = alpha.at[0].set(1)

    if (
        J.output_grid_shape[-1] % 2 == 0
        and hidden_modes_shape[-1] <= input_modes_shape[-1]
    ):
        alpha = alpha.at[-1].set(1)

    # Compute diagonal
    diag_sconv = jnp.sum(
        Sigma_R_real  # shape: (M_1, ..., M_D, C_hidden, C_in)
        * alpha[:, None, None]  # shape: (M_D, 1, 1)
        * z_abs_sq[..., None, :],  # shape: (M_1, ..., M_D, 1, C_in)
        axis=tuple(range(D)) + (-1,),
    )  # shape: (C_hidden,)

    ###################################
    # (Interpolated) Pointwise Linear #
    ###################################

    v = J.v_in
    v_interp = gridded_fourier_interpolation(
        v,
        axes=tuple(range(len(J.output_grid_shape))),
        output_grid_shape=J.output_grid_shape,
    )

    diag_pointwise = jnp.sum(
        v_interp[..., None, :] ** 2 * Sigma.W,
        axis=-1,
    ).reshape(J.output_grid_size, J.num_hidden_channels)

    #############
    # FNO Block #
    #############

    diag_fno_block = diag_sconv + diag_pointwise

    if J.b is not None:
        diag_fno_block += Sigma.b

    ##############
    # Projection #
    ##############

    J_proj = JSigmaJT.projection_pointwise_jacobian  # shape: (N_out, C_out, C_hidden)

    diag = jnp.sum(
        diag_fno_block[:, None, :] * J_proj**2,
        axis=-1,
    )  # shape: (N_out, C_out)

    return diag.reshape(-1)


##########################################
# (LastFNOBlockWeightJacobian, Identity) #
##########################################


@linox.congruence_transform.dispatch
def _(
    A: LastFNOBlockWeightJacobian, B: linox.Identity
) -> CongruenceTransform_LastFNOBlockWeightJacobian_CircularlySymmetricDiagonal:
    return CongruenceTransform_LastFNOBlockWeightJacobian_CircularlySymmetricDiagonal(
        A,
        CircularlySymmetricDiagonal(
            R_real=jnp.ones_like(A.R, dtype=jnp.float32),
            W=jnp.ones_like(A.W),
            b=jnp.ones_like(A.b) if A.b is not None else None,
        ),
    )
