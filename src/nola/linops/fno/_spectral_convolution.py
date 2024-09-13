import functools
import operator

import jax
from jax import numpy as jnp
import linox
from linox._arithmetic import CongruenceTransform

from jax.typing import ArrayLike
from linox.typing import ShapeLike


class FixedInputSpectralConvolution(linox.LinearOperator):
    r"""Linear operator representation of the spectral convolution in an FNO block as a
    function of the spectral weight tensor :math:`R` with fixed input signal :math:`v`.

    This linear operator assumes that :math:`R \in \R^{2 \times C \times M \times C}`
    contains the real and imaginary parts of the spectral weights along the first index.
    In one spatial dimension, this linear operator implements the operation

    .. math::
        R \mapsto \left(
            \operatorname{rfft}^{-1} \left(
                \sum_{c = 1}^C (R_{0c'mc} + i R_{0c'mc}) z_{mc}
            \right)_{m = 1}^M
        \right)_{c' = 1}^C,

    where :math:`z \in \C^{M \times C}` is given by :math:`z_{\colon c} =
    \operatorname{rfft}(v_{\colon c})`.
    The generalization to multiple spatial dimensions is analogous.

    Parameters
    ----------
    input_signal :
        Input signal :math:`v` with shape :math:`(Nin_1, \ldots, Nin_D, C)`.
    output_grid_shape :
        Spatial shape :math:`(Nout_1, \ldots, Nout_D)` of the output signal.
        This is passed to the `s` argument in `jax.numpy.fft.irfftn`.
    """

    def __init__(
        self,
        input_signal: ArrayLike,
        output_grid_shape: ShapeLike | None = None,
        input_signal_rfft: ArrayLike | None = None,
    ):
        if jnp.ndim(input_signal) < 2:
            raise ValueError("`input_signal` must have at least 2 dimensions.")

        self._input_signal = jnp.asarray(input_signal)  # shape = (Nin_1, ..., Nin_D, C)

        self._D = self._input_signal.ndim - 1
        self._C = self._input_signal.shape[-1]

        if output_grid_shape is None:
            output_grid_shape = self._input_signal.shape[:-1]

        self._output_grid_shape = linox.utils.as_shape(output_grid_shape)

        if len(self._output_grid_shape) != self._D:
            raise ValueError(
                f"Expected `output_grid_shape` to have length {self._D}, "
                f"but got {len(self._output_grid_shape)}."
            )

        self._Nout = functools.reduce(operator.mul, self._output_grid_shape, 1)

        # Precompute the Fourier transform of the input signal
        if input_signal_rfft is None:
            input_signal_rfft = jnp.fft.rfftn(
                self._input_signal,
                axes=tuple(range(self._D)),
                norm="forward",
            )

        self._input_signal_rfft = jnp.asarray(
            input_signal_rfft
        )  # shape = (M_1, ..., M_D, C)
        self._z_flat = self._input_signal_rfft.reshape(
            -1, self._input_signal_rfft.shape[-1]
        )  # shape = (M, C)

        self._Ms = self._input_signal_rfft.shape[:-1]
        self._M = self._z_flat.shape[0]

        super().__init__(
            shape=(self._Nout * self._C, 2 * self._C * self._M * self._C),
            dtype=self._input_signal.dtype,
        )

    @property
    def input_signal(self) -> jax.Array:
        return self._input_signal

    @property
    def output_grid_shape(self) -> tuple[int, ...]:
        return self._output_grid_shape

    @property
    def input_signal_rfft(self) -> jax.Array:
        return self._input_signal_rfft

    @property
    def num_channels(self) -> int:
        return self._C

    @property
    def num_modes(self) -> int:
        return self._M

    def _matmul(self, R: jax.Array) -> jax.Array:
        batch_shape = R.shape[:-2]
        ncols = R.shape[-1]

        R = R.reshape(batch_shape + (2, self._C, self._M, self._C, ncols))
        R = (
            R[..., 0, :, :, :, :] + 1j * R[..., 1, :, :, :, :]
        )  # shape = (..., C, M, C, ncols)

        v = jnp.fft.irfftn(
            jnp.reshape(
                jnp.sum(R * self._z_flat[..., None], axis=-2),
                R.shape[:-3] + self._Ms + (ncols,),
            ),
            s=self._output_grid_shape,
            axes=tuple(range(-1 - self._D, -1)),
            norm="forward",
        )  # shape = (..., C) + output_grid_shape + (ncols,)

        v = jnp.moveaxis(
            v, -2 - self._D, -2
        )  # shape = batch_shape + output_grid_shape + (C, ncols)

        return v.reshape(batch_shape + (self.shape[0], ncols))


class FixedInputSpectralConvolutionOuterProduct(CongruenceTransform):
    pass


@linox.congruence_transform.dispatch
def _(
    A: FixedInputSpectralConvolution, B: linox.Identity
) -> FixedInputSpectralConvolutionOuterProduct:
    return FixedInputSpectralConvolutionOuterProduct(A, B)


@linox.diagonal.dispatch
def _(AAT: FixedInputSpectralConvolutionOuterProduct) -> jax.Array:
    A = AAT._A

    z = A.input_signal_rfft
    z_abs_sq = z.real**2 + z.imag**2

    diag_val = jnp.sum(z_abs_sq[..., 0, :])

    if A.input_signal.shape[-2] % 2 == 0:
        diag_val += 4 * jnp.sum(z_abs_sq[..., 1:-1, :])
        diag_val += jnp.sum(z_abs_sq[..., -1, :])
    else:
        diag_val += 4 * jnp.sum(z_abs_sq[..., 1:, :])

    return jnp.full(
        AAT.shape[0],
        diag_val,
        dtype=AAT.dtype,
    )
