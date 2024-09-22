import functools
import operator

import jax
from jax import numpy as jnp
import linox
from linox._arithmetic import CongruenceTransform

from nola.models.fno.dft import irfftn

from jax.typing import ArrayLike
from linox.typing import ShapeLike


class FixedInputSpectralConvolution(linox.LinearOperator):
    r"""Linear operator representation of the spectral convolution in an FNO block as a
    function of the spectral weight tensor :math:`R` with fixed input signal :math:`v`.

    This linear operator assumes that

    .. math::
        R \in \R^{2 \times M_1 \dotsb \times M_D \times C_{out} \times C_{in}}`

    contains the real and imaginary parts of the spectral weights along the first index.
    This linear operator implements the operation

    .. math::
        R \mapsto \left(
            \operatorname{rfft}^{-1} \left(
                \sum_{c = 1}^{C_{in}} (R_{0mc'c} + i R_{1mc'c}) z_{mc}
            \right)_{m = (1, \dotsc, 1)}^{(M_1, \dotsc, M_D)}
        \right)_{c' = 1}^{C_{out}},

    where :math:`z \in \C^{M_1 \times \dotsb \times M_D \times C}`.

    Parameters
    ----------
    z :
        (Truncated) real discrete Fourier transform of the signal :math:`v` with shape
        :math:`(M_1, \ldots, M_D, C_in)`.
        This has to be "forward" normalized.
    output_grid_shape :
        Spatial shape :math:`(Nout_1, \ldots, Nout_D)` of the output signal.
    num_output_channels :
        Number :math:`C_{out}` of channels in the output signal. By default, this is set
        to :math:`C_{in}`.
    """

    def __init__(
        self,
        z: ArrayLike,
        output_grid_shape: ShapeLike,
        num_output_channels: int | None = None,
    ):
        if jnp.ndim(z) < 2:
            raise ValueError("`z` must have at least 2 dimensions.")

        self._z = jnp.asarray(z)  # shape = (M_1, ..., M_D, C)

        self._Ms = self._z.shape[:-1]
        self._M = functools.reduce(operator.mul, self._Ms, 1)
        self._D = len(self._Ms)

        self._C_in = self._z.shape[-1]

        self._output_grid_shape = linox.utils.as_shape(output_grid_shape)

        if len(self._output_grid_shape) != self._D:
            raise ValueError(
                f"Expected `output_grid_shape` to have length {self._D}, "
                f"but got {len(self._output_grid_shape)}."
            )

        self._N = functools.reduce(operator.mul, self._output_grid_shape, 1)

        self._C_out = (
            num_output_channels if num_output_channels is not None else self._C_in
        )

        super().__init__(
            shape=(self._N * self._C_out, 2 * self._M * self._C_out * self._C_in),
            dtype=jnp.empty_like(self._z, shape=()).real.dtype,
        )

    @property
    def z(self) -> jax.Array:
        return self._z

    @property
    def output_grid_shape(self) -> tuple[int, ...]:
        return self._output_grid_shape

    @property
    def num_input_channels(self) -> int:
        return self._C_in

    @property
    def num_output_channels(self) -> int:
        return self._C_out

    @property
    def num_modes(self) -> int:
        return self._M

    def _matmul(self, R: jax.Array) -> jax.Array:
        batch_shape = R.shape[:-2]
        ncols = R.shape[-1]

        R = R.reshape(batch_shape + (2, self._M, self._C_out, self._C_in, ncols))
        R = (
            R[..., 0, :, :, :, :] + 1j * R[..., 1, :, :, :, :]
        )  # shape = (..., M, C_out, C_in, ncols)
        R = R.reshape(batch_shape + self._Ms + (self._C_out, self._C_in, ncols))

        v = irfftn(
            jnp.sum(R * self._z[..., None, :, None], axis=-2),
            grid_shape=self._output_grid_shape,
            axes=tuple(range(-2 - self._D, -2)),
            norm="forward",
        )  # shape = batch_shape + output_grid_shape + (C_out, ncols)

        return v.reshape(batch_shape + (self.shape[0], ncols))


########################################################################################
# Congruence Transforms ################################################################
########################################################################################


#############################################
# (FixedInputSpectralConvolution, Identity) #
#############################################


class CongruenceTransform_FixedInputSpectralConvolution_Identity(CongruenceTransform):
    pass


@linox.congruence_transform.dispatch
def _(
    A: FixedInputSpectralConvolution, B: linox.Identity
) -> CongruenceTransform_FixedInputSpectralConvolution_Identity:
    return CongruenceTransform_FixedInputSpectralConvolution_Identity(A, B)


@linox.diagonal.dispatch
def _(AAT: CongruenceTransform_FixedInputSpectralConvolution_Identity) -> jax.Array:
    A = AAT._A

    z = A.z
    z_abs_sq = z.real**2 + z.imag**2

    diag_val = jnp.sum(z_abs_sq[..., 0, :])

    if any(m_out < m_in for m_out, m_in in zip(A.output_grid_shape[:-1], z.shape[:-2])):
        raise NotImplementedError()

    m = A.output_grid_shape[-1] // 2 + 1

    if A.output_grid_shape[-1] % 2 == 0 and m <= z.shape[-2]:
        diag_val += 4 * jnp.sum(z_abs_sq[..., 1 : min(m, z.shape[-2]) - 1, :])
        diag_val += jnp.sum(z_abs_sq[..., m - 1, :])
    else:
        diag_val += 4 * jnp.sum(z_abs_sq[..., 1 : min(m, z.shape[-2]), :])

    return jnp.full(
        AAT.shape[0],
        diag_val,
        dtype=AAT.dtype,
    )


###########################################
# (FixedInputSpectralConvolution, Scalar) #
###########################################


@linox.congruence_transform.dispatch
def _(
    A: FixedInputSpectralConvolution,
    B: linox.Scalar,
):
    return B.scalar * CongruenceTransform_FixedInputSpectralConvolution_Identity(A, B)
