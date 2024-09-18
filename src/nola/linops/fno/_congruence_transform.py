import jax
from jax import numpy as jnp
import linox
from linox._arithmetic import CongruenceTransform

from ._spectral_convolution import FixedInputSpectralConvolution

########################################################################################
# (FixedInputSpectralConvolution, Identity) ############################################
########################################################################################


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

    if A.output_grid_shape[-1] % 2 == 0:
        diag_val += 4 * jnp.sum(z_abs_sq[..., 1:-1, :])
        diag_val += jnp.sum(z_abs_sq[..., -1, :])
    else:
        diag_val += 4 * jnp.sum(z_abs_sq[..., 1:, :])

    return jnp.full(
        AAT.shape[0],
        diag_val,
        dtype=AAT.dtype,
    )


########################################################################################
# (FixedInputSpectralConvolution, Scalar) ##############################################
########################################################################################


@linox.congruence_transform.dispatch
def _(
    A: FixedInputSpectralConvolution,
    B: linox.Scalar,
):
    return B.scalar * CongruenceTransform_FixedInputSpectralConvolution_Identity(A, B)
