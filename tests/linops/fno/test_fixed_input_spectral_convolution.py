import jax
from jax import numpy as jnp
import linox
import numpy as np

from pytest_cases import AUTO, fixture, parametrize

from nola.linops.fno import FixedInputSpectralConvolution


@fixture(scope="module")
@parametrize("C", (1, 2, 3, 4), idgen=AUTO)
@parametrize(
    "input_grid_shape",
    [
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (1, 1),
        (3, 3),
        (2, 3),
        (3, 2),
        (1, 1, 1),
        (4, 2, 4),
        (2, 4, 2),
        (3, 3, 3),
    ],
    idgen=AUTO,
)
def input_signal(input_grid_shape: tuple[int], C: int) -> jax.Array:
    key = jax.random.key(42)

    return jax.random.normal(key, shape=input_grid_shape + (C,))


@fixture(scope="module")
def input_signal_rdft(input_signal: jax.Array) -> jax.Array:
    return jnp.fft.rfftn(
        input_signal,
        axes=tuple(range(input_signal.ndim - 1)),
        norm="forward",
    )


@fixture(scope="module")
def fixed_input_spectral_convolution(
    input_signal: jax.Array,
    input_signal_rdft: jax.Array,
) -> FixedInputSpectralConvolution:
    return FixedInputSpectralConvolution(
        input_signal_rdft,
        output_grid_shape=input_signal.shape[:-1],
    )


def test_matmul_identity_weight_reproduces_input_signal(
    input_signal: jax.Array,
    fixed_input_spectral_convolution: FixedInputSpectralConvolution,
):
    linop = fixed_input_spectral_convolution

    # Create and reshape identity weights
    R_real = jnp.eye(linop.num_input_channels)
    R_real = jnp.broadcast_to(
        R_real,
        (linop.num_modes, linop.num_output_channels, linop.num_input_channels),
    )
    R = jnp.stack([R_real, jnp.zeros_like(R_real)], axis=0)
    R = R.reshape(-1)

    output_signal_flat = linop @ R
    output_signal = output_signal_flat.reshape(
        linop.output_grid_shape + (linop.num_output_channels,)
    )

    np.testing.assert_allclose(
        output_signal,
        input_signal,
        rtol=1e-6,
        atol=1e-6,
    )


def test_outer_product_diagonal(
    fixed_input_spectral_convolution: FixedInputSpectralConvolution,
):
    A = fixed_input_spectral_convolution

    I = linox.Identity(A.shape[1])
    AAT = linox.congruence_transform(A, I)

    np.testing.assert_allclose(
        linox.diagonal(AAT),
        jnp.diag(AAT.todense()),
        rtol=1e-6,
        atol=1e-6,
    )
