import jax
from jax import numpy as jnp
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
    key = jax.random.PRNGKey(42)

    return jax.random.normal(key, shape=input_grid_shape + (C,))


@fixture(scope="module")
def fixed_input_spectral_convolution(
    input_signal: jax.Array,
) -> FixedInputSpectralConvolution:
    return FixedInputSpectralConvolution(input_signal)


def test_matmul_identity_weight_reproduces_input_signal(
    fixed_input_spectral_convolution: FixedInputSpectralConvolution,
):
    linop = fixed_input_spectral_convolution

    # Create and reshape identity weights
    R_real = jnp.eye(linop._C)
    R_real = jnp.broadcast_to(R_real[:, None, :], (linop._C, linop._M, linop._C))
    R = jnp.stack([R_real, jnp.zeros_like(R_real)], axis=0)
    R = R.reshape(-1)

    output_signal_flat = linop @ R
    output_signal = output_signal_flat.reshape(linop._output_grid_shape + (linop._C,))

    np.testing.assert_allclose(
        output_signal,
        linop._input_signal,
        rtol=1e-6,
        atol=1e-6,
    )
