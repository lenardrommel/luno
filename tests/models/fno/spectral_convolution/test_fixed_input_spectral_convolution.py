import jax
from jax import numpy as jnp
import linox
import numpy as np

import pytest
from pytest_cases import AUTO, fixture, parametrize

import nola
from nola.models.fno import FixedInputSpectralConvolution


@fixture(scope="module")
def fixed_input_spectral_convolution(
    grid_shape_out: tuple[int, ...],
    num_channels_out: int,
    z_in: jax.Array,
) -> FixedInputSpectralConvolution:
    return FixedInputSpectralConvolution(
        z_in,
        output_grid_shape=grid_shape_out,
        num_output_channels=num_channels_out,
    )


def test_matmul(
    fixed_input_spectral_convolution: FixedInputSpectralConvolution,
    R: jax.Array,
    v_out_ref: jax.Array,
):
    R_real = jnp.stack((R.real, R.imag), axis=0)
    v_out = fixed_input_spectral_convolution @ R_real.reshape(-1)

    np.testing.assert_allclose(
        v_out,
        v_out_ref.reshape(-1),
        rtol=1e-6,
        atol=1e-6,
    )


@parametrize("batch_shape", ((), (3,), (2, 3)), idgen=AUTO)
@parametrize("ncols", (1, 2, 3))
def test_matmul_batching(
    fixed_input_spectral_convolution: FixedInputSpectralConvolution,
    batch_shape: tuple[int, ...],
    ncols: int,
):
    key = jax.random.key(5842)
    Rs = jax.random.normal(
        key,
        shape=batch_shape + (fixed_input_spectral_convolution.shape[-1], ncols),
    )

    vs_out = fixed_input_spectral_convolution @ Rs

    vs_out_ref = jnp.vectorize(
        jax.vmap(
            lambda R: fixed_input_spectral_convolution @ R,
            in_axes=1,
            out_axes=1,
        ),
        signature="(n,k)->(m,k)",
    )(Rs)

    np.testing.assert_allclose(
        vs_out,
        vs_out_ref,
        rtol=1e-5,
        atol=1e-5,
    )


def test_matmul_identity_weight_reproduces_resampled_input(
    num_channels_in: int,
    grid_shape_out: tuple[int, ...],
    num_channels_out: int,
    R: jax.Array,
    z_in: jax.Array,
    fixed_input_spectral_convolution: FixedInputSpectralConvolution,
):
    if num_channels_in != num_channels_out:
        pytest.skip()

    # Create and reshape identity weights
    R_id = jnp.eye(*R.shape[-2:])
    R_id = jnp.broadcast_to(R_id, R.shape)
    R_id = jnp.stack([R_id, jnp.zeros_like(R_id)], axis=0)
    R_id = R_id.reshape(-1)

    v_out = fixed_input_spectral_convolution @ R_id

    # Resample the truncated input signal
    v_out_ref = nola.models.fno.dft.irfftn(
        z_in,
        grid_shape=grid_shape_out,
        axes=tuple(range(len(grid_shape_out))),
        norm="forward",
    )

    np.testing.assert_allclose(
        v_out,
        v_out_ref.reshape(-1),
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
