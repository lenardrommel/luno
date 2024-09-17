import jax
from jax import numpy as jnp
from neuralop.layers.spectral_convolution import SpectralConv
import numpy as np
import torch

from pytest_cases import fixture

from nola.models.fno import spectral_convolution


@fixture(scope="module")
def num_channels_in() -> int:
    return 4


@fixture(scope="module")
def num_channels_out() -> int:
    return 2


@fixture(scope="module")
def num_modes() -> tuple[int, ...]:
    return (8, 4)


@fixture(scope="module")
def grid_shape_in() -> tuple[int, ...]:
    return (32, 16)


@fixture(scope="module")
def reference_impl(
    num_channels_in: int,
    num_channels_out: int,
    num_modes: tuple[int, ...],
) -> SpectralConv:
    torch.manual_seed(453879)

    return SpectralConv(
        in_channels=num_channels_in,
        out_channels=num_channels_out,
        n_modes=num_modes,
        bias=False,
    )


@fixture(scope="module")
def R(reference_impl: SpectralConv) -> jax.Array:
    R = reference_impl.weight[0].to_tensor().detach()

    R = jnp.asarray(R)  # shape = (C_in, C_out, M_1, ..., M_D)
    R = jnp.moveaxis(R, 1, -1)  # shape = (C_in, M_1, ..., M_D, C_out)
    R = jnp.moveaxis(R, 0, -1)  # shape = (M_1, ..., M_D, C_out, C_in)

    R = jnp.fft.fftshift(R, axes=tuple(range(R.ndim - 3)))

    return R


@fixture(scope="module")
def v_in(grid_shape_in: tuple[int, ...], num_channels_in: int) -> jax.Array:
    key = jax.random.key(345786)
    return jax.random.normal(key, grid_shape_in + (num_channels_in,))


def test_spectral_convolution(
    v_in: jax.Array,
    R: jax.Array,
    reference_impl: SpectralConv,
):
    v_out = spectral_convolution(v_in, R)

    v_in_torch = torch.as_tensor(np.asarray(v_in))  # shape = (N_1, N_2, ..., N_D, C_in)
    v_in_torch = torch.moveaxis(v_in_torch, -1, 0)  # shape = (C_in, N_1, N_2, ..., N_D)
    v_out_ref = reference_impl(v_in_torch[None, ...])[0, ...].detach().numpy()
    v_out_ref = np.moveaxis(v_out_ref, 0, -1)  # shape = (N_1, N_2, ..., N_D, C_out)

    np.testing.assert_allclose(
        v_out,
        v_out_ref,
        atol=1e-6,
        rtol=1e-6,
    )
