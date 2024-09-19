import jax
from jax import numpy as jnp
import neuralop
import neuralop.layers
import neuralop.layers.spectral_convolution
import numpy as np
import torch

from pytest_cases import fixture, parametrize

import nola

from typing import NamedTuple


class FixtureParameters(NamedTuple):
    grid_shape_in: tuple[int, ...]
    num_channels_in: int
    num_modes: tuple[int, ...]
    grid_shape_out: tuple[int, ...]
    num_channels_out: int

    def __repr__(self) -> str:
        return "-".join(
            f"{field}={repr(value)}" for field, value in zip(self._fields, iter(self))
        )


@fixture(scope="session")
@parametrize(
    "params",
    [
        FixtureParameters(
            grid_shape_in=(16, 16),
            num_channels_in=4,
            num_modes=(16, 16),
            grid_shape_out=(32, 32),
            num_channels_out=4,
        )
    ],
    idgen=lambda params: repr(params),
)
def _fixture_parameters(params: FixtureParameters) -> FixtureParameters:
    return params


@fixture(scope="session")
def grid_shape_in(_fixture_parameters: FixtureParameters) -> tuple[int, ...]:
    return _fixture_parameters.grid_shape_in


@fixture(scope="session")
def num_channels_in(_fixture_parameters: FixtureParameters) -> int:
    return _fixture_parameters.num_channels_in


@fixture(scope="session")
def grid_shape_out(_fixture_parameters: FixtureParameters) -> tuple[int, ...]:
    return _fixture_parameters.grid_shape_out


@fixture(scope="session")
def num_channels_out(_fixture_parameters: FixtureParameters) -> int:
    return _fixture_parameters.num_channels_out


@fixture(scope="session")
def _neuralop_spectral_conv(
    _fixture_parameters: FixtureParameters,
) -> neuralop.layers.spectral_convolution.SpectralConv:
    torch.manual_seed(453879)

    return neuralop.layers.spectral_convolution.SpectralConv(
        in_channels=_fixture_parameters.num_channels_in,
        out_channels=_fixture_parameters.num_channels_out,
        n_modes=_fixture_parameters.num_modes,
        max_n_modes=None,
        bias=False,
        n_layers=1,
        separable=False,
        output_scaling_factor=None,
    )


@fixture(scope="session")
def R(
    _neuralop_spectral_conv: neuralop.layers.spectral_convolution.SpectralConv,
) -> jax.Array:
    R = _neuralop_spectral_conv.weight[0]  # shape = (C_in, C_out, M_1, ..., M_D)

    R = jnp.asarray(R.to_tensor().detach())
    R = jnp.moveaxis(R, 1, -1)  # shape = (C_in, M_1, ..., M_D, C_out)
    R = jnp.moveaxis(R, 0, -1)  # shape = (M_1, ..., M_D, C_out, C_in)

    R = jnp.fft.fftshift(R, axes=tuple(range(R.ndim - 3)))

    return R


@fixture(scope="session")
def modes_shape(R: jax.Array) -> tuple[int, ...]:
    return R.shape[:-2]


@fixture(scope="session")
def v_in(grid_shape_in: tuple[int, ...], num_channels_in: int) -> jax.Array:
    return jax.random.normal(
        jax.random.key(345786),
        shape=grid_shape_in + (num_channels_in,),
    )


@fixture(scope="session")
def z_in(v_in: jax.Array, modes_shape: tuple[int, ...]) -> jax.Array:
    return nola.models.fno.dft.rfftn(
        v_in,
        modes_shape=modes_shape,
        axes=tuple(range(len(modes_shape))),
        norm="forward",
    )


@fixture(scope="session")
def v_out_ref(
    v_in: jax.Array,
    _neuralop_spectral_conv: neuralop.layers.spectral_convolution.SpectralConv,
    grid_shape_out: tuple[int, ...],
) -> jax.Array:
    v_in_torch = torch.as_tensor(np.asarray(v_in))  # shape = (N_1, N_2, ..., N_D, C_in)
    v_in_torch = torch.moveaxis(v_in_torch, -1, 0)  # shape = (C_in, N_1, N_2, ..., N_D)
    v_in_torch = v_in_torch[None, ...]  # shape = (1, C_in, N_1, N_2, ..., N_D)

    v_out_ref_torch = _neuralop_spectral_conv(v_in_torch, output_shape=grid_shape_out)

    v_out_ref = jnp.asarray(
        v_out_ref_torch.detach().numpy()  # shape = (1, C_out, N_1, N_2, ..., N_D)
    )
    v_out_ref = v_out_ref[0, ...]  # shape = (C_out, N_1, N_2, ..., N_D)
    v_out_ref = jnp.moveaxis(v_out_ref, 0, -1)  # shape = (N_1, N_2, ..., N_D, C_out)

    return v_out_ref
