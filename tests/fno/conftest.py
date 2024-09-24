import cases
import jax
from jax import numpy as jnp
import neuralop
import neuralop.layers
import neuralop.layers.fno_block
import neuralop.layers.mlp
import neuralop.layers.spectral_convolution
import numpy as np
import torch

from pytest_cases import fixture, parametrize_with_cases

import nola


@fixture(scope="session")
@parametrize_with_cases("params", cases=cases, scope="session")
def _case(params: cases.Case) -> cases.Case:
    return params


@fixture(scope="session")
def grid_shape_in(_case: cases.Case) -> tuple[int, ...]:
    return _case.grid_shape_in


@fixture(scope="session")
def num_channels_in(_case: cases.Case) -> int:
    return _case.num_channels_in


@fixture(scope="session")
def grid_shape_out(_case: cases.Case) -> tuple[int, ...]:
    return _case.grid_shape_out


@fixture(scope="session")
def num_channels_out(_case: cases.Case) -> int:
    return _case.num_channels_out


@fixture(scope="session")
def _neuralop_fno_block(
    _case: cases.Case,
) -> neuralop.layers.fno_block.FNOBlocks:
    torch.manual_seed(453879)

    return neuralop.layers.fno_block.FNOBlocks(
        in_channels=_case.num_channels_in,
        out_channels=_case.num_channels_hidden,
        n_modes=_case.num_modes,
        output_scaling_factor=None,
        n_layers=1,
        max_n_modes=None,
        fno_block_precision="full",
        use_mlp=False,
        stabilizer=None,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        separable=False,
        factorization=None,
    )


@fixture(scope="session")
def _neuralop_spectral_conv(
    _neuralop_fno_block: neuralop.layers.fno_block.FNOBlocks,
) -> neuralop.layers.spectral_convolution.SpectralConv:
    return _neuralop_fno_block.convs


@fixture(scope="session")
def _neuralop_skip_conv(
    _neuralop_fno_block: neuralop.layers.fno_block.FNOBlocks,
):
    return _neuralop_fno_block.fno_skips[0]


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
def W(_neuralop_skip_conv) -> jax.Array:
    W = _neuralop_skip_conv.weight.data

    W = jnp.asarray(W.detach().numpy())  # shape = (C_in, C_out, 1, ..., 1)
    W = W.reshape(W.shape[:2])  # shape = (C_in, C_out)

    return W


@fixture(scope="session")
def b(
    _neuralop_spectral_conv: neuralop.layers.spectral_convolution.SpectralConv,
) -> jax.Array:
    b = _neuralop_spectral_conv.bias[0, ...]

    b = jnp.asarray(b.detach().numpy())  # shape = (C_out, 1, ..., 1)
    b = b.reshape(b.shape[0])  # shape = (C_out,)

    return b


@fixture(scope="session")
def v_in(grid_shape_in: tuple[int, ...], num_channels_in: int) -> jax.Array:
    return jax.random.normal(
        jax.random.key(345786),
        shape=grid_shape_in + (num_channels_in,),
    )


@fixture(scope="session")
def v_in_torch(v_in: jax.Array) -> torch.Tensor:
    v_in_torch = torch.as_tensor(np.asarray(v_in))  # shape = (N_1, N_2, ..., N_D, C_in)
    v_in_torch = torch.moveaxis(v_in_torch, -1, 0)  # shape = (C_in, N_1, N_2, ..., N_D)
    v_in_torch = v_in_torch[None, ...]  # shape = (1, C_in, N_1, N_2, ..., N_D)

    return v_in_torch
