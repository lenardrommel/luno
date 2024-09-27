import jax
import numpy as np
import torch

from pytest_cases import fixture, parametrize_with_cases, unpack_fixture

import nola

from .cases import FNOBlockCase


@fixture(scope="session")
@parametrize_with_cases("params", cases=".cases", scope="session")
def _case(params: FNOBlockCase) -> FNOBlockCase:
    return params


@fixture(scope="session")
def input_grid_shape(_case: FNOBlockCase) -> tuple[int, ...]:
    return _case.input_grid_shape


@fixture(scope="session")
def num_input_channels(_case: FNOBlockCase) -> int:
    return _case.num_input_channels


@fixture(scope="session")
def output_grid_shape(_case: FNOBlockCase) -> tuple[int, ...] | None:
    return _case.output_grid_shape


@fixture(scope="session")
def num_output_channels(_case: FNOBlockCase) -> int:
    return _case.num_output_channels


@fixture(scope="session")
def weights(
    _case: FNOBlockCase,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    key = jax.random.key(
        789645
        + sum(_case.input_grid_shape)
        + _case.num_input_channels
        + sum(_case.modes_shape)
        + _case.num_output_channels
        + (453789 if _case.output_grid_shape is None else sum(_case.output_grid_shape))
    )

    key, subkey = jax.random.split(key)
    R_re_im = jax.random.normal(
        subkey,
        shape=(2,)
        + _case.modes_shape
        + (_case.num_output_channels, _case.num_input_channels),
    )
    R = R_re_im[0] + 1j * R_re_im[1]

    key, subkey = jax.random.split(key)
    W = jax.random.normal(
        subkey,
        shape=(_case.num_output_channels, _case.num_input_channels),
    )

    key, subkey = jax.random.split(key)
    b = jax.random.normal(
        subkey,
        shape=(_case.num_output_channels,),
    )

    return R, W, b


R, W, b = unpack_fixture("R, W, b", weights)


@fixture(scope="session")
def v_in(input_grid_shape: tuple[int, ...], num_input_channels: int) -> jax.Array:
    return jax.random.normal(
        jax.random.key(345786),
        shape=input_grid_shape + (num_input_channels,),
    )


@fixture(scope="session")
def v_in_torch(v_in: jax.Array) -> torch.Tensor:
    v_in_torch = torch.as_tensor(np.asarray(v_in))  # shape = (N_1, N_2, ..., N_D, C_in)
    v_in_torch = torch.moveaxis(v_in_torch, -1, 0)  # shape = (C_in, N_1, N_2, ..., N_D)
    v_in_torch = v_in_torch[None, ...]  # shape = (1, C_in, N_1, N_2, ..., N_D)

    return v_in_torch


@fixture(scope="session")
def _fno_block_out(
    v_in: jax.Array,
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    output_grid_shape: tuple[int, ...],
):
    return nola.models.fno.fno_block(
        v_in,
        R,
        W,
        b,
        output_grid_shape=output_grid_shape,
    )


@fixture(scope="session")
def v_out(_fno_block_out) -> jax.Array:
    return _fno_block_out[0]


@fixture(scope="session")
def v_out_sconv(_fno_block_out) -> jax.Array:
    return _fno_block_out[1]["v_out_sconv"]
