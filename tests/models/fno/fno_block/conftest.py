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
def grid_shape_in(_case: FNOBlockCase) -> tuple[int, ...]:
    return _case.grid_shape_in


@fixture(scope="session")
def num_channels_in(_case: FNOBlockCase) -> int:
    return _case.num_channels_in


@fixture(scope="session")
def grid_shape_out(_case: FNOBlockCase) -> tuple[int, ...] | None:
    return _case.grid_shape_out


@fixture(scope="session")
def num_channels_out(_case: FNOBlockCase) -> int:
    return _case.num_channels_out


@fixture(scope="session")
def weights(
    _case: FNOBlockCase,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    key = jax.random.key(
        789645
        + sum(_case.grid_shape_in)
        + _case.num_channels_in
        + sum(_case.num_modes)
        + _case.num_channels_out
        + (453789 if _case.grid_shape_out is None else sum(_case.grid_shape_out))
    )

    key, subkey = jax.random.split(key)
    R_re_im = jax.random.normal(
        subkey,
        shape=(2,) + _case.num_modes + (_case.num_channels_out, _case.num_channels_in),
    )
    R = R_re_im[0] + 1j * R_re_im[1]

    key, subkey = jax.random.split(key)
    W = jax.random.normal(
        subkey,
        shape=(_case.num_channels_out, _case.num_channels_in),
    )

    key, subkey = jax.random.split(key)
    b = jax.random.normal(
        subkey,
        shape=(_case.num_channels_out,),
    )

    return R, W, b


R, W, b = unpack_fixture("R, W, b", weights)


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


@fixture(scope="session")
def _fno_block_out(
    v_in: jax.Array,
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    grid_shape_out: tuple[int, ...],
):
    return nola.models.fno.fno_block(v_in, R, W, b, output_grid_shape=grid_shape_out)


@fixture(scope="session")
def v_out(_fno_block_out) -> jax.Array:
    return _fno_block_out[0]


@fixture(scope="session")
def v_out_sconv(_fno_block_out) -> jax.Array:
    return _fno_block_out[1]["v_out_sconv"]
