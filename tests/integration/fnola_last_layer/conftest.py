import jax
from jax import numpy as jnp

import pytest
from pytest_cases import AUTO, fixture, parametrize, parametrize_with_cases
from tests.models.fno.fno_block.cases import FNOBlockCase

from collections.abc import Callable


@fixture(scope="session")
@parametrize_with_cases(
    "_case",
    cases="tests.models.fno.fno_block.cases",
    scope="session",
)
def _fno_block_case(_case: FNOBlockCase) -> FNOBlockCase:
    return _case


@fixture(scope="session")
def input_grid_shape(_fno_block_case: FNOBlockCase) -> tuple[int, ...]:
    return _fno_block_case.grid_shape_in


@fixture(scope="session")
def num_input_channels(_fno_block_case: FNOBlockCase) -> int:
    return _fno_block_case.num_channels_in


@fixture(scope="session")
def num_modes(_fno_block_case: FNOBlockCase) -> tuple[int, ...]:
    return _fno_block_case.num_modes


@fixture(scope="session")
def num_hidden_channels(_fno_block_case: FNOBlockCase) -> tuple[int, ...]:
    return _fno_block_case.num_channels_out


@fixture(scope="session")
def output_grid_shape(_fno_block_case: FNOBlockCase) -> tuple[int, ...]:
    return _fno_block_case.grid_shape_out


@fixture(scope="session")
@parametrize("num_output_channels", [None, 1, 2], idgen=AUTO)
def num_output_channels(num_output_channels: int, num_hidden_channels: int) -> int:
    return (
        num_output_channels if num_output_channels is not None else num_hidden_channels
    )


@fixture(scope="session")
def v_in(input_grid_shape: tuple[int, ...], num_input_channels: int) -> jax.Array:
    return jax.random.normal(
        jax.random.key(234567 + sum(input_grid_shape) + num_input_channels),
        shape=input_grid_shape + (num_input_channels,),
    )


@fixture(scope="session")
def R(
    num_modes: tuple[int, ...],
    num_input_channels: int,
    num_hidden_channels: int,
) -> jax.Array:
    modes_shape = num_modes[:-1] + (num_modes[-1] // 2 + 1,)

    R_real_imag = jax.random.normal(
        jax.random.key(
            6379975 + sum(num_modes) + num_input_channels + num_hidden_channels
        ),
        shape=(2,) + modes_shape + (num_hidden_channels, num_input_channels),
    )

    return R_real_imag[0] + R_real_imag[1] * 1j


@fixture(scope="session")
def W(num_input_channels: int, num_hidden_channels: int) -> jax.Array:
    return jax.random.normal(
        jax.random.key(28467 + num_input_channels + num_hidden_channels),
        shape=(num_hidden_channels, num_input_channels),
    )


@fixture(scope="session")
def b(num_hidden_channels: int) -> jax.Array:
    return jax.random.normal(
        jax.random.key(987264 + num_hidden_channels),
        shape=(num_hidden_channels,),
    )


def case_projection_mlp(
    num_hidden_channels: int,
    num_output_channels: int,
) -> Callable[[jax.Array], jax.Array]:
    key = jax.random.key(3245)

    key, subkey = jax.random.split(key)
    W1 = jax.random.normal(key, shape=(num_hidden_channels, num_hidden_channels))
    W2 = jax.random.normal(subkey, shape=(num_output_channels, num_hidden_channels))

    def Q(x: jax.Array) -> jax.Array:
        return (W2 @ jnp.tanh(W1 @ x[..., None]))[..., 0]

    return Q


def case_projection_identity(
    num_hidden_channels: int,
    num_output_channels: int,
) -> Callable[[jax.Array], jax.Array]:
    pytest.mark.skipif(
        num_output_channels != num_hidden_channels,
        reason=(
            "Identity projection only makes sense if num_output_channels == "
            "num_hidden_channels."
        ),
    )

    return lambda x: x


@fixture(scope="session")
@parametrize_with_cases("Q", cases=".", prefix="case_projection_", scope="session")
def projection(Q: Callable[[jax.Array], jax.Array]) -> Callable[[jax.Array], jax.Array]:
    return Q
