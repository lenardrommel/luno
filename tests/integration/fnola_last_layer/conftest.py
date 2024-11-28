from collections.abc import Callable

import jax
import linox
import lugano
from jax import numpy as jnp
from pytest_cases import (
    AUTO,
    fixture,
    parametrize,
    parametrize_with_cases,
    unpack_fixture,
)

from tests.models.fno.fno_block.cases import FNOBlockCase


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
    return _fno_block_case.input_grid_shape


@fixture(scope="session")
def num_input_channels(_fno_block_case: FNOBlockCase) -> int:
    return _fno_block_case.num_input_channels


@fixture(scope="session")
def modes_shape(_fno_block_case: FNOBlockCase) -> tuple[int, ...]:
    return _fno_block_case.modes_shape


@fixture(scope="session")
def num_hidden_channels(_fno_block_case: FNOBlockCase) -> tuple[int, ...]:
    return _fno_block_case.num_output_channels


@fixture(scope="session")
def output_grid_shape(_fno_block_case: FNOBlockCase) -> tuple[int, ...]:
    return _fno_block_case.output_grid_shape


@fixture(scope="session")
def v_in(input_grid_shape: tuple[int, ...], num_input_channels: int) -> jax.Array:
    return jax.random.normal(
        jax.random.key(234567 + sum(input_grid_shape) + num_input_channels),
        shape=input_grid_shape + (num_input_channels,),
    )


@fixture(scope="session")
def R(
    modes_shape: tuple[int, ...],
    num_input_channels: int,
    num_hidden_channels: int,
) -> jax.Array:
    R_real_imag = jax.random.normal(
        jax.random.key(
            6379975 + sum(modes_shape) + num_input_channels + num_hidden_channels
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


def case_weight_covariance_scaled_identity(
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
) -> linox.LinearOperator:
    var = 0.42

    return var * linox.Identity(2 * R.size + W.size + b.size)


@fixture(scope="session")
def _random_circularly_symmetric_diagonal(
    modes_shape: tuple[int, ...],
    num_input_channels: int,
    num_hidden_channels: int,
) -> lugano.covariances.fno.CircularlySymmetricDiagonal:
    key = jax.random.key(
        36789 + sum(modes_shape) + num_input_channels + num_hidden_channels
    )

    key, subkey = jax.random.split(key)
    R_real = jax.random.gamma(
        key,
        1.0,
        shape=modes_shape + (num_hidden_channels, num_input_channels),
    )

    key, subkey = jax.random.split(subkey)
    W = jax.random.gamma(key, 1.0, shape=(num_hidden_channels, num_input_channels))

    key, subkey = jax.random.split(subkey)
    b = jax.random.gamma(subkey, 1.0, shape=(num_hidden_channels,))

    return lugano.covariances.fno.CircularlySymmetricDiagonal(R_real, W, b)


def case_weight_covariance_circularly_symmetric_diagonal(
    _random_circularly_symmetric_diagonal: lugano.covariances.fno.CircularlySymmetricDiagonal,
) -> lugano.covariances.fno.CircularlySymmetricDiagonal:
    return _random_circularly_symmetric_diagonal


@parametrize("rank", (10,), idgen=AUTO)
def case_weight_covariance_diagonal_plus_low_rank(
    _random_circularly_symmetric_diagonal: lugano.covariances.fno.CircularlySymmetricDiagonal,
    rank: int,
) -> linox.PositiveDiagonalPlusSymmetricLowRank:
    key = jax.random.key(65789 + _random_circularly_symmetric_diagonal.shape[0])
    U, S, _ = jnp.linalg.svd(
        jax.random.normal(
            key,
            shape=(_random_circularly_symmetric_diagonal.shape[0], rank),
        ),
        full_matrices=False,
    )

    return linox.PositiveDiagonalPlusSymmetricLowRank(
        _random_circularly_symmetric_diagonal,
        linox.SymmetricLowRank(U, S**2),
    )


@fixture(scope="session")
@parametrize_with_cases(
    "weight_covariance",
    cases=".",
    prefix="case_weight_covariance_",
    scope="session",
)
def weight_covariance(weight_covariance: linox.LinearOperator) -> linox.LinearOperator:
    return weight_covariance


@parametrize("num_output_channels", [None, 1, 2], idgen=AUTO)
def case_projection_mlp(
    num_hidden_channels: int,
    num_output_channels: int | None,
) -> Callable[[jax.Array], jax.Array]:
    if num_output_channels is None:
        num_output_channels = num_hidden_channels

    key, subkey = jax.random.split(jax.random.key(3245))
    W1 = jax.random.normal(key, shape=(num_hidden_channels, num_hidden_channels))
    W2 = jax.random.normal(subkey, shape=(num_output_channels, num_hidden_channels))

    def Q(x: jax.Array) -> jax.Array:
        return (W2 @ jnp.tanh(W1 @ x[..., None]))[..., 0]

    return Q, num_output_channels


def case_projection_identity(
    num_hidden_channels: int,
) -> tuple[Callable[[jax.Array], jax.Array], int]:
    return lambda x: x, num_hidden_channels


@fixture(scope="session")
@parametrize_with_cases(
    "projection_and_num_output_channels",
    cases=".",
    prefix="case_projection_",
    scope="session",
)
def _projection_and_num_output_channels(
    projection_and_num_output_channels: tuple[Callable[[jax.Array], jax.Array], int],
) -> tuple[Callable[[jax.Array], jax.Array], int]:
    return projection_and_num_output_channels


projection, num_output_channels = unpack_fixture(
    "projection,num_output_channels",
    _projection_and_num_output_channels,
)
