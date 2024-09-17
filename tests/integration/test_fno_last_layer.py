from dataclasses import InitVar, dataclass, field
import functools
import operator

import jax
from jax import numpy as jnp
import linox

from pytest_cases import fixture, parametrize

import nola


@dataclass
class Case:
    key: InitVar[jax.Array]

    num_channels: int
    spatial_grid_shape: tuple[int, ...]

    num_outputs: int

    prior_prec: InitVar[float]
    rank: InitVar[int]

    mean: jax.Array = field(init=False)
    prec: linox.IsotropicScalingPlusLowRank = field(init=False)
    cov: linox.IsotropicScalingPlusLowRank = field(init=False)

    @functools.cached_property
    def modes_shape(self) -> tuple[int, ...]:
        dummy_signal = jnp.zeros(self.spatial_grid_shape)
        return jnp.fft.rfftn(
            dummy_signal,
            axes=tuple(range(len(self.spatial_grid_shape))),
        ).shape

    @functools.cached_property
    def R(self) -> jax.Array:
        R_real = self.mean[: -self.W.size].reshape(
            2,
            self.num_channels,
            *self.modes_shape,
            self.num_channels,
        )

        return R_real[0, ...] + 1j * R_real[1, ...]

    @functools.cached_property
    def W(self) -> jax.Array:
        return self.mean[-self.num_channels * self.num_channels :].reshape(
            self.num_channels,
            self.num_channels,
        )

    def __post_init__(self, key: jax.Array, prior_prec: float, rank: int) -> None:
        _R_size = (
            2
            * self.num_channels
            * functools.reduce(operator.mul, self.modes_shape, 1)
            * self.num_channels
        )
        _W_size = self.num_channels * self.num_channels

        # Sample random mean
        key, subkey = jax.random.split(key)
        self.mean = jax.random.normal(subkey, shape=(_R_size + _W_size,))

        # Sample random precision updates
        key, subkey = jax.random.split(key)
        prec_dd = jax.random.normal(subkey, shape=(_R_size + _W_size, rank))
        prec_dd_U, prec_dd_S, _ = jnp.linalg.svd(prec_dd)
        self.prec = linox.IsotropicScalingPlusLowRank(
            prior_prec,
            prec_dd_U,
            prec_dd_S,
        )

        # Compute corresponding covariance matrix
        self.cov = linox.linverse(self.prec)


@fixture(scope="module")
@parametrize(
    "case",
    (
        Case(
            key=jax.random.key(425786),
            num_channels=2,
            spatial_grid_shape=(10,),
            num_outputs=1,
            prior_prec=1.0,
            rank=10,
        ),
    ),
)
def case(case: Case) -> Case:
    return case
