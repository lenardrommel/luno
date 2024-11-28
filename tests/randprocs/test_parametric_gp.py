from collections.abc import Callable

import jax
import linox
import lugano
from jax import numpy as jnp
from jax.typing import ArrayLike
from pytest_cases import fixture

from tests.utils import assert_samples_marginally_gaussian


@fixture(scope="module")
def num_features() -> int:
    return 5


@fixture(scope="module")
def features(num_features: int) -> Callable[[ArrayLike], jax.Array]:
    def features(x):
        return linox.Matrix(jnp.asarray(x) ** jnp.arange(num_features))

    return features


@fixture(scope="module")
def parametric_gp(
    num_features: int,
    features: Callable[[ArrayLike], jax.Array],
) -> lugano.randprocs.ParametricGaussianProcess:
    weight_mean = jax.random.normal(
        jax.random.key(10657),
        shape=(num_features,),
    )
    weight_cov = linox.Identity(num_features)

    return lugano.randprocs.ParametricGaussianProcess.from_weights_and_features(
        weight_mean,
        weight_cov,
        features,
    )


def test_sample_fn(parametric_gp: lugano.randprocs.ParametricGaussianProcess):
    key = jax.random.PRNGKey(34890)
    samples = parametric_gp.sample(key, size=(1000,))

    xs = jnp.linspace(0.0, 1.0, 100)[:, None]

    samples_xs = samples(xs)
    mean_xs, std_xs = parametric_gp.mean_and_std(xs)

    assert_samples_marginally_gaussian(samples_xs, mean_xs, std_xs)


def test_sample(parametric_gp: lugano.randprocs.ParametricGaussianProcess):
    key = jax.random.PRNGKey(89756)
    xs = jnp.linspace(0.0, 1.0, 100)[:, None]

    samples_xs = parametric_gp.sample(key, xs, size=(1000,))

    mean_xs, std_xs = parametric_gp.mean_and_std(xs)

    assert_samples_marginally_gaussian(samples_xs, mean_xs, std_xs)
