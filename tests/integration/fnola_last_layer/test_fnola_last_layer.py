import jax
from jax import numpy as jnp
import linox

from pytest_cases import fixture
from tests.utils import assert_samples_marginally_gaussian

import nola

from collections.abc import Callable


@fixture(scope="session")
def prior_var() -> float:
    return 0.42


@fixture(scope="session")
def rank() -> int:
    return 10


@fixture(scope="session")
def weight_cov(
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    prior_var: float,
    rank: int,
) -> linox.IsotropicScalingPlusSymmetricLowRank:
    key = jax.random.key(65789)
    U, S, _ = jnp.linalg.svd(
        jax.random.normal(key, shape=(2 * R.size + W.size + b.size, rank)),
        full_matrices=False,
    )

    return linox.IsotropicScalingPlusSymmetricLowRank(prior_var, U, S)


@fixture(scope="session")
def fnola_last_layer(
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    weight_cov: linox.IsotropicScalingPlusSymmetricLowRank,
    projection: Callable[[jax.Array], jax.Array],
) -> nola.FNOLALastLayer:
    return nola.FNOLALastLayer(
        fno_head=lambda x: x,
        R=R,
        W=W,
        b=b,
        weight_cov=weight_cov,
        projection=projection,
        num_output_channels=2,
    )


def test_sample(
    fnola_last_layer: nola.FNOLALastLayer,
    v_in: jax.Array,
    output_grid_shape: tuple[int, ...],
):
    parametric_gp = fnola_last_layer(v_in)

    xs = nola.models.fno.FFTGrid(output_grid_shape)

    key = jax.random.key(34890)
    samples_xs = parametric_gp.sample(key, xs, size=(2000,))

    mean_xs, std_xs = parametric_gp.mean_and_std(xs)

    assert_samples_marginally_gaussian(samples_xs, mean_xs, std_xs)
