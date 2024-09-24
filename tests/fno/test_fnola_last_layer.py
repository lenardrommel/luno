import jax
from jax import numpy as jnp
import linox
import numpy as np

from pytest_cases import fixture

import nola

from collections.abc import Callable


@fixture(scope="session")
def prior_prec() -> float:
    return 0.42


@fixture(scope="session")
def rank() -> int:
    return 10


@fixture(scope="session")
def weight_cov(
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    prior_prec: float,
    rank: int,
) -> linox.IsotropicScalingPlusSymmetricLowRank:
    key = jax.random.key(65789)
    U, S, _ = jnp.linalg.svd(
        jax.random.normal(key, shape=(2 * R.size + W.size + b.size, rank)),
        full_matrices=False,
    )

    return linox.IsotropicScalingPlusSymmetricLowRank(prior_prec, U, S)


@fixture(scope="session")
def projection(num_channels_out: int) -> Callable[[jax.Array], jax.Array]:
    key = jax.random.key(3245)

    key, subkey = jax.random.split(key)
    W1 = jax.random.normal(key, shape=(num_channels_out, num_channels_out))
    W2 = jax.random.normal(subkey, shape=(2, num_channels_out))

    def Q(x: jax.Array) -> jax.Array:
        return (W2 @ jnp.tanh(W1 @ x[..., None]))[..., 0]

    return Q


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
    grid_shape_out: tuple[int, ...],
):
    parametric_gp = fnola_last_layer(v_in)

    xs = nola.models.fno.FFTGrid(grid_shape_out)

    key = jax.random.key(34890)
    samples_xs = parametric_gp.sample(key, xs, size=(2000,))

    mean_xs, std_xs = parametric_gp.mean_and_std(xs)

    assert_samples_marginally_gaussian(samples_xs, mean_xs, std_xs)


def assert_samples_marginally_gaussian(
    samples: jax.Array,
    mean: jax.Array,
    std: jax.Array,
    axis: int = 0,
):
    samples = jnp.sort(samples, axis=axis)

    samples_standardized = (
        samples - jnp.expand_dims(mean, axis=axis)
    ) / jnp.expand_dims(std, axis=axis)

    # Map standardized samples through standard normal cdf and compare to uniform cdf
    samples_norm_cdf = jax.scipy.stats.norm.cdf(samples_standardized)
    uniform_cdf = jnp.linspace(0.0, 1.0, samples_norm_cdf.shape[axis])

    np.testing.assert_allclose(
        np.moveaxis(samples_norm_cdf, axis, -1) - uniform_cdf,
        0.0,
        atol=6e-2,
    )
