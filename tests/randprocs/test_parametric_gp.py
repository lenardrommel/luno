import jax
from jax import numpy as jnp
import linox
import numpy as np

from pytest_cases import fixture

import nola


@fixture
def parametric_gp() -> nola.randprocs.ParametricGaussianProcess:
    key, subkey = jax.random.split(jax.random.PRNGKey(10657))

    input_signal = jax.random.normal(subkey, (5, 2))

    def feature_fns(x):
        return nola.linops.fno.FixedInputSpectralConvolution(
            input_signal,
            x.shape[:-1],
        )  # TODO: This is a bit of a hack...

    weight_dim = feature_fns(jnp.zeros((1, 1))).shape[-1]

    weight_mean = jax.random.normal(key, (weight_dim,))
    weight_cov = linox.Identity(weight_dim)

    return nola.randprocs.ParametricGaussianProcess.from_weights_and_features(
        weight_mean,
        weight_cov,
        feature_fns,
    )


def test_sample_fn(parametric_gp: nola.randprocs.ParametricGaussianProcess):
    key = jax.random.PRNGKey(34890)
    samples = parametric_gp.sample(key, size=(1000,))

    xs = jnp.linspace(0.0, 1.0, 100)[:, None]

    samples_xs = samples(xs)
    mean_xs = parametric_gp.mean(xs)
    std_xs = parametric_gp.std(xs)

    assert_samples_marginally_gaussian(samples_xs, mean_xs, std_xs)


def test_sample(parametric_gp: nola.randprocs.ParametricGaussianProcess):
    key = jax.random.PRNGKey(89756)
    xs = jnp.linspace(0.0, 1.0, 100)[:, None]

    samples_xs = parametric_gp.sample(key, xs, size=(1000,))

    mean_xs = parametric_gp.mean(xs)
    std_xs = parametric_gp.std(xs)

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
