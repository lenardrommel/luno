import jax
from jax import numpy as jnp
import linox

from pytest_cases import fixture
from tests.utils import assert_samples_marginally_gaussian

import nola


@fixture
def parametric_gp() -> nola.randprocs.ParametricGaussianProcess:
    key, subkey = jax.random.split(jax.random.PRNGKey(10657))

    input_signal = jax.random.normal(subkey, (5, 2))
    z = nola.models.fno.dft.rfftn(input_signal, axes=(-2,), norm="forward")

    def feature_fns(x):
        return nola.jacobians.fno.SpectralConvolutionWeightJacobian(
            z,
            x.shape[:-1],
        )

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
    mean_xs, std_xs = parametric_gp.mean_and_std(xs)

    assert_samples_marginally_gaussian(samples_xs, mean_xs, std_xs)


def test_sample(parametric_gp: nola.randprocs.ParametricGaussianProcess):
    key = jax.random.PRNGKey(89756)
    xs = jnp.linspace(0.0, 1.0, 100)[:, None]

    samples_xs = parametric_gp.sample(key, xs, size=(1000,))

    mean_xs, std_xs = parametric_gp.mean_and_std(xs)

    assert_samples_marginally_gaussian(samples_xs, mean_xs, std_xs)
