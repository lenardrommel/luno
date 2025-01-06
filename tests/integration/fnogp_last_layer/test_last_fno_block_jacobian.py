from collections.abc import Callable

import jax
import linox
import numpy as np
from jax import numpy as jnp
from lugano.covariances.fno import CircularlySymmetricDiagonal
from lugano.jacobians.fno import LastFNOBlockWeightJacobian
from pytest_cases import fixture


@fixture(scope="session")
def jacobian(
    v_in: jax.Array,
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    output_grid_shape: tuple[int, ...],
    projection: Callable[[jax.Array], jax.Array],
    num_output_channels: int,
):
    return LastFNOBlockWeightJacobian(
        v_in=v_in,
        R=R,
        W=W,
        b=b,
        output_grid_shape=output_grid_shape,
        projection=projection,
        num_output_channels=num_output_channels,
    )


def test_transpose(jacobian: LastFNOBlockWeightJacobian):
    np.testing.assert_allclose(
        jacobian.T @ jnp.eye(jacobian.shape[0]),
        jnp.transpose(jacobian.todense()),
        rtol=1e-5,
        atol=1e-5,
    )


def test_linearized_pushforward_marginal_covariance(
    jacobian: LastFNOBlockWeightJacobian,
    weight_covariance: CircularlySymmetricDiagonal,
):
    JSigmaJT = linox.congruence_transform(jacobian, weight_covariance)

    np.testing.assert_allclose(
        linox.diagonal(JSigmaJT),
        jnp.diag(JSigmaJT.todense()),
        rtol=1e-5,
        atol=1e-5,
    )
