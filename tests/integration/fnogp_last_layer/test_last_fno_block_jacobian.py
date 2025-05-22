from collections.abc import Callable

import jax
import linox
import numpy as np
from jax import numpy as jnp
from pytest_cases import fixture

from luno.covariances.fno import CircularlySymmetricDiagonal
from luno.jacobians.fno import LastFNOBlockWeightJacobian

ATOL = 1e-5
RTOL = 1e-5

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
        rtol=RTOL,
        atol=ATOL,
    )


def test_linearized_pushforward_marginal_covariance_compare_isotropic_circularly_symmetric(
    jacobian: LastFNOBlockWeightJacobian,
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    random_symmetric_low_rank: linox.SymmetricLowRank,
):
    var = 0.42

    isotropic_weight_precision = linox.IsotropicScalingPlusSymmetricLowRank(
        var,
        random_symmetric_low_rank.U,
        random_symmetric_low_rank.S,
    )
    circularly_symmetric_weight_precision = linox.PositiveDiagonalPlusSymmetricLowRank(
        CircularlySymmetricDiagonal(
            jnp.full_like(R.real, var),
            jnp.full_like(W, var),
            jnp.full_like(b, var),
        ),
        random_symmetric_low_rank,
    )

    isotropic_weight_covariance = linox.linverse(isotropic_weight_precision)
    circularly_symmetric_weight_covariance = linox.linverse(
        circularly_symmetric_weight_precision
    )

    JSigmaJT_isotropic = linox.congruence_transform(
        jacobian, isotropic_weight_covariance
    )
    JSigmaJT_circularly_symmetric = linox.congruence_transform(
        jacobian, circularly_symmetric_weight_covariance
    )

    np.testing.assert_allclose(
        linox.diagonal(JSigmaJT_isotropic),
        linox.diagonal(JSigmaJT_circularly_symmetric),
        rtol=RTOL,
        atol=ATOL,
    )
