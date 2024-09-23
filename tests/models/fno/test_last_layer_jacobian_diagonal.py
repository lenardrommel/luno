import jax
from jax import numpy as jnp
import linox
import numpy as np

from pytest_cases import parametrize_with_cases

from nola.jacobians.fno import LastFNOBlockWeightJacobian


def case_identity_projection(
    v_in: jax.Array,
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    grid_shape_out: tuple[int, ...],
) -> LastFNOBlockWeightJacobian:
    return LastFNOBlockWeightJacobian(
        v_in=v_in,
        R_0=R,
        W_0=W,
        b_0=b,
        output_grid_shape=grid_shape_out,
        projection=lambda x: x,
        num_output_channels=W.shape[1],
    )


def case_quadratic_projection(
    v_in: jax.Array,
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    grid_shape_out: tuple[int, ...],
) -> LastFNOBlockWeightJacobian:
    return LastFNOBlockWeightJacobian(
        v_in=v_in,
        R_0=R,
        W_0=W,
        b_0=b,
        output_grid_shape=grid_shape_out,
        projection=lambda x: jnp.sum(x**2, axis=-1, keepdims=True),
        num_output_channels=1,
    )


@parametrize_with_cases("linop", cases=".")
def test_outer_product_diagonal(linop: LastFNOBlockWeightJacobian):
    AAT = linox.congruence_transform(linop, linox.Identity(linop.shape[1]))

    np.testing.assert_allclose(
        linox.diagonal(AAT),
        jnp.diag(AAT.todense()),
        rtol=1e-6,
        atol=1e-6,
    )
