import jax
from jax import numpy as jnp
import linox
import numpy as np

from pytest_cases import fixture

from nola.jacobians.fno import LastFNOBlockWeightJacobian

from collections.abc import Callable


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


def test_outer_product_diagonal(jacobian: LastFNOBlockWeightJacobian):
    JJT = linox.congruence_transform(jacobian, linox.Identity(jacobian.shape[1]))

    np.testing.assert_allclose(
        linox.diagonal(JJT),
        jnp.diag(JJT.todense()),
        rtol=1e-6,
        atol=1e-6,
    )
