from collections.abc import Callable

import jax
import linox
import lugano
from pytest_cases import fixture

from tests.utils import assert_samples_marginally_gaussian


@fixture(scope="session")
def fnogp_last_layer(
    R: jax.Array,
    W: jax.Array,
    b: jax.Array,
    weight_covariance: linox.IsotropicScalingPlusSymmetricLowRank,
    projection: Callable[[jax.Array], jax.Array],
    num_output_channels: int,
) -> lugano.FNOGPLastLayer:
    return lugano.FNOGPLastLayer(
        fno_head=lambda x: x,
        R=R,
        W=W,
        b=b,
        weight_cov=weight_covariance,
        projection=projection,
        num_output_channels=num_output_channels,
    )


def test_sample(
    fnogp_last_layer: lugano.FNOGPLastLayer,
    v_in: jax.Array,
    input_grid_shape: tuple[int, ...],
    output_grid_shape: tuple[int, ...] | None,
):
    parametric_gp = fnogp_last_layer(v_in)

    xs = lugano.models.fno.FFTGrid(
        output_grid_shape if output_grid_shape is not None else input_grid_shape
    )

    key = jax.random.key(34890)
    samples_xs = parametric_gp.sample(key, xs, size=(2000,))

    mean_xs, std_xs = parametric_gp.mean_and_std(xs)

    assert_samples_marginally_gaussian(samples_xs, mean_xs, std_xs)
