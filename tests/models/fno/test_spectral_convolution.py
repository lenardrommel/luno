import jax
import numpy as np

from nola.models.fno import spectral_convolution


def test_spectral_convolution(
    v_in: jax.Array,
    R: jax.Array,
    grid_shape_out: tuple[int, ...],
    v_out_ref: jax.Array,
):
    v_out, _ = spectral_convolution(v_in, R, grid_shape_out=grid_shape_out)

    np.testing.assert_allclose(
        v_out,
        v_out_ref,
        atol=1e-6,
        rtol=1e-6,
    )
