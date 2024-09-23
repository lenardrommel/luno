import jax
import numpy as np

from nola.models.fno import spectral_convolution


def test_spectral_convolution(
    v_in: jax.Array,
    z_in: jax.Array,
    R: jax.Array,
    grid_shape_out: tuple[int, ...],
    v_out_sconv_ref: jax.Array,
):
    v_out_sconv, intermediates = spectral_convolution(
        v_in,
        R,
        output_grid_shape=grid_shape_out,
    )

    np.testing.assert_allclose(
        v_out_sconv,
        v_out_sconv_ref,
        atol=1e-6,
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        intermediates["z_in"],
        z_in,
        atol=1e-6,
        rtol=1e-6,
    )
