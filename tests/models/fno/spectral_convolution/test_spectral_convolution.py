import jax
import numpy as np

from nola.models.fno import spectral_convolution


def test_spectral_convolution(
    v_in: jax.Array,
    R: jax.Array,
    v_out_ref: jax.Array,
):
    v_out = spectral_convolution(v_in, R)

    np.testing.assert_allclose(
        v_out,
        v_out_ref,
        atol=1e-6,
        rtol=1e-6,
    )
