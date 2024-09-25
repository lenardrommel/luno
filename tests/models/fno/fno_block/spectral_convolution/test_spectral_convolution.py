import jax
import numpy as np


def test_spectral_convolution(
    v_out_sconv: jax.Array,
    v_out_sconv_ref: jax.Array,
):
    np.testing.assert_allclose(
        v_out_sconv,
        v_out_sconv_ref,
        atol=1e-6,
        rtol=1e-6,
    )
