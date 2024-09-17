import jax
from jax import numpy as jnp
import numpy as np

import nola


def test_rfftn_2d():
    # Sample a random signal
    key = jax.random.key(985367)
    v = jax.random.normal(key, (16, 15, 3))

    # Transform and truncate
    num_modes = (3, 3)

    z_trunc = nola.models.fno.dft.rfftn(
        v,
        modes_shape=(2 * num_modes[0], num_modes[1]),
        axes=(0, 1),
    )

    # Reference implementation
    z_ref = jnp.fft.rfftn(v, axes=(0, 1), norm="forward")

    z_trunc_ref = jnp.concatenate(
        (z_ref[: num_modes[0], : num_modes[1]], z_ref[-num_modes[0] :, : num_modes[1]]),
        axis=0,
    )

    np.testing.assert_allclose(z_trunc, z_trunc_ref, rtol=1e-5, atol=1e-5)
