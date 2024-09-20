import jax
from jax import numpy as jnp
import numpy as np

import nola


def test_matmul():
    key = jax.random.key(234587)

    def f(x):
        return x[1:4] ** 2

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (20, 5))

    Df_x = jax.vmap(jax.jacfwd(f), in_axes=0, out_axes=0)(x)

    linop = nola.models.PointwiseJVP(f, (3, 5), x)

    key, subkey = jax.random.split(key)
    v = jax.random.normal(subkey, (4, 20, 5, 8))

    v_flat = v.reshape(4, 20 * 5, 8)

    np.testing.assert_allclose(
        jnp.reshape(linop @ v_flat, (4, 20, 3, 8)),
        Df_x @ v,
    )
