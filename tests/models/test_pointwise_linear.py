import jax
from jax import numpy as jnp
import numpy as np

import nola


def test_identity_weight_matrix_reproduces_input():
    N = 100
    C = 10

    key = jax.random.key(345678)

    key, subkey = jax.random.split(key)
    v_in = jax.random.normal(subkey, (N, C))

    linop = nola.models.FixedInputPointwiseLinearTransform(v_in)

    W = jnp.eye(C).reshape(-1)

    np.testing.assert_allclose((linop @ W).reshape(N, C), v_in)
