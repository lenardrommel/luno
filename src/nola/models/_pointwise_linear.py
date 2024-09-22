import jax
from jax import numpy as jnp

from jax.typing import ArrayLike


def pointwise_linear(v_in: ArrayLike, W: ArrayLike) -> jax.Array:
    """Applies a pointwise linear transformation to the discretized input function."""

    v_in = jnp.asarray(v_in)
    W = jnp.asarray(W)

    return (W @ v_in[..., None])[..., 0]
