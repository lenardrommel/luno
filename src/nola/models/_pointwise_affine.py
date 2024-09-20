import jax
from jax import numpy as jnp

from jax.typing import ArrayLike


def pointwise_affine(v_in: ArrayLike, W: ArrayLike, b: ArrayLike) -> jax.Array:
    """Applies an affine function pointwise to the discretized input function."""

    v_in = jnp.asarray(v_in)
    W = jnp.asarray(W)
    b = jnp.asarray(b)

    return (W @ v_in[..., None])[..., 0] + b
