__all__ = ["CircularlySymmetricDiagonal"]

import jax
from jax import numpy as jnp
from linox import Diagonal

from jax.typing import ArrayLike


class CircularlySymmetricDiagonal(Diagonal):
    def __init__(self, R_real: ArrayLike, W: ArrayLike, b: ArrayLike | None) -> None:
        self._R_real = jnp.asarray(R_real)
        self._W = jnp.asarray(W)
        self._b = jnp.asarray(b) if b is not None else None

        diag = (
            self._R_real.reshape(-1, order="C"),
            self._R_real.reshape(-1, order="C"),
            self._W.reshape(-1, order="C"),
        )

        if self._b is not None:
            diag += (self._b.reshape(-1, order="C"),)

        super().__init__(jnp.concatenate(diag, axis=0))

    @property
    def R_real(self) -> jax.Array:
        return self._R_real

    @property
    def W(self) -> jax.Array:
        return self._W

    @property
    def b(self) -> jax.Array | None:
        return self._b
