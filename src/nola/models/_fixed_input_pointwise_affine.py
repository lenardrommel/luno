import functools
import operator

import jax
from jax import numpy as jnp
import linox

from jax.typing import ArrayLike


class FixedInputPointwiseAffineTransform(linox.LinearOperator):
    r"""Linear operator representation of the skip connection in a neural operator layer
    as a function of the weight matrix :math:`W` and bias vector :math:`b` with fixed
    input :math:`v_\text{in}`.

    In one spatial dimension, this linear operator implements the operation

    .. math::
        (W, b) \mapsto (W (v_\text{in})_{n:} + b)_{n = 1}^N.

    The generalization to multiple spatial dimensions is analogous.

    Parameters
    ----------
    v_in :
        Input :math:`v_\text{in}` to the neural operator layer with shape
        :math:`(N_1, \ldots, N_D, C)`.
    """

    def __init__(self, v_in: ArrayLike) -> None:
        if jnp.ndim(v_in) < 2:
            raise ValueError("`v_in` must have at least 2 dimensions.")

        self._v_in = jnp.asarray(v_in)

        self._grid_shape = self._v_in.shape[:-1]
        self._C = self._v_in.shape[-1]

        self._N = functools.reduce(operator.mul, self._grid_shape, 1)

        super().__init__(
            shape=(self._N * self._C, self._C * self._C + self._C),
            dtype=self._v_in.dtype,
        )

    def _matmul(self, Wb: jax.Array) -> jax.Array:
        batch_shape = Wb.shape[:-2]
        ncols = Wb.shape[-1]

        W = Wb[..., : -self._C, :].reshape(*batch_shape, self._C, self._C, ncols)
        b = Wb[..., -self._C :, :].reshape(*batch_shape, self._C, ncols)

        v_out = jnp.einsum("...ijk,...j->...ik", W, self._v_in) + b

        return v_out.reshape(*batch_shape, self._N * self._C, ncols)
