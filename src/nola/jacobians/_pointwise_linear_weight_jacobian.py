import jax
from jax import numpy as jnp
import linox

from jax.typing import ArrayLike


# TODO: This is a Kronecker product v (x) I_{C_{out}}
class PointwiseLinearWeightJacobian(linox.LinearOperator):
    r"""Jacobian of a pointwise linear layer (a 1x1 convolution) with respect to its
    weight matrix :math:`W`.

    Since the layer is linear in its weights, the Jacobian only depends on the layer
    input :math:`v`.

    This linear operator implements the operation

    .. math::
        vec(W) \mapsto vec((W v_{n:})_{n = 1}^N).

    Parameters
    ----------
    v :
        Input :math:`v` with shape :math:`(N_1, \ldots, N_D, C)`.
    """

    def __init__(self, v: ArrayLike, num_output_channels: int | None = None) -> None:
        if jnp.ndim(v) < 2:
            raise ValueError("`v` must have at least 2 dimensions.")

        self._v = jnp.asarray(v)
        self._C_in = self._v.shape[-1]

        self._v_flat = self._v.reshape(-1, self._C_in)  # shape: (N, C_in)
        self._N = self._v_flat.shape[0]

        self._C_out = (
            num_output_channels if num_output_channels is not None else self._C_in
        )

        super().__init__(
            shape=(self._N * self._C_out, self._C_out * self._C_in),
            dtype=self._v.dtype,
        )

    def _matmul(self, W: jax.Array) -> jax.Array:
        batch_shape = W.shape[:-2]
        ncols = W.shape[-1]

        W = W.reshape(*batch_shape, self._C_out, self._C_in, ncols)

        v_out = jnp.einsum("...ijk,...nj->...nik", W, self._v)

        return v_out.reshape(*batch_shape, self._N * self._C_out, ncols)
