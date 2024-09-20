import jax
from linox._arithmetic import ProductLinearOperator

from collections.abc import Callable
from jax.typing import ArrayLike

from .._pointwise_jvp import PointwiseJVP
from ._fixed_input_fno_block import FixedInputFNOBlock


class FixedInputFNOHead(ProductLinearOperator):
    def __init__(
        self,
        v_in: ArrayLike,
        z_in: ArrayLike,
        v_out: ArrayLike,
        projection: Callable[[jax.Array], jax.Array],
        num_output_channels: int,
    ) -> None:
        self._fno_block = FixedInputFNOBlock(
            v_in,
            z_in,
            output_grid_shape=v_out.shape[:-1],
            num_output_channels=v_out.shape[-1],
        )

        self._projection_jacobian = PointwiseJVP(
            projection,
            Df_shape=(num_output_channels, self._fno_block.num_output_channels),
            x=v_out,
        )

        super().__init__(self._projection_jacobian, self._fno_block)
