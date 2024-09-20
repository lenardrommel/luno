import linox

from jax.typing import ArrayLike

from .._fixed_input_pointwise_affine import FixedInputPointwiseAffineTransform
from ._fixed_input_spectral_convolution import FixedInputSpectralConvolution


class FixedInputFNOBlock(linox.BlockMatrix):
    def __init__(
        self,
        v: ArrayLike,
        z: ArrayLike,
        output_grid_shape: tuple[int, ...],
        num_output_channels: int | None = None,
    ):
        self._fixed_input_spectral_convolution = FixedInputSpectralConvolution(
            z,
            output_grid_shape=output_grid_shape,
            num_output_channels=num_output_channels,
        )

        self._fixed_input_pointwise_affine_transform = (
            FixedInputPointwiseAffineTransform(
                v,
                num_output_channels=num_output_channels,
            )
        )

        # TODO: Interpolation of the skip connection

        super().__init__(
            [
                [
                    self._fixed_input_spectral_convolution,
                    self._fixed_input_pointwise_affine_transform,
                ]
            ]
        )

    @property
    def num_output_channels(self) -> int:
        return self._fixed_input_spectral_convolution.num_output_channels
