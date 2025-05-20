import numpy as np

from numpy.typing import DTypeLike


class FFTGrid(np.ndarray):
    def __new__(cls, shape: tuple[int, ...], dtype: DTypeLike = np.double):
        grids = tuple(
            np.linspace(0.0, 1.0, n, endpoint=False, dtype=dtype) for n in shape
        )

        obj = np.stack(
            np.meshgrid(
                *grids,
                copy=True,
                sparse=False,
                indexing="ij",
            ),
            axis=-1,
        ).view(cls)

        obj.grids = grids

        return obj
