import numpy as np


class FFTGrid(np.ndarray):
    def __new__(cls, shape: tuple[int, ...]):
        grids = tuple(np.linspace(0.0, 1.0, n + 1)[:-1] for n in shape)

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
