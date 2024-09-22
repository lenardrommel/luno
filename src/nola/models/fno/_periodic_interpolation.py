from . import dft


def gridded_fourier_interpolation(
    v,
    axes: tuple[int, ...],
    output_grid_shape: tuple[int, ...],
):
    v_rdft = dft.rfftn(
        v,
        axes=axes,
        norm="forward",
    )

    v_interp = dft.irfftn(
        v_rdft,
        grid_shape=output_grid_shape,
        axes=axes,
        norm="forward",
    )

    return v_interp
