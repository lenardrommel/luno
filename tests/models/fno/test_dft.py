import fractions

import jax
from jax import numpy as jnp
import numpy as np

from pytest_cases import AUTO, parametrize

import nola


@parametrize(
    "grid_shape,modes_shape",
    (
        ((3,), (2,)),
        ((16, 15), (16, 3)),
        ((16, 16), (6, 4)),
        ((16, 32), (8, 8)),
    ),
    idgen=AUTO,
)
def test_rfftn_truncation(grid_shape: tuple[int, ...], modes_shape: tuple[int, ...]):
    signal = jax.random.normal(
        jax.random.key(985367 + sum(grid_shape) + sum(modes_shape)),
        shape=grid_shape,
    )

    z_trunc = nola.models.fno.dft.rfftn(
        signal,
        modes_shape=modes_shape,
        axes=tuple(range(signal.ndim)),
        norm="forward",
    )

    # Reference implementation
    z = jnp.fft.rfftn(signal, axes=tuple(range(signal.ndim)), norm="forward")

    z_trunc_ref = jnp.fft.fftshift(z, axes=tuple(range(z.ndim - 1)))
    z_trunc_ref = z_trunc_ref[
        *(
            slice((n - m) // 2, n + (-(n - m) // 2))
            for n, m in zip(z_trunc_ref.shape[:-1], modes_shape[:-1], strict=True)
        ),
        : modes_shape[-1],
    ]
    z_trunc_ref = jnp.fft.ifftshift(z_trunc_ref, axes=tuple(range(z.ndim - 1)))

    np.testing.assert_allclose(
        z_trunc,
        z_trunc_ref,
        rtol=1e-6,
        atol=1e-7,
    )


@parametrize(
    "input_grid_shape,output_grid_shape",
    (
        ((4,), (6,)),
        ((6,), (9,)),
        ((5,), (10,)),
        ((6, 3), (10, 15)),
        ((15, 14), (64, 64)),
    ),
    idgen=AUTO,
)
def test_odd_irfftn_interpolates(
    input_grid_shape: tuple[int, ...],
    output_grid_shape: tuple[int, ...],
):
    signal = jax.random.normal(
        jax.random.key(423897 + sum(input_grid_shape) + sum(output_grid_shape)),
        shape=input_grid_shape,
    )

    signal_rfft = jnp.fft.rfftn(signal, axes=tuple(range(signal.ndim)), norm="forward")

    signal_interp = nola.models.fno.dft.irfftn(
        signal_rfft,
        grid_shape=output_grid_shape,
        axes=tuple(range(signal.ndim)),
        last_axis_from_even=input_grid_shape[-1] % 2 == 0,
        norm="forward",
    )

    shape_ratios = tuple(
        fractions.Fraction(n_out, n_in)
        for n_out, n_in in zip(output_grid_shape, input_grid_shape)
    )

    np.testing.assert_allclose(
        signal_interp[
            *(
                slice(0, n, ratio.numerator)
                for n, ratio in zip(output_grid_shape, shape_ratios)
            )
        ],
        signal[
            *(
                slice(0, n, ratio.denominator)
                for n, ratio in zip(input_grid_shape, shape_ratios)
            )
        ],
        rtol=1e-5,
        atol=1e-5,
    )


def debug_plot_1d(signal, signal_interp):
    from matplotlib import pyplot as plt

    signal = jnp.pad(signal, ((0, 1),), mode="wrap")
    signal_interp = jnp.pad(signal_interp, ((0, 1),), mode="wrap")

    plt.plot(jnp.linspace(0, 1, signal.shape[0]), signal, "o-")
    plt.plot(jnp.linspace(0, 1, signal_interp.shape[0]), signal_interp, "+-")

    plt.show()


def debug_plot_2d(signal, signal_rfft, signal_interp):
    from matplotlib import pyplot as plt

    signal_np_irfft = jnp.fft.irfftn(
        signal_rfft,
        s=signal_interp.shape,
        axes=tuple(range(signal_rfft.ndim)),
        norm="forward",
    )

    signal = jnp.pad(signal, ((0, 1), (0, 1)), mode="wrap")
    signal_interp = jnp.pad(signal_interp, ((0, 1), (0, 1)), mode="wrap")
    signal_np_irfft = jnp.pad(signal_np_irfft, ((0, 1), (0, 1)), mode="wrap")

    _, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(signal, aspect=1)
    ax[0, 1].imshow(signal_interp, aspect=1)
    ax[1, 0].imshow(signal_interp, aspect=1)
    ax[1, 1].imshow(signal_np_irfft, aspect=1)

    plt.show()
