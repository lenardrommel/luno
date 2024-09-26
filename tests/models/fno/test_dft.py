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
