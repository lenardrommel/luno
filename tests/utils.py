import numpy as np
import scipy.stats

from numpy.typing import ArrayLike


def assert_samples_marginally_gaussian(
    samples: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    axis: int = 0,
):
    samples = np.sort(samples, axis=axis)

    mean = np.expand_dims(mean, axis=axis)
    std = np.expand_dims(std, axis=axis)

    samples_standardized = (samples - mean) / (std + 1e-6)

    # Map standardized samples through standard normal cdf and compare to uniform cdf
    samples_norm_cdf = scipy.stats.norm.cdf(samples_standardized)
    uniform_cdf = np.broadcast_to(
        np.moveaxis(
            np.expand_dims(
                np.linspace(0.0, 1.0, samples_norm_cdf.shape[axis]),
                axis=tuple(range(1, samples_norm_cdf.ndim)),
            ),
            0,
            axis,
        ),
        shape=samples_norm_cdf.shape,
    )

    np.testing.assert_allclose(
        samples_norm_cdf,
        uniform_cdf,
        atol=6e-2,
    )
