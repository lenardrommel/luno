from collections.abc import Callable

import jax
from jax import numpy as jnp
import linox

from jax.typing import ArrayLike
from linox.typing import ShapeLike


class ParametricGaussianProcess:
    def __init__(
        self,
        weight_mean: ArrayLike,
        weight_cov: linox.LinearOperator,
        feature_fn: Callable[[ArrayLike], linox.LinearOperator],
    ):
        self._weight_mean = jnp.array(weight_mean)
        self._weight_cov = linox.utils.as_linop(weight_cov)

        self._feature_fn = feature_fn

    @property
    def weight_mean(self) -> ArrayLike:
        return self._weight_mean

    @property
    def weight_cov(self) -> linox.LinearOperator:
        return self._weight_cov

    @property
    def feature_fn(self) -> Callable[[ArrayLike], linox.LinearOperator]:
        return self._feature_fn

    def sample(
        self,
        key: jax.Array,
        x: ArrayLike | None = None,
        size: ShapeLike = (),
    ) -> ArrayLike:
        weight_cov_lsqrt = linox.lsqrt(self._weight_cov)

        weight_sample = jax.random.normal(
            key,
            shape=size + weight_cov_lsqrt.shape[-1:],
        )[..., None]
        weight_sample = weight_cov_lsqrt @ weight_sample
        weight_sample += self._weight_mean[..., None]

        def sample_fn(x: ArrayLike) -> jax.Array:
            return (self._feature_fn(x) @ weight_sample)[..., 0]

        if x is None:
            return sample_fn

        return sample_fn(x)

    def mean(self, x: ArrayLike, /) -> ArrayLike:
        x = jnp.array(x)

        return self._feature_fn(x) @ self._weight_mean

    def cov(
        self,
        x0: ArrayLike,
        x1: ArrayLike | None = None,
        /,
    ) -> linox.LinearOperator:
        if x1 is None:
            return linox.congruence_transform(self._feature_fn(x0), self._weight_cov)

        return self._feature_fn(x0) @ self._weight_cov @ self._feature_fn(x1).T

    def var(self, x: ArrayLike, /) -> ArrayLike:
        x = jnp.array(x)

        return linox.diagonal(self.cov(x))

    def std(self, x: ArrayLike, /) -> ArrayLike:
        return jnp.sqrt(self.var(x))
