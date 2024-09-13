import jax
from jax import numpy as jnp
import linox

from collections.abc import Callable
from jax.typing import ArrayLike
from linox.typing import ShapeLike


class ParametricGaussianProcess:
    def __init__(
        self,
        mean_fn: Callable[[ArrayLike], ArrayLike],
        weight_cov: linox.LinearOperator,
        feature_fns: Callable[[ArrayLike], linox.LinearOperator],
    ):
        self._mean_fn = mean_fn

        self._weight_cov = linox.utils.as_linop(weight_cov)

        self._feature_fns = feature_fns

    @classmethod
    def from_weights_and_features(
        cls,
        weight_mean: ArrayLike,
        weight_cov: linox.LinearOperator,
        feature_fns: Callable[[ArrayLike], linox.LinearOperator],
    ):
        return cls(
            lambda x: feature_fns(x) @ weight_mean,
            weight_cov,
            feature_fns,
        )

    @property
    def weight_cov(self) -> linox.LinearOperator:
        return self._weight_cov

    @property
    def feature_fns(self) -> Callable[[ArrayLike], linox.LinearOperator]:
        return self._feature_fns

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

        def sample_fn(x: ArrayLike) -> jax.Array:
            return self.mean(x) + (self._feature_fns(x) @ weight_sample)[..., 0]

        if x is None:
            return sample_fn

        return sample_fn(x)

    def mean(self, x: ArrayLike, /) -> ArrayLike:
        return self._mean_fn(x)

    def cov(
        self,
        x0: ArrayLike,
        x1: ArrayLike | None = None,
        /,
    ) -> linox.LinearOperator:
        if x1 is None:
            return linox.congruence_transform(self._feature_fns(x0), self._weight_cov)

        return self._feature_fns(x0) @ self._weight_cov @ self._feature_fns(x1).T

    def var(self, x: ArrayLike, /) -> ArrayLike:
        x = jnp.array(x)

        return linox.diagonal(self.cov(x))

    def std(self, x: ArrayLike, /) -> ArrayLike:
        return jnp.sqrt(self.var(x))
