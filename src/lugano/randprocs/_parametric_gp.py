from collections.abc import Callable

import jax
import linox
from jax import numpy as jnp
from jax.typing import ArrayLike, DTypeLike
from linox.typing import ShapeLike


class ParametricGaussianProcess:
    def __init__(
        self,
        weight_cov: linox.LinearOperator,
        mean_and_features: (
            Callable[
                [ArrayLike],
                tuple[jax.Array, linox.LinearOperator],
            ]
            | None
        ) = None,
    ):
        self.__mean_and_features = mean_and_features
        self._weight_cov = linox.utils.as_linop(weight_cov)

    @classmethod
    def from_weights_and_features(
        cls,
        weight_mean: ArrayLike,
        weight_cov: linox.LinearOperator,
        features: Callable[[ArrayLike], linox.LinearOperator],
    ):
        def mean_and_features(x: ArrayLike) -> tuple[jax.Array, linox.LinearOperator]:
            features_x = features(x)

            return features_x @ weight_mean, features_x

        return cls(weight_cov, mean_and_features)

    @property
    def weight_cov(self) -> linox.LinearOperator:
        return self._weight_cov

    def mean_and_features(
        self, x: ArrayLike, /
    ) -> tuple[ArrayLike, linox.LinearOperator]:
        return self.__mean_and_features(x)

    def sample(
        self,
        key: jax.Array,
        x: ArrayLike | None = None,
        size: ShapeLike = (),
        dtype: DTypeLike = jnp.float32,
    ) -> ArrayLike:
        if x.dtype != dtype:
            msg = (
                f"GP input dtype {x.dtype} doesn't match explicit sample dtype {dtype}."
            )
            raise TypeError(msg)

        weight_cov_lsqrt = linox.lsqrt(self._weight_cov)

        weight_sample = jax.random.normal(
            key,
            shape=size + weight_cov_lsqrt.shape[-1:],
            dtype=dtype,
        )[..., None]
        weight_sample = weight_cov_lsqrt @ weight_sample

        def sample_fn(x: ArrayLike) -> jax.Array:
            mean_x, features_x = self.mean_and_features(x)

            return mean_x + (features_x @ weight_sample)[..., 0]

        if x is None:
            return sample_fn

        return sample_fn(x)

    def mean_and_cov(self, x: ArrayLike, /) -> tuple[jax.Array, linox.LinearOperator]:
        mean_x, features_x = self.mean_and_features(x)

        return mean_x, linox.congruence_transform(features_x, self._weight_cov)

    def mean(self, x: ArrayLike, /) -> ArrayLike:
        return self.mean_and_features(x)[0]

    def cov(
        self,
        x0: ArrayLike,
        x1: ArrayLike | None = None,
        /,
    ) -> linox.LinearOperator:
        if x1 is None:
            _, cov_x0 = self.mean_and_cov(x0)

            return cov_x0

        _, features_x0 = self.mean_and_features(x0)
        _, features_x1 = self.mean_and_features(x1)

        return features_x0 @ self._weight_cov @ features_x1.T

    def mean_and_var(self, x: ArrayLike, /) -> tuple[ArrayLike, ArrayLike]:
        mean_x, cov_x = self.mean_and_cov(x)

        return mean_x, linox.diagonal(cov_x)

    def var(self, x: ArrayLike, /) -> ArrayLike:
        _, var_x = self.mean_and_var(x)

        return var_x

    def mean_and_std(self, x: ArrayLike, /) -> tuple[ArrayLike, ArrayLike]:
        mean_x, var_x = self.mean_and_var(x)

        return mean_x, jnp.sqrt(var_x)

    def std(self, x: ArrayLike, /) -> ArrayLike:
        _, std_x = self.mean_and_std(x)

        return std_x
