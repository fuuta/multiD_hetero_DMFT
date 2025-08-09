from typing import Sequence, Callable
from dataclasses import dataclass
from math import prod

import jax
import jax.numpy as jnp
import jax.random as jr


def sample_with_condition(
    rndkey: jax.Array,
    shape: Sequence[int],
    condition_fn: Callable[[jnp.ndarray], jnp.ndarray],
    sample_fn: Callable = jr.normal,
) -> tuple[jax.Array, jax.Array]:
    """
    Sample from a distribution until the condition is met.

    Args:
        rndkey: JAX random key.
        shape: Shape of the output array.
        condition_fn: Function that takes a sample and returns a boolean indicating if the condition is met.
        sample_fn: Function to generate samples (default is normal distribution).

    Returns:
        A tuple of the sampled array and the updated random key.
    """
    if isinstance(shape, int):
        n_need = shape
    else:
        n_need = prod(shape)
    n_sample = n_need * 10
    ret = jnp.zeros([n_need], dtype=jnp.float32)
    n_valid_samples = 0

    while True:
        samples = sample_fn(rndkey, shape=[n_sample])
        rndkey, subkey = jax.random.split(rndkey)
        mask = condition_fn(samples)
        valid_samples = samples[mask]
        ret = ret.at[
            n_valid_samples : min(n_valid_samples + valid_samples.size, ret.size)
        ].set(
            valid_samples[
                : min(n_valid_samples + valid_samples.size, ret.size) - n_valid_samples
            ]
        )
        n_valid_samples += valid_samples.size
        if n_valid_samples >= n_need:
            break

    ret = ret.reshape(shape)
    rndkey, subkey = jax.random.split(rndkey)
    return ret, rndkey


@dataclass(frozen=True)
class NormalDist:
    mean: float = 0.0
    sigma: float = 1.0

    def __post_init__(self):
        assert self.sigma >= 0.0

    def sample(
        self, rndkey: jax.Array, shape: Sequence[int]
    ) -> tuple[jax.Array, jax.Array]:
        ret = jr.normal(rndkey, shape=shape) * self.sigma + self.mean
        rndkey, subkey = jax.random.split(rndkey)
        return ret, rndkey


@dataclass(frozen=True)
class LogNormalDist:
    mean: float = 0.0
    sigma: float = 1.0

    def __post_init__(self):
        assert self.sigma >= 0.0

    def sample(
        self, rndkey: jax.Array, shape: Sequence[int]
    ) -> tuple[jax.Array, jax.Array]:
        ret = jnp.exp(jr.normal(rndkey, shape=shape) * self.sigma + self.mean)
        rndkey, subkey = jax.random.split(rndkey)
        return ret, rndkey


@dataclass(frozen=True)
class TruncatedNormalDist:
    min: float
    max: float
    mean: float = 0.0
    sigma: float = 1.0

    def __post_init__(self):
        assert self.sigma >= 0.0

    def sample(
        self, rndkey: jax.Array, shape: Sequence[int]
    ) -> tuple[jax.Array, jax.Array]:
        def condition_fn(samples):
            return (samples >= self.min) & (samples <= self.max)

        def sample_fn(rndkey, shape):
            return jr.normal(rndkey, shape=shape) * self.sigma + self.mean

        return sample_with_condition(rndkey, shape, condition_fn, sample_fn)


@dataclass(frozen=True)
class TruncatedLogNormalDist:
    min: float
    max: float
    mean: float = 0.0
    sigma: float = 1.0
    flip_sign: bool = False  # If True, samples will be flipped in sign

    def __post_init__(self):
        assert self.sigma >= 0.0

    def sample(
        self, rndkey: jax.Array, shape: Sequence[int]
    ) -> tuple[jax.Array, jax.Array]:
        def condition_fn(samples):
            return (samples >= self.min) & (samples <= self.max)

        def sample_fn(rndkey, shape):
            return self.sigma * jnp.exp(
                jr.normal(rndkey, shape=shape) * self.sigma + self.mean
            )
        
        if self.flip_sign:
            def flip_sign_sample_fn(rndkey, shape):
                ret = sample_fn(rndkey, shape)
                return -ret
            return sample_with_condition(rndkey, shape, condition_fn, flip_sign_sample_fn)
        else:
            return sample_with_condition(rndkey, shape, condition_fn, sample_fn)


@dataclass(frozen=True)
class GammaDist:
    k: float
    scale: float

    def __post_init__(self):
        assert self.k > 0.0

    def sample(
        self, rndkey: jax.Array, shape: list[int]
    ) -> tuple[jax.Array, jax.Array]:
        ret = jr.gamma(rndkey, a=self.k, shape=shape) * self.scale
        return ret, rndkey


@dataclass(frozen=True)
class UniformDist:
    min_val: float = 0.0
    max_val: float = 1.0

    def __post_init__(self):
        assert self.min_val <= self.max_val

    def sample(
        self, rndkey: jax.Array, shape: list[int]
    ) -> tuple[jax.Array, jax.Array]:
        ret = jr.uniform(rndkey, shape=shape, minval=self.min_val, maxval=self.max_val)
        return ret, rndkey
    
    @property
    def variance(self) -> float:
        """Variance of the uniform distribution."""
        return (self.max_val - self.min_val) ** 2 / 12.0


@dataclass(frozen=True)
class TwoValDist:
    p: float
    low_val: float = 0.0
    high_val: float = 1.0

    def __post_init__(self):
        assert 0.0 <= self.p <= 1.0
        assert self.high_val >= self.low_val

    def sample(
        self, rndkey: jax.Array, shape: list[int]
    ) -> tuple[jax.Array, jax.Array]:
        bernoulli_sample = jr.uniform(rndkey, shape=shape) < self.p
        ret = bernoulli_sample * self.low_val + (1 - bernoulli_sample) * self.high_val
        rndkey, subkey = jax.random.split(rndkey)
        return ret, rndkey


ProbabilityDistribution = (
    NormalDist
    | LogNormalDist
    | TruncatedNormalDist
    | TruncatedLogNormalDist
    | GammaDist
    | UniformDist
    | TwoValDist
)
