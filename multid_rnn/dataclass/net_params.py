from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Type, Callable, Literal

import jax
import jax.numpy as jnp
from plum import dispatch
from serde import serialize

from ..utils.logging_utils import get_logger
from .prob_dist import ProbabilityDistribution, NormalDist
from .external_input import ExternalInputParams

logger = get_logger()


# class Jax2DArraySerializer:
#     @dispatch
#     def serialize(self, value: jax.Array) -> list[list[float]]:
#         if len(value.shape) != 2 or value.shape[0] != value.shape[1]:
#             raise ValueError(f"Expected a square 2D array, but got shape {value.shape}")
#         return value.tolist()


# class Jax2DArrayDeserializer:
#     def deserialize(self, cls: Type[jax.Array], value: Any) -> jax.Array:
#         return jnp.asarray(value)

#     def deserialize(self, cls: Type[None], value: Any) -> None:
#         return None


@dataclass(frozen=True)
class HeteroGPA1Var: ...


@dataclass(frozen=True)
class HeteroGPA2var: ...


@dataclass(frozen=True)
class GammaheteroGPA2var: ...


@dataclass(frozen=True)
class BetaheteroGPA2var: ...


@dataclass(frozen=True)
class BetaheteroADA2var: ...


@dataclass(frozen=True)
class GammaheteroADA2var: ...


NetworkType = (
    HeteroGPA1Var
    | HeteroGPA2var
    | GammaheteroGPA2var
    | BetaheteroGPA2var
    | BetaheteroADA2var
    | GammaheteroADA2var
)


@dataclass(frozen=True)
class HeterogeneousParameter:
    index: tuple[int, int] | Literal["gain"]
    dist: ProbabilityDistribution
    label: str = "unknown"
    shift: float = 0.0

    def validate(self):
        pass

    def __post_init__(self):
        self.validate()
        if isinstance(self.index, tuple):
            if self.index[0] == 1 and self.index[1] == 0:
                object.__setattr__(self, "label", "beta")
            elif self.index[0] == 1 and self.index[1] == 1:
                object.__setattr__(self, "label", "gamma")
            else:
                if self.index[0] == self.index[1]:
                    object.__setattr__(self, "label", f"decay var{self.index[0]}")
                else:
                    object.__setattr__(self, "label", f"effect from var{self.index[0]} to var{self.index[1]}")
                logger.warning("Index of HeterogeneousParameter is not interpreted")
        elif self.index == "gain":
            object.__setattr__(self, "label", "gain")
        else:
            raise ValueError("Invalid index for HeterogeneousParameter. Must be a tuple or 'gain'.")


@dataclass(frozen=True)
class BaseActf(ABC):
    @abstractmethod
    def to_f(self) -> Callable[[jnp.ndarray], jnp.ndarray]: ...


@dataclass(frozen=True)
class Tanh(BaseActf):
    def to_f(self):
        return jnp.tanh


@dataclass(frozen=True)
class PWL(BaseActf):
    def to_f(self):
        def f(x: jnp.ndarray) -> jnp.ndarray:
            return ((x <= -1) * (-1) + (x > -1) * (x < 1) * x + (x >= 1) * 1).astype(jnp.float64)
        return f


@dataclass(frozen=True)
class TestActf(BaseActf): ...


Actf = Tanh | PWL | TestActf

# @serialize(
#     class_serializer=Jax2DArraySerializer(), class_deserializer=Jax2DArrayDeserializer()
# )
@serialize()
@dataclass(frozen=True)
class NetworkParameters:
    n_neuron: int
    hetero_info: HeterogeneousParameter | None
    coupling_strength: float
    activation_function: Actf
    coefficient: list[list[float]]
    n_dimension: int | None = None
    net_type: NetworkType | None = None
    external_input: ExternalInputParams | None = None
    noise: NormalDist | None = None

    def validate(self):
        assert self.n_neuron > 0, "Number of neurons must be positive."
        assert self.n_dimension is None or self.n_dimension > 0, "Number of dimensions must be positive."
        assert self.coupling_strength >= 0, "Coupling strength must be non-negative."
        assert isinstance(self.activation_function, BaseActf), (
            "Invalid activation function type."
        )
        coef_jnp = jnp.asarray(self.coefficient)
        assert len(coef_jnp.shape) == 2
        assert coef_jnp.shape[0] == coef_jnp.shape[1]
        if self.hetero_info is not None:
            if self.hetero_info.index != "gain":
                assert (
                    coef_jnp[self.hetero_info.index[0], self.hetero_info.index[1]] == 1.0
                    or coef_jnp[self.hetero_info.index[0], self.hetero_info.index[1]] == -1.0
                ), "coeff for heterogeneous parameter must be 1 or -1."

    def __post_init__(self):
        self.validate()

    @property
    def _n_dimension(self) -> int:
        return len(self.coefficient)
    
    @property
    def n_hetero(self) -> int:
        return 0 if self.hetero_info is None else 1

    @property
    def _net_type(self) -> str:
        coef_jnp = jnp.asarray(self.coefficient)
        n_dim = coef_jnp.shape[0]
        if self.hetero_info is not None:
            if self.hetero_info.label == "unknown":
                hetero_param_name = (
                    f"coef[{self.hetero_info.index[0]},{self.hetero_info.index[1]}]"
                )
            else:
                hetero_param_name = self.hetero_info.label
            hetero_dist = self.hetero_info.dist.__class__.__name__.lower()
        else:
            hetero_param_name = None
            hetero_dist = "nonhetero"
        if n_dim == 2:
            if hetero_param_name == "gamma":
                if coef_jnp[1, 0] < 0:
                    net_name = "ADA"
                else:
                    net_name = "GPA"
            else:
                net_name = ""
        else:
            net_name = ""
        if hetero_param_name is None:
            return f"{hetero_dist}{net_name}{n_dim}var"
        else:
            return f"{hetero_dist}{hetero_param_name}{net_name}{n_dim}var"
