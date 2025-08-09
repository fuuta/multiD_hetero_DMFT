from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TrialResult:
    initial_rndkey: jax.Array
    ys_history: np.ndarray
    J: np.ndarray
    nd_coeff: np.ndarray

    hetero_var_vals: np.ndarray | None

    f_jacob: Callable | None

    dys_history: np.ndarray | None

    eI_history: np.ndarray | None
    eIsource_history: np.ndarray | None
    Win: np.ndarray | None
    description: str = ""
