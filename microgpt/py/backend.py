# What the training code needs from the python backend besides its model and optimisers (each backend has the same
# names, so train.py and checkpoint.py work with any of them)
import random
from contextlib import AbstractContextManager, nullcontext

import numpy as np

SUPPORTS_COMPILE = False


def set_precision(model, params: str, compute: str) -> AbstractContextManager:
    return nullcontext()  # Python numbers are always 64-bit


def state_dict(model) -> dict[str, np.ndarray]:
    # Each weight matrix by name, as float32 numbers for a checkpoint
    return {name: np.asarray(w, dtype=np.float32) for name, w in model.state_dict().items()}


def load_state_dict(model, weights: dict[str, np.ndarray]) -> None:
    model.load_state_dict(weights)


def seed(value: int) -> None:
    random.seed(value)  # the weights start from Python's random numbers


def set_threads(threads: int) -> None:
    pass  # Python runs on one thread
