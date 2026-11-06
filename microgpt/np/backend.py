# The numpy backend's versions of microgpt/py/backend.py
from contextlib import AbstractContextManager, nullcontext

from threadpoolctl import threadpool_limits

from microgpt.py.backend import load_state_dict, seed, state_dict  # the same as python's
from microgpt.np.model import DTYPES

SUPPORTS_COMPILE = False


def set_precision(model, params: str, compute: str) -> AbstractContextManager:
    # Casts the weights to params; numpy computes in its weights' format
    assert compute == params, 'numpy computes in the format of its weights'
    for p in model.parameters():
        p.data, p.grad = p.data.astype(DTYPES[params]), p.grad.astype(DTYPES[params])
    return nullcontext()


def set_threads(threads: int) -> None:
    if threads:  # 0 leaves numpy at its default
        threadpool_limits(threads)
