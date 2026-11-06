# The torch backend's versions of microgpt/py/backend.py
import random
from contextlib import AbstractContextManager

import numpy as np
import torch

from microgpt.np.backend import set_threads as set_numpy_threads
from microgpt.pt.model import DTYPES

SUPPORTS_COMPILE = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # CUDA_VISIBLE_DEVICES= keeps a GPU machine on the CPU


def set_precision(model, params: str, compute: str) -> AbstractContextManager:
    # Moves the model to DEVICE with its weights in params; the maths runs in compute when it differs (mixed precision,
    # autocast)
    model.to(DEVICE, DTYPES[params])
    return torch.autocast(DEVICE, dtype=DTYPES[compute], enabled=compute != params)


def state_dict(model) -> dict[str, np.ndarray]:
    # named_parameters lists a tied matrix once, unlike torch's state_dict
    return {name: p.detach().float().cpu().numpy() for name, p in model.named_parameters()}


def load_state_dict(model, weights: dict[str, np.ndarray]) -> None:
    missing, unexpected = model.load_state_dict({k: torch.from_numpy(w) for k, w in weights.items()}, strict=False)
    tied = ['lm_head.weight'] if model.config.tie_word_embeddings else []  # saved as model.embed_tokens.weight
    assert missing == tied and not unexpected, f'missing weights {missing}, unexpected {unexpected}'


def seed(value: int) -> None:
    random.seed(value)
    torch.manual_seed(value)


def set_threads(threads: int) -> None:
    if threads:  # 0 leaves numpy and torch at their default
        set_numpy_threads(threads)
        torch.set_num_threads(threads)
