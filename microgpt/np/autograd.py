# Like autograd.py, but each Tensor holds a whole numpy array, so one operation does the work of many Values
from __future__ import annotations

from collections.abc import Callable

import numpy as np


class Tensor:
    # Like Value in autograd.py, but data is an array, and each node keeps a function that turns its gradient into the
    # gradients of its children
    def __init__(self, data: np.ndarray, children: tuple[Tensor, ...] = (),
                 grad_fn: Callable[[np.ndarray], tuple[np.ndarray, ...]] | None = None):
        self.data = data
        self.grad = np.zeros_like(data)
        self._children = children
        self._grad_fn = grad_fn

    def __add__(self, other: Tensor) -> Tensor:
        return Tensor(self.data + other.data, (self, other),
                      lambda g: (unbroadcast(g, self.data.shape), unbroadcast(g, other.data.shape)))

    def backward(self) -> None:
        topo, visited = [], set()

        def build_topo(v: Tensor) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            if v._grad_fn:
                for child, grad in zip(v._children, v._grad_fn(v.grad)):
                    child.grad += grad


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    # Sum the gradient over the axes that numpy broadcast, so it matches the input's shape again
    while grad.ndim > len(shape):
        grad = grad.sum(0)
    return grad.sum(tuple(i for i, n in enumerate(shape) if n == 1), keepdims=True)


def embed(weight: Tensor, ids: np.ndarray) -> Tensor:
    def grad_fn(g: np.ndarray) -> tuple[np.ndarray]:
        grad = np.zeros_like(weight.data)
        np.add.at(grad, ids, g)
        return (grad,)
    return Tensor(weight.data[ids], (weight,), grad_fn)


def linear(x: Tensor, weight: Tensor) -> Tensor:
    # weight: [out, in], as in model.py
    return Tensor(x.data @ weight.data.T, (x, weight),
                  lambda g: (g @ weight.data, g.reshape(-1, g.shape[-1]).T @ x.data.reshape(-1, x.data.shape[-1])))


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(a.data @ b.data, (a, b), lambda g: (g @ b.data.swapaxes(-1, -2), a.data.swapaxes(-1, -2) @ g))


def scale(x: Tensor, c: float) -> Tensor:
    return Tensor(x.data * c, (x,), lambda g: (g * c,))


def relu(x: Tensor) -> Tensor:
    return Tensor(np.maximum(x.data, 0), (x,), lambda g: (g * (x.data > 0),))


def rmsnorm(x: Tensor) -> Tensor:
    s = (np.mean(x.data ** 2, -1, keepdims=True) + 1e-5) ** -0.5
    return Tensor(x.data * s, (x,), lambda g: (s * g - x.data * s ** 3 * np.mean(g * x.data, -1, keepdims=True),))


def softmax(x: Tensor) -> Tensor:
    e = np.exp(x.data - x.data.max(-1, keepdims=True))
    y = e / e.sum(-1, keepdims=True)
    return Tensor(y, (x,), lambda g: (y * (g - np.sum(g * y, -1, keepdims=True)),))


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    B, T, C = x.data.shape  # -> [B, num_heads, T, head_dim]
    return Tensor(x.data.reshape(B, T, num_heads, C // num_heads).transpose(0, 2, 1, 3), (x,),
                  lambda g: (g.transpose(0, 2, 1, 3).reshape(B, T, C),))


def merge_heads(x: Tensor) -> Tensor:
    B, H, T, D = x.data.shape  # -> [B, T, H * D]
    return Tensor(x.data.transpose(0, 2, 1, 3).reshape(B, T, H * D), (x,),
                  lambda g: (g.reshape(B, T, H, D).transpose(0, 2, 1, 3),))


def transpose_last(x: Tensor) -> Tensor:
    return Tensor(x.data.swapaxes(-1, -2), (x,), lambda g: (g.swapaxes(-1, -2),))


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    # The mean over all positions of -log(probability of the target)
    idx = targets[..., None]
    shifted = logits.data - logits.data.max(-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    picked = np.take_along_axis(log_probs, idx, -1)[..., 0]

    def grad_fn(g: np.ndarray) -> tuple[np.ndarray]:
        grad = np.exp(log_probs)  # softmax - one_hot(target)
        np.put_along_axis(grad, idx, np.take_along_axis(grad, idx, -1) - 1, -1)
        return (g * grad / picked.size,)
    return Tensor(np.array(-picked.mean()), (logits,), grad_fn)
