import numpy as np

from microgpt.np.autograd import Tensor
from microgpt.optim import Optimizer as BaseOptimizer


def clip_grad_norm(params: list[Tensor], max_norm: float) -> None:
    # Scales all gradients down together so that their combined size is at most max_norm
    norm = np.sqrt(sum((p.grad ** 2).sum() for p in params))
    for p in params:
        p.grad *= min(1.0, max_norm / (norm + 1e-6))


class Optimizer(BaseOptimizer):
    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                p.grad = np.zeros_like(p.data)


class AdamW(Optimizer):
    def __init__(self, params: list, lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params, {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay})

    def step(self) -> None:
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                zeros = np.zeros_like(p.data)
                state = self.state.setdefault(id(p), {'step': 0, 'm': zeros, 'v': zeros.copy()})
                state['step'] += 1
                state['m'][:] = beta1 * state['m'] + (1 - beta1) * p.grad
                state['v'][:] = beta2 * state['v'] + (1 - beta2) * p.grad ** 2
                m_hat = state['m'] / (1 - beta1 ** state['step'])
                v_hat = state['v'] / (1 - beta2 ** state['step'])
                p.data -= group['lr'] * (m_hat / (v_hat ** 0.5 + group['eps']) + group['weight_decay'] * p.data)


class SGD(Optimizer):
    # Stochastic gradient descent with momentum: each step moves the weights against a running average of gradients
    def __init__(self, params: list, lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(params, {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay})

    def step(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                buf = self.state.setdefault(id(p), np.zeros_like(p.data))
                buf[:] = group['momentum'] * buf + p.grad + group['weight_decay'] * p.data
                p.data -= group['lr'] * buf


def newton_schulz(G: np.ndarray, steps: int = 5) -> np.ndarray:
    # Adapted from Keller Jordan's modded-nanogpt (MIT licence; see THIRD_PARTY_NOTICES.md)
    # Approximately orthogonalise G (push all its singular values towards 1) with a quintic iteration
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (np.linalg.norm(G) + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X


class Muon(Optimizer):
    # For weight matrices: Nesterov momentum, then orthogonalise the update. hyperball=True is MuonH: each matrix keeps
    # its Frobenius norm R, moving by lr * R along the normalised update and scaled back to norm R. weight_decay shrinks
    # the matrix first (not for MuonH, whose size is fixed)
    def __init__(self, params: list, lr: float = 0.02, momentum: float = 0.95, hyperball: bool = False,
                 weight_decay: float = 0.0):
        defaults = {'lr': lr, 'momentum': momentum, 'hyperball': hyperball, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                radius = np.linalg.norm(p.data)
                buf = self.state.setdefault(id(p), np.zeros_like(p.data))
                buf[:] = group['momentum'] * buf + p.grad
                update = newton_schulz(p.grad + group['momentum'] * buf)
                if group['hyperball']:
                    p.data -= group['lr'] * radius * update / (np.linalg.norm(update) + 1e-7)
                    p.data *= radius / (np.linalg.norm(p.data) + 1e-7)
                else:
                    p.data *= 1 - group['lr'] * group['weight_decay']
                    p.data -= group['lr'] * max(1.0, p.data.shape[0] / p.data.shape[1]) ** 0.5 * update
