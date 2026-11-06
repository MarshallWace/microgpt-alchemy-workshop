from collections import defaultdict

import torch
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.optim import SGD, AdamW


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    # Adapted from Keller Jordan's modded-nanogpt (MIT licence; see THIRD_PARTY_NOTICES.md)
    # Approximately orthogonalise each matrix in G [..., m, n] (push its singular values towards 1)
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = G.shape[-2] > G.shape[-1]
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return (X.mT if transposed else X).float()


class Muon(torch.optim.Optimizer):
    # As microgpt/np/optim.py's Muon (with its weight_decay); matrices of the same shape are orthogonalised together,
    # in bfloat16
    def __init__(self, params: list, lr: float = 0.02, momentum: float = 0.95, hyperball: bool = False,
                 weight_decay: float = 0.0):
        defaults = {'lr': lr, 'momentum': momentum, 'hyperball': hyperball, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            by_shape = defaultdict(list)
            for p in group['params']:
                by_shape[p.shape].append(p)
            for shape, params in by_shape.items():
                bufs = [self.state[p].setdefault('momentum_buffer', torch.zeros_like(p)) for p in params]
                torch._foreach_mul_(bufs, group['momentum'])
                torch._foreach_add_(bufs, [p.grad for p in params])
                updates = newton_schulz(torch.stack([p.grad + group['momentum'] * b for p, b in zip(params, bufs)]))
                weights = torch.stack(params)
                if group['hyperball']:
                    radius = weights.norm(dim=(-2, -1), keepdim=True)
                    weights -= group['lr'] * radius * updates / (updates.norm(dim=(-2, -1), keepdim=True) + 1e-7)
                    weights *= radius / (weights.norm(dim=(-2, -1), keepdim=True) + 1e-7)
                else:
                    weights *= 1 - group['lr'] * group['weight_decay']
                    weights -= group['lr'] * max(1.0, shape[0] / shape[1]) ** 0.5 * updates
                torch._foreach_copy_(params, list(weights))
