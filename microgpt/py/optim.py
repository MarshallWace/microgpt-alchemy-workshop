from microgpt.optim import Optimizer
from microgpt.py.autograd import Value


def clip_grad_norm(params: list[Value], max_norm: float) -> None:
    # Scales all gradients down together so that their combined size is at most max_norm
    norm = sum(p.grad ** 2 for p in params) ** 0.5
    for p in params:
        p.grad *= min(1.0, max_norm / (norm + 1e-6))


class AdamW(Optimizer):
    def __init__(self, params: list, lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params, {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay})

    def step(self) -> None:
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                state = self.state.setdefault(id(p), {'step': 0, 'm': 0.0, 'v': 0.0})
                state['step'] += 1
                state['m'] = beta1 * state['m'] + (1 - beta1) * p.grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * p.grad ** 2
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
                state = self.state.setdefault(id(p), {'buf': 0.0})
                state['buf'] = group['momentum'] * state['buf'] + p.grad + group['weight_decay'] * p.data
                p.data -= group['lr'] * state['buf']


Matrix = list[list[float]]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def norm(a: Matrix) -> float:
    return sum(x * x for row in a for x in row) ** 0.5


def newton_schulz(G: Matrix, steps: int = 5) -> Matrix:
    # Adapted from Keller Jordan's modded-nanogpt (MIT licence; see THIRD_PARTY_NOTICES.md)
    # Approximately orthogonalise G (push all its singular values towards 1) with a quintic iteration
    a, b, c = 3.4445, -4.7750, 2.0315
    X = [[x / (norm(G) + 1e-7) for x in row] for row in G]
    transposed = len(X) > len(X[0])
    if transposed:
        X = transpose(X)
    for _ in range(steps):
        A = matmul(X, transpose(X))
        B = [[b * x + c * y for x, y in zip(r, s)] for r, s in zip(A, matmul(A, A))]
        X = [[a * x + y for x, y in zip(r, s)] for r, s in zip(X, matmul(B, X))]
    return transpose(X) if transposed else X


class Muon(Optimizer):
    # For weight matrices (lists of rows of Values): Nesterov momentum, then orthogonalise the update. hyperball=True
    # is MuonH: each matrix keeps its size (Frobenius norm R), moving by lr * R along the normalised update and scaled
    # back to size R. weight_decay shrinks the matrix first (not for MuonH, whose size is fixed)
    def __init__(self, params: list[list[list[Value]]], lr: float = 0.02, momentum: float = 0.95,
                 hyperball: bool = False, weight_decay: float = 0.0):
        defaults = {'lr': lr, 'momentum': momentum, 'hyperball': hyperball, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            for m in group['params']:
                self.update(m, group['lr'], group['momentum'], group['hyperball'], group['weight_decay'])

    def update(self, m: list[list[Value]], lr: float, momentum: float, hyperball: bool, weight_decay: float) -> None:
        radius = norm([[v.data for v in row] for row in m])
        buf = self.state.setdefault(id(m), [[0.0] * len(row) for row in m])
        for row, buf_row in zip(m, buf):
            buf_row[:] = [momentum * b + v.grad for v, b in zip(row, buf_row)]
        update = newton_schulz([[v.grad + momentum * b for v, b in zip(row, buf_row)] for row, buf_row in zip(m, buf)])
        scale = radius / (norm(update) + 1e-7) if hyperball else max(1.0, len(m) / len(m[0])) ** 0.5
        shrink = 1.0 if hyperball else 1 - lr * weight_decay
        for row, update_row in zip(m, update):
            for v, u in zip(row, update_row):
                v.data = v.data * shrink - lr * scale * u
        if hyperball:
            size = norm([[v.data for v in row] for row in m])
            for row in m:
                for v in row:
                    v.data *= radius / (size + 1e-7)
