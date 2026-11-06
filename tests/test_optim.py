import numpy as np
import pytest
import torch

from microgpt.config import BACKENDS, AdamWConfig, ModelConfig, MuonConfig, SGDConfig
from microgpt.np.autograd import Tensor
from microgpt.optim import leaves
from microgpt.py.autograd import Value

START = np.array([[0.5, -1.0, 2.0], [1.5, 0.3, -0.7]])


def steps(backend: str, optimizer: str, **settings) -> np.ndarray:
    # Three steps of an optimiser with weight decay, from START with gradients 1, 2, 3
    if backend == 'torch':
        params = [torch.nn.Parameter(torch.tensor(START))]
    elif backend == 'numpy':
        params = [Tensor(START.copy())]
    else:
        params = [Value(x) for x in START.ravel()]
    optimizer = getattr(BACKENDS[backend], optimizer)(params, lr=0.1, weight_decay=0.2, **settings)
    for grad in (1.0, 2.0, 3.0):
        for p in params:
            full = torch.full_like if backend == 'torch' else lambda p, g: np.full_like(p.data, g)
            p.grad = grad if backend == 'python' else full(p, grad)
        optimizer.step()
    return np.array([p.detach().numpy() if backend == 'torch' else p.data for p in params], dtype=float).ravel()


@pytest.mark.parametrize('backend', ['python', 'numpy'])
def test_sgd_matches_torch(backend):
    """Weight decay is added to the gradient before momentum, as in torch.optim.SGD."""
    np.testing.assert_allclose(steps(backend, 'SGD', momentum=0.9), steps('torch', 'SGD', momentum=0.9), atol=1e-12)


@pytest.mark.parametrize('backend', ['python', 'numpy'])
def test_adamw_matches_torch(backend):
    """Bias-corrected averages and decoupled weight decay, as in torch.optim.AdamW."""
    np.testing.assert_allclose(steps(backend, 'AdamW', betas=(0.8, 0.9)), steps('torch', 'AdamW', betas=(0.8, 0.9)),
                               atol=1e-12)


def muon_matrices(backend: str, config: MuonConfig) -> tuple:
    # A tiny model's Muon optimiser, its matrices, and a function giving their Frobenius norms
    BACKENDS[backend].seed(0)
    model = BACKENDS[backend].GPT(ModelConfig(vocab_size=20, max_seq_len=8, num_layers=1, hidden_dim=8, num_heads=2))
    _, muon = config.build(BACKENDS[backend], model)
    matrices = muon.param_groups[0]['params']
    norm = lambda w: float(np.linalg.norm(w.detach().numpy() if backend == 'torch' else w.data if backend == 'numpy'
                                          else [[v.data for v in row] for row in w]))
    return model, muon, lambda: np.array([norm(w) for w in matrices])


@pytest.mark.parametrize('backend', list(BACKENDS))
def test_muonh_keeps_each_matrix_size(backend):
    """MuonH moves the block matrices without changing their Frobenius norm, even with weight decay."""
    model, muon, norms = muon_matrices(backend, MuonConfig(hyperball=True, weight_decay=0.2))
    before = norms()
    BACKENDS[backend].loss(model, [list(range(9))]).backward()
    muon.step()
    np.testing.assert_allclose(norms(), before, rtol=1e-6)


@pytest.mark.parametrize('backend', list(BACKENDS))
def test_muon_weight_decay_is_decoupled(backend):
    """With no gradient, a plain Muon step only shrinks each matrix, by 1 - muon_lr x weight_decay."""
    _, muon, norms = muon_matrices(backend, MuonConfig(muon_lr=0.1, weight_decay=0.2))
    before = norms()
    if backend == 'torch':
        for w in muon.param_groups[0]['params']:
            w.grad = torch.zeros_like(w)
    else:
        muon.zero_grad()
    muon.step()
    np.testing.assert_allclose(norms(), before * (1 - 0.1 * 0.2), rtol=1e-6)


@pytest.mark.parametrize('backend', list(BACKENDS))
@pytest.mark.parametrize('config', [MuonConfig(), AdamWConfig(), SGDConfig()], ids=type)
def test_every_weight_has_one_optimizer(backend, config):
    """Each weight is updated by exactly one of the optimisers the config builds."""
    model = BACKENDS[backend].GPT(ModelConfig(vocab_size=20, max_seq_len=8, num_layers=1, hidden_dim=8, num_heads=2,
                                              tie_word_embeddings=True))
    updated = [id(p) for optimizer in config.build(BACKENDS[backend], model) for group in optimizer.param_groups
               for p in leaves(group['params'])]
    assert sorted(updated) == sorted(id(p) for p in model.parameters())


@pytest.mark.parametrize('backend', list(BACKENDS))
def test_decay_embeddings_false_skips_the_embeddings(backend):
    model = BACKENDS[backend].GPT(ModelConfig(vocab_size=20, max_seq_len=8, num_layers=1, hidden_dim=8, num_heads=2))
    named = dict(model.named_parameters())
    groups = AdamWConfig(weight_decay=0.1, decay_embeddings=False).param_groups(named)
    no_decay = {id(p) for p in leaves(groups[1]['params'])}
    assert all(id(p) in no_decay for name, w in named.items() if 'emb' in name or 'norm' in name for p in leaves([w]))
    assert not any(id(p) in no_decay for name, w in named.items() if 'emb' not in name and 'norm' not in name
                   for p in leaves([w]))
