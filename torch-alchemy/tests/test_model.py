import pytest
import torch

from alchemy.checkpoint import load_checkpoint, save_checkpoint
from alchemy.config import AdamWConfig, ModelConfig, MuonConfig, SGDConfig
from alchemy.model import GPT, loss


def tiny(tie: bool = False) -> GPT:
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=300, max_seq_len=8, num_layers=2, hidden_dim=16, num_heads=2,
                           tie_word_embeddings=tie))


@pytest.mark.parametrize('tie', [False, True])
def test_initial_weight_sizes(tie):
    """The output layer starts at std 1/sqrt(hidden_dim), and untied token embeddings at 1/hidden_dim."""
    model = GPT(ModelConfig(vocab_size=4000, hidden_dim=64, num_layers=1, tie_word_embeddings=tie))
    assert model.lm_head.weight.std().item() == pytest.approx(64 ** -0.5, rel=0.05)
    assert model.model.embed_tokens.weight.std().item() == pytest.approx(64 ** -0.5 if tie else 1 / 64, rel=0.05)


@pytest.mark.parametrize('tie', [False, True])
def test_save_and_load_give_the_same_model(tie, tmp_path):
    """A saved model loads with the same config and outputs (a tied output layer stays tied)."""
    original = tiny(tie)
    save_checkpoint(str(tmp_path / 'model.safetensors'), original)
    loaded = load_checkpoint(str(tmp_path / 'model.safetensors'))
    assert loaded.config == original.config
    assert (loaded.lm_head.weight is loaded.model.embed_tokens.weight) == tie
    ids = torch.tensor([[299, 5, 104, 101]])
    torch.testing.assert_close(loaded(ids), original(ids))


def test_muonh_keeps_each_matrix_size():
    """MuonH moves the block matrices without changing their Frobenius norm; the rest go to AdamW."""
    model = tiny()
    adamw, muon = MuonConfig(hyperball=True).build(model)
    matrices = muon.param_groups[0]['params']
    before = [w.norm().item() for w in matrices]
    loss(model, torch.arange(9)[None]).backward()
    muon.step()
    assert [w.norm().item() for w in matrices] == pytest.approx(before, rel=1e-6)


@pytest.mark.parametrize('config', [MuonConfig(), AdamWConfig(), SGDConfig()], ids=type)
def test_every_weight_has_one_optimizer(config):
    """Each weight is updated by exactly one of the optimisers the config builds."""
    model = tiny(tie=True)
    updated = [id(p) for optimizer in config.build(model) for group in optimizer.param_groups for p in group['params']]
    assert sorted(updated) == sorted(id(p) for p in model.parameters())


def test_decay_embeddings_false_skips_the_tied_embeddings():
    model = tiny(tie=True)
    decay, no_decay = AdamWConfig(weight_decay=0.1, decay_embeddings=False).param_groups(dict(model.named_parameters()))
    assert any(p is model.lm_head.weight for p in no_decay['params'])
    assert not any(p is model.lm_head.weight for p in decay['params'])
