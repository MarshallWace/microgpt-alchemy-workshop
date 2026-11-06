import pytest

from microgpt.config import ModelConfig
from microgpt.pt.model import GPT


@pytest.mark.parametrize('tie', [False, True])
def test_initial_weight_sizes(tie):
    """The output layer starts at std 1/sqrt(hidden_dim), and untied token embeddings at 1/hidden_dim."""
    model = GPT(ModelConfig(vocab_size=4000, hidden_dim=64, num_layers=1, tie_word_embeddings=tie))
    assert model.lm_head.weight.std().item() == pytest.approx(64 ** -0.5, rel=0.05)
    embedding_std = model.model.embed_tokens.weight.std().item()
    assert embedding_std == pytest.approx(64 ** -0.5 if tie else 1 / 64, rel=0.05)
