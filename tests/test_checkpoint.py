import numpy as np
import pytest

from microgpt.checkpoint import load_checkpoint, save_checkpoint
from microgpt.config import BACKENDS, ModelConfig
from microgpt.tokenizer import BPETokenizer

TOKENIZER = BPETokenizer.train(['the film was good', 'the film was bad'], 300)
IDS = [TOKENIZER.bos_token_id] + TOKENIZER.encode('the film')


def model(backend: str, tie: bool):
    BACKENDS[backend].seed(0)
    config = ModelConfig(vocab_size=TOKENIZER.vocab_size, max_seq_len=8, num_layers=2, hidden_dim=16, num_heads=2,
                         tie_word_embeddings=tie)
    return BACKENDS[backend].GPT(config)


@pytest.mark.parametrize('backend', list(BACKENDS))
@pytest.mark.parametrize('tie', [False, True])
def test_save_and_load_give_the_same_model(backend, tie, tmp_path):
    """A saved model loads with the same config and outputs."""
    original = model(backend, tie)
    save_checkpoint(str(tmp_path / 'model.safetensors'), original, backend)
    loaded = load_checkpoint(str(tmp_path / 'model.safetensors'))
    assert loaded.config == original.config
    np.testing.assert_allclose(loaded.logits(IDS), original.logits(IDS), atol=1e-6)


def test_python_checkpoints_run_on_numpy(tmp_path):
    """The python and numpy models are the same model: a python checkpoint loads on numpy with the same outputs."""
    original = model('python', tie=False)
    save_checkpoint(str(tmp_path / 'model.safetensors'), original, 'python')
    loaded = load_checkpoint(str(tmp_path / 'model.safetensors'))
    assert type(loaded) is BACKENDS['numpy'].GPT
    np.testing.assert_allclose(loaded.logits(IDS), original.logits(IDS), atol=1e-6)
    np.testing.assert_allclose(loaded.hidden_states(IDS), original.hidden_states(IDS), atol=1e-6)
