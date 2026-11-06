import json
import math

import chz
import pytest
import torch

from alchemy.checkpoint import load_checkpoint
from alchemy.config import AdamWConfig, CLIConfig, DataConfig, ModelConfig, MuonConfig
from alchemy.model import GPT
from alchemy.train import eval_metrics, lr_multiplier, train

# A small corpus in place of SST, so the tests need no download
TEXTS = [f'the film was {a} and the acting was {b}.' for a in ('good', 'bad', 'dull', 'great', 'fine')
         for b in ('superb', 'awful', 'flat', 'warm')] * 5


@pytest.fixture(autouse=True)
def offline_texts(monkeypatch):
    monkeypatch.setattr('alchemy.train.load_texts', lambda datasets, split='train': {name: TEXTS for name in datasets})



def test_lr_warms_up_then_decays():
    """Warmup as a fraction of training or as steps reaches the full lr, then a linear fall to min_lr_ratio starts."""
    fraction, steps = AdamWConfig(warmup=0.1, min_lr_ratio=0.1), AdamWConfig(warmup=10, min_lr_ratio=0.1)
    assert [lr_multiplier(fraction, s, 100) for s in (0, 4, 9, 10)] == pytest.approx([0.1, 0.5, 1.0, 1.0])
    assert lr_multiplier(fraction, 9, 100) == pytest.approx(lr_multiplier(steps, 9, 100))
    assert lr_multiplier(fraction, 100, 100) == pytest.approx(0.1)


def test_eval_bits_per_byte():
    """A model that gives every one of V tokens the same probability loses log V per token. Bits per byte count only
    the targets that are text: their loss in bits over their bytes (here one byte per token, none for <bos>)."""
    model = GPT(ModelConfig(vocab_size=50, max_seq_len=8, hidden_dim=8, num_heads=2))
    torch.nn.init.zeros_(model.lm_head.weight)  # every logit 0
    sizes = torch.ones(50, dtype=torch.long)
    sizes[8] = 0  # token 8 plays <bos>
    batches = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[7, 8]])]  # targets 2 3 5 6 8: four are text
    metrics = eval_metrics(model, {'text': batches}, sizes, torch.autocast('cpu', enabled=False))
    assert metrics['eval/loss'] == pytest.approx(math.log(50))
    assert metrics['eval/bpb'] == metrics['eval/text/bpb'] == pytest.approx(math.log2(50))


def test_train_writes_a_model_and_logs(tmp_path):
    """A short bf16 run with Muon stops at max_steps, logs held-out loss at the end and
    saves a model that loads."""
    config = CLIConfig(batch_size=2, max_steps=4, log_every=2, eval_windows=2, log_dir=str(tmp_path), precision='bf16',
                       data=DataConfig(weights={'sst': 1.0}, vocab_size=300),
                       model=ModelConfig(hidden_dim=8, num_heads=2), optimizer=MuonConfig())
    train(config)
    rows = [json.loads(line) for line in open(tmp_path / 'metrics.jsonl')]
    assert [r['step'] for r in rows if 'train/loss' in r] == [2, 4]
    assert 'eval/sst/bpb' in rows[-1]
    assert rows[-1]['optim/AdamW/lr'] == pytest.approx(0.003 * lr_multiplier(config.optimizer, 3, 4))  # last step's
    assert load_checkpoint(str(tmp_path / 'model.safetensors')).config.hidden_dim == 8


def test_init_from_continues_with_the_checkpoints_tokenizer(tmp_path):
    """Training saves tokenizer.json next to the model; init_from loads both, as two separate files."""
    first = CLIConfig(batch_size=2, max_steps=2, eval_windows=0, log_dir=str(tmp_path / 'a'),
                      data=DataConfig(weights={'sst': 1.0}, vocab_size=300),
                      model=ModelConfig(hidden_dim=8, num_heads=2))
    train(first)
    train(chz.replace(first, log_dir=str(tmp_path / 'b'), init_from=str(tmp_path / 'a' / 'model.safetensors')))
    assert (tmp_path / 'a' / 'tokenizer.json').read_text() == (tmp_path / 'b' / 'tokenizer.json').read_text()


def test_resumed_run_matches_an_uninterrupted_one(tmp_path):
    """A run stopped after a step checkpoint and started again in the same log_dir ends with exactly the weights of a
    run that was never stopped, and keeps its metrics."""
    def run(name: str) -> CLIConfig:
        return CLIConfig(batch_size=2, max_steps=4, save_every=2, log_every=1, eval_windows=0,
                         log_dir=str(tmp_path / name), data=DataConfig(weights={'sst': 1.0}, vocab_size=300),
                         model=ModelConfig(hidden_dim=8, num_heads=2), optimizer=MuonConfig())
    train(run('straight'))
    train(run('stopped'))
    for path in ('model.safetensors', 'step000004.safetensors', 'step000004.pt'):  # as if it stopped after step 2
        (tmp_path / 'stopped' / path).unlink()
    train(run('stopped'))
    straight, resumed = (load_checkpoint(str(tmp_path / n / 'model.safetensors')) for n in ('straight', 'stopped'))
    for (name, a), b in zip(straight.state_dict().items(), resumed.state_dict().values()):
        assert torch.equal(a, b), name
    steps = [json.loads(line)['step'] for line in (tmp_path / 'stopped' / 'metrics.jsonl').read_text().splitlines()]
    assert [s for s in steps if s is not None] == [1, 2, 3, 4]
