# A checkpoint is a .safetensors file: each weight by name, plus the model config as text. The tokenizer is saved
# separately, as tokenizer.json in the same folder, and a step checkpoint also has a .pt file of the training state, to
# resume from.
import json
from pathlib import Path

import chz
import torch
from safetensors import safe_open
from safetensors.torch import load_model, save_model

from alchemy.config import ModelConfig
from alchemy.model import GPT


def save_checkpoint(path: str, model: GPT) -> None:
    # A tied output layer is saved once, as model.embed_tokens.weight
    save_model(model, path, {'config': json.dumps(chz.asdict(model.config))})


def load_checkpoint(path: str) -> GPT:
    with safe_open(path, 'pt') as f:
        model = GPT(ModelConfig(**json.loads(f.metadata()['config'])))
    load_model(model, path)
    return model


def save_train_state(checkpoint: str, step: int, model: GPT, optimizers: list, schedulers: list) -> None:
    # A step checkpoint to resume from: the weights (checkpoint.safetensors), and the step, the optimizers' and lr
    # schedules' state and the random state (checkpoint.pt). The data needs nothing: each batch follows from the seed
    # and the step.
    save_checkpoint(f'{checkpoint}.safetensors', model)
    torch.save({'step': step, 'optimizers': [o.state_dict() for o in optimizers],
                'schedulers': [s.state_dict() for s in schedulers], 'rng': torch.get_rng_state()}, f'{checkpoint}.pt')


def load_train_state(checkpoint: str, model: GPT, optimizers: list, schedulers: list) -> int:
    # Restores what save_train_state saved, and returns the step to continue after
    load_model(model, f'{checkpoint}.safetensors')
    state = torch.load(f'{checkpoint}.pt', weights_only=True)
    for optimizer, saved in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(saved)
    for scheduler, saved in zip(schedulers, state['schedulers']):
        scheduler.load_state_dict(saved)
    torch.set_rng_state(state['rng'])
    return state['step']


def latest_checkpoint(log_dir: str) -> str | None:
    # The newest step checkpoint in log_dir (without its suffix), if any
    states = sorted(Path(log_dir).glob('step*.pt'))
    return str(states[-1].with_suffix('')) if states else None
