# A checkpoint is a .safetensors file: each weight matrix by name, plus the backend and model config as text. The
# tokenizer is saved separately, as tokenizer.json in the same folder, and a step checkpoint also has a .pkl file of the
# training state, to resume from.
import json
import pickle
from pathlib import Path

import chz
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from microgpt.config import BACKENDS, ModelConfig
from microgpt.optim import Optimizer


def save_checkpoint(path: str, model, backend: str) -> None:
    metadata = {'backend': backend, 'config': json.dumps(chz.asdict(model.config))}
    save_file(BACKENDS[backend].state_dict(model), path, metadata)


def read_metadata(path: str) -> dict[str, str]:
    with safe_open(path, 'np') as f:
        return f.metadata()


def load_checkpoint(path: str, backend: str | None = None):
    # The checkpoint's model, on the backend it was trained with unless another is given (python checkpoints run on
    # numpy, the same model but much faster)
    metadata = read_metadata(path)
    backend = backend or {'python': 'numpy'}.get(metadata['backend'], metadata['backend'])
    model = BACKENDS[backend].GPT(ModelConfig(**json.loads(metadata['config'])))
    BACKENDS[backend].load_state_dict(model, load_file(path))
    return model


def save_train_state(checkpoint: str, step: int, model, optimizers: list[Optimizer], backend: str) -> None:
    # A step checkpoint to resume from: the weights (checkpoint.safetensors), and the step and the optimizers' state
    # (checkpoint.pkl). The data and the learning rate need nothing: they follow from the seed and the step, and nothing
    # else in training is random. Weights are saved as float32, so a python backend run (float64) resumes from weights
    # rounded to float32.
    save_checkpoint(f'{checkpoint}.safetensors', model, backend)
    with open(f'{checkpoint}.pkl', 'wb') as f:
        pickle.dump({'step': step, 'optimizers': [o.state_dict() for o in optimizers]}, f)


def load_train_state(checkpoint: str, model, optimizers: list[Optimizer], backend: str) -> int:
    # Restores what save_train_state saved, and returns the step to continue after
    BACKENDS[backend].load_state_dict(model, load_file(f'{checkpoint}.safetensors'))
    with open(f'{checkpoint}.pkl', 'rb') as f:
        state = pickle.load(f)
    for optimizer, saved in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(saved)
    return state['step']


def latest_checkpoint(log_dir: str) -> str | None:
    # The newest step checkpoint in log_dir (without its suffix), if any
    states = sorted(Path(log_dir).glob('step*.pkl'))
    return str(states[-1].with_suffix('')) if states else None
