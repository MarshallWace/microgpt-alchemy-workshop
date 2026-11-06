# Scores a model: the linear probe's accuracy on SST (alchemy/probe.py)
#   python3 -m scripts.linear_probe model=runs/pretrain/model.safetensors                # an alchemy checkpoint
#   python3 -m scripts.linear_probe model=my_gpt.featurize  # your model: featurize(texts) -> array [texts, features]
import importlib
from pathlib import Path

import chz

from alchemy.probe import evaluate_probe, checkpoint_featurizer


@chz.chz
class ProbeConfig:
    model: str              # a checkpoint (runs/pretrain/model.safetensors) or your own module.featurize
    dataset: str = 'sst'    # labelled texts to probe (DATASETS in the data module)


def probe(config: ProbeConfig) -> dict:
    if config.model.endswith('.safetensors'):
        assert Path(config.model).exists(), f'{config.model} does not exist: train it first'
        featurize = checkpoint_featurizer(config.model)
    else:
        module, function = config.model.rsplit('.', 1)
        featurize = getattr(importlib.import_module(module), function)
    return evaluate_probe(featurize, config.dataset)


if __name__ == '__main__':
    probe(chz.entrypoint(ProbeConfig))
