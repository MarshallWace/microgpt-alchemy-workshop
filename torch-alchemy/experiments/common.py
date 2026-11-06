import importlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import chz

from alchemy.train import train as train_alchemy
from alchemy.utils.cli_utils import check_log_dir


def settings(config: Any) -> dict[str, Any]:
    flat = {}
    for key in chz.chz_fields(config):
        value = getattr(config, f'X_{key}', getattr(config, key))  # as set, before chz fills in defaults
        if chz.is_chz(value):
            flat[key] = type(value).__name__
            flat.update(settings(value))
        else:
            flat[key] = value
    return flat


NOT_IN_NAMES = ('log_dir', 'log_every', 'eval_every', 'eval_windows', 'save_every')


def run_names(configs: list[Any]) -> list[str]:
    # Named by the settings that differ between the runs, e.g. lr=0.01,batch_size=8
    runs = [settings(config) for config in configs]
    keys = list(dict.fromkeys(key for run in runs for key in run if key not in NOT_IN_NAMES))
    varying = [key for key in keys if len({repr(run[key]) for run in runs if key in run}) > 1]
    show = lambda value: '+'.join(f'{k}:{v:g}' for k, v in value.items()) if isinstance(value, dict) else value
    return [','.join(f'{key}={show(run[key])}' for key in varying if key in run) or 'default' for run in runs]


def run_dirs(name: str, configs: list[Any], names: list[str] | None = None) -> dict[str, Any]:
    # runs/<name>/<run name>; runs are named by run_names unless names are given
    dirs = [f'runs/{name}/{run}' for run in names or run_names(configs)]
    return {d: chz.replace(config, log_dir=d)
            for d, config in zip(dirs, configs)}


def train_runs(name: str, configs: list[Any], workers: int = 1, train: Callable[[Any], None] = train_alchemy,
               names: list[str] | None = None) -> None:
    # Each run starts afresh: an earlier run in its log_dir is deleted, as tinker's sweeps do
    runs = run_dirs(name, configs, names)
    for log_dir in runs:
        check_log_dir(log_dir, 'delete')
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(train, runs.values()))


def load_experiment(name: str):
    return importlib.import_module(f'experiments.{name}')
