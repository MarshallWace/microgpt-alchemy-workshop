# Plot a finished experiment's losses: python3 -m experiments.plot experiment=tune_lr
import chz
import numpy as np
from rich.console import Console
from rich.table import Table

from alchemy.utils.plot import last_metric, plot, plot_landscape
from experiments.common import load_experiment, run_dirs, settings


def plot_sweep(name: str, runs: dict) -> None:
    # The runs' losses on one chart in runs/<name>/plot.png, and each run's final loss
    plot(tuple(runs), 'train/loss', 'progress/epoch', f'runs/{name}/plot.png')
    table = Table(title=name, header_style='bold magenta')
    table.add_column('run')
    table.add_column('final train/loss')
    for log_dir in runs:
        table.add_row(log_dir.split('/')[-1], f'{last_metric(log_dir, 'train/loss'):.3f}')
    Console().print(table)


def plot_grid(name: str, runs: dict, x: str, y: str) -> None:
    # A grid over two settings as a loss landscape in runs/<name>/landscape.png
    values = {log_dir: settings(config) for log_dir, config in runs.items()}
    xs = sorted({v[x] for v in values.values()})
    ys = sorted({v[y] for v in values.values()})
    losses = np.zeros((len(ys), len(xs)))
    for log_dir, v in values.items():
        losses[ys.index(v[y]), xs.index(v[x])] = last_metric(log_dir, 'train/loss')
    plot_landscape(losses, x, tuple(xs), y, tuple(ys), f'runs/{name}/landscape.png')


def main(experiment: str) -> None:
    # experiment: one in this folder, e.g. tune_lr
    experiment = load_experiment(experiment)
    runs = run_dirs(experiment.EXPERIMENT, experiment.build_runs())
    if hasattr(experiment, 'AXES'):
        plot_grid(experiment.EXPERIMENT, runs, *experiment.AXES)
    else:
        plot_sweep(experiment.EXPERIMENT, runs)


if __name__ == '__main__':
    chz.entrypoint(main)
