# Score every run of a finished experiment with the linear probe, and plot the scores against the final losses:
# python3 -m experiments.score experiment=tune_embd
from typing import Any

import chz
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from experiments.common import load_experiment, run_dirs
from microgpt.probe import checkpoint_featurizer, evaluate_probe
from microgpt.utils.plot import CMAP, last_metric, save, style

RANDOM_WEIGHTS = 53.7  # SST test accuracy (%) of the default model before any training


def plot_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, colour: Any) -> None:
    # A quadratic fit, with a 95% confidence band from refitting on 1,000 resamples of the runs
    grid = np.linspace(x.min(), x.max(), 100)
    rng = np.random.default_rng(0)
    fits = np.array([np.polyval(np.polyfit(x[i], y[i], 2), grid) for i in rng.integers(0, len(x), (1000, len(x)))])
    ax.plot(grid, np.polyval(np.polyfit(x, y, 2), grid), color=colour, linewidth=1.8, zorder=2)
    ax.fill_between(grid, *np.percentile(fits, [2.5, 97.5], axis=0), color=colour, alpha=0.2, linewidth=0, zorder=1)


def mark_random_weights(ax: plt.Axes, accuracy: float = RANDOM_WEIGHTS) -> None:
    ax.axhline(accuracy, color='grey', linestyle='--', linewidth=1)
    ax.text(0.99, accuracy + 0.2, 'random weights', color='grey', ha='right', va='bottom', fontsize=9,
            transform=ax.get_yaxis_transform())


def plot_scores(runs: list[str], losses: list[float], dev: list[float], test: list[float], out: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sweep, x = len(runs) <= 6, np.array(losses)
    for scores, label, marker, colour in ((dev, 'dev', 'o', CMAP(0.0)), (test, 'test', 's', CMAP(1.0))):
        y = 100 * np.array(scores)
        if sweep:
            ax.plot(x, y, label=label, marker=marker, markersize=7, linewidth=1.8, color=colour, zorder=3)
        else:
            ax.scatter(x, y, marker=marker, s=28, color=colour, alpha=0.6, label=label, zorder=3)
            plot_fit(ax, x, y, colour)
    if sweep:
        for run, loss, accuracy in zip(runs, losses, dev):
            ax.annotate(run, (loss, 100 * accuracy), textcoords='offset points', xytext=(6, 6), fontsize=9)
    accuracies = 100 * np.array([*dev, *test])
    ax.set_ylim(min(accuracies.min(), RANDOM_WEIGHTS) - 2, accuracies.max() + 2)  # the data and the baseline, not the band's tails
    mark_random_weights(ax)
    ax.invert_xaxis()
    style(ax, 'final train/loss (lower is better)', 'linear probe accuracy (%)')
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    save(fig, out)


def main(experiment: str) -> None:
    # experiment: one in this folder, e.g. tune_embd
    experiment = load_experiment(experiment)
    table = Table(title=experiment.EXPERIMENT, header_style='bold magenta')
    for column in ('run', 'final train/loss', 'dev', 'test'):
        table.add_column(column)
    rows = []  # (run, final loss, dev accuracy, test accuracy)
    for log_dir, config in run_dirs(experiment.EXPERIMENT, experiment.build_runs()).items():
        run = log_dir.split('/')[-1]
        scores = evaluate_probe(checkpoint_featurizer(f'{config.log_dir}/model.safetensors'))
        rows.append((run, last_metric(log_dir, 'train/loss'), scores['dev_acc'], scores['test_acc']))
        table.add_row(run, f'{rows[-1][1]:.3f}', f'{rows[-1][2]:.1%}', f'{rows[-1][3]:.1%}')
    Console().print(table)
    plot_scores(*map(list, zip(*rows)), f'runs/{experiment.EXPERIMENT}/scores.png')


if __name__ == '__main__':
    chz.entrypoint(main)
