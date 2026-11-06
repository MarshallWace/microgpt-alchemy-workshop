# Reading a run's metrics.jsonl, and plotting runs' metrics: against the step, or over a grid of two settings
import json
from pathlib import Path

import chz
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

CMAP = LinearSegmentedColormap.from_list('step_law', ['#19516b', '#6e5e9c', '#cf6294', '#fa8c6c', '#ffae19'])


def style(ax: plt.Axes, x: str, y: str) -> None:
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.spines[['top', 'right']].set_visible(False)


def save(fig: plt.Figure, out: str) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'saved {out}')


def read_metrics(log_dir: str) -> list[dict]:
    with open(Path(log_dir) / 'metrics.jsonl') as f:
        return [json.loads(line) for line in f]


def last_metric(log_dir: str, metric: str) -> float:
    # The metric's last logged value, or nan if the run has none (yet)
    if not (Path(log_dir) / 'metrics.jsonl').exists():
        return float('nan')
    values = [row[metric] for row in read_metrics(log_dir) if row.get(metric) is not None]
    return values[-1] if values else float('nan')


def plot(log_dirs: tuple[str, ...], metric: str = 'train/loss', x: str = 'step', out: str = 'plot.png') -> None:
    # Runs are coloured in the order given, so list a sweep in order (e.g. small to large)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, log_dir in enumerate(log_dirs):
        rows = [row for row in read_metrics(log_dir) if row.get(x) is not None and metric in row]
        ax.plot([row[x] for row in rows], [row[metric] for row in rows], marker='o', markersize=4, linewidth=1.8,
                color=CMAP(i / max(1, len(log_dirs) - 1)), label=log_dir)
    style(ax, x, metric)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    save(fig, out)


def plot_landscape(losses: np.ndarray, x_name: str, x_values: tuple, y_name: str, y_values: tuple, out: str,
                   metric: str = 'final train/loss') -> None:
    # A grid of runs that vary two settings. losses: [len(y_values), len(x_values)] of a metric where lower is better;
    # colour = how far each run is above the best one.
    above_best = 100 * (losses / losses.min() - 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    # Log colour scale: plenty of contrast near the best, but the worst runs still look different
    norm = LogNorm(vmin=0.5, vmax=above_best.max())
    image = ax.imshow(np.maximum(above_best, 0.5), origin='lower', cmap=CMAP, norm=norm)
    for (i, j), loss in np.ndenumerate(losses):
        dark = norm(max(above_best[i, j], 0.5)) < 0.5
        ax.text(j, i, f'{loss:.2f}', ha='center', va='center', color='white' if dark else 'black', zorder=3)
    best_i, best_j = np.unravel_index(losses.argmin(), losses.shape)
    ax.plot(best_j, best_i, marker='*', markersize=48, markerfacecolor='none', markeredgecolor='white',
            markeredgewidth=2, alpha=0.4, zorder=2)
    ax.set_xticks(range(len(x_values)), x_values)
    ax.set_yticks(range(len(y_values)), y_values)
    style(ax, x_name, y_name)
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100) if t <= above_best.max()]
    fig.colorbar(image, label=f'% above the best {metric}', ticks=ticks, format='%g%%')
    save(fig, out)


if __name__ == '__main__':
    # python3 -m alchemy.utils.plot log_dirs=runs/pretrain,runs/long x=progress/epoch
    chz.entrypoint(plot)
