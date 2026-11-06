# Learning rate and batch size affect each other, so search both together, as in Step Law
# (https://arxiv.org/abs/2503.04715): a grid of runs that all read the same text.
# Train: python3 -m experiments.tune_lr_batch
# Plot:  python3 -m experiments.plot experiment=tune_lr_batch
# Score: python3 -m experiments.score experiment=tune_lr_batch
from experiments.common import train_runs
from alchemy.config import AdamWConfig, CLIConfig

EXPERIMENT = 'tune_lr_batch'
LEARNING_RATES = (0.001, 0.002, 0.004, 0.008, 0.016)
BATCH_SIZES = (8, 16, 32, 64, 128)
AXES = ('lr', 'batch_size')  # plotted as a loss landscape


def build_runs() -> list[CLIConfig]:
    # Everything else stays at the CLIConfig defaults
    return [CLIConfig(optimizer=AdamWConfig(lr=lr), batch_size=batch_size)
            for lr in LEARNING_RATES for batch_size in BATCH_SIZES]


if __name__ == '__main__':
    train_runs(EXPERIMENT, build_runs(), workers=4)  # more at once can freeze a laptop
