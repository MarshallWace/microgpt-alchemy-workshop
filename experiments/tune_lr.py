# 2. How big should each update be? Learning rate.
# Train: python3 -m experiments.tune_lr
# Plot:  python3 -m experiments.plot experiment=tune_lr
# Score: python3 -m experiments.score experiment=tune_lr
from experiments.common import train_runs
from microgpt.config import AdamWConfig, CLIConfig

EXPERIMENT = 'tune_lr'
LEARNING_RATES = (0.01, 0.003, 0.001)


def build_runs() -> list[CLIConfig]:
    # Everything else stays at the CLIConfig defaults
    return [CLIConfig(optimizer=AdamWConfig(lr=lr)) for lr in LEARNING_RATES]


if __name__ == '__main__':
    train_runs(EXPERIMENT, build_runs(), workers=3)
