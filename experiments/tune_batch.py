# 3. How much text per step? Batch size. Every run trains for the same number of epochs, so reads the
# same amount of text.
# Train: python3 -m experiments.tune_batch
# Plot:  python3 -m experiments.plot experiment=tune_batch
# Score: python3 -m experiments.score experiment=tune_batch
from experiments.common import train_runs
from microgpt.config import CLIConfig

EXPERIMENT = 'tune_batch'
BATCH_SIZES = (8, 32, 128)


def build_runs() -> list[CLIConfig]:
    # Everything else stays at the CLIConfig defaults
    return [CLIConfig(batch_size=batch_size) for batch_size in BATCH_SIZES]


if __name__ == '__main__':
    train_runs(EXPERIMENT, build_runs(), workers=3)
