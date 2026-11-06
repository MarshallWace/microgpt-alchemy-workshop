# 1. Can the model learn? Vector size.
# Train: python3 -m experiments.tune_embd
# Plot:  python3 -m experiments.plot experiment=tune_embd
# Score: python3 -m experiments.score experiment=tune_embd
from experiments.common import train_runs
from alchemy.config import CLIConfig, ModelConfig

EXPERIMENT = 'tune_embd'
VECTOR_SIZES = (16, 64, 128)


def build_runs() -> list[CLIConfig]:
    # Everything else stays at the CLIConfig defaults
    return [CLIConfig(model=ModelConfig(hidden_dim=hidden_dim)) for hidden_dim in VECTOR_SIZES]


if __name__ == '__main__':
    train_runs(EXPERIMENT, build_runs(), workers=3)
