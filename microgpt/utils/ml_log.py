# From Thinking Machines Lab's Tinker Cookbook (Apache-2.0; see THIRD_PARTY_NOTICES.md)
# Source: https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/utils/ml_log.py
# (only the console and JSON loggers), and truncate_metrics as in tinker_cookbook/stores/training_store.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import json

import chz
from rich.console import Console
from rich.table import Table


def _maybe_truncate_repr(value: Any) -> str:
    repr_value = repr(value)
    if len(repr_value) > 256:
        return repr_value[:128] + " ... " + repr_value[-128:]
    return repr_value


class Logger(ABC):
    """What every logger can log: the config, metrics at a step and text."""

    @abstractmethod
    def log_hparams(self, config: Any) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    def log_text(self, text: str) -> None:
        pass


class PrettyPrintLogger(Logger):
    """Displays the config and metrics as Rich tables in the console."""

    def __init__(self):
        self.console = Console()

    def log_hparams(self, config: Any) -> None:
        self.console.print("[bold cyan]Configuration:[/bold cyan]")
        for key, value in chz.asdict(config).items():
            self.console.print(f"  {key}: {_maybe_truncate_repr(value)}")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not metrics:
            return
        metric_col_width = min(max(max(len(k) for k in metrics), 20), 60)
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=metric_col_width)
        table.add_column("Value", style="green")
        if step is not None:
            table.title = f"Step {step}"
        for key, value in sorted(metrics.items()):
            table.add_row(key, f"{value:.6f}" if isinstance(value, float) else str(value))
        self.console.print(table)

    def log_text(self, text: str) -> None:
        self.console.print(text)


class JsonLogger(Logger):
    """Writes the config to config.json and appends one line per log_metrics call to metrics.jsonl."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_hparams(self, config: Any) -> None:
        (self.log_dir / "config.json").write_text(json.dumps(chz.asdict(config, include_type=True), indent=2))

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not metrics:
            return
        with open(self.log_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")


def truncate_metrics(log_dir: str, step: int) -> None:
    """Drops the metrics.jsonl rows of steps after step (and rows without a step), before a run resumes from step."""
    path = Path(log_dir) / "metrics.jsonl"
    rows = [line for line in path.read_text().splitlines() if (json.loads(line)["step"] or step + 1) <= step]
    path.write_text("".join(f"{row}\n" for row in rows))


class MultiplexLogger(Logger):
    """Forwards every call to each of its loggers."""

    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers

    def log_hparams(self, config: Any) -> None:
        for logger in self.loggers:
            logger.log_hparams(config)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def log_text(self, text: str) -> None:
        for logger in self.loggers:
            logger.log_text(text)


def setup_logging(log_dir: str, config: Any) -> MultiplexLogger:
    ml_logger = MultiplexLogger([JsonLogger(log_dir), PrettyPrintLogger()])
    ml_logger.log_hparams(config)
    return ml_logger
