import math
import time
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import ModuleType

import chz
import numpy as np

from microgpt.checkpoint import latest_checkpoint, load_checkpoint, load_train_state, save_checkpoint, save_train_state
from microgpt.config import BACKENDS, CLIConfig, OptimizerConfig
from microgpt.data import DatasetMix, PackedDataset
from microgpt.load_data import load_texts
from microgpt.optim import Optimizer
from microgpt.sampling import log_softmax
from microgpt.tokenizer import TOKENIZERS, Tokenizer, checkpoint_tokenizer
from microgpt.utils.cli_utils import check_log_dir
from microgpt.utils.ml_log import setup_logging, truncate_metrics

EvalSets = dict[str, np.ndarray]  # dataset -> held-out windows [num_windows, max_seq_len]


def build_tokenizer(config: CLIConfig, texts: dict[str, list[str]]) -> Tokenizer:
    # With init_from, the tokenizer.json in its folder; or a new data.tokenizer trained on the texts
    if config.init_from:
        return checkpoint_tokenizer(config.init_from)
    return TOKENIZERS[config.data.tokenizer].train([t for name in texts for t in texts[name]], config.data.vocab_size)


def build_model(config: CLIConfig, backend: ModuleType, vocab_size: int):
    # The init_from checkpoint, or a new model
    model = load_checkpoint(config.init_from, config.backend) if config.init_from else \
        backend.GPT(chz.replace(config.model, vocab_size=vocab_size))
    assert model.config.vocab_size == vocab_size, 'the tokenizer does not match the model'
    return model


def num_steps(config: CLIConfig, data: DatasetMix) -> int:
    # num_epochs of the data, or max_steps if that is sooner
    steps = max(1, round(config.num_epochs * data.num_windows / config.batch_size))
    return steps if config.max_steps is None else min(steps, config.max_steps)


def lr_multiplier(optimizer: OptimizerConfig, step: int, total_steps: int) -> float:
    # The peak lr is multiplied by this at step 0, 1, ...: it rises linearly over the warmup (a fraction of training
    # up to 1, or a number of steps), then falls with lr_schedule to min_lr_ratio over the steps after it
    warmup_steps = optimizer.warmup * total_steps if optimizer.warmup <= 1 else optimizer.warmup
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    done = (step - warmup_steps) / max(1.0, total_steps - warmup_steps)
    decay = {'linear': 1 - done, 'cosine': (1 + math.cos(math.pi * done)) / 2, 'constant': 1.0}[optimizer.lr_schedule]
    return optimizer.min_lr_ratio + (1 - optimizer.min_lr_ratio) * decay


def eval_sets(config: CLIConfig, tokenizer: Tokenizer, max_seq_len: int) -> EvalSets:
    # For each dataset: about eval_windows windows from the start of its dev split, packed as in training
    max_tokens = config.eval_windows * max_seq_len
    return {name: PackedDataset(tokenizer, texts, max_seq_len, max_tokens).windows()[0]
            for name, texts in load_texts(config.data.weights, 'dev').items()}


def eval_metrics(model, sets: EvalSets, sizes: np.ndarray,
                 autocast: AbstractContextManager = nullcontext()) -> dict[str, float]:
    # Loss per token; and bits per byte (comparable across tokenizers): the loss on the targets that are text, over
    # their bytes (sizes: each token's, as nanochat does). For each dataset, and over all of them.
    def summary(prefix: str, nats: float, targets: float, text_nats: float, num_bytes: float) -> dict[str, float]:
        return {f'{prefix}/loss': nats / targets, f'{prefix}/bpb': text_nats / num_bytes / math.log(2)}

    metrics, totals = {}, np.zeros(4)  # nats, targets, nats on text, bytes
    for name, windows in sets.items():
        sums = np.zeros(4)
        for window in windows:  # one at a time: every backend's model.logits reads one token list
            with autocast:
                logprobs = log_softmax(model.logits(window[:-1].tolist()))
            losses, num_bytes = -logprobs[np.arange(len(window) - 1), window[1:]], sizes[window[1:]]
            sums += [losses.sum(), losses.size, (losses * (num_bytes > 0)).sum(), num_bytes.sum()]
        metrics.update(summary(f'eval/{name}', *sums))
        totals += sums
    return {**summary('eval', *totals), **metrics}


def train_step(backend: ModuleType, model, optimizers: list[Optimizer], batch: np.ndarray, max_grad_norm: float,
               autocast: AbstractContextManager) -> float:
    with autocast:
        loss = backend.loss(model, batch)
    loss.backward()

    if max_grad_norm:
        backend.clip_grad_norm(model.parameters(), max_grad_norm)
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad()
    return float(loss.data)


def train(config: CLIConfig) -> None:
    # Logging (resuming from the newest step checkpoint in log_dir, if there is one), seeds and threads
    resume = latest_checkpoint(config.log_dir)
    if resume:
        truncate_metrics(config.log_dir, int(Path(resume).name.removeprefix('step')))
    logger = setup_logging(config.log_dir, config)
    backend = BACKENDS[config.backend]
    backend.seed(config.seed)
    backend.set_threads(config.threads)

    # Tokenizer, model and training data
    texts = load_texts(config.data.weights)
    tokenizer = build_tokenizer(config, texts)
    tokenizer.save(f'{config.log_dir}/tokenizer.json')
    model = build_model(config, backend, tokenizer.vocab_size)
    autocast = backend.set_precision(model, config.precision.params, config.precision.compute)
    if config.compile:
        assert backend.SUPPORTS_COMPILE, f'backend={config.backend} cannot compile'
        model.compile()
    data = DatasetMix(tokenizer, texts, config.data.weights, model.config.max_seq_len, config.seed)

    # How long to train
    total_steps = num_steps(config, data)
    tokens_per_step = config.batch_size * model.config.max_seq_len  # every window has max_seq_len tokens
    logger.log_metrics({'data/num_windows': data.num_windows, 'data/num_tokens': data.num_tokens,
                        'model/vocab_size': tokenizer.vocab_size, 'train/num_steps': total_steps,
                        'model/num_params': sum(w.size for w in backend.state_dict(model).values())})

    # Optimizers (each group's peak lr kept as initial_lr, for the schedule) and held-out text
    optimizers = config.optimizer.build(backend, model)
    for group in (group for optimizer in optimizers for group in optimizer.param_groups):
        group['initial_lr'] = group['lr']
    evals = eval_sets(config, tokenizer, model.config.max_seq_len) if config.eval_windows else {}
    sizes = np.array(tokenizer.token_bytes())
    done = load_train_state(resume, model, optimizers, config.backend) if resume else 0

    # Training loop
    losses, start, last_log = [], time.time(), time.time()
    for step in range(done + 1, total_steps + 1):  # step: the number of steps done after this one
        # Training step, at this step's learning rate
        lr_scale = lr_multiplier(config.optimizer, step - 1, total_steps)
        for group in (group for optimizer in optimizers for group in optimizer.param_groups):
            group['lr'] = group['initial_lr'] * lr_scale
        batch = data.batch(step - 1, config.batch_size)
        losses.append(train_step(backend, model, optimizers, batch, config.optimizer.max_grad_norm, autocast))

        # Log metrics, with held-out loss when it is due
        last = step == total_steps
        evaluate = bool(evals) and (last or config.eval_every > 0 and step % config.eval_every == 0)
        if last or evaluate or step % config.log_every == 0:
            now = time.time()
            metrics = eval_metrics(model, evals, sizes, autocast) if evaluate else {}
            logger.log_metrics(metrics | {
                'train/loss': sum(losses) / len(losses),
                'progress/done_frac': step / total_steps,
                'progress/epoch': step * tokens_per_step / data.num_tokens,
                'throughput/total_tokens': step * tokens_per_step,
                'throughput/tokens_per_second': len(losses) * tokens_per_step / (now - last_log),
                'time/elapsed': now - start,
                **{f'optim/{type(o).__name__}/lr': o.param_groups[0]['lr'] for o in optimizers},
            }, step)
            losses, last_log = [], time.time()

        # Save a checkpoint to resume from
        if config.save_every > 0 and step % config.save_every == 0:
            save_train_state(f'{config.log_dir}/step{step:06d}', step, model, optimizers, config.backend)

    # Save the final model
    save_checkpoint(f'{config.log_dir}/model.safetensors', model, config.backend)
    logger.log_text(f'saved model to {config.log_dir}/model.safetensors')


if __name__ == '__main__':
    cli_config = chz.entrypoint(CLIConfig)
    check_log_dir(cli_config.log_dir, cli_config.behavior_if_log_dir_exists)
    train(cli_config)
