import functools
import math
import time
from pathlib import Path

import chz
import torch
from torch.optim.lr_scheduler import LambdaLR

from alchemy.checkpoint import latest_checkpoint, load_checkpoint, load_train_state, save_checkpoint, save_train_state
from alchemy.config import CLIConfig, OptimizerConfig
from alchemy.data import DatasetMix, PackedDataset
from alchemy.load_data import load_texts
from alchemy.model import GPT, loss
from alchemy.tokenizer import BPETokenizer, checkpoint_tokenizer
from alchemy.utils.cli_utils import check_log_dir
from alchemy.utils.ml_log import setup_logging, truncate_metrics

EvalSets = dict[str, list[torch.Tensor]]  # dataset -> batches of held-out windows


def build_tokenizer(config: CLIConfig, texts: dict[str, list[str]]) -> BPETokenizer:
    # data.tokenizer, or with init_from the tokenizer.json in its folder, or a new one trained on the texts
    if config.data.tokenizer:
        return BPETokenizer.load(config.data.tokenizer)
    if config.init_from:
        return checkpoint_tokenizer(config.init_from)
    return BPETokenizer.train([text for name in texts for text in texts[name]], config.data.vocab_size)


def build_model(config: CLIConfig, vocab_size: int) -> GPT:
    # The init_from checkpoint, or a new model
    model = load_checkpoint(config.init_from) if config.init_from else \
        GPT(chz.replace(config.model, vocab_size=vocab_size))
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


def eval_sets(config: CLIConfig, tokenizer: BPETokenizer, max_seq_len: int) -> EvalSets:
    # For each dataset: about eval_windows windows from the start of its dev split, packed as in training, in batches
    sets = {}
    for name, texts in load_texts(config.data.weights, 'dev').items():
        dev = PackedDataset(tokenizer, texts, max_seq_len, max_tokens=config.eval_windows * max_seq_len)
        windows, _ = dev.windows()
        sets[name] = [batch.to(config.device) for batch in windows.split(config.batch_size)]
    return sets


def token_bytes(tokenizer: BPETokenizer, device: str) -> torch.Tensor:
    # Each token's size in bytes, 0 for <bos>, which is not text
    sizes = torch.tensor([len(token) for token in tokenizer.vocab])
    sizes[tokenizer.bos_token_id] = 0
    return sizes.to(device)


@torch.no_grad()
def eval_metrics(model: GPT, sets: EvalSets, sizes: torch.Tensor, autocast: torch.autocast) -> dict[str, float]:
    # Loss per token; and bits per byte (comparable across tokenizers): the loss on the targets that are text, over
    # their bytes (sizes: each token's, as nanochat does). For each dataset, and over all of them.
    def summary(prefix: str, nats: float, targets: float, text_nats: float, num_bytes: float) -> dict[str, float]:
        return {f'{prefix}/loss': nats / targets, f'{prefix}/bpb': text_nats / num_bytes / math.log(2)}

    metrics, totals = {}, torch.zeros(4, dtype=torch.float64)  # nats, targets, nats on text, bytes
    for name, batches in sets.items():
        sums = torch.zeros(4, dtype=torch.float64)
        for batch in batches:
            with autocast:
                losses = loss(model, batch, reduction='none')
            num_bytes = sizes[batch[:, 1:]]
            sums += torch.stack([losses.sum(), losses.new_tensor(losses.numel()), (losses * (num_bytes > 0)).sum(),
                                 num_bytes.sum().float()]).double().cpu()
        metrics.update(summary(f'eval/{name}', *sums.tolist()))
        totals += sums
    return {**summary('eval', *totals.tolist()), **metrics}


def train_step(model: GPT, optimizers: list[torch.optim.Optimizer], batch: torch.Tensor, max_grad_norm: float,
               autocast: torch.autocast) -> float:
    with autocast:
        batch_loss = loss(model, batch)
    batch_loss.backward()

    if max_grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad()
    return batch_loss.item()


def train(config: CLIConfig) -> None:
    # Logging (resuming from the newest step checkpoint in log_dir, if there is one), seeds and threads
    resume = latest_checkpoint(config.log_dir)
    if resume:
        truncate_metrics(config.log_dir, int(Path(resume).name.removeprefix('step')))
    logger = setup_logging(config.log_dir, config)
    torch.manual_seed(config.seed)
    if config.threads:
        torch.set_num_threads(config.threads)

    # Tokenizer, model and training data
    texts = load_texts(config.data.weights)
    tokenizer = build_tokenizer(config, texts)
    tokenizer.save(f'{config.log_dir}/tokenizer.json')
    model = build_model(config, tokenizer.vocab_size).to(config.device)
    if config.compile:
        model.compile()
    data = DatasetMix(tokenizer, texts, config.data.weights, model.config.max_seq_len, config.seed)

    # How long to train
    total_steps = num_steps(config, data)
    tokens_per_step = config.batch_size * model.config.max_seq_len  # every window has max_seq_len tokens
    logger.log_metrics({'data/num_windows': data.num_windows, 'data/num_tokens': data.num_tokens,
                        'model/vocab_size': tokenizer.vocab_size, 'train/num_steps': total_steps,
                        'model/num_params': sum(p.numel() for p in model.parameters())})

    # Optimizers, learning-rate schedule, precision and held-out text
    optimizers = config.optimizer.build(model)
    schedule = functools.partial(lr_multiplier, config.optimizer, total_steps=total_steps)
    schedulers = [LambdaLR(optimizer, schedule) for optimizer in optimizers]
    autocast = torch.autocast(torch.device(config.device).type, torch.bfloat16, enabled=config.precision == 'bf16')
    evals = eval_sets(config, tokenizer, model.config.max_seq_len) if config.eval_windows else {}
    sizes = token_bytes(tokenizer, config.device)
    done = load_train_state(resume, model, optimizers, schedulers) if resume else 0

    # Training loop
    losses, start, last_log = [], time.time(), time.time()
    for step in range(done + 1, total_steps + 1):  # step: the number of steps done after this one
        # Training step
        batch = data.batch(step - 1, config.batch_size).to(config.device)
        losses.append(train_step(model, optimizers, batch, config.optimizer.max_grad_norm, autocast))

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

        # Move to the next step's learning rate, and save a checkpoint to resume from
        for scheduler in schedulers:
            scheduler.step()
        if config.save_every > 0 and step % config.save_every == 0:
            save_train_state(f'{config.log_dir}/step{step:06d}', step, model, optimizers, schedulers)

    # Save the final model
    save_checkpoint(f'{config.log_dir}/model.safetensors', model)
    logger.log_text(f'saved model to {config.log_dir}/model.safetensors')


if __name__ == '__main__':
    cli_config = chz.entrypoint(CLIConfig)
    check_log_dir(cli_config.log_dir, cli_config.behavior_if_log_dir_exists)
    train(cli_config)
