# Sampling from a trained model: SamplingParams, sample() returning a SampleResponse of SampledSequences, and
# compute_logprobs(). Prompts are lists of token ids.
from dataclasses import dataclass
from typing import Literal

import chz
import torch
from torch.nn import functional as F

from alchemy.model import GPT

MAX_TOKENS = 256  # when max_tokens is None


@chz.chz
class SamplingParams:
    max_tokens: int | None = None    # None: until a stop token (at most MAX_TOKENS)
    seed: int | None = None          # for reproducible samples
    stop: tuple[int, ...] | None = None  # token ids that end a sample; None: the end of a text; (): never
    temperature: float = 1.0         # 0: always the most likely token
    top_k: int = -1                  # sample from the k most likely tokens only; -1: all
    top_p: float = 1.0               # sample from the most likely tokens that cover p of the probability


@dataclass(frozen=True)
class SampledSequence:
    stop_reason: Literal['length', 'stop']
    tokens: list[int]
    logprobs: list[float]            # the model's log-probability of each sampled token


@dataclass(frozen=True)
class SampleResponse:
    sequences: list[SampledSequence]


@torch.no_grad()
def logprobs(model: GPT, ids: list[int]) -> torch.Tensor:
    # Each position's log-probabilities of the next token [len(ids), vocab_size]
    return F.log_softmax(model(torch.tensor([ids], device=model.device))[0].cpu().double(), -1)


def next_token(logprob: torch.Tensor, params: SamplingParams, generator: torch.Generator) -> int:
    if params.temperature == 0:
        return int(logprob.argmax())
    probs, order = F.softmax(logprob / params.temperature, -1).sort(descending=True)
    keep = probs[:params.top_k] if params.top_k > 0 else probs
    kept = keep.cumsum(0) / keep.sum()  # top-p over what top-k kept, renormalised (as Hugging Face)
    keep = keep[:1 + int(torch.searchsorted(kept, params.top_p))]
    return int(order[torch.multinomial(keep, 1, generator=generator)])


def sample(model: GPT, prompt: list[int], num_samples: int, sampling_params: SamplingParams,
           end_of_text: int | None = None) -> SampleResponse:
    # end_of_text: the token id that ends a text, used when sampling_params.stop is None
    params, generator = sampling_params, torch.Generator()
    generator.seed() if params.seed is None else generator.manual_seed(params.seed)
    stop = set(params.stop if params.stop is not None else [end_of_text] if end_of_text is not None else [])
    sequences = []
    for _ in range(num_samples):
        ids, tokens, token_logprobs, reason = list(prompt), [], [], 'length'
        while len(tokens) < (params.max_tokens or MAX_TOKENS):
            logprob = logprobs(model, ids[-model.config.max_seq_len:])[-1]
            token = next_token(logprob, params, generator)
            if token in stop:
                reason = 'stop'
                break
            ids.append(token)
            tokens.append(token)
            token_logprobs.append(float(logprob[token]))
        sequences.append(SampledSequence(reason, tokens, token_logprobs))
    return SampleResponse(sequences)


def compute_logprobs(model: GPT, prompt: list[int]) -> list[float | None]:
    # The log-probability of each token given up to max_seq_len tokens before it, as when sampling (None for the first)
    n = model.config.max_seq_len
    rows = list(logprobs(model, prompt[:n]))[:len(prompt) - 1]
    rows += [logprobs(model, prompt[k - n:k])[-1] for k in range(n + 1, len(prompt))]
    return [None] + [float(row[token]) for row, token in zip(rows, prompt[1:])]
