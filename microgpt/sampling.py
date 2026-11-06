# Sampling from a trained model: SamplingParams, sample() returning a SampleResponse of SampledSequences, and
# compute_logprobs(). Prompts are lists of token ids.
import random
from dataclasses import dataclass
from typing import Literal

import chz
import numpy as np

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


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits, -1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


def next_token(logprobs: np.ndarray, params: SamplingParams, rng: random.Random) -> int:
    if params.temperature == 0:
        return int(np.argmax(logprobs))
    probs = np.exp(log_softmax(logprobs / params.temperature))
    order = np.argsort(-probs)
    keep = order[:params.top_k] if params.top_k > 0 else order
    kept = np.cumsum(probs[keep]) / probs[keep].sum()  # top-p over what top-k kept, renormalised (as Hugging Face)
    keep = keep[:1 + int(np.searchsorted(kept, params.top_p))]
    return int(rng.choices(keep, weights=probs[keep])[0])


def sample(model, prompt: list[int], num_samples: int, sampling_params: SamplingParams,
           end_of_text: int | None = None) -> SampleResponse:
    # end_of_text: the token id that ends a text, used when sampling_params.stop is None
    params, rng = sampling_params, random.Random(sampling_params.seed)
    stop = set(params.stop if params.stop is not None else [end_of_text] if end_of_text is not None else [])
    sequences = []
    for _ in range(num_samples):
        ids, tokens, logprobs, reason = list(prompt), [], [], 'length'
        while len(tokens) < (params.max_tokens or MAX_TOKENS):
            logprob = log_softmax(model.logits(ids[-model.config.max_seq_len:])[-1])
            token = next_token(logprob, params, rng)
            if token in stop:
                reason = 'stop'
                break
            ids.append(token)
            tokens.append(token)
            logprobs.append(float(logprob[token]))
        sequences.append(SampledSequence(reason, tokens, logprobs))
    return SampleResponse(sequences)


def compute_logprobs(model, prompt: list[int]) -> list[float | None]:
    # The log-probability of each token given up to max_seq_len tokens before it, as when sampling (None for the first)
    n = model.config.max_seq_len
    rows = list(log_softmax(model.logits(prompt[:n])))[:len(prompt) - 1]
    rows += [log_softmax(model.logits(prompt[k - n:k])[-1]) for k in range(n + 1, len(prompt))]
    return [None] + [float(row[token]) for row, token in zip(rows, prompt[1:])]
