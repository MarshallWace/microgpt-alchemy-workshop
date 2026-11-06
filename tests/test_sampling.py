import random

import numpy as np
import pytest

from microgpt.config import BACKENDS, ModelConfig
from microgpt.sampling import SamplingParams, compute_logprobs, next_token, sample


@pytest.fixture(params=list(BACKENDS))
def model(request):
    BACKENDS[request.param].seed(0)
    config = ModelConfig(vocab_size=20, max_seq_len=8, num_layers=1, hidden_dim=8, num_heads=2)
    return BACKENDS[request.param].GPT(config)


def test_same_seed_same_samples(model):
    params = SamplingParams(max_tokens=12, seed=1, stop=())
    assert sample(model, [0, 1], 2, params) == sample(model, [0, 1], 2, params)


def test_greedy_is_top_1(model):
    """Temperature 0, top_k=1 and top_p=0 all pick the most likely token."""
    greedy = [sample(model, [0, 1], 1, SamplingParams(max_tokens=12, stop=(), **settings)).sequences[0].tokens
              for settings in ({'temperature': 0.0}, {'top_k': 1}, {'top_p': 0.0})]
    assert greedy[0] == greedy[1] == greedy[2]


def test_logprobs_match_sampling(model):
    """compute_logprobs scores a sample as sampling did, also after the context window fills up."""
    sequence = sample(model, [0, 1], 1, SamplingParams(max_tokens=12, seed=1, stop=())).sequences[0]
    scored = compute_logprobs(model, [0, 1] + sequence.tokens)
    assert scored[0] is None
    assert scored[2:] == pytest.approx(sequence.logprobs, abs=1e-5)


def test_sampling_stops_at_end_of_text(model):
    sequence = sample(model, [0, 1], 1, SamplingParams(max_tokens=200, seed=1), end_of_text=3).sequences[0]
    assert 3 not in sequence.tokens
    assert sequence.stop_reason == 'stop' or len(sequence.tokens) == 200


def test_top_p_applies_to_what_top_k_kept():
    """top_p is a share of the probability top_k kept: of 0.45 / 0.35 / 0.2, top_k=2 leaves 0.56 / 0.44, so top_p=0.5
    keeps only the first token."""
    logprobs, rng = np.log([0.45, 0.35, 0.2]), random.Random(0)
    params = SamplingParams(top_k=2, top_p=0.5)
    assert {next_token(logprobs, params, rng) for _ in range(50)} == {0}
