# The same GPT as microgpt/py/model.py, but on whole arrays, and a batch of texts at a time. The weights have the same
# names as in microgpt/py/model.py, so checkpoints work with both.
import random
from typing import TYPE_CHECKING

import numpy as np

from microgpt.np.autograd import (Tensor, cross_entropy, embed, linear, matmul, merge_heads, relu, rmsnorm, scale,
                                    softmax, split_heads, transpose_last)

if TYPE_CHECKING:
    from microgpt.config import ModelConfig


DTYPES = {'fp64': np.float64, 'fp32': np.float32}


def matrix(nout: int, nin: int, std: float = 0.08) -> Tensor:
    # The same random numbers, in the same order, as nn_utils.matrix
    return Tensor(np.array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)]))


class GPT:
    def __init__(self, config: 'ModelConfig'):
        self.config = config
        d, hidden = config.hidden_dim, config.intermediate_dim
        self.wtok_emb = matrix(config.vocab_size, d, std=0.08 * config.embedding_init_scale)
        self.wpos_emb = matrix(config.max_seq_len, d)
        self.blocks = [{name: matrix(*shape) for name, shape in [
            ('attn.wq', (d, d)), ('attn.wk', (d, d)), ('attn.wv', (d, d)), ('attn.wo', (d, d)),
            ('mlp.fc', (hidden, d)), ('mlp.proj', (d, hidden))]} for _ in range(config.num_layers)]
        self.lm_head = self.wtok_emb if config.tie_word_embeddings else matrix(config.vocab_size, d)

    def named_parameters(self) -> dict[str, Tensor]:
        named = {'wtok_emb.weight': self.wtok_emb, 'wpos_emb.weight': self.wpos_emb,
                 **{f'blocks.{i}.{name}.weight': w for i, block in enumerate(self.blocks) for name, w in block.items()}}
        return named if self.lm_head is self.wtok_emb else {**named, 'lm_head.weight': self.lm_head}  # tied: once

    def parameters(self) -> list[Tensor]:
        return list(self.named_parameters().values())

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: p.data for name, p in self.named_parameters().items()}

    def load_state_dict(self, weights: dict) -> None:
        params = self.named_parameters()
        assert weights.keys() == params.keys(), f'expected weights {sorted(params)}, got {sorted(weights)}'
        for name, p in params.items():
            p.data = np.array(weights[name], dtype=p.data.dtype)  # keeps the precision set_precision chose
            p.grad = np.zeros_like(p.data)

    def attention(self, x: Tensor, block: dict[str, Tensor]) -> Tensor:
        q, k, v = (split_heads(linear(x, block[f'attn.{name}']), self.config.num_heads) for name in ('wq', 'wk', 'wv'))
        T, head_dim = q.data.shape[2], q.data.shape[3]
        scores = scale(matmul(q, transpose_last(k)), head_dim ** -0.5)
        causal = Tensor(np.triu(np.full((T, T), -np.inf, dtype=q.data.dtype), k=1))  # no looking at later tokens
        return linear(merge_heads(matmul(softmax(scores + causal), v)), block['attn.wo'])

    def last_layer(self, ids: np.ndarray) -> Tensor:
        # ids: [B, T] -> the residual stream after the last block [B, T, hidden_dim]
        x = rmsnorm(embed(self.wtok_emb, ids) + embed(self.wpos_emb, np.arange(ids.shape[1])))
        for block in self.blocks:
            x = x + self.attention(rmsnorm(x), block)
            x = x + linear(relu(linear(rmsnorm(x), block['mlp.fc'])), block['mlp.proj'])
        return x

    def forward(self, ids: np.ndarray) -> Tensor:
        # ids: [B, T] -> next-token logits [B, T, vocab_size]
        return linear(self.last_layer(ids), self.lm_head)

    def hidden_states(self, ids: list[int]) -> np.ndarray:
        # The last layer's vector after each token, [T, hidden_dim]: what the probe (microgpt/probe.py) uses as features
        return self.last_layer(np.array([ids])).data[0].astype(np.float64)

    def logits(self, ids: list[int]) -> np.ndarray:
        # The next-token scores after each token, [T, vocab_size]
        return self.forward(np.array([ids])).data[0]


def loss(model: GPT, batch: list[list[int]]) -> Tensor:
    # Every position predicts the next token; the texts in a batch have the same length
    ids = np.array(batch)
    return cross_entropy(model.forward(ids[:, :-1]), ids[:, 1:])
