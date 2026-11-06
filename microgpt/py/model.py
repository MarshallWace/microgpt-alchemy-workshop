# Adapted from Andrej Karpathy's microgpt (MIT licence; see THIRD_PARTY_NOTICES.md)
from typing import TYPE_CHECKING

from microgpt.py.cache import KVCache
from microgpt.py.autograd import Value
from microgpt.py.module import Module
from microgpt.py.nn_utils import cross_entropy, matrix, rmsnorm, scaled_dot_product_attention

if TYPE_CHECKING:
    from microgpt.config import ModelConfig


class RMSNorm(Module):
    def forward(self, x: list[Value]) -> list[Value]:
        return rmsnorm(x)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, std: float = 0.08):
        self.weight = matrix(out_features, in_features, std)

    def forward(self, x: list[Value]) -> list[Value]:
        return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in self.weight]


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, std: float = 0.08):
        self.weight = matrix(num_embeddings, embedding_dim, std)

    def forward(self, idx: int) -> list[Value]:
        return self.weight[idx]


class SelfAttention(Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.wq = Linear(hidden_dim, hidden_dim)
        self.wk = Linear(hidden_dim, hidden_dim)
        self.wv = Linear(hidden_dim, hidden_dim)
        self.wo = Linear(hidden_dim, hidden_dim)

    def forward(self, x: list[Value], cache: KVCache, layer_idx: int) -> list[Value]:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        all_keys, all_values = cache.update(k, v, layer_idx)
        x_attn = []
        for h in range(self.num_heads):
            hs = h * self.head_dim
            q_h = q[hs:hs + self.head_dim]
            k_h = [ki[hs:hs + self.head_dim] for ki in all_keys]
            v_h = [vi[hs:hs + self.head_dim] for vi in all_values]
            x_attn.extend(scaled_dot_product_attention(q_h, k_h, v_h))
        return self.wo(x_attn)


class MLP(Module):
    def __init__(self, dim: int, hidden: int):
        self.fc = Linear(dim, hidden)
        self.proj = Linear(hidden, dim)

    def forward(self, x: list[Value]) -> list[Value]:
        x = self.fc(x)
        x = [xi.relu() for xi in x]
        x = self.proj(x)
        return x


class TransformerBlock(Module):
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int):
        self.attn_norm = RMSNorm()
        self.attn = SelfAttention(hidden_dim, num_heads)
        self.mlp_norm = RMSNorm()
        self.mlp = MLP(hidden_dim, intermediate_dim)

    def forward(self, x: list[Value], cache: KVCache, layer_idx: int) -> list[Value]:
        x_residual = x
        x = self.attn_norm(x)
        x = self.attn(x, cache, layer_idx)
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = [a + b for a, b in zip(x, x_residual)]
        return x


class GPT(Module):
    def __init__(self, config: 'ModelConfig'):
        self.config = config  # saved with the weights, so a checkpoint can rebuild the same model
        self.wtok_emb = Embedding(config.vocab_size, config.hidden_dim, std=0.08 * config.embedding_init_scale)
        self.wpos_emb = Embedding(config.max_seq_len, config.hidden_dim)
        self.emb_norm = RMSNorm()
        self.blocks = [TransformerBlock(config.hidden_dim, config.num_heads, config.intermediate_dim)
                       for _ in range(config.num_layers)]
        self.lm_head = Linear(config.hidden_dim, config.vocab_size)
        if config.tie_word_embeddings:  # the output layer reuses the token embeddings
            self.lm_head.weight = self.wtok_emb.weight

    def forward(self, token_ids: list[int]) -> list[list[Value]]:
        # Returns next-token logits for every position
        return [self.lm_head(x) for x in self.last_layer(token_ids)]

    def last_layer(self, token_ids: list[int]) -> list[list[Value]]:
        # The vector after the last block for every position. Tokens are read one at a time, and the cache keeps each
        # one's keys and values for the tokens after it to look at
        cache = KVCache()
        return [self._step(token_id, pos, cache) for pos, token_id in enumerate(token_ids)]

    def _step(self, token_id: int, pos_id: int, cache: KVCache) -> list[Value]:
        tok_emb = self.wtok_emb(token_id)
        pos_emb = self.wpos_emb(pos_id)
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = self.emb_norm(x)
        for li, block in enumerate(self.blocks):
            x = block(x, cache, li)
        return x

    def logits(self, ids: list[int]) -> list[list[float]]:
        # The next-token scores after each token, [T, vocab_size]
        return [[logit.data for logit in row] for row in self(ids)]

    def hidden_states(self, ids: list[int]) -> list[list[float]]:
        # The last layer's vector after each token, [T, hidden_dim]
        return [[v.data for v in x] for x in self.last_layer(ids)]


def loss(model: GPT, batch: list[list[int]]) -> Value:
    # Every position predicts the next token; the texts in a batch have the same length
    return sum(cross_entropy(model(seq[:-1]), seq[1:]) for seq in batch) / len(batch)
