# A Llama-style GPT in PyTorch, with Hugging Face Llama's module and weight names (model.layers.0.self_attn.q_proj...).
# Compared with microgpt/py/model.py: rotary position embeddings (RoPE) instead of learned position vectors, RMSNorm
# with a learned scale, normalised queries and keys (QK-norm), a SwiGLU MLP, and a final norm before the output.
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from microgpt.config import ModelConfig


ROPE_THETA = 10_000.0
DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16}


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In fp32 whatever the input's format (bf16 under autocast), then back to it
        x32 = x.float()
        return self.weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-5)).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, -1)
    return torch.cat((-second, first), -1)


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    # Rotate each pair of numbers in a query or key by an angle that grows with the token's position
    T, head_dim = x.shape[-2], x.shape[-1]
    inv_freq = 1.0 / ROPE_THETA ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device) / head_dim)
    angles = torch.outer(torch.arange(T, dtype=torch.float32, device=x.device), inv_freq)
    angles = torch.cat((angles, angles), -1)
    return x * angles.cos() + rotate_half(x) * angles.sin()


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = (proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj))
        q, k = apply_rope(self.q_norm(q)), apply_rope(self.k_norm(k))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_dim)
        self.self_attn = Attention(hidden_dim, num_heads)
        self.post_attention_layernorm = RMSNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Transformer(nn.Module):
    # Token ids [B, T] -> the last layer's vectors [B, T, hidden_dim]
    def __init__(self, vocab_size: int, num_layers: int, hidden_dim: int, num_heads: int, intermediate_dim: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(DecoderLayer(hidden_dim, num_heads, intermediate_dim) for _ in range(num_layers))
        self.norm = RMSNorm(hidden_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class GPT(nn.Module):
    # Token ids [B, T] -> next-token logits [B, T, vocab_size], plus logits and hidden_states for token lists, as every
    # backend has
    def __init__(self, config: 'ModelConfig'):
        super().__init__()
        self.config = config
        self.model = Transformer(config.vocab_size, config.num_layers, config.hidden_dim, config.num_heads,
                                 config.intermediate_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        # Normal values cut off at 3 standard deviations, of size 1/sqrt(inputs) for the linear layers and the output
        # layer, and 1/hidden_dim for untied token embeddings; then the token embeddings are multiplied by
        # embedding_init_scale
        def normal(weight: torch.Tensor, std: float) -> None:
            nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std)

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                normal(module.weight, module.weight.shape[1] ** -0.5)
        normal(self.lm_head.weight, self.config.hidden_dim ** -0.5)
        if not self.config.tie_word_embeddings:
            normal(self.model.embed_tokens.weight, 1 / self.config.hidden_dim)
        self.model.embed_tokens.weight.mul_(self.config.embedding_init_scale)

    @property
    def device(self) -> torch.device:
        return self.lm_head.weight.device

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(ids))

    @torch.no_grad()
    def logits(self, ids: list[int]) -> np.ndarray:
        return self(torch.tensor([ids], device=self.device))[0].double().cpu().numpy()

    @torch.no_grad()
    def hidden_states(self, ids: list[int]) -> np.ndarray:
        return self.model(torch.tensor([ids], device=self.device))[0].double().cpu().numpy()


def loss(model: GPT, batch: list[list[int]]) -> torch.Tensor:
    # Every position predicts the next token; the texts in a batch have the same length
    ids = torch.tensor(batch, device=model.device)
    logits = model(ids[:, :-1])
    return F.cross_entropy(logits.float().flatten(0, 1), ids[:, 1:].flatten())
