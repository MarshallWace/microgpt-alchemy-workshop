# A TopK sparse autoencoder (Gao et al. 2024, https://arxiv.org/abs/2406.04093) on one layer's residual stream, and
# what the SAE experiments share: loading a model with its tokenizer, the layer's activations, and clamping features.
import json

import chz
import torch
from safetensors import safe_open
from safetensors.torch import load_model, save_model
from torch import nn
from torch.nn import functional as F

from alchemy.checkpoint import load_checkpoint
from alchemy.load_data import load_split
from alchemy.model import GPT
from alchemy.sampling import SamplingParams, sample
from alchemy.tokenizer import BPETokenizer, checkpoint_tokenizer


def load_run(checkpoint: str) -> tuple[GPT, BPETokenizer]:
    # The model, and the tokenizer.json next to it (a BPE checkpoint of the main microgpt repo works too)
    return load_checkpoint(checkpoint).eval(), checkpoint_tokenizer(checkpoint)


@torch.no_grad()
def residual(model: GPT, ids: torch.Tensor, layer: int) -> torch.Tensor:
    # The residual stream after decoder layer `layer` (0 = the first) [B, T, hidden_dim]
    x = model.model.embed_tokens(ids)
    for block in model.model.layers[:layer + 1]:
        x = block(x)
    return x


class TopKSAE(nn.Module):
    # activation -> features (only the k largest are kept) -> reconstruction. Activations are divided by scale first,
    # so their mean squared norm is hidden_dim; max_act is each feature's largest activation on held-out text.
    def __init__(self, hidden_dim: int, num_features: int, k: int, layer: int):
        super().__init__()
        self.k, self.layer = k, layer
        self.W_dec = nn.Parameter(F.normalize(torch.randn(num_features, hidden_dim), dim=1))
        self.W_enc = nn.Parameter(self.W_dec.data.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(num_features))
        self.b_dec = nn.Parameter(torch.zeros(hidden_dim))
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('max_act', torch.zeros(num_features))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.relu((x / self.scale - self.b_dec) @ self.W_enc + self.b_enc)
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ self.W_dec + self.b_dec) * self.scale

    def save(self, path: str, quality: dict[str, float]) -> None:
        # quality: how well it reconstructs, measured by train.py; shown on the SAE page
        save_model(self, path, {'k': str(self.k), 'layer': str(self.layer), 'quality': json.dumps(quality)})

    @classmethod
    def load(cls, path: str) -> 'TopKSAE':
        with safe_open(path, 'pt') as f:
            meta, shape = f.metadata(), f.get_slice('W_dec').get_shape()
        sae = cls(shape[1], shape[0], int(meta['k']), int(meta['layer']))
        load_model(sae, path)
        sae.quality = json.loads(meta.get('quality', '{}'))
        return sae


def clamp_features(model: GPT, sae: TopKSAE, clamps: dict[int, float]):
    # While the returned hook is registered, each feature f in clamps is set to clamps[f] x its max activation at the
    # SAE's layer, at every position; the SAE's reconstruction error is added back, so nothing else changes (as
    # Anthropic's Golden Gate Claude). Remove it with .remove().
    def hook(module, inputs, output):
        with torch.no_grad():
            z = sae.encode(output)
            error = output - sae.decode(z)
            for f, times in clamps.items():
                z[..., f] = times * sae.max_act[f]
            return sae.decode(z) + error
    return model.model.layers[sae.layer].register_forward_hook(hook)


@torch.no_grad()
def layer_activations(model: GPT, windows: torch.Tensor, layer: int) -> torch.Tensor:
    # The residual stream after `layer` at every token of the windows [num_windows * length, hidden_dim]
    return torch.cat([residual(model, batch, layer).flatten(0, 1) for batch in windows.split(128)])


@torch.no_grad()
def sparse_features(sae: TopKSAE, acts: torch.Tensor, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Every token's k active features and their activations [N, k]; none where keep is False (e.g. at <bos>)
    indices, values = [], []
    for x, k in zip(acts.split(8192), keep.split(8192)):
        top = (sae.encode(x) * k[:, None]).topk(sae.k, dim=-1)
        indices.append(top.indices)
        values.append(top.values)
    return torch.cat(indices), torch.cat(values)


def context(tokens: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, scale: float, f: int, p: int,
            length: int) -> list:
    # The tokens around position p in its window, feature f's activation at each (0-100 of scale), and p's place
    row, col = divmod(p, length)
    lo, hi = row * length + max(0, col - 16), row * length + min(length, col + 5)
    a = ((indices[lo:hi] == f) * values[lo:hi]).sum(-1) / scale
    return [tokens[lo:hi].tolist(), (100 * a).round().int().tolist(), p - lo]


@torch.no_grad()
def pooled_sst(model: GPT, tokenizer: BPETokenizer, layer: int, encode) -> dict:
    # For SST train and dev: each sentence's encode(activations at layer) maxed over its tokens, and its labels
    n, out = model.config.max_seq_len - 1, {}
    for split in ('train', 'dev'):
        data = load_split('sst', split)
        rows = [encode(residual(model, torch.tensor([[tokenizer.bos_token_id] + tokenizer.encode(t)[-n:]]),
                                layer)[0, 1:]).max(0).values for t, _ in data]
        out[split] = torch.stack(rows), torch.tensor([label for _, label in data])
    return out


def single_accuracy(pooled: dict) -> torch.Tensor:
    # Each column alone on dev, split halfway between the class means on train. Signed: + if higher on positive
    (Xt, yt), (Xd, yd) = pooled['train'], pooled['dev']
    pos, neg = Xt[yt == 1].mean(0), Xt[yt == 0].mean(0)
    correct = ((Xd - (pos + neg) / 2) * (pos - neg).sign() > 0) == yd[:, None]
    return correct.float().mean(0) * (pos - neg).sign()


@torch.no_grad()
def generate(model: GPT, tokenizer: BPETokenizer, prompt: str, num_samples: int, params: SamplingParams,
             sae: TopKSAE | None = None, clamps: dict[int, float] | None = None) -> list[str]:
    # num_samples continuations of the prompt, with seeds params.seed (or 0), +1, ...: the same for every call, so
    # samples with and without clamps differ only by the clamps
    ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
    hook = clamp_features(model, sae, clamps) if clamps else None
    try:
        return [tokenizer.decode(sample(model, ids, 1, chz.replace(params, seed=(params.seed or 0) + i),
                                        tokenizer.eos_token_id).sequences[0].tokens) for i in range(num_samples)]
    finally:
        if hook:
            hook.remove()


def steer(model: GPT, tokenizer: BPETokenizer, sae: TopKSAE, prompt: str, clamps: dict[int, float], num_samples: int,
          params: SamplingParams) -> dict[str, list[str]]:
    # The same samples with the features clamped and without
    return {'clamped': generate(model, tokenizer, prompt, num_samples, params, sae, clamps),
            'normal': generate(model, tokenizer, prompt, num_samples, params)}
