# A richer featurizer for the linear probe (allowed: see torch-alchemy/docs/07-probe.md). For each text: the last
# token's vector and the mean over all tokens, at every layer of our own torch model (the final layer after its norm).
# Score it with: python3 -m scripts.linear_probe model=my_model.featurize   (CHECKPOINT env var picks the model)
import os

import numpy as np
import torch

from microgpt.checkpoint import load_checkpoint
from microgpt.tokenizer import checkpoint_tokenizer

CHECKPOINT = os.environ.get('CHECKPOINT', 'runs/final/model.safetensors')
_cache = {}


def _load():
    if 'model' not in _cache:
        torch.set_num_threads(os.cpu_count() or 1)
        model = load_checkpoint(CHECKPOINT)
        model.eval()
        _cache['model'], _cache['tok'] = model, checkpoint_tokenizer(CHECKPOINT)
    return _cache['model'], _cache['tok']


@torch.no_grad()
def featurize(texts: list[str]) -> np.ndarray:
    model, tok = _load()
    bos, n = tok.bos_token_id, model.config.max_seq_len - 1
    rows = []
    for text in texts:
        ids = torch.tensor([[bos] + tok.encode(text)[-n:]])
        x = model.model.embed_tokens(ids)
        feats = []
        for i, layer in enumerate(model.model.layers):
            x = layer(x)
            h = model.model.norm(x) if i == len(model.model.layers) - 1 else x
            content = h[0, 1:] if h.shape[1] > 1 else h[0]  # skip <bos> in the mean
            feats += [h[0, -1], content.mean(0)]
        rows.append(torch.cat(feats).double().numpy())
    return np.stack(rows)
