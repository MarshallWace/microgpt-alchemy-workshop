# Train a TopK sparse autoencoder on one layer's residual stream, and report how good it is
#   python3 -m experiments.sae.train checkpoint=runs/pretrain/model.safetensors layer=-1
import time

import chz
import torch
from torch.nn import functional as F

from alchemy.data import PackedDataset
from alchemy.load_data import load_texts
from experiments.sae.sae import TopKSAE, load_run, residual


@chz.chz
class SAEConfig:
    checkpoint: str            # a trained model (its tokenizer.json next to it)
    layer: int = -1            # the decoder layer whose output is read (0 = the first, -1 = the last)
    num_features: int = 2048
    k: int = 32                # features kept per token
    dataset: str = 'imdb'      # text to read: its train split
    tokens: int = 1_500_000    # training activations, plus 200,000 held out
    epochs: int = 4
    lr: float = 1e-3
    out: str = 'runs/sae/sae.safetensors'
    threads: int = 4


def activations(model, tokenizer, dataset: str, layer: int, tokens: int) -> torch.Tensor:
    # The layer's output at every token of the first `tokens` tokens of the dataset, except at <bos>, whose activations
    # are unlike the rest [N, hidden_dim]
    texts = load_texts({dataset: 1.0})[dataset]
    windows, _ = PackedDataset(tokenizer, texts, model.config.max_seq_len, max_tokens=tokens).windows()
    acts = torch.cat([residual(model, batch, layer) for batch in windows.split(256)])
    return acts[windows != tokenizer.bos_token_id]


def main(config: SAEConfig) -> None:
    torch.manual_seed(0)
    torch.set_num_threads(config.threads)
    model, tokenizer = load_run(config.checkpoint)
    layer = config.layer % model.config.num_layers
    acts = activations(model, tokenizer, config.dataset, layer, config.tokens + 200_000)
    train_acts, held_acts = acts[:-200_000], acts[-200_000:]

    # 1. Train: reconstruction error only, the TopK keeps it sparse
    sae = TopKSAE(acts.shape[1], config.num_features, config.k, layer)
    sae.scale.fill_((train_acts.pow(2).sum(-1).mean() / acts.shape[1]).sqrt())
    sae.b_dec.data = (train_acts / sae.scale).median(0).values
    optimizer, start = torch.optim.Adam(sae.parameters(), lr=config.lr), time.time()
    for epoch in range(config.epochs):
        for batch in train_acts[torch.randperm(len(train_acts))].split(4096):
            loss = ((sae.decode(sae.encode(batch)) - batch) / sae.scale).pow(2).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)
        print(f'epoch {epoch + 1}: reconstruction error {loss.item():.2f} ({time.time() - start:.0f}s)')

    # 2. Quality on held-out activations: variance explained, dead features, and the model's loss with the layer's
    # output replaced by the SAE's reconstruction (against replacing it by its mean)
    with torch.no_grad():
        z = torch.cat([sae.encode(batch) for batch in held_acts.split(8192)])
        recon = torch.cat([sae.decode(batch) for batch in z.split(8192)])
        sae.max_act.copy_(z.max(0).values)
        explained = 1 - (recon - held_acts).pow(2).sum() / (held_acts - held_acts.mean(0)).pow(2).sum()
    texts = load_texts({config.dataset: 1.0}, 'dev')[config.dataset]
    dev, _ = PackedDataset(tokenizer, texts, model.config.max_seq_len, max_tokens=25_000).windows()
    clean, spliced, ablated = (lm_loss(model, sae, dev, tokenizer.bos_token_id, replace)
                               for replace in (None, 'sae', train_acts.mean(0)))
    print(f'variance explained {explained:.1%}, dead features {(sae.max_act == 0).float().mean():.1%}')
    print(f'loss: clean {clean:.3f}, with the SAE {spliced:.3f}, with the mean {ablated:.3f} -> '
          f'{(ablated - spliced) / (ablated - clean):.1%} of the loss recovered')
    sae.save(config.out, {'variance_explained': explained.item(), 'dead': (sae.max_act == 0).float().mean().item(),
                          'loss_recovered': (ablated - spliced) / (ablated - clean), 'dataset': config.dataset})
    print(f'saved {config.out}')


@torch.no_grad()
def lm_loss(model, sae: TopKSAE, ids: torch.Tensor, bos: int, replace) -> float:
    # The next-token loss with the SAE layer's output (except at <bos>) left alone (None), replaced by the SAE's
    # reconstruction ('sae'), or replaced by a fixed vector
    x = residual(model, ids, sae.layer)
    if replace is not None:
        new = sae.decode(sae.encode(x)) if isinstance(replace, str) else replace.expand_as(x)
        x = torch.where((ids != bos).unsqueeze(-1), new, x)
    for block in model.model.layers[sae.layer + 1:]:
        x = block(x)
    logits = model.lm_head(model.model.norm(x))[:, :-1]
    return F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten()).item()


if __name__ == '__main__':
    main(chz.entrypoint(SAEConfig))
