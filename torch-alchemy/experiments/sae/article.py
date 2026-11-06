# One page for the SAE, in the style of Anthropic's Towards Monosemanticity: an article (the question, the setup,
# deep dives into a few sentiment features, a look at all features), a steering explorer (precomputed samples with
# features clamped; live, with your own prompts, under experiments.sae.serve), and a browser of every feature.
# Computes the data and fills in article_template.html.
#   python3 -m experiments.sae.article checkpoint=runs/pretrain/model.safetensors      # writes runs/sae/index.html
import datetime
import json
from collections import Counter
from pathlib import Path

import chz
import torch

from alchemy.load_data import load_split
from alchemy.probe import fit_probe
from alchemy.sampling import SamplingParams
from experiments.sae.sae import (TopKSAE, context, generate, layer_activations, load_run, pooled_sst,
                                 single_accuracy, sparse_features)

TEMPLATE = Path(__file__).with_name('article_template.html')


@chz.chz
class ArticleConfig:
    checkpoint: str              # the model the SAE was trained on
    sae: str = 'runs/sae/sae.safetensors'
    features: tuple[int, ...] = (904, 1067, 1241)  # the deep dives (the template's text is written for these)
    tokens: int = 300_000        # held-out IMDB tokens to find examples in
    prompts: tuple[str, ...] = ('i thought the acting was', 'this movie', 'the ending')
    strengths: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)  # clamps precomputed for the explorer
    steer_each: int = 10         # the explorer's features: this many most positive and most negative, and the dives
    samples: int = 2             # per prompt and clamp
    out: str = 'runs/sae/index.html'


def labelled_windows(tokenizer, length: int, max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    # IMDB dev reviews packed as in training, cut into windows, and each token's review label (1 = positive)
    tokens, labels = [tokenizer.bos_token_id], [-1]
    for text, label in load_split('imdb', 'dev'):
        ids = tokenizer.encode(text) + [tokenizer.bos_token_id]
        tokens += ids
        labels += [label] * len(ids)
        if len(tokens) >= max_tokens:
            break
    n = len(tokens) // length
    return torch.tensor(tokens[:n * length]).view(n, length), torch.tensor(labels[:n * length])


def histogram(values: torch.Tensor, lo: float, hi: float, bins: int) -> list[int]:
    return torch.histc(values.float(), bins=bins, min=lo, max=hi).int().tolist()


def browser(tokenizer, sae: TopKSAE, tokens, indices, values, accuracy, effects, length: int) -> list[dict]:
    # For every feature: a label, how often it fires, a histogram of its activations, the next tokens it pushes up
    # and down, its SST accuracy alone, and examples where it fires most and about half as strongly
    position = torch.arange(len(tokens))[:, None].expand_as(indices).flatten()
    feature, act = indices.flatten(), values.flatten()
    feature, act, position = feature[act > 0], act[act > 0], position[act > 0]
    order = act.argsort(descending=True)
    order = order[feature[order].argsort(stable=True)]  # by feature, then by activation, largest first
    feature, act, position = feature[order], act[order], position[order]
    starts = torch.searchsorted(feature, torch.arange(len(sae.max_act) + 1))
    out = []
    for f in range(len(sae.max_act)):
        acts, where, top = act[starts[f]:starts[f + 1]], position[starts[f]:starts[f + 1]], sae.max_act[f].item()
        example = lambda p: context(tokens, indices, values, top, f, int(p), length)
        strongest = [example(p) for p in where[:10]]
        # The label: the words most often strongly shaded in the strongest examples (not punctuation)
        words = Counter(tokenizer.decode([t]).strip().lower() for ids, a, _ in strongest
                        for t, x in zip(ids, a) if x >= 30)
        out.append({
            'label': ' · '.join([w for w, _ in words.most_common() if any(c.isalpha() for c in w)][:3]),
            'density': len(acts) / int((tokens != tokenizer.bos_token_id).sum()), 'max': round(top, 2),
            'sst': round(accuracy[f].item(), 3), 'hist': histogram(acts / max(top, 1e-6), 0, 1, 10),
            'up': effects[f].topk(8).indices.tolist(), 'down': (-effects[f]).topk(8).indices.tolist(),
            'strongest': strongest, 'half': [example(where[i]) for i in (acts < top / 2).nonzero().flatten()[:5]],
        })
    return out


@torch.no_grad()
def main(config: ArticleConfig) -> None:
    torch.set_num_threads(4)
    model, tokenizer = load_run(config.checkpoint)
    sae = TopKSAE.load(config.sae)
    length = model.config.max_seq_len
    windows, labels = labelled_windows(tokenizer, length, config.tokens)
    tokens = windows.flatten()
    keep = tokens != tokenizer.bos_token_id
    acts = layer_activations(model, windows, sae.layer)
    indices, values = sparse_features(sae, acts, keep)
    effects = (sae.W_dec * model.model.norm.weight) @ model.lm_head.weight.T  # direct effect on the next token
    generator = torch.Generator().manual_seed(0)

    # Global: each feature and each raw dimension alone on SST, the probe on all of them, how often features fire
    pooled_sae, pooled_raw = (pooled_sst(model, tokenizer, sae.layer, e) for e in (sae.encode, lambda x: x))
    acc_sae, acc_raw = single_accuracy(pooled_sae), single_accuracy(pooled_raw)
    probe = {name: fit_probe(*(p[s][i].double().numpy() for s in ('train', 'dev') for i in (0, 1)))[0]
             for name, p in (('sae', pooled_sae), ('raw', pooled_raw))}
    density = torch.bincount(indices[values > 0], minlength=len(sae.max_act)) / keep.sum()
    acts_c, kept_labels = acts[keep] - acts[keep].mean(0), labels[keep]
    top_share = lambda x: (kept_labels[x.topk(int(0.01 * len(x))).indices] == 1).float().mean().item()

    # The explorer: the same samples for each prompt without clamps, and with each feature clamped at each strength
    params = SamplingParams(max_tokens=30, temperature=0.8)
    order = acc_sae.argsort()
    steered = list(dict.fromkeys([*config.features, *order.flip(0)[:config.steer_each].tolist(),
                                  *order[:config.steer_each].tolist()]))
    explorer = {'prompts': config.prompts, 'strengths': config.strengths, 'features': steered,
                'normal': {p: generate(model, tokenizer, p, config.samples, params) for p in config.prompts},
                'clamped': {f: {p: [generate(model, tokenizer, p, config.samples, params, sae, {f: s})
                                    for s in config.strengths] for p in config.prompts} for f in steered}}

    dives = []
    for f in config.features:
        a, top = ((indices == f) * values).sum(-1), sae.max_act[f].item()
        active = (a > 0) & keep
        bins, pos = (a[active] / top * 10).long().clamp(max=9), labels[active] == 1

        def pick(lo: float, hi: float, n: int) -> list:
            where = (active & (a >= lo * top) & (a < hi * top)).nonzero().flatten()
            return [context(tokens, indices, values, top, f, int(p), length)
                    for p in where[torch.randperm(len(where), generator=generator)[:n]]]

        # The residual-stream dimension whose activations correlate most with the feature's
        a_c = a[keep] - a[keep].mean()
        corr = (acts_c * a_c[:, None]).sum(0) / (acts_c.norm(dim=0) * a_c.norm() + 1e-9)
        dim = int(corr.abs().argmax())
        dives.append({
            'id': f, 'max': round(top, 2), 'density': density[f].item(), 'sst': acc_sae[f].item(),
            'spec': [[int(((bins == b) & pos).sum()), int(((bins == b) & ~pos).sum())] for b in range(10)],
            'pos_share': pos.float().mean().item(), 'top_share': top_share(a[keep]),
            'examples': {'strongest': [context(tokens, indices, values, top, f, int(p), length)
                                       for p in a.topk(4).indices],
                         'about half as strong': pick(0.4, 0.6, 3), 'weak': pick(0.1, 0.25, 3)},
            'up': effects[f].topk(10).indices.tolist(), 'down': (-effects[f]).topk(10).indices.tolist(),
            'logit_hist': histogram(effects[f], effects[f].min().item(), effects[f].max().item(), 40),
            'neuron': {'dim': dim, 'corr': corr[dim].item(), 'sst': acc_raw[dim].item(),
                       'top_share': top_share(acts[keep][:, dim] * corr[dim].sign())},
        })

    numbers = {
        'features': len(sae.max_act), 'k': sae.k, 'layer': sae.layer + 1, 'layers': len(model.model.layers),
        'hidden': acts.shape[1], 'vocab': tokenizer.vocab_size, 'tokens': int(keep.sum()),
        'positive_rate': (kept_labels == 1).float().mean().item(), 'quality': getattr(sae, 'quality', {}),
        'dead': int((density == 0).sum()), 'best_raw': [int(acc_raw.abs().argmax()), acc_raw.abs().max().item()],
        'over_60': [int((acc_sae.abs() > 0.6).sum()), int((acc_raw.abs() > 0.6).sum())],
        'probe_sae': probe['sae'], 'probe_raw': probe['raw'], 'checkpoint': config.checkpoint, 'sae': config.sae,
        'date': datetime.date.today().strftime('%b %-d, %Y'),
    }
    data = {'vocab': [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)], 'numbers': numbers,
            'dives': dives, 'explorer': explorer,
            'features': browser(tokenizer, sae, tokens, indices, values, acc_sae, effects, length),
            'sst_hist': {'sae': histogram(acc_sae.abs(), 0.5, 0.8, 30), 'raw': histogram(acc_raw.abs(), 0.5, 0.8, 30)},
            'density_hist': histogram(density[density > 0].log10(), -6, 0, 24)}
    page = TEMPLATE.read_text().replace('/*DATA*/', json.dumps(data, separators=(',', ':')).replace('</', '<\\/'))
    Path(config.out).write_text(page, encoding='utf-8')
    print(f'saved {config.out} ({len(page) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main(chz.entrypoint(ArticleConfig))
