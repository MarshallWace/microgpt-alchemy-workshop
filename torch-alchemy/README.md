# torch-alchemy

**Train a small Llama-style language model on a CPU, and see how much it learns about movie-review sentiment without
ever seeing a label.**

A small-scale reproduction of OpenAI's [sentiment neuron](https://arxiv.org/abs/1704.01444), in PyTorch. The model
only learns to predict the next token of movie reviews; a linear probe then measures how well its internal features
tell positive from negative.

[docs/](docs/README.md) explains how every part works, with diagrams.

| Path | What it is |
|---|---|
| `alchemy/model.py` | The GPT: RoPE, RMSNorm, QK-norm, SwiGLU, Hugging Face Llama weight names |
| `alchemy/tokenizer.py` | Byte-level BPE, as in GPT-2 |
| `alchemy/load_data.py` | The datasets, SST and IMDB, loaded from Hugging Face |
| `alchemy/data.py` | Texts packed into one token stream, cut into windows; dataset mixtures |
| `alchemy/config.py` | Every setting (chz), and AdamW / Muon / SGD |
| `alchemy/optim.py` | Muon and MuonH |
| `alchemy/train.py` | The training loop: lr schedule, eval in bits per byte, logging, checkpoints |
| `alchemy/checkpoint.py` | `.safetensors` checkpoints with the model config; step checkpoints to resume from |
| `alchemy/sampling.py` | `sample()` and `compute_logprobs()` |
| `alchemy/probe.py` | The linear probe (the score), explained in [docs/07-probe.md](docs/07-probe.md) |
| `scripts/` | `linear_probe.py` scores a model; `sample_lm.py` continues prompts |
| `experiments/` | Sweeps: train a few runs, plot them, score them |

## Setup

torch-alchemy uses the workshop's environment: in the repository root, run `uv sync` and
`source .venv/bin/activate` (see the main README). Then run every command from this folder:

```
python3 -m pytest tests      # a few seconds
```

## Train

The defaults train a 1-layer model with vectors of 128 numbers for 3 epochs over SST's 7,000 movie-review sentences
(about 30 seconds):

```
python3 -m alchemy.train                                   # saves the model and logs in runs/pretrain/
python3 -m alchemy.utils.plot log_dirs=runs/pretrain x=progress/epoch
```

`python3 -m alchemy.train --help` lists every setting (see `CLIConfig` in `alchemy/config.py`). Change any of them as
`name=value`, for example:

```
python3 -m alchemy.train model.num_layers=2 log_dir=runs/two_layers
python3 -m alchemy.train optimizer=MuonConfig log_dir=runs/muon
```

The logged metrics:

- `train/loss`: the loss per token, averaged since the last table.
- `eval/loss` and `eval/bpb`: the loss on held-out dev text, and in bits per byte, which compares across tokenizer
  sizes. Measured at the end, and every `eval_every` steps.
- `optim/<optimizer>/lr`, `progress/done_frac`, `progress/epoch`, `throughput/tokens_per_second`,
  `throughput/total_tokens` and `time/elapsed`.

Training stops after `num_epochs` or `max_steps`, whichever comes first. `init_from=<checkpoint>` continues from a
saved model.

## Score

```
$ python3 -m scripts.linear_probe model=runs/pretrain/model.safetensors
128 features, 6920 training labels
linear probe: dev 57.6%, test 55.6% (C=0.125, 89 weights used)
...
```

The probe freezes the model and takes its last layer at the text's last token. It then fits an
L1-regularised logistic regression on SST's labels. **The test accuracy is the score.** Guessing gives 50%, and
random weights about 55%. [docs/07-probe.md](docs/07-probe.md) explains how it works.

```
python3 -m scripts.sample_lm checkpoint=runs/pretrain/model.safetensors prompt='the acting was'
```

prints the probe's sentiment for the prompt and how the model continues it.

## Experiments

Each file in `experiments/` trains runs that differ in one or two settings:

```
python3 -m experiments.tune_lr                       # trains runs/tune_lr/lr=0.01, lr=0.003, lr=0.001
python3 -m experiments.plot experiment=tune_lr       # their losses on one chart
python3 -m experiments.score experiment=tune_lr      # their probe scores against their losses
```

`tune_embd`, `tune_lr` and `tune_batch` sweep one setting each. `tune_lr_batch` searches a 5 × 5 grid and plots a loss
landscape, as in [Step Law](https://arxiv.org/abs/2503.04715).

## More data

```
python3 -m alchemy.train data.weights.imdb=0.8 data.weights.sst=0.2 max_steps=5000 log_dir=runs/mix
```

Rules: pretrain only on the SST and IMDB training splits, without labels; the features must come from your own model;
don't change `fit_probe` or `evaluate_probe`; make choices on dev.

## Credits

- The datasets are loaded from Hugging Face (`alchemy/load_data.py`): SST, the Stanford Sentiment Treebank
  ([`stanfordnlp/sst`](https://huggingface.co/datasets/stanfordnlp/sst); Socher et al., EMNLP 2013), and IMDB
  movie reviews ([`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb); Maas et al., ACL 2011).
- `alchemy/utils/ml_log.py` and `alchemy/utils/cli_utils.py` are taken from Thinking Machines'
  [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook), and `alchemy/sampling.py` follows
  Tinker's sampling interface.
- `alchemy/model.py` is a scaled-down Llama with Hugging Face [Transformers](https://github.com/huggingface/transformers)'
  module and weight names; its initialisation follows
  [Deep Learning Alchemy](https://github.com/deep-learning-alchemy/assignments)'s.
- `alchemy/tokenizer.py` uses GPT-2's pre-tokenisation regex ([OpenAI](https://github.com/openai/gpt-2)).
- Muon's Newton-Schulz orthogonalisation is adapted from Keller Jordan's
  [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
- The linear probe reproduces OpenAI's [sentiment neuron](https://arxiv.org/abs/1704.01444) evaluation.

torch-alchemy is part of the microgpt workshop, licensed under the Apache License 2.0 (`../LICENSE`); the code it
takes from other projects keeps their licences, listed in `../THIRD_PARTY_NOTICES.md`.
