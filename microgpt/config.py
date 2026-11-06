import abc
import re
from types import ModuleType

import chz

import microgpt.np
import microgpt.py
from microgpt.optim import Optimizer, leaves
from microgpt.utils.cli_utils import LogdirBehavior

# Each backend is a package with the same names: GPT and loss (model.py); AdamW, SGD, Muon and clip_grad_norm
# (optim.py); set_precision, state_dict, load_state_dict, seed, set_threads and SUPPORTS_COMPILE (backend.py)
BACKENDS = {'python': microgpt.py, 'numpy': microgpt.np}
try:  # torch is optional
    import microgpt.pt
    BACKENDS['torch'] = microgpt.pt
except ImportError:
    pass


@chz.chz
class ModelConfig:
    max_seq_len: int = 32    # context window
    num_layers: int = 1
    hidden_dim: int = 128
    num_heads: int = 4
    # The MLP's width; 4 x hidden_dim unless set
    intermediate_dim: int | None = chz.field(default=None, munger=lambda self, value: value or 4 * self.hidden_dim)
    tie_word_embeddings: bool = False  # the output layer reuses the token embeddings
    embedding_init_scale: float = 1.0  # multiplies the token embeddings' initial random size
    vocab_size: int = 0      # set from the tokenizer


def depth_preset(depth: int, max_seq_len: int = 32) -> ModelConfig:
    # One knob for model size: width 64 per layer, heads of 32 numbers,
    # e.g. model=depth_preset model.depth=2
    return ModelConfig(max_seq_len=max_seq_len, num_layers=depth, hidden_dim=64 * depth, num_heads=2 * depth)


@chz.chz
class DataConfig:
    # Datasets ('sst', 'imdb') and how often each is sampled, e.g. data.weights.imdb=0.8 data.weights.sst=0.2
    weights: dict[str, float] = chz.field(default_factory=lambda: {'sst': 1.0})
    tokenizer: str = 'char'  # 'char', 'word' or 'bpe' (microgpt/tokenizer.py)
    vocab_size: int = 0      # cap on the word/BPE vocabulary (0 = the tokenizer's default)


@chz.chz
class PrecisionConfig:
    # Number formats ('fp64', 'fp32', 'bf16'): compute=bf16 with params=fp32 is
    # mixed precision (torch only)
    params: str = 'fp32'
    compute: str = 'fp32'


@chz.chz
class OptimizerConfig(abc.ABC):
    lr: float = 0.003
    lr_schedule: str = 'linear'  # 'linear', 'cosine' or 'constant'
    warmup: float = 0.0          # the lr rises linearly from 0 over this fraction of training (up to 1) or steps
    min_lr_ratio: float = 0.0    # the lr falls to this fraction of lr, not to 0
    max_grad_norm: float = 0.0   # if set, gradients are scaled down to at most this size (gradient clipping)
    weight_decay: float = 0.0
    decay_embeddings: bool = True  # False: no weight decay on the embeddings (tied: the output layer too)

    def param_groups(self, params: dict) -> list[dict]:
        # Weight decay for every weight but the norms' scales (and the embeddings, unless decay_embeddings)
        no_decay = lambda name: 'norm' in name or (not self.decay_embeddings and 'emb' in name)
        decay = {'params': leaves([w for n, w in params.items() if not no_decay(n)]), 'weight_decay': self.weight_decay}
        rest = {'params': leaves([w for n, w in params.items() if no_decay(n)]), 'weight_decay': 0.0}
        return [group for group in (decay, rest) if group['params']]

    @abc.abstractmethod
    def build(self, backend: ModuleType, model) -> list[Optimizer]:
        ...


@chz.chz
class AdamWConfig(OptimizerConfig):
    beta1: float = 0.85
    beta2: float = 0.99

    def build(self, backend: ModuleType, model) -> list[Optimizer]:
        groups = self.param_groups(dict(model.named_parameters()))
        return [backend.AdamW(groups, lr=self.lr, betas=(self.beta1, self.beta2))]


def is_block_matrix(name: str) -> bool:
    # A weight matrix inside a transformer block (e.g. blocks.0.attn.wq.weight, model.layers.0.mlp.up_proj.weight), not
    # an embedding, the output layer or a norm's scale
    return re.search(r'\.\d+\.', f'.{name}') is not None and 'norm' not in name


@chz.chz
class MuonConfig(AdamWConfig):
    # Muon for the weight matrices inside the blocks (muon_lr, and weight_decay unless hyperball); AdamW (lr, betas,
    # weight_decay) for the rest
    muon_lr: float = 0.02
    hyperball: bool = False  # MuonH: each matrix keeps its size

    def build(self, backend: ModuleType, model) -> list[Optimizer]:
        named = dict(model.named_parameters())
        rest = self.param_groups({name: w for name, w in named.items() if not is_block_matrix(name)})
        matrices = [w for name, w in named.items() if is_block_matrix(name)]
        return [backend.AdamW(rest, lr=self.lr, betas=(self.beta1, self.beta2)),
                backend.Muon(matrices, lr=self.muon_lr, hyperball=self.hyperball,
                             weight_decay=self.weight_decay)]


@chz.chz
class SGDConfig(OptimizerConfig):
    lr: float = 0.1
    momentum: float = 0.9

    def build(self, backend: ModuleType, model) -> list[Optimizer]:
        return [backend.SGD(self.param_groups(dict(model.named_parameters())), lr=self.lr, momentum=self.momentum)]


@chz.chz
class CLIConfig:
    backend: str = 'numpy'
    batch_size: int = 32
    num_epochs: float = 3.0
    max_steps: int | None = None   # stop after this many steps if that is sooner
    seed: int = 42
    threads: int = 1         # CPU threads for numpy and torch (0 = every core); small models are fastest on 1
    compile: bool = False    # torch only: torch.compile the model, faster after slow first steps
    init_from: str = ''      # a checkpoint to continue training
    log_dir: str = 'runs/pretrain'  # config.json, metrics.jsonl, model.safetensors and step checkpoints
    log_every: int = 100     # steps between logged tables
    eval_every: int = 0      # steps between evals of eval/loss; 0 = only at the end
    eval_windows: int = 64   # held-out windows from each dataset's dev split, for eval/loss; 0 = none
    save_every: int = 0      # steps between step checkpoints; training again in log_dir resumes from the newest
    # If log_dir already exists when training from the command line: 'ask', 'resume', 'delete' or 'raise'
    behavior_if_log_dir_exists: LogdirBehavior = 'ask'
    data: DataConfig = chz.field(default_factory=DataConfig)
    model: ModelConfig = chz.field(default_factory=ModelConfig)
    precision: PrecisionConfig = chz.field(default_factory=PrecisionConfig)
    optimizer: OptimizerConfig = chz.field(default_factory=AdamWConfig, blueprint_unspecified=AdamWConfig)
