import torch

from alchemy.data import DatasetMix, PackedDataset
from alchemy.tokenizer import BOS, BPETokenizer

TOKENIZER = BPETokenizer.train(['abcdef'], 257)  # one token per byte


def test_texts_are_joined_by_one_bos():
    """<bos> a b c <bos> d e <bos>, cut into windows of max_seq_len tokens that do not overlap."""
    data = PackedDataset(TOKENIZER, ['abc', 'de'], max_seq_len=3)
    windows, rest = data.windows()
    assert [TOKENIZER.decode(w.tolist()) for w in windows] == [f'{BOS}ab', f'c{BOS}d']
    assert TOKENIZER.decode(rest.tolist()) == f'e{BOS}'


def test_each_window_once_per_epoch():
    """An epoch reads every window once; the next epoch reads them again in a different order."""
    mixture = DatasetMix(TOKENIZER, {'a': ['abcdef'] * 10}, {'a': 1.0}, max_seq_len=4, seed=0)
    n = mixture.num_windows
    first, second = mixture.batch(0, n), mixture.batch(1, n)
    whole = sorted(map(tuple, mixture.windows.tolist()))
    assert sorted(map(tuple, first.tolist())) == sorted(map(tuple, second.tolist())) == whole
    assert not torch.equal(first, second)


def test_order_depends_only_on_seed_and_step():
    """Batches are the same for the same seed and step, also across an epoch boundary, and differ for another seed."""
    make = lambda seed: DatasetMix(TOKENIZER, {'a': ['abcdef'] * 10}, {'a': 1.0}, max_seq_len=4, seed=seed)
    torch.manual_seed(123)  # the global random state does not matter
    assert torch.equal(make(0).batch(2, 6), make(0).batch(2, 6))
    assert not torch.equal(make(0).batch(0, 6), make(1).batch(0, 6))


def test_max_tokens_stops_adding_texts():
    data = PackedDataset(TOKENIZER, ['abc', 'de', 'f'], max_seq_len=2, max_tokens=5)  # 5 tokens after 'abc'
    assert data.num_texts == 2


def test_mixture_has_exactly_each_weights_share():
    """Each dataset fills its weight's share of an epoch, repeating a small dataset's windows if it needs more."""
    mixture = DatasetMix(TOKENIZER, {'a': ['aaaaaa'] * 20, 'b': ['bbbbbb'] * 4}, {'a': 0.5, 'b': 0.5}, max_seq_len=4,
                      seed=0)
    windows = mixture.batch(0, mixture.num_windows)
    from_b = sum('b' in TOKENIZER.decode(w.tolist()).replace(BOS, '') for w in windows)
    assert from_b == mixture.num_windows // 2


def test_weight_0_is_not_trained_on():
    mixture = DatasetMix(TOKENIZER, {'a': ['aaaaaa'] * 20, 'b': ['bbbbbb'] * 20}, {'a': 1.0, 'b': 0.0}, max_seq_len=4,
                         seed=0)
    assert not any('b' in TOKENIZER.decode(w.tolist()).replace(BOS, '') for w in mixture.batch(0, mixture.num_windows))
