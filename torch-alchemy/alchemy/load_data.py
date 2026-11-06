# The datasets, loaded from Hugging Face the first time and cached (in ~/.cache/huggingface): each split as
# (text, label) pairs, label 0 = negative, 1 = positive
import functools
import random
import re

from datasets import load_dataset


def detokenize(text: str) -> str:
    # The treebank's tokens joined back into written text, as the reviews were written:
    # It does n't work , `` really '' .  ->  It doesn't work, "really".
    for token, written in (('-LRB-', '('), ('-RRB-', ')'), ('\\/', '/'), ('\\*', '*'), ('``', '"'), ("''", '"'),
                           ('`', "'"), (' can not ', ' cannot ')):
        text = text.replace(token, written)
    spaces = r" (?=[,.!?;:%)](?!\.\.)|'(s|re|ve|ll|d|m)\b|n't\b)|(?<=\() |(?<=') (?=\d)|(?<=s) (?=' )"  # ones to drop
    text = re.sub(spaces, '', text)
    return re.sub(r'(["\']) (.*?) \1', r'\1\2\1', text)  # no spaces inside quotes


def sst() -> dict[str, list[tuple[str, int]]]:
    # The Stanford Sentiment Treebank (Socher et al., EMNLP 2013), stanfordnlp/sst, as SST-2: each sentence's
    # sentiment score (0 to 1) as negative up to 0.4 and positive above 0.6, the neutral ones in between left out; the
    # text as written, not in the treebank's tokens
    data = load_dataset('stanfordnlp/sst', data_dir='default', revision='refs/convert/parquet')  # not the ptb/ trees
    binary = lambda split: [(detokenize(row['sentence']), int(row['label'] > 0.6)) for row in data[split]
                            if not 0.4 < row['label'] <= 0.6]
    return {'train': binary('train'), 'dev': binary('validation'), 'test': binary('test')}


def imdb() -> dict[str, list[tuple[str, int]]]:
    # IMDB movie reviews (Maas et al., ACL 2011), stanfordnlp/imdb: its 50k labelled reviews (25k positive), shuffled
    # into 45k / 2.5k / 2.5k
    data = load_dataset('stanfordnlp/imdb')
    clean = lambda text: ' '.join(text.replace('<br />', ' ').split())
    rows = [(clean(row['text']), row['label']) for split in ('train', 'test') for row in data[split]]
    random.Random(0).shuffle(rows)
    return {'test': rows[:2500], 'dev': rows[2500:5000], 'train': rows[5000:]}


DATASETS = {'sst': sst, 'imdb': imdb}


@functools.cache
def load_splits(dataset: str) -> dict[str, list[tuple[str, int]]]:
    return DATASETS[dataset]()


def load_split(dataset: str, split: str) -> list[tuple[str, int]]:
    return load_splits(dataset)[split]


def load_texts(datasets: dict[str, float], split: str = 'train') -> dict[str, list[str]]:
    return {name: [text for text, _ in load_split(name, split)] for name in datasets}
