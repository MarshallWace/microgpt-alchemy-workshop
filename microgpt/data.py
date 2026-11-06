from array import array

import numpy as np

from microgpt.tokenizer import Tokenizer


class PackedDataset:
    # A dataset packed as one stream of tokens: <bos> text <bos> text ..., so <bos> also ends the text before it. The
    # stream is cut into windows of max_seq_len tokens that do not overlap.
    def __init__(self, tokenizer: Tokenizer, texts: list[str], max_seq_len: int, max_tokens: int | None = None):
        # max_tokens: stop adding texts once the stream is this long (num_texts were added)
        tokens, self.max_seq_len, self.num_texts = array('i', [tokenizer.bos_token_id]), max_seq_len, 0
        for text in texts:
            if max_tokens is not None and len(tokens) > max_tokens:
                break
            tokens.extend(tokenizer.encode(text))
            tokens.append(tokenizer.bos_token_id)
            self.num_texts += 1
        self.tokens = np.array(tokens, dtype=np.int64)
        self.num_windows = len(self.tokens) // max_seq_len
        assert self.num_windows, f'fewer than max_seq_len = {max_seq_len} tokens'

    def windows(self) -> tuple[np.ndarray, np.ndarray]:
        # Every whole window in order [num_windows, max_seq_len], and the shorter window of the tokens after them
        end = self.num_windows * self.max_seq_len
        return self.tokens[:end].reshape(self.num_windows, self.max_seq_len), self.tokens[end:]


class DatasetMix:
    # The datasets' windows, each dataset's repeated or cut to its weight's share of an epoch (which has as many windows
    # as all the datasets together). Every epoch reads each of these once, in an order fixed by the seed and the epoch
    # (as torch's DistributedSampler and OLMo-core do), so it does not depend on anything else that is random. A dataset
    # with weight 0 is not trained on.
    def __init__(self, tokenizer: Tokenizer, texts: dict[str, list[str]], weights: dict[str, float], max_seq_len: int,
                 seed: int):
        weights = {name: weight for name, weight in weights.items() if weight > 0}
        datasets = [PackedDataset(tokenizer, texts[name], max_seq_len).windows()[0] for name in weights]
        total, rng = sum(len(windows) for windows in datasets), np.random.default_rng(seed)
        index, offset = [], 0
        for windows, weight in zip(datasets, weights.values()):
            n, count = len(windows), round(weight / sum(weights.values()) * total)
            # Whole passes over the dataset, then a random part of one more
            index.append(offset + np.concatenate([np.tile(np.arange(n), count // n), rng.permutation(n)[:count % n]]))
            offset += n
        self.windows, self.index, self.seed = np.concatenate(datasets), np.concatenate(index), seed
        self.num_windows = len(self.index)  # in an epoch
        self.num_tokens = self.num_windows * max_seq_len
        self.last_order: tuple[int, np.ndarray] | None = None  # the epoch read last, and its order

    def order(self, epoch: int) -> np.ndarray:
        if self.last_order is None or self.last_order[0] != epoch:
            self.last_order = epoch, np.random.default_rng(self.seed + epoch).permutation(self.num_windows)
        return self.last_order[1]

    def batch(self, step: int, batch_size: int) -> np.ndarray:
        # The windows at positions step * batch_size, ... of the epochs read one after another [batch_size, max_seq_len]
        epoch, position = np.divmod(np.arange(step * batch_size, (step + 1) * batch_size), self.num_windows)
        rows = np.concatenate([self.order(int(e))[position[epoch == e]] for e in np.unique(epoch)])
        return self.windows[self.index[rows]]
