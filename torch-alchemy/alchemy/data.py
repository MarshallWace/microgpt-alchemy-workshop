from array import array

import torch

from alchemy.tokenizer import BPETokenizer


class PackedDataset:
    # A dataset packed as one stream of tokens: <bos> text <bos> text ..., so <bos> also ends the text before it. The
    # stream is cut into windows of max_seq_len tokens that do not overlap.
    def __init__(self, tokenizer: BPETokenizer, texts: list[str], max_seq_len: int, max_tokens: int | None = None):
        # max_tokens: stop adding texts once the stream is this long (num_texts were added)
        tokens, self.max_seq_len, self.num_texts = array('i', [tokenizer.bos_token_id]), max_seq_len, 0
        for text in texts:
            if max_tokens is not None and len(tokens) > max_tokens:
                break
            tokens.extend(tokenizer.encode(text))
            tokens.append(tokenizer.bos_token_id)
            self.num_texts += 1
        self.tokens = torch.frombuffer(tokens, dtype=torch.int32).long()
        self.num_windows = len(self.tokens) // max_seq_len
        assert self.num_windows, f'fewer than max_seq_len = {max_seq_len} tokens'

    def windows(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Every whole window in order [num_windows, max_seq_len], and the shorter window of the tokens after them (for
        # evaluation)
        end = self.num_windows * self.max_seq_len
        return self.tokens[:end].view(self.num_windows, self.max_seq_len), self.tokens[end:]


class DatasetMix:
    # The datasets' windows, each dataset's repeated or cut to its weight's share of an epoch (which has as many windows
    # as all the datasets together). Every epoch reads each of these once, in an order fixed by the seed and the epoch
    # (as torch's DistributedSampler and OLMo-core do), so it does not depend on anything else that is random. A dataset
    # with weight 0 is not trained on.
    def __init__(self, tokenizer: BPETokenizer, texts: dict[str, list[str]], weights: dict[str, float],
                 max_seq_len: int, seed: int):
        weights = {name: weight for name, weight in weights.items() if weight > 0}
        datasets = [PackedDataset(tokenizer, texts[name], max_seq_len).windows()[0] for name in weights]
        total, generator = sum(len(windows) for windows in datasets), torch.Generator().manual_seed(seed)
        index, offset = [], 0
        for windows, weight in zip(datasets, weights.values()):
            n, count = len(windows), round(weight / sum(weights.values()) * total)
            # Whole passes over the dataset, then a random part of one more
            passes = [torch.arange(n).repeat(count // n), torch.randperm(n, generator=generator)[:count % n]]
            index.append(offset + torch.cat(passes))
            offset += n
        self.windows, self.index, self.seed = torch.cat(datasets), torch.cat(index), seed
        self.num_windows = len(self.index)  # in an epoch
        self.num_tokens = self.num_windows * max_seq_len
        self.last_order: tuple[int, torch.Tensor] | None = None  # the epoch read last, and its order

    def order(self, epoch: int) -> torch.Tensor:
        if self.last_order is None or self.last_order[0] != epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self.last_order = epoch, torch.randperm(self.num_windows, generator=generator)
        return self.last_order[1]

    def batch(self, step: int, batch_size: int) -> torch.Tensor:
        # The windows at positions step * batch_size, ... of the epochs read one after another
        k = torch.arange(step * batch_size, (step + 1) * batch_size)
        epoch, position = k // self.num_windows, k % self.num_windows
        rows = torch.cat([self.order(int(e))[position[epoch == e]] for e in epoch.unique()])
        return self.windows[self.index[rows]]
