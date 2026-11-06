# Character, word and byte-pair-encoding tokenizers: how text is cut into tokens. A trained tokenizer is saved as
# tokenizer.json next to the model.
import collections
import heapq
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import regex

BOS = '<bos>'  # starts every text, and so also ends the one before it
UNK = '<unk>'  # stands for a word the word tokenizer does not know


class Tokenizer:
    # What every tokenizer has: a vocabulary (one string per token, including BOS) and the token id of each string
    name = ''  # its key in TOKENIZERS, saved in tokenizer.json

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.stoi = {token: i for i, token in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.bos_token_id = self.eos_token_id = self.stoi[BOS]  # one token starts and ends texts

    def state(self) -> dict:
        # What from_state needs to rebuild it: saved in tokenizer.json
        return {'vocab': self.vocab}

    @classmethod
    def from_state(cls, state: dict) -> 'Tokenizer':
        return cls(**state)

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump({'type': self.name, **self.state()}, f)

    def token_bytes(self) -> list[int]:
        # Each token's size in bytes of text, for bits per byte; 0 for <bos>, which is not text
        return [0 if i == self.bos_token_id else self.num_bytes(token) for i, token in enumerate(self.vocab)]

    @staticmethod
    def num_bytes(token) -> int:
        return len(token.encode())


class CharTokenizer(Tokenizer):
    # One token per character
    name = 'char'

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 0) -> 'CharTokenizer':
        # Every character in the texts is a token; vocab_size is for tokenizers with a vocabulary cap
        return cls(sorted(set(''.join(texts))) + [BOS])

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]  # characters not in the vocab are skipped

    def decode(self, token_ids: list[int]) -> str:
        return ''.join(self.vocab[t] for t in token_ids)


class WordTokenizer(Tokenizer):
    # One token per lowercased word or punctuation mark. Keeps words seen at least twice, or only the most frequent
    # ones if vocab_size is set; everything else becomes UNK.
    name = 'word'

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 0) -> 'WordTokenizer':
        counts = collections.Counter(w for t in texts for w in cls.split(t))
        keep = counts.most_common(vocab_size - 2 if vocab_size else None)
        return cls(sorted(w for w, c in keep if c >= 2) + [UNK, BOS])

    @staticmethod
    def split(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+|[^\sa-z0-9']", text.lower())

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(w, self.stoi[UNK]) for w in self.split(text)]

    def decode(self, token_ids: list[int]) -> str:
        return ' '.join(self.vocab[t] for t in token_ids)

    @staticmethod
    def num_bytes(token: str) -> int:
        return len(token.encode()) + 1  # and the space before it (approximate: <unk> stands for a word of any size)


# GPT-2's pre-tokenization: contractions, letters, numbers, other symbols and whitespace, each with a leading space
# (from OpenAI's GPT-2, Modified MIT licence; see THIRD_PARTY_NOTICES.md)
GPT2_SPLIT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
ALPHABET = [bytes([b]) for b in range(256)]  # BPE's first 256 tokens: one per byte
TEXTS_PER_PROCESS = 5000  # BPE training pre-tokenizes chunks of this many texts in parallel

BytePair = tuple[bytes, bytes]


def pretokenize(text: str) -> list[bytes]:
    # The UTF-8 bytes of each piece GPT-2's regex splits the text into; merges never cross pretokens
    return [pretoken.encode('utf-8') for pretoken in GPT2_SPLIT.findall(text)]


def count_pretokens(texts: list[str]) -> collections.Counter[bytes]:
    return collections.Counter(pretoken for text in texts for pretoken in pretokenize(text))


def count_pretokens_parallel(texts: list[str]) -> collections.Counter[bytes]:
    # count_pretokens on chunks of the texts, one process per chunk (up to one per core)
    chunks = [texts[i:i + TEXTS_PER_PROCESS] for i in range(0, len(texts), TEXTS_PER_PROCESS)]
    if len(chunks) <= 1:
        return count_pretokens(texts)
    counts = collections.Counter()
    with ProcessPoolExecutor(min(len(chunks), os.cpu_count() or 1)) as pool:
        for chunk_counts in pool.map(count_pretokens, chunks):
            counts.update(chunk_counts)
    return counts


def merge_pretoken(tokens: list[bytes], pair: BytePair) -> list[bytes]:
    # A pretoken's tokens with each occurrence of pair joined into one token
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


class ReverseLexOrderPair:
    # Sorts pairs largest first, so that of the pairs with the most occurrences the heap pops the lexicographically
    # largest
    def __init__(self, pair: BytePair):
        self.pair = pair

    def __lt__(self, other: 'ReverseLexOrderPair') -> bool:
        return self.pair > other.pair


class BPETokenizer(Tokenizer):
    # Byte-level byte-pair encoding, as in GPT-2: text is split with GPT-2's regex, each piece is read as UTF-8 bytes,
    # and the most frequent adjacent pair is merged into a new token, again and again. Each token is a string of
    # bytes, so any text can be encoded, and decode(encode(text)) == text.
    name = 'bpe'

    def __init__(self, merges: list[BytePair]):
        self.merges = {pair: rank for rank, pair in enumerate(merges)}  # rank: the order they were learned
        self.vocab = ALPHABET + [a + b for a, b in merges] + [BOS.encode()]  # each token's bytes
        self.stoi = {token: i for i, token in enumerate(self.vocab[:-1])}  # bytes -> token id, but not <bos>
        self.vocab_size = len(self.vocab)
        self.bos_token_id = self.eos_token_id = self.vocab_size - 1  # one token starts and ends texts
        self.cache: dict[bytes, list[bytes]] = {}

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 0) -> 'BPETokenizer':
        # Merges until the vocabulary (256 bytes, the merges and <bos>) has vocab_size tokens (0: 4,000 merges), or no
        # pair repeats. Pair counts are updated incrementally, only for the pretokens containing the merged pair (found
        # through pair_pretokens), and a max-heap of counts finds the next pair; entries whose count changed are
        # skipped.
        pretoken_counts = count_pretokens_parallel(texts)
        tokens = {pretoken: [bytes([b]) for b in pretoken] for pretoken in pretoken_counts}  # each one's tokens now
        pair_counts, pair_pretokens = collections.Counter(), collections.defaultdict(set)
        for pretoken, count in pretoken_counts.items():
            for pair in zip(tokens[pretoken], tokens[pretoken][1:]):
                pair_counts[pair] += count
                pair_pretokens[pair].add(pretoken)
        heap = [(-count, ReverseLexOrderPair(pair)) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        merges = []
        while heap and len(merges) < (vocab_size - len(ALPHABET) - 1 if vocab_size else 4000):
            count, best = heapq.heappop(heap)
            if -count != pair_counts[best.pair]:
                continue
            merges.append(best.pair)
            changed = set()
            for pretoken in pair_pretokens.pop(best.pair):
                old, tokens[pretoken] = tokens[pretoken], merge_pretoken(tokens[pretoken], best.pair)
                for pair in zip(old, old[1:]):
                    pair_counts[pair] -= pretoken_counts[pretoken]
                    pair_pretokens[pair].discard(pretoken)
                for pair in zip(tokens[pretoken], tokens[pretoken][1:]):
                    pair_counts[pair] += pretoken_counts[pretoken]
                    pair_pretokens[pair].add(pretoken)
                changed.update(zip(old, old[1:]), zip(tokens[pretoken], tokens[pretoken][1:]))
            for pair in changed - {best.pair}:
                if pair_counts[pair] > 0:
                    heapq.heappush(heap, (-pair_counts[pair], ReverseLexOrderPair(pair)))
        return cls(merges)

    def pieces(self, text: str) -> list[bytes]:
        # The text's tokens, as bytes: each pretoken's bytes, merged in the order the merges were learned
        out = []
        for pretoken in pretokenize(text):
            if pretoken not in self.cache:
                tokens = [bytes([b]) for b in pretoken]
                while len(tokens) > 1:
                    pair = min(zip(tokens, tokens[1:]), key=lambda p: self.merges.get(p, float('inf')))
                    if pair not in self.merges:
                        break
                    tokens = merge_pretoken(tokens, pair)
                self.cache[pretoken] = tokens
            out.extend(self.cache[pretoken])
        return out

    def encode(self, text: str) -> list[int]:
        return [self.stoi[p] for p in self.pieces(text)]

    def decode(self, token_ids: list[int]) -> str:
        # errors='replace': a sample can end halfway through a character's bytes
        return b''.join(self.vocab[t] for t in token_ids).decode('utf-8', errors='replace')

    def state(self) -> dict:
        # The merges in the order they were learned, as pairs of token ids; merge i makes token 256 + i
        return {'merges': [[self.stoi[a], self.stoi[b]] for a, b in self.merges]}

    @classmethod
    def from_state(cls, state: dict) -> 'BPETokenizer':
        vocab, merges = list(ALPHABET), []
        for a, b in state['merges']:
            merges.append((vocab[a], vocab[b]))
            vocab.append(vocab[a] + vocab[b])
        return cls(merges)

    @staticmethod
    def num_bytes(token: bytes) -> int:
        return len(token)


TOKENIZERS = {cls.name: cls for cls in (CharTokenizer, WordTokenizer, BPETokenizer)}


def load_tokenizer(path: str) -> Tokenizer:
    # A tokenizer.json written by Tokenizer.save
    with open(path) as f:
        state = json.load(f)
    return TOKENIZERS[state.pop('type')].from_state(state)


def checkpoint_tokenizer(checkpoint: str) -> Tokenizer:
    # The tokenizer.json that training saved next to a checkpoint
    return load_tokenizer(str(Path(checkpoint).with_name('tokenizer.json')))
