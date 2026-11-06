from alchemy.tokenizer import BPETokenizer

TEXTS = ['The film was great!', "It wasn't good, it was bad.", 'Café déjà vu 🎬']


def test_bpe_is_lossless():
    """Byte-level BPE turns any text, even text it never saw, back into exactly the same text."""
    tokenizer = BPETokenizer.train(TEXTS, 300)
    for text in TEXTS + ['Unseen words: zyxwvut, ∑ and 日本語']:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_save_and_load_give_the_same_tokenizer(tmp_path):
    """tokenizer.json (the merges) rebuilds a tokenizer that encodes the same way."""
    tokenizer = BPETokenizer.train(TEXTS * 2, 300)
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    rebuilt = BPETokenizer.load(str(tmp_path / 'tokenizer.json'))
    assert rebuilt.encode(TEXTS[1]) == tokenizer.encode(TEXTS[1])
    assert rebuilt.vocab_size == tokenizer.vocab_size <= 300
