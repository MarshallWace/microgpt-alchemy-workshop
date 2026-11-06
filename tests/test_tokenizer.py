import pytest

from microgpt.tokenizer import TOKENIZERS, BPETokenizer, CharTokenizer, WordTokenizer, load_tokenizer

TEXTS = ['The film was great!', "It wasn't good, it was bad.", 'Café déjà vu 🎬']


def test_bpe_is_lossless():
    """Byte-level BPE turns any text, even text it never saw, back into exactly the same text."""
    tokenizer = BPETokenizer.train(TEXTS, 300)
    for text in TEXTS + ['Unseen words: zyxwvut, ∑ and 日本語']:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_word_tokenizer_maps_rare_words_to_unk():
    tokenizer = WordTokenizer.train(['the film the film', 'a once'])
    assert tokenizer.decode(tokenizer.encode('the film once')) == 'the film <unk>'


@pytest.mark.parametrize('name', list(TOKENIZERS))
def test_save_and_load_give_the_same_tokenizer(name, tmp_path):
    """tokenizer.json rebuilds a tokenizer that encodes the same way."""
    tokenizer = TOKENIZERS[name].train(TEXTS * 2, 300)
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    rebuilt = load_tokenizer(str(tmp_path / 'tokenizer.json'))
    assert type(rebuilt) is type(tokenizer) and rebuilt.encode(TEXTS[1]) == tokenizer.encode(TEXTS[1])
    assert rebuilt.bos_token_id == rebuilt.eos_token_id == tokenizer.bos_token_id
    assert rebuilt.vocab_size == tokenizer.vocab_size <= 300


def test_token_bytes_add_up_to_the_text():
    """Bits per byte divide by each token's bytes: for BPE they add up to the text's, and <bos> has none."""
    tokenizer = BPETokenizer.train(TEXTS, 300)
    sizes = tokenizer.token_bytes()
    assert sum(sizes[t] for t in tokenizer.encode(TEXTS[2])) == len(TEXTS[2].encode())
    assert sizes[tokenizer.bos_token_id] == 0


def test_char_tokenizer_skips_unknown_characters():
    assert CharTokenizer.train(['ab']).encode('abc') == CharTokenizer.train(['ab']).encode('ab')
