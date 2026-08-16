import os
from itertools import islice

import pytest

from cminbpe import RegexTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return RegexTokenizer()


data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taylorswift.txt")

special_tokens = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}

llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()

TEST_TEXT = """
Either/Or is a philosophical work by Søren Kierkegaard, published in 1843. It is a complex and multi-layered text that explores the nature of human existence, ethics, and the tension between aesthetic and ethical modes of life. The work is divided into two main parts, each representing a different perspective on life. The first part, "Either," presents the aesthetic viewpoint, which emphasizes the pursuit of pleasure, beauty, and personal satisfaction. The second part, "Or," presents the ethical viewpoint, which emphasizes moral responsibility, duty, and the importance of making choices that align with one's values and principles. Kierkegaard uses a variety of literary forms, including essays, letters, and fictional narratives, to convey his ideas. The work is known for its exploration of existential themes, such as the individual's struggle with freedom, choice, and the search for meaning in life. It also delves into the concept of despair and the tension between the finite and infinite aspects of human existence. Kierkegaard's writing style is often indirect and employs pseudonyms, allowing him to present different perspectives and voices within the text. This approach encourages readers to engage critically with the ideas presented and to reflect on their own lives and choices. Overall, Either/Or is a seminal work in existential philosophy, offering profound insights into the human condition and the complexities of ethical decision-making. It continues to be studied and discussed by scholars and readers interested in philosophy, literature, and the exploration of human existence."""


def load_test_text(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return f.read()


def test_cminbpe_train_set_meges(tokenizer):
    tokenizer.train(TEST_TEXT, 300, backend="c", verbose=False)
    assert isinstance(tokenizer.merges, dict)
    assert len(tokenizer.merges) >= 0
    for pair, token_id in tokenizer.merges.items():
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert all(isinstance(v, int) for v in pair)
        assert isinstance(token_id, int)


def test_cminbpe_train_on_realistic_text(tokenizer):
    text = load_test_text(data_path)
    tokenizer.train(text, 300, backend="c", verbose=False)
    assert isinstance(tokenizer.merges, dict)
    assert len(tokenizer.merges) > 0


def test_cminbpe_train_with_empty_text(tokenizer):
    empty_text = ""
    with pytest.raises(ValueError):
        tokenizer.train(empty_text, 300, backend="c", verbose=False)


def test_cminbpe_encode_decode(tokenizer):
    tokenizer.train(TEST_TEXT, 300, backend="c", verbose=False)
    encoded = tokenizer.encode(TEST_TEXT, backend="c")
    decoded = tokenizer.decode(encoded)
    assert decoded == TEST_TEXT


def test_cminbpe_encode_decode_with_special_tokens(tokenizer):
    tokenizer.train(TEST_TEXT, 300, backend="c", verbose=False)
    special_tokens = set(["<PAD>", "<UNK>", "<EOS>"])
    encoded = tokenizer.encode(TEST_TEXT, allowed_special=special_tokens, backend="c")
    decoded = tokenizer.decode(encoded)
    assert decoded == TEST_TEXT


def test_cminbpe_encode_decode_with_special_tokens_llama_like_test(tokenizer):
    tokenizer.train(llama_text, 300, backend="c", verbose=False)
    tokenizer.register_special_tokens(special_tokens)
    encoded = tokenizer.encode(
        llama_text, allowed_special=set(islice(special_tokens, 3)), backend="c"
    )
    decoded = tokenizer.decode(encoded)
    assert decoded == llama_text


def test_cminbpe_encode_decode_with_special_tokens_llama_like_test_allowed_special(
    tokenizer,
):
    tokenizer.train(llama_text, 300, backend="c", verbose=False)
    tokenizer.register_special_tokens(special_tokens)
    encoded = tokenizer.encode(llama_text, allowed_special="all", backend="c")
    decoded = tokenizer.decode(encoded)
    assert decoded == llama_text
