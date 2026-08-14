import os

import pytest

from cminbpe import RegexTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return RegexTokenizer()


TEST_TEXT = """
Either/Or is a philosophical work by Søren Kierkegaard, published in 1843. It is a complex and multi-layered text that explores the nature of human existence, ethics, and the tension between aesthetic and ethical modes of life. The work is divided into two main parts, each representing a different perspective on life. The first part, "Either," presents the aesthetic viewpoint, which emphasizes the pursuit of pleasure, beauty, and personal satisfaction. The second part, "Or," presents the ethical viewpoint, which emphasizes moral responsibility, duty, and the importance of making choices that align with one's values and principles. Kierkegaard uses a variety of literary forms, including essays, letters, and fictional narratives, to convey his ideas. The work is known for its exploration of existential themes, such as the individual's struggle with freedom, choice, and the search for meaning in life. It also delves into the concept of despair and the tension between the finite and infinite aspects of human existence. Kierkegaard's writing style is often indirect and employs pseudonyms, allowing him to present different perspectives and voices within the text. This approach encourages readers to engage critically with the ideas presented and to reflect on their own lives and choices. Overall, Either/Or is a seminal work in existential philosophy, offering profound insights into the human condition and the complexities of ethical decision-making. It continues to be studied and discussed by scholars and readers interested in philosophy, literature, and the exploration of human existence."""


def load_test_text(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return f.read()


def test_cminbpe_train_returns_dict(tokenizer):
    merges = tokenizer.train_cbackend(TEST_TEXT, 300, False)
    assert isinstance(merges, dict)
    assert len(merges) >= 0
    for pair, token_id in merges.items():
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert all(isinstance(v, int) for v in pair)
        assert isinstance(token_id, int)


def test_cminbpe_train_on_realistic_text(tokenizer):
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "taylorswift.txt"
    )
    text = load_test_text(data_path)
    merges = tokenizer.train_cbackend(text, 300, False)
    assert isinstance(merges, dict)
    assert len(merges) > 0
