import os

import pytest

from cminbpe import RegexTokenizer

# common test data

# a few strings to test the tokenizers on
test_strings = [
    "",  # empty string
    "?",  # single character
    "hello world!!!? (안녕하세요!) lol123 😉",  # fun small string
    "FILE:taylorswift.txt",  # FILE: is handled as a special string in unpack()
]


def unpack(text):
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        taylorswift_file = os.path.join(dirname, text[5:])
        with open(taylorswift_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return text


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


@pytest.fixture(scope="session")
def tokenizer_file_prefix(tmp_path_factory):
    # create a temporary dir for the tokenizer files
    tmp_dir = tmp_path_factory.mktemp("data")
    return tmp_dir / "tokenizer"


@pytest.mark.parametrize("tokenizer_factory", [RegexTokenizer])
@pytest.mark.parametrize("special_tokens", [{}, special_tokens])
def test_save_load_tokenizer(special_tokens, tokenizer_factory, tokenizer_file_prefix):
    # create a tokenizer and train it on some text
    text = llama_text
    vocab_size = 256 + 64
    tokenizer = tokenizer_factory()
    tokenizer.train(text, vocab_size)
    tokenizer.register_special_tokens(special_tokens)
    # encode the text with the tokenizer
    ids = tokenizer.encode(text, allowed_special="all")
    # save the tokenizer to a file
    tokenizer.save(str(tokenizer_file_prefix))

    # load the tokenizer back from the file
    tokenizer = tokenizer_factory()
    tokenizer.load(str(tokenizer_file_prefix) + ".model")

    # check that the loaded tokenizer has the same pattern and merges as the original
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(tokenizer.encode(text, allowed_special="all")) == text
    assert tokenizer.encode(text, allowed_special="all") == ids


# Test train and encode decode round trip
@pytest.mark.parametrize("tokenizer_factory", [RegexTokenizer])
@pytest.mark.parametrize("text", test_strings)
def test_train_encode_decode_round_trip(tokenizer_factory, text, tokenizer_file_prefix):
    text = unpack(text)
    vocab_size = 256 + 64
    tokenizer = tokenizer_factory()
    tokenizer.train(text, vocab_size, verbose=False)
    tokenizer.register_special_tokens(special_tokens)
    ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(ids)
    assert decoded_text == text
