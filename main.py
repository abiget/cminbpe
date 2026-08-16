import time

from datasets import load_dataset

from cminbpe import RegexTokenizer

SIZE_LIMIT = 100  # Limit the number of lines to load from the dataset``


def load_wikitext_dataset(size: int):
    # Load the wikitext dataset from the Hugging Face Hub
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_lines = [line for line in dataset["train"]["text"] if line.strip()]
    return "".join(train_lines[:size])  # limit to the first size lines


def main():
    print("Loading the wikitext dataset...")
    # Load the wikitext dataset

    train_text = load_wikitext_dataset(SIZE_LIMIT)
    test_string = "This is a test string for encoding and decoding."

    print("Starting MinBPE training...")
    tokenizer = RegexTokenizer()
    start_time = time.time()
    tokenizer.train(train_text, 10000, backend="python", verbose=False)
    end_time = time.time()
    print(
        f"MinBPE Python Implementation Training completed in {end_time - start_time:.2f} seconds."
    )
    encoded = tokenizer.encode(test_string, backend="python")
    decoded = tokenizer.decode(encoded)
    assert test_string == decoded, "Roundtrip encoding and decoding failed!"

    print("Starting MinBPE C Extension Implementation Training...")
    tokenizer_c = RegexTokenizer()
    start_time_c = time.time()
    tokenizer_c.train(train_text, 10000, backend="c", verbose=False)
    end_time_c = time.time()
    print(
        f"MinBPE C Extension Implementation Training completed in {end_time_c - start_time_c:.2f} seconds."
    )

    encoded_c = tokenizer_c.encode(test_string, backend="c")
    decoded_c = tokenizer_c.decode(encoded_c)
    assert (
        test_string == decoded_c
    ), "Roundtrip encoding and decoding failed for C extension!"

    print(f"Vocabulary size (Python): {len(tokenizer.vocab)}")
    print(f"Decoded string (Python): {decoded}")
    print(f"Vocabulary size (C Extension): {len(tokenizer_c.vocab)}")
    print(f"Decoded string (C Extension): {decoded_c}")


if __name__ == "__main__":
    main()
