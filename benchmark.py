import time

from datasets import load_dataset

from cminbpe import RegexTokenizer

SIZE_LIMIT = 1000  # Limit the number of lines to load from the dataset``


def load_wikitext_dataset(size: int):
    # Load the wikitext dataset from the Hugging Face Hub
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_lines = [line for line in dataset["train"]["text"] if line.strip()]
    return "".join(train_lines[:size])  # limit to the first size lines


def print_tokenizer_info(tokenizer: RegexTokenizer, decoded: str, backend: str):
    print(f"Vocabulary size ({backend.capitalize()}): {len(tokenizer.vocab)}")
    print(f"Decoded string ({backend.capitalize()}): {decoded}")


def benchmark_tokenizer(
    tokenizer: RegexTokenizer,
    train_text: str,
    test_string: str,
    backend: str,
    vocab_size: int = 10000,
    verbose: bool = False,
):
    print(f"Starting MinBPE {backend.capitalize()} Implementation Training...")
    start_time = time.time()
    tokenizer.train(train_text, vocab_size, backend=backend, verbose=verbose)
    end_time = time.time()
    print(
        f"MinBPE {backend.capitalize()} Implementation Training completed in {end_time - start_time:.6f} seconds."
    )

    start_time = time.time()
    encoded = tokenizer.encode(test_string, backend=backend)
    end_time = time.time()
    print(
        f"MinBPE {backend.capitalize()} Implementation Encoding completed in {end_time - start_time:.6f} seconds."
    )
    decoded = tokenizer.decode(encoded)
    assert (
        test_string == decoded
    ), f"Roundtrip encoding and decoding failed for {backend}!"

    # Print tokenizer information
    print_tokenizer_info(tokenizer, decoded, backend)


def main():
    print("Loading the wikitext dataset...")
    # Load the wikitext dataset

    train_text = load_wikitext_dataset(SIZE_LIMIT)
    test_string = "This is a test string for encoding and decoding."

    # Benchmark the Python implementation
    tokenizer = RegexTokenizer()
    benchmark_tokenizer(
        tokenizer,
        train_text,
        test_string,
        backend="python",
        verbose=False,
        vocab_size=10000,
    )

    # Benchmark the C extension implementation
    tokenizer_c = RegexTokenizer()
    benchmark_tokenizer(
        tokenizer_c,
        train_text,
        test_string,
        backend="c",
        verbose=False,
        vocab_size=10000,
    )


if __name__ == "__main__":
    main()
