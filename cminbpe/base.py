import unicodedata


def get_stats(ids: list, counts=None) -> dict:
    """
    Given a list of integer ids, return a dictionary of the counts of consecutive pairs of ids.
    Args:
        ids: list of integer ids
        counts: optional existing counts to update
    Returns:
        counts: dictionary of counts of consecutive pairs of ids

    Example: [1, 2, 3, 2, 3] -> {(1, 2): 1, (2, 3): 2}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list, pair: tuple, idx: int) -> list:
    """
    In the list of interger ids, replace all consecutive occurrences of pair
    with the new integer token idx.
    Args:
        ids: list of integer ids
        pair: tuple of two integers to merge
        idx: integer token to replace the pair with
    Returns:
        new_ids: list of integer ids with the pair replaced by idx

    Example: ids = [1, 2, 3, 2, 3], pair = (2, 3), idx = 4 -> [1, 4, 4]
    """

    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def replace_control_characters(s: str) -> str:
    """
    Replace control characters in a string with their Unicode escape sequences.
    Args:
        s: input string
    Returns:
        new_s: string with control characters replaced by Unicode escape sequences

    Example: "Hello\x00World" -> "Hello\\u0000World"
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(
                f"\\u{ord(ch):04x}"
            )  # Replace control characters with their Unicode escape sequences
            # why double backslash? Because we want to represent the escape sequence as a string, so we need to escape the backslash itself.

    return "".join(chars)


def render_token(t: bytes) -> str:
    """
    Render a token (bytes) as a string, replacing control characters with their Unicode escape sequences.
    Args:
        t: token as bytes
    Returns:
        s: token as string
    Example: b'Hello\x00World' -> 'Hello\\u0000World'
    """
    s = t.decode("utf-8", errors="replace")
    s = replace_control_characters(s)
    return s


class Tokenizer:

    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.pattern = ""  # str
        self.special_tokens = {}  # str -> int, e.g, {'<|endoftext|>': 100257}
        self.vocab = {}
        # should be empty until train or load is called.
        # int -> bytes, e.g, {0: b'\x00', 1: b'\x01', ...}

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocab size of size vocab_size from text
        raise NotImplementedError()

    def encode(self, text):
        # Tokenizer can encode text into a list of integer
        raise NotImplementedError()

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError()

    def _build_vocab(self):
        # Vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            vocab[idx] = vocab[p0] + vocab[p1]

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        self.vocab = vocab

    def save(self, file_prefix):
        """
        Save the tokenizer to a file with the given prefix.  The model is saved in a .model file, and the vocab is saved in a .vocab file.  The .model
        file contains the merges and special tokens, while the .vocab file contains the vocab for humans to look at.
        Args:
            file_prefix: prefix for the files to save, e.g., "tokenizer" will save "tokenizer.model" and "tokenizer.vocab"
        Returns:
            None
        """
        model_file = file_prefix + ".model"
        with open(model_file, "w") as f:
            # write the version, patttern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the number of special token
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            # merge dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        # write the vocab: for humans to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)

                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """
        Load the tokenizer from a .model file.  The .model file contains the merges and special tokens, while the .vocab file contains the vocab for humans to look at.
        Args:
            model_file: path to the .model file to load
        Returns:
            None
        """
        assert model_file.endswith(".model")

        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            # read version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read special tokens
            num_special = int(f.readline().strip())

            # read special tokens
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)

            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self._build_vocab()
