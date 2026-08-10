import unicodedata


def get_stats(ids, counts=None):
    """
    Given a list of integer ids, return a dictionary of the counts of consecutive pairs of ids.
    Example: [1, 2, 3, 2, 3] -> {(1, 2): 1, (2, 3): 2}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of interger ids, replace all consecutive occurrences of pair
    with the new integer token idx.
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
    s = t.decode("utf-8", errors="replace")
    s = replace_control_characters(s)
    return s


class Tokenizer:

    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.pattern = ""  # str
        self.special_tokens = {}  # str -> int, e.g, {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()
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
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab

    def save(self, file_prefix):
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
        assert model_file.endswith(".model")

        merges = {}
        self.special_tokens = {}
        idx = 256
        with open(model_file, "r", encoding="uft-8") as f:
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
                self.special_tokens[special] = special_idx

            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

            self.merges = merges
            self.special_tokens = self.special_tokens
            self.vocab = self._build_vocab()
