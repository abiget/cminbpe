from typing import Set

import regex as re

from .base import Tokenizer, get_stats, merge

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.inverse_special_tokens = {}
        self._special_pattern = None  # compiled regex pattern for special tokens
        # frozenset of special tokens used to compile the special pattern, used to avoid recompiling the same pattern
        self._special_pattern_key = None

    def _train_python(self, text: str, vocab_size: int, verbose=False):
        """
        Chunk-to-chunk consideration is important:
        Word boundary protection:
            If the tokenizer allowed a merge like o +  w -> "o w", a single token
            could swallow the end of one word and the start of the next. This makes
            it harder for the neural network to learn where words begin and end.

        Space efficiency:
            Allowing spaces to merge with every possible ending letter (a, b, c, d, ...)
            would quickly flood the vocabulary with redundant "letter-plus-space" tokens,
            wasting valuable vocabulary slots.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text processing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}  # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stas will update it in place, adding up counts
                get_stats(chunk_ids, stats)

            # no more pairs to merge break
            if not stats:
                break
            pair = max(stats, key=stats.get)

            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurences of the pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # prints
            if verbose:
                print(
                    f"merge {i + 1} / {num_merges} : {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurences"
                )
        self.merges = merges
        self.vocab = vocab

    def _train_cbackend(
        self,
        text: str,
        vocab_size: int,
        next_token_idx: int = 256,
        verbose: bool = False,
    ):
        """
        Train the BPE tokenizer using the C backend for performance.

        This function uses the C implementation of the BPE training algorithm for faster execution.
        It takes the same parameters as the `train` method and returns the merges dictionary.

        Args:
            text (str): The input text to train on.
            vocab_size (int): The desired vocabulary size. It must be greater than or equal to 256.
            next_token_idx (int, optional): The starting index for new tokens. Defaults to
            verbose (bool): If True, prints progress information during training.
        Returns:
            dict: A dictionary mapping pairs of token IDs to their new merged token ID.

        Example:
            tokenizer = RegexTokenizer()
            merges = tokenizer.train_cbackend(text, vocab_size=300, verbose=True)
        """
        from ._minbpe import train as c_backend_train

        assert vocab_size >= 256, "vocab_size must be at least 256"
        assert next_token_idx >= 256, "next_token_idx must be at least 256"

        if len(text) == 0:
            raise ValueError(
                "Input text is empty. Please provide valid text for training."
            )

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        ids = [chunk.encode("utf-8") for chunk in text_chunks]

        merges = c_backend_train(ids, vocab_size, verbose, next_token_idx)

        if merges is None:
            raise RuntimeError(
                "C backend training failed. Please check the input text and parameters."
            )
        self.merges = merges
        self._build_vocab()

    def train(self, text: str, vocab_size: int, backend: str = "python", verbose=False):
        """
        Train the BPE tokenizer using the specified backend.

        Args:
            text (str): The input text to train on.
            vocab_size (int): The desired vocabulary size. It must be greater than or equal to 256.
            backend (str): The backend to use for training. Supported values are "python" and "c". Defaults to "python".
            verbose (bool): If True, prints progress information during training.
        Returns:
            None: The function updates the tokenizer's merges and vocab attributes in place.

        Example:
            tokenizer = RegexTokenizer()
            tokenizer.train(text, vocab_size=300, backend="c", verbose=True)
        """
        if backend == "python":
            self._train_python(text, vocab_size, verbose=verbose)
        elif backend == "c":
            self._train_cbackend(text, vocab_size, verbose=verbose)
        else:
            raise ValueError(
                f"Invalid backend '{backend}'. Supported backends are 'python' and 'c'."
            )

    def _split_and_encode_chunks(self, text: str):
        """
        Split the input text into chunks based on the compiled
        regex pattern and encode each chunk into a list of bytes.

        Args:
            text (str): The input text to split and encode.
        Returns:
            list: A list of lists, where each inner list contains the byte representation of a chunk of text.
        """
        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)
        # encoding the text chunks into bytes
        bytes_list = [chunk.encode("utf-8") for chunk in text_chunks]
        return bytes_list

    def _encode_bytes_cbackend(self, bytes_list: list[bytes]):
        """
        Encode the input text using the C backend for performance.

        This function uses the C implementation of the BPE encoding algorithm for faster execution.
        It takes the input text and returns a list of token IDs.

        Args:
            bytes_list (list[bytes]): A list of lists, where each inner list contains the byte representation of a chunk of text.
        Returns:
            list: A list of token IDs representing the encoded text.
        Example:
            tokenizer = RegexTokenizer()
            token_ids = tokenizer._encode_bytes_cbackend(bytes_list)
        """

        from ._minbpe import encode as c_backend_encode

        # call the C backend encode function
        token_ids = c_backend_encode(bytes_list, self.merges)

        if token_ids is None:
            raise RuntimeError(
                "C backend encoding failed. Please check the input text and merges."
            )

        return token_ids

    def _resolve_special_tokens(self, allowed_special: Set[str] | str, text: str):
        """
        Resolve the special tokens based on the allowed_special parameter and the input text.
        Args:
            allowed_special (Set[str] | str): A set of allowed special tokens or a string indicating the allowed special tokens ("all", "none", "none_raise").
            text (str): The input text to check for special tokens.
        Returns:
            dict: A dictionary of resolved special tokens (str -> int) based on the allowed_special parameter and the input text.
        """
        if allowed_special == "all":
            return self.special_tokens
        elif allowed_special == "none":
            return {}
        elif allowed_special == "none_raise":
            assert all(
                token not in text for token in self.special_tokens
            ), "Special tokens found in text when allowed_special is set to 'none_raise'."
            return {}
        elif isinstance(allowed_special, set):
            return {
                k: v for k, v in self.special_tokens.items() if k in allowed_special
            }
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

    def _get_special_pattern(self, special: Set[str]):
        """
        Get a compiled regex pattern for the given special tokens.
        This function caches the compiled pattern to avoid recompilation for the same set of special tokens.
        Args:
            special (Set[str]): A set of special tokens.
        Returns:
            re.Pattern: A compiled regex pattern that matches any of the special tokens.
        """
        key = frozenset(special)
        if self._special_pattern_key != key:
            ordered = sorted(
                special, key=len, reverse=True
            )  # sort by length to avoid partial matches
            self._special_pattern = re.compile(
                "(" + "|".join(re.escape(k) for k in ordered) + ")"
            )
            self._special_pattern_key = key

        return self._special_pattern

    def _encode_cbackend_specials(
        self, text: str, allowed_special: Set[str] | str = "none_raise"
    ):
        """
        Encode the input text using the C backend while handling special tokens.

        Args:
            text (str): The input text to encode.
            allowed_special (Set[str] | str): A set of allowed special tokens or a string indicating the allowed special tokens ("all", "none", "none_raise").
        Returns:
            list: A list of token IDs representing the encoded text, including special tokens if present.
        """
        special = self._resolve_special_tokens(allowed_special, text)
        if not special:
            chunk_bytes = self._split_and_encode_chunks(text)
            # return only the token ids
            return self._encode_bytes_cbackend(chunk_bytes)[0]

        # otherwise, we have to be careful with potential special tokens in text
        special_pattern = self._get_special_pattern(special)
        segments = re.split(special_pattern, text)

        # flat list of all ordinary byte segments to be encoded
        all_byte_chunks = []
        # number of ordinary chunk bytes per segment, to be used later to reconstruct the final token ids
        segment_sizes = []
        # types of each segment ('special', id) or ('ordinary', None)
        segment_types = []

        for segment in segments:
            if segment in special:
                segment_types.append(("special", special[segment]))
            else:
                segment_bytes = self._split_and_encode_chunks(segment)
                segment_types.append(("ordinary", None))
                all_byte_chunks.extend(segment_bytes)
                segment_sizes.append(len(segment_bytes))

        # encode all ordinary byte chunks using the C backend
        flat_ids, chunks_sizes = self._encode_bytes_cbackend(all_byte_chunks)

        if flat_ids is None or chunks_sizes is None:
            raise RuntimeError(
                "C backend encoding failed. Please check the input text and merges."
            )

        seg_iter = iter(segment_sizes)
        offset = [0]
        for size in chunks_sizes:
            offset.append(offset[-1] + size)

        # reconstruct the final token ids based on segment types and sizes
        final_ids = []
        chunk_ptr = 0
        for i, (segment_type, segment_value) in enumerate(segment_types):
            if segment_type == "special":
                final_ids.append(segment_value)
            else:  # 'ordinary'
                n = next(seg_iter)  # number of ordinary chunks in this segment
                final_ids.extend(flat_ids[offset[chunk_ptr] : offset[chunk_ptr + n]])
                chunk_ptr += n

        return final_ids

    def register_special_tokens(self, special_tokens: dict[str, int]):
        # special_tokens is a dict of str -> int
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def _encode_chunk(self, text_bytes: bytes):
        # return the token ids
        # let's begin.  first convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            # is it possible that a pair can have more than one merge index where in the code that allows for that?  I think not, because the merges are unique.  So we can just find the pair with the lowest merge index
            # so i don't see the point of finding the min when each merges index is unique.  But let's do it anyway, just in case.  It will be a bit slower, but it will be more robust.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                # no more pairs to merge
                break

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text: str):
        """Encoding that ignores any special tokens."""
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then the result joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def _encode_python(self, text: str, allowed_special: Set[str] | str = "none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {
                k: v for k, v in self.special_tokens.items() if k in allowed_special
            }
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            # shortcut: if no special tokens, just use ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunk = re.split(special_pattern, text)

        # now all the special chars are separated from the rest of the text
        # all chunks of text are encoded separately, then resuts are joined
        ids = []
        for part in special_chunk:
            if part in special:
                ids.append(special[part])
            else:
                # this is an ordinary chunk of text, so we can encode it normally
                ids.extend(self.encode_ordinary(part))

        return ids

    def encode(
        self,
        text: str,
        allowed_special: Set[str] | str = "none_raise",
        backend: str = "python",
    ):
        """
        Wrapper for encoding that handles special tokens and allows for backend selection.

        Args:
            text (str): The input text to encode.
            allowed_special (Set[str] | str): A set of allowed special tokens or a string indicating the allowed special tokens ("all", "none", "none_raise").
            backend (str): The backend to use for encoding. Supported values are "python" and "c". Defaults to "python".
        Returns:
            list: A list of token IDs representing the encoded text, including special tokens if present.

        Example:
            tokenizer = RegexTokenizer()
            token_ids = tokenizer.encode(text, allowed_special="all", backend="c")
        """
        if backend == "python":
            return self._encode_python(text, allowed_special=allowed_special)
        elif backend == "c":
            return self._encode_cbackend_specials(text, allowed_special=allowed_special)
        else:
            raise ValueError(
                f"Invalid backend '{backend}'. Supported backends are 'python' and 'c'."
            )

    def decode(self, ids: list[int]):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")

        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
