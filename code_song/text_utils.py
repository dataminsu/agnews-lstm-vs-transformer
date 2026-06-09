"""Pure-Python, torchtext-free replacements for the two torchtext utilities the
pipeline used: the ``basic_english`` tokenizer and ``build_vocab_from_iterator``.

WHY THIS EXISTS
---------------
torchtext's final release (0.18.0) is pinned to torch 2.3.0 and ships a compiled
extension (``libtorchtext.so``). On Blackwell GPUs (e.g. RTX 5090, sm_120) you must
install torch >= 2.7 (CUDA 12.8), and there is no torchtext build for that torch,
so ``import torchtext`` fails with an ABI ``undefined symbol`` error. This module
removes the dependency entirely.

EXACT COMPATIBILITY
-------------------
The tokenizer and vocab logic here are byte-for-byte copies of torchtext 0.18.0's
implementation (torchtext/data/utils.py and torchtext/vocab/vocab_factory.py), so
the produced tokens and vocabulary indices are IDENTICAL to the runs the team
already did with torchtext. ``verify_text_utils.py`` proves this against a real
torchtext install. Only stdlib (re, collections) is imported here — no torch.
"""
from __future__ import annotations

import re
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional


# --------------------------------------------------------------------------- #
# Tokenizer — exact copy of torchtext.data.utils._basic_english_normalize
# --------------------------------------------------------------------------- #
_patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]
_replacements = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]
_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))


def _basic_english_normalize(line: str) -> List[str]:
    """Lowercase, pad/strip a fixed set of punctuation, split on whitespace.
    Identical to torchtext 0.18.0's basic_english tokenizer."""
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


def get_tokenizer(tokenizer, language: str = "en"):
    """Drop-in for torchtext.data.utils.get_tokenizer. Only the names the project
    uses are supported: ``None`` (str.split) and ``"basic_english"``; a callable is
    returned unchanged."""
    if tokenizer is None:
        return str.split
    if tokenizer == "basic_english":
        if language != "en":
            raise ValueError("Basic normalization is only available for English (en)")
        return _basic_english_normalize
    if callable(tokenizer):
        return tokenizer
    raise ValueError(
        f"tokenizer {tokenizer!r} is not supported by the torchtext-free shim "
        f"(only None and 'basic_english')."
    )


# --------------------------------------------------------------------------- #
# Vocab — exact behavior of torchtext.vocab.Vocab for the methods the pipeline
# uses: v[token], v(tokens), set_default_index, get_itos, len, plus the usual
# lookup helpers. Index of a token == its position in the itos list.
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, itos: List[str]):
        self._itos = list(itos)
        self._stoi = {tok: i for i, tok in enumerate(self._itos)}
        self._default_index: Optional[int] = None

    # --- defaults ---
    def set_default_index(self, index: Optional[int]) -> None:
        self._default_index = index

    def get_default_index(self) -> Optional[int]:
        return self._default_index

    # --- single-token lookup ---
    def __getitem__(self, token: str) -> int:
        idx = self._stoi.get(token)
        if idx is not None:
            return idx
        if self._default_index is not None:
            return self._default_index
        # Matches torchtext: OOV without a default index is an error.
        raise RuntimeError(
            f"Token {token!r} not found in Vocab and default index is not set."
        )

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __len__(self) -> int:
        return len(self._itos)

    # --- batched lookup: v(tokens) == v.lookup_indices(tokens) ---
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self.__getitem__(t) for t in tokens]

    def __call__(self, tokens: List[str]) -> List[int]:
        return self.lookup_indices(tokens)

    # --- reverse lookup / introspection ---
    def get_itos(self) -> List[str]:
        return list(self._itos)

    def get_stoi(self) -> Dict[str, int]:
        return dict(self._stoi)

    def lookup_token(self, index: int) -> str:
        return self._itos[index]

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return [self._itos[i] for i in indices]


def vocab(ordered_dict: Dict, min_freq: int = 1, specials: Optional[List[str]] = None,
          special_first: bool = True) -> Vocab:
    """Exact copy of torchtext.vocab.vocab factory semantics."""
    specials = specials or []
    for token in list(specials):
        ordered_dict.pop(token, None)

    tokens = [tok for tok, freq in ordered_dict.items() if freq >= min_freq]

    if special_first:
        tokens[0:0] = list(specials)
    else:
        tokens.extend(specials)
    return Vocab(tokens)


def build_vocab_from_iterator(iterator: Iterable, min_freq: int = 1,
                              specials: Optional[List[str]] = None,
                              special_first: bool = True,
                              max_tokens: Optional[int] = None) -> Vocab:
    """Exact copy of torchtext.vocab.build_vocab_from_iterator.

    Tokens are ranked by (descending frequency, then ascending token string),
    capped at ``max_tokens - len(specials)``, filtered by ``min_freq``, and the
    specials are prepended (special_first=True). The resulting index == position.
    """
    counter: Counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    specials = specials or []

    # First sort by descending frequency, then lexicographically (identical key).
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    if max_tokens is None:
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
    else:
        assert len(specials) < max_tokens, \
            "len(specials) >= max_tokens, so the vocab will be entirely special tokens."
        ordered_dict = OrderedDict(sorted_by_freq_tuples[: max_tokens - len(specials)])

    return vocab(ordered_dict, min_freq=min_freq, specials=specials, special_first=special_first)
