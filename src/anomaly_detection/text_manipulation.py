"""Text standardization and positional insertion utilities."""

import re
import unicodedata


def standardize_text(text: str) -> str:
    """Strip accents/non-ASCII characters and collapse runs of horizontal whitespace.

    Uses NFKD normalization so accented characters degrade to their closest
    ASCII equivalent (e.g. "e" -> "e") before non-ASCII characters are dropped.
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[ \t]+", " ", ascii_text)


def insert_after_line(text: str, k: int, insertion: str) -> str:
    """Insert `insertion` right after line `k` (0-indexed) of `text`."""
    lines = text.splitlines(keepends=True)
    if not 0 <= k < len(lines):
        raise IndexError(f"line index {k} out of range for text with {len(lines)} lines")
    pos = sum(len(line) for line in lines[: k + 1])
    return text[:pos] + insertion + text[pos:]


def insert_after_word(text: str, l: int, insertion: str) -> str:
    """Insert `insertion` right after word `l` (0-indexed) of `text`.

    Words are runs of non-whitespace characters.
    """
    words = list(re.finditer(r"\S+", text))
    if not 0 <= l < len(words):
        raise IndexError(f"word index {l} out of range for text with {len(words)} words")
    pos = words[l].end()
    return text[:pos] + insertion + text[pos:]


def insert_after_sentence(text: str, m: int, insertion: str) -> str:
    """Insert `insertion` right after sentence `m` (0-indexed) of `text`.

    Sentences are runs of characters terminated by one or more of `.`, `!`, `?`.
    Trailing text with no terminating punctuation is not counted as a sentence.
    """
    sentences = list(re.finditer(r"[^.!?]+[.!?]+", text))
    if not 0 <= m < len(sentences):
        raise IndexError(f"sentence index {m} out of range for text with {len(sentences)} sentences")
    pos = sentences[m].end()
    return text[:pos] + insertion + text[pos:]
