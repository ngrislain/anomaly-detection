"""Text standardization and positional insertion utilities."""

import re
import string

# Latin-script letter blocks, accented forms included. 0xD7 (x) and 0xF7 (:)
# sit inside the Latin-1 Supplement range but aren't letters, so that range is
# split around them.
_LATIN_LETTER_RANGES = (
    (0x0041, 0x005A),  # A-Z
    (0x0061, 0x007A),  # a-z
    (0x00C0, 0x00D6),  # A-grave .. O-diaeresis
    (0x00D8, 0x00F6),  # O-stroke .. o-diaeresis
    (0x00F8, 0x00FF),  # o-stroke .. y-diaeresis
    (0x0100, 0x017F),  # Latin Extended-A
    (0x0180, 0x024F),  # Latin Extended-B
)

_ALLOWED_PUNCTUATION = ".,;:!?'\"()-"


def _is_allowed_char(c: str) -> bool:
    if c in string.digits or c.isspace() or c in _ALLOWED_PUNCTUATION:
        return True
    code = ord(c)
    return any(lo <= code <= hi for lo, hi in _LATIN_LETTER_RANGES)


def standardize_text(text: str) -> str:
    """Keep only Latin-script letters (accents preserved), digits, basic
    punctuation (`.,;:!?'"()-`), and whitespace; drop everything else and
    collapse runs of horizontal whitespace.
    """
    filtered = "".join(c for c in text if _is_allowed_char(c))
    return re.sub(r"[ \t]+", " ", filtered)


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
