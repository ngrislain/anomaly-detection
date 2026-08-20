"""Example: compare character n-gram models and Qwen3.5-0.8B at spotting an
out-of-place Basque sentence inserted into an English Wikipedia article.

Run with:
    uv run python examples/anomaly_detection_example.py
"""

import math
import re

from anomaly_detection.detector import detect_anomalies
from anomaly_detection.models import NgramModel, QwenModel
from anomaly_detection.text_manipulation import insert_after_sentence
from anomaly_detection.wikipedia_search import get_wikipedia_article

TRAINING_QUERIES = ["Machine learning", "Statistics", "Computer science", "Mathematics"]
NGRAM_ORDERS = (2, 3, 5)
BASQUE_SENTENCE = (
    " Gaur egun, kartveliar hizkuntzen eta euskararen arteko"
    " ahaidetasun lotura ukatzen dute adituek."
)
# Sentence 1 ends at a genuine full stop; later indices fall inside the "e.g."
# abbreviation, which the simple sentence regex treats as a boundary.
INSERT_AFTER_SENTENCE = 1
EXCERPT_CHARS = 900
# The sentence regex splits abbreviations such as "e.g." into tiny fragments;
# their means are dominated by one or two characters, so ignore them when
# ranking which sentence is most surprising.
MIN_SENTENCE_CHARS = 20


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of each sentence, so scores can be aggregated per sentence."""
    return [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]+", text)]


def mean_log_prob(pairs: list[tuple[str, float]], start: int, end: int) -> float:
    values = [v for _, v in pairs[start:end] if math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


def report(name: str, pairs: list[tuple[str, float]], text: str, anomaly: tuple[int, int]) -> None:
    spans = [(s, e) for s, e in sentence_spans(text) if e - s >= MIN_SENTENCE_CHARS]
    scored = [(mean_log_prob(pairs, s, e), s, e) for s, e in spans]
    ranked = sorted(scored)  # most surprising (lowest mean log-prob) first

    print(f"--- {name} ---")
    print(f"mean log-prob over whole text: {mean_log_prob(pairs, 0, len(text)):.3f}")
    worst_score, worst_start, worst_end = ranked[0]
    caught = worst_start >= anomaly[0] and worst_end <= anomaly[1] + 1
    print(f"most surprising sentence ({worst_score:.3f}): {text[worst_start:worst_end].strip()!r}")
    print(f"flagged the inserted sentence: {caught}")
    for score, s, e in ranked[:3]:
        marker = "<-- inserted" if s >= anomaly[0] and e <= anomaly[1] + 1 else ""
        print(f"  {score:7.3f}  {text[s:e].strip()[:60]!r} {marker}")
    print()


def main() -> None:
    print("Fetching training articles...")
    corpus = [get_wikipedia_article(q, "us") for q in TRAINING_QUERIES]

    article = get_wikipedia_article("Artificial intelligence", "us")[:EXCERPT_CHARS]
    text = insert_after_sentence(article, INSERT_AFTER_SENTENCE, BASQUE_SENTENCE)

    # Character span of the inserted sentence, used to check whether each model
    # actually flags it rather than something else.
    start = text.index(BASQUE_SENTENCE)
    anomaly = (start, start + len(BASQUE_SENTENCE))
    print(f"Inserted at chars {anomaly[0]}-{anomaly[1]}: {BASQUE_SENTENCE.strip()!r}\n")

    for n in NGRAM_ORDERS:
        print(f"Training {n}-gram model...")
        ngram = NgramModel(n=n).fit(corpus)
        report(f"{n}-gram", detect_anomalies(ngram, text), text, anomaly)

    print("Loading Qwen (first run downloads ~1.8GB)...")
    qwen = QwenModel()
    print(f"running on device: {qwen.device}")
    report(qwen.model_name, detect_anomalies(qwen, text), text, anomaly)


if __name__ == "__main__":
    main()
