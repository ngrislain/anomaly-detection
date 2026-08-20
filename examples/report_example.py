"""Example: build a standalone HTML anomaly report into out/.

Scores a clean English Wikipedia excerpt against two contaminated copies of it
-- one with a Basque sentence spliced in, one with a French sentence at the same
position -- under three models: an n-gram fitted on English, the same order
n-gram fitted on Basque, and Qwen.

Basque and French go into separate copies rather than the same one, so each is
measured against an identical background and the two are directly comparable.

The Basque-fitted n-gram is the control. It should behave as the mirror image
of the English one -- comfortable exactly where the English model is alarmed --
which shows that "anomalous" is a statement about the training distribution
rather than about the text.

The inserted text comes from eu.wikipedia's "Euskara", the passage explaining
that Basque is a language isolate with no known relatives.

Run with:
    uv run python examples/report_example.py
"""

import re

from anomaly_detection.models import NgramModel
from anomaly_detection.report import DEFAULT_N, build_report
from anomaly_detection.text_manipulation import insert_after_sentence
from anomaly_detection.wikipedia_search import get_wikipedia_article

ENGLISH_QUERIES = ["Machine learning", "Statistics", "Computer science", "Mathematics"]
BASQUE_QUERIES = ["Euskal Herria", "Matematika", "Historia", "Europa"]
EXCERPT_CHARS = 1200
MIN_PARAGRAPH_CHARS = 400


def french_sentence() -> str:
    """A sentence from fr.wikipedia's "Langue basque" making the same point.

    Placed straight after the Basque one, it gives a second anomaly that is far
    closer to English, so the two can be compared against the same background.
    """
    article = get_wikipedia_article("Langue basque", "fr")
    for sentence in re.findall(r"[^.!?\n]+[.!?]+", article):
        if "seul isolat" in sentence:
            return sentence.strip()
    raise RuntimeError("no French sentence about Basque being an isolate found")


def basque_paragraph() -> str:
    """The passage from eu.wikipedia's "Euskara" on Basque being an isolate.

    Selected by content rather than position: "bakartua" is Basque for
    "isolated", and the length floor skips the shorter lead paragraph that
    mentions it only in passing.
    """
    article = get_wikipedia_article("Euskara", "eu")
    for paragraph in article.split("\n"):
        if len(paragraph) >= MIN_PARAGRAPH_CHARS and "bakartua" in paragraph:
            return paragraph
    raise RuntimeError("no paragraph about Basque being a language isolate found")


def main() -> None:
    print("Fetching articles...")
    fit_text = "\n\n".join(get_wikipedia_article(q, "us") for q in ENGLISH_QUERIES)
    basque_corpus = [get_wikipedia_article(q, "eu") for q in BASQUE_QUERIES]
    eval_text = get_wikipedia_article("Artificial intelligence", "us")[:EXCERPT_CHARS]

    basque = " " + re.match(r"[^.!?]+[.!?]+", basque_paragraph()).group().strip()
    french = " " + french_sentence()

    # Each language goes into its own copy of the text, at the same position, so
    # the two are compared against an identical background. Sentence 1 ends at a
    # genuine full stop; later indices fall inside the "e.g." abbreviation,
    # which the simple sentence regex treats as a boundary.
    with_basque = insert_after_sentence(eval_text, 1, basque)
    with_french = insert_after_sentence(eval_text, 1, french)

    print(f"Fitting {DEFAULT_N}-gram on Basque ({sum(map(len, basque_corpus)):,} chars)...")
    basque_ngram = NgramModel(n=DEFAULT_N).fit(basque_corpus)

    print("Scoring and rendering (loads Qwen on first run)...")
    path = build_report(
        fit_text=fit_text,
        eval_text=eval_text,
        beta=0,
        min_scale=1.5,
        max_scale=3.0,
        alt_text={
            "Basque sentence inserted": with_basque,
            "French sentence inserted": with_french,
        },
        extra_models={f"{DEFAULT_N}-gram (Basque)": basque_ngram},
        # The Basque-fitted model is only informative against the Basque text.
        extra_on="Basque sentence inserted",
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
