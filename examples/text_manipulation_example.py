"""Example: standardize text and insert a French sentence into an English
Wikipedia article at a chosen line, word, and sentence position.

Run with:
    uv run python examples/text_manipulation_example.py
"""

from anomaly_detection.text_manipulation import (
    insert_after_line,
    insert_after_sentence,
    insert_after_word,
    standardize_text,
)
from anomaly_detection.wikipedia_search import get_wikipedia_article


def main() -> None:
    article = get_wikipedia_article("Machine learning", "us")
    french_sentence = " Ceci est une phrase en francais inseree dans l'article."

    standardized_sentence = standardize_text(french_sentence)
    print("Standardized French sentence:", standardized_sentence)

    after_line = insert_after_line(article, k=0, insertion=french_sentence)
    print("\n=== Inserted after line 0 ===")
    print(after_line[:400], "...\n")

    after_word = insert_after_word(article, l=9, insertion=french_sentence)
    print("=== Inserted after word 9 ===")
    print(after_word[:400], "...\n")

    after_sentence = insert_after_sentence(article, m=0, insertion=french_sentence)
    print("=== Inserted after sentence 0 ===")
    print(after_sentence[:400], "...\n")


if __name__ == "__main__":
    main()
