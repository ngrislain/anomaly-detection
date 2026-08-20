"""Example: fetch a Wikipedia article's ASCII text for a query and country code.

Run with:
    uv run python examples/wikipedia_search_example.py
"""

from anomaly_detection.wikipedia_search import get_wikipedia_article


def main() -> None:
    examples = [
        ("Eiffel Tower", "fr"),
        ("Machine learning", "us"),
        ("Datadog", "fr"),
        ("Datadog", "us"),
    ]
    for query, country_code in examples:
        text = get_wikipedia_article(query, country_code)
        print(f"=== {query!r} ({country_code}) ===")
        print(text[:300], "...\n")


if __name__ == "__main__":
    main()
