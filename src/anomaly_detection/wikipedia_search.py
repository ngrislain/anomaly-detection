"""Search Wikipedia and fetch the plain-text body of the top-matching article.

Results are memoized to disk with joblib so repeated lookups for the same
(query, country_code) pair never hit the network twice.
"""

import re
import unicodedata

import requests
from joblib import Memory

_CACHE_DIR = ".cache/wikipedia_search"
_memory = Memory(location=_CACHE_DIR, verbose=0)

_USER_AGENT = "anomaly-detection/0.1 (wikipedia_search module)"

# Wikipedia editions are keyed by language code, not country code, so common
# country codes are mapped to the language of their Wikipedia edition here.
# Anything not listed is assumed to already be a valid Wikipedia language code.
_COUNTRY_TO_WIKI_LANG = {
    "us": "en", "gb": "en", "au": "en", "ca": "en", "nz": "en", "ie": "en",
    "fr": "fr", "be": "fr", "ch": "de",
    "de": "de", "at": "de",
    "es": "es", "mx": "es", "ar": "es", "cl": "es", "co": "es", "pe": "es",
    "it": "it",
    "pt": "pt", "br": "pt",
    "nl": "nl",
    "ru": "ru",
    "cn": "zh", "tw": "zh", "hk": "zh",
    "jp": "ja",
    "kr": "ko",
    "in": "hi",
    "sa": "ar", "ae": "ar", "eg": "ar",
    "pl": "pl",
    "se": "sv",
    "no": "no",
    "dk": "da",
    "fi": "fi",
    "gr": "el",
    "tr": "tr",
    "il": "he",
    "id": "id",
    "vn": "vi",
    "th": "th",
}


def _wiki_language(country_code: str) -> str:
    code = country_code.strip().lower()
    return _COUNTRY_TO_WIKI_LANG.get(code, code)


def _to_ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[ \t]+", " ", ascii_text)


def _search_top_title(lang: str, query: str) -> str | None:
    response = requests.get(
        f"https://{lang}.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json().get("query", {}).get("search", [])
    return results[0]["title"] if results else None


def _fetch_article_text(lang: str, title: str) -> str:
    response = requests.get(
        f"https://{lang}.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    return page.get("extract", "")


@_memory.cache
def get_wikipedia_article(query: str, country_code: str) -> str:
    """Return the ASCII-only text of the top Wikipedia search result.

    Args:
        query: search terms.
        country_code: ISO country code (or Wikipedia language code) used to
            pick the Wikipedia language edition to search.

    Returns:
        The article's plain text with all non-ASCII characters stripped, or
        an empty string if no article was found.
    """
    lang = _wiki_language(country_code)
    title = _search_top_title(lang, query)
    if title is None:
        return ""
    text = _fetch_article_text(lang, title)
    return _to_ascii(text)
