"""
wikipedia_tool.py

Controlled Wikipedia lookup tool for Lynx.

This is not a general webcrawler. It searches Wikipedia for a topic,
fetches the article summary, and returns a small source object that can
be injected into Lynx's prompt and saved into SQLite.
"""

from dataclasses import dataclass
from urllib.parse import quote

import requests


USER_AGENT = (
    "LynxLocalAssistant/0.1 "
    "(local personal AI assistant; contact: local-user)"
)


@dataclass
class WikipediaResult:
    query: str
    title: str
    summary: str
    url: str
    language: str = "en"


def _headers() -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }


def _search_title(query: str, language: str = "en") -> str | None:
    """Search Wikipedia and return the best matching page title."""

    endpoint = f"https://{language}.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
    }

    response = requests.get(
        endpoint,
        params=params,
        headers=_headers(),
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("query", {}).get("search", [])

    if not results:
        return None

    return results[0].get("title")


def _fetch_summary(title: str, language: str = "en") -> dict:
    """Fetch the Wikipedia REST summary for a page title."""

    encoded_title = quote(title.replace(" ", "_"), safe="")
    endpoint = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"

    response = requests.get(
        endpoint,
        headers=_headers(),
        timeout=20,
    )
    response.raise_for_status()

    return response.json()


def lookup_wikipedia_summary(
    query: str,
    language: str = "en",
) -> WikipediaResult | None:
    """
    Search Wikipedia for query and return a summary result.

    Returns None if no suitable article is found.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        return None

    title = _search_title(cleaned_query, language=language)

    if not title:
        return None

    data = _fetch_summary(title, language=language)

    summary = data.get("extract", "").strip()
    result_title = data.get("title", title).strip()
    url = data.get("content_urls", {}).get("desktop", {}).get("page", "").strip()

    if not summary:
        return None

    return WikipediaResult(
        query=cleaned_query,
        title=result_title,
        summary=summary,
        url=url,
        language=language,
    )


def extract_wikipedia_query(user_message: str) -> str | None:
    """
    Detect explicit Wikipedia lookup commands.

    Supported examples:
    - look up Wikipedia: Alan Turing
    - lookup Wikipedia: Alan Turing
    - search Wikipedia: Alan Turing
    - wikipedia: Alan Turing
    - wiki: Alan Turing
    """

    text = user_message.strip()

    prefixes = [
        "look up wikipedia:",
        "lookup wikipedia:",
        "search wikipedia:",
        "wikipedia:",
        "wiki:",
    ]

    lowered = text.lower()

    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()

    return None


def format_wikipedia_context(result: WikipediaResult) -> str:
    """Format a Wikipedia result for prompt injection."""

    return (
        f"Wikipedia article title: {result.title}\n"
        f"Query: {result.query}\n"
        f"URL: {result.url}\n\n"
        f"Summary:\n{result.summary}"
    )
