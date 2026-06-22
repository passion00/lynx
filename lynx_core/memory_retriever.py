"""
memory_retriever.py

Tiered memory retrieval for Lynx.

Retrieval order:
1. durable facts table
2. conversation summaries, newest to oldest
3. raw messages, newest to oldest

The function stops as soon as it finds relevant context at a cheaper layer.
"""

from __future__ import annotations

import re
from typing import Iterable

from lynx_core.database import LynxDatabase


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "his",
    "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "our",
    "she", "so", "that", "the", "their", "there", "they", "this", "to", "was",
    "we", "what", "when", "where", "which", "who", "why", "will", "with", "you",
    "your", "user", "users", "s",
}

SYNONYMS = {
    "name": {"name", "named", "called"},
    "named": {"name", "named", "called"},
    "called": {"name", "named", "called"},
    "father": {"father", "dad", "parent"},
    "dad": {"father", "dad", "parent"},
    "mother": {"mother", "mom", "mum", "parent"},
    "mom": {"mother", "mom", "mum", "parent"},
    "cat": {"cat", "pet"},
    "pet": {"cat", "pet"},
    "lynx": {"lynx", "project", "assistant"},
    "project": {"lynx", "project", "assistant"},
}


def tokenize(text: str) -> list[str]:
    """Tokenize text for simple relevance scoring."""

    lowered = text.lower().replace("’", "'")
    raw_tokens = re.findall(r"[a-zA-Z0-9_çğıöşüÇĞİÖŞÜ']+", lowered)

    tokens: list[str] = []
    for token in raw_tokens:
        token = token.strip("'")
        if token.endswith("'s"):
            token = token[:-2]
        if not token:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)

    return tokens


def expand_tokens(tokens: Iterable[str]) -> set[str]:
    """Expand query tokens with small synonym groups."""

    expanded: set[str] = set()

    for token in tokens:
        expanded.add(token)
        if token in SYNONYMS:
            expanded.update(SYNONYMS[token])
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
        if token.endswith("ed") and len(token) > 4:
            expanded.add(token[:-2])

    return expanded


def relevance_score(query: str, text: str) -> int:
    """Return a simple relevance score between the query and a candidate text."""

    query_tokens = expand_tokens(tokenize(query))
    text_tokens = expand_tokens(tokenize(text))

    if not query_tokens or not text_tokens:
        return 0

    score = len(query_tokens & text_tokens)

    lowered_query = query.lower()
    lowered_text = text.lower()

    # Useful phrase boosts for common memory questions.
    if "father" in query_tokens and "father" in text_tokens:
        score += 2
    if "mother" in query_tokens and "mother" in text_tokens:
        score += 2
    if "name" in query_tokens and ({"name", "named", "called"} & text_tokens):
        score += 2
    if "lynx" in query_tokens and "lynx" in text_tokens:
        score += 2
    if "who am i" in lowered_query and ("orion" in lowered_text or "virel" in lowered_text):
        score += 4

    return score


def first_relevant_fact(
    db: LynxDatabase,
    user_message: str,
    fact_limit: int = 200,
) -> dict | None:
    """Scan active facts from newest to oldest and return the first relevant fact."""

    for fact in db.list_active_facts(limit=fact_limit):
        if relevance_score(user_message, fact["fact"]) > 0:
            return fact

    return None


def first_relevant_summary(
    db: LynxDatabase,
    user_message: str,
    summary_limit: int = 50,
) -> dict | None:
    """Scan summaries from newest to oldest and return the first relevant summary."""

    for summary in db.list_recent_summaries(limit=summary_limit):
        if relevance_score(user_message, summary["summary"]) > 0:
            return summary

    return None


def first_relevant_raw_message(
    db: LynxDatabase,
    user_message: str,
    message_limit: int = 500,
    current_message_id: int | None = None,
) -> dict | None:
    """Scan raw messages from newest to oldest and return the first relevant message."""

    for message in db.list_recent_messages(limit=message_limit):
        if current_message_id is not None and message["message_id"] == current_message_id:
            continue
        if relevance_score(user_message, message["content"]) > 0:
            return message

    return None


def retrieve_relevant_context(
    db: LynxDatabase,
    user_message: str,
    summary_limit: int = 50,
    fact_limit: int = 200,
    message_limit: int = 500,
    current_message_id: int | None = None,
) -> str:
    """
    Retrieve memory context for the current user message.

    The search stops at the first memory layer that produces a relevant result:
    facts, then summaries, then raw messages.
    """

    relevant_fact = first_relevant_fact(
        db=db,
        user_message=user_message,
        fact_limit=fact_limit,
    )

    if relevant_fact is not None:
        return (
            "Relevant durable fact found in Lynx memory: "
            f"{relevant_fact['fact']}"
        )

    relevant_summary = first_relevant_summary(
        db=db,
        user_message=user_message,
        summary_limit=summary_limit,
    )

    if relevant_summary is not None:
        return (
            "Relevant conversation summary found in Lynx memory: "
            f"Conversation ID {relevant_summary['conversation_id']}. "
            f"Summary: {relevant_summary['summary']}"
        )

    relevant_message = first_relevant_raw_message(
        db=db,
        user_message=user_message,
        message_limit=message_limit,
        current_message_id=current_message_id,
    )

    if relevant_message is not None:
        return (
            "Relevant raw chat message found in Lynx memory: "
            f"Conversation ID {relevant_message['conversation_id']}, "
            f"role {relevant_message['role']}. "
            f"Message: {relevant_message['content']}"
        )

    return ""
