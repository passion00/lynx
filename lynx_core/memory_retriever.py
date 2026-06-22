"""
memory_retriever.py

Simple automatic memory retrieval for Lynx.

Design:
- Use saved conversation summaries as an index.
- Score summaries locally in Python.
- Load full chat history only for the most relevant conversations.
- Ask the model to distill useful context.
- Return that context so lynx.py can inject it before answering.

No commands.
No manual recall.
No JSON selection step.
"""

import re

import requests

from lynx_core.model_server import CHAT_URL, MODEL_NAME


DISTILLATION_SYSTEM_PROMPT = """
You are Lynx's memory distiller.

You will receive:
1. The user's current message.
2. Summaries and old chat history from previous Lynx conversations.

Your job is NOT to answer the user.
Your job is to extract useful remembered context for the main assistant.

Rules:
- Extract only information relevant to the user's current message.
- If the user asks "Who am I?", include identity/profile/project facts about the user.
- Preserve names, project names, decisions, preferences, and technical context.
- Be concise.
- Do not invent facts.
- If no useful memory exists, return exactly: EMPTY
""".strip()


def tokenize(text: str) -> set[str]:
    """
    Convert text into lowercase searchable tokens.
    """
    return set(re.findall(r"[a-zA-Z0-9_ğüşöçıİĞÜŞÖÇ]+", text.lower()))


def is_identity_question(text: str) -> bool:
    """
    Detect questions like:
    - Who am I?
    - What do you know about me?
    - Do you remember me?
    """

    lowered = text.lower().strip()

    identity_phrases = [
        "who am i",
        "who i am",
        "what do you know about me",
        "do you remember me",
        "remember me",
        "my name",
        "about me",
        "tell me about myself",
        "beni hatırlıyor musun",
        "ben kimim",
    ]

    return any(phrase in lowered for phrase in identity_phrases)


def expanded_query_tokens(user_message: str) -> set[str]:
    """
    Add helpful search terms based on the kind of user message.
    """

    tokens = tokenize(user_message)

    if is_identity_question(user_message):
        tokens.update(
            {
                "user",
                "name",
                "identity",
                "profile",
                "pseudonym",
                "orion",
                "virel",
                "lynx",
                "assistant",
                "project",
                "building",
                "local",
            }
        )

    return tokens


def score_summary(user_message: str, summary: str) -> int:
    """
    Score how relevant a summary is to the current user message.

    This is intentionally simple and deterministic.
    """

    query_tokens = expanded_query_tokens(user_message)
    summary_tokens = tokenize(summary)

    score = 0

    for token in query_tokens:
        if token in summary_tokens:
            score += 2

    lowered_summary = summary.lower()

    if is_identity_question(user_message):
        identity_markers = [
            "user",
            "name",
            "orion",
            "virel",
            "profile",
            "pseudonym",
            "building lynx",
            "local ai assistant",
        ]

        for marker in identity_markers:
            if marker in lowered_summary:
                score += 5

    return score


def select_relevant_summaries(
    user_message: str,
    summaries: list[dict],
    max_results: int = 3,
) -> list[dict]:
    """
    Select the most relevant saved summaries.

    If the user asks an identity/profile question and scores are weak,
    fall back to recent summaries. This helps Lynx discover profile info
    from a fresh database.
    """

    scored = []

    for summary_item in summaries:
        score = score_summary(user_message, summary_item["summary"])
        scored.append((score, summary_item))

    scored.sort(key=lambda item: item[0], reverse=True)

    selected = [item for score, item in scored if score > 0][:max_results]

    if selected:
        return selected

    # Identity questions are special: "Who am I?" has few keywords.
    # If no score matched, inspect a few recent summaries anyway.
    if is_identity_question(user_message):
        return summaries[:max_results]

    return []


def format_chat_history(messages: list[dict[str, str]], max_chars: int = 12000) -> str:
    """
    Convert old database messages into readable text.
    """

    parts = []

    for message in messages:
        role = message["role"].upper()
        content = message["content"].strip()

        if not content:
            continue

        parts.append(f"{role}:\n{content}")

    text = "\n\n".join(parts)

    if len(text) <= max_chars:
        return text

    return (
        "[Earlier part omitted because the conversation was long.]\n\n"
        + text[-max_chars:]
    )


def ask_model_for_distillation(
    user_message: str,
    memory_text: str,
    max_tokens: int = 350,
) -> str:
    """
    Ask the model to distill useful context from retrieved memory.
    """

    prompt = f"""
User's current message:
{user_message}

Retrieved previous memory:
{memory_text}
""".strip()

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": DISTILLATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }

    response = requests.post(CHAT_URL, json=payload, timeout=None)
    response.raise_for_status()

    data = response.json()

    try:
        distilled = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as error:
        raise RuntimeError(f"Unexpected model response format: {data}") from error

    if distilled.upper() == "EMPTY":
        return ""

    return distilled


def retrieve_relevant_context(
    db,
    user_message: str,
    summary_limit: int = 30,
) -> str:
    """
    Retrieve useful remembered context for the user's current message.

    Returns:
        A concise context string, or "" if nothing useful was found.
    """

    summaries = db.list_recent_summaries(limit=summary_limit)

    if not summaries:
        return ""

    selected_summaries = select_relevant_summaries(
        user_message=user_message,
        summaries=summaries,
        max_results=3,
    )

    if not selected_summaries:
        return ""

    memory_blocks = []

    for item in selected_summaries:
        conversation_id = item["conversation_id"]

        messages = db.get_conversation_messages(conversation_id)
        chat_history = format_chat_history(messages)

        block = f"""
Conversation ID: {conversation_id}
Conversation started at: {item["conversation_started_at"]}

Saved summary:
{item["summary"]}

Full chat history:
{chat_history}
""".strip()

        memory_blocks.append(block)

    memory_text = "\n\n====================\n\n".join(memory_blocks)

    return ask_model_for_distillation(
        user_message=user_message,
        memory_text=memory_text,
        max_tokens=350,
    )
