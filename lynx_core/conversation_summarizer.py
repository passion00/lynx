"""
conversation_summarizer.py

Summarizes the current Lynx conversation before shutdown.

This module is called when the user types "kill".
It requires the llama-server to still be running.
"""

import requests

from lynx_core.model_server import CHAT_URL, MODEL_NAME


SUMMARY_SYSTEM_PROMPT = """
You are Lynx's conversation summarizer.

Your job is to summarize the user's most recent chat session.
Focus on:
- important decisions
- project progress
- code changes
- bugs or errors encountered
- unresolved next steps
- durable information useful for future sessions
- identity/profile facts if the user shared them

Do not include small talk unless it affected the project.
Return only the summary.
""".strip()


def format_messages_for_summary(messages: list[dict[str, str]]) -> str:
    """
    Convert messages into plain readable text.
    """

    formatted_parts = []

    for message in messages:
        role = message.get("role", "").upper()
        content = message.get("content", "").strip()

        if role not in {"USER", "ASSISTANT"}:
            continue

        if not content:
            continue

        formatted_parts.append(f"{role}:\n{content}")

    return "\n\n".join(formatted_parts)


def trim_from_end(text: str, max_chars: int) -> str:
    """
    Keep the most recent part of a long conversation.
    """

    if len(text) <= max_chars:
        return text

    return (
        "[Earlier part of conversation omitted because it was too long.]\n\n"
        + text[-max_chars:]
    )


def summarize_conversation(
    messages: list[dict[str, str]],
    max_summary_tokens: int = 250,
    max_input_chars: int = 12000,
) -> str:
    """
    Send conversation messages to the model and return a short summary.
    """

    conversation_text = format_messages_for_summary(messages)

    if not conversation_text.strip():
        return ""

    conversation_text = trim_from_end(conversation_text, max_input_chars)

    user_prompt = f"""
Summarize this Lynx chat session in no more than approximately {max_summary_tokens} tokens.

Conversation:

{conversation_text}
""".strip()

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SUMMARY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0.2,
        "max_tokens": max_summary_tokens,
    }

    response = requests.post(CHAT_URL, json=payload, timeout=None)
    response.raise_for_status()

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as error:
        raise RuntimeError(f"Unexpected model response format: {data}") from error


def summarize_and_store_current_conversation(
    db,
    conversation_id: int,
    active_messages: list[dict[str, str]] | None = None,
    max_summary_tokens: int = 250,
) -> str:
    """
    Summarize the current conversation before shutdown.

    Priority:
    1. Use messages passed from lynx.py's session transcript.
    2. If that is empty, fall back to SQLite messages.
    3. Save the summary to the database.
    """

    messages: list[dict[str, str]] = []

    if active_messages is not None:
        messages = [
            message
            for message in active_messages
            if message.get("role") in {"user", "assistant"}
            and message.get("content", "").strip()
        ]

    if not messages:
        messages = [
            message
            for message in db.get_conversation_messages(conversation_id)
            if message.get("role") in {"user", "assistant"}
            and message.get("content", "").strip()
        ]

    if not messages:
        return ""

    summary = summarize_conversation(
        messages=messages,
        max_summary_tokens=max_summary_tokens,
    )

    if summary:
        db.save_conversation_summary(conversation_id, summary)

    return summary
