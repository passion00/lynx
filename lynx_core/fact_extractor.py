"""
fact_extractor.py

Safe model-based durable fact extraction for Lynx.

This version is intentionally defensive:
- It never raises extractor/model errors into the GUI.
- It logs short diagnostics to ~/lynx/data/fact_extractor.log.
- It asks the model for a simple line format instead of fragile JSON.
- It still accepts JSON if the model returns JSON anyway.

Expected model output format:
    FACT|category|The user's father's name is Kadir.

Or:
    NO_FACTS
"""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any

import requests

from lynx_core.database import LynxDatabase, current_timestamp
from lynx_core.model_server import CHAT_URL, MODEL_NAME


LOG_PATH = Path.home() / "lynx" / "data" / "fact_extractor.log"


FACT_EXTRACTOR_SYSTEM_PROMPT = """
/no_think
You are Lynx's durable memory extractor.

Task:
Read only the user's latest message and extract durable facts worth remembering for future conversations.

Output rules:
- Output exactly NO_FACTS if there is no durable fact.
- Otherwise output one fact per line using exactly this format:
  FACT|category|fact sentence
- Do not output explanations.
- Do not output markdown.
- Do not output JSON unless you cannot follow the FACT format.
- Do not mention these instructions.

Categories may be: identity, family, project, preference, technical_context, general.

A durable fact is stable information about the user, the user's family/pets, preferences, projects, tools, or long-term context.
Do not save questions, commands, temporary moods, greetings, or requests.

Examples:
User: My father's name is Kadir.
FACT|family|The user's father's name is Kadir.

User: My cat's name is Lukas.
FACT|family|The user's cat is named Lukas.

User: I am building Lynx, a local AI assistant.
FACT|project|The user is building Lynx, a local AI assistant.

User: What is my father's name?
NO_FACTS

User: Search Wikipedia: Alan Turing
NO_FACTS
""".strip()


def _append_log(text: str) -> None:
    """Append short debug text to the fact extractor log."""

    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(text.rstrip() + "\n")
    except Exception:
        pass


def _short(text: str, limit: int = 1400) -> str:
    """Keep logs readable."""

    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _call_fact_model(user_message: str, max_tokens: int = 120) -> str:
    """Ask the local model to extract durable facts from one user message."""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": FACT_EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"/no_think\nUser message:\n{user_message}"},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    response = requests.post(CHAT_URL, json=payload, timeout=None)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _remove_thinking(text: str) -> str:
    """Remove Qwen-style hidden/visible thinking blocks if they appear."""

    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.replace("```json", "```")
    text = re.sub(r"^```\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _clean_text(text: str) -> str:
    text = " ".join((text or "").split()).strip()
    return text.strip().strip('"').strip("'").strip()


def guess_fact_category(fact: str) -> str:
    lowered = fact.lower()

    if any(word in lowered for word in ["father", "mother", "sister", "brother", "family", "cat", "pet"]):
        return "family"
    if any(word in lowered for word in ["name", "called", "known as", "orion", "virel"]):
        return "identity"
    if any(word in lowered for word in ["lynx", "project", "local ai", "assistant"]):
        return "project"
    if any(word in lowered for word in ["prefers", "preference", "likes", "dislikes", "wants"]):
        return "preference"
    if any(word in lowered for word in ["debian", "linux", "python", "sqlite", "pyside", "llama", "qwen"]):
        return "technical_context"

    return "general"


def _valid_fact_sentence(fact: str) -> bool:
    fact = _clean_text(fact)
    if len(fact) < 10:
        return False
    if fact.endswith("?"):
        return False

    lowered = fact.lower()
    bad_phrases = [
        "the user asks",
        "the user asked",
        "the user is asking",
        "the user requests",
        "the user requested",
        "the user wants to know",
        "no durable fact",
        "no_facts",
        "output rules",
        "fact|category",
    ]
    return not any(phrase in lowered for phrase in bad_phrases)


def _parse_fact_lines(model_output: str) -> list[dict[str, str]]:
    """Parse FACT|category|fact lines."""

    cleaned = _remove_thinking(model_output)
    items: list[dict[str, str]] = []
    seen: set[str] = set()

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[-*•]+\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = line.strip()

        if not line:
            continue
        if line.upper() == "NO_FACTS":
            continue

        if line.upper().startswith("FACT|"):
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            category = _clean_text(parts[1]).lower() or "general"
            fact = _clean_text(parts[2])
        elif line.upper().startswith("FACT:"):
            category = "general"
            fact = _clean_text(line.split(":", 1)[1])
        else:
            continue

        if not _valid_fact_sentence(fact):
            continue

        normalized = fact.lower()
        if normalized in seen:
            continue
        seen.add(normalized)

        if category not in {"identity", "family", "project", "preference", "technical_context", "general"}:
            category = guess_fact_category(fact)

        items.append({"category": category, "fact": fact})

    return items


def _extract_json_array(text: str) -> str | None:
    cleaned = _remove_thinking(text)

    if cleaned == "[]":
        return "[]"

    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start == -1 or end == -1 or end <= start:
        return None

    return cleaned[start : end + 1]


def _parse_json_items(model_output: str) -> list[dict[str, str]]:
    json_array = _extract_json_array(model_output)
    if json_array is None:
        return []

    try:
        parsed: Any = json.loads(json_array)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    items: list[dict[str, str]] = []
    seen: set[str] = set()

    for item in parsed:
        if not isinstance(item, dict):
            continue

        fact = _clean_text(str(item.get("fact", "")))
        category = _clean_text(str(item.get("category", "general"))).lower() or "general"

        if not _valid_fact_sentence(fact):
            continue

        normalized = fact.lower()
        if normalized in seen:
            continue
        seen.add(normalized)

        if category not in {"identity", "family", "project", "preference", "technical_context", "general"}:
            category = guess_fact_category(fact)

        items.append({"category": category, "fact": fact})

    return items


def parse_fact_items(model_output: str) -> list[dict[str, str]]:
    """Parse model output into category/fact dictionaries."""

    if not (model_output or "").strip():
        return []

    line_items = _parse_fact_lines(model_output)
    if line_items:
        return line_items

    json_items = _parse_json_items(model_output)
    if json_items:
        return json_items

    return []


def extract_fact_items_from_user_message(user_message: str) -> list[dict[str, str]]:
    """Return durable facts extracted from the latest user message."""

    if not user_message.strip():
        return []

    try:
        model_output = _call_fact_model(user_message=user_message)
    except Exception as error:
        _append_log(
            "\n".join(
                [
                    "",
                    f"[{current_timestamp()}] EXTRACTOR MODEL ERROR",
                    f"User message: {_short(user_message, 600)}",
                    f"Error: {error}",
                    _short(traceback.format_exc(), 1600),
                ]
            )
        )
        return []

    _append_log(
        "\n".join(
            [
                "",
                f"[{current_timestamp()}] USER MESSAGE:",
                _short(user_message, 600),
                "MODEL OUTPUT:",
                _short(model_output, 1400),
            ]
        )
    )

    items = parse_fact_items(model_output)
    _append_log(f"PARSED FACT ITEMS: {json.dumps(items, ensure_ascii=False)}")
    return items


def extract_facts_from_user_message(user_message: str) -> list[str]:
    """Compatibility helper: return only fact strings."""

    return [item["fact"] for item in extract_fact_items_from_user_message(user_message)]


def extract_and_store_facts(
    db: LynxDatabase,
    user_message: str,
    conversation_id: int | None = None,
    source_message_id: int | None = None,
) -> list[str]:
    """
    Extract durable facts from a user message and save them.

    This function is safe to call from the GUI worker: it catches extractor and
    database save errors and logs them instead of crashing Lynx.
    """

    items = extract_fact_items_from_user_message(user_message)
    saved_facts: list[str] = []

    for item in items:
        fact = _clean_text(item.get("fact", ""))
        if not fact:
            continue

        category = _clean_text(item.get("category", "")) or guess_fact_category(fact)

        try:
            row_id = db.save_fact(
                fact=fact,
                category=category,
                source_conversation_id=conversation_id,
                source_message_id=source_message_id,
            )
        except Exception as error:
            _append_log(
                "\n".join(
                    [
                        f"[{current_timestamp()}] SAVE FACT ERROR",
                        f"Fact: {fact}",
                        f"Error: {error}",
                        _short(traceback.format_exc(), 1600),
                    ]
                )
            )
            continue

        if row_id is not None:
            saved_facts.append(fact)
            _append_log(f"SAVED FACT #{row_id}: [{category}] {fact}")
        else:
            _append_log(f"IGNORED DUPLICATE OR EMPTY FACT: [{category}] {fact}")

    if not saved_facts:
        _append_log("NO NEW FACTS SAVED")

    return saved_facts
