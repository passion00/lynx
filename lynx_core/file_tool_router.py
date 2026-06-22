"""
file_tool_router.py

Autonomous filesystem tool router for Lynx.

This module lets the local model decide whether a normal user message needs a
filesystem tool. The router is intentionally limited: the model may only choose
from safe Python functions in file_tools.py. No shell commands are executed.

Security boundary remains in file_tools.py:
- read/list/search only for normal readable files
- /proc, /sys, /dev, /run blocked
- writes only create new files inside ~/lynx/playground
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import requests

from lynx_core.file_tools import (
    FileToolResult,
    list_directory,
    read_text_file,
    run_file_tool_from_message,
    search_file_names,
    search_text_files,
    write_playground_file,
)
from lynx_core.model_server import get_chat_url
from lynx_core.settings import load_settings


LOG_PATH = Path.home() / "lynx" / "data" / "file_tool_router.log"

VALID_ACTIONS = {
    "read_file",
    "list_directory",
    "search_file_names",
    "search_text_files",
    "write_playground_file",
}


SYSTEM_PROMPT = """You are Lynx's filesystem tool router.
Your job is to decide whether the user's message requires inspecting local files.
You do not answer the user. You only choose a tool or choose no tool.

Available tools:
1. read_file
   Use when the user asks to read, open, inspect, view, analyze, summarize, or check a specific file.
   Required field: path

2. list_directory
   Use when the user asks what files are in a folder, wants to browse a directory, or asks to inspect a folder.
   Required field: path

3. search_file_names
   Use when the user asks to find files/folders by name.
   Required fields: root, pattern

4. search_text_files
   Use when the user asks to search inside files for text/code/terms.
   Required fields: root, query

5. write_playground_file
   Use only when the user asks Lynx to create, save, export, or write a file.
   This tool can only create NEW files inside ~/lynx/playground. It cannot overwrite.
   Required fields: relative_path, content

Rules:
- If the user is asking a normal knowledge/conversation question, return use_tool=false.
- If the user asks about local files, folders, code files, logs, documents, downloads, desktop, home folder, project folder, or filesystem contents, return use_tool=true.
- Prefer the user's home directory "~" when they say "my files", "my computer", or "my PC" without a more specific path.
- Prefer "~/lynx" when the user says "Lynx project" or "project files".
- Prefer "~/Downloads" for downloads, "~/Documents" for documents, and "~/Desktop" for desktop.
- Use "/" only when the user explicitly asks for the whole system, whole PC, or root filesystem.
- For broad requests like "inspect my folder", start with list_directory. Do not recursively scan unless the user asks to search/find.
- Do not invent exact filenames. If the file is not specified, use list_directory or search_file_names instead of read_file.
- Return exactly one JSON object and no other text.

JSON formats:
{"use_tool": false, "reason": "short reason"}
{"use_tool": true, "action": "read_file", "path": "~/example.txt", "reason": "short reason"}
{"use_tool": true, "action": "list_directory", "path": "~/Documents", "reason": "short reason"}
{"use_tool": true, "action": "search_file_names", "root": "~", "pattern": "invoice", "reason": "short reason"}
{"use_tool": true, "action": "search_text_files", "root": "~/lynx", "query": "Qwen", "reason": "short reason"}
{"use_tool": true, "action": "write_playground_file", "relative_path": "notes/result.txt", "content": "text to write", "reason": "short reason"}
"""


EXAMPLES = [
    (
        "Can you look at my Documents folder and tell me what is there?",
        {"use_tool": True, "action": "list_directory", "path": "~/Documents", "reason": "user asks to inspect a folder"},
    ),
    (
        "Open my bashrc and explain it.",
        {"use_tool": True, "action": "read_file", "path": "~/.bashrc", "reason": "user asks to read a specific file"},
    ),
    (
        "Find files named lynx on my computer.",
        {"use_tool": True, "action": "search_file_names", "root": "~", "pattern": "lynx", "reason": "user asks to find files by name"},
    ),
    (
        "Search my Lynx project for Qwen.",
        {"use_tool": True, "action": "search_text_files", "root": "~/lynx", "query": "Qwen", "reason": "user asks to search inside project files"},
    ),
    (
        "Save a short hello note into playground as hello.txt",
        {"use_tool": True, "action": "write_playground_file", "relative_path": "hello.txt", "content": "Hello from Lynx.\n", "reason": "user asks to create a playground file"},
    ),
    (
        "What is the capital of France?",
        {"use_tool": False, "reason": "general knowledge question"},
    ),
]


def _log(message: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(message.rstrip() + "\n")
    except Exception:
        pass


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from possibly messy model output."""

    cleaned = text.strip()

    # Remove common Markdown fences if the model ignores instructions.
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(cleaned)):
        char = cleaned[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:index + 1]
                try:
                    data = json.loads(candidate)
                    return data if isinstance(data, dict) else None
                except Exception:
                    return None

    return None


def _normalize_plan(plan: dict[str, Any]) -> dict[str, Any] | None:
    use_tool = bool(plan.get("use_tool", False))
    if not use_tool:
        return {"use_tool": False, "reason": str(plan.get("reason", "no tool needed"))}

    action = str(plan.get("action", "")).strip()
    if action not in VALID_ACTIONS:
        return None

    normalized: dict[str, Any] = {
        "use_tool": True,
        "action": action,
        "reason": str(plan.get("reason", "file tool selected")),
    }

    if action in {"read_file", "list_directory"}:
        path = str(plan.get("path", "")).strip()
        if not path:
            return None
        normalized["path"] = path

    elif action == "search_file_names":
        root = str(plan.get("root", "~")).strip() or "~"
        pattern = str(plan.get("pattern", "")).strip()
        if not pattern:
            return None
        normalized["root"] = root
        normalized["pattern"] = pattern

    elif action == "search_text_files":
        root = str(plan.get("root", "~")).strip() or "~"
        query = str(plan.get("query", "")).strip()
        if not query:
            return None
        normalized["root"] = root
        normalized["query"] = query

    elif action == "write_playground_file":
        relative_path = str(plan.get("relative_path", "")).strip()
        content = str(plan.get("content", ""))
        if not relative_path:
            return None
        normalized["relative_path"] = relative_path
        normalized["content"] = content

    return normalized


def decide_file_tool(user_message: str) -> dict[str, Any] | None:
    """
    Ask the local model whether a filesystem tool should be used.
    Returns a normalized plan dict, or None if routing failed.
    """

    stripped = user_message.strip()
    if not stripped:
        return {"use_tool": False, "reason": "empty message"}

    settings = load_settings()

    examples_text = "\n".join(
        f"User: {prompt}\nJSON: {json.dumps(result, ensure_ascii=False)}"
        for prompt, result in EXAMPLES
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Examples:\n{examples_text}\n\nCurrent user message:\n{stripped}\n\nReturn JSON now:"},
    ]

    payload = {
        "model": settings.model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 220,
    }

    try:
        response = requests.post(get_chat_url(settings), json=payload, timeout=None)
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
    except Exception as error:
        _log(f"ROUTER ERROR: {error}")
        return None

    _log("USER: " + stripped)
    _log("RAW: " + raw[:2000].replace("\n", "\\n"))

    parsed = _extract_json_object(raw)
    if parsed is None:
        _log("PARSE FAILED")
        return None

    normalized = _normalize_plan(parsed)
    if normalized is None:
        _log("NORMALIZE FAILED: " + repr(parsed))
        return None

    _log("PLAN: " + json.dumps(normalized, ensure_ascii=False))
    return normalized


def execute_file_tool_plan(plan: dict[str, Any]) -> FileToolResult | None:
    """Execute a normalized tool plan using only safe file_tools.py functions."""

    if not plan or not plan.get("use_tool"):
        return None

    action = plan.get("action")

    try:
        if action == "read_file":
            path = str(plan["path"])
            return FileToolResult(f"Autonomous file read: {path}", read_text_file(path))

        if action == "list_directory":
            path = str(plan["path"])
            return FileToolResult(f"Autonomous directory listing: {path}", list_directory(path))

        if action == "search_file_names":
            root = str(plan["root"])
            pattern = str(plan["pattern"])
            return FileToolResult(
                f"Autonomous filename search: {pattern}",
                search_file_names(root, pattern),
            )

        if action == "search_text_files":
            root = str(plan["root"])
            query = str(plan["query"])
            return FileToolResult(
                f"Autonomous text search: {query}",
                search_text_files(root, query),
            )

        if action == "write_playground_file":
            relative_path = str(plan["relative_path"])
            content = str(plan.get("content", ""))
            return FileToolResult(
                f"Autonomous playground write: {relative_path}",
                write_playground_file(relative_path, content),
            )

    except Exception as error:
        return FileToolResult("Autonomous file tool error", str(error), is_error=True)

    return FileToolResult("Autonomous file tool error", f"Unknown action: {action}", is_error=True)


def run_autonomous_file_tool(user_message: str) -> FileToolResult | None:
    """
    Run file tools autonomously.

    Explicit command syntax is still honored first. If no explicit command is
    detected, Lynx asks the local model whether a file tool should be used.
    """

    explicit_result = run_file_tool_from_message(user_message)
    if explicit_result is not None:
        return explicit_result

    plan = decide_file_tool(user_message)
    if plan is None or not plan.get("use_tool"):
        return None

    return execute_file_tool_plan(plan)
