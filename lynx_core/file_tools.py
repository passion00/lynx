"""
file_tools.py

Read-only filesystem inspection tools for Lynx.

Safety design:
- Lynx can list, read, and search files that the current Linux user can read.
- Lynx does not edit files anywhere on the system.
- The only write operation is creating a NEW file inside ~/lynx/playground.
- Existing playground files are not overwritten.
- No shell commands are executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


LYNX_DIR = Path.home() / "lynx"
PLAYGROUND_DIR = LYNX_DIR / "playground"

MAX_READ_BYTES = 2_000_000
MAX_TEXT_CHARS = 30_000
MAX_LIST_ENTRIES = 250
MAX_SEARCH_RESULTS = 80
MAX_TEXT_SEARCH_FILE_BYTES = 1_000_000
MAX_WALK_DIRS = 20_000

# These are virtual/special filesystems. Reading them as normal files can hang,
# produce unstable data, or expose device interfaces. Normal user files remain readable.
BLOCKED_ROOTS = [
    Path("/proc"),
    Path("/sys"),
    Path("/dev"),
    Path("/run"),
]


@dataclass
class FileToolResult:
    title: str
    content: str
    is_error: bool = False


def _resolve_user_path(path_text: str) -> Path:
    path_text = path_text.strip().strip('"').strip("'")
    if not path_text:
        raise ValueError("No path was provided.")
    return Path(path_text).expanduser().resolve(strict=False)


def _is_under(path: Path, base: Path) -> bool:
    path = path.resolve(strict=False)
    base = base.resolve(strict=False)
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _is_blocked_path(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    return any(resolved == root or _is_under(resolved, root) for root in BLOCKED_ROOTS)


def _human_size(size: int | None) -> str:
    if size is None:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _safe_stat(path: Path):
    try:
        return path.stat()
    except OSError:
        return None


def read_text_file(path_text: str) -> str:
    path = _resolve_user_path(path_text)

    if _is_blocked_path(path):
        return f"Refused to read special/virtual filesystem path: {path}"

    if not path.exists():
        return f"File does not exist: {path}"

    if path.is_dir():
        return f"Path is a directory, not a file: {path}"

    stat_result = _safe_stat(path)
    if stat_result is None:
        return f"Could not stat file: {path}"

    if stat_result.st_size > MAX_READ_BYTES:
        return (
            f"Refused to read file larger than {_human_size(MAX_READ_BYTES)}: "
            f"{path} ({_human_size(stat_result.st_size)})"
        )

    try:
        data = path.read_bytes()
    except Exception as error:
        return f"Could not read file {path}: {error}"

    if b"\x00" in data[:4096]:
        return f"Refused to display likely-binary file: {path}"

    text = data.decode("utf-8", errors="replace")
    truncated = False
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        truncated = True

    header = (
        f"File read result\n"
        f"Path: {path}\n"
        f"Size: {_human_size(stat_result.st_size)}\n"
        f"Truncated: {'yes' if truncated else 'no'}\n"
        f"--- file content starts ---\n"
    )
    footer = "\n--- file content ends ---"
    return header + text + footer


def list_directory(path_text: str) -> str:
    path = _resolve_user_path(path_text)

    if _is_blocked_path(path):
        return f"Refused to list special/virtual filesystem path: {path}"

    if not path.exists():
        return f"Directory does not exist: {path}"

    if not path.is_dir():
        return f"Path is not a directory: {path}"

    try:
        entries = list(path.iterdir())
    except Exception as error:
        return f"Could not list directory {path}: {error}"

    def sort_key(item: Path):
        return (not item.is_dir(), item.name.lower())

    entries.sort(key=sort_key)
    shown = entries[:MAX_LIST_ENTRIES]

    lines = [
        "Directory listing",
        f"Path: {path}",
        f"Total entries: {len(entries)}",
        f"Showing: {len(shown)}",
        "--- entries ---",
    ]

    for item in shown:
        stat_result = _safe_stat(item)
        size = _human_size(stat_result.st_size if stat_result else None)
        kind = "DIR " if item.is_dir() else "FILE"
        suffix = "/" if item.is_dir() else ""
        lines.append(f"{kind} {size:>10}  {item.name}{suffix}")

    if len(entries) > len(shown):
        lines.append(f"... {len(entries) - len(shown)} more entries not shown")

    return "\n".join(lines)


def search_file_names(root_text: str, pattern: str) -> str:
    root = _resolve_user_path(root_text)
    pattern = pattern.strip().lower()

    if not pattern:
        return "No filename search pattern was provided."

    if _is_blocked_path(root):
        return f"Refused to search special/virtual filesystem path: {root}"

    if not root.exists():
        return f"Search root does not exist: {root}"

    if not root.is_dir():
        return f"Search root is not a directory: {root}"

    results: list[str] = []
    visited_dirs = 0

    for current_root, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current_root)
        visited_dirs += 1
        if visited_dirs > MAX_WALK_DIRS:
            results.append(f"Stopped after scanning {MAX_WALK_DIRS} directories.")
            break

        # Prune blocked directories.
        dirnames[:] = [
            dirname for dirname in dirnames
            if not _is_blocked_path(current_path / dirname)
        ]

        for name in dirnames + filenames:
            if pattern in name.lower():
                results.append(str(current_path / name))
                if len(results) >= MAX_SEARCH_RESULTS:
                    break
        if len(results) >= MAX_SEARCH_RESULTS:
            break

    if not results:
        return f"No filenames containing {pattern!r} were found under {root}."

    return "\n".join([
        "Filename search result",
        f"Root: {root}",
        f"Pattern: {pattern}",
        f"Results shown: {len(results)}",
        "--- matches ---",
        *results,
    ])


def search_text_files(root_text: str, query: str) -> str:
    root = _resolve_user_path(root_text)
    query = query.strip()
    query_lower = query.lower()

    if not query:
        return "No text search query was provided."

    if _is_blocked_path(root):
        return f"Refused to search special/virtual filesystem path: {root}"

    if not root.exists():
        return f"Search root does not exist: {root}"

    if not root.is_dir():
        return f"Search root is not a directory: {root}"

    results: list[str] = []
    visited_dirs = 0
    scanned_files = 0

    for current_root, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current_root)
        visited_dirs += 1
        if visited_dirs > MAX_WALK_DIRS:
            results.append(f"Stopped after scanning {MAX_WALK_DIRS} directories.")
            break

        dirnames[:] = [
            dirname for dirname in dirnames
            if not _is_blocked_path(current_path / dirname)
        ]

        for filename in filenames:
            file_path = current_path / filename
            stat_result = _safe_stat(file_path)
            if stat_result is None or stat_result.st_size > MAX_TEXT_SEARCH_FILE_BYTES:
                continue

            try:
                data = file_path.read_bytes()
            except Exception:
                continue

            if b"\x00" in data[:4096]:
                continue

            scanned_files += 1
            text = data.decode("utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if query_lower in line.lower():
                    trimmed = line.strip()
                    if len(trimmed) > 240:
                        trimmed = trimmed[:240] + "..."
                    results.append(f"{file_path}:{line_number}: {trimmed}")
                    break

            if len(results) >= MAX_SEARCH_RESULTS:
                break
        if len(results) >= MAX_SEARCH_RESULTS:
            break

    if not results:
        return f"No text matches for {query!r} were found under {root}. Scanned files: {scanned_files}."

    return "\n".join([
        "Text search result",
        f"Root: {root}",
        f"Query: {query}",
        f"Scanned text-like files: {scanned_files}",
        f"Results shown: {len(results)}",
        "--- matches ---",
        *results,
    ])


def write_playground_file(relative_path_text: str, content: str) -> str:
    """
    Create a new text file inside ~/lynx/playground only.
    This refuses absolute paths, parent-directory escape, and overwriting.
    """

    relative_path_text = relative_path_text.strip().strip('"').strip("'")
    if not relative_path_text:
        return "No playground filename was provided."

    relative_path = Path(relative_path_text)
    if relative_path.is_absolute():
        return "Refused: playground output path must be relative, not absolute."

    target = (PLAYGROUND_DIR / relative_path).resolve(strict=False)
    playground = PLAYGROUND_DIR.resolve(strict=False)

    if not _is_under(target, playground):
        return "Refused: output path escapes ~/lynx/playground."

    if target.exists():
        return f"Refused to overwrite existing playground file: {target}"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except Exception as error:
        return f"Could not write playground file {target}: {error}"

    return "\n".join([
        "Playground file created",
        f"Path: {target}",
        f"Characters written: {len(content)}",
    ])


def run_file_tool_from_message(user_message: str) -> FileToolResult | None:
    """
    Parse explicit file-tool commands.

    Supported commands:
      read file: /path/to/file
      inspect file: /path/to/file
      list directory: /path/to/directory
      ls: /path/to/directory
      search files: pattern in /path/to/root
      search text: query in /path/to/root
      write playground: relative/path.txt <<< content

    Returns None when the user message is not an explicit file-tool command.
    """

    stripped = user_message.strip()
    lowered = stripped.lower()

    try:
        if lowered.startswith("read file:"):
            path = stripped.split(":", 1)[1].strip()
            return FileToolResult("File read", read_text_file(path))

        if lowered.startswith("inspect file:"):
            path = stripped.split(":", 1)[1].strip()
            return FileToolResult("File read", read_text_file(path))

        if lowered.startswith("list directory:"):
            path = stripped.split(":", 1)[1].strip()
            return FileToolResult("Directory listing", list_directory(path))

        if lowered.startswith("ls:"):
            path = stripped.split(":", 1)[1].strip()
            return FileToolResult("Directory listing", list_directory(path))

        if lowered.startswith("search files:"):
            rest = stripped.split(":", 1)[1].strip()
            marker = " in "
            idx = rest.lower().rfind(marker)
            if idx == -1:
                return FileToolResult(
                    "Filename search error",
                    "Use: search files: pattern in /path/to/root",
                    is_error=True,
                )
            pattern = rest[:idx].strip()
            root = rest[idx + len(marker):].strip()
            return FileToolResult("Filename search", search_file_names(root, pattern))

        if lowered.startswith("search text:"):
            rest = stripped.split(":", 1)[1].strip()
            marker = " in "
            idx = rest.lower().rfind(marker)
            if idx == -1:
                return FileToolResult(
                    "Text search error",
                    "Use: search text: query in /path/to/root",
                    is_error=True,
                )
            query = rest[:idx].strip()
            root = rest[idx + len(marker):].strip()
            return FileToolResult("Text search", search_text_files(root, query))

        if lowered.startswith("write playground:"):
            rest = stripped.split(":", 1)[1].strip()
            marker = "<<<"
            if marker not in rest:
                return FileToolResult(
                    "Playground write error",
                    "Use: write playground: relative/path.txt <<< content to write",
                    is_error=True,
                )
            filename, content = rest.split(marker, 1)
            return FileToolResult("Playground write", write_playground_file(filename, content.strip()))

    except Exception as error:
        return FileToolResult("File tool error", str(error), is_error=True)

    return None
