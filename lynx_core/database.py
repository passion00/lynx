"""
database.py

SQLite database layer for Lynx.

This module stores raw conversations and messages permanently.
It does not perform summarization or intelligent memory extraction yet.
"""

import sqlite3
from datetime import datetime
from pathlib import Path


def current_timestamp() -> str:
    """
    Return current local timestamp as ISO-like text.
    """
    return datetime.now().isoformat(timespec="seconds")


class LynxDatabase:
    def __init__(self, db_path: Path | None = None):
        """
        Create a LynxDatabase object.

        db_path:
            Optional custom path for the SQLite database.
            If not provided, Lynx uses ~/lynx/data/lynx.db
        """

        if db_path is None:
            db_path = Path.home() / "lynx" / "data" / "lynx.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

        self._enable_foreign_keys()
        self._create_tables()

    def _enable_foreign_keys(self) -> None:
        """
        Enable SQLite foreign key enforcement.
        """
        self.connection.execute("PRAGMA foreign_keys = ON;")

    def _create_tables(self) -> None:
        """
        Create database tables if they do not already exist.
        """

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT
            );
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY (conversation_id)
                    REFERENCES conversations (id)
                    ON DELETE CASCADE
            );
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
            ON messages (conversation_id);
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_created_at
            ON messages (created_at);
            """
        )

        self.connection.commit()

    def start_conversation(self, title: str | None = None) -> int:
        """
        Start a new conversation and return its ID.
        """

        if title is None:
            title = "Untitled conversation"

        cursor = self.connection.execute(
            """
            INSERT INTO conversations (title, started_at)
            VALUES (?, ?);
            """,
            (title, current_timestamp()),
        )

        self.connection.commit()
        return int(cursor.lastrowid)

    def end_conversation(self, conversation_id: int) -> None:
        """
        Mark a conversation as ended.
        """

        self.connection.execute(
            """
            UPDATE conversations
            SET ended_at = ?
            WHERE id = ?;
            """,
            (current_timestamp(), conversation_id),
        )

        self.connection.commit()

    def save_message(self, conversation_id: int, role: str, content: str) -> int:
        """
        Save one message to the database.

        role should normally be:
            "user"
            "assistant"
            "system"
        """

        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Invalid message role: {role}")

        cursor = self.connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?);
            """,
            (conversation_id, role, content, current_timestamp()),
        )

        self.connection.commit()
        return int(cursor.lastrowid)

    def get_conversation_messages(self, conversation_id: int) -> list[dict[str, str]]:
        """
        Return all messages from a conversation.
        """

        cursor = self.connection.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC;
            """,
            (conversation_id,),
        )

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in cursor.fetchall()
        ]

    def list_recent_conversations(self, limit: int = 10) -> list[dict[str, str]]:
        """
        Return recent conversations.
        """

        cursor = self.connection.execute(
            """
            SELECT id, title, started_at, ended_at
            FROM conversations
            ORDER BY id DESC
            LIMIT ?;
            """,
            (limit,),
        )

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
            }
            for row in cursor.fetchall()
        ]

    def close(self) -> None:
        """
        Close the database connection.
        """
        self.connection.close()
