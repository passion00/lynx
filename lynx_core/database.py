"""
database.py

SQLite database layer for Lynx.

This module stores:
- raw conversations
- raw messages
- conversation summaries
- external web source summaries
"""

import sqlite3
from datetime import datetime
from pathlib import Path


def current_timestamp() -> str:
    """Return current local timestamp as ISO-like text."""
    return datetime.now().isoformat(timespec="seconds")


class LynxDatabase:
    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = Path.home() / "lynx" / "data" / "lynx.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

        self._enable_foreign_keys()
        self._create_tables()

    def _enable_foreign_keys(self) -> None:
        self.connection.execute("PRAGMA foreign_keys = ON;")

    def _create_tables(self) -> None:
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
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY (conversation_id)
                    REFERENCES conversations (id)
                    ON DELETE CASCADE
            );
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS web_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                source_type TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT,
                query TEXT,
                summary TEXT NOT NULL,
                fetched_at TEXT NOT NULL,

                FOREIGN KEY (conversation_id)
                    REFERENCES conversations (id)
                    ON DELETE SET NULL
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

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_summaries_conversation_id
            ON conversation_summaries (conversation_id);
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_web_sources_conversation_id
            ON web_sources (conversation_id);
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_web_sources_source_type
            ON web_sources (source_type);
            """
        )

        self.connection.commit()

    def start_conversation(self, title: str | None = None) -> int:
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

    def save_conversation_summary(self, conversation_id: int, summary: str) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO conversation_summaries
                (conversation_id, summary, created_at)
            VALUES (?, ?, ?);
            """,
            (conversation_id, summary, current_timestamp()),
        )

        self.connection.commit()
        return int(cursor.lastrowid)

    def save_web_source(
        self,
        source_type: str,
        title: str,
        summary: str,
        url: str | None = None,
        query: str | None = None,
        conversation_id: int | None = None,
    ) -> int:
        """Save an external web source summary."""

        cursor = self.connection.execute(
            """
            INSERT INTO web_sources
                (conversation_id, source_type, title, url, query, summary, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                conversation_id,
                source_type,
                title,
                url,
                query,
                summary,
                current_timestamp(),
            ),
        )

        self.connection.commit()
        return int(cursor.lastrowid)

    def get_conversation_messages(self, conversation_id: int) -> list[dict[str, str]]:
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

    def list_recent_conversations(self, limit: int = 10) -> list[dict]:
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

    def list_recent_summaries(self, limit: int = 30) -> list[dict]:
        cursor = self.connection.execute(
            """
            SELECT
                conversation_summaries.id AS summary_id,
                conversation_summaries.conversation_id AS conversation_id,
                conversation_summaries.summary AS summary,
                conversation_summaries.created_at AS created_at,
                conversations.title AS conversation_title,
                conversations.started_at AS conversation_started_at
            FROM conversation_summaries
            JOIN conversations
                ON conversation_summaries.conversation_id = conversations.id
            ORDER BY conversation_summaries.id DESC
            LIMIT ?;
            """,
            (limit,),
        )

        return [
            {
                "summary_id": row["summary_id"],
                "conversation_id": row["conversation_id"],
                "summary": row["summary"],
                "created_at": row["created_at"],
                "conversation_title": row["conversation_title"],
                "conversation_started_at": row["conversation_started_at"],
            }
            for row in cursor.fetchall()
        ]

    def list_recent_web_sources(self, limit: int = 20) -> list[dict]:
        cursor = self.connection.execute(
            """
            SELECT
                id,
                conversation_id,
                source_type,
                title,
                url,
                query,
                summary,
                fetched_at
            FROM web_sources
            ORDER BY id DESC
            LIMIT ?;
            """,
            (limit,),
        )

        return [
            {
                "id": row["id"],
                "conversation_id": row["conversation_id"],
                "source_type": row["source_type"],
                "title": row["title"],
                "url": row["url"],
                "query": row["query"],
                "summary": row["summary"],
                "fetched_at": row["fetched_at"],
            }
            for row in cursor.fetchall()
        ]

    def close(self) -> None:
        self.connection.close()
