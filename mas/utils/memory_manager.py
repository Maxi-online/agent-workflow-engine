"""Persistent, user-scoped SQLite memory without destructive initialization."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


class MemoryManager:
    def __init__(self, path=None):
        self.path = Path(path or os.getenv("AGENT_MEMORY_PATH", str(ROOT / "var/memory.sqlite3")))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.http_client = None
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL,
                    role TEXT NOT NULL,content TEXT NOT NULL,timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS summaries(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL,
                    summary TEXT NOT NULL,timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS last_agents(
                    user_id TEXT PRIMARY KEY,agent_name TEXT NOT NULL,timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);
                CREATE INDEX IF NOT EXISTS messages_user_idx ON messages(user_id,id);
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def set_http_client(self, client: Any):
        self.http_client = client

    def save_memory(self, user_id, message, response):
        with self.connect() as connection:
            connection.executemany(
                "INSERT INTO messages(user_id,role,content) VALUES(?,?,?)",
                [(user_id, "user", message), (user_id, "assistant", response)],
            )

    def get_combined_memory(self, user_id, limit=10):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT role,content,timestamp FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit * 2),
            ).fetchall()
            summary = connection.execute(
                "SELECT summary,timestamp FROM summaries WHERE user_id=? ORDER BY id DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        messages = []
        if summary:
            messages.append(
                {
                    "role": "system",
                    "content": "Previous summary:\n" + summary[0],
                    "timestamp": summary[1],
                }
            )
        messages.extend(
            {"role": role, "content": content, "timestamp": timestamp}
            for role, content, timestamp in reversed(rows)
        )
        return messages

    def set_summary(self, user_id, summary):
        if not isinstance(summary, str):
            summary = json.dumps(summary, ensure_ascii=False)
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO summaries(user_id,summary) VALUES(?,?)", (user_id, summary)
            )

    def get_summary(self, user_id):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT summary FROM summaries WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)
            ).fetchone()
        return row[0] if row else None

    def set_last_agent(self, user_id, agent_name):
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO last_agents(user_id,agent_name) VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET agent_name=excluded.agent_name,timestamp=CURRENT_TIMESTAMP",
                (user_id, agent_name),
            )

    def get_last_agent(self, user_id):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT agent_name FROM last_agents WHERE user_id=?", (user_id,)
            ).fetchone()
        return row[0] if row else None

    def clear_memory(self, user_id):
        with self.connect() as connection:
            for table in ("messages", "summaries", "last_agents"):
                connection.execute("DELETE FROM " + table + " WHERE user_id=?", (user_id,))


class LazyMemory:
    """Keep module imports free of database creation and migrations."""

    def __init__(self):
        self.instance = None
        self.lock = threading.Lock()

    def __getattr__(self, name):
        with self.lock:
            if self.instance is None:
                self.instance = MemoryManager()
        return getattr(self.instance, name)


memory_manager = LazyMemory()
