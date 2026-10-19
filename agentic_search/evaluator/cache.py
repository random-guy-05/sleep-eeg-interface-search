"""SQLite cache for expensive source-only candidate fits."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SourceCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS source_metrics (
                held_site TEXT NOT NULL,
                quartet_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (held_site, quartet_key)
            )""")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def get(self, held_site: str, quartet_key: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM source_metrics WHERE held_site=? AND quartet_key=?",
                             (held_site, quartet_key)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, held_site: str, quartet_key: str, payload: dict[str, Any]) -> None:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO source_metrics (held_site, quartet_key, payload_json) VALUES (?, ?, ?)",
                       (held_site, quartet_key, serialized))
            saved = db.execute("SELECT payload_json FROM source_metrics WHERE held_site=? AND quartet_key=?",
                               (held_site, quartet_key)).fetchone()[0]
        if saved != serialized:
            raise ValueError("Source cache key collision produced inconsistent results")
