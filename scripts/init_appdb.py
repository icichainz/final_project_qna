#!/usr/bin/env python3
"""Create (or migrate-forward) the conversation-history database.

SQLite schema for chainlit's SQLAlchemyDataLayer (threads sidebar, resume,
feedback). Idempotent: CREATE TABLE IF NOT EXISTS only — safe to re-run.

  python scripts/init_appdb.py            # data/app.db (config.APP_DB)
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gcf_qna import config

DDL = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL,
    "createdAt" TEXT
);
CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "parentId" TEXT,
    "streaming" INTEGER NOT NULL DEFAULT 0,
    "waitForAnswer" INTEGER,
    "isError" INTEGER,
    "metadata" TEXT,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INTEGER,
    "defaultOpen" INTEGER,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INTEGER,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "value" INTEGER NOT NULL,
    "comment" TEXT
);
CREATE TABLE IF NOT EXISTS credentials (
    "identifier" TEXT PRIMARY KEY,
    "passwordHash" TEXT NOT NULL,
    "createdAt" TEXT
);
CREATE INDEX IF NOT EXISTS ix_threads_user ON threads("userIdentifier");
CREATE INDEX IF NOT EXISTS ix_steps_thread ON steps("threadId");
CREATE INDEX IF NOT EXISTS ix_elements_thread ON elements("threadId");
CREATE INDEX IF NOT EXISTS ix_feedbacks_for ON feedbacks("forId");
"""


def main() -> None:
    db = Path(config.APP_DB)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(DDL)
        con.commit()
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        print(f"{db}: {', '.join(tables)}  (WAL on)")
    finally:
        con.close()


if __name__ == "__main__":
    main()
