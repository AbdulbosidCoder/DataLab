"""SQLite: таблица segments (схема как в speech-lab)."""
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video       TEXT NOT NULL,
    audio_path  TEXT NOT NULL UNIQUE,
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    duration    REAL NOT NULL,
    speaker     TEXT,
    auto_text   TEXT,
    text        TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending / approved / rejected / overlap
    annotator   TEXT,
    notes       TEXT,
    claimed_by  TEXT,
    claimed_at  TEXT,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_segments_status ON segments(status);
CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video);

-- Загрузки из веб-интерфейса: файл -> очередь обработки VoiceLab
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    video       TEXT,
    status      TEXT NOT NULL DEFAULT 'queued',  -- queued / processing / done / error
    message     TEXT,
    segments    INTEGER,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    if "sha256" not in {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}:  # хэш файла для поиска повторных загрузок
        conn.execute("ALTER TABLE jobs ADD COLUMN sha256 TEXT")
    return conn


def insert_segments(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Добавляет сегменты; уже существующие audio_path пропускаются."""
    cur = conn.executemany(
        """INSERT OR IGNORE INTO segments
           (video, audio_path, start, end, duration, speaker, auto_text, text, status, notes)
           VALUES (:video, :audio_path, :start, :end, :duration, :speaker,
                   :auto_text, :text, :status, :notes)""",
        rows,
    )
    conn.commit()
    return cur.rowcount
