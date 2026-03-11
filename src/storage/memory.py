"""ユーザーごとの会話履歴を SQLite に永続化するモジュール。

--- 技術解説: SQLite ---
SQLite はファイルベースの軽量データベース。
サーバーが再起動しても会話履歴が消えない。
Python 標準ライブラリに含まれているため追加インストール不要。

Render の無料プランではサーバーが頻繁に再起動するが、
SQLite ファイルが残っていれば会話を復元できる。
"""

import json
import sqlite3
from pathlib import Path

MAX_HISTORY = 10  # 直近10メッセージ（5往復分）をLLMに渡す

# DB ファイルのパス（プロジェクトルートの data/ 以下に保存）
_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH = _DB_DIR / "conversations.db"

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """SQLite コネクションを取得する（初回のみ DB を初期化）。"""
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH))
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_id ON messages (user_id)"
        )
        _conn.commit()
    return _conn


def get_history(user_id: str) -> list[dict[str, str]]:
    """指定ユーザーの直近の会話履歴をリストで返す。"""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT role, content FROM messages
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, MAX_HISTORY),
    ).fetchall()
    # DESC で取得したので逆順にして時系列順にする
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def add_message(user_id: str, role: str, content: str) -> None:
    """会話履歴にメッセージを1件追加する。"""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content),
    )
    conn.commit()
