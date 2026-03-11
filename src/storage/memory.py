"""ユーザーごとの会話履歴をメモリ上に保持するモジュール。

--- 技術解説: インメモリ vs DB ---
現段階ではサーバーのメモリ（辞書）に会話を保持する。
メリット: 実装がシンプル、DBが不要
デメリット: サーバー再起動で消える

Phase 2 で SQLite や Qdrant に永続化すれば、
サーバー再起動後も会話を復元できるようになる。
今はまず「会話の流れを覚える」機能を最小限で実装する。

--- 技術解説: collections.deque ---
deque（デック）は「両端キュー」。maxlen を指定すると、
上限を超えた古い要素が自動的に削除される。
リストだと手動で古い要素を消す必要があるが、deque なら不要。
"""

from collections import deque

# ユーザーIDをキー、会話履歴（deque）を値とする辞書。
# 各ユーザーの直近 MAX_HISTORY ターン分の会話を保持する。
MAX_HISTORY = 10  # 直近10ターン（user + assistant で5往復分）

_histories: dict[str, deque[dict[str, str]]] = {}


def get_history(user_id: str) -> list[dict[str, str]]:
    """指定ユーザーの会話履歴をリストで返す。

    Args:
        user_id: LINE のユーザーID（一意な文字列）

    Returns:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    if user_id not in _histories:
        return []
    return list(_histories[user_id])


def add_message(user_id: str, role: str, content: str) -> None:
    """会話履歴にメッセージを1件追加する。

    Args:
        user_id: LINE のユーザーID
        role: "user" または "assistant"
        content: メッセージのテキスト
    """
    if user_id not in _histories:
        _histories[user_id] = deque(maxlen=MAX_HISTORY)
    _histories[user_id].append({"role": role, "content": content})
