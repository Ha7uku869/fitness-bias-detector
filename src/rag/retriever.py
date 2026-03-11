"""RAG（検索拡張生成）のメインモジュール。

embedder と store を組み合わせて、会話の保存と類似検索を行う。

--- 技術解説: RAG（Retrieval-Augmented Generation）とは ---
LLM に質問を投げる前に、関連する情報をデータベースから検索（Retrieval）し、
その情報をプロンプトに追加（Augmented）して応答を生成（Generation）する手法。

通常の LLM は学習データに含まれない情報は回答できないが、
RAG を使うと外部データベースの情報を活用できる。

このプロジェクトでは、過去の会話を検索して LLM のプロンプトに追加することで、
「以前の会話を踏まえた応答」を実現する。

例: ユーザーが「ベンチプレスが伸びない」と言った時、
過去に「胸トレでフォームを見直した」という会話があれば、
それを参考にした応答を生成できる。
"""

import logging

from src.rag.embedder import embed
from src.rag.store import search, upsert

logger = logging.getLogger(__name__)


async def store_message(user_id: str, text: str, role: str) -> None:
    """会話メッセージを埋め込みベクトル化して Qdrant に保存する。

    Args:
        user_id: LINE ユーザーID
        text: メッセージのテキスト
        role: "user" または "assistant"

    --- 技術解説: 非同期での保存 ---
    ベクトルの生成と保存は応答速度に影響するため、
    本番環境ではバックグラウンドタスクとして実行するのが理想的。
    ただし MVP では簡潔さを優先して同期的に実行する。
    """
    try:
        vector = await embed(text)
        upsert(user_id, text, role, vector)
    except Exception:
        # RAG の保存失敗はユーザー体験に直接影響しないため、
        # ログに記録して処理を継続する
        logger.exception("RAG への会話保存に失敗（user_id=%s, role=%s）", user_id, role)


async def retrieve_similar(
    user_id: str, query: str, limit: int = 3
) -> list[str]:
    """クエリに類似する過去の会話を検索して返す。

    Args:
        user_id: 検索対象のユーザーID
        query: 検索クエリ（通常はユーザーの最新メッセージ）
        limit: 返す結果の最大数

    Returns:
        類似度が高い過去の会話テキストのリスト。
        検索に失敗した場合は空リストを返す。

    --- 技術解説: なぜ空リストを返す？ ---
    RAG は応答の品質を向上させる「補助機能」であるため、
    RAG が失敗しても基本的な応答は生成できるべき。
    エラー時に空リストを返すことで、RAG なしでも動作を継続できる。
    これは「graceful degradation（優雅な劣化）」パターンと呼ばれる。
    """
    try:
        vector = await embed(query)
        results = search(vector, user_id, limit=limit)
        return [r["text"] for r in results]
    except Exception:
        logger.exception("RAG からの類似会話検索に失敗（user_id=%s）", user_id)
        return []
