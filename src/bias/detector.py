"""認知バイアスの検知と応答生成を行うモジュール。

bias/prompts.py のシステムプロンプトを llm/client.py に渡して、
ユーザーの発言に対する応答を生成する。

会話履歴を受け取って LLM に渡すことで、
文脈を踏まえた応答が可能になっている。

--- 技術解説: RAG の統合 ---
RAG（検索拡張生成）を組み込むことで、直近の会話履歴だけでなく、
過去の類似する会話を参考にした応答を生成できる。
例えば1ヶ月前にベンチプレスの悩みを相談していた場合、
今回また同じ話題が出た時に過去の文脈を踏まえた応答ができる。
"""

import logging

from src.bias.prompts import SYSTEM_PROMPT
from src.llm.client import chat
from src.rag.retriever import retrieve_similar, store_message

logger = logging.getLogger(__name__)


async def respond(
    user_message: str,
    user_id: str = "",
    history: list[dict[str, str]] | None = None,
) -> str:
    """ユーザーのメッセージに対して認知バイアスを考慮した応答を返す。

    RAG を使って過去の類似会話を検索し、コンテキストとして
    LLM に渡すことで、より文脈を踏まえた応答を生成する。

    Args:
        user_message: ユーザーが LINE で送ったテキスト
        user_id: LINE ユーザーID（RAG の検索・保存に使用）
        history: 直近の会話履歴

    Returns:
        認知バイアスの検知結果を含む応答テキスト

    --- 技術解説: RAG コンテキストの渡し方 ---
    検索で見つかった過去の会話をユーザーメッセージに付加する。
    LLM はこの追加コンテキストを参考にして応答を生成する。
    RAG が失敗しても（空リストが返っても）基本的な応答は生成できる。
    """
    # --- Step 1: 過去の類似会話を検索 ---
    rag_context = ""
    if user_id:
        similar_messages = await retrieve_similar(user_id, user_message)
        if similar_messages:
            # 検索結果をコンテキスト文字列に整形
            context_lines = "\n".join(f"- {msg}" for msg in similar_messages)
            rag_context = (
                "\n\n【参考: このユーザーの過去の関連する会話】\n"
                f"{context_lines}\n"
            )
            logger.info(
                "RAG: %d 件の類似会話を取得しました（user_id=%s）",
                len(similar_messages),
                user_id,
            )

    # --- Step 2: RAG コンテキストを付加してメッセージを構築 ---
    augmented_message = user_message
    if rag_context:
        augmented_message = user_message + rag_context

    # --- Step 3: LLM で応答を生成 ---
    reply = await chat(
        user_message=augmented_message,
        system_prompt=SYSTEM_PROMPT,
        history=history,
    )

    # --- Step 4: 今回の会話を RAG ストアに保存 ---
    if user_id:
        await store_message(user_id, user_message, "user")
        await store_message(user_id, reply, "assistant")

    return reply
