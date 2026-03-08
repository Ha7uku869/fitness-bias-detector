"""認知バイアスの検知と応答生成を行うモジュール。

bias/prompts.py のシステムプロンプトを llm/client.py に渡して、
ユーザーの発言に対する応答を生成する。

現在は Gemini にプロンプトを渡すだけのシンプルな構成だが、
Phase 2 以降で以下の拡張を予定:
- 過去の会話履歴を RAG で取得して文脈を補強
- 検知したバイアスの種類を構造化データとして記録
"""

from src.bias.prompts import SYSTEM_PROMPT
from src.llm.client import chat


async def respond(user_message: str) -> str:
    """ユーザーのメッセージに対して認知バイアスを考慮した応答を返す。

    Args:
        user_message: ユーザーが LINE で送ったテキスト

    Returns:
        認知バイアスの検知結果を含む応答テキスト
    """
    return await chat(user_message=user_message, system_prompt=SYSTEM_PROMPT)
