"""認知バイアスの検知と応答生成を行うモジュール。

bias/prompts.py のシステムプロンプトを llm/client.py に渡して、
ユーザーの発言に対する応答を生成する。

会話履歴を受け取って LLM に渡すことで、
文脈を踏まえた応答が可能になっている。
"""

from src.bias.prompts import SYSTEM_PROMPT
from src.llm.client import chat


async def respond(
    user_message: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """ユーザーのメッセージに対して認知バイアスを考慮した応答を返す。

    Args:
        user_message: ユーザーが LINE で送ったテキスト
        history: 過去の会話履歴

    Returns:
        認知バイアスの検知結果を含む応答テキスト
    """
    return await chat(
        user_message=user_message,
        system_prompt=SYSTEM_PROMPT,
        history=history,
    )
