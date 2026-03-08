"""Groq API とやりとりするクライアントモジュール。

Groq は OpenAI 互換の API を提供しているため、openai ライブラリの
base_url を変えるだけで接続できる。Llama 3 等のオープンソースモデルを
無料で高速に利用できる。

--- 技術解説: OpenAI 互換 API ---
多くの LLM プロバイダ（Groq, Together AI, Ollama 等）が
OpenAI の Chat Completions API と同じインターフェースを採用している。
これにより、openai ライブラリの base_url を差し替えるだけで
異なるプロバイダに接続でき、コードの変更が最小限で済む。
"""

from openai import AsyncOpenAI

from src.config import get_settings

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Groq クライアントを取得する（初回のみ生成）。

    AsyncOpenAI に base_url を指定することで、
    リクエスト先を OpenAI ではなく Groq に向ける。
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    return _client


async def chat(user_message: str, system_prompt: str) -> str:
    """ユーザーのメッセージに対して LLM の応答を返す。

    Args:
        user_message: ユーザーが送ったテキスト
        system_prompt: LLM の振る舞いを制御するシステムプロンプト

    Returns:
        LLM が生成した応答テキスト

    --- 技術解説 ---
    llama-3.3-70b-versatile は Meta の Llama 3.3 70B モデル。
    オープンソースモデルの中ではトップクラスの性能で、
    日本語の対話にも十分対応できる。Groq では無料で利用可能。
    """
    client = _get_client()

    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
