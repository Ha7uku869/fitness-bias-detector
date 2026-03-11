"""FastAPI アプリケーションのエントリーポイント。

LINE Bot の Webhook を受け取り、認知バイアス検知の応答を返す。

--- 技術解説: Webhook とは ---
Webhook は「イベント発生時に指定した URL に HTTP リクエストを送る仕組み」。
LINE でユーザーがメッセージを送ると、LINE プラットフォームが
このサーバーの /callback エンドポイントに POST リクエストを送ってくる。

--- 技術解説: FastAPI ---
FastAPI は Python の非同期 Web フレームワーク。
Flask に似ているが、以下の特徴がある:
- 型ヒントからAPIドキュメントを自動生成（Swagger UI）
- async/await をネイティブサポート
- Pydantic による入力バリデーション
"""

import logging

from fastapi import FastAPI, HTTPException, Request
from linebot.v3 import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from src.bias.detector import respond
from src.config import get_settings
from src.storage.memory import add_message, get_history

logger = logging.getLogger(__name__)

app = FastAPI(title="筋トレメタ認知AI")


def _get_line_clients() -> tuple[WebhookParser, MessagingApi]:
    """LINE SDK のクライアントを初期化して返す。

    WebhookParser: LINE からの Webhook リクエストを解析する
    MessagingApi:  LINE にメッセージを返信する

    --- 技術解説: 署名検証 ---
    LINE は Webhook リクエストに署名（X-Line-Signature ヘッダ）を付ける。
    channel_secret を使ってこの署名を検証することで、
    リクエストが本当に LINE から来たものかを確認できる。
    これがないと、第三者が偽のリクエストを送れてしまう（セキュリティ上重要）。
    """
    settings = get_settings()
    parser = WebhookParser(channel_secret=settings.line_channel_secret)
    config = Configuration(access_token=settings.line_channel_access_token)
    api_client = ApiClient(configuration=config)
    messaging_api = MessagingApi(api_client)
    return parser, messaging_api


# アプリ起動時に1回だけ初期化
_parser: WebhookParser | None = None
_messaging_api: MessagingApi | None = None


def _ensure_line_clients() -> tuple[WebhookParser, MessagingApi]:
    """LINE クライアントを遅延初期化する。"""
    global _parser, _messaging_api
    if _parser is None or _messaging_api is None:
        _parser, _messaging_api = _get_line_clients()
    return _parser, _messaging_api


@app.get("/health")
async def health() -> dict[str, str]:
    """ヘルスチェック用エンドポイント。サーバーが動いているか確認する。"""
    return {"status": "ok"}


@app.post("/callback")
async def callback(request: Request) -> dict[str, str]:
    """LINE Webhook のエンドポイント。

    LINE プラットフォームからメッセージイベントを受け取り、
    認知バイアス検知の応答を返す。

    処理の流れ:
    1. LINE からの署名を検証（不正なリクエストを弾く）
    2. リクエストボディを解析してイベントを取得
    3. テキストメッセージイベントに対して応答を生成
    4. LINE API を使って返信
    """
    parser, messaging_api = _ensure_line_clients()

    # X-Line-Signature ヘッダから署名を取得
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode("utf-8")

    # 署名を検証してイベントを解析
    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # 各イベントを処理
    for event in events:
        # テキストメッセージ以外（画像、スタンプ等）は無視
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessageContent):
            continue

        user_id = event.source.user_id
        user_text = event.message.text
        logger.info("Received message from %s: %s", user_id, user_text)

        # 会話履歴を取得して応答を生成
        history = get_history(user_id)

        try:
            reply_text = await respond(user_text, history=history)
        except Exception:
            logger.exception("Failed to generate response")
            reply_text = "すみません、応答の生成に失敗しました。もう一度お試しください。"

        # 会話履歴に今回のやりとりを保存
        add_message(user_id, "user", user_text)
        add_message(user_id, "assistant", reply_text)

        # LINE に返信
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)],
            )
        )

    return {"status": "ok"}
