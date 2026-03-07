# Personal AI - Cognitive Bias Detection with Workout Logging

LINE Botを通じた筋トレログの構造化と認知バイアス検知AIです。

## Tech Stack
- **Chat UI**: LINE Bot (Messaging API)
- **LLM**: Gemini API (gemini-2.0-flash)
- **Embedding**: Gemini Embedding API (text-embedding-004)
- **Vector DB**: Qdrant Cloud
- **Backend**: Python, FastAPI

## Setup

```bash
cp .env.example .env
# .envを編集してAPIキーを設定

pip install -e ".[dev]"
uvicorn src.api.main:app --reload
```

## Phases
- **Phase 1**: LINE Bot + ログ構造化 + メタ認知応答
- **Phase 2**: RAGパイプライン
- **Phase 3**: Streamlitダッシュボード
- **Phase 4-5**: 外部データ統合 & CI/CD
