# omi-whisper-transcribe

Omi デバイスの音声を WebSocket 経由で受信し、ローカルの ASR エンジンで文字起こしするサービス。
Omi の `custom_stt` モードを使い、クラウド STT を完全にバイパスします。

対応エンジン:
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** (large-v3, デフォルト) — 高精度・Hotwords デコーダバイアス対応
- **[Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)** — 軽量 (~3.4 GB VRAM)・52言語対応・context boost 対応

## アーキテクチャ

```
Omi デバイス → (WebSocket :8090) → omi-whisper-transcribe → ASR Engine (GPU)
                                         ↑                   ├─ faster-whisper (ENGINE=whisper)
                              管理 API (:8091)                └─ Qwen3-ASR 1.7B (ENGINE=qwen3asr)
                                         ↑
                              Viewer  (:8092)
```

- **単体デプロイ**: GPU を専有し、モデルを常時ロード
- **GPU 共有デプロイ**: [gpu-shared](../gpu-shared/) 経由で qwen-tts と GPU を共有。管理 API でモデルのロード/アンロードを外部制御

## 主な機能

| 機能 | 説明 |
|------|------|
| リアルタイム文字起こし | WebSocket で受信した音声を ASR エンジン (Whisper / Qwen3-ASR) で逐次処理 |
| ASR エンジン切り替え | `ENGINE` 環境変数で Whisper ↔ Qwen3-ASR を切り替え |
| エンジン比較ツール | 同じ録音を両エンジンで処理し、CER・RTF を比較 (`tools/compare_engines.py`) |
| VAD ベースセグメンテーション | Silero VAD で発話区間を検出し、自然な区切りで文字起こし |
| 無音検出 | RMS ベースで無音を判定し、GPU リソースを節約 |
| 音声前処理 | ラウドネス正規化 + ノイズ除去 (スペクトルゲーティング) |
| ハルシネーションフィルタ | 異常出力・ギブリッシュ仮名・句読点のみ等を除去 |
| セグメント分割 | 長いセグメントを文境界で自然に分割 |
| Hotwords | JMnedict 辞書ベースの固有名詞バイアス |
| LLM 後処理 (3段階パイプライン) | 用語候補抽出 → 表記補正 → 品質監査の3段階で文字起こし精度を向上 |
| LLM スクリーニング | 辞書補正 + 信頼度スコアで LLM 不要なセグメントを自動スキップ (API call 60-80% 削減) |
| セッションタグ (3軸分類) | LLM による日本語3軸タグ付け (会話種類・成果物・事業領域 + 要約・判断メモ) |
| RAG 分類 | 過去セッションの埋め込みベクトルによる k-NN 加重投票分類 (LLM コスト 80-90% 削減) |
| タクソノミー自動精緻化 | 月次バッチでタクソノミーの妥当性を検証し改善提案を自動生成 (Gemini Batch API) |
| バッチ話者分離 | pyannote で全録音の話者分離を一括実行 |
| マルチスピーカー声紋登録 | 複数話者の声紋を登録し、セグメントごとに話者タグ付け |
| リアルタイム声紋フィルタ | is_user 声紋との比較で非ユーザー音声 (TV等) を自動除去 |
| 話者タグ補正 | セグメント単位で話者タグを手動修正、自動学習でサンプル追加 |
| 音響シーン分類 (PANNs CNN14) | AudioSet 527クラスモデルで6次元音響分析: シーン・空間・ノイズ源・イベント・音楽・話者数・品質 (CPU 推論) |
| VibeVoice-ASR バッチ文字起こし | microsoft/VibeVoice-ASR-HF (9B, 4-bit NF4 量子化) による比較用バッチ文字起こし (デフォルト無効、API のみ。Viewer UI ボタンは削除済み) |
| 深夜スケジューラ | 指定時刻に話者分離 + タグ付けを自動一括実行 (デフォルト: 毎日 4:00) |
| 空セッションスキップ | セグメントのないセッションは自動削除 |
| トレーニングデータエクスポート | 6 フォーマット (OpenAI / ShareGPT / Comprehensive / Knowledge / Decision Log / SOP) |
| Safe Training Data エクスポート | LLM で匿名化・構造化 + evidence-based knowledge_card 抽出 (2-step pipeline) |
| 事業継承 RAG (Succession RAG) | sanitized KC をベクター検索 → gpt-5-nano で根拠付き回答する Q&A システム (月コスト ~$1) |
| タグレビューキュー | 優先度付きレビューモードで不確実なタグを効率的に承認・修正 (キーボードショートカット対応) |
| 録音 Viewer | 録音一覧・再生・検索・設定管理の Web UI |

## 必要環境

| 項目 | 要件 |
|------|------|
| GPU | NVIDIA (VRAM 6GB 以上、large-v3 は 12GB 推奨) |
| Docker | Docker + NVIDIA Container Toolkit |
| OS | Linux / WSL2 |

## クイックスタート

### 1. .env 設定

```bash
cd omi_whisper_transcribe
cp .env.example .env
# .env を編集して以下を設定:
#   HF_AUTH_TOKEN=hf_xxx    (話者分離・声紋登録に必須。pyannote の利用規約同意が必要)
#   GEMINI_API_KEY=...      (LLM 後処理・セッションタグに必須)
#   OMI_WHISPER_TOKEN=...   (Cloudflare Tunnel 経由で公開する場合)
```

### 2. 起動

#### 単体デプロイ (GPU 専有)

```bash
cd omi_whisper_transcribe
docker compose up -d --build
```

#### GPU 共有デプロイ (qwen-tts と共有)

```bash
cd gpu-shared
docker compose up -d --build
```

> **重要**: gpu-shared 構成では、**すべてのコンテナを `gpu-shared/` ディレクトリから起動**してください。
> `omi_whisper_transcribe/` から `docker compose up` すると Viewer が別ネットワークに配置され、
> Bridge (管理 API) に接続できなくなります (502 エラー)。

初回はモデルダウンロード (約 3GB) に数分かかります。Docker Volume `whisper-models` にキャッシュされるので 2 回目以降は高速です。

### 3. 確認

```bash
# ログ (単体の場合はカレントディレクトリで実行)
docker compose logs -f omi-whisper

# 管理 API でモデル状態を確認
curl http://localhost:8091/model/status

# Viewer
open http://localhost:8092
```

## ポート

| ポート | プロトコル | 用途 |
|--------|-----------|------|
| 8090 | WebSocket | 音声受信 + 文字起こし結果送信 |
| 8091 | HTTP | 管理 API |
| 8092 | HTTP | 録音 Viewer |

## 管理 API (port 8091)

### モデル管理

| メソッド | パス | 説明 |
|----------|------|------|
| POST | `/model/load` | モデルロード (冪等) |
| POST | `/model/unload` | モデルアンロード (セッション中は 409) |
| GET | `/model/status` | モデル状態を返す |
| GET | `/whisper/config` | Whisper 設定パラメータを返す (読み取り専用) |

### バッチ話者分離

| メソッド | パス | 説明 |
|----------|------|------|
| POST | `/diarize/start` | バッチ話者分離を開始 (`{"force": true}` で再処理) |
| GET | `/diarize/status` | 進捗状態を返す |
| POST | `/diarize/cancel` | 実行中のバッチをキャンセル |

### Hotwords

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/hotwords/status` | 有効な hotword 数・カテゴリ情報 |
| GET | `/hotwords/settings` | 設定 JSON を返す |
| PUT | `/hotwords/settings` | 設定を更新 |
| POST | `/hotwords/custom` | カスタム hotword を追加 |
| DELETE | `/hotwords/custom` | カスタム hotword を削除 |

### 話者管理・声紋

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/speakers/status` | 話者数・閾値等の状態 |
| GET | `/speakers/list` | 登録話者一覧 |
| GET | `/speakers/list-detailed` | 登録話者一覧 (サンプルメタデータ付き、embedding 除外) |
| POST | `/speakers/add` | 話者を登録 `{"name", "is_user"}` |
| PUT | `/speakers/{id}` | 話者情報を更新 |
| DELETE | `/speakers/{id}` | 話者を削除 |
| POST | `/speakers/{id}/voiceprint` | 声紋サンプルを追加 |
| DELETE | `/speakers/{id}/voiceprint/{sample_id}` | 声紋サンプルを削除 |
| PUT | `/speakers/threshold` | 類似度閾値を変更 |
| PUT | `/speakers/tag-correction` | セグメント個別の話者タグ修正 (`manually_tagged` で上書き保護) |
| POST | `/speakers/tag/start` | バッチ話者タグ付けを開始 |
| GET | `/speakers/tag/status` | タグ付け進捗を返す |
| POST | `/speakers/tag/cancel` | タグ付けをキャンセル |
| DELETE | `/speakers/all` | 全話者を削除 |

### セッションタグ (3軸分類)

| メソッド | パス | 説明 |
|----------|------|------|
| POST | `/session-tags/start` | バッチセッションタグ付けを開始 |
| GET | `/session-tags/status` | タグ付け進捗を返す |
| POST | `/session-tags/cancel` | タグ付けをキャンセル |
| POST | `/taxonomy/reload` | タクソノミー定義を再読み込み |
| POST | `/keywords/learn` | タグ修正からキーワード候補を抽出 |

### RAG 分類

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/rag/status` | RAG インデックス状態 (セッション数・モデル・ready) |
| POST | `/rag/rebuild` | RAG インデックスを削除して再構築 |

### タクソノミー自動精緻化

| メソッド | パス | 説明 |
|----------|------|------|
| POST | `/taxonomy/refine/run` | 精緻化パイプラインを手動実行 (バックグラウンド) |
| GET | `/taxonomy/refine/status` | 実行状態・pending 有無・過去レポート |
| GET | `/taxonomy/refine/pending` | 承認待ち提案の詳細 |
| POST | `/taxonomy/refine/approve` | 提案を承認してタクソノミーを更新 (`{"accepted_indices": [0,1]}` で個別承認可) |
| POST | `/taxonomy/refine/reject` | 提案を却下 |

### 一括タグ付け (セッション + 話者)

| メソッド | パス | 説明 |
|----------|------|------|
| POST | `/tags/start` | セッションタグ + 話者タグを一括実行 |
| GET | `/tags/status` | 一括タグ付け進捗を返す |
| POST | `/tags/cancel` | 一括タグ付けをキャンセル |

### バッチ LLM 補正

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/batch/status` | LLM バッチ補正ステータスを返す |
| GET | `/batch/jobs` | バッチジョブ一覧 |
| POST | `/batch/retry` | 失敗ジョブの再試行 |
| POST | `/batch/process` | 手動でバッチ補正をトリガー |

### 深夜スケジューラ

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/nightly/status` | スケジューラ状態 (next_run, last_run, running, phase, last_completed_date) |
| POST | `/nightly/trigger` | 手動即時実行 (話者分離 + タグ付け) |

最後に完了した日付は `config/nightly_state.json` に永続化される。コンテナが
深夜バッチ実行中にクラッシュ／OOM 等で再起動した場合、起動時に「当日のウィンドウ
を過ぎているのに今日まだ完了していない」状態を検知して即時 catch-up 実行する。
バッチ diarize 開始時には Whisper モデルを unload して RAM/VRAM を解放する
(OOM 対策)。

### Safe Training Data (匿名化エクスポート)

| メソッド | パス | 説明 |
|----------|------|------|
| POST | `/sanitize/start` | バッチ匿名化を開始 (`{"sessions": [...]}`) |
| GET | `/sanitize/status` | 進捗状態を返す |
| POST | `/sanitize/cancel` | 実行中のバッチをキャンセル |
| GET | `/sanitize/download` | 生成された JSONL ファイルをダウンロード |

### 事業継承 RAG (Succession RAG)

sanitized `knowledge_cards` を **Anthropic Contextual Retrieval** 方式で
検索して gpt-5-nano で根拠付き回答を生成する Retrieval-Augmented Q&A
システム。月次コストは ~$1-2。fine-tuning なしで即運用可能。

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/succession/status` | インデックス情報 (size, version, retrieval_mode 等) |
| POST | `/succession/rebuild` | インデックス再構築 (`{"force": bool}`) |
| POST | `/succession/ask` | 質問に回答 (`{"query": str, "k": int=5}`) |

Viewer からも `/api/succession/*` で同じ機能を使えます (Settings > 事業相談 タブ)。

**動作原理 (Contextual Retrieval + Hybrid RRF)**:

[Anthropic の公式ブログ](https://www.anthropic.com/news/contextual-retrieval)
で発表された Contextual Retrieval 方式を採用。純粋な dense search 比で
**失敗率 49% 削減** (reranker 併用で 67%) が報告されています。

```
1. config/safe_training_*.jsonl をすべて読み込み (SanitizedIndex 経由)
2. 各 knowledge_card について gpt-5-nano で 1 文 (50-80 字) の
   "context" を生成 (Anthropic Contextual Embeddings 方式)
   例: "物販事業における FBA と自己配送の使い分け方針"
3. "context + question" を gemini-embedding-001 で埋め込み (dense)
4. 同じテキストを CJK 向け BM25 tokenizer (ASCII 単語 + 日本語
   unigram/bigram) でトークン化し rank-bm25 に投入 (sparse)
5. インデックスを config/succession_index.{npz,json} に永続化
6. クエリ時: query を同じ tokenizer + embedding で処理
7. dense top-20 + BM25 top-20 を Reciprocal Rank Fusion (RRF, k=60) で融合
8. 上位 K 件を gpt-5-nano に渡して根拠付き回答を生成
9. 応答 + 参照した KC (similarity / bm25_score / rrf_score 付き) を返却
```

Contextual Retrieval が効くケース:
- 固有名詞・型番・金額が絡む質問 (BM25 が強い)
- 言い換え・同義語を含む質問 (dense が強い)
- 両方混在する実用クエリ → **RRF で両方の長所を取る**

**CLI**:

```bash
# インデックス構築 (初回 or sanitize 追加後)
GEMINI_API_KEY=xxx OPENAI_API_KEY=xxx \
  python tools/build_succession_index.py --force

# 質問 (HTTP)
curl -X POST http://localhost:8092/api/succession/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "FBA と自己配送の使い分けはどう判断してた?", "k": 5}'
```

**環境変数**:

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `SUCCESSION_ANSWER_MODEL` | `gpt-5-nano` | 回答生成モデル (Responses API) |
| `SUCCESSION_EMBED_MODEL` | `gemini-embedding-001` | 埋め込みモデル |
| `SUCCESSION_CONTEXT_MODEL` | `gpt-5-nano` | KC への 1 文コンテキスト生成に使うモデル |

**プロンプト方針**: 「提供された KC の内容のみ」に基づいて答える。外部知識や一般論で
補完せず、該当なしの場合は率直に「記録がありません」と返す。推測禁止。回答末尾で
どの KC を参照したかを `[#1] [#2]` 形式で引用。

**Build コスト**: 455 KC で contextualize に ~$0.02、embed に ~$0.01、合計 ~$0.03/回。
差分更新時は既存の context を reuse するので新規 KC 分のみ課金される。

### 音響シーン分類

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/audio-scene/status` | 音響シーン分類の有効/無効、モデル情報、閾値、バッチ進捗 |

### VibeVoice-ASR バッチ (API のみ、Viewer UI ボタンは削除済み)

| メソッド | パス | 説明 |
|----------|------|------|
| POST | `/vibevoice/start` | VibeVoice-ASR バッチ文字起こし開始 (Phase 1: 文字起こし → Phase 2: 声紋タグ付け) |
| GET | `/vibevoice/status` | 実行状態・進捗を取得 |
| POST | `/vibevoice/cancel` | 実行中のバッチをキャンセル |

### タグレビューキュー (Viewer API, port 8092)

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/api/review/queue?limit=1&offset=0` | 優先度順にレビュー対象セッションを返却 (transcript 付き) |
| GET | `/api/review/stats` | キュー統計 (残数、済数、今日の処理数) |
| POST | `/api/review/accept` | セッションを承認 (`{"date","time"}`) → `manually_corrected` + `review_outcome` を書き込み + フィードバック記録 |
| POST | `/api/review/skip` | スキップ (永続化なし、メモリのみ。リロードで再表示) |

#### 優先度スコアリング (5軸 + 交互作用ボーナス × 重要度乗数)

`priority_score = base_score × importance_multiplier`

**ベーススコア (5軸 + 交互作用)**

| 軸 | 重み | シグナル | 計算方法 |
|----|------|---------|---------|
| A. モデル不確実性 | 0.40 | rag_similarity, quality | RAG: `(1-sim)*0.40` (連続値)、LLM noisy: 0.34、partial: 0.23、clean: 0.17 |
| B. 新規性 | 0.20 | conversation_type | 「その他」0.20、NULL 0.18、「独り言・メモ」「雑談・移動中メモ」0.05、他 0.0 |
| C. コンテンツ複雑度 | 0.15 | segment_count | `MIN(segments, 50) / 50 * 0.15` (連続値) |
| D. 音声シーン曖昧性 | 0.12 | external_audio, scene_confidence | external>=0.5 → +0.072、scene_confidence<0.7 → `(1-conf)*0.048` |
| E. 時間的鮮度 | 0.13 | start_at | `MAX(0, 1 - days/90) * 0.13` (90日で線形減衰) |
| F. 交互作用ボーナス | +0.10 | 複合 | RAG低信頼+noisy → +0.10、RAG低信頼+外部音声高 → +0.08 |

**重要度乗数**

| importance_v2 | 乗数 |
|--------------|------|
| 高 | 1.5 |
| 中 | 1.0 |
| 低 / NULL | 0.8 |

重みはフィードバックループ (`config/review_weights.json`) により EMA で自動調整されます。Accept/Edit 操作のたびにシグナルバケット別の修正率が更新され、修正頻度の高い軸の重みが増加します。

## 環境変数

### ASR エンジン選択

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `ENGINE` | `whisper` | ASR エンジン (`whisper` または `qwen3asr`) |

`ENGINE=qwen3asr` を指定すると、Qwen3-ASR 1.7B (bfloat16, ~3.4 GB VRAM) がリアルタイム文字起こしに使用されます。
Whisper 固有の環境変数 (BEAM_SIZE, VAD_FILTER 等) は Qwen3-ASR モード時には無視されます。
Hotwords は Qwen3-ASR の context boost 機能を通じて自動的に適用されます。

### Whisper モデル

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `WHISPER_MODEL` | `large-v3` | モデル名 (tiny, base, small, medium, large-v3) |
| `COMPUTE_TYPE` | `int8_float16` | 量子化 (float16, int8_float16, int8) |
| `LANGUAGE` | `ja` | 言語コード (ja, en, auto) |
| `BEAM_SIZE` | `5` | ビームサイズ |
| `VAD_FILTER` | `true` | VAD フィルタ (ハルシネーション対策) |
| `BUFFER_SECONDS` | `5.0` | 音声バッファ秒数 |
| `INITIAL_PROMPT` | (空) | 初期プロンプト |
| `NO_REPEAT_NGRAM_SIZE` | `5` | N-gram 繰り返し防止 (0=無効) |
| `PATIENCE` | `1.5` | ビームサーチ patience (探索深度、0=無効) |
| `LOG_PROB_THRESHOLD` | `-0.5` | 対数確率の閾値 |
| `NO_SPEECH_THRESHOLD` | `0.6` | 非発話判定の閾値 |
| `COMPRESSION_RATIO_THRESHOLD` | `2.4` | 圧縮率の閾値 |
| `CONDITION_ON_PREVIOUS_TEXT` | `true` | 前セグメントの文脈を使って認識精度を向上 |
| `REPETITION_PENALTY` | `1.1` | 繰り返しペナルティ (1.0=無効) |
| `HALLUCINATION_SILENCE_THRESHOLD` | `1.0` | 無音区間ハルシネーション検出閾値 (秒、空=無効) |
| `PROMPT_CARRY_ENABLED` | `true` | チャンク間コンテキスト引き継ぎ (前チャンク出力を次の initial_prompt に) |
| `WHISPER_VAD_MIN_SPEECH_MS` | `250`/`500` | Whisper 内部 VAD 最小発話長 (ms)。VAD_SEG 有効時は 250、無効時は 500 |
| `WHISPER_VAD_MIN_SILENCE_MS` | `500` | Whisper 内部 VAD 最小無音長 (ms) |
| `WHISPER_VAD_SPEECH_PAD_MS` | `200` | Whisper 内部 VAD 発話パディング (ms) |
| `WHISPER_VAD_THRESHOLD` | `0.5` | Whisper 内部 VAD 閾値 |

### VAD セグメンテーション

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `VAD_SEG_ENABLED` | `true` | VAD ベースセグメンテーションの有効化 |
| `VAD_CHECK_INTERVAL` | `1.0` | VAD チェック間隔 (秒) |
| `VAD_SILENCE_TRIGGER` | `1.2` | 文字起こしトリガーとなる無音時間 (秒) |
| `VAD_MAX_BUFFER` | `20.0` | 強制文字起こしまでの最大バッファ (秒) |
| `VAD_THRESHOLD` | `0.5` | Silero VAD の信頼度閾値 |
| `VAD_MIN_SILENCE_MS` | `500` | 発話間の最小無音時間 (ms) |
| `VAD_SPEECH_PAD_MS` | `300` | 発話前後のパディング (ms) |

### 無音検出

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `SILENCE_THRESHOLD` | `200` | 無音検出 RMS 閾値 (0=無効) |
| `SILENCE_TIMEOUT` | `30.0` | 無音タイムアウト秒数 |

### 録音

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `SAVE_AUDIO` | `true` | 録音保存 |
| `RECORDINGS_DIR` | `./recordings` | 保存先 |
| `MAX_RECORDING_DAYS` | `30` | 自動削除日数 (0=無効) |

### 音声前処理

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `NORMALIZE_ENABLED` | `true` | ラウドネス正規化 |
| `NORMALIZE_TARGET_DBFS` | `-16.0` | ターゲットラウドネス (dBFS) |
| `NOISE_REDUCE_ENABLED` | `false` | スペクトルゲーティング方式のノイズ除去 |
| `NOISE_REDUCE_PROP_DECREASE` | `0.8` | ノイズ低減率 (0.0〜1.0) |

### LLM 後処理 (3段階パイプライン)

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `LLM_ENABLED` | `true` | LLM 後処理の有効化 |
| `LLM_MODEL` | `gemini-3.1-flash-lite-preview` | LLM モデル名 |
| `LLM_TIMEOUT` | `10` | リクエストタイムアウト (秒) |
| `LLM_MIN_TEXT_LENGTH` | `20` | 処理対象の最小テキスト長 |
| `LLM_BATCH_INTERVAL` | `60` | バッチフラッシュ間隔 (秒) |
| `LLM_BATCH_SIZE` | `20` | バッチフラッシュのセグメント数 |
| `LLM_PIPELINE_STAGES` | `B` | パイプラインステージ (A=候補抽出, B=補正, C=監査) |
| `LLM_FLUSH_MODE` | `smart` | フラッシュモード (`smart`=トピック境界検出, `fixed`=固定) |
| `LLM_SCREENING_ENABLED` | `true` | スクリーニング (辞書補正 + 信頼度分類で LLM スキップ) |
| `LLM_BATCH_MODE` | `sync` | バッチ補正モード (`sync`=同期処理, `batch`=Gemini Batch API 50%割引) |
| `RECOVERY_SCAN_DAYS` | `7` | 起動時リカバリ: 直近N日の未処理セッションを自動エンキュー (0=無効) |
| `LLM_BASE_URL` | Gemini OpenAI 互換 | LLM API のベース URL |
| `CONFIDENCE_LOGPROB_THRESHOLD` | `-0.5` | 信頼度スクリーニング閾値 (低信頼度→dirty) |

### セッションタグ (3軸分類)

セッション終了時に LLM が自動で日本語3軸タグを付与します:

- **会話の種類** (conversation_type): 商談、社内ミーティング、1on1 等 (単一選択)
- **会話の成果物** (outcomes): 意思決定あり、TODO発生 等 (複数選択)
- **事業領域** (domains): 水中超音波、物販、経営 等 (複数選択、任意)

タクソノミー (分類軸の項目) は `config/tag_taxonomy.json` で定義し、Viewer の Settings > Taxonomy タブから編集可能です。`domain_descriptions` でドメインごとの説明文を定義すると、LLM プロンプトに説明付きで提示され分類精度が向上します。さらに `keyword_anchors` で各ドメインに関連キーワードのリストを定義すると、LLM がキーワードレベルでドメインの境界を判断でき、分類精度がさらに向上します。`keyword_anchors` は Viewer の Taxonomy タブ内のキーワードエディタからも追加・削除できます。

**ドメイン事前フィルタ**: LLM コール前にセッションテキストを `keyword_anchors` で走査し、ヒットしたキーワード数に基づいて候補ドメインを特定します。候補一覧をユーザーメッセージに注入することで LLM のラベル空間を縮小し、分類精度を向上させます。候補外のドメインも正当な理由があれば LLM が選択可能です。

**Correction-driven Learning**: Viewer でセッションのドメインタグを手動修正すると、そのセッションの文字起こしテキストと entities から新しいキーワード候補を自動抽出し `config/pending_keyword_suggestions.json` に蓄積します。同じキーワードが複数セッションで検出されると occurrences がインクリメントされます。候補は Settings > Taxonomy の「キーワード候補」セクションで一覧表示され、承認すると `keyword_anchors` に即座に追加、拒否すると非表示になります。

Voice タグ (話者識別) が利用可能な場合、各セグメントの話者情報 (自分/登録者/unknown) を LLM プロンプトに含めて分類精度を向上させます。

**外部音声比率の動的判定**: `external_audio_ratio` は声紋 (voiceprint) と PANNs シーン分類 (`audio_scene`) を動的にブレンドして算出します。

1. セッション内の `speaker_similarity` の p75 から `voice_quality` (0〜1) を計算：
   - p75 ≥ 0.50 → 1.0 (声紋が確実に user を識別)
   - p75 ≤ 0.25 → 0.0 (ノイズレベル、声紋が機能していない — 例: 古いモデルのエンロール残り)
   - 中間は線形補間
2. PANNs の `scene_label_en` を参照 (`media` → 1.0, `conversation` 等 → 0.0)
3. 両シグナルを `voice_quality` で重み付け: `ratio = w_voice * unknown_ratio + (1 - w_voice) * pann_media`

これにより、声紋モデルの入れ替え等で声紋比較が壊れていても、PANNs が「会話」と判定したセッションに「TV/外部音声」バッジが誤って付くことはなくなります。`external_audio_ratio >= 0.8` のセッションは従来通り「TV/外部音声」バッジを表示し、ドメイン分類を空配列に強制します。

**声紋モデル変更後の再エンロール**: 声紋の埋め込みモデルを変更した場合、`tools/auto_reenroll_user.py` で自動再エンロールできます。PANNs=会話 かつ single speaker かつ music 未検出のセッションを候補として自動抽出し、既存サンプルをクリアして新モデルで再生成します。

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `TAGGING_ENABLED` | `true` | セッション終了時の自動タグ付け |
| `TAGGING_MODEL` | `gemini-3.1-flash-lite-preview` | タグ付け用 LLM モデル名 |
| `DOMAIN_PREFILTER_ENABLED` | `true` | ドメイン事前フィルタ (keyword_anchors によるキーワード走査) |
| `DOMAIN_PREFILTER_MIN_MATCHES` | `1` | 候補ドメインとなる最小キーワードヒット数 |
| `RAG_ENABLED` | `true` | RAG 分類 (過去セッション埋め込みベースの k-NN 加重投票) |
| `RAG_K` | `7` | k-NN 検索時の近傍数 |
| `RAG_CONFIDENCE_THRESHOLD` | `0.5` | RAG 確信度閾値 (top 投票比率がこれ以上で LLM スキップ) |
| `RAG_MIN_SESSIONS` | `20` | RAG 有効化に必要な最小インデックスセッション数 |

### タクソノミー自動精緻化

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `TAXONOMY_REFINE_ENABLED` | `true` | タクソノミー自動精緻化機能の有効/無効 |
| `TAXONOMY_SAMPLE_RATIO` | `0.2` | サンプリング率 (各 conversation_type から何割抽出するか) |
| `TAXONOMY_PROPOSAL_MODEL` | `gemini-2.5-flash` | Step 3 (提案生成) に使う LLM モデル |
| `TAXONOMY_VALIDATION_MODEL` | `gemini-3.1-flash-lite-preview` | Step 4 (検証) に使う LLM モデル |

### 深夜スケジューラ

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `NIGHTLY_SCHEDULE_HOUR` | `4` | 深夜バッチ実行時刻 (0=無効, 1-23=実行時刻) |

### リアルタイム声紋フィルタ

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `VOICE_FILTER_ENABLED` | `false` | リアルタイム声紋フィルタ (非ユーザー音声を除去) |
| `VOICE_FILTER_THRESHOLD` | `0` | 類似度閾値 (0=speakers.json の値を使用) |

### 機能フラグ

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `DIARIZATION_ENABLED` | `false` | セッション終了時の自動話者分離 |
| `HOTWORDS_ENABLED` | `true` | Hotwords (デコーダバイアス) |

### 音響シーン分類 (6次元分析)

1回の PANNs CNN14 推論 (527クラス) から以下6次元を抽出:

| 次元 | 出力キー | 説明 |
|------|---------|------|
| シーン | `scene_label` | 電話 / TV・ラジオ / 会話 / 静寂 / 騒音環境 (優先度ベース判定) |
| 空間 | `spatial` | 室内(小/大) / 公共空間 / 屋外(都市/自然) (max-score) |
| ノイズ源 | `noise_sources` | 交通 / 空調 / ファン / タイピング / 風 / 雨 / 工事 / 衣擦れ / サイレン (多ラベル) |
| イベント | `events` | 笑い / 泣き声 / 咳 / くしゃみ / 拍手・歓声 / 叫び / ため息 (多ラベル) |
| 音楽 | `music` | 検出有無 + タイプ: 生演奏 / 録音・BGM / 歌唱 (Noisy OR + max比較) |
| 話者数 | `speaker_count` | 独り言 / 会話 / 群衆 (max-score) |
| 品質 | `quality_issues` | 歪み / ノイズ / ハム / エコー / 残響 / 風切り音 + クリッピング検出 (多ラベル) |

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `AUDIO_SCENE_ENABLED` | `true` | PANNs CNN14 音響シーン分類の有効/無効 |
| `AUDIO_SCENE_DEVICE` | `cpu` | 推論デバイス (cpu / cuda) |
| `AUDIO_SCENE_THRESHOLD` | `0.3` | シーン/空間分類の信頼度閾値 |
| `AUDIO_SCENE_MEDIA_THRESHOLD` | `0.6` | Noisy OR によるメディア判定閾値 (TV/アニメ検出感度) |

### VibeVoice-ASR

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `VIBEVOICE_ENABLED` | `false` | VibeVoice-ASR バッチ文字起こしの有効/無効 (実験的機能。有効にするとモデルのダウンロードが行われます) |

**実装詳細**: 4-bit NF4 量子化 (~6.6GB VRAM), Flash Attention 2 (LLM デコーダのみ dict 形式適用), TF32 テンソルコア有効化, `max_new_tokens` 音声長比例制限 (100 tokens/秒, 上限 16384), `repetition_penalty=1.05` (1.2→軽減、JSON キーペナルティ抑制), `acoustic_tokenizer_chunk_size` による内部チャンキング制御, 音声正規化 (-16 dBFS), 入力トークン除外による安定した JSON パース, 日本語コンテキストプロンプト + Hotwords 連携, `processor.decode(return_format="parsed")` 最適化, **非言語タグ除去** ([Unintelligible Speech], [Silence] 等の公式6タグ + [Hard Speech] をフィルタ), **15秒セグメント分割** (文境界で分割し Whisper 互換の細粒度出力に変換), **Prefetch パイプライン** (次セッションの ffmpeg 変換 + 音声ロードを GPU 推論と並行実行), **セッション長ソート** (短い音声から順に処理して OOM リスク低減)。

### 外部サービス

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `HF_AUTH_TOKEN` | (空) | HuggingFace トークン (話者分離・声紋登録に必須) |
| `GEMINI_API_KEY` | (空) | Google Gemini API キー (LLM 後処理・セッションタグ・Chat Completions 匿名化に必須) |
| `OPENAI_API_KEY` | (空) | OpenAI API キー (Responses API 匿名化に使用、設定時は Gemini より優先) |
| `SANITIZE_MODEL` | `gemini-2.5-flash` | Chat Completions 匿名化で使うモデル名 |

### ネットワーク・認証

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `HOST` | `0.0.0.0` | バインドアドレス |
| `PORT` | `8090` | WebSocket ポート |
| `MANAGEMENT_PORT` | `8091` | 管理 API ポート |
| `OMI_WHISPER_TOKEN` | (空) | 認証トークン (空=認証なし) |

### モデル選択の目安

| モデル | VRAM (int8) | 速度 | 日本語精度 |
|--------|-------------|------|-----------|
| large-v3 | ~5GB | RTF 0.21x | 最高 |
| medium | ~3GB | 中 | 良好 |
| small | ~1.5GB | 速い | そこそこ |

RTX 3060 12GB なら `large-v3` + `int8_float16` がベストバランス。

## 無音検出

RMS ベースの無音判定で GPU リソースを節約:

- 無音バッファは whisper 処理と WAV 書き込みをスキップ
- 30秒連続で無音が続くと録音セッションを自動終了 (WAV 確定保存)
- 発話が再開すると新しいセッションを自動開始
- WebSocket 接続自体は切断しない (Omi 側のライフサイクルに任せる)

`SILENCE_THRESHOLD=0` で無効化可能。閾値 200 は環境ノイズ (50〜300) をカットし、小声 (500〜) は通す設定。

## 音声前処理

音声が Whisper に渡される前に 2 段階の前処理を実行:

1. **ノイズ除去** (`NOISE_REDUCE_ENABLED=true` で有効): noisereduce ライブラリのスペクトルゲーティングで定常騒音 (空調、エンジン音、電車走行音) を除去。CPU (numpy) ベースのため VRAM 消費なし
2. **ラウドネス正規化** (`NORMALIZE_ENABLED=true`): ターゲット dBFS に音量を正規化。音量のばらつきを抑えて認識精度を安定化

## ハルシネーションフィルタ

Whisper 特有の異常出力を後処理で除去:

- **繰り返しパターン**: 同じフレーズや文字の連続出力
- **ギブリッシュ仮名**: 低信頼度 + 意味のないカタカナ/ひらがな羅列 (助詞・文法パターンがあれば保持)
- **句読点・記号のみ**: 実質内容のないセグメント
- **末尾カンマ断片**: 漢字仮名の短い断片がカンマ区切りで連続するパターン
- **no_speech 判定**: `no_speech_prob` が高く、かつ Whisper 信頼度が低いセグメント
- **YouTube 定型句**: 「ご視聴いただき…」「チャンネル登録」「ご清聴ありがとう」等
- **店名・ブランド名暴走**: hotwords 登録語がノイズから連続生成されるパターン (2 個以上の既知固有名詞がカンマ区切りで連続)
- **孤立定型句**: セグメント全体が「ごちそうさまでした」「いらっしゃいませ」「これが今日の映像です」等の場合のみ除去

## セグメント分割

15秒を超える長いセグメントを文境界 (`。` `？` `！`) で自動分割。タイムスタンプは文字数比で按分し、5 文字未満の断片は前のセグメントにマージ。`SEGMENT_MAX_DURATION=0` で無効化。

## バッチ話者分離

[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) (CC-BY-4.0) を使用。
Viewer の「Diarize」ボタンから、未処理の全録音に対して一括で話者分離を実行できます。

### 動作仕様

- `HF_AUTH_TOKEN` があれば Diarizer が常に利用可能 (バッチ用)
- `DIARIZATION_ENABLED=true` の場合のみ、セッション終了時に自動で話者分離も実行
- GPU ロックで Whisper セッションと排他制御 (処理中は Whisper をアンロード)
- キャンセルは現在処理中のセッション完了後に安全に停止
- エラー時も元のセグメントは保持される (テキスト消失なし)

## マルチスピーカー声紋登録

[pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) (CC-BY-4.0, 256次元) で話者の声紋を登録し、各セグメントに話者タグを付与します。

### 仕組み

1. **話者登録**: 名前と `is_user` フラグ (自分/他者) を設定
2. **声紋追加**: Viewer のセッション詳細でセグメントを選択 → 声紋を抽出・登録
   - 複数セッションから追記式で追加可能 (追加するほど精度向上)
   - セッション詳細から新規話者の作成も可能
3. **バッチタグ付け**: 全録音に対して一括で話者タグ付けを実行
   - コサイン類似度で最も近い登録話者を判定
   - 閾値未満のセグメントは `unknown` に分類

### 設定

- 類似度閾値: デフォルト 0.55 (UI スライダーで 0.3〜0.9 調整可)
- VRAM 管理: diarizer と同パターン (ロード→処理→アンロード)
- `config/speakers.json` に話者情報・embedding・サンプルを保存

## リアルタイム声紋フィルタ

Whisper 出力後に `is_user` フラグの付いた登録声紋とコサイン類似度を比較し、非ユーザーのセグメント (TV 音声等) を自動除去します。

- Whisper の入力音声は一切変更しない (Post-transcription フィルタ)
- embedding モデル (~50MB VRAM) は Whisper (~5GB) とセッション中に共存
- 0.5 秒未満のセグメントは判定スキップ (keep)
- `VOICE_FILTER_ENABLED=true` で有効化、`VOICE_FILTER_THRESHOLD=0` で speakers.json の閾値を使用

## Hotwords

[JMnedict](https://www.edrdg.org/enamdict/enamdict_doc.html) (CC-BY-SA 3.0) から抽出した日本語固有名詞で faster-whisper のデコーダロジットをバイアスし、固有名詞の認識精度を向上させます。

- `tools/build_hotwords.py` で `config/hotwords.json` を生成
- カテゴリ別 ON/OFF (地名、企業名、駅名、姓、名、製品名)
- カスタム語の追加も可能 (カテゴリ語より優先的にトークン予算に含まれる)
- トークン予算制限 (`max_tokens=200`): Whisper の内部上限 (448 トークン) を超えないよう自動制御
- `HOTWORDS_ENABLED=false` で無効化

## LLM 後処理 (3段階パイプライン)

Gemini API を使い、文字起こし結果の品質を3段階のパイプラインで向上させます。

```
Layer 2 (辞書置換) 出力
    │
    ▼
3-A: 用語候補抽出 ─── テキスト変更なし、候補 JSON を返す
    │
    ▼
3-B: 表記補正 ──── transcript + glossary + 3-A候補 → 差分 corrections 配列
    │
    ▼
3-C: 品質監査 ──── 修正済みテキストの問題点を検出 → 監査 JSON
    │  └─ high severity 項目あり → 3-B を再実行 (最大1回)
    ▼
インプレース更新 + ログ出力
```

### Stage 3-A: 用語候補抽出

固有名詞・技術用語・壊れカタカナ・誤認識語を過剰気味に抽出。本文は書き換えない。

- カテゴリ: `proper_noun`, `technical_term`, `broken_katakana`, `misrecognized`
- 参考用語リスト (辞書 + hotwords) との照合を優先
- 低 confidence でも候補として含める (3-B で最終判断)

### Stage 3-B: 表記補正

transcript + 参考用語リスト + 3-A 候補を統合し、差分のみの corrections 配列を返す。

- snippet ベースの置換 (全文再生成しない)
- ポストフィルタ: 空 snippet 除去、no-change 除去、句読点のみ変更除去、snippet 存在チェック
- ソース: `glossary`, `candidate_3a`, `context`, `dictionary`

### Stage 3-C: 品質監査

修正済みテキストに残る品質問題を検出。

- 検出対象: `uncorrected_term`, `meaningless`, `broken_katakana`, `trailing_fragment`
- `severity: high` が 1 件以上 → 3-B を再実行 (最大1回)
- 再実行時は 3-C の指摘を追加ヒントとして 3-B に渡す

### ステージ選択

`LLM_PIPELINE_STAGES` 環境変数で有効にするステージを選択:

| 値 | 動作 |
|---|---|
| `B` (デフォルト) | 3-B のみ (スクリーニングと組み合わせてコスト最適) |
| `ABC` | フルパイプライン |
| `AB` | 候補抽出 + 補正 (監査なし) |
| `BC` | 補正 + 監査 (候補抽出なし) |

### Smart Flush

`LLM_FLUSH_MODE=smart` (デフォルト) では、20セグメント到達時にセグメント間のギャップ (2秒以上) を確認し、トピック境界で自動フラッシュ。ギャップがなければ 1.5 倍 (30セグメント) まで待機。タイムアウトでは常に強制フラッシュ。

### その他

- セッション用語集: セグメントテキストに関連する用語のみフィルタ (最大 50 語)
- セッション終了・無音タイムアウト時にも自動フラッシュ
- 登録済み用語辞書 (`config/term_corrections.json`) に基づく自動修正
- 未知の技術用語・固有名詞の修正候補を提案 (`config/pending_suggestions.json`)
- Viewer の Settings > LLM Suggestions で提案を承認/却下

### LLM スクリーニング

LLM を呼ぶ前にルールベースの前処理で補正不要なセグメントを自動スキップし、API コストを大幅に削減します。

```
全セグメント
    │
    ▼
SegmentScreener.screen()
    ├─ Step 1: 辞書補正 (term_corrections.json)
    ├─ Step 2: 自動学習補正 (pending_suggestions.json から高頻度パターン)
    └─ Step 3: 分類 (dirty / clean)
          │
          ├─ dirty → LLM Pipeline (3-A/3-B/3-C)
          └─ clean → LLM スキップ (0 API calls)
```

#### 分類ルール (`segment_classifier.py`)

| ルール | 条件 | 判定 |
|--------|------|------|
| 1 | `llm_corrected=True` (既に補正済み) | clean |
| 2 | テキスト長 < 20文字 (ただし `avg_logprob < -0.7` なら **dirty**) | clean / dirty |
| 3 | 辞書/hotwords にない未知カタカナ語 (3文字以上) | **dirty** |
| 4 | Whisper 信頼度が低い (`avg_logprob < -0.5`) | **dirty** |
| 5 | 純日本語 (カタカナ率 < 20%, 漢字仮名率 > 70%) | clean |
| 6 | デフォルト | clean |

- **約300語の一般外来語** (レストラン、テスト、コンピュータ等) は既知として扱い、dirty にしない
- **Confidence-based screening** (Naderi et al., Interspeech 2024): `avg_logprob` が低いセグメントは LLM に送信。Whisper が不確かだった漢字誤読・ひらがなエラー等をキャッチ
- 旧トランスクリプト (`avg_logprob` なし) は信頼度ルールをスキップし後方互換を維持

#### 効果

| ケース | LLM コスト |
|--------|-----------|
| 日常会話セッション (カタカナ少) | **0 API calls** (全 clean) |
| 技術話題混在 | 1 call (dirty のみ送信) |
| 全セグメント dirty | 1 call (変わらず) |

実測: 50 セッション / 356 セグメントで **clean 79.8%, dirty 20.2%** → 従来比 38% 以上の API call 削減。

### コスト見積もり (Gemini 2.5 Flash-Lite)

| 項目 | 1バッチ (20seg) | 月額 (8h/日) |
|---|---|---|
| B + スクリーニング (標準) | ~$0.0001 | ~$1.44 |
| ABC (フルパイプライン) | ~$0.0015 | ~$21.60 |
| ABC + 再実行 | ~$0.0025 | ~$36.00 |

## セッションタグ

Gemini API でセッション終了時に自動でメタデータタグを付与:

#### 基本フィールド

| フィールド | 型 | 内容 |
|-----------|-----|------|
| `topics` | リスト | 会話の主要トピック |
| `activity` | enum | 会話、ミーティング、移動中、作業中 等 |
| `intent` | enum | 情報共有、相談、雑談、報告 等 |
| `mood` | enum | ポジティブ、ニュートラル、ネガティブ |
| `quality` | enum | 文字起こし品質 (clear, noisy, mixed) |
| `entities` | リスト | 言及された固有名詞 |
| `summary` | テキスト | 日本語の要約 (1〜2文) |

#### 事業継承 AI 用フィールド

| フィールド | 型 | 内容 |
|-----------|-----|------|
| `source_type` | enum | 会話形式 (meeting, monologue, discussion, phone_call, negotiation 等) |
| `business_domain` | enum | 事業領域 (sales, technology, finance, management, product_planning 等) |
| `importance` | enum | 重要度 (high, medium, low) |
| `confidentiality` | enum | 機密レベル (public, internal, confidential) |
| `decision_required` | bool | 意思決定が行われたか |
| `decision_summary` | テキスト | 何が決まったか (50 文字以内) |
| `reasoning_summary` | テキスト | なぜそう決めたか (50 文字以内) |
| `action_items` | リスト | 次のステップ (最大 5 件) |
| `outcome` | enum/null | 結果 (pending, completed, cancelled, deferred) |
| `next_action` | テキスト | 次にやること (50 文字以内) |

- `TAGGING_ENABLED=true` でリアルタイム自動タグ付け
- Viewer の「Tag」ボタンからバッチ実行も可能
- `tools/tag_all.py` で CLI からも実行可能

### RAG 分類 (Phase 4)

過去のタグ付き済みセッションの埋め込みベクトル (Gemini `gemini-embedding-001`) を使い、
新セッションを k-NN 加重投票で分類します。LLM を呼ばずに 8-9 割の分類を完結し、
曖昧なケースのみ LLM にフォールバックします。

- **コスト削減**: Embedding API は LLM 推論の ~1/100。LLM 呼び出しが 1-2 割に減少
- **精度向上**: 過去事例の正解ラベルを活用した類似度ベース分類
- **自動学習**: 新セッションのタグ付け完了時にインデックスへ自動追加
- **起動時自動構築**: インデックスが存在しない場合、起動時にバックグラウンドで自動構築
- RAG のみの場合: `conversation_type`, `outcomes`, `domains`, `importance`, `quality` を返却 (summary/entities/decision_memo は空)
- LLM フォールバック時: 全フィールドを含む完全タグを返却
- Viewer の Settings > RAG タブでインデックス状態確認・再構築が可能

## 録音 Viewer

`omi-recordings-viewer/` に録音閲覧用の Web アプリを同梱 (ポート 8092):

- 録音一覧・検索 (SQLite FTS)
- 音声再生 + セグメントハイライト
- セッションタグバッジ (活動・気分・トピック) + 要約表示
- 話者バッジ表示 (緑=自分、青=登録話者、グレー=unknown)
- セグメント選択による声紋登録 (新規話者のインライン作成対応)
- バッチ処理パネル (話者分離・話者タグ付け・セッションタグ付け)
- トレーニングデータエクスポート (6 フォーマット、フィルタ、ストリーミングダウンロード)
- 設定管理 (プロンプト、用語辞書、LLM 提案、プリセット、Hotwords、Voice、Taxonomy)
- タグレビューキュー (優先度付き、Accept/Edit/Skip、キーボードショートカット A/E/S)
- 30秒ごとにインクリメンタルスキャン (最新日付以降のみ)

## トレーニングデータエクスポート

Viewer の **Export** タブまたは API からセッションデータを JSONL 形式でエクスポートできます。事業継承 AI のファインチューニングデータ生成を想定しています。

**V2 の変更 (KC ソース切替)**: チャット形式 (`openai` / `conversation`) と構造化形式 (`knowledge` / `decision_log` / `sop`) は、raw transcript の話者マッピングではなく **Safe Training Data エクスポート (後述) の `config/safe_training_*.jsonl` から KC (knowledge card) / summary / keypoints を引いて学習レコードを合成** します。自然会話の話者ラベルを `assistant` ロールに流用する旧方式は、事業判断とは無関係の雑談を教師信号に昇格させる問題があったため廃止されました。

このため:
1. まず **Safe Training Data パイプライン** (Viewer の Sanitize タブ or `/api/sanitize/start`) を高価値セッションに対して実行
2. `config/safe_training_*.jsonl` が生成される
3. Export タブで対象フォーマットを選択してダウンロード → 上記 JSONL を読み込んで fine-tuning 用レコードを生成

### フォーマット

| フォーマット | 用途 | 出力内容 | ソース |
|---|---|---|---|
| **OpenAI Fine-tuning** | GPT ファインチューニング | `{"messages": [system, user, assistant]}` — 1 セッションから KC ごとの Q&A タスク + 要約タスクを複数行で出力 | sanitized JSONL |
| **ShareGPT / LLaMA-Factory** | オープンソースモデル (Llama/Qwen/Mistral 等) の LoRA/QLoRA | `{"conversations": [{from:human}, {from:gpt}], "system": "..."}` — LLaMA-Factory 公式仕様通り (id/metadata は出力しない) | sanitized JSONL |
| **Comprehensive** | 全メタデータ付き生データ (分析・デバッグ用) | session + tags (V2 タクソノミー) + turns + sanitized メタ情報 | raw transcript + (あれば) sanitized |
| **Knowledge Base** | RAG ナレッジ蓄積 | title, body, qa_pairs[], category, entities, importance | sanitized > tags.summary fallback |
| **Decision Log** | 意思決定記録 | decision, reasoning, outcomes, knowledge_cards[], domains | tags.decision_memo + outcomes + sanitized KC |
| **SOP** | 手順書 / TODO 抽出 | steps (keypoints から), context, domain | outcomes=TODO発生/方針転換 + sanitized keypoints |

- **OpenAI/ShareGPT**: sanitized KC が存在しない場合は空を返す (そのセッションはスキップ)
- **Knowledge**: summary (sanitized or tags) があれば出力
- **Decision Log**: `decision_memo` 非空 OR `outcomes` に意思決定あり OR sanitized KC の confidence≥medium
- **SOP**: `outcomes` に TODO発生/方針転換 を含むセッションから sanitized keypoints を手順として抽出
- マルチターン / DPO / ツール使用への将来拡張は ShareGPT 形式で対応可能

### フィルタ

| パラメータ | 説明 | デフォルト |
|---|---|---|
| `date_from` / `date_to` | 日付範囲 | 全期間 |
| `quality` | 文字起こし品質 (カンマ区切り) | 全て |
| `conversation_type` | 会話の種類で絞り込み (V2 タクソノミー) | 全て |
| `domain` | 事業領域で絞り込み (V2 タクソノミー) | 全て |
| `importance_v2` | 重要度で絞り込み (高/中/低) | 全て |
| `business_domain` | 旧事業領域 (後方互換) | 全て |
| `speaker_mode` | `conversation` (全員) / `user_only` (自分のみ) | conversation |
| `min_segments` | 最小セグメント数 | 3 |
| `max_per_class` | conversation_type ごとの最大件数 (クラス均等化) | 0 (無制限) |
| `min_class_samples` | この件数未満のクラスを除外 (過学習防止) | 0 (除外なし) |
| `val_ratio` | 検証データの割合 (0.2=20%)。>0 で ZIP (train+val) 出力 | 0 (分割なし) |
| `include_monologue` | 独白セッションを含めるか | フォーマットによる |
| `require_tagged` | session_tagged=1 のみ (KC 抽出に必須) | false |
| `require_summary` | tags.summary 非空のみ | false |
| `exclude_personal_only` | domains=["個人"] のみを除外 (複合は残す) | false |
| `external_audio_max` | `tags_external_audio < N` (NULL 許容) | 制限なし |
| `high_value_or` | 事業継承価値のある OR 条件 (重要度中+ / 有用 conv_type / decision_memo 非空 / outcomes に実質的成果物) | false |

### API

```bash
# プレビュー (件数・サイズ推定)
curl "http://localhost:8092/api/export/stats?format=knowledge&domain=経営"

# ダウンロード (Safe Training Data 実行後)
curl -o openai_ft.jsonl \
  "http://localhost:8092/api/export/training?format=openai&require_tagged=true&require_summary=true&high_value_or=true&external_audio_max=0.8&exclude_personal_only=true&min_segments=5"

# ShareGPT (LLaMA-Factory 向け)
curl -o sharegpt.jsonl \
  "http://localhost:8092/api/export/training?format=conversation&require_tagged=true&require_summary=true&high_value_or=true&external_audio_max=0.8&exclude_personal_only=true&min_segments=5"

# Knowledge Base
curl -o knowledge.jsonl \
  "http://localhost:8092/api/export/training?format=knowledge&require_summary=true"

# 意思決定ログ
curl -o decisions.jsonl \
  "http://localhost:8092/api/export/training?format=decision_log"

# SOP (手順書)
curl -o sop.jsonl \
  "http://localhost:8092/api/export/training?format=sop"
```

**推奨フロー** (1,000件超のセッションがある場合):

```bash
# 1. 高価値セッション (目安 50-80件) を sanitize パイプラインに投入
curl -X POST "http://localhost:8092/api/sanitize/start" \
  -H "Content-Type: application/json" \
  -d '{}' \
  -G \
  --data-urlencode "require_tagged=true" \
  --data-urlencode "require_summary=true" \
  --data-urlencode "high_value_or=true" \
  --data-urlencode "external_audio_max=0.8" \
  --data-urlencode "exclude_personal_only=true" \
  --data-urlencode "min_segments=5" \
  --data-urlencode "quality=clean,partial"

# 2. 進捗確認
curl "http://localhost:8092/api/sanitize/status"

# 3. 完了後、同じフィルタで export
curl -o openai_ft.jsonl "http://localhost:8092/api/export/training?format=openai&require_tagged=true&..."
```

## Safe Training Data エクスポート (LLM 匿名化)

LLM で文字起こしデータを匿名化・構造化し、安全なトレーニングデータとしてエクスポートします。

### 2-step Pipeline

```
Step 1: 匿名化・構造化 (SANITIZE_SCHEMA)
  入力: raw_text (ASR 出力)
  出力: risk, summary, keypoints, safe_conversation, issues_detected

Step 2: Knowledge Card 抽出 (KC_EXTRACTION_SCHEMA, multi-card)
  入力: Step 1 の cleaned output のみ (raw text は使わない)
  出力: extractable, knowledge_cards[] (最大 5 件 / session)
        各 card: question, answer, evidence[], confidence
```

**Multi-KC (V2)**: 1 セッションから最大 5 個の Q&A ペアを異なる視点
(fact / reason / method / comparison / risk / lesson × 技術 / 経営 / 業務 /
顧客) から抽出。Evol-Instruct / Explicit Diversity Conditions の手法に
準拠。後方互換のため従来の単一 `knowledge_card` フィールドも併記。

### プロバイダ

| プロバイダ | API | 特徴 |
|---|---|---|
| **OpenAI Responses API** (推奨) | `SANITIZE_MODEL=gpt-5-nano` + `OPENAI_API_KEY` | GPT-5 nano + Structured Outputs で schema 準拠を保証、2-step KC 抽出対応 |
| **Chat Completions API** (フォールバック) | `SANITIZE_MODEL=gemini-2.5-flash` + `GEMINI_API_KEY` | Gemini 2.5 Flash、json_object 形式。KC 抽出品質は Responses API より低い |

`SANITIZE_MODEL` が `gpt-*` で始まると自動的に Responses API (2-step pipeline) が有効化されます。事業継承 AI 向けの学習データを作るなら **必ず `gpt-5-nano` 推奨**。

### Knowledge Card の抽出ルール

- **「生成」ではなく「根拠付き抽出」**: テキストに書かれていない外部知識を使わない
- `evidence` (根拠) が必須: safe_conversation からの引用を含める
- `confidence` が `medium` 以上のみ出力 (`low` / `none` は破棄)
- `keypoints` が 3 件未満のセッションは抽出をスキップ
- **質問 15 文字以上、回答 20 文字以上**の品質ゲート
- **曖昧な質問を自動拒否**: 「〜について教えてください」「どう思いますか」「要点を再確認」等のパターンを正規表現で検出 → 破棄
- **セッション内 exact-question 重複排除** (filter_knowledge_cards)
- **セッション横断の semantic 重複排除**: 別ツール `tools/dedupe_sanitized.py` で embedding ベースのクラスタリング (cosine ≥ 0.92)

### セッション横断のセマンティック重複排除

Multi-KC で生成された KC の中には、異なるセッション間で「Whisper Large は
どの GPU で動作しますか?」と「RTX3060 の VRAM で Whisper Large は動くか?」
のような実質同義の質問が発生する。NeMo Curator の semantic deduplication
方式に従い、Gemini `gemini-embedding-001` で質問文を埋め込み、cosine ≥ 0.92 で
クラスタリングして各グループから confidence 最高の 1 件のみ残す。

```bash
# 単一ファイルを重複排除
python tools/dedupe_sanitized.py \
  --input config/safe_training_20260413_113103.jsonl \
  --output config/safe_training_deduped.jsonl \
  --threshold 0.92

# 全 safe_training_*.jsonl を統合して重複排除
python tools/dedupe_sanitized.py \
  --input 'config/safe_training_*.jsonl' \
  --output config/safe_training_merged_deduped.jsonl
```

環境変数: `GEMINI_API_KEY` (embedding API 用)。
コスト: 100 KC ≈ $0.013 (約 2 円)。

### リスク判定

| リスク | 処理 |
|--------|------|
| **low** | 個人情報なし → そのまま JSONL に出力 |
| **medium** | 個人情報あり → 匿名化して出力 (人名→Aさん、会社→X社 等) |
| **high** | PII 複合 → ドロップ (出力しない) |

### 出力 JSONL 構造

```json
{
  "session_id": "2026-03-31/09-35-13",
  "risk": "low",
  "summary": {"input": "...", "output": "..."},
  "keypoints": ["...", "..."],
  "safe_conversation": [{"speaker": "user", "text": "..."}],
  "knowledge_card": {
    "question": "...",
    "answer": "...",
    "evidence": ["会話からの引用1", "引用2"],
    "confidence": "high"
  },
  "issues_detected": ["asr_noise"]
}
```

`knowledge_card` は抽出できない場合 `null` になります。

## 録音データ

セッションごとに `recordings/YYYY-MM-DD/HH-MM-SS/` に保存:
- `audio.ogg` -- Opus 圧縮 (or `audio.wav` -- PCM 16kHz 16-bit mono)
- `transcript.json` -- メタデータ + 文字起こしセグメント + タグ

セグメントのないセッション (無音のみ) は自動で削除されます。
データ量目安: 常時 ON 8h/日で WAV 約 27GB/月。

## ツール

`tools/` に運用・保守用スクリプトを同梱:

| スクリプト | 用途 |
|-----------|------|
| `tools/build_hotwords.py` | JMnedict から `config/hotwords.json` を生成 |
| `tools/retranscribe_all.py` | 全セッションの再文字起こし (`--restore` で復元、`--verify` で検証) |
| `tools/tag_all.py` | 全セッションの一括タグ付け (CLI) |
| `tools/verify_screening.py` | LLM スクリーニングの精度検証 (パターンマッチ分析) |
| `tools/verify_confidence_screening.py` | 信頼度スコアベースのスクリーニング効果検証 |
| `tools/sanitize_jsonl.py` | 生成済み JSONL ファイルの再サニタイズ |
| `tools/compare_*.py` | 各種品質比較スクリプト (パイプライン、スクリーニング、サニタイズ等) |
| `tools/compare_vibevoice.py` | VibeVoice-ASR (9B, NF4) vs Whisper 精度比較 (`--sessions N`, `--ids ID...`) |
| `tools/compare_engines.py` | Whisper vs Qwen3-ASR 精度比較 (`--sessions N`, `--ids ID...`, `--hotwords`) |
| `tools/deep_verify_screening.py` | スクリーニング精度の深掘り検証 |

## GPU 共有デプロイ

qwen-tts と GPU を共有する場合は [gpu-shared](../gpu-shared/) を使用してください。
omi-whisper 自体は GPU 共有の知識を持たず、管理 API を通じて外部から制御されます。

### 起動・停止

gpu-shared 構成では **必ず `gpu-shared/` ディレクトリから** Docker Compose コマンドを実行します。

```bash
cd gpu-shared

# 全コンテナ起動
docker compose up -d --build

# 個別サービス再起動
docker compose up -d --build omi-whisper          # Bridge
docker compose up -d --build omi-recordings-viewer # Viewer
docker compose up -d --build qwen-tts             # TTS

# ログ確認
docker compose logs -f omi-whisper
docker compose logs -f omi-recordings-viewer

# 状態確認
docker compose ps
```

### ネットワーク構成

gpu-shared は `gpu-net` ブリッジネットワークで全サービスを接続します。
`omi_whisper_transcribe/` から個別に `docker compose up` すると別ネットワーク (`omi_whisper_transcribe_default`) に配置され、
Bridge への接続が失敗します。

```
gpu-net (bridge)
├── omi-whisper-transcribe   (8090/8091 internal)
├── omi-recordings-viewer    (8092 → external)
├── qwen-tts                 (8080 internal)
└── gpu-shared               (8000/8090 → external, proxy)
```

## Cloudflare Tunnel

外出先からアクセスする場合:

1. Tunnel を作成し `whisper.example.com` → `http://localhost:8090` にルーティング
2. Cloudflare Access で `/transcribe*` パスを Bypass に設定
3. `OMI_WHISPER_TOKEN` でアプリ側のトークン認証を有効化
4. Omi アプリの Custom STT に `wss://whisper.example.com/transcribe?token=<TOKEN>` を設定

## プロジェクト構成

```
omi_whisper_transcribe/
├── omi_whisper_bridge.py     # メイン: WebSocket + 文字起こし + 管理 API
├── gpu_manager.py            # GPU リソース管理 (レガシー、未使用)
├── model_manager.py          # Whisper モデルのロード/アンロード管理
├── audio_preprocessor.py     # ノイズ除去 + ラウドネス正規化
├── llm_pipeline.py           # 3段階 LLM パイプライン (候補抽出→補正→監査)
├── llm_pipeline_schema.py    # パイプライン用 JSON Schema 定義
├── llm_pipeline_prompt.py    # パイプライン用システムプロンプト
├── llm_corrector.py          # Gemini によるバッチ用語修正 (レガシー / retranscribe 用)
├── batch_corrector.py        # バッチ LLM 補正ワーカー (sync/async モード)
├── segment_screener.py       # LLM スクリーニング (辞書→自動学習→分類)
├── segment_classifier.py     # dirty/clean 分類ルール (カタカナ + 信頼度)
├── suggestion_learner.py     # LLM 提案からの自動学習パターン抽出
├── correction_metrics.py     # スクリーニング統計メトリクス
├── session_tagger.py         # Gemini によるセッションタグ付け
├── rag_classifier.py         # RAG 分類 (Embedding + k-NN 加重投票)
├── taxonomy_refiner.py       # タクソノミー自動精緻化 (月次バッチ, Gemini Batch API)
├── keyword_learner.py        # タグ修正ドリブンのキーワード候補学習
├── diarizer.py               # pyannote バッチ話者分離
├── speaker_filter.py         # 声紋登録・話者タグ付け
├── responses_sanitizer.py    # OpenAI Responses API 匿名化 (2-step pipeline)
├── training_sanitizer.py     # Chat Completions API 匿名化 (Gemini フォールバック)
├── sanitize_schema.py        # 匿名化 JSON Schema + KC 抽出スキーマ + 品質ゲート
├── sanitize_prompt.py        # 匿名化 + KC 抽出のプロンプト定義
├── sanitize_provider.py      # 匿名化プロバイダ抽象インターフェース
├── audio_scene_classifier.py  # PANNs CNN14 6次元音響分析
├── vibevoice_transcriber.py   # VibeVoice-ASR バッチ文字起こし (9B, 4-bit NF4, Prefetch)
├── transcriber_qwen3asr.py   # Qwen3-ASR 1.7B リアルタイム文字起こし (bfloat16, ~3.4 GB VRAM)
├── hotword_loader.py         # Hotwords ロード・管理
├── term_corrector.py         # 用語辞書管理
├── tts_proxy.py              # TTS プロキシ (レガシー、未使用)
├── config/                   # 設定ファイル (JSON)
│   ├── tag_taxonomy.json     # セッションタグ3軸分類 (会話種類・成果物・事業領域)
│   ├── rag_embeddings.npz   # RAG 埋め込みベクトルインデックス (自動生成)
│   ├── rag_metadata.json    # RAG セッションメタデータ (自動生成)
│   ├── pending_taxonomy.json # 承認待ちタクソノミー提案 (自動生成)
│   ├── taxonomy_reports/    # 月次精緻化レポート (自動生成)
├── tools/                    # 運用・検証スクリプト
├── tests/                    # ユニットテスト (pytest)
├── omi-recordings-viewer/    # Viewer Web アプリ
│   ├── app.py                # aiohttp サーバー + API
│   ├── indexer.py            # SQLite インデックス管理
│   ├── exporter.py           # トレーニングデータエクスポート (6 フォーマット)
│   ├── templates/            # Jinja2 テンプレート
│   └── static/               # JS + CSS
├── Dockerfile                # Bridge コンテナ
├── docker-compose.yml        # 単体デプロイ構成
└── .env.example              # 環境変数テンプレート
```

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| GPU が認識されない | NVIDIA Container Toolkit のインストールと `nvidia-ctk runtime configure` を確認 |
| VRAM 不足 | `COMPUTE_TYPE=int8` または `WHISPER_MODEL=medium` に変更 |
| ハルシネーション | `VAD_FILTER=true` を確認、`BUFFER_SECONDS` を短くする |
| Unauthorized | `?token=` の値と `OMI_WHISPER_TOKEN` が一致しているか確認 |
| 話者分離が 403 | `HF_AUTH_TOKEN` の権限を確認 (Access public gated repos が必要) |
| Viewer が Loading で止まる | ブラウザのハードリロード (Cmd+Shift+R)、リトライ機能で自動復帰 |
| バッチ処理中にシステム停止 | VRAM 不足の可能性。ログで OOM を確認。GPU ロックが自動でクリーンアップ |

## 使用モデル・ライセンス

| モデル / ライブラリ | 用途 | ライセンス |
|---------------------|------|-----------|
| [faster-whisper large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | 文字起こし | MIT |
| [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | 話者分離 | CC-BY-4.0 |
| [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | 声紋抽出 | CC-BY-4.0 |
| [JMnedict](https://www.edrdg.org/enamdict/enamdict_doc.html) | 固有名詞辞書 | CC-BY-SA 3.0 |
| [Silero VAD](https://github.com/snakers4/silero-vad) | 音声区間検出 | MIT |
| [noisereduce](https://github.com/timsainb/noisereduce) | ノイズ除去 | MIT |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | リアルタイム文字起こし (1.7B, bfloat16) | Apache 2.0 |
| [microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF) | バッチ文字起こし (9B) | MIT |
| [Gemini API](https://ai.google.dev/) | LLM 後処理・セッションタグ・匿名化 | Google ToS |
| [OpenAI Responses API](https://platform.openai.com/) | 匿名化・KC 抽出 (GPT-5 nano) | OpenAI ToS |
