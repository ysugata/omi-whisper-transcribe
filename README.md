# omi-whisper-transcribe

Omi デバイスの音声を WebSocket 経由で受信し、ローカルの [faster-whisper](https://github.com/SYSTRAN/faster-whisper) で文字起こしするサービス。
Omi の `custom_stt` モードを使い、クラウド STT を完全にバイパスします。

## アーキテクチャ

```
Omi デバイス → (WebSocket :8090) → omi-whisper-transcribe → faster-whisper (GPU)
                                         ↑
                              管理 API (:8091)
                                         ↑
                              Viewer  (:8092)
```

- **単体デプロイ**: GPU を専有し、モデルを常時ロード
- **GPU 共有デプロイ**: [gpu-shared](../gpu-shared/) 経由で qwen-tts と GPU を共有。管理 API でモデルのロード/アンロードを外部制御

## 主な機能

| 機能 | 説明 |
|------|------|
| リアルタイム文字起こし | WebSocket で受信した音声を faster-whisper で逐次処理 |
| VAD ベースセグメンテーション | Silero VAD で発話区間を検出し、自然な区切りで文字起こし |
| 無音検出 | RMS ベースで無音を判定し、GPU リソースを節約 |
| 音声前処理 | ラウドネス正規化 + ノイズ除去 (スペクトルゲーティング) |
| ハルシネーションフィルタ | 異常出力・ギブリッシュ仮名・句読点のみ等を除去 |
| セグメント分割 | 長いセグメントを文境界で自然に分割 |
| Hotwords | JMnedict 辞書ベースの固有名詞バイアス |
| LLM 後処理 | Gemini によるバッチ用語修正・提案 (コスト最適化済み) |
| セッションタグ | LLM による自動タグ付け (活動・気分・トピック・要約) |
| カスタムユーザータグ | ユーザー定義タグの作成・付与・LLM 自動適用 |
| バッチ話者分離 | pyannote で全録音の話者分離を一括実行 |
| マルチスピーカー声紋登録 | 複数話者の声紋を登録し、セグメントごとに話者タグ付け |
| 空セッションスキップ | セグメントのないセッションは自動削除 |
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

```bash
docker compose up -d --build
```

初回はモデルダウンロード (約 3GB) に数分かかります。Docker Volume `whisper-models` にキャッシュされるので 2 回目以降は高速です。

### 3. 確認

```bash
# ログ
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
| POST | `/speakers/add` | 話者を登録 `{"name", "is_user"}` |
| PUT | `/speakers/{id}` | 話者情報を更新 |
| DELETE | `/speakers/{id}` | 話者を削除 |
| POST | `/speakers/{id}/voiceprint` | 声紋サンプルを追加 |
| DELETE | `/speakers/{id}/voiceprint/{sample_id}` | 声紋サンプルを削除 |
| PUT | `/speakers/threshold` | 類似度閾値を変更 |
| POST | `/speakers/tag/start` | バッチ話者タグ付けを開始 |
| GET | `/speakers/tag/status` | タグ付け進捗を返す |
| POST | `/speakers/tag/cancel` | タグ付けをキャンセル |
| DELETE | `/speakers/all` | 全話者を削除 |

### セッションタグ

| メソッド | パス | 説明 |
|----------|------|------|
| POST | `/session-tags/start` | バッチセッションタグ付けを開始 |
| GET | `/session-tags/status` | タグ付け進捗を返す |
| POST | `/session-tags/cancel` | タグ付けをキャンセル |

### ユーザータグ

| メソッド | パス | 説明 |
|----------|------|------|
| POST | `/user-tags/reload` | ユーザータグ定義を再読み込み |

## 環境変数

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
| `NO_REPEAT_NGRAM_SIZE` | `3` | N-gram 繰り返し防止 (0=無効) |
| `LOG_PROB_THRESHOLD` | `-0.5` | 対数確率の閾値 |
| `NO_SPEECH_THRESHOLD` | `0.6` | 非発話判定の閾値 |
| `COMPRESSION_RATIO_THRESHOLD` | `2.4` | 圧縮率の閾値 |

### VAD セグメンテーション

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `VAD_SEG_ENABLED` | `true` | VAD ベースセグメンテーションの有効化 |
| `VAD_CHECK_INTERVAL` | `1.0` | VAD チェック間隔 (秒) |
| `VAD_SILENCE_TRIGGER` | `0.8` | 文字起こしトリガーとなる無音時間 (秒) |
| `VAD_MAX_BUFFER` | `30.0` | 強制文字起こしまでの最大バッファ (秒) |
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

### LLM 後処理

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `LLM_ENABLED` | `true` | LLM 後処理の有効化 |
| `LLM_MODEL` | `gemini-2.5-flash-lite` | LLM モデル名 |
| `LLM_TIMEOUT` | `10` | リクエストタイムアウト (秒) |
| `LLM_MIN_TEXT_LENGTH` | `20` | 処理対象の最小テキスト長 |
| `LLM_BATCH_INTERVAL` | `60` | バッチフラッシュ間隔 (秒) |
| `LLM_BATCH_SIZE` | `20` | バッチフラッシュのセグメント数 |

### セッションタグ

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `TAGGING_ENABLED` | `true` | セッション終了時の自動タグ付け |
| `TAGGING_MODEL` | `gemini-2.5-flash-lite` | タグ付け用 LLM モデル名 |

### 機能フラグ

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `DIARIZATION_ENABLED` | `false` | セッション終了時の自動話者分離 |
| `HOTWORDS_ENABLED` | `true` | Hotwords (デコーダバイアス) |

### 外部サービス

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `HF_AUTH_TOKEN` | (空) | HuggingFace トークン (話者分離・声紋登録に必須) |
| `GEMINI_API_KEY` | (空) | Google Gemini API キー (LLM 後処理・セッションタグに必須) |

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

## Hotwords

[JMnedict](https://www.edrdg.org/enamdict/enamdict_doc.html) (CC-BY-SA 3.0) から抽出した日本語固有名詞で faster-whisper のデコーダロジットをバイアスし、固有名詞の認識精度を向上させます。

- `tools/build_hotwords.py` で `config/hotwords.json` を生成
- カテゴリ別 ON/OFF (地名、企業名、駅名、姓、名、製品名)
- カスタム語の追加も可能
- `HOTWORDS_ENABLED=false` で無効化

## LLM 後処理

Gemini API を使い、文字起こし結果の用語修正と新規用語の提案をバッチで行います。

- セグメントを `LLM_BATCH_INTERVAL` 秒 (デフォルト 60) または `LLM_BATCH_SIZE` 個 (デフォルト 20) ごとにまとめて 1 回の API コールで処理
- セッション終了・無音タイムアウト時にも自動フラッシュ
- 登録済み用語辞書 (`config/term_corrections.json`) に基づく自動修正
- 未知の技術用語・固有名詞の修正候補を提案 (`config/pending_suggestions.json`)
- Viewer の Settings > LLM Suggestions で提案を承認/却下

## セッションタグ

Gemini API でセッション終了時に自動でメタデータタグを付与:

| フィールド | 型 | 内容 |
|-----------|-----|------|
| `topics` | リスト | 会話の主要トピック |
| `activity` | enum | 会話、ミーティング、移動中、作業中 等 |
| `intent` | enum | 情報共有、相談、雑談、報告 等 |
| `mood` | enum | ポジティブ、ニュートラル、ネガティブ |
| `quality` | enum | 文字起こし品質 (clear, noisy, mixed) |
| `entities` | リスト | 言及された固有名詞 |
| `summary` | テキスト | 日本語の要約 (1〜2文) |

- `TAGGING_ENABLED=true` でリアルタイム自動タグ付け
- Viewer の「Tag」ボタンからバッチ実行も可能
- `tools/tag_all.py` で CLI からも実行可能

## カスタムユーザータグ

ユーザーがシンプルラベル形式のカスタムタグ (名前・色・説明) を定義し、セッションに付与できます。

### 仕組み

1. **タグ定義**: Viewer の Settings > Tags タブで作成・編集・削除
2. **手動付与**: セッション詳細モーダルでタグチップをクリックして ON/OFF
3. **LLM 自動付与**: セッションタグ付け時にユーザータグ定義もプロンプトに含め、該当するタグを自動適用
4. **フィルタ**: ヘッダーのドロップダウンでユーザータグ別に絞り込み

タグ定義は `config/user_tags.json` に保存。

## 録音 Viewer

`omi-recordings-viewer/` に録音閲覧用の Web アプリを同梱 (ポート 8092):

- 録音一覧・検索 (SQLite FTS)
- 音声再生 + セグメントハイライト
- セッションタグバッジ (活動・気分・トピック) + 要約表示
- カスタムユーザータグバッジ + フィルタ
- 話者バッジ表示 (緑=自分、青=登録話者、グレー=unknown)
- セグメント選択による声紋登録 (新規話者のインライン作成対応)
- バッチ処理パネル (話者分離・話者タグ付け・セッションタグ付け)
- 設定管理 (プロンプト、用語辞書、LLM 提案、プリセット、Hotwords、Voice、Tags)
- 30秒ごとにインクリメンタルスキャン (最新日付以降のみ)

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

## GPU 共有デプロイ

qwen-tts と GPU を共有する場合は [gpu-shared](../gpu-shared/) を使用してください。
omi-whisper 自体は GPU 共有の知識を持たず、管理 API を通じて外部から制御されます。

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
├── gpu_manager.py            # GPU リソース管理 (qwen-tts との排他制御)
├── model_manager.py          # Whisper モデルの外部制御 (gpu-shared 用)
├── audio_preprocessor.py     # ノイズ除去 + ラウドネス正規化
├── llm_corrector.py          # Gemini によるバッチ用語修正
├── session_tagger.py         # Gemini によるセッションタグ付け
├── diarizer.py               # pyannote バッチ話者分離
├── speaker_filter.py         # 声紋登録・話者タグ付け
├── hotword_loader.py         # Hotwords ロード・管理
├── term_corrector.py         # 用語辞書管理
├── tts_proxy.py              # TTS プロキシ (gpu-shared 連携)
├── config/                   # 設定ファイル (JSON)
├── tools/                    # 運用スクリプト
├── tests/                    # ユニットテスト
├── omi-recordings-viewer/    # Viewer Web アプリ
│   ├── app.py                # aiohttp サーバー + API
│   ├── indexer.py            # SQLite インデックス管理
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
| [Gemini API](https://ai.google.dev/) | LLM 後処理・セッションタグ | Google ToS |
