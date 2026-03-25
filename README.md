# omi-whisper-transcribe

Omi デバイスの音声を WebSocket 経由で受信し、ローカルの [faster-whisper](https://github.com/SYSTRAN/faster-whisper) で文字起こしするサービス。
Omi の `custom_stt` モードを使い、クラウド STT を完全にバイパスします。

## アーキテクチャ

```
Omi デバイス → (WebSocket :8090) → omi-whisper-transcribe → faster-whisper (GPU)
                                         ↑
                              管理 API (:8091)
```

- **単体デプロイ**: GPU を専有し、モデルを常時ロード
- **GPU 共有デプロイ**: [gpu-shared](../gpu-shared/) 経由で qwen-tts と GPU を共有。管理 API でモデルのロード/アンロードを外部制御

## 必要環境

| 項目 | 要件 |
|------|------|
| GPU | NVIDIA (VRAM 6GB 以上、large-v3 は 12GB 推奨) |
| Docker | Docker + NVIDIA Container Toolkit |
| OS | Linux / WSL2 |

## クイックスタート

[gpu-shared](../gpu-shared/) 経由でデプロイします。

### 1. .env 設定

```bash
cd gpu-shared
cp .env.example .env
# .env を編集して以下を設定:
#   HF_AUTH_TOKEN=hf_xxx    (話者分離を使う場合、pyannote の利用規約同意が必要)
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

# Viewer (録音一覧 + バッチ話者分離)
open http://localhost:8092
```

## ポート

| ポート | プロトコル | 用途 |
|--------|-----------|------|
| 8090 | WebSocket | 音声受信 + 文字起こし結果送信 |
| 8091 | HTTP | 管理 API (内部、gpu-shared 経由) |
| 8092 | HTTP | 録音 Viewer |

## 管理 API (port 8091)

| エンドポイント | メソッド | 動作 |
|---|---|---|
| `/model/load` | POST | モデルロード (冪等) |
| `/model/unload` | POST | モデルアンロード (セッション中は 409) |
| `/model/status` | GET | `{"loaded": bool, "session_count": int, "model_name": "...", "compute_type": "..."}` |
| `/diarize/start` | POST | バッチ話者分離を開始 (実行中は 409、HF_AUTH_TOKEN 未設定は 503) |
| `/diarize/status` | GET | バッチ話者分離の進捗状態を JSON で返す |
| `/diarize/cancel` | POST | 実行中のバッチ話者分離をキャンセル |

## 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `WHISPER_MODEL` | `large-v3` | モデル名 (tiny, base, small, medium, large-v3) |
| `COMPUTE_TYPE` | `int8_float16` | 量子化 (float16, int8_float16, int8) |
| `LANGUAGE` | `ja` | 言語 (ja, en, auto) |
| `BEAM_SIZE` | `5` | ビームサイズ |
| `VAD_FILTER` | `true` | VAD フィルタ (ハルシネーション対策) |
| `BUFFER_SECONDS` | `3.0` | 音声バッファ秒数 |
| `PORT` | `8090` | WebSocket ポート |
| `MANAGEMENT_PORT` | `8091` | 管理 API ポート |
| `SAVE_AUDIO` | `true` | 録音保存 |
| `RECORDINGS_DIR` | `./recordings` | 保存先 |
| `MAX_RECORDING_DAYS` | `30` | 自動削除日数 (0=無効) |
| `SILENCE_THRESHOLD` | `200` | 無音検出 RMS 閾値 (0=無効) |
| `SILENCE_TIMEOUT` | `30.0` | 無音タイムアウト秒数 |
| `DIARIZATION_ENABLED` | `false` | セッション終了時の自動話者分離 |
| `HF_AUTH_TOKEN` | (空) | HuggingFace トークン (話者分離に必須) |
| `OMI_WHISPER_TOKEN` | (空) | 認証トークン (空=認証なし) |

### モデル選択の目安

| モデル | VRAM (int8) | 速度 | 日本語精度 |
|--------|-------------|------|-----------|
| large-v3 | ~5GB | RTF 0.21x | 最高 |
| medium | ~3GB | 中 | 良好 |
| small | ~1.5GB | 速い | そこそこ |

RTX 3060 12GB なら `large-v3` + `int8_float16` がベストバランス。

## 録音データ

セッションごとに `recordings/YYYY-MM-DD/HH-MM-SS/` に保存:
- `audio.wav` -- PCM 16kHz 16-bit mono
- `transcript.json` -- メタデータ + 文字起こしセグメント

データ量目安: 常時 ON 8h/日で WAV 約 27GB/月。

## 無音検出

RMS ベースの無音判定で GPU リソースを節約:

- 無音バッファは whisper 処理と WAV 書き込みをスキップ
- 30秒連続で無音が続くと録音セッションを自動終了 (WAV 確定保存)
- 発話が再開すると新しいセッションを自動開始
- WebSocket 接続自体は切断しない (Omi 側のライフサイクルに任せる)

`SILENCE_THRESHOLD=0` で無効化可能。閾値 200 は環境ノイズ (50〜300) をカットし、小声 (500〜) は通す設定。

## バッチ話者分離

Viewer の「Diarize」ボタンから、未処理の全録音に対して一括で話者分離を実行できます。

```
[Browser]  ←→  [Viewer (8092)]  ←→  [Bridge (8091)]
  UI + 進捗表示     プロキシ API         GPU + Diarizer
```

### 使い方

1. `.env` に `HF_AUTH_TOKEN=hf_xxx` を設定 ([pyannote](https://huggingface.co/pyannote/speaker-diarization-3.1) の利用規約に同意が必要)
2. Viewer を開くとヘッダーに「Diarize」ボタンが表示される
3. ボタンを押すと確認ダイアログ → バッチ処理開始
4. 進捗パネルでリアルタイムに進行状況を確認 (2秒ポーリング)
5. 完了後、各セッションの transcript.json に話者情報が追加される

### 動作仕様

- `HF_AUTH_TOKEN` があれば Diarizer が常に利用可能 (バッチ用)
- `DIARIZATION_ENABLED=true` の場合のみ、セッション終了時に自動で話者分離も実行
- GPU ロックで Whisper セッションと排他制御 (処理中は Whisper をアンロード)
- キャンセルは現在処理中のセッション完了後に安全に停止
- エラー時も元のセグメントは保持される (テキスト消失なし)

## 録音 Viewer

`omi-recordings-viewer/` に録音閲覧用の Web アプリを同梱 (ポート 8092):

- 録音一覧・検索 (SQLite FTS)
- 音声再生 + セグメントハイライト
- バッチ話者分離 (上記参照)
- 設定管理 (プロンプト、用語辞書、LLM 提案、プリセット)
- 30秒ごとにインクリメンタルスキャン (最新日付以降のみ)

## Cloudflare Tunnel

外出先からアクセスする場合:

1. Tunnel を作成し `whisper.example.com` → `http://localhost:8090` にルーティング
2. Cloudflare Access で `/transcribe*` パスを Bypass に設定
3. `OMI_WHISPER_TOKEN` でアプリ側のトークン認証を有効化
4. Omi アプリの Custom STT に `wss://whisper.example.com/transcribe?token=<TOKEN>` を設定

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| GPU が認識されない | NVIDIA Container Toolkit のインストールと `nvidia-ctk runtime configure` を確認 |
| VRAM 不足 | `COMPUTE_TYPE=int8` または `WHISPER_MODEL=medium` に変更 |
| ハルシネーション | `VAD_FILTER=true` を確認、`BUFFER_SECONDS` を短くする |
| Unauthorized | `?token=` の値と `OMI_WHISPER_TOKEN` が一致しているか確認 |
