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

### 1. トークン生成 (任意)

Cloudflare Tunnel 経由で公開する場合のみ:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### 2. 起動

```bash
cd omi_whisper_transcribe

# トークンを設定する場合
export OMI_WHISPER_TOKEN=<生成したトークン>

docker compose up -d --build
```

初回はモデルダウンロード (約 3GB) に数分かかります。Docker Volume `whisper-models` にキャッシュされるので 2 回目以降は高速です。

### 3. 確認

```bash
# ログ
docker compose logs -f

# 管理 API でモデル状態を確認
curl http://localhost:8091/model/status
```

## ポート

| ポート | プロトコル | 用途 |
|--------|-----------|------|
| 8090 | WebSocket | 音声受信 + 文字起こし結果送信 |
| 8091 | HTTP | 管理 API |

## 管理 API (port 8091)

| エンドポイント | メソッド | 動作 |
|---|---|---|
| `/model/load` | POST | モデルロード (冪等) |
| `/model/unload` | POST | モデルアンロード (セッション中は 409) |
| `/model/status` | GET | `{"loaded": bool, "session_count": int, "model_name": "...", "compute_type": "..."}` |

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

## 録音 Viewer

`omi-recordings-viewer/` に録音閲覧用の Web アプリを同梱:

- 録音一覧・検索 (SQLite FTS)、音声再生、セグメントハイライト
- 30秒ごとにインクリメンタルスキャン (最新日付以降のみ)
- ポート 8092 でアクセス

gpu-shared デプロイでは自動起動。単体デプロイでは別途起動が必要:

```bash
cd omi-recordings-viewer
docker build -t omi-recordings-viewer .
docker run -d -p 8092:8092 -v ../recordings:/app/recordings:ro omi-recordings-viewer
```

## GPU 共有デプロイ

qwen-tts と GPU を共有する場合は [gpu-shared](../gpu-shared/) を使用してください。
omi-whisper 自体は GPU 共有の知識を持たず、管理 API を通じて外部から制御されます。

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
