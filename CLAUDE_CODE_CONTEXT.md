# omi_whisper_transcribe — プロジェクト概要・設計経緯

Claude Code でソース分析・改善を行う際のコンテキスト情報。

---

## プロジェクト概要

Omi AI ウェアラブルデバイスの音声を、クラウド STT サービスを使わずに
ローカル PC の faster-whisper (RTX 3060) で文字起こしするセルフホスト構成。

## 技術スタック

- **STT エンジン**: faster-whisper (CTranslate2 ベース、large-v3 モデル、int8_float16)
- **実行環境**: WSL2 + Docker (nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04)
- **GPU**: NVIDIA RTX 3060 12GB
- **ネットワーク**: Cloudflare Tunnel + Access (外出先からも利用可能)
- **Omi 接続方式**: `custom_stt=enabled` モード (WebSocket)

## ファイル構成

```
omi_whisper_transcribe/
├── omi_whisper_bridge.py      # メインアプリ (WebSocket サーバー + Whisper + 録音保存)
├── test_whisper.py            # GPU/モデル動作テストスクリプト
├── Dockerfile                 # CUDA 12.3 + cuDNN 9 ベースイメージ
├── docker-compose.yml         # GPU パススルー + ボリュームマウント
├── requirements-docker.txt    # Docker 用 pip 依存 (CUDA ライブラリは不要)
├── requirements.txt           # ローカル実行用 pip 依存
├── .env.example               # 認証トークンのテンプレート
└── README.md                  # セットアップガイド
```

---

## 設計判断の経緯

### 1. なぜ faster-whisper か (vs whisper.cpp)

RTX 3060 (CUDA GPU) がある環境では faster-whisper が最適と判断。
- CTranslate2 の CUDA 最適化で最大4倍高速
- int8 量子化対応で VRAM 消費が少ない (large-v3 で約5GB)
- ストリーミング対応の API ラッパー (speaches 等) が豊富
- whisper.cpp は CPU 最適化が主軸で、GPU 環境ではメリットが薄い
- ハルシネーション (無音時の繰り返し) も faster-whisper の方がマシ

### 2. なぜ custom_stt モードか (vs バックエンド丸ごとセルフホスト)

Omi には3つの STT 接続ルートがある:
- A: 組み込み STT (Deepgram/Soniox/Speechmatics) → クラウド API 前提、ローカル不可
- B: custom_stt モード → アプリが STT 結果を JSON で返す。**採用**
- C: バックエンド丸ごとセルフホスト → Firebase/Pinecone/Redis 等の依存が重すぎ

B を選んだ理由: ブリッジアプリ1本書くだけで済む。STT エンジンの選択が完全に自由。

### 3. なぜ Cloudflare Tunnel か (vs Tailscale)

外出先からも使いたいという要件に対して:
- Tailscale: セキュリティは最高 (エンドポイント非公開) だが、スマホに常時 VPN アプリが必要。バッテリー消費、他 VPN との競合あり。
- Cloudflare Tunnel: スマホ側追加アプリ不要。URL 指定だけで完結。

Cloudflare Access の問題:
- Omi アプリは WebSocket ハンドシェイク時にカスタム HTTP ヘッダーを付けられない
- CF-Access-Client-Id / CF-Access-Client-Secret ヘッダーが送れない
- → 解決策: `/transcribe*` パスだけ Access Bypass にして、アプリ側でトークン認証

セキュリティの3層構造:
1. TLS 暗号化 (Cloudflare 提供の wss://)
2. Cloudflare Access で /transcribe 以外のパスを保護
3. ブリッジアプリ側で URL クエリパラメータのトークン検証

### 4. なぜ Speechmatics セルフホストではないか

Omi は stt_service=speechmatics で直接接続できるが:
- セルフホストにはエンタープライズ契約のライセンスが必要 (個人には非現実的)
- クラウド API は月8時間無料、超過分は $1.35/時間 (Enhanced リアルタイム)
- 常時ON (8h/日) だと月 $313+ になる
- faster-whisper セルフホストなら電気代のみ (月2,000-3,000円)

### 5. 音声保存機能

- セッション (WebSocket 接続〜切断) ごとに WAV + JSON を保存
- 保存先: `recordings/YYYY-MM-DD/HH-MM-SS/audio.wav` + `transcript.json`
- transcript.json のセグメント start/end は WAV の秒数と一致 → 音声とテキストが紐付き
- MAX_RECORDING_DAYS で自動削除 (デフォルト30日)
- 常時ON 8h/日で WAV 約27GB/月
- 将来 NAS に保存先を変更可能 (docker-compose のマウント先変更のみ)

---

## Omi custom_stt プロトコル仕様

### WebSocket 接続

Omi バックエンドが WebSocket でブリッジアプリに接続。
custom_stt=enabled の場合、バックエンドは STT サービスを使わず、
アプリが返すトランスクリプトをそのまま採用する。

### 受信 (Omi → ブリッジ)

- **バイナリメッセージ**: PCM 16-bit LE, 16kHz, mono の音声チャンク
- **2バイト以下のバイナリ**: Keep-alive ping (無視)
- **テキストメッセージ**: JSON コマンド (speaker_assigned 等)

### 送信 (ブリッジ → Omi)

```json
{
  "segments": [
    {
      "text": "文字起こしテキスト",
      "speaker": "SPEAKER_00",
      "speaker_id": 0,
      "start": 0.0,
      "end": 2.3,
      "is_user": true,
      "person_id": null
    }
  ]
}
```

重要:
- `segments` キーを持つオブジェクトであること (生配列はNG)
- `type` フィールドは含めない、または "Results" にする
- 90秒の無通信でコネクション切断

---

## omi_whisper_bridge.py の処理フロー

1. 起動時に faster-whisper モデルを GPU にロード
2. 古い録音を自動クリーンアップ (MAX_RECORDING_DAYS)
3. WebSocket サーバーを 0.0.0.0:8090 で起動
4. 接続ごとに:
   a. URL の ?token= を検証 (AUTH_TOKEN が設定されている場合)
   b. SessionRecorder を初期化 (WAV ファイルを開く)
   c. バイナリ音声チャンクを受信 → audio_buffer に追加 + WAV に書き込み
   d. バッファが BUFFER_SECONDS (3秒) に達したら:
      - run_in_executor で faster-whisper を呼び出し (非ブロッキング)
      - 結果を Omi 形式の JSON に変換して送信
      - セグメントを SessionRecorder に蓄積
   e. 接続終了時に残りバッファを処理し、WAV を閉じて JSON を書き出す

---

## 現在の制限・改善候補

### 既知の制限
- 話者分離 (diarization) 未対応: 全セグメントが SPEAKER_00 固定
- Opus デコード未対応: Omi が Opus で送ってくる場合は別途デコード処理が必要
- WAV 非圧縮保存: 長期保存にはストレージ消費が大きい (Opus/FLAC 変換未実装)
- 単一セッション前提: 同時接続は未テスト

### 改善候補
- 話者分離: pyannote-audio または WhisperX 統合
- 音声圧縮: 録音後バックグラウンドで Opus/FLAC に変換するジョブ
- NAS 保存: docker-compose のマウント先を SMB 共有に変更
- 検索UI: recordings/ 内の JSON を全文検索する簡易 Web UI
- Prometheus メトリクス: RTF、セッション数、VRAM 使用率の監視
- Omi からの Opus 受信対応: opuslib でデコードしてから Whisper に渡す
