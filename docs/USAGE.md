# 使い方マニュアル

Omi ウェアラブルデバイスの音声をローカル GPU で文字起こし・整理・活用するまでの、
セットアップから日常運用までの手順をまとめたマニュアルです。
機能の網羅的なリファレンス（環境変数・管理 API の全一覧）は [README.md](../README.md) を参照してください。

## 目次

1. [全体像](#全体像)
2. [初期セットアップ](#初期セットアップ)
3. [Omi デバイスの接続](#omi-デバイスの接続)
4. [日々の使い方（自動処理の流れ）](#日々の使い方自動処理の流れ)
5. [Viewer の使い方](#viewer-の使い方)
6. [声紋の登録（初回セットアップ）](#声紋の登録初回セットアップ)
7. [タグのレビューとタクソノミー運用](#タグのレビューとタクソノミー運用)
8. [トレーニングデータのエクスポート](#トレーニングデータのエクスポート)
9. [事業継承 RAG / LLM Wiki](#事業継承-rag--llm-wiki)
10. [管理 API チートシート](#管理-api-チートシート)
11. [セキュリティ上の注意](#セキュリティ上の注意)
12. [困ったときは](#困ったときは)

---

## 全体像

```
Omi デバイス ──BLE──> Omi アプリ (スマホ)
                        │  custom_stt (WebSocket, 音声ストリーム)
                        ▼
              omi-whisper-transcribe (:8090)      ← Bridge (GPU で文字起こし)
                        │ 管理 API (:8091)
                        ▼
              omi-recordings-viewer (:8092)       ← Viewer (ブラウザ UI)
```

- **Bridge**: 音声を受信して faster-whisper (または Qwen3-ASR) で文字起こしし、
  `recordings/YYYY-MM-DD/HH-MM-SS/` に音声 (OGG) と `transcript.json` を保存。
  セッション終了後に LLM 補正・タグ付けを自動実行します。
- **Viewer**: 録音の一覧・検索・再生・タグ修正・声紋管理・エクスポート・事業継承チャットを行う Web UI。

## 初期セットアップ

### 1. 前提環境

| 項目 | 要件 |
|------|------|
| GPU | NVIDIA (large-v3 は VRAM 12GB 推奨、`COMPUTE_TYPE=int8` なら 6GB〜) |
| OS | Linux または WSL2 |
| ソフトウェア | Docker + Docker Compose + NVIDIA Container Toolkit |

NVIDIA Container Toolkit が未導入の場合:

```bash
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# 動作確認
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

### 2. API キーの準備

| キー | 用途 | 必須度 |
|------|------|--------|
| `GEMINI_API_KEY` | LLM 補正・セッションタグ・匿名化 | ほぼ必須（ないと文字起こしのみ） |
| `HF_AUTH_TOKEN` | 話者分離 (pyannote)・声紋登録 | 話者機能を使うなら必須 |
| `OPENAI_API_KEY` | gpt-5-nano 匿名化・事業継承 RAG | 該当機能を使うなら必須 |

> **pyannote の注意**: HuggingFace 上で
> [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
> 等の利用規約に同意し、トークンに *Access public gated repos* 権限を付けてください。

### 3. .env の作成

```bash
cd omi_whisper_transcribe
cp .env.example .env
# GEMINI_API_KEY / HF_AUTH_TOKEN / OMI_WHISPER_TOKEN を設定
```

### 4. 起動

```bash
# 単体デプロイ (GPU 専有)
docker compose up -d --build

# gpu-shared 構成 (qwen-tts と GPU 共有) の場合は必ず gpu-shared/ から起動
cd ../gpu-shared && docker compose up -d --build
```

初回はモデルダウンロード（約 3GB）に数分かかります。

### 5. 動作確認

```bash
docker compose logs -f omi-whisper            # 起動ログにエラーがないこと
curl http://localhost:8091/model/status       # {"loaded": true, ...} ならモデル準備完了
```

ブラウザで `http://localhost:8092` を開き、Viewer が表示されれば OK です
（録音がまだないので一覧は空です）。

## Omi デバイスの接続

Omi アプリの **custom STT モード** を使い、クラウド STT を完全にバイパスします。

### LAN 内で使う場合

1. ホスト PC の IP アドレスを確認（例: `192.168.1.10`）。
   WSL2 の場合はポートフォワード設定か `netsh interface portproxy` が必要です。
2. Omi アプリで Developer Settings（開発者設定）を開き、**Custom STT** の
   WebSocket URL に以下を設定:

   ```
   ws://192.168.1.10:8090/transcribe
   ```

   `OMI_WHISPER_TOKEN` を設定している場合は `?token=<TOKEN>` を付加します。
3. デバイスを装着して話すと、Omi アプリ上にこのサーバーの文字起こし結果が表示されます。

### 外出先から使う場合 (Cloudflare Tunnel)

1. Tunnel を作成し `whisper.example.com` → `http://localhost:8090` にルーティング
2. Cloudflare Access で `/transcribe*` パスを Bypass に設定
3. `.env` の `OMI_WHISPER_TOKEN` を設定（必須）
4. Omi アプリの Custom STT に `wss://whisper.example.com/transcribe?token=<TOKEN>` を設定

### SD カード録音の取り込み (オフライン録音)

Omi デバイスの SD カードに溜まった録音は管理 API から取り込めます（1 ファイル 100MB まで）:

```bash
curl -X POST http://localhost:8091/sdcard/upload \
  --data-binary @recording.bin
```

## 日々の使い方（自動処理の流れ）

装着して話す以外の操作は基本的に不要です。処理は 3 層で自動的に進みます。

| タイミング | 処理 |
|-----------|------|
| リアルタイム | VAD で発話区間を検出 → Whisper で文字起こし → ハルシネーション除去 → Omi アプリへ返送 |
| セッション終了直後 | 録音を OGG に圧縮、LLM 補正（用語・表記）とセッションタグ（会話種類・成果物・事業領域・要約）を付与 |
| 深夜バッチ (デフォルト 4:00) | 話者分離 → 声紋タグ付け → 音響シーン分類 → 未タグセッションのタグ付け。毎月 1 日はタクソノミー自動精緻化も実行 |

- セッションは **無音 30 秒**（`SILENCE_TIMEOUT`）または WebSocket 切断で終了します。
- 空のセッション（発話なし）は自動削除されます。
- 録音は `MAX_RECORDING_DAYS`（デフォルト 30 日）を過ぎると自動削除されます。
- 深夜バッチの対象は直近 `BATCH_MAX_DAYS`（デフォルト 7 日）のセッションです。

すぐにバッチを回したい場合は Viewer の「タグ付け」ボタン、または
`curl -X POST http://localhost:8091/nightly/trigger` を使います。

## Viewer の使い方

`http://localhost:8092` を開くと録音一覧が表示されます。

### 一覧・検索・フィルタ

- **検索ボックス**: 文字起こし本文の全文検索（Enter で実行）。
- **Filter ボタン**: 会話種類 / 成果物 / 事業領域 / 重要度 / 話者 / 音響シーンで絞り込み。
- セッションカードには要約・タグ・話者・「TV/外部音声」バッジ（外部音声比率 ≥ 0.8）などが表示されます。

### セッション詳細（カードをクリック）

- **音声再生**: 再生位置に合わせて該当セグメントがハイライトされます。
  セグメントをクリックするとその位置にシークします。
- **話者の修正**: セグメントの話者ラベルをクリックすると修正ドロップダウンが開きます。
  修正内容は声紋の学習サンプルとして自動反映されます。
- **タグの編集**: タグ表示部の編集ボタンから会話種類・成果物・事業領域・重要度を修正できます。
  ドメインタグを手動修正すると、そのセッションから新しいキーワード候補が自動抽出されます
  （→ [タグのレビューとタクソノミー運用](#タグのレビューとタクソノミー運用)）。

### バッチ実行

ヘッダーの「タグ付け」ボタンから、話者タグ → 音響シーン → セッションタグの一括バッチを起動できます。
進捗はパネルに表示され、キャンセルも可能です。GPU は Whisper セッションと排他制御されるため、
バッチ実行中は新規の文字起こし接続が一時的に待たされます（1013 で拒否されることがあります）。

### レビューモード（ヘッダーの「レビュー」ボタン）

タグの確からしさが低いセッションから優先的に提示されます（不確実度・新規性・鮮度などのスコア順）。

| キー | 動作 |
|------|------|
| `A` | 現在のタグを承認 (Accept) |
| `E` | タグを編集 (Edit) |
| `S` / `Space` | スキップ |
| `D` | サイドドロワー（用語・タクソノミー提案）の開閉 |
| `Esc` | レビューモードを閉じる |

### 設定（歯車アイコン）

| タブ | 内容 |
|------|------|
| ASR Engine | 現在の Whisper/Qwen3-ASR 設定の確認、エンジン切り替え |
| 用語補正 | 誤認識→正しい表記の辞書 (`term_corrections.json`) の追加・編集・削除 |
| Hotwords | 固有名詞バイアスのカテゴリ設定・カスタム hotword の追加 |
| LLM 提案 | LLM が検出した表記ゆれ候補の承認・拒否（承認すると辞書と hotwords に反映） |
| Taxonomy | タグ分類軸（会話種類・成果物・事業領域・キーワードアンカー）の編集、キーワード候補の承認 |
| RAG | RAG 分類のインデックス状態確認・再構築 |
| Refiner | 月次タクソノミー精緻化の実行・提案の承認/拒否 |
| Voice | 話者と声紋の管理（→ 次章） |
| Export | トレーニングデータエクスポート・サニタイズ実行 |
| 事業相談 | 事業継承 RAG へのチャット (Q&A) |

## 声紋の登録（初回セットアップ）

自分の声を「ユーザー」として登録すると、話者タグ付け・外部音声（TV 等）の判定精度が上がります。

1. まず数セッション録音し、深夜バッチか手動バッチで **話者分離** を実行しておく
2. Viewer → Settings → **Voice** タブを開く
3. 「+ New Speaker」で自分の名前を登録し **is_user** を ON
4. 自分がはっきり話しているセッション・セグメントを選んで声紋サンプルとして追加
   （複数サンプル登録すると精度が上がります。サンプルはインライン再生で確認可能）
5. 類似度しきい値（デフォルト 0.5 前後）は Voice タブで調整可能

以後、バッチタグ付けで各セグメントに「自分/登録話者/unknown」が付与されます。
セグメント単位の手動修正（詳細画面）を繰り返すと自動学習でサンプルが追加されていきます。

> 声紋の埋め込みモデルを変更した場合は `tools/auto_reenroll_user.py` で再エンロールしてください。

## タグのレビューとタクソノミー運用

タグ付けは「LLM/RAG による自動付与 → 人間のレビュー → 学習」のループで精度が上がっていきます。

1. **自動付与**: セッション終了時と深夜バッチで自動タグ付け。
   RAG 分類（過去セッションの類似度投票）で確信度が高い場合は LLM を呼ばずコスト削減。
2. **レビュー**: レビューモードで不確実なタグから順に承認・修正。
3. **学習**:
   - ドメイン修正 → キーワード候補が自動抽出され、Taxonomy タブで承認すると
     `keyword_anchors` に追加（以後の事前フィルタ・分類精度に反映）
   - 承認済みセッションは RAG インデックスに追加され、以後の類似セッション分類に使われる
4. **月次精緻化**: 毎月 1 日の深夜バッチでタクソノミー全体の妥当性を LLM が検証し、
   改善提案を生成。Settings → Refiner で承認/拒否。

## トレーニングデータのエクスポート

Settings → **Export** タブから、蓄積した会話をファインチューニング用データとして出力できます。

1. **フィルタ設定**: 期間・品質・会話種類・重要度・「個人のみを除外」・外部音声比率上限など
2. **サニタイズ実行**（chat 系フォーマットに必須）: 「Sanitize」で LLM 匿名化を実行。
   固有名詞の匿名化と knowledge card（Q&A 形式の知見）抽出が行われ、
   `config/safe_training_*.jsonl` に保存されます
3. **エクスポート**: フォーマット（OpenAI / ShareGPT / Comprehensive / Knowledge / Decision Log / SOP）を
   選んでダウンロード。train/val 分割（層化サンプリング）も可能で、その場合は ZIP になります

## 事業継承 RAG / LLM Wiki

サニタイズ済み knowledge card を知識ベースとして、根拠付きで質問に答える Q&A システムです
（`GEMINI_API_KEY` と `OPENAI_API_KEY` の両方が必要）。

1. サニタイズを実行して knowledge card を蓄積（前章）
2. インデックス構築: Settings → 事業相談 →「Rebuild」
   （または `python tools/build_succession_index.py`）
3. 事業相談タブで質問すると、関連 KC を検索して根拠を引用しながら回答します

**LLM Wiki** (`~/knowledge/`) を運用している場合は `tools/build_wiki.py` で
KC からトピック別 markdown を生成でき、Wiki のトピックが RAG の親ドキュメントとして
検索に組み込まれます（`tools/cron_wiki_update.sh` で夜間自動更新可能）。

## 管理 API チートシート

管理 API (`:8091`) はすべて認証なしの HTTP です。よく使うもの:

```bash
# モデル状態 / 設定
curl http://localhost:8091/model/status
curl http://localhost:8091/whisper/config

# バッチ実行 (話者分離 / 統合タグ付け / ナイトリー即時実行)
curl -X POST http://localhost:8091/diarize/start
curl -X POST http://localhost:8091/tags/start
curl -X POST http://localhost:8091/nightly/trigger

# 進捗確認 / キャンセル
curl http://localhost:8091/tags/status
curl -X POST http://localhost:8091/tags/cancel

# エンジン切り替え (whisper <-> qwen3asr)
curl -X POST http://localhost:8091/engine/switch -d '{"engine": "qwen3asr"}'

# RAG インデックス再構築 / 状態
curl -X POST http://localhost:8091/rag/rebuild
curl http://localhost:8091/rag/status

# 事業継承 RAG に質問
curl -X POST http://localhost:8091/succession/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "主要顧客との取引条件は？"}'
```

全エンドポイントの一覧は [README.md の管理 API 章](../README.md#管理-api-port-8091) を参照。

## セキュリティ上の注意

- **管理 API (8091) と Viewer (8092) には認証がありません。**
  LAN 外に公開しないでください。外部公開するのは Cloudflare Tunnel 経由の
  `/transcribe`（トークン認証あり）のみにし、8091/8092 はファイアウォールで塞ぐこと。
- 録音には会話がそのまま含まれます。`recordings/` と `config/`（辞書・声紋・キュー）を
  バックアップ対象・アクセス制御対象として扱ってください。
- 外部にデータを出すのは Gemini / OpenAI API 呼び出し（LLM 補正・タグ・匿名化・RAG）のみです。
  完全ローカルにしたい場合は `LLM_ENABLED=false TAGGING_ENABLED=false` で文字起こし専用になります。

## 困ったときは

| 症状 | 確認すること |
|------|-------------|
| Omi アプリに文字が出ない | `docker compose logs omi-whisper` で接続ログを確認。トークン不一致 (4401)、GPU ビジー (1013) など |
| 文字起こしが始まらない | `curl :8091/model/status` で `loaded` を確認。バッチ実行中は待たされます |
| Viewer に録音が出ない | Viewer は起動時と定期スキャンでインデックスします。コンテナ再起動で再同期 |
| 話者分離が 403 | HF トークンの権限と pyannote 規約同意を確認 |
| タグが付かない | `GEMINI_API_KEY` 設定と `docker compose logs omi-whisper` の LLM エラーを確認 |
| VRAM 不足 / OOM | `COMPUTE_TYPE=int8` や `WHISPER_MODEL=medium` に変更。バッチとセッションの同時実行を避ける |

その他は [README.md のトラブルシューティング](../README.md#トラブルシューティング) を参照してください。
