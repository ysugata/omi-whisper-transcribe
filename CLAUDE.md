# CLAUDE.md

## 言語

- ユーザーへの応答（サマリー、説明、確認など）は **すべて日本語** で行うこと
- コード中のコメントやコミットメッセージは英語でも可

## Git ワークフロー

### Worktree ベースの並列開発

ソースを書き換える場合は **git worktree** を使い、メインの作業ディレクトリを汚さずに並列開発する。

```bash
# 1. まず main を最新化（古い main から分岐しないように）
git fetch origin main

# 2. worktree を作成（ベースブランチを明示）
git worktree add ../omi_whisper_transcribe-<機能名> -b <branch名> origin/main
# 例: git worktree add ../omi_whisper_transcribe-fix-filter -b fix/filter-user-tag origin/main

# 3. worktree 内で開発・テスト・コミット・プッシュ

# 4. マージ後にクリーンアップ
git worktree remove ../omi_whisper_transcribe-<機能名>
git branch -d <branch名>
git push origin --delete <branch名>
```

### ブランチルール

- ブランチ名は内容に応じて `feature/xxx` や `fix/xxx` 等、適切に決める
- main に直接コミットしない
- 作業完了後はコミット＆プッシュし、特に指示がなければ main にマージする
- マージ後は main にチェックアウトして pull し、最新の状態にしておく
- **マージ完了後はローカル・リモート両方のブランチを削除する**
- **worktree のディレクトリも削除する** (`git worktree remove`)

### 自動管理 (cron)

- **30分ごと**: `git fetch origin main` で main を最新に保つ
- **毎日 AM3:30**: `~/scripts/git-worktree-cleanup.sh` でマージ済み worktree を自動削除

## テスト — 必須ステップ

コードを修正・追加した場合、**コミット前に以下をすべて実行**すること。
「ビルド成功 ≠ 動作確認」。ビルドが通っても実際に動かさなければテスト完了ではない。

### Step 1: ユニットテスト

- 変更した関数・モジュールの pytest を実行
- 既存テストがない場合は最低限のテストを書く

### Step 2: Docker ビルド

> **重要: worktree 使用時のビルド順序**
> Docker Compose は `omi_whisper_transcribe/` (main) のコードをマウント/COPYする。
> **worktree で開発している場合、先に main へマージしてからビルド**すること。
> worktree ディレクトリからビルドしても、コンテナには main のコードが入るため、
> 変更が反映されず結合テストで古いコードをテストしてしまう。
>
> 正しい手順: worktree でコミット → main にマージ → `docker compose build`

```bash
docker compose build <変更したサービス>
```

### Step 3: コンテナ起動 + 結合テスト

ビルド後、**必ず変更した機能を実際のコンテナで動かして検証する**。

> **重要**: この環境は gpu-shared 構成のため、Docker Compose コマンドは
> **`/home/dev/gpu-shared/` ディレクトリから実行**すること。
> `omi_whisper_transcribe/` から実行すると Viewer が別ネットワークに配置され、
> Bridge に接続できなくなる。

変更内容に応じて以下のチェックを行う:

#### Bridge (omi_whisper_bridge.py, 各 Python モジュール) を変更した場合
1. `cd /home/dev/gpu-shared && docker compose up -d --build omi-whisper` でコンテナを起動
2. `cd /home/dev/gpu-shared && docker compose logs omi-whisper` で起動エラーがないことを確認
3. 変更した管理 API エンドポイントに curl でリクエストし、期待するレスポンスを確認

#### Viewer (app.py, indexer.py, templates, JS, CSS) を変更した場合
1. `cd /home/dev/gpu-shared && docker compose up -d --build omi-recordings-viewer` でコンテナを起動
2. `cd /home/dev/gpu-shared && docker compose logs omi-recordings-viewer` で起動エラーがないことを確認
3. 変更した API エンドポイントに curl でリクエストし、期待するレスポンスを確認
4. UI を変更した場合: `curl http://localhost:8092/` で HTML を取得し、追加した要素が含まれることを確認

#### Docker 設定 (docker-compose.yml, Dockerfile) を変更した場合
1. `cd /home/dev/gpu-shared && docker compose up -d` で全コンテナを起動
2. `cd /home/dev/gpu-shared && docker compose ps` で全コンテナが running であることを確認
3. 変更に関連するポート・ボリューム・環境変数が正しく機能することを確認

#### 設定ファイル (config/*.json) のスキーマ変更や読み込みロジックを変更した場合
1. コンテナを再起動して設定が正しくロードされることをログで確認

### テスト不合格時

- テストが通らない状態では **絶対にコミットしない**
- 結合テストで問題が見つかった場合、修正してから再テスト

## ドキュメント

- 機能追加・変更・削除を行った場合、**README.md も合わせて更新**すること
- 更新対象: 機能一覧、環境変数テーブル、管理 API テーブル、プロジェクト構成、ツール一覧など、変更に関連するセクション
- コミットに README の更新を含める（別コミットにしない）

## 設計方針

### アーキテクチャ
- **omi_whisper_transcribe は単体でも動く設計を維持**する。gpu-shared 固有のロジックを入れない
- gpu-shared は include で継承するだけ、追加変更は最小限
- Viewer は Bridge の管理 API を介してデータにアクセスする (直接の Python import はしない)

### Python 環境
- コンテナは **Python 3.10** (nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04)
- `asyncio.timeout` は使えない → `asyncio.wait_for` を使う
- `CancelledError` は `BaseException` のサブクラス → `except Exception` では捕まらない

### VRAM 管理
- GPU を使うバッチ処理 (diarize, speaker tag, etc.) は必ず gpu_lock を取得してから実行
- バッチ完了後は明示的にモデルをアンロードして VRAM を解放する
- Whisper セッション中にバッチ処理が走らないよう排他制御する

### 設定ファイル
- ランタイム設定は `config/` ディレクトリに JSON 形式で保存
- Bridge と Viewer の両方からマウントされるため、両者で共有される
