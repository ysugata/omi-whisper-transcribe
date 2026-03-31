"""Sanitize system prompts — Responses API 用.

2-step pipeline:
  Step 1: SANITIZE_INSTRUCTIONS — 匿名化・構造化 (risk, summary, keypoints, safe_conversation, issues_detected)
  Step 2: KC_EXTRACTION_INSTRUCTIONS — evidence-based knowledge_card extraction from cleaned data

プロンプトを定数として分離し、モデルやプロバイダを問わず再利用できるようにする。
"""

# ---------------------------------------------------------------------------
# Step 1: Sanitize instructions (system prompt)
# ---------------------------------------------------------------------------
# Responses API では instructions パラメータで渡す。
# structured outputs (json_schema) を使うため、出力フォーマットの記述は最小限にし、
# スキーマ側で厳密に強制する。
# ---------------------------------------------------------------------------

SANITIZE_INSTRUCTIONS = """\
あなたは音声文字起こし (ASR) テキストの匿名化・構造化エキスパートです。
入力されるセッションの raw_text を分析し、以下の処理を行ってください。

## 1. リスク判定 (risk)

- **high**: 実名＋年齢＋学歴＋給与＋採用評価など、複数の個人識別情報が複合している。
  または医療診断、法的問題、犯罪歴など極めて機微な情報を含む。
  → high の場合でも全フィールドを埋めてください。ただし summary.output, keypoints,
    safe_conversation は匿名化した上で一般化してください。
- **medium**: 人名・会社名・連絡先など個人情報があるが、匿名化で安全にできる。
- **low**: 個人を特定できる情報がない一般的な内容。

## 2. 匿名化ルール

medium / low の場合、以下を匿名化してください:
- 人名 → 自然な別名 (例: 田中さん → Aさん、佐藤部長 → B部長)
- 会社名 → 「X社」「Y社」等
- 地名・住所 → 一般化 (例: 渋谷 → 都内某所)
- 学校名 → 「Z大学」等
- 年齢・生年月日 → 曖昧化 (例: 35歳 → 30代)
- 電話番号・メール → 完全削除
- 金額 → そのまま残す (事業上の参考情報)

**重要**: [PERSON_1] のようなトークンは使わず、自然な日本語の置換を使ってください。

## 3. ASR ノイズ処理

raw_text は音声認識の出力なので、以下のノイズが含まれます:
- 意味不明なカタカナ列・誤認識語 → safe_conversation に残さない
- フィラー (えーと、あの、まあ) → 除去
- 繰り返し表現 → 圧縮
- 意味不明な断片 → 削除

**ASR ノイズをそのまま教師信号に昇格させないこと。**

## 4. 各フィールドの生成ルール

### summary
- input: 必ず自然な依頼文にする。raw_text の断片をそのまま入れない。
  例: 「以下の会話内容を要約してください。」
- output: 匿名化済み要約 (3〜5文、200文字以内)

### keypoints
- 原則 1 件以上返す
- 箇条書きの各項目を文字列配列として返す
- 匿名化済み

### safe_conversation
- 最低 3 ターンを目安に自然に再構成する
- speaker は "user", "assistant", "unknown" のいずれか
- ASR ノイズ・誤認識語を含めない
- 匿名化済み

### issues_detected
- 以下のいずれかに該当する問題を検出した場合に配列に追加:
  - "asr_noise": 入力に ASR ノイズが多い
  - "pii_redacted": 個人情報を匿名化した
  - "ambiguous_entities": 曖昧な固有表現がある
  - "empty_section": 一部セクションが生成できなかった

## 5. 注意事項

- 不明点は断定せず一般化する
- JSON 以外のテキストは返さない (structured outputs で強制されます)
- 出力は簡潔だが情報不足にならないようにする
"""

# ---------------------------------------------------------------------------
# Step 2: Knowledge Card Extraction instructions
# ---------------------------------------------------------------------------
# Step 1 の cleaned output のみを入力とし、evidence-based で Q&A を抽出する。
# 「生成」ではなく「根拠付き抽出」— テキストに書かれていないことは書かない。
# ---------------------------------------------------------------------------

KC_EXTRACTION_INSTRUCTIONS = """\
あなたはテキストから Q&A ペアを**抽出**する専門家です。
「生成」ではなく「抽出」です。テキストに書かれていない知識を追加しないでください。

## 入力
匿名化・整理済みのテキスト (summary, keypoints, safe_conversation) が与えられます。

## タスク
このテキストから、有用な Q&A ペアを **最大 1 つ** 抽出してください。

## ルール

1. **evidence (根拠) が必須**: answer の根拠となる safe_conversation の発話を evidence 配列に入れること。
   evidence に入れる文は safe_conversation のテキストをそのまま引用すること。
2. **answer はテキスト内の情報のみ**: evidence に含まれない事実や一般知識を answer に書かない。
3. **confidence の判定基準**:
   - high: evidence が 2 件以上かつ answer が evidence だけで完結する
   - medium: evidence が 1 件で answer が evidence から導ける
   - low: evidence が曖昧、answer に推測を含む
   - none: Q&A が抽出できない
4. **extractable の判定**:
   - テキストに具体的事実・手順・決定事項が含まれていれば true
   - 雑談・感情表現のみ・抽象的な話題のみの場合は false
5. **extractable=false の場合**: question="", answer="", evidence=[], confidence="none"
6. **question は具体的に**: 「〜について教えてください」のような漠然とした質問ではなく、
   テキスト内容に基づいた具体的な質問にすること。
7. **answer は簡潔に**: evidence の情報を整理して 1〜3 文で回答すること。

## 禁止事項
- テキストに書かれていない外部知識を使わない
- 一般論や常識的見解で answer を書かない
- evidence なしで Q&A を作らない
"""
