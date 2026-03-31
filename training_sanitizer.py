"""Training data sanitizer module using Gemini API.

Evaluates risk level of each session and generates anonymized,
safe training records for LLM fine-tuning.
Uses OpenAI-compatible API (same as SessionTagger / LLMCorrector).
"""

import json
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """あなたは音声文字起こしテキストの匿名化・変換エキスパートです。
入力されるセッションテキスト（タグ情報＋発話セグメント）を分析し、以下の処理を行ってください。

## 1. リスク判定

入力テキストのプライバシーリスクを判定してください:

- **high**: 実名＋年齢＋学歴＋採用評価＋給与など、複数の個人識別情報が組み合わさっている。
  または医療診断、法的問題、犯罪歴など極めて機微な情報を含む。
  → recordsは空配列を返す。
- **medium**: 人名・会社名・連絡先など個人情報があるが、一般化・匿名化で安全にできる。
- **low**: 一般的な内容で個人を特定できる情報がない。

## 2. 匿名化ルール

medium/low の場合、以下を匿名化してください:
- 人名 → 自然な別名に置換（例: 田中さん → Aさん、佐藤部長 → B部長）
- 会社名 → 「X社」「Y社」等に置換
- 地名・住所 → 一般化（例: 渋谷 → 都内某所）
- 学校名 → 「Z大学」等に置換
- 年齢・生年月日 → 削除または曖昧化（例: 35歳 → 30代）
- 電話番号・メール → 完全削除
- 金額 → そのまま残す（事業上の参考情報のため）

**重要**: [PERSON_1] のようなトークンは使わないでください。自然な日本語の置換を使用してください。

## 3. テキスト品質改善

匿名化と同時に以下を行ってください:
- ASR（音声認識）エラーの修正
- フィラー（えーと、あの、まあ）の除去
- 繰り返し表現の圧縮
- 意味不明な断片の削除

## 4. 変換タイプ

以下の4タイプのレコードを生成してください（内容に適したもののみ）:

### A. summary — セッション要約
- input: 「以下の会話を要約してください。」
- output: 匿名化された要約文（3-5文）

### B. keypoints — 要点抽出
- input: 「以下の会話から要点を抽出してください。」
- output: 匿名化された要点リスト（箇条書き）

### C. safe_conversation — 安全な会話再構成
- messages: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}] 形式
- 匿名化された会話を自然な対話形式に再構成

### D. knowledge_card — ナレッジカード
- input: トピックに関する質問文
- output: 会話から得られる知識・ノウハウの回答

**全タイプ生成する必要はありません。内容に適したもののみ生成してください。**
短い雑談セッションなら summary のみで十分です。

## 出力フォーマット

以下の JSON のみを返してください（他のテキストは不要）:

```json
{
  "risk_level": "low|medium|high",
  "records": [
    {"type": "summary", "input": "...", "output": "..."},
    {"type": "keypoints", "input": "...", "output": "..."},
    {"type": "safe_conversation", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]},
    {"type": "knowledge_card", "input": "...", "output": "..."}
  ]
}
```

high の場合: `{"risk_level": "high", "records": []}`

**重要**: 各レコードの output は200文字以内に収めてください。簡潔さを優先してください。"""

_VALID_RISK_LEVELS = {"low", "medium", "high"}
_VALID_RECORD_TYPES = {"summary", "keypoints", "safe_conversation", "knowledge_card"}

# Limits for input truncation
_MAX_SEGMENTS = 80
_MAX_CHARS = 4000


class TrainingSanitizer:
    """Sanitizes session transcripts for safe training data export."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 8192,
        timeout: int = 30,
    ) -> None:
        self._enabled = bool(api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # OpenAI GPT-5+ models require max_completion_tokens instead of max_tokens
        self._use_max_completion_tokens = model.startswith("gpt-")
        self._client = None

        if self._enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
                log.info(f"TrainingSanitizer initialized: model={model}")
            except Exception as e:
                log.error(f"TrainingSanitizer init failed: {e}")
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _build_input(self, transcript: dict) -> str:
        """Build input text from transcript data (tags + segments)."""
        parts = []

        # Include session tags if available
        session = transcript.get("session", {})
        tags = session.get("tags", {})
        if tags:
            tag_lines = []
            for key in ("summary", "topics", "entities", "activity",
                        "business_domain", "importance",
                        "decision_summary", "action_items"):
                val = tags.get(key)
                if val:
                    if isinstance(val, list):
                        val = ", ".join(str(v) for v in val)
                    tag_lines.append(f"{key}: {val}")
            if tag_lines:
                parts.append("[セッション情報]\n" + "\n".join(tag_lines))

        # Include segments
        segments = transcript.get("segments", [])
        if not segments:
            return ""

        seg_lines = []
        total_chars = 0
        for seg in segments[:_MAX_SEGMENTS]:
            text = seg.get("text", "").strip()
            if not text:
                continue
            speaker = seg.get("person_name") or seg.get("speaker", "")
            line = f"[{speaker}] {text}" if speaker else text
            total_chars += len(line)
            if total_chars > _MAX_CHARS:
                break
            seg_lines.append(line)

        if seg_lines:
            parts.append("[発話]\n" + "\n".join(seg_lines))

        return "\n\n".join(parts)

    def _parse_response(self, raw: str) -> Optional[dict]:
        """Parse JSON response from LLM."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(f"TrainingSanitizer: invalid JSON: {raw[:200]}")
            return None

        if not isinstance(data, dict):
            log.warning("TrainingSanitizer: response is not a dict")
            return None

        return data

    def _validate_response(self, data: dict) -> Optional[dict]:
        """Validate and normalize the sanitizer response."""
        risk_level = str(data.get("risk_level", "")).lower().strip()
        if risk_level not in _VALID_RISK_LEVELS:
            log.warning(f"TrainingSanitizer: invalid risk_level: {risk_level}")
            return None

        # high risk → force empty records
        if risk_level == "high":
            return {"risk_level": "high", "records": []}

        records = data.get("records", [])
        if not isinstance(records, list):
            log.warning("TrainingSanitizer: records is not a list")
            return None

        valid_records = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec_type = str(rec.get("type", "")).strip()
            if rec_type not in _VALID_RECORD_TYPES:
                continue

            if rec_type == "safe_conversation":
                messages = rec.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    continue
                # Validate message structure
                ok = True
                for msg in messages:
                    if not isinstance(msg, dict):
                        ok = False
                        break
                    if "role" not in msg or "content" not in msg:
                        ok = False
                        break
                if not ok:
                    continue
                valid_records.append({
                    "type": rec_type,
                    "messages": messages,
                })
            else:
                inp = rec.get("input", "")
                out = rec.get("output", "")
                if not inp or not out:
                    continue
                valid_records.append({
                    "type": rec_type,
                    "input": str(inp),
                    "output": str(out),
                })

        return {"risk_level": risk_level, "records": valid_records}

    def sanitize_session(self, transcript: dict) -> Optional[dict]:
        """Sanitize a session transcript and return safe training records.

        Returns:
            {"risk_level": "low|medium|high", "records": [...]} on success,
            None on error.
        """
        if not self._enabled:
            return None

        input_text = self._build_input(transcript)
        if not input_text or len(input_text) < 20:
            log.debug("TrainingSanitizer: input text too short, skipping")
            return None

        try:
            kwargs = dict(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input_text},
                ],
                response_format={"type": "json_object"},
            )
            if self._use_max_completion_tokens:
                # GPT-5+ models: max_completion_tokens, no custom temperature
                kwargs["max_completion_tokens"] = self._max_tokens
            else:
                kwargs["max_tokens"] = self._max_tokens
                kwargs["temperature"] = self._temperature
            response = self._client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content
            log.debug(f"TrainingSanitizer raw response: {raw[:300]}")

            parsed = self._parse_response(raw)
            if parsed is None:
                return None

            result = self._validate_response(parsed)
            if result is None:
                return None

            log.info(
                f"TrainingSanitizer: risk={result['risk_level']}, "
                f"records={len(result['records'])}"
            )
            return result

        except Exception as e:
            log.warning(f"TrainingSanitizer API error: {e}")
            return None
