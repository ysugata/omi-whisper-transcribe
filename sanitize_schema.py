"""Sanitize JSON Schema — OpenAI Structured Outputs 用.

strict: true で安定動作するように設計。
- 全オブジェクトに additionalProperties: false
- 全プロパティを required に含める
- knowledge_card は anyOf で null 許容
- enum で揺れを防止
"""

import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON Schema definition for OpenAI Responses API (strict mode)
# ---------------------------------------------------------------------------
# この schema は text={"format": {"type": "json_schema", ...}} で使う。
# strict: true 制約:
#   - additionalProperties: false 必須
#   - required に全プロパティ列挙必須
#   - anyOf は {"type": "null"} との組み合わせで nullable 表現
# ---------------------------------------------------------------------------

SANITIZE_SCHEMA = {
    "type": "json_schema",
    "name": "sanitize_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "risk": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
            "summary": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["input", "output"],
                "additionalProperties": False,
            },
            "keypoints": {
                "type": "array",
                "items": {"type": "string"},
            },
            "safe_conversation": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {
                            "type": "string",
                            "enum": ["user", "assistant", "unknown"],
                        },
                        "text": {"type": "string"},
                    },
                    "required": ["speaker", "text"],
                    "additionalProperties": False,
                },
            },
            "knowledge_card": {
                # null 許容: 作れない場合は null を返す
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                        },
                        "required": ["question", "answer"],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ],
            },
            "issues_detected": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "asr_noise",
                        "pii_redacted",
                        "ambiguous_entities",
                        "empty_section",
                    ],
                },
            },
        },
        "required": [
            "risk",
            "summary",
            "keypoints",
            "safe_conversation",
            "knowledge_card",
            "issues_detected",
        ],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Validation & Quality Gate
# ---------------------------------------------------------------------------

# カタカナ 3 文字以上の意味不明パターン (簡易判定)
_GIBBERISH_KANA = re.compile(r"[ァ-ヶー]{6,}")

# PII らしきパターン (簡易判定)
_PII_PATTERN = re.compile(
    r"\d{2,4}[-−]\d{2,4}[-−]\d{2,4}"  # 電話番号
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"  # メール
    r"|\d{1,3}歳"  # 年齢
)


def validate_schema(data: dict) -> bool:
    """出力が最低限のスキーマ構造を満たしているか検証する.

    strict mode なので通常は不要だが、念のため defensive に検証。
    """
    if not isinstance(data, dict):
        return False

    # risk
    if data.get("risk") not in ("low", "medium", "high"):
        return False

    # summary
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return False
    if not summary.get("input") or not summary.get("output"):
        return False

    # keypoints
    if not isinstance(data.get("keypoints"), list):
        return False

    # safe_conversation
    conv = data.get("safe_conversation")
    if not isinstance(conv, list):
        return False

    # knowledge_card (nullable)
    kc = data.get("knowledge_card")
    if kc is not None:
        if not isinstance(kc, dict):
            return False
        if "question" not in kc or "answer" not in kc:
            return False

    # issues_detected
    if not isinstance(data.get("issues_detected"), list):
        return False

    return True


def run_quality_gates(data: dict) -> list[str]:
    """品質ゲートを実行し、warnings リストを返す.

    以下をチェック:
    - summary.input が不自然
    - keypoints が空
    - safe_conversation が 2 ターン未満
    - safe_conversation に意味不明語が多い
    - risk=low なのに PII らしき内容が多い
    - knowledge_card が作れない
    """
    warnings: list[str] = []

    # 1. summary.input の自然さチェック
    summary_input = data.get("summary", {}).get("input", "")
    if len(summary_input) < 5 or summary_input == data.get("summary", {}).get("output", ""):
        warnings.append("summary_input_unnatural")

    # 2. keypoints が空
    keypoints = data.get("keypoints", [])
    if not keypoints:
        warnings.append("keypoints_empty")

    # 3. safe_conversation が 2 ターン未満
    conv = data.get("safe_conversation", [])
    if len(conv) < 2:
        warnings.append("conversation_too_short")

    # 4. safe_conversation に意味不明語 (カタカナ 6 文字以上の連続)
    gibberish_count = 0
    for turn in conv:
        text = turn.get("text", "")
        if _GIBBERISH_KANA.search(text):
            gibberish_count += 1
    if gibberish_count > 0:
        warnings.append(f"conversation_gibberish_detected:{gibberish_count}")

    # 5. risk=low なのに PII パターンが多い
    if data.get("risk") == "low":
        pii_hits = 0
        all_text = (
            data.get("summary", {}).get("output", "")
            + " ".join(data.get("keypoints", []))
            + " ".join(t.get("text", "") for t in conv)
        )
        pii_hits = len(_PII_PATTERN.findall(all_text))
        if pii_hits >= 2:
            warnings.append(f"risk_low_but_pii_found:{pii_hits}")

    # 6. knowledge_card が作れない
    if data.get("knowledge_card") is None:
        warnings.append("knowledge_card_null")

    return warnings
