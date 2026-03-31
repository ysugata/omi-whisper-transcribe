"""Sanitize JSON Schema — OpenAI Structured Outputs 用.

strict: true で安定動作するように設計。
- 全オブジェクトに additionalProperties: false
- 全プロパティを required に含める
- enum で揺れを防止

2-step pipeline:
  Step 1 (SANITIZE_SCHEMA): risk, summary, keypoints, safe_conversation, issues_detected
  Step 2 (KC_EXTRACTION_SCHEMA): evidence-based knowledge_card extraction from cleaned data
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

    # 6. knowledge_card の品質検証 (Step 2 で追加された場合)
    kc = data.get("knowledge_card")
    if kc is not None and isinstance(kc, dict):
        kc_warnings = validate_knowledge_card(kc)
        warnings.extend(kc_warnings)

    return warnings


# ---------------------------------------------------------------------------
# Step 2: Knowledge Card Extraction Schema
# ---------------------------------------------------------------------------
# Step 1 の cleaned output (summary, keypoints, safe_conversation) のみを入力とし、
# evidence-based で knowledge_card を抽出する。
# extractable=false の場合は question/answer/evidence は空文字列/空配列。
# ---------------------------------------------------------------------------

KC_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "name": "knowledge_card_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "extractable": {
                "type": "boolean",
            },
            "question": {
                "type": "string",
            },
            "answer": {
                "type": "string",
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
            },
            "confidence": {
                "type": "string",
                "enum": ["none", "low", "medium", "high"],
            },
        },
        "required": ["extractable", "question", "answer", "evidence", "confidence"],
        "additionalProperties": False,
    },
}

# Minimum keypoints to attempt KC extraction
KC_MIN_KEYPOINTS = 3


def validate_knowledge_card(kc: dict) -> list[str]:
    """Knowledge card の品質ゲートを実行し warnings を返す.

    Quality gates:
    1. evidence が 1 件以上あること
    2. confidence が low/none でないこと
    3. question が抽象的すぎないこと (10 文字以上)
    4. answer が一般論でないこと (15 文字以上)
    """
    warnings: list[str] = []

    if not kc.get("extractable", False):
        return warnings  # extractable=false なら検証不要

    # 1. evidence >= 1
    evidence = kc.get("evidence", [])
    if not evidence:
        warnings.append("kc_no_evidence")
        return warnings  # evidence なしは致命的 → 他の検証は不要

    # 2. confidence != low/none
    confidence = kc.get("confidence", "none")
    if confidence in ("none", "low"):
        warnings.append(f"kc_weak_confidence:{confidence}")

    # 3. question の長さ
    question = kc.get("question", "")
    if len(question) < 10:
        warnings.append("kc_question_too_short")

    # 4. answer の長さ
    answer = kc.get("answer", "")
    if len(answer) < 15:
        warnings.append("kc_answer_too_short")

    return warnings


def should_keep_knowledge_card(kc: dict) -> bool:
    """Knowledge card を最終出力に含めるべきか判定する.

    False を返す場合、knowledge_card は null にフォールバックする。
    """
    if not kc.get("extractable", False):
        return False

    # evidence must exist
    if not kc.get("evidence"):
        return False

    # confidence must be medium or high
    if kc.get("confidence", "none") in ("none", "low"):
        return False

    # question/answer must have substance
    if len(kc.get("question", "")) < 10:
        return False
    if len(kc.get("answer", "")) < 15:
        return False

    return True
