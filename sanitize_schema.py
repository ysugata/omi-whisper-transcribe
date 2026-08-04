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

    # 6. knowledge_cards の品質検証 (Step 2 で追加された場合)
    cards = data.get("knowledge_cards")
    if isinstance(cards, list):
        for i, kc in enumerate(cards):
            if isinstance(kc, dict):
                for w in validate_knowledge_card(kc):
                    warnings.append(f"kc[{i}]:{w}")
    else:
        # Legacy single-card fallback
        kc = data.get("knowledge_card")
        if kc is not None and isinstance(kc, dict):
            warnings.extend(validate_knowledge_card(kc))

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
            "extractable": {"type": "boolean"},
            "cards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["none", "low", "medium", "high"],
                        },
                        "preconditions": {"type": "string"},
                        "failure_signals": {"type": "string"},
                    },
                    "required": [
                        "question", "answer", "evidence", "confidence",
                        "preconditions", "failure_signals",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["extractable", "cards"],
        "additionalProperties": False,
    },
}

# Minimum keypoints to attempt KC extraction
KC_MIN_KEYPOINTS = 3

# Maximum KCs per session (must match prompt instruction)
KC_MAX_PER_SESSION = 5

# Vague/generic question patterns to reject during quality gate
_VAGUE_QUESTION_PATTERNS = [
    r"について教えて",
    r"どう思いますか",
    r"要点を(再)?確認",
    r"主なトピックは何",
    r"どのような(内容|話|議論)",
    r"何について(話|議論)",
    r"全体の(内容|流れ|概要)",
]
_VAGUE_QUESTION_RE = re.compile("|".join(_VAGUE_QUESTION_PATTERNS))


def validate_knowledge_card(kc: dict) -> list[str]:
    """Per-card quality gate — returns list of warning strings.

    Used for logging/metrics. The hard accept/reject decision is in
    should_keep_knowledge_card().
    """
    warnings: list[str] = []

    evidence = kc.get("evidence", [])
    if not evidence:
        warnings.append("kc_no_evidence")
        return warnings

    confidence = kc.get("confidence", "none")
    if confidence in ("none", "low"):
        warnings.append(f"kc_weak_confidence:{confidence}")

    question = kc.get("question", "")
    if len(question) < 15:
        warnings.append("kc_question_too_short")

    answer = kc.get("answer", "")
    if len(answer) < 20:
        warnings.append("kc_answer_too_short")

    if _VAGUE_QUESTION_RE.search(question):
        warnings.append("kc_question_vague")

    return warnings


def should_keep_knowledge_card(kc: dict) -> bool:
    """Hard quality gate — decide whether to keep a single KC in the output.

    Rejects:
        - no evidence
        - confidence none/low
        - question < 15 chars
        - answer < 20 chars
        - vague question pattern match
    """
    if not kc.get("evidence"):
        return False

    if kc.get("confidence", "none") in ("none", "low"):
        return False

    question = kc.get("question", "")
    if len(question) < 15:
        return False

    answer = kc.get("answer", "")
    if len(answer) < 20:
        return False

    if _VAGUE_QUESTION_RE.search(question):
        return False

    return True


def filter_knowledge_cards(data: dict) -> list[dict]:
    """Extract and filter the `cards` array from a KC extraction response.

    Supports both the new multi-card format (`cards: [...]`) and the
    legacy single-card format (flat `question/answer/evidence/confidence`).
    Returns cards that pass should_keep_knowledge_card(), deduplicated
    by exact-question-match within the session.
    """
    if not data.get("extractable", False):
        return []

    candidates: list[dict] = []

    cards = data.get("cards")
    if isinstance(cards, list):
        candidates = [c for c in cards if isinstance(c, dict)]
    elif "question" in data and "answer" in data:
        # Legacy single-card shape
        candidates = [{
            "question": data.get("question", ""),
            "answer": data.get("answer", ""),
            "evidence": data.get("evidence", []),
            "confidence": data.get("confidence", "none"),
        }]

    # Apply hard gate + exact-question dedup
    kept: list[dict] = []
    seen_questions: set[str] = set()
    for c in candidates:
        if not should_keep_knowledge_card(c):
            continue
        q = c.get("question", "").strip()
        if q in seen_questions:
            continue
        seen_questions.add(q)
        kept.append(c)
        if len(kept) >= KC_MAX_PER_SESSION:
            break

    return kept
