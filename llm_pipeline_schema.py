"""JSON Schema definitions for the 3-stage LLM pipeline.

Provider-agnostic object schemas + OpenAI-compatible response_format wrappers.
"""

# ---------------------------------------------------------------------------
# Helper: wrap pure JSON Schema into OpenAI-compatible response_format
# ---------------------------------------------------------------------------

def wrap_openai_schema(name: str, schema: dict) -> dict:
    """Build OpenAI-compatible response_format from a raw JSON Schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


# ---------------------------------------------------------------------------
# Stage 3-A: Term candidate extraction (no text modification)
# ---------------------------------------------------------------------------

CANDIDATE_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "inferred_domain": {"type": "string"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "integer"},
                    "text_span": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "proper_noun",
                            "technical_term",
                            "broken_katakana",
                            "misrecognized",
                        ],
                    },
                    "suggested_form": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "line",
                    "text_span",
                    "category",
                    "suggested_form",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["inferred_domain", "candidates"],
    "additionalProperties": False,
}

CANDIDATE_SCHEMA = wrap_openai_schema("term_candidates", CANDIDATE_OBJECT_SCHEMA)


# ---------------------------------------------------------------------------
# Stage 3-B: Notation correction (diff-only corrections array)
# ---------------------------------------------------------------------------

CORRECTION_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "integer"},
                    "original_snippet": {"type": "string"},
                    "corrected": {"type": "string"},
                    "reason": {"type": "string"},
                    "source": {
                        "type": "string",
                        "enum": [
                            "glossary",
                            "candidate_3a",
                            "context",
                            "dictionary",
                        ],
                    },
                },
                "required": [
                    "line",
                    "original_snippet",
                    "corrected",
                    "reason",
                    "source",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["corrections"],
    "additionalProperties": False,
}

CORRECTION_SCHEMA = wrap_openai_schema("notation_corrections", CORRECTION_OBJECT_SCHEMA)


# ---------------------------------------------------------------------------
# Stage 3-C: Quality audit
# ---------------------------------------------------------------------------

AUDIT_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "integer"},
                    "text_span": {"type": "string"},
                    "issue_type": {
                        "type": "string",
                        "enum": [
                            "uncorrected_term",
                            "meaningless",
                            "broken_katakana",
                            "trailing_fragment",
                        ],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "suggestion": {"type": "string"},
                },
                "required": [
                    "line",
                    "text_span",
                    "issue_type",
                    "severity",
                    "suggestion",
                ],
                "additionalProperties": False,
            },
        },
        "pass": {"type": "boolean"},
    },
    "required": ["items", "pass"],
    "additionalProperties": False,
}

AUDIT_SCHEMA = wrap_openai_schema("quality_audit", AUDIT_OBJECT_SCHEMA)


# ---------------------------------------------------------------------------
# Session tagging (used by BatchCorrector for Gemini Batch API)
# ---------------------------------------------------------------------------

TAGGING_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "conversation_type": {"type": "string"},
        "outcomes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "domains": {
            "type": "array",
            "items": {"type": "string"},
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
        },
        "summary": {"type": "string"},
        "decision_memo": {"type": "string"},
        "quality": {
            "type": "string",
            "enum": ["clean", "partial", "noisy", "fragmented"],
        },
        "importance": {"type": "string"},
    },
    "required": [
        "conversation_type", "outcomes", "domains",
        "entities", "summary", "decision_memo",
        "quality", "importance",
    ],
    "additionalProperties": False,
}

TAGGING_SCHEMA = wrap_openai_schema("session_tagging", TAGGING_OBJECT_SCHEMA)
