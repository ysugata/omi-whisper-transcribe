"""Training data export — convert transcript sessions to fine-tuning formats.

Since the V2 rewrite, chat-style formats (openai, conversation) source their
data from sanitized JSONL files (config/safe_training_*.jsonl) rather than
raw transcript segments. The raw speaker-role-mapping approach produced
unusable noise; the sanitized pipeline extracts evidence-based Q&A pairs
which are proper instruction-tuning material.

Supported sanitizer output formats:
- ResponsesSanitizer: {session_id, risk, summary, keypoints, safe_conversation,
  knowledge_card, issues_detected}
- TrainingSanitizer: separate records per type
  {type: summary|keypoints|safe_conversation|knowledge_card, session_id,
   risk_level, input, output} (or messages for safe_conversation)
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sanitized JSONL index
# ---------------------------------------------------------------------------


def _parse_bullet_list(text: str) -> list[str]:
    """Parse a bullet-list string into a list of items.

    Accepts "- item" / "・ item" / "* item" bullets. Falls back to newline-split.
    """
    items = []
    for line in (text or "").split("\n"):
        s = line.strip()
        if not s:
            continue
        s = s.lstrip("-・*•●◯ \t").strip()
        if s:
            items.append(s)
    return items


class SanitizedIndex:
    """In-memory index of sanitized records, keyed by session_id.

    Loads all config/safe_training_*.jsonl files and aggregates records
    across both ResponsesSanitizer (1 record per session) and
    TrainingSanitizer (multiple records per session) output formats.
    """

    def __init__(self, config_dir: Path | str | None = None) -> None:
        self._by_session: dict[str, dict[str, Any]] = {}
        self._source_files: list[str] = []
        if config_dir:
            self.load(Path(config_dir))

    def load(self, config_dir: Path) -> None:
        """Load all safe_training_*.jsonl files from the given directory."""
        if not config_dir.exists():
            logger.debug("SanitizedIndex: config dir does not exist: %s", config_dir)
            return

        files = sorted(config_dir.glob("safe_training_*.jsonl"))
        for fp in files:
            try:
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self._merge_record(rec)
            except OSError as e:
                logger.warning("SanitizedIndex: failed to read %s: %s", fp, e)
                continue
            self._source_files.append(fp.name)
        logger.info(
            "SanitizedIndex loaded: %d sessions from %d files",
            len(self._by_session), len(self._source_files),
        )

    def _bucket(self, session_id: str) -> dict[str, Any]:
        return self._by_session.setdefault(session_id, {
            "session_id": session_id,
            "summary_input": None,
            "summary_output": None,
            "keypoints": None,
            "safe_conversation": None,
            "knowledge_cards": [],
            "risk": None,
        })

    def _merge_record(self, rec: dict) -> None:
        sid = rec.get("session_id")
        if not sid:
            return
        bucket = self._bucket(sid)

        # ResponsesSanitizer format: single record per session with all fields.
        # Detection: has both 'risk' and 'safe_conversation' keys.
        if "risk" in rec and "safe_conversation" in rec:
            bucket["risk"] = bucket["risk"] or rec.get("risk")
            summary = rec.get("summary") or {}
            if isinstance(summary, dict):
                bucket["summary_input"] = (
                    bucket["summary_input"] or summary.get("input")
                )
                bucket["summary_output"] = (
                    bucket["summary_output"] or summary.get("output")
                )
            kps = rec.get("keypoints")
            if isinstance(kps, list) and kps and not bucket["keypoints"]:
                bucket["keypoints"] = [str(k) for k in kps if k]
            conv = rec.get("safe_conversation")
            if isinstance(conv, list) and conv and not bucket["safe_conversation"]:
                bucket["safe_conversation"] = [
                    {
                        "speaker": (t.get("speaker") or t.get("role") or "unknown"),
                        "text": (t.get("text") or t.get("content") or ""),
                    }
                    for t in conv
                    if isinstance(t, dict)
                ]
            # New multi-card field takes precedence over the legacy single-card
            seen_questions = {c["question"] for c in bucket["knowledge_cards"]}
            kcs_multi = rec.get("knowledge_cards")
            if isinstance(kcs_multi, list):
                for kc in kcs_multi:
                    if not isinstance(kc, dict):
                        continue
                    q = str(kc.get("question", "")).strip()
                    a = str(kc.get("answer", "")).strip()
                    if not q or not a or q in seen_questions:
                        continue
                    seen_questions.add(q)
                    bucket["knowledge_cards"].append({
                        "question": q,
                        "answer": a,
                        "evidence": kc.get("evidence") or [],
                        "confidence": kc.get("confidence") or "medium",
                    })
            # Legacy single-card field (back-compat with pre-multi-KC runs)
            kc = rec.get("knowledge_card")
            if isinstance(kc, dict):
                q = str(kc.get("question", "")).strip()
                a = str(kc.get("answer", "")).strip()
                if q and a and q not in seen_questions:
                    seen_questions.add(q)
                    bucket["knowledge_cards"].append({
                        "question": q,
                        "answer": a,
                        "evidence": kc.get("evidence") or [],
                        "confidence": kc.get("confidence") or "medium",
                    })
            return

        # TrainingSanitizer format: multiple records per session, distinguished
        # by 'type' field. risk_level is per-record.
        rtype = rec.get("type")
        if not rtype:
            return
        bucket["risk"] = bucket["risk"] or rec.get("risk_level")

        if rtype == "summary":
            if not bucket["summary_input"]:
                bucket["summary_input"] = rec.get("input") or ""
            if not bucket["summary_output"]:
                bucket["summary_output"] = rec.get("output") or ""
        elif rtype == "keypoints":
            out = rec.get("output")
            if bucket["keypoints"] is None:
                if isinstance(out, list):
                    bucket["keypoints"] = [str(x) for x in out if x]
                elif isinstance(out, str):
                    bucket["keypoints"] = _parse_bullet_list(out)
        elif rtype == "safe_conversation":
            if bucket["safe_conversation"] is None:
                msgs = rec.get("messages") or []
                if isinstance(msgs, list):
                    bucket["safe_conversation"] = [
                        {
                            "speaker": (m.get("role") or m.get("speaker") or "unknown"),
                            "text": (m.get("content") or m.get("text") or ""),
                        }
                        for m in msgs
                        if isinstance(m, dict)
                    ]
        elif rtype == "knowledge_card":
            q = (rec.get("input") or "").strip()
            a = (rec.get("output") or "").strip()
            # Dedup by question: SanitizedIndex loads every safe_training_*
            # file ever generated, so overlapping runs re-emit the same KCs.
            seen = {c["question"] for c in bucket["knowledge_cards"]}
            if q and a and q not in seen:
                bucket["knowledge_cards"].append({
                    "question": q,
                    "answer": a,
                    "evidence": [],
                    "confidence": "medium",
                })

    def get(self, session_id: str) -> dict | None:
        return self._by_session.get(session_id)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._by_session

    def __len__(self) -> int:
        return len(self._by_session)


# ---------------------------------------------------------------------------
# SanitizedIndex cache
# ---------------------------------------------------------------------------

# Keyed by config dir; value is (directory signature, built index). The
# signature is the sorted (name, mtime_ns, size) of every safe_training_*
# file — cheap to compute per request, so the multi-MB JSONL parse only
# happens when a file actually changed.
_INDEX_CACHE: dict[str, tuple[tuple, "SanitizedIndex"]] = {}


def _sanitized_dir_signature(config_dir: Path) -> tuple:
    sig = []
    for f in sorted(config_dir.glob("safe_training_*.jsonl")):
        try:
            st = f.stat()
        except OSError:
            continue
        sig.append((f.name, st.st_mtime_ns, st.st_size))
    return tuple(sig)


def get_sanitized_index(config_dir: Path | str) -> "SanitizedIndex":
    """Return a (possibly cached) SanitizedIndex for config_dir.

    Synchronous — async callers should run it in an executor so a cache
    miss (full JSONL parse) does not block the event loop.
    """
    config_dir = Path(config_dir)
    key = str(config_dir)
    sig = _sanitized_dir_signature(config_dir)
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached[0] == sig:
        return cached[1]
    index = SanitizedIndex(config_dir)
    _INDEX_CACHE[key] = (sig, index)
    return index


# ---------------------------------------------------------------------------
# Shared helpers (unchanged)
# ---------------------------------------------------------------------------


def transcript_stub_from_row(row: dict) -> dict:
    """Rebuild a minimal transcript dict from a sessions DB row.

    Retention cleanup (MAX_RECORDING_DAYS) deletes transcript.json from disk
    while the DB row and sanitized safe_training data survive. The KC-sourced
    formats only need session metadata (tags) from the transcript, so this
    stub lets those sessions still be exported.
    """
    def _split(v: str | None) -> list[str]:
        return [s for s in (v or "").split(",") if s]

    tags: dict = {
        "conversation_type": row.get("tags_conversation_type"),
        "outcomes": _split(row.get("tags_outcomes")),
        "domains": _split(row.get("tags_domains")),
        "importance": row.get("tags_importance_v2"),
        "quality": row.get("tags_quality"),
        "summary": row.get("tags_summary"),
        "decision_memo": row.get("tags_decision_memo"),
        "action_items": _split(row.get("tags_action_items")),
        "topics": _split(row.get("tags_topics")),
        "entities": _split(row.get("tags_entities")),
        "business_domain": row.get("tags_business_domain"),
        "external_audio_ratio": row.get("tags_external_audio"),
    }
    tags = {k: v for k, v in tags.items() if v}
    return {
        "session": {
            "start": row.get("start_at"),
            "end": row.get("end_at"),
            "duration_seconds": row.get("duration_seconds"),
            "language": row.get("language"),
            "tags": tags,
        },
        "segments": [],
    }


def merge_turns(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments from the same speaker into turns."""
    if not segments:
        return []

    turns: list[dict] = []
    for seg in segments:
        speaker = seg.get("person_name") or seg.get("speaker") or "unknown"
        is_user = bool(seg.get("is_user", False))
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = seg.get("end", turns[-1]["end"])
        else:
            turns.append({
                "speaker": speaker,
                "is_user": is_user,
                "text": text,
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            })
    return turns


def _session_meta(transcript: dict) -> dict:
    """Extract session metadata from transcript."""
    session = transcript.get("session", {})
    tags = session.get("tags", {})
    return {
        "start": session.get("start"),
        "duration_seconds": session.get("duration_seconds"),
        "language": session.get("language"),
        "num_speakers": len(set(
            seg.get("person_name") or seg.get("speaker") or "unknown"
            for seg in transcript.get("segments", [])
            if (seg.get("text") or "").strip()
        )),
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# System prompts for task-synthesized records
# ---------------------------------------------------------------------------

_SYS_QA = (
    "あなたは経験豊富な事業経営者の思考を継承したアシスタントです。"
    "質問には実際の意思決定や経験に基づいて具体的に答えてください。"
)

_SYS_SUMMARIZE = (
    "次の会話を、要点を押さえた3〜5文で簡潔に要約してください。"
)

_SYS_KEYPOINTS = (
    "次の会話から、事業上の意思決定・気づき・学びを箇条書きで抽出してください。"
)


def _conversation_to_text(conv: list[dict]) -> str:
    """Serialize a safe_conversation list into a readable string."""
    lines = []
    for t in conv:
        speaker = t.get("speaker", "unknown")
        text = t.get("text", "").strip()
        if not text:
            continue
        # Normalize role names for serialization
        if speaker == "user":
            label = "経営者"
        elif speaker == "assistant":
            label = "相手"
        elif speaker == "unknown":
            label = "話者"
        else:
            label = speaker
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Format: comprehensive — unchanged, sources from raw transcript
# ---------------------------------------------------------------------------


def to_comprehensive(
    session_id: str,
    transcript: dict,
    speaker_mode: str = "conversation",
    include_monologue: bool = True,
    sanitized: dict | None = None,
) -> dict | None:
    """Convert a session to comprehensive JSONL format (1 dict per session).

    Preserves raw transcript + tags for analysis/debugging. Not used for
    fine-tuning directly.
    """
    segments = transcript.get("segments", [])
    if not segments:
        return None

    meta = _session_meta(transcript)
    tags = meta["tags"]
    turns = merge_turns(segments)

    if speaker_mode == "user_only":
        turns = [t for t in turns if t["is_user"]]

    if not turns:
        return None

    session_info = {
        "start": meta["start"],
        "duration_seconds": meta["duration_seconds"],
        "language": meta["language"],
        "num_speakers": meta["num_speakers"],
    }

    tags_info: dict[str, Any] = {}
    for key in (
        "conversation_type", "outcomes", "domains", "importance",
        "decision_memo", "topics", "activity", "mood", "quality",
        "entities", "summary", "intent", "source_type", "business_domain",
        "confidentiality", "decision_required", "decision_summary",
        "reasoning_summary", "action_items", "outcome", "next_action",
        "external_audio_ratio",
    ):
        val = tags.get(key)
        if val is not None and val != "" and val != []:
            tags_info[key] = val

    result: dict[str, Any] = {
        "session_id": session_id,
        "session": session_info,
        "tags": tags_info,
        "turns": [
            {
                "speaker": t["speaker"],
                "is_user": t["is_user"],
                "text": t["text"],
                "start": t["start"],
                "end": t["end"],
            }
            for t in turns
        ],
    }
    if sanitized is not None:
        result["sanitized"] = {
            "risk": sanitized.get("risk"),
            "summary": sanitized.get("summary_output"),
            "keypoints": sanitized.get("keypoints") or [],
            "knowledge_card_count": len(sanitized.get("knowledge_cards") or []),
        }
    return result


# ---------------------------------------------------------------------------
# Format: openai (messages) — sourced from sanitized KC + summary
# ---------------------------------------------------------------------------


def to_openai(
    session_id: str,
    transcript: dict,
    speaker_mode: str = "conversation",
    include_monologue: bool = False,
    sanitized: dict | None = None,
) -> list[dict]:
    """Emit {"messages": [...]} records derived from sanitized KC data.

    Produces one record per task:
    1. Q&A task from each knowledge_card
    2. Summarization task (if safe_conversation + summary are both present)

    Returns [] if no usable sanitized data is available (caller should skip).
    """
    if sanitized is None:
        return []

    out: list[dict] = []
    meta = _session_meta(transcript)
    tags = meta["tags"]
    domains = tags.get("domains") or []
    domain_str = ", ".join(domains) if isinstance(domains, list) else str(domains)

    system_qa = _SYS_QA
    if domain_str:
        system_qa = f"{_SYS_QA}\n事業領域: {domain_str}"

    # 1. Knowledge-card driven Q&A tasks
    for kc in sanitized.get("knowledge_cards") or []:
        q = (kc.get("question") or "").strip()
        a = (kc.get("answer") or "").strip()
        if len(q) < 10 or len(a) < 15:
            continue
        out.append({"messages": [
            {"role": "system", "content": system_qa},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]})

    # 2. Summarization task (needs both input conversation and output summary)
    conv = sanitized.get("safe_conversation")
    summary_out = (sanitized.get("summary_output") or "").strip()
    if conv and summary_out and len(summary_out) >= 20:
        conv_text = _conversation_to_text(conv)
        if len(conv_text) >= 80:
            out.append({"messages": [
                {"role": "system", "content": _SYS_SUMMARIZE},
                {"role": "user", "content": conv_text},
                {"role": "assistant", "content": summary_out},
            ]})

    return out


# ---------------------------------------------------------------------------
# Format: conversation (ShareGPT / LLaMA-Factory)
# ---------------------------------------------------------------------------


def to_conversation(
    session_id: str,
    transcript: dict,
    speaker_mode: str = "conversation",
    include_monologue: bool = False,
    sanitized: dict | None = None,
) -> list[dict]:
    """Emit ShareGPT-formatted records from sanitized KC data.

    Format per LLaMA-Factory spec:
        {"conversations": [{"from": "human", "value": ...},
                           {"from": "gpt", "value": ...}],
         "system": "..."}

    Note: id/metadata are intentionally NOT top-level (not in the spec).
    """
    if sanitized is None:
        return []

    out: list[dict] = []
    meta = _session_meta(transcript)
    tags = meta["tags"]
    domains = tags.get("domains") or []
    domain_str = ", ".join(domains) if isinstance(domains, list) else str(domains)

    system_qa = _SYS_QA
    if domain_str:
        system_qa = f"{_SYS_QA}\n事業領域: {domain_str}"

    for kc in sanitized.get("knowledge_cards") or []:
        q = (kc.get("question") or "").strip()
        a = (kc.get("answer") or "").strip()
        if len(q) < 10 or len(a) < 15:
            continue
        out.append({
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": a},
            ],
            "system": system_qa,
        })

    conv = sanitized.get("safe_conversation")
    summary_out = (sanitized.get("summary_output") or "").strip()
    if conv and summary_out and len(summary_out) >= 20:
        conv_text = _conversation_to_text(conv)
        if len(conv_text) >= 80:
            out.append({
                "conversations": [
                    {"from": "human", "value": conv_text},
                    {"from": "gpt", "value": summary_out},
                ],
                "system": _SYS_SUMMARIZE,
            })

    return out


# ---------------------------------------------------------------------------
# Format: knowledge — RAG-ready records with QA + evidence
# ---------------------------------------------------------------------------


def to_knowledge(
    session_id: str,
    transcript: dict,
    speaker_mode: str = "conversation",
    include_monologue: bool = True,
    sanitized: dict | None = None,
) -> dict | None:
    """Emit a knowledge record for RAG ingestion.

    Prefers sanitized summary + keypoints + KC pairs. Falls back to tags.summary
    from the raw transcript if no sanitized data exists.
    """
    meta = _session_meta(transcript)
    tags = meta["tags"]

    summary_text = ""
    keypoints: list[str] = []
    qa_pairs: list[dict] = []

    if sanitized is not None:
        summary_text = (sanitized.get("summary_output") or "").strip()
        keypoints = sanitized.get("keypoints") or []
        for kc in sanitized.get("knowledge_cards") or []:
            q = (kc.get("question") or "").strip()
            a = (kc.get("answer") or "").strip()
            if q and a:
                qa_pairs.append({
                    "question": q,
                    "answer": a,
                    "evidence": kc.get("evidence") or [],
                    "confidence": kc.get("confidence") or "medium",
                })

    if not summary_text:
        summary_text = (tags.get("summary") or "").strip()

    if not summary_text and not qa_pairs:
        return None

    domains = tags.get("domains", [])
    category = ", ".join(domains) if domains else tags.get("business_domain", "other")

    body_parts = []
    if summary_text:
        body_parts.append(summary_text)
    if keypoints:
        body_parts.append("\n".join(f"- {k}" for k in keypoints))

    return {
        "id": session_id,
        "title": summary_text[:60] if summary_text else f"Session {session_id}",
        "body": "\n\n".join(body_parts),
        "qa_pairs": qa_pairs,
        "category": category,
        "topics": tags.get("topics", []),
        "entities": tags.get("entities", []),
        "source_sessions": [session_id],
        "importance": tags.get("importance", "低"),
        "updated_at": transcript.get("session", {}).get("start"),
    }


# ---------------------------------------------------------------------------
# Format: decision_log — decisions + reasoning from tags + sanitized KC
# ---------------------------------------------------------------------------


def to_decision_log(
    session_id: str,
    transcript: dict,
    speaker_mode: str = "conversation",
    include_monologue: bool = True,
    sanitized: dict | None = None,
) -> dict | None:
    """Extract a decision log entry.

    Trigger conditions:
        - tags.decision_memo is non-empty, OR
        - tags.outcomes contains "意思決定あり", OR
        - sanitized data has a knowledge_card with confidence=high
    """
    meta = _session_meta(transcript)
    tags = meta["tags"]

    decision_memo = (tags.get("decision_memo") or "").strip()
    outcomes = tags.get("outcomes") or []
    has_decision_outcome = (
        isinstance(outcomes, list) and "意思決定あり" in outcomes
    )

    # Legacy V1 fields (may not exist)
    legacy_decision_required = tags.get("decision_required")
    legacy_decision_summary = tags.get("decision_summary")

    # Sanitized KC enrichment
    kc_enrich: list[dict] = []
    if sanitized is not None:
        for kc in sanitized.get("knowledge_cards") or []:
            if kc.get("confidence") in ("high", "medium"):
                kc_enrich.append({
                    "question": kc.get("question", ""),
                    "answer": kc.get("answer", ""),
                    "evidence": kc.get("evidence") or [],
                    "confidence": kc.get("confidence"),
                })

    has_kc = bool(kc_enrich)
    if not (decision_memo or has_decision_outcome or legacy_decision_required or has_kc):
        return None

    session = transcript.get("session", {})
    segments = transcript.get("segments", [])
    turns = merge_turns(segments)
    context = " ".join(t["text"] for t in turns[:3])[:200]

    title = decision_memo[:60] or legacy_decision_summary or (
        (sanitized.get("summary_output") or "")[:60] if sanitized else ""
    ) or "Decision"

    return {
        "decision_id": session_id,
        "title": title,
        "context": context,
        "decision": decision_memo or legacy_decision_summary or "",
        "reasoning": tags.get("reasoning_summary") or "",
        "outcomes": outcomes if isinstance(outcomes, list) else [],
        "knowledge_cards": kc_enrich,
        "summary": (sanitized.get("summary_output") or tags.get("summary", "")) if sanitized else tags.get("summary", ""),
        "domains": tags.get("domains", []),
        "importance": tags.get("importance", ""),
        "source_session": session_id,
        "date": session.get("start"),
    }


# ---------------------------------------------------------------------------
# Format: sop — procedural knowledge from outcomes=TODO発生 + sanitized KC
# ---------------------------------------------------------------------------


def to_sop(
    session_id: str,
    transcript: dict,
    speaker_mode: str = "conversation",
    include_monologue: bool = True,
    sanitized: dict | None = None,
) -> dict | None:
    """Extract SOP (procedural knowledge) from a session.

    Trigger conditions:
        - tags.outcomes contains "TODO発生" or "方針転換", OR
        - legacy tags.action_items is non-empty
    """
    meta = _session_meta(transcript)
    tags = meta["tags"]

    outcomes = tags.get("outcomes") or []
    has_todo = isinstance(outcomes, list) and (
        "TODO発生" in outcomes or "方針転換" in outcomes
    )
    legacy_action_items = tags.get("action_items") or []

    if not has_todo and not legacy_action_items:
        return None

    # Use sanitized keypoints as SOP steps if available, fall back to legacy
    steps: list[str] = []
    if sanitized is not None:
        steps = list(sanitized.get("keypoints") or [])
    if not steps and legacy_action_items:
        steps = [str(a) for a in legacy_action_items if a]

    if not steps:
        return None

    domains = tags.get("domains", [])
    domain = ", ".join(domains) if domains else tags.get("business_domain", "other")

    context = ""
    if sanitized:
        context = sanitized.get("summary_output") or ""
    if not context:
        context = tags.get("summary", "")

    return {
        "id": session_id,
        "title": (context[:60] if context else f"SOP {session_id}"),
        "steps": steps,
        "context": context,
        "domain": domain,
        "source_session": session_id,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


FORMATTERS = {
    "comprehensive": to_comprehensive,
    "openai": to_openai,
    "conversation": to_conversation,
    "knowledge": to_knowledge,
    "decision_log": to_decision_log,
    "sop": to_sop,
}

# Formats that may emit multiple JSONL lines per session
_MULTI_LINE_FORMATS = {"openai", "conversation"}


def export_session(
    fmt: str,
    session_id: str,
    transcript: dict,
    speaker_mode: str = "conversation",
    include_monologue: bool = False,
    sanitized: dict | None = None,
) -> list[str]:
    """Export a single session in the given format.

    Returns a list of JSONL lines (may be empty). Multi-line formats
    (openai, conversation) can yield multiple lines per session.
    """
    formatter = FORMATTERS.get(fmt)
    if not formatter:
        raise ValueError(f"Unknown format: {fmt}")

    result = formatter(
        session_id, transcript, speaker_mode,
        include_monologue=include_monologue,
        sanitized=sanitized,
    )

    if result is None:
        return []
    if isinstance(result, list):
        return [json.dumps(r, ensure_ascii=False) for r in result if r]
    return [json.dumps(result, ensure_ascii=False)]
