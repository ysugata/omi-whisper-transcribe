"""Training data export — convert transcript sessions to fine-tuning formats."""

import json
from typing import Any


def merge_turns(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments from the same speaker into turns.

    Returns list of dicts with keys: speaker, is_user, text, start, end.
    """
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
            # Merge with previous turn
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


def _build_system_message(tags: dict, is_monologue: bool = False) -> str:
    """Build a system message from session tags."""
    parts = []
    if is_monologue:
        parts.append("あなたは事業経営者です。以下は独り言/思考です。")
    else:
        parts.append("あなたは事業経営者です。")

    if tags.get("activity"):
        parts.append(f"状況: {tags['activity']}")
    if tags.get("topics"):
        topics = tags["topics"] if isinstance(tags["topics"], list) else [tags["topics"]]
        parts.append(f"話題: {', '.join(topics)}")
    if tags.get("mood"):
        parts.append(f"ムード: {tags['mood']}")
    return "\n".join(parts)


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


def to_comprehensive(session_id: str, transcript: dict,
                      speaker_mode: str = "conversation") -> dict | None:
    """Convert a session to comprehensive JSONL format (1 dict per session)."""
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

    tags_info = {}
    for key in ("topics", "activity", "mood", "quality", "entities", "summary", "intent"):
        if tags.get(key):
            tags_info[key] = tags[key]

    return {
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


def to_openai(session_id: str, transcript: dict,
              speaker_mode: str = "conversation") -> dict | None:
    """Convert a session to OpenAI fine-tuning JSONL format.

    - is_user speaker → assistant role (AI learns this)
    - other speakers → user role (input to AI)
    - monologue → synthetic user prompt + assistant = all speech
    """
    segments = transcript.get("segments", [])
    if not segments:
        return None

    meta = _session_meta(transcript)
    tags = meta["tags"]
    turns = merge_turns(segments)

    if not turns:
        return None

    has_user = any(t["is_user"] for t in turns)
    has_other = any(not t["is_user"] for t in turns)
    is_monologue = has_user and not has_other

    system_msg = _build_system_message(tags, is_monologue=is_monologue)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]

    if is_monologue:
        # Monologue: synthesize user prompt, combine all speech as assistant
        topics = tags.get("topics", [])
        topic_str = ", ".join(topics) if topics else "日常"
        messages.append({"role": "user", "content": f"{topic_str}について考えてください"})
        all_text = " ".join(t["text"] for t in turns)
        messages.append({"role": "assistant", "content": all_text})
    else:
        if speaker_mode == "user_only":
            # Only user turns as assistant, skip others
            for t in turns:
                if t["is_user"]:
                    messages.append({"role": "assistant", "content": t["text"]})
        else:
            # Full conversation
            for t in turns:
                role = "assistant" if t["is_user"] else "user"
                messages.append({"role": role, "content": t["text"]})

    # Must have at least one assistant message
    if not any(m["role"] == "assistant" for m in messages):
        return None

    return {"messages": messages}


def to_conversation(session_id: str, transcript: dict,
                    speaker_mode: str = "conversation") -> dict | None:
    """Convert a session to ShareGPT/LLaMA-Factory conversation format."""
    segments = transcript.get("segments", [])
    if not segments:
        return None

    meta = _session_meta(transcript)
    tags = meta["tags"]
    turns = merge_turns(segments)

    if not turns:
        return None

    has_user = any(t["is_user"] for t in turns)
    has_other = any(not t["is_user"] for t in turns)
    is_monologue = has_user and not has_other

    system_msg = _build_system_message(tags, is_monologue=is_monologue)
    conversations: list[dict[str, str]] = []

    if is_monologue:
        topics = tags.get("topics", [])
        topic_str = ", ".join(topics) if topics else "日常"
        conversations.append({"from": "human", "value": f"{topic_str}について考えてください"})
        all_text = " ".join(t["text"] for t in turns)
        conversations.append({"from": "gpt", "value": all_text})
    else:
        if speaker_mode == "user_only":
            for t in turns:
                if t["is_user"]:
                    conversations.append({"from": "gpt", "value": t["text"]})
        else:
            for t in turns:
                role = "gpt" if t["is_user"] else "human"
                conversations.append({"from": role, "value": t["text"]})

    # Must have at least one gpt turn
    if not any(c["from"] == "gpt" for c in conversations):
        return None

    metadata = {}
    for key in ("topics", "activity", "quality", "mood"):
        if tags.get(key):
            metadata[key] = tags[key]

    return {
        "id": session_id,
        "system": system_msg,
        "conversations": conversations,
        "metadata": metadata,
    }


FORMATTERS = {
    "comprehensive": to_comprehensive,
    "openai": to_openai,
    "conversation": to_conversation,
}


def export_session(fmt: str, session_id: str, transcript: dict,
                   speaker_mode: str = "conversation") -> str | None:
    """Export a single session in the given format. Returns JSONL line or None."""
    formatter = FORMATTERS.get(fmt)
    if not formatter:
        raise ValueError(f"Unknown format: {fmt}")

    result = formatter(session_id, transcript, speaker_mode)
    if result is None:
        return None
    return json.dumps(result, ensure_ascii=False)
