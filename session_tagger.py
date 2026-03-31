"""Session-level tagging module using Gemini API.

Tags each recording session with topics, activity, intent, mood,
quality, entities, and a short Japanese summary.
Uses OpenAI-compatible API (same as LLMCorrector).
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """あなたは音声文字起こしテキストの分類・要約者です。
セッションの全発話テキストが入力されます。以下の分類を JSON で返してください。

出力フォーマット (JSON のみ、他のテキストは不要):
{
  "topics": ["topic1", "topic2"],
  "activity": "one_of_enum",
  "intent": "one_of_enum",
  "mood": "one_of_enum",
  "quality": "one_of_enum",
  "entities": ["Entity1", "Entity2"],
  "summary": "日本語で1-2文の要約",
  "source_type": "one_of_enum",
  "business_domain": "one_of_enum",
  "importance": "one_of_enum",
  "confidentiality": "one_of_enum",
  "decision_required": true_or_false,
  "decision_summary": "何が決まったか (50文字以内、なければnull)",
  "reasoning_summary": "なぜそう決めたか (50文字以内、なければnull)",
  "action_items": ["次のステップ"],
  "outcome": "one_of_enum_or_null",
  "next_action": "次にやること (50文字以内、なければnull)"
}

### topics (1-3個、英語)
自由形式の短いトピックラベル。例: "technology", "cooking", "work", "health", "travel", "gaming", "finance", "family", "education"

### activity (英語 enum、1つ)
casual_chat, meeting, brainstorming, planning, debugging, learning, storytelling, phone_call, dictation, other

### intent (英語 enum、1つ)
informing, planning, problem_solving, socializing, reflecting, instructing, requesting, other

### mood (英語 enum、1つ)
positive, neutral, negative, mixed, excited, frustrated

### quality (英語 enum、1つ)
clean (明瞭で完全), partial (部分的に不明瞭), noisy (ノイズ多い), fragmented (断片的)

### entities
テキスト中に登場する固有名詞 (人名、地名、サービス名、技術名など) をそのまま抽出。なければ空配列。

### summary
日本語で1-2文。何について話していたかの概要。

### source_type (英語 enum、1つ)
会話の形式: meeting, monologue, negotiation, phone_call, dictation, casual_chat, discussion, presentation, other

### business_domain (英語 enum、1つ)
事業領域: sales, hiring, product_planning, purchasing, customer_service, management, finance, technology, personal, other

### importance (英語 enum、1つ)
重要度: high (事業判断に直結), medium (参考情報), low (雑談等)

### confidentiality (英語 enum、1つ)
機密レベル: public (社外共有可), internal (社内限定), confidential (関係者限定)

### decision_required (boolean)
このセッションで何かの意思決定が行われた/必要とされたか。true or false。

### decision_summary
何が決まったか、日本語50文字以内。決定事項がなければ null。

### reasoning_summary
なぜそう決めたか (判断の根拠)、日本語50文字以内。なければ null。

### action_items
次のステップのリスト (日本語)。最大5件。なければ空配列。

### outcome (英語 enum or null)
結果: pending (保留中), completed (完了), cancelled (中止), deferred (延期)。該当なしなら null。

### next_action
次にやること、日本語50文字以内。なければ null。"""


# Valid enum values for validation
_VALID_ACTIVITY = {
    "casual_chat", "meeting", "brainstorming", "planning", "debugging",
    "learning", "storytelling", "phone_call", "dictation", "other",
}
_VALID_INTENT = {
    "informing", "planning", "problem_solving", "socializing",
    "reflecting", "instructing", "requesting", "other",
}
_VALID_MOOD = {
    "positive", "neutral", "negative", "mixed", "excited", "frustrated",
}
_VALID_QUALITY = {"clean", "partial", "noisy", "fragmented"}
_VALID_SOURCE_TYPE = {
    "meeting", "monologue", "negotiation", "phone_call", "dictation",
    "casual_chat", "discussion", "presentation", "other",
}
_VALID_BUSINESS_DOMAIN = {
    "sales", "hiring", "product_planning", "purchasing", "customer_service",
    "management", "finance", "technology", "personal", "other",
}
_VALID_IMPORTANCE = {"high", "medium", "low"}
_VALID_CONFIDENTIALITY = {"public", "internal", "confidential"}
_VALID_OUTCOME = {"pending", "completed", "cancelled", "deferred"}


class SessionTagger:
    """Tags a session with topics, activity, intent, mood, quality, entities, summary."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0,
        max_tokens: int = 768,
        timeout: int = 15,
        user_tags_path: str = "",
    ) -> None:
        self._enabled = bool(api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None
        self._user_tags_path = user_tags_path
        self._user_tags: list[dict] = []

        if user_tags_path:
            self._load_user_tags()

        if self._enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
                log.info(f"SessionTagger initialized: model={model}")
            except Exception as e:
                log.error(f"SessionTagger init failed: {e}")
                self._enabled = False

    def _load_user_tags(self) -> None:
        """Load user tag definitions from JSON file."""
        if not self._user_tags_path or not os.path.exists(self._user_tags_path):
            self._user_tags = []
            return
        try:
            with open(self._user_tags_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._user_tags = data
                log.info(f"SessionTagger: loaded {len(data)} user tag definitions")
            else:
                self._user_tags = []
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"SessionTagger: failed to load user tags: {e}")
            self._user_tags = []

    def reload_user_tags(self) -> None:
        """Reload user tag definitions (called when tags are updated via CRUD)."""
        self._load_user_tags()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _build_system_prompt(self) -> str:
        """Build the system prompt, including user tag definitions if any."""
        prompt = SYSTEM_PROMPT
        if self._user_tags:
            prompt += "\n\n### user_tags (ユーザー定義タグ、該当するもの全て)\n"
            prompt += "利用可能なタグ:\n"
            for tag in self._user_tags:
                name = tag.get("name", "")
                desc = tag.get("description", "")
                if name:
                    prompt += f'- "{name}": {desc}\n' if desc else f'- "{name}"\n'
            prompt += '該当するタグ名のリストを user_tags に入れてください。該当なしなら空配列。'
        return prompt

    def _build_input(self, segments: list[dict]) -> str:
        """Build session text from segments."""
        lines = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if text:
                speaker = seg.get("person_name") or seg.get("speaker", "")
                if speaker:
                    lines.append(f"[{speaker}] {text}")
                else:
                    lines.append(text)
        return "\n".join(lines)

    def _parse_response(self, raw: str) -> Optional[dict]:
        """Parse JSON response from LLM."""
        # Try to extract JSON from response (may have markdown code fences)
        raw = raw.strip()
        if raw.startswith("```"):
            # Remove markdown code fences
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(f"SessionTagger: invalid JSON response: {raw[:200]}")
            return None

        if not isinstance(data, dict):
            log.warning(f"SessionTagger: response is not a dict")
            return None

        return data

    def _validate_tags(self, tags: dict) -> dict:
        """Validate and normalize tag values."""
        result = {}

        # topics: list of strings
        topics = tags.get("topics", [])
        if isinstance(topics, list):
            result["topics"] = [str(t).lower().strip() for t in topics[:5] if t]
        else:
            result["topics"] = []

        # activity: enum
        activity = str(tags.get("activity", "other")).lower().strip()
        result["activity"] = activity if activity in _VALID_ACTIVITY else "other"

        # intent: enum
        intent = str(tags.get("intent", "other")).lower().strip()
        result["intent"] = intent if intent in _VALID_INTENT else "other"

        # mood: enum
        mood = str(tags.get("mood", "neutral")).lower().strip()
        result["mood"] = mood if mood in _VALID_MOOD else "neutral"

        # quality: enum
        quality = str(tags.get("quality", "clean")).lower().strip()
        result["quality"] = quality if quality in _VALID_QUALITY else "clean"

        # entities: list of strings (preserve case)
        entities = tags.get("entities", [])
        if isinstance(entities, list):
            result["entities"] = [str(e).strip() for e in entities[:20] if e]
        else:
            result["entities"] = []

        # summary: string (Japanese)
        summary = str(tags.get("summary", "")).strip()
        result["summary"] = summary[:200] if summary else ""

        # user_tags: list of tag names → convert to tag IDs
        user_tags_raw = tags.get("user_tags", [])
        if isinstance(user_tags_raw, list) and self._user_tags:
            # Build name→id map
            name_to_id = {t["name"]: t["id"] for t in self._user_tags if t.get("name") and t.get("id")}
            resolved = []
            for tag_name in user_tags_raw:
                tag_name_str = str(tag_name).strip()
                if tag_name_str in name_to_id:
                    resolved.append(name_to_id[tag_name_str])
            result["user_tags"] = resolved
        else:
            result["user_tags"] = []

        # --- Business succession fields ---

        # source_type: enum
        source_type = str(tags.get("source_type", "other")).lower().strip()
        result["source_type"] = source_type if source_type in _VALID_SOURCE_TYPE else "other"

        # business_domain: enum
        business_domain = str(tags.get("business_domain", "other")).lower().strip()
        result["business_domain"] = business_domain if business_domain in _VALID_BUSINESS_DOMAIN else "other"

        # importance: enum
        importance = str(tags.get("importance", "low")).lower().strip()
        result["importance"] = importance if importance in _VALID_IMPORTANCE else "low"

        # confidentiality: enum
        confidentiality = str(tags.get("confidentiality", "internal")).lower().strip()
        result["confidentiality"] = confidentiality if confidentiality in _VALID_CONFIDENTIALITY else "internal"

        # decision_required: bool
        result["decision_required"] = bool(tags.get("decision_required", False))

        # decision_summary: string or None (truncate to 100 chars)
        ds = tags.get("decision_summary")
        if ds and isinstance(ds, str) and ds.strip():
            result["decision_summary"] = ds.strip()[:100]
        else:
            result["decision_summary"] = None

        # reasoning_summary: string or None (truncate to 100 chars)
        rs = tags.get("reasoning_summary")
        if rs and isinstance(rs, str) and rs.strip():
            result["reasoning_summary"] = rs.strip()[:100]
        else:
            result["reasoning_summary"] = None

        # action_items: list of strings (max 5)
        action_items = tags.get("action_items", [])
        if isinstance(action_items, list):
            result["action_items"] = [str(a).strip() for a in action_items[:5] if a and str(a).strip()]
        else:
            result["action_items"] = []

        # outcome: enum or None
        outcome = tags.get("outcome")
        if outcome and isinstance(outcome, str):
            outcome = outcome.lower().strip()
            result["outcome"] = outcome if outcome in _VALID_OUTCOME else None
        else:
            result["outcome"] = None

        # next_action: string or None (truncate to 100 chars)
        na = tags.get("next_action")
        if na and isinstance(na, str) and na.strip():
            result["next_action"] = na.strip()[:100]
        else:
            result["next_action"] = None

        return result

    def tag_session(self, segments: list[dict]) -> Optional[dict]:
        """Tag a session's segments and return tags dict.

        Returns None on error (API failure, invalid response, etc.).
        """
        if not self._enabled:
            return None

        if not segments:
            return None

        session_text = self._build_input(segments)
        if not session_text or len(session_text) < 10:
            log.debug("SessionTagger: session text too short, skipping")
            return None

        try:
            system_prompt = self._build_system_prompt()
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": session_text},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            raw = response.choices[0].message.content
            log.debug(f"SessionTagger raw response: {raw[:300]}")

            parsed = self._parse_response(raw)
            if parsed is None:
                return None

            tags = self._validate_tags(parsed)
            tags["tagged_at"] = datetime.now().isoformat()

            log.info(
                f"SessionTagger: activity={tags['activity']}, "
                f"mood={tags['mood']}, topics={tags['topics']}, "
                f"summary={tags['summary'][:50]}"
            )
            return tags

        except Exception as e:
            log.warning(f"SessionTagger API error: {e}")
            return None
