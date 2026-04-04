"""Correction-driven keyword learning.

When a user manually corrects session domain tags, this module extracts
keyword candidates from the session transcript and accumulates them in
``config/pending_keyword_suggestions.json``.  Approved candidates are
written back to ``keyword_anchors`` in ``tag_taxonomy.json``.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-words: common katakana particles / filler that are not domain keywords
# ---------------------------------------------------------------------------
_KATAKANA_STOPWORDS = {
    "デス", "マス", "ナイ", "ダケ", "タチ", "コト", "モノ", "トコロ",
    "ソレ", "コレ", "アレ", "ドコ", "ナニ", "ダレ", "イツ", "ドウ",
    "ヤッパリ", "チョット", "スゴイ", "スゴク", "ホント", "ホントウ",
    "タブン", "ゼンゼン", "ゼッタイ", "マジ", "ケド", "ダカラ",
    "ソウ", "アノ", "エット", "ネー", "ハイ", "ウン",
}


class KeywordLearner:
    """Extract, accumulate and manage keyword suggestions."""

    def __init__(
        self,
        taxonomy_path: str,
        suggestions_path: str,
        recordings_dir: str,
    ) -> None:
        self.taxonomy_path = Path(taxonomy_path)
        self.suggestions_path = Path(suggestions_path)
        self.recordings_dir = Path(recordings_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn_from_correction(
        self,
        date: str,
        time: str,
        old_domains: list,
        new_domains: list,
    ) -> dict:
        """Learn keywords from a tag correction.

        Only *added* domains trigger learning.  Returns summary counts.
        """
        added = set(new_domains) - set(old_domains)
        if not added:
            return {"candidates_added": 0, "total_pending": self._count_pending()}

        # Read session transcript
        transcript_path = self.recordings_dir / date / time / "transcript.json"
        if not transcript_path.exists():
            log.warning("Transcript not found: %s", transcript_path)
            return {"candidates_added": 0, "total_pending": self._count_pending()}

        try:
            data = json.loads(transcript_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to read transcript %s: %s", transcript_path, exc)
            return {"candidates_added": 0, "total_pending": self._count_pending()}

        segments = data.get("segments", [])
        session_text = " ".join(seg.get("text", "") for seg in segments)
        entities: list = []
        session_info = data.get("session", {})
        tags = session_info.get("tags", {})
        if isinstance(tags.get("entities"), list):
            entities = tags["entities"]

        taxonomy = self._load_taxonomy()
        all_anchors = taxonomy.get("keyword_anchors", {})

        suggestions = self._load_suggestions()
        session_id = f"{date}/{time}"
        added_count = 0

        for domain in added:
            candidates = self._extract_candidates(
                session_text, entities, domain, all_anchors,
            )
            for kw in candidates:
                existing = self._find_suggestion(suggestions, kw, domain)
                if existing:
                    existing["occurrences"] += 1
                    if session_id not in existing["sessions"]:
                        existing["sessions"].append(session_id)
                    existing["updated_at"] = datetime.now().isoformat(timespec="seconds")
                else:
                    suggestions.append({
                        "id": f"kws_{uuid.uuid4().hex[:8]}",
                        "keyword": kw,
                        "domain": domain,
                        "occurrences": 1,
                        "sessions": [session_id],
                        "source": "entity" if kw in entities else "text_scan",
                        "status": "pending",
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    })
                    added_count += 1

        self._save_suggestions(suggestions)
        return {
            "candidates_added": added_count,
            "total_pending": self._count_pending(suggestions),
        }

    def get_suggestions(self, status: str = "pending") -> list:
        """Return suggestions filtered by status, sorted by occurrences desc."""
        suggestions = self._load_suggestions()
        filtered = [s for s in suggestions if s.get("status") == status]
        filtered.sort(key=lambda s: s.get("occurrences", 0), reverse=True)
        return filtered

    def approve_suggestion(self, suggestion_id: str) -> Optional[dict]:
        """Approve a suggestion: update status and add to keyword_anchors."""
        suggestions = self._load_suggestions()
        target = None
        for s in suggestions:
            if s["id"] == suggestion_id:
                target = s
                break
        if not target:
            return None

        target["status"] = "approved"
        target["updated_at"] = datetime.now().isoformat(timespec="seconds")

        # Add to taxonomy keyword_anchors
        taxonomy = self._load_taxonomy()
        anchors = taxonomy.setdefault("keyword_anchors", {})
        domain_keywords = anchors.setdefault(target["domain"], [])
        if target["keyword"] not in domain_keywords:
            domain_keywords.append(target["keyword"])
        self._save_taxonomy(taxonomy)

        self._save_suggestions(suggestions)
        return target

    def reject_suggestion(self, suggestion_id: str) -> Optional[dict]:
        """Reject a suggestion."""
        suggestions = self._load_suggestions()
        target = None
        for s in suggestions:
            if s["id"] == suggestion_id:
                target = s
                break
        if not target:
            return None

        target["status"] = "rejected"
        target["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._save_suggestions(suggestions)
        return target

    # ------------------------------------------------------------------
    # Candidate extraction
    # ------------------------------------------------------------------

    def _extract_candidates(
        self,
        text: str,
        entities: list,
        target_domain: str,
        all_anchors: dict,
    ) -> List[str]:
        """Extract keyword candidates for *target_domain* from text/entities."""
        existing = set(all_anchors.get(target_domain, []))
        # Build set of all keywords across ALL domains for cross-domain check
        other_domain_keywords: set = set()
        for dom, kws in all_anchors.items():
            if dom != target_domain:
                other_domain_keywords.update(kws)

        candidates: list = []
        seen: set = set()

        def _add(word: str):
            w = word.strip()
            if not w or len(w) <= 2:
                return
            if w in seen or w in existing:
                return
            if w in other_domain_keywords:
                return
            if w in _KATAKANA_STOPWORDS:
                return
            seen.add(w)
            candidates.append(w)

        # 1. Entities
        for ent in entities:
            if isinstance(ent, str):
                _add(ent)

        # 2. Text scan — katakana words (3+ chars)
        for m in re.findall(r"[\u30A0-\u30FF]{3,}", text):
            _add(m)

        # 3. Text scan — English words (3+ chars)
        for m in re.findall(r"[A-Za-z]{3,}", text):
            _add(m)

        # 4. Kanji + katakana / katakana + kanji compound words
        for m in re.findall(
            r"[\u4E00-\u9FFF]+[\u30A0-\u30FF]+|[\u30A0-\u30FF]+[\u4E00-\u9FFF]+",
            text,
        ):
            _add(m)

        return candidates

    # ------------------------------------------------------------------
    # File I/O helpers
    # ------------------------------------------------------------------

    def _load_suggestions(self) -> list:
        try:
            if self.suggestions_path.exists():
                return json.loads(
                    self.suggestions_path.read_text(encoding="utf-8")
                )
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to load suggestions: %s", exc)
        return []

    def _save_suggestions(self, suggestions: list) -> None:
        self.suggestions_path.parent.mkdir(parents=True, exist_ok=True)
        self.suggestions_path.write_text(
            json.dumps(suggestions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_taxonomy(self) -> dict:
        try:
            if self.taxonomy_path.exists():
                return json.loads(
                    self.taxonomy_path.read_text(encoding="utf-8")
                )
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to load taxonomy: %s", exc)
        return {}

    def _save_taxonomy(self, data: dict) -> None:
        self.taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
        self.taxonomy_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _count_pending(self, suggestions: Optional[list] = None) -> int:
        if suggestions is None:
            suggestions = self._load_suggestions()
        return sum(1 for s in suggestions if s.get("status") == "pending")

    @staticmethod
    def _find_suggestion(suggestions: list, keyword: str, domain: str):
        """Find existing pending suggestion for (keyword, domain) pair."""
        for s in suggestions:
            if (
                s.get("keyword") == keyword
                and s.get("domain") == domain
                and s.get("status") == "pending"
            ):
                return s
        return None
