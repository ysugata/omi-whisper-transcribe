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
# Stop-words: generic words that are NOT useful as domain-specific keywords
# ---------------------------------------------------------------------------
_KATAKANA_STOPWORDS = {
    # Filler / particles
    "デス", "マス", "ナイ", "ダケ", "タチ", "コト", "モノ", "トコロ",
    "ソレ", "コレ", "アレ", "ドコ", "ナニ", "ダレ", "イツ", "ドウ",
    "ヤッパリ", "チョット", "スゴイ", "スゴク", "ホント", "ホントウ",
    "タブン", "ゼンゼン", "ゼッタイ", "マジ", "ケド", "ダカラ",
    "ソウ", "アノ", "エット", "ネー", "ハイ", "ウン",
    # Generic katakana words (technology, business, daily)
    "プラス", "マイナス", "メイン", "メール", "リンク", "サーバー",
    "システム", "データ", "ファイル", "フォルダ", "ダウンロード",
    "アップロード", "アクセス", "チャンネル", "コントロール",
    "パスワード", "アカウント", "プロフィール", "コメント",
    "グループ", "メンバー", "リスト", "ページ", "サイト",
    "ボタン", "クリック", "チェック", "スタート", "ストップ",
    "オープン", "クローズ", "セット", "リセット", "アップデート",
    "バージョン", "パーセンテージ", "コスト", "プラン", "プラム",
    "フレンド", "ワット", "クォーター", "デイラリー",
    "ポイント", "レベル", "タイプ", "モード", "ステータス",
    "エラー", "イベント", "テスト", "サンプル", "ケース",
    "プロセス", "メソッド", "パターン", "モデル", "ルール",
    "インポイント", "コンソール", "サーブ", "リロード", "リリース",
}

_ENGLISH_STOPWORDS = {
    # Common English words that slip through from ASR
    "the", "and", "for", "that", "this", "with", "from", "have", "has",
    "not", "are", "was", "were", "been", "being", "will", "would", "could",
    "should", "can", "may", "might", "shall", "must", "need",
    "get", "got", "set", "let", "put", "run", "use", "try", "ask",
    "say", "said", "see", "saw", "take", "took", "make", "made",
    "all", "any", "some", "many", "much", "more", "most", "other",
    "also", "just", "only", "very", "really", "actually", "basically",
    "like", "well", "then", "than", "when", "where", "what", "which",
    "who", "how", "why", "about", "after", "before", "into", "over",
    # Generic tech words not domain-specific
    "http", "https", "www", "com", "org", "net", "html", "css",
    "api", "url", "app", "web", "log", "dev", "src", "bin",
    "system", "server", "client", "user", "data", "file", "code",
    "test", "debug", "error", "null", "true", "false", "type",
    "click", "link", "page", "site", "view", "list", "item",
    "start", "stop", "open", "close", "save", "load", "send",
    "Chrome", "Google", "YouTube", "Facebook", "Twitter", "Amazon",
    "activate", "download", "upload", "update", "install",
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
            if s.get("id") == suggestion_id:
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
            if s.get("id") == suggestion_id:
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
        """Extract keyword candidates for *target_domain* from text/entities.

        Quality filters:
        - Katakana words must be 4+ chars (3-char katakana is almost always noise)
        - English words must be 4+ chars and start with uppercase (proper nouns)
        - Must appear 2+ times in the text (frequency filter)
        - Excluded by stop-word lists and cross-domain check
        """
        existing = set(all_anchors.get(target_domain, []))
        other_domain_keywords: set = set()
        for dom, kws in all_anchors.items():
            if dom != target_domain:
                other_domain_keywords.update(kws)

        # Pre-compute word frequencies for quality filtering
        text_lower = text.lower()

        candidates: list = []
        seen: set = set()

        def _add(word: str, min_occurrences: int = 2):
            w = word.strip()
            if not w or len(w) <= 2:
                return
            if w in seen or w in existing:
                return
            if w in other_domain_keywords:
                return
            if w in _KATAKANA_STOPWORDS:
                return
            if w.lower() in _ENGLISH_STOPWORDS or w in _ENGLISH_STOPWORDS:
                return
            # Frequency check: must appear at least min_occurrences times
            if min_occurrences > 1 and text.count(w) < min_occurrences:
                return
            seen.add(w)
            candidates.append(w)

        # 1. Entities (trusted source — lower threshold)
        for ent in entities:
            if isinstance(ent, str) and len(ent) >= 3:
                _add(ent, min_occurrences=1)

        # 2. Katakana words (4+ chars to avoid noise like プラス, メイン)
        for m in re.findall(r"[\u30A0-\u30FF]{4,}", text):
            _add(m)

        # 3. English proper nouns (capitalized, 4+ chars)
        for m in re.findall(r"[A-Z][a-zA-Z]{3,}", text):
            _add(m)

        # 4. Kanji + katakana compound words (likely domain-specific)
        for m in re.findall(
            r"[\u4E00-\u9FFF]+[\u30A0-\u30FF]{2,}|[\u30A0-\u30FF]{2,}[\u4E00-\u9FFF]+",
            text,
        ):
            if len(m) >= 4:
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
        tmp = self.suggestions_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(suggestions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.suggestions_path)

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
        # tmp + rename: tag_taxonomy.json は refiner / session_tagger と
        # 共有される。途中で truncate されると全タグ付けが degrade する。
        self.taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.taxonomy_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.taxonomy_path)

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
