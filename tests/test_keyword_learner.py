"""Tests for keyword_learner.py."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from keyword_learner import KeywordLearner


@pytest.fixture
def tmp_env(tmp_path):
    """Create temporary taxonomy, suggestions, and recordings."""
    taxonomy_path = tmp_path / "config" / "tag_taxonomy.json"
    suggestions_path = tmp_path / "config" / "pending_keyword_suggestions.json"
    recordings_dir = tmp_path / "recordings"

    taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
    recordings_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = {
        "version": 1,
        "conversation_type": ["商談"],
        "outcomes": ["意思決定あり"],
        "domains": ["水中超音波", "物販", "AI"],
        "importance": ["高", "中", "低"],
        "keyword_anchors": {
            "水中超音波": ["超音波", "トランスデューサ"],
            "物販": ["Amazon", "FBA"],
            "AI": ["ChatGPT", "LLM"],
        },
    }
    taxonomy_path.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")

    learner = KeywordLearner(
        taxonomy_path=str(taxonomy_path),
        suggestions_path=str(suggestions_path),
        recordings_dir=str(recordings_dir),
    )
    return learner, taxonomy_path, suggestions_path, recordings_dir


def _create_session(recordings_dir: Path, date: str, time: str, text: str, entities=None):
    """Helper: create a minimal transcript.json."""
    session_dir = recordings_dir / date / time
    session_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "segments": [{"text": text, "start": 0, "end": 10}],
        "session": {
            "tags": {
                "entities": entities or [],
            }
        },
    }
    (session_dir / "transcript.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class TestExtractCandidates:
    """Tests for _extract_candidates."""

    def test_extract_candidates_from_entities(self, tmp_env):
        learner, _, _, _ = tmp_env
        all_anchors = {"水中超音波": ["超音波"], "物販": ["Amazon"]}
        candidates = learner._extract_candidates(
            text="テスト文章",
            entities=["振動子", "ソナー装置", "水中通信"],
            target_domain="水中超音波",
            all_anchors=all_anchors,
        )
        assert "振動子" in candidates
        assert "ソナー装置" in candidates

    def test_extract_candidates_katakana(self, tmp_env):
        learner, _, _, _ = tmp_env
        all_anchors = {"水中超音波": ["超音波"]}
        # Words must appear 2+ times to pass frequency filter
        candidates = learner._extract_candidates(
            text="キャビテーションが発生し、インピーダンスが変化した。キャビテーションとインピーダンスの関係",
            entities=[],
            target_domain="水中超音波",
            all_anchors=all_anchors,
        )
        assert "キャビテーション" in candidates
        assert "インピーダンス" in candidates

    def test_extract_candidates_english(self, tmp_env):
        learner, _, _, _ = tmp_env
        all_anchors = {"AI": ["ChatGPT"]}
        # Proper nouns (capitalized, 4+ chars), appearing 2+ times
        candidates = learner._extract_candidates(
            text="We used Llama with Llama model for fine-tuning",
            entities=[],
            target_domain="AI",
            all_anchors=all_anchors,
        )
        assert "Llama" in candidates

    def test_extract_excludes_existing_anchors(self, tmp_env):
        learner, _, _, _ = tmp_env
        all_anchors = {"水中超音波": ["超音波", "トランスデューサ"]}
        candidates = learner._extract_candidates(
            text="超音波とトランスデューサの実験",
            entities=["超音波"],
            target_domain="水中超音波",
            all_anchors=all_anchors,
        )
        assert "超音波" not in candidates
        assert "トランスデューサ" not in candidates

    def test_extract_excludes_cross_domain(self, tmp_env):
        learner, _, _, _ = tmp_env
        # "Amazon" belongs to 物販 domain
        all_anchors = {"水中超音波": ["超音波"], "物販": ["Amazon"]}
        candidates = learner._extract_candidates(
            text="Amazon で部品を注文",
            entities=["Amazon"],
            target_domain="水中超音波",
            all_anchors=all_anchors,
        )
        assert "Amazon" not in candidates

    def test_extract_excludes_short_words(self, tmp_env):
        learner, _, _, _ = tmp_env
        all_anchors = {}
        candidates = learner._extract_candidates(
            text="AB アイ",
            entities=["AB", "アイ"],
            target_domain="AI",
            all_anchors=all_anchors,
        )
        assert "AB" not in candidates
        assert "アイ" not in candidates

    def test_extract_excludes_stopwords(self, tmp_env):
        learner, _, _, _ = tmp_env
        all_anchors = {}
        candidates = learner._extract_candidates(
            text="チョットスゴイコト",
            entities=[],
            target_domain="AI",
            all_anchors=all_anchors,
        )
        assert "チョット" not in candidates
        assert "スゴイ" not in candidates


class TestLearnFromCorrection:
    """Tests for learn_from_correction."""

    def test_learn_from_correction_added_domain(self, tmp_env):
        learner, _, suggestions_path, recordings_dir = tmp_env
        _create_session(
            recordings_dir, "2026-04-05", "10-00-00",
            "キャビテーションの実験をした",
            entities=["キャビテーション"],
        )
        result = learner.learn_from_correction(
            "2026-04-05", "10-00-00",
            old_domains=[], new_domains=["水中超音波"],
        )
        assert result["candidates_added"] > 0
        suggestions = json.loads(suggestions_path.read_text(encoding="utf-8"))
        keywords = [s["keyword"] for s in suggestions]
        assert "キャビテーション" in keywords

    def test_learn_from_correction_no_change(self, tmp_env):
        learner, _, _, recordings_dir = tmp_env
        _create_session(recordings_dir, "2026-04-05", "10-00-00", "テスト")
        result = learner.learn_from_correction(
            "2026-04-05", "10-00-00",
            old_domains=["水中超音波"], new_domains=["水中超音波"],
        )
        assert result["candidates_added"] == 0

    def test_learn_from_correction_removed_domain(self, tmp_env):
        """Removing a domain should NOT trigger learning."""
        learner, _, _, recordings_dir = tmp_env
        _create_session(recordings_dir, "2026-04-05", "10-00-00", "テスト")
        result = learner.learn_from_correction(
            "2026-04-05", "10-00-00",
            old_domains=["水中超音波", "物販"], new_domains=["水中超音波"],
        )
        assert result["candidates_added"] == 0

    def test_learn_increments_occurrences(self, tmp_env):
        learner, _, suggestions_path, recordings_dir = tmp_env
        _create_session(
            recordings_dir, "2026-04-05", "10-00-00",
            "キャビテーション実験",
            entities=["キャビテーション"],
        )
        _create_session(
            recordings_dir, "2026-04-06", "14-00-00",
            "キャビテーション測定",
            entities=["キャビテーション"],
        )
        learner.learn_from_correction(
            "2026-04-05", "10-00-00",
            old_domains=[], new_domains=["水中超音波"],
        )
        learner.learn_from_correction(
            "2026-04-06", "14-00-00",
            old_domains=[], new_domains=["水中超音波"],
        )
        suggestions = json.loads(suggestions_path.read_text(encoding="utf-8"))
        kav = [s for s in suggestions if s["keyword"] == "キャビテーション"]
        assert len(kav) == 1
        assert kav[0]["occurrences"] == 2
        assert len(kav[0]["sessions"]) == 2

    def test_learn_missing_transcript(self, tmp_env):
        learner, _, _, _ = tmp_env
        result = learner.learn_from_correction(
            "2099-01-01", "00-00-00",
            old_domains=[], new_domains=["水中超音波"],
        )
        assert result["candidates_added"] == 0


class TestApproveReject:
    """Tests for approve_suggestion / reject_suggestion."""

    def _seed_suggestion(self, suggestions_path: Path, sid="kws_test1"):
        suggestions = [{
            "id": sid,
            "keyword": "キャビテーション",
            "domain": "水中超音波",
            "occurrences": 2,
            "sessions": ["2026-04-05/10-00-00"],
            "source": "entity",
            "status": "pending",
            "created_at": "2026-04-05T10:00:00",
            "updated_at": "2026-04-05T10:00:00",
        }]
        suggestions_path.write_text(
            json.dumps(suggestions, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def test_approve_suggestion(self, tmp_env):
        learner, taxonomy_path, suggestions_path, _ = tmp_env
        self._seed_suggestion(suggestions_path)
        result = learner.approve_suggestion("kws_test1")
        assert result is not None
        assert result["status"] == "approved"

        # Check keyword_anchors updated
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        assert "キャビテーション" in taxonomy["keyword_anchors"]["水中超音波"]

    def test_reject_suggestion(self, tmp_env):
        learner, taxonomy_path, suggestions_path, _ = tmp_env
        self._seed_suggestion(suggestions_path)
        result = learner.reject_suggestion("kws_test1")
        assert result is not None
        assert result["status"] == "rejected"

        # keyword_anchors NOT updated
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        assert "キャビテーション" not in taxonomy["keyword_anchors"]["水中超音波"]

    def test_approve_duplicate_keyword(self, tmp_env):
        """Approving a keyword that already exists should not create duplicates."""
        learner, taxonomy_path, suggestions_path, _ = tmp_env
        # Manually add keyword to anchors first
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        taxonomy["keyword_anchors"]["水中超音波"].append("キャビテーション")
        taxonomy_path.write_text(
            json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self._seed_suggestion(suggestions_path)
        result = learner.approve_suggestion("kws_test1")
        assert result["status"] == "approved"

        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        count = taxonomy["keyword_anchors"]["水中超音波"].count("キャビテーション")
        assert count == 1

    def test_approve_nonexistent(self, tmp_env):
        learner, _, _, _ = tmp_env
        result = learner.approve_suggestion("nonexistent_id")
        assert result is None

    def test_reject_nonexistent(self, tmp_env):
        learner, _, _, _ = tmp_env
        result = learner.reject_suggestion("nonexistent_id")
        assert result is None
