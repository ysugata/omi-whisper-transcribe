"""Unit tests for RagClassifier."""

import json
import os
import tempfile

import numpy as np
import pytest

from rag_classifier import RagClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classifier(tmpdir, min_sessions=2, k=3, confidence_threshold=0.5):
    """Create a RagClassifier with a temp index dir (no real API key)."""
    return RagClassifier(
        api_key="fake-key",
        recordings_dir=str(tmpdir),
        index_dir=str(tmpdir),
        model="gemini-embedding-001",
        k=k,
        confidence_threshold=confidence_threshold,
        min_sessions=min_sessions,
    )


def _fake_embedding(dim=768, seed=42):
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


SAMPLE_TAGS_A = {
    "conversation_type": "商談",
    "outcomes": ["意思決定あり"],
    "domains": ["水中超音波"],
    "importance": "高",
    "quality": "clean",
}

SAMPLE_TAGS_B = {
    "conversation_type": "商談",
    "outcomes": ["情報共有のみ"],
    "domains": ["水中超音波"],
    "importance": "高",
    "quality": "clean",
}

SAMPLE_TAGS_C = {
    "conversation_type": "雑談・移動中メモ",
    "outcomes": ["なし"],
    "domains": [],
    "importance": "低",
    "quality": "clean",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractPlainText:
    def test_basic(self):
        segments = [
            {"text": "  Hello  ", "person_name": "taro", "is_user": True},
            {"text": "world", "person_name": "unknown"},
            {"text": "", "person_name": ""},
        ]
        result = RagClassifier._extract_plain_text(segments)
        assert result == "Hello\nworld"

    def test_empty(self):
        assert RagClassifier._extract_plain_text([]) == ""

    def test_no_text_key(self):
        segments = [{"other": "data"}]
        assert RagClassifier._extract_plain_text(segments) == ""


class TestWeightedVote:
    def test_single_domain(self):
        rc = _make_classifier(tempfile.mkdtemp())
        neighbors = [
            {"similarity": 0.9, "tags": SAMPLE_TAGS_A},
            {"similarity": 0.85, "tags": SAMPLE_TAGS_B},
            {"similarity": 0.8, "tags": SAMPLE_TAGS_A},
        ]
        result = rc._weighted_vote(neighbors)
        assert "商談" in result["conversation_type"]
        assert result["conversation_type"]["商談"] == pytest.approx(0.9 + 0.85 + 0.8)

    def test_mixed(self):
        rc = _make_classifier(tempfile.mkdtemp())
        neighbors = [
            {"similarity": 0.9, "tags": SAMPLE_TAGS_A},
            {"similarity": 0.8, "tags": SAMPLE_TAGS_C},
        ]
        result = rc._weighted_vote(neighbors)
        assert "商談" in result["conversation_type"]
        assert "雑談・移動中メモ" in result["conversation_type"]


class TestIsConfident:
    def test_clear_winner(self):
        rc = _make_classifier(tempfile.mkdtemp(), confidence_threshold=0.5)
        votes = {
            "conversation_type": {"商談": 2.5, "雑談・移動中メモ": 0.3},
            "importance": {"高": 2.0, "低": 0.5},
            "outcomes": {"意思決定あり": 2.0},
            "domains": {"水中超音波": 2.0},
        }
        assert rc._is_confident(votes) is True

    def test_tie(self):
        rc = _make_classifier(tempfile.mkdtemp(), confidence_threshold=0.6)
        votes = {
            "conversation_type": {"商談": 1.0, "雑談・移動中メモ": 1.0},
            "importance": {"高": 1.0, "低": 1.0},
            "outcomes": {},
            "domains": {},
        }
        # top/total = 0.5, which is below threshold 0.6
        assert rc._is_confident(votes) is False

    def test_empty_axis(self):
        rc = _make_classifier(tempfile.mkdtemp())
        votes = {
            "conversation_type": {},
            "importance": {"高": 1.0},
        }
        assert rc._is_confident(votes) is False


class TestVotesToTags:
    def test_basic(self):
        rc = _make_classifier(tempfile.mkdtemp())
        votes = {
            "conversation_type": {"商談": 2.5, "雑談・移動中メモ": 0.3},
            "importance": {"高": 2.0, "低": 0.5},
            "quality": {"clean": 2.5},
            "outcomes": {"意思決定あり": 2.0, "情報共有のみ": 0.5},
            "domains": {"水中超音波": 2.5, "AI": 0.1},
        }
        tags = rc._votes_to_tags(votes)
        assert tags["conversation_type"] == "商談"
        assert tags["importance"] == "高"
        assert tags["quality"] == "clean"
        assert "水中超音波" in tags["domains"]


class TestAddSessionAndSearch:
    def test_add_and_search(self):
        tmpdir = tempfile.mkdtemp()
        rc = _make_classifier(tmpdir, min_sessions=1)
        emb = _fake_embedding(768, seed=1)
        rc.add_session("2026-01-01/12-00-00", emb, SAMPLE_TAGS_A)
        assert rc.session_count() == 1

        results = rc._search_similar(emb, k=1)
        assert len(results) == 1
        assert results[0]["similarity"] > 0.99
        assert results[0]["tags"]["conversation_type"] == "商談"

    def test_duplicate_skip(self):
        tmpdir = tempfile.mkdtemp()
        rc = _make_classifier(tmpdir)
        emb = _fake_embedding(768)
        rc.add_session("2026-01-01/12-00-00", emb, SAMPLE_TAGS_A)
        rc.add_session("2026-01-01/12-00-00", emb, SAMPLE_TAGS_A)
        assert rc.session_count() == 1


class TestClassify:
    def test_not_ready(self):
        tmpdir = tempfile.mkdtemp()
        rc = _make_classifier(tmpdir, min_sessions=100)
        emb = _fake_embedding(768)
        rc.add_session("s1", emb, SAMPLE_TAGS_A)
        result = rc.classify.__wrapped__(rc, "test") if hasattr(rc.classify, '__wrapped__') else None
        # classify needs embed_text which requires API, so test not_ready path
        assert not rc.is_ready()

    def test_confident_with_mocked_embed(self, monkeypatch):
        tmpdir = tempfile.mkdtemp()
        rc = _make_classifier(tmpdir, min_sessions=2, k=3, confidence_threshold=0.4)

        # Seed 3 similar sessions
        base_emb = _fake_embedding(768, seed=10)
        for i in range(3):
            # Add slight noise
            noise = np.random.RandomState(i + 100).randn(768).astype(np.float32) * 0.01
            emb = (np.array(base_emb) + noise).tolist()
            rc.add_session(f"2026-01-{i+1:02d}/12-00-00", emb, SAMPLE_TAGS_A)

        assert rc.is_ready()

        # Mock embed_text to return the base embedding
        monkeypatch.setattr(rc, "embed_text", lambda text: base_emb)

        result = rc.classify("テスト会話テキスト")
        assert result["confident"] is True
        assert result["tags"]["conversation_type"] == "商談"
        assert result["top_similarity"] > 0.9

    def test_ambiguous_with_mocked_embed(self, monkeypatch):
        tmpdir = tempfile.mkdtemp()
        rc = _make_classifier(tmpdir, min_sessions=2, k=4, confidence_threshold=0.7)

        # Add 2 sessions of type A and 2 of type C
        emb_a = _fake_embedding(768, seed=20)
        emb_c = _fake_embedding(768, seed=30)
        rc.add_session("s1", emb_a, SAMPLE_TAGS_A)
        rc.add_session("s2", emb_a, SAMPLE_TAGS_A)
        rc.add_session("s3", emb_c, SAMPLE_TAGS_C)
        rc.add_session("s4", emb_c, SAMPLE_TAGS_C)

        # Query embedding is right between them
        mid = ((np.array(emb_a) + np.array(emb_c)) / 2).tolist()
        monkeypatch.setattr(rc, "embed_text", lambda text: mid)

        result = rc.classify("曖昧なテスト")
        # With high threshold and equal representation, should not be confident
        assert result["confident"] is False


class TestIndexPersistence:
    def test_save_and_load(self):
        tmpdir = tempfile.mkdtemp()
        rc1 = _make_classifier(tmpdir, min_sessions=1)
        emb1 = _fake_embedding(768, seed=50)
        emb2 = _fake_embedding(768, seed=51)
        rc1.add_session("s1", emb1, SAMPLE_TAGS_A)
        rc1.add_session("s2", emb2, SAMPLE_TAGS_C)
        assert rc1.session_count() == 2

        # Load into a new instance
        rc2 = _make_classifier(tmpdir, min_sessions=1)
        assert rc2.session_count() == 0
        rc2.load_index()
        assert rc2.session_count() == 2

        # Verify search works
        results = rc2._search_similar(emb1, k=1)
        assert len(results) == 1
        assert results[0]["session_id"] == "s1"


class TestGetStatus:
    def test_status(self):
        tmpdir = tempfile.mkdtemp()
        rc = _make_classifier(tmpdir, min_sessions=5)
        status = rc.get_status()
        assert status["ready"] is False
        assert status["total_sessions"] == 0
        assert status["min_sessions"] == 5
        assert status["model"] == "gemini-embedding-001"
