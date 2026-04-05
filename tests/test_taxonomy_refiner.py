"""Tests for taxonomy_refiner.py."""

import json
import math
import os
import shutil
import tempfile

import numpy as np
import pytest

from taxonomy_refiner import TaxonomyRefiner, _extract_plain_text


@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temporary recordings and config directories."""
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    config = tmp_path / "config"
    config.mkdir()
    return str(recordings), str(config)


def _make_session(recordings_dir, day, time_str, conv_type, domains=None,
                  text="テストテキスト。これはテスト用のセッションです。"):
    """Create a mock session directory with transcript."""
    day_dir = os.path.join(recordings_dir, day)
    time_dir = os.path.join(day_dir, time_str)
    os.makedirs(time_dir, exist_ok=True)

    transcript = {
        "session": {
            "session_tagged": True,
            "tags": {
                "conversation_type": conv_type,
                "outcomes": ["情報共有"],
                "domains": domains or ["技術"],
                "importance": "中",
            },
        },
        "segments": [
            {"text": text, "start": 0, "end": 5},
        ],
    }
    with open(os.path.join(time_dir, "transcript.json"), "w") as f:
        json.dump(transcript, f)


class FakeRagClassifier:
    """Minimal mock of RagClassifier for testing."""

    def __init__(self, embeddings=None, metadata=None):
        import threading
        self._lock = threading.Lock()
        self._embeddings = embeddings
        self._metadata = metadata or []

    def embed_text(self, text):
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(768).tolist()

    def embed_texts_batch(self, texts):
        return [self.embed_text(t) for t in texts]


class TestExtractPlainText:
    def test_basic(self):
        segments = [{"text": "Hello"}, {"text": "World"}, {"text": ""}]
        assert _extract_plain_text(segments) == "Hello\nWorld"

    def test_empty(self):
        assert _extract_plain_text([]) == ""


class TestStep1StratifiedSampling:
    def test_stratified(self, tmp_dirs):
        rec_dir, cfg_dir = tmp_dirs
        # Create sessions of different types
        for i in range(10):
            _make_session(rec_dir, "2026-04-01", f"10-00-{i:02d}", "会議")
        for i in range(6):
            _make_session(rec_dir, "2026-04-01", f"11-00-{i:02d}", "雑談")

        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None, sample_ratio=0.5,
        )
        samples = refiner._step1_sample()

        # Should sample from both types
        types = [s["current_tags"]["conversation_type"] for s in samples]
        assert "会議" in types
        assert "雑談" in types

        # ~50% of each: ceil(10*0.5)=5, ceil(6*0.5)=3
        meeting_count = types.count("会議")
        chat_count = types.count("雑談")
        assert meeting_count == 5
        assert chat_count == 3

    def test_empty_recordings(self, tmp_dirs):
        rec_dir, cfg_dir = tmp_dirs
        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )
        samples = refiner._step1_sample()
        assert samples == []


class TestKmeans:
    def test_basic_two_clusters(self):
        """Two well-separated groups should be correctly clustered."""
        rng = np.random.RandomState(42)
        group_a = rng.randn(10, 8) + np.array([5, 0, 0, 0, 0, 0, 0, 0])
        group_b = rng.randn(10, 8) + np.array([-5, 0, 0, 0, 0, 0, 0, 0])
        embeddings = np.vstack([group_a, group_b]).astype(np.float32)

        labels = TaxonomyRefiner._kmeans(embeddings, k=2)
        assert len(labels) == 20

        # All of group_a should have same label, all of group_b should have same label
        assert len(set(labels[:10])) == 1
        assert len(set(labels[10:])) == 1
        assert labels[0] != labels[10]

    def test_single_cluster(self):
        embeddings = np.random.randn(5, 4).astype(np.float32)
        labels = TaxonomyRefiner._kmeans(embeddings, k=1)
        assert len(labels) == 5
        assert all(l == 0 for l in labels)

    def test_empty(self):
        labels = TaxonomyRefiner._kmeans(np.array([]), k=3)
        assert len(labels) == 0

    def test_k_larger_than_n(self):
        embeddings = np.random.randn(3, 4).astype(np.float32)
        labels = TaxonomyRefiner._kmeans(embeddings, k=10)
        assert len(labels) == 3


class TestGetEmbeddingsFromIndex:
    def test_reuses_rag_embeddings(self, tmp_dirs):
        rec_dir, cfg_dir = tmp_dirs
        # Create fake RAG index
        embeddings = np.random.randn(2, 768).astype(np.float32)
        metadata = [
            {"session_id": "2026-04-01/10-00-00", "tags": {}},
            {"session_id": "2026-04-01/10-00-01", "tags": {}},
        ]
        rag = FakeRagClassifier(embeddings=embeddings, metadata=metadata)

        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=rag,
        )

        samples = [
            {"session_id": "2026-04-01/10-00-00", "text": "test 1", "current_tags": {}},
            {"session_id": "2026-04-01/10-00-01", "text": "test 2", "current_tags": {}},
            {"session_id": "2026-04-01/10-00-02", "text": "test 3 new text", "current_tags": {}},
        ]

        result = refiner._get_embeddings_for_samples(samples)
        assert result.shape == (3, 768)
        # First two should match RAG index
        np.testing.assert_array_equal(result[0], embeddings[0])
        np.testing.assert_array_equal(result[1], embeddings[1])
        # Third should be newly embedded (not zero)
        assert np.linalg.norm(result[2]) > 0


class TestClusterCoherence:
    def test_high_coherence(self, tmp_dirs):
        rec_dir, cfg_dir = tmp_dirs
        # Create identical sessions => high coherence
        rag = FakeRagClassifier()

        # Override embed to return near-identical vectors
        original_embed = rag.embed_text
        def fixed_embed(text):
            return [1.0] * 768
        rag.embed_text = fixed_embed
        rag.embed_texts_batch = lambda texts: [fixed_embed(t) for t in texts]

        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=rag,
        )

        samples = [
            {"session_id": f"2026-04-01/10-00-{i:02d}", "text": "same text",
             "current_tags": {"conversation_type": "会議", "domains": ["技術"]}}
            for i in range(5)
        ]

        clusters = refiner._step2_cluster(samples)
        # With identical embeddings, all should be in one cluster with coherence ~1.0
        assert len(clusters) >= 1
        # The cluster containing all items should have high coherence
        main_cluster = max(clusters, key=lambda c: c["size"])
        assert main_cluster["coherence_score"] >= 0.99


class TestProposeNoChange:
    def test_no_change_response(self, tmp_dirs):
        rec_dir, cfg_dir = tmp_dirs
        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )

        taxonomy = {"conversation_type": ["会議", "雑談"], "domains": ["技術"]}

        # Mock direct API call (Step 3 uses generate_content, not Batch)
        refiner._generate_content = lambda *a, **kw: \
            '{"no_change": true, "changes": [], "proposed_taxonomy": {}}'

        result = refiner._step3_propose([{"cluster_id": 0, "size": 5,
                                           "representative_texts": ["test"],
                                           "current_domains": {},
                                           "current_types": {},
                                           "coherence_score": 0.8}],
                                          taxonomy)
        assert result["no_change"] is True


class TestValidateScoring:
    def test_scoring(self, tmp_dirs):
        rec_dir, cfg_dir = tmp_dirs
        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )

        samples = [
            {"session_id": "s1", "text": "test",
             "current_tags": {"conversation_type": "会議", "domains": ["技術"]}},
            {"session_id": "s2", "text": "test",
             "current_tags": {"conversation_type": "雑談", "domains": ["生活"]}},
        ]

        proposed_taxonomy = {
            "conversation_type": ["会議", "雑談", "報告"],
            "domains": ["技術", "生活", "健康"],
        }

        # Simulate responses: one matching, one with new classification
        responses = [
            '{"conversation_type": "会議", "domains": ["技術"]}',
            '{"conversation_type": "報告", "domains": ["健康"]}',
        ]

        result = refiner._score_validation(samples, responses, proposed_taxonomy)
        assert result["total_validated"] == 2
        assert result["per_axis_scores"]["conversation_type"] == 1.0
        # Both have valid types in proposed taxonomy
        assert result["improved_count"] + result["unchanged_count"] + result["degraded_count"] == 2


class TestApproveUpdatesTaxonomy:
    def test_approve(self, tmp_dirs):
        rec_dir, cfg_dir = tmp_dirs
        taxonomy_path = os.path.join(cfg_dir, "tag_taxonomy.json")

        # Create initial taxonomy
        initial = {
            "conversation_type": ["会議", "雑談"],
            "outcomes": ["情報共有"],
            "domains": ["技術"],
            "keyword_anchors": {},
            "domain_descriptions": {},
        }
        with open(taxonomy_path, "w") as f:
            json.dump(initial, f)

        # Create pending proposal
        pending_path = os.path.join(cfg_dir, "pending_taxonomy.json")
        pending = {
            "created_at": "2026-04-01T00:00:00",
            "proposed_taxonomy": {
                "conversation_type": ["会議", "雑談", "報告"],
                "outcomes": ["情報共有", "意思決定"],
                "domains": ["技術", "健康"],
                "keyword_anchors": {"健康": ["病院", "薬"]},
                "domain_descriptions": {"健康": "健康関連の話題"},
            },
            "changes": [{"type": "new", "axis": "domains", "before": [], "after": ["健康"]}],
            "validation": {"accuracy_vs_current": 0.95},
        }
        with open(pending_path, "w") as f:
            json.dump(pending, f)

        # Create report
        reports_dir = os.path.join(cfg_dir, "taxonomy_reports")
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, "2026-04.json"), "w") as f:
            json.dump({"status": "pending_approval"}, f)

        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )

        result = refiner.approve_proposal()
        assert result["status"] == "approved"
        assert result["accepted"] == 1

        # Verify taxonomy was updated
        with open(taxonomy_path) as f:
            updated = json.load(f)
        assert "報告" in updated["conversation_type"]
        assert "健康" in updated["domains"]
        assert updated["keyword_anchors"]["健康"] == ["病院", "薬"]

        # Verify pending was removed
        assert not os.path.exists(pending_path)

        # Verify report status updated
        with open(os.path.join(reports_dir, "2026-04.json")) as f:
            report = json.load(f)
        assert report["status"] == "approved"


class TestSelectiveApproval:
    """Test approving individual changes from a proposal."""

    def test_partial_approve(self, tmp_dirs):
        """Approve one change, reject another."""
        rec_dir, cfg_dir = tmp_dirs
        taxonomy_path = os.path.join(cfg_dir, "tag_taxonomy.json")

        initial = {
            "conversation_type": ["会議", "雑談"],
            "outcomes": ["情報共有"],
            "domains": ["技術"],
            "keyword_anchors": {},
            "domain_descriptions": {},
        }
        with open(taxonomy_path, "w") as f:
            json.dump(initial, f)

        pending_path = os.path.join(cfg_dir, "pending_taxonomy.json")
        pending = {
            "created_at": "2026-04-01T00:00:00",
            "proposed_taxonomy": {
                "conversation_type": ["会議", "雑談", "報告"],
                "outcomes": ["情報共有", "意思決定"],
                "domains": ["技術", "健康"],
                "keyword_anchors": {"健康": ["病院"]},
                "domain_descriptions": {"健康": "健康関連"},
            },
            "changes": [
                {"type": "new", "axis": "conversation_type",
                 "description": "報告を追加", "before": [], "after": ["報告"]},
                {"type": "new", "axis": "domains",
                 "description": "健康を追加", "before": [], "after": ["健康"]},
            ],
            "validation": {"accuracy_vs_current": 0.95},
        }
        with open(pending_path, "w") as f:
            json.dump(pending, f)

        reports_dir = os.path.join(cfg_dir, "taxonomy_reports")
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, "2026-04.json"), "w") as f:
            json.dump({"status": "pending_approval"}, f)

        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )

        # Accept only change 0 (報告), reject change 1 (健康)
        result = refiner.approve_proposal(accepted_indices=[0])
        assert result["status"] == "partial"
        assert result["accepted"] == 1
        assert result["rejected"] == 1

        with open(taxonomy_path) as f:
            updated = json.load(f)

        # "報告" should be added (accepted)
        assert "報告" in updated["conversation_type"]
        # "健康" should NOT be added (rejected/reverted)
        assert "健康" not in updated["domains"]
        # keyword_anchors for 健康 should not exist
        assert "健康" not in updated.get("keyword_anchors", {})

        assert not os.path.exists(pending_path)

    def test_reject_all_via_empty_indices(self, tmp_dirs):
        """Passing empty accepted_indices should revert all changes."""
        rec_dir, cfg_dir = tmp_dirs
        taxonomy_path = os.path.join(cfg_dir, "tag_taxonomy.json")

        initial = {
            "conversation_type": ["会議"],
            "outcomes": [],
            "domains": ["技術"],
            "keyword_anchors": {},
            "domain_descriptions": {},
        }
        with open(taxonomy_path, "w") as f:
            json.dump(initial, f)

        pending_path = os.path.join(cfg_dir, "pending_taxonomy.json")
        pending = {
            "created_at": "2026-04-01T00:00:00",
            "proposed_taxonomy": {
                "conversation_type": ["会議", "雑談"],
                "outcomes": [],
                "domains": ["技術", "AI"],
                "keyword_anchors": {"AI": ["GPT"]},
                "domain_descriptions": {"AI": "AI関連"},
            },
            "changes": [
                {"type": "new", "axis": "conversation_type",
                 "before": [], "after": ["雑談"]},
                {"type": "new", "axis": "domains",
                 "before": [], "after": ["AI"]},
            ],
            "validation": {},
        }
        with open(pending_path, "w") as f:
            json.dump(pending, f)

        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )

        result = refiner.approve_proposal(accepted_indices=[])
        assert result["accepted"] == 0
        assert result["rejected"] == 2

        with open(taxonomy_path) as f:
            updated = json.load(f)
        assert "雑談" not in updated["conversation_type"]
        assert "AI" not in updated["domains"]

    def test_revert_merge(self, tmp_dirs):
        """Rejecting a merge change should restore the original items."""
        rec_dir, cfg_dir = tmp_dirs
        taxonomy_path = os.path.join(cfg_dir, "tag_taxonomy.json")

        initial = {
            "conversation_type": ["会議", "ミーティング"],
            "outcomes": [],
            "domains": [],
            "keyword_anchors": {},
            "domain_descriptions": {},
        }
        with open(taxonomy_path, "w") as f:
            json.dump(initial, f)

        pending_path = os.path.join(cfg_dir, "pending_taxonomy.json")
        pending = {
            "created_at": "2026-04-01T00:00:00",
            "proposed_taxonomy": {
                "conversation_type": ["社内ミーティング"],
                "outcomes": [],
                "domains": [],
                "keyword_anchors": {},
                "domain_descriptions": {},
            },
            "changes": [
                {"type": "merge", "axis": "conversation_type",
                 "description": "会議とミーティングを統合",
                 "before": ["会議", "ミーティング"],
                 "after": ["社内ミーティング"]},
            ],
            "validation": {},
        }
        with open(pending_path, "w") as f:
            json.dump(pending, f)

        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )

        # Reject the merge
        result = refiner.approve_proposal(accepted_indices=[])

        with open(taxonomy_path) as f:
            updated = json.load(f)
        # Original items should be restored
        assert "会議" in updated["conversation_type"]
        assert "ミーティング" in updated["conversation_type"]
        assert "社内ミーティング" not in updated["conversation_type"]

    def test_revert_remove(self, tmp_dirs):
        """Rejecting a remove change should restore the removed item."""
        rec_dir, cfg_dir = tmp_dirs
        taxonomy_path = os.path.join(cfg_dir, "tag_taxonomy.json")

        initial = {
            "conversation_type": ["会議", "雑談"],
            "outcomes": [],
            "domains": [],
            "keyword_anchors": {},
            "domain_descriptions": {},
        }
        with open(taxonomy_path, "w") as f:
            json.dump(initial, f)

        pending_path = os.path.join(cfg_dir, "pending_taxonomy.json")
        pending = {
            "created_at": "2026-04-01T00:00:00",
            "proposed_taxonomy": {
                "conversation_type": ["会議"],
                "outcomes": [],
                "domains": [],
                "keyword_anchors": {},
                "domain_descriptions": {},
            },
            "changes": [
                {"type": "remove", "axis": "conversation_type",
                 "description": "雑談を削除",
                 "before": ["雑談"], "after": []},
            ],
            "validation": {},
        }
        with open(pending_path, "w") as f:
            json.dump(pending, f)

        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )

        # Reject the remove → 雑談 should be restored
        result = refiner.approve_proposal(accepted_indices=[])

        with open(taxonomy_path) as f:
            updated = json.load(f)
        assert "雑談" in updated["conversation_type"]


class TestRejectCleansPending:
    def test_reject(self, tmp_dirs):
        rec_dir, cfg_dir = tmp_dirs
        pending_path = os.path.join(cfg_dir, "pending_taxonomy.json")

        # Create pending
        with open(pending_path, "w") as f:
            json.dump({"proposed_taxonomy": {}}, f)

        # Create report
        reports_dir = os.path.join(cfg_dir, "taxonomy_reports")
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, "2026-04.json"), "w") as f:
            json.dump({"status": "pending_approval"}, f)

        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )

        result = refiner.reject_proposal()
        assert result["status"] == "rejected"
        assert not os.path.exists(pending_path)

        # Verify report status
        with open(os.path.join(reports_dir, "2026-04.json")) as f:
            report = json.load(f)
        assert report["status"] == "rejected"


class TestReportPersistence:
    def test_save_and_get_reports(self, tmp_dirs):
        rec_dir, cfg_dir = tmp_dirs
        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )

        report = {
            "started_at": "2026-04-01T00:00:00",
            "finished_at": "2026-04-01T00:05:00",
            "status": "no_change",
            "sample_count": 50,
        }
        refiner._save_report(report)

        reports = refiner.get_reports()
        assert len(reports) >= 1
        assert reports[0]["status"] == "no_change"
        assert "filename" in reports[0]

    def test_get_status(self, tmp_dirs):
        rec_dir, cfg_dir = tmp_dirs
        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )

        status = refiner.get_status()
        assert status["running"] is False
        assert status["has_pending"] is False


class TestParseProposalJson:
    def test_markdown_codeblock(self, tmp_dirs):
        rec_dir, cfg_dir = tmp_dirs
        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )

        text = '```json\n{"no_change": false, "changes": [{"type": "new"}], "proposed_taxonomy": {"conversation_type": ["A"]}}\n```'
        result = refiner._parse_proposal_json(text, {})
        assert result["no_change"] is False
        assert len(result["changes"]) == 1

    def test_invalid_json(self, tmp_dirs):
        rec_dir, cfg_dir = tmp_dirs
        refiner = TaxonomyRefiner(
            api_key="fake", recordings_dir=rec_dir, config_dir=cfg_dir,
            rag_classifier=None,
        )

        result = refiner._parse_proposal_json("not json", {"fallback": True})
        assert result["no_change"] is True
