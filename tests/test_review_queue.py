"""Tests for review queue priority ordering and statistics."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "omi-recordings-viewer"))

from indexer import Indexer


def _make_session(rec_dir, date, time, tags=None, segments=None, audio_scene=None):
    """Helper to create a session directory with transcript.json."""
    d = rec_dir / date / time
    d.mkdir(parents=True, exist_ok=True)
    session = {"start": f"{date}T{time.replace('-', ':')}", "duration_seconds": 60}
    if tags:
        session["tags"] = tags
        session["session_tagged"] = True
    if audio_scene:
        session["audio_scene"] = audio_scene
    segs = segments if segments is not None else [{"start": 0, "end": 5, "text": "hello"}]
    (d / "transcript.json").write_text(
        json.dumps({"session": session, "segments": segs}, ensure_ascii=False)
    )


@pytest.fixture
def tmp_env(tmp_path):
    rec_dir = tmp_path / "recordings"
    db_path = str(tmp_path / "test.db")
    return db_path, rec_dir


@pytest.mark.asyncio
async def test_review_queue_priority_ordering(tmp_env):
    """RAG low similarity should rank higher than LLM noisy, which ranks higher than RAG high similarity."""
    db_path, rec_dir = tmp_env

    # Session 1: RAG with very low similarity (highest model uncertainty)
    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "その他",
        "outcomes": ["情報共有"],
        "domains": [],
        "importance": "中",
        "tag_source": "rag",
        "rag_similarity": 0.15,
        "tag_version": 2,
    })

    # Session 2: LLM with noisy quality (medium priority)
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "業務会話",
        "outcomes": ["意思決定"],
        "domains": ["個人"],
        "importance": "中",
        "quality": "noisy",
        "tag_source": "llm",
        "tag_version": 2,
    })

    # Session 3: RAG with high similarity (lowest priority)
    _make_session(rec_dir, "2026-04-01", "12-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": ["情報共有"],
        "domains": [],
        "importance": "低",
        "tag_source": "rag",
        "rag_similarity": 0.85,
        "tag_version": 2,
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 3

    # First should be RAG low similarity (highest priority_score)
    assert queue[0]["time"] == "10-00-00", f"Expected RAG low sim first, got {queue[0]['time']}"
    # priority_score of first should be > second
    assert queue[0]["priority_score"] > queue[1]["priority_score"]
    # Last should be RAG high similarity
    assert queue[2]["time"] == "12-00-00", f"Expected RAG high sim last, got {queue[2]['time']}"

    await indexer.close()


@pytest.mark.asyncio
async def test_review_queue_excludes_reviewed(tmp_env):
    """Sessions with manually_corrected=1 should not appear in the queue."""
    db_path, rec_dir = tmp_env

    # Session 1: not reviewed
    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_source": "llm",
        "tag_version": 2,
    })

    # Session 2: already reviewed
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "業務会話",
        "outcomes": [],
        "importance": "中",
        "tag_source": "llm",
        "manually_corrected": True,
        "corrected_at": "2026-04-01T12:00:00",
        "tag_version": 2,
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 1
    assert queue[0]["time"] == "10-00-00"

    await indexer.close()


@pytest.mark.asyncio
async def test_review_queue_excludes_v1_and_untagged(tmp_env):
    """Sessions with tags_version < 2 should not appear in the queue."""
    db_path, rec_dir = tmp_env

    # Session 1: V2 tagged (should appear)
    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_version": 2,
    })

    # Session 2: V1 tagged (should NOT appear)
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "activity": "meeting",
        "tag_version": 1,
    })

    # Session 3: untagged (should NOT appear)
    _make_session(rec_dir, "2026-04-01", "12-00-00")

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 1
    assert queue[0]["time"] == "10-00-00"

    await indexer.close()


@pytest.mark.asyncio
async def test_review_queue_excludes_empty_sessions(tmp_env):
    """Sessions with segment_count=0 should not appear in the queue."""
    db_path, rec_dir = tmp_env

    # Session 1: has segments
    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_version": 2,
    })

    # Session 2: no segments
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_version": 2,
    }, segments=[])

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 1
    assert queue[0]["time"] == "10-00-00"

    await indexer.close()


@pytest.mark.asyncio
async def test_review_stats(tmp_env):
    """Review stats should correctly count pending/reviewed/today."""
    db_path, rec_dir = tmp_env

    # 2 pending, 1 reviewed (today)
    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話", "tag_version": 2,
    })
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "業務会話", "tag_version": 2,
    })
    _make_session(rec_dir, "2026-04-01", "12-00-00", tags={
        "conversation_type": "日常会話",
        "manually_corrected": True,
        "corrected_at": "2099-01-01T00:00:00",
        "tag_version": 2,
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    stats = await indexer.get_review_stats()
    assert stats["total_pending"] == 2
    assert stats["total_reviewed"] == 1
    # reviewed_today depends on actual date, so just check type
    assert isinstance(stats["reviewed_today"], int)

    await indexer.close()


@pytest.mark.asyncio
async def test_review_queue_offset(tmp_env):
    """Offset should skip the first N items."""
    db_path, rec_dir = tmp_env

    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "その他", "tag_version": 2,
        "tag_source": "rag", "rag_similarity": 0.3,
    })
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "日常会話", "tag_version": 2,
        "tag_source": "llm",
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    # offset=0 should return highest priority
    q0 = await indexer.get_review_queue(limit=1, offset=0)
    assert len(q0) == 1

    # offset=1 should return the other one
    q1 = await indexer.get_review_queue(limit=1, offset=1)
    assert len(q1) == 1
    assert q1[0]["time"] != q0[0]["time"]

    # offset=2 should return nothing
    q2 = await indexer.get_review_queue(limit=1, offset=2)
    assert len(q2) == 0

    await indexer.close()


@pytest.mark.asyncio
async def test_new_columns_in_parse_transcript(tmp_env):
    """Verify that _parse_transcript correctly extracts new review columns."""
    db_path, rec_dir = tmp_env

    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "tag_source": "rag",
        "rag_similarity": 0.72,
        "manually_corrected": True,
        "corrected_at": "2026-04-01T10:00:00",
        "tag_version": 2,
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    s = await indexer.get_session("2026-04-01", "10-00-00")
    assert s["tags_tag_source"] == "rag"
    assert abs(s["tags_rag_similarity"] - 0.72) < 0.001
    assert s["tags_manually_corrected"] == 1
    assert s["tags_corrected_at"] == "2026-04-01T10:00:00"

    await indexer.close()


# ============================================================
# New tests for improved scoring
# ============================================================


@pytest.mark.asyncio
async def test_continuous_rag_similarity(tmp_env):
    """RAG similarity should produce strictly monotonically decreasing priority scores."""
    db_path, rec_dir = tmp_env

    for i, sim in enumerate([0.10, 0.40, 0.60, 0.90]):
        time_str = f"1{i}-00-00"
        _make_session(rec_dir, "2026-04-01", time_str, tags={
            "conversation_type": "日常会話",
            "outcomes": [],
            "importance": "低",
            "tag_source": "rag",
            "rag_similarity": sim,
            "tag_version": 2,
        })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 4

    scores = [item["priority_score"] for item in queue]
    # Should be strictly decreasing (sorted by priority DESC)
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1], f"Score {scores[i]} should be > {scores[i+1]}"

    # Verify the order: lowest similarity first
    assert queue[0]["time"] == "10-00-00"  # sim 0.10
    assert queue[-1]["time"] == "13-00-00"  # sim 0.90

    await indexer.close()


@pytest.mark.asyncio
async def test_importance_affects_priority(tmp_env):
    """importance_v2=高 should produce higher priority than importance_v2=低."""
    db_path, rec_dir = tmp_env

    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "高",
        "tag_source": "llm",
        "quality": "clean",
        "tag_version": 2,
    })
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_source": "llm",
        "quality": "clean",
        "tag_version": 2,
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 2
    assert queue[0]["time"] == "10-00-00", "高 importance should rank higher"
    assert queue[0]["priority_score"] > queue[1]["priority_score"]

    await indexer.close()


@pytest.mark.asyncio
async def test_external_audio_bonus(tmp_env):
    """Sessions with external_audio >= 0.5 should get higher priority."""
    db_path, rec_dir = tmp_env

    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_source": "llm",
        "quality": "clean",
        "external_audio_ratio": 0.8,
        "tag_version": 2,
    })
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_source": "llm",
        "quality": "clean",
        "external_audio_ratio": 0.0,
        "tag_version": 2,
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 2
    assert queue[0]["time"] == "10-00-00", "High external audio should rank higher"
    assert queue[0]["priority_score"] > queue[1]["priority_score"]

    await indexer.close()


@pytest.mark.asyncio
async def test_interaction_term(tmp_env):
    """RAG low similarity + noisy quality should get interaction bonus >= 0.08."""
    db_path, rec_dir = tmp_env

    # With interaction: RAG low sim + noisy
    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_source": "rag",
        "rag_similarity": 0.3,
        "quality": "noisy",
        "tag_version": 2,
    })

    # Without interaction: RAG low sim only (no noisy)
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_source": "rag",
        "rag_similarity": 0.3,
        "quality": "clean",
        "tag_version": 2,
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 2

    with_interaction = next(q for q in queue if q["time"] == "10-00-00")
    without_interaction = next(q for q in queue if q["time"] == "11-00-00")

    bonus = with_interaction["priority_score"] - without_interaction["priority_score"]
    assert bonus >= 0.08, f"Interaction bonus should be >= 0.08, got {bonus:.4f}"

    await indexer.close()


@pytest.mark.asyncio
async def test_segment_count_complexity(tmp_env):
    """Sessions with more segments should have higher priority."""
    db_path, rec_dir = tmp_env

    # 30 segments
    segs_30 = [{"start": i, "end": i + 1, "text": f"seg {i}"} for i in range(30)]
    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_source": "llm",
        "quality": "clean",
        "tag_version": 2,
    }, segments=segs_30)

    # 1 segment
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_source": "llm",
        "quality": "clean",
        "tag_version": 2,
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 2
    assert queue[0]["time"] == "10-00-00", "More segments should rank higher"
    assert queue[0]["priority_score"] > queue[1]["priority_score"]

    await indexer.close()


@pytest.mark.asyncio
async def test_feedback_weight_adjustment(tmp_env):
    """After repeated corrections, model_uncertainty weight should increase and sum to 1.0."""
    db_path, rec_dir = tmp_env

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()

    # Simulate 10 corrections for rag_low_sim sessions
    for i in range(10):
        session = {
            "tags_tag_source": "rag",
            "tags_rag_similarity": 0.3,
            "tags_quality": "clean",
            "tags_conversation_type": "日常会話",
            "tags_external_audio": 0.0,
            "tags_importance_v2": "低",
        }
        indexer.record_review_feedback(session, was_corrected=True)

    # Load saved weights
    weights_path = rec_dir.parent / "config" / "review_weights.json"
    assert weights_path.exists(), "review_weights.json should be created"

    data = json.loads(weights_path.read_text(encoding="utf-8"))
    adjusted = data["adjusted_weights"]

    # model_uncertainty should be boosted above default
    assert adjusted["model_uncertainty"] > Indexer._DEFAULT_REVIEW_WEIGHTS["model_uncertainty"], \
        f"Expected model_uncertainty > {Indexer._DEFAULT_REVIEW_WEIGHTS['model_uncertainty']}, got {adjusted['model_uncertainty']}"

    # Sum should be ~1.0
    total = sum(adjusted.values())
    assert abs(total - 1.0) < 0.001, f"Weights should sum to 1.0, got {total}"

    await indexer.close()


def test_classify_session_signals():
    """Test signal bucket classification accuracy."""
    # RAG low similarity
    signals = Indexer.classify_session_signals({
        "tags_tag_source": "rag",
        "tags_rag_similarity": 0.3,
        "tags_quality": "clean",
        "tags_conversation_type": "その他",
        "tags_external_audio": 0.8,
        "tags_importance_v2": "高",
    })
    assert "rag_low_sim" in signals
    assert "sono_ta" in signals
    assert "high_external_audio" in signals
    assert "high_importance" in signals

    # RAG high similarity
    signals = Indexer.classify_session_signals({
        "tags_tag_source": "rag",
        "tags_rag_similarity": 0.7,
        "tags_quality": "clean",
        "tags_conversation_type": "日常会話",
        "tags_external_audio": 0.0,
        "tags_importance_v2": "低",
    })
    assert "rag_high_sim" in signals
    assert "sono_ta" not in signals
    assert "high_external_audio" not in signals
    assert "high_importance" not in signals

    # LLM noisy
    signals = Indexer.classify_session_signals({
        "tags_tag_source": "llm",
        "tags_rag_similarity": None,
        "tags_quality": "noisy",
        "tags_conversation_type": "業務会話",
        "tags_external_audio": 0.0,
        "tags_importance_v2": "中",
    })
    assert "llm_noisy" in signals
    assert "rag_low_sim" not in signals

    # LLM clean
    signals = Indexer.classify_session_signals({
        "tags_tag_source": "llm",
        "tags_rag_similarity": None,
        "tags_quality": "clean",
        "tags_conversation_type": "日常会話",
        "tags_external_audio": 0.0,
        "tags_importance_v2": "低",
    })
    assert "llm_clean" in signals
