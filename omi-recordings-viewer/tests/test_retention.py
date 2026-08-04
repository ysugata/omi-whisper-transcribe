"""Tests for retention-aware behavior: files_present tracking and
export fallback for sessions whose files were deleted by retention cleanup."""

import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exporter import export_session, transcript_stub_from_row
from indexer import Indexer


@pytest.fixture
def tmp_env(tmp_path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    db_path = str(tmp_path / "index.db")
    return recordings, db_path


def _create_session(recordings, date, time_str, tags=None):
    session_dir = recordings / date / time_str
    session_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "session": {
            "start": f"{date}T{time_str.replace('-', ':')}",
            "duration_seconds": 30.0,
            "model": "large-v3",
            "language": "ja",
        },
        "segments": [{"text": "hello world", "start": 0, "end": 1}],
    }
    if tags:
        data["session"]["tags"] = tags
    (session_dir / "transcript.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


V2_TAGS = {
    "conversation_type": "商談・打合せ",
    "outcomes": ["意思決定あり"],
    "domains": ["製造"],
    "importance": "高",
    "quality": "clear",
    "summary": "テスト用サマリー",
    "tag_version": 2,
}


@pytest.mark.asyncio
async def test_files_present_marked_on_deletion(tmp_env):
    """Deleting a session dir marks files_present=0 on next sync."""
    recordings, db_path = tmp_env
    _create_session(recordings, "2026-03-29", "10-00-00")
    _create_session(recordings, "2026-03-29", "11-00-00")

    indexer = Indexer(db_path, str(recordings))
    await indexer.open()
    await indexer.sync()

    shutil.rmtree(recordings / "2026-03-29" / "10-00-00")
    await indexer.sync()

    s1 = await indexer.get_session("2026-03-29", "10-00-00")
    s2 = await indexer.get_session("2026-03-29", "11-00-00")
    assert s1["files_present"] == 0
    assert s2["files_present"] == 1
    await indexer.close()


@pytest.mark.asyncio
async def test_files_present_restored_on_reappearance(tmp_env):
    """A session dir restored to disk flips files_present back to 1."""
    recordings, db_path = tmp_env
    _create_session(recordings, "2026-03-29", "10-00-00")

    indexer = Indexer(db_path, str(recordings))
    await indexer.open()
    await indexer.sync()

    shutil.rmtree(recordings / "2026-03-29" / "10-00-00")
    await indexer.sync()
    assert (await indexer.get_session("2026-03-29", "10-00-00"))["files_present"] == 0

    _create_session(recordings, "2026-03-29", "10-00-00")
    await indexer.sync()
    assert (await indexer.get_session("2026-03-29", "10-00-00"))["files_present"] == 1
    await indexer.close()


@pytest.mark.asyncio
async def test_review_queue_excludes_deleted_sessions(tmp_env):
    """Sessions without files must not enter the review queue (accept would 404)."""
    recordings, db_path = tmp_env
    _create_session(recordings, "2026-03-29", "10-00-00", tags=V2_TAGS)
    _create_session(recordings, "2026-03-29", "11-00-00", tags=V2_TAGS)

    indexer = Indexer(db_path, str(recordings))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10)
    assert len(queue) == 2

    shutil.rmtree(recordings / "2026-03-29" / "10-00-00")
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10)
    assert [(q["date"], q["time"]) for q in queue] == [("2026-03-29", "11-00-00")]

    stats = await indexer.get_review_stats()
    assert stats["total_pending"] == 1
    await indexer.close()


def test_transcript_stub_from_row_rebuilds_tags():
    row = {
        "start_at": "2026-03-29T10:00:00",
        "end_at": "2026-03-29T10:05:00",
        "duration_seconds": 300.0,
        "language": "ja",
        "tags_conversation_type": "商談・打合せ",
        "tags_outcomes": "意思決定あり,TODO発生",
        "tags_domains": "製造,営業",
        "tags_importance_v2": "高",
        "tags_quality": "clear",
        "tags_summary": "サマリー",
        "tags_decision_memo": None,
    }
    stub = transcript_stub_from_row(row)
    tags = stub["session"]["tags"]
    assert tags["conversation_type"] == "商談・打合せ"
    assert tags["outcomes"] == ["意思決定あり", "TODO発生"]
    assert tags["domains"] == ["製造", "営業"]
    assert tags["importance"] == "高"
    assert "decision_memo" not in tags
    assert stub["segments"] == []


def test_openai_export_works_with_stub_and_sanitized():
    """KC-format export must produce records from sanitized data even when
    the transcript is a DB-derived stub (file deleted by retention)."""
    row = {
        "start_at": "2026-03-29T10:00:00",
        "duration_seconds": 300.0,
        "language": "ja",
        "tags_conversation_type": "商談・打合せ",
        "tags_domains": "製造",
        "tags_importance_v2": "高",
    }
    sanitized = {
        "session_id": "2026-03-29/10-00-00",
        "summary_input": None,
        "summary_output": "経営者は新しい製造ラインの投資判断について議論し、導入を決定した。",
        "keypoints": ["投資額を確認", "導入時期を決定"],
        "safe_conversation": [
            {"speaker": "user", "text": "新ラインの投資はどう考えるべきか。" * 5},
            {"speaker": "assistant", "text": "回収期間と稼働率次第だと思います。" * 5},
        ],
        "knowledge_cards": [{
            "question": "製造ライン投資の判断基準は何ですか？",
            "answer": "回収期間が3年以内で、稼働率80%以上を見込めることを基準とした。",
            "evidence": [],
            "confidence": "high",
        }],
        "risk": "low",
    }
    stub = transcript_stub_from_row(row)
    lines = export_session(
        "openai", "2026-03-29/10-00-00", stub, "user_vs_others",
        include_monologue=False, sanitized=sanitized,
    )
    assert len(lines) == 2  # 1 KC record + 1 summarization record
    rec = json.loads(lines[0])
    assert rec["messages"][0]["role"] == "system"
    assert "製造" in rec["messages"][0]["content"]

    # Without sanitized data the stub alone must yield nothing
    assert export_session(
        "openai", "2026-03-29/10-00-00", stub, "user_vs_others",
        include_monologue=False, sanitized=None,
    ) == []


def test_decision_log_kc_only_no_crash():
    """KC-only sanitized bucket (summary_output=None) must not TypeError."""
    from exporter import to_decision_log
    sanitized = {
        "session_id": "2026-03-29/10-00-00",
        "summary_input": None,
        "summary_output": None,
        "keypoints": None,
        "safe_conversation": None,
        "knowledge_cards": [{
            "question": "仕入先の選定基準は？",
            "answer": "品質と納期を最優先に、価格は二番目とする。",
            "evidence": [],
            "confidence": "high",
        }],
        "risk": "low",
    }
    transcript = {"session": {"tags": {}}, "segments": []}
    rec = to_decision_log("2026-03-29/10-00-00", transcript, sanitized=sanitized)
    assert rec is not None
    assert rec["title"] == "Decision"


def test_sanitized_index_dedups_training_sanitizer_kcs(tmp_path):
    """Same KC in two safe_training files (overlapping runs) must appear once."""
    from exporter import SanitizedIndex
    rec = {
        "type": "knowledge_card",
        "input": "テスト質問はこれですか？",
        "output": "はい、テスト回答はこれです。",
        "session_id": "2026-03-29/10-00-00",
        "risk_level": "low",
    }
    for name in ("safe_training_20260101_000000.jsonl",
                 "safe_training_20260102_000000.jsonl"):
        (tmp_path / name).write_text(
            json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    idx = SanitizedIndex(tmp_path)
    bucket = idx.get("2026-03-29/10-00-00")
    assert len(bucket["knowledge_cards"]) == 1


@pytest.mark.asyncio
async def test_export_sessions_require_files(tmp_env):
    """require_files=True (comprehensive) excludes retention-deleted sessions."""
    recordings, db_path = tmp_env
    _create_session(recordings, "2026-03-29", "10-00-00")
    _create_session(recordings, "2026-03-29", "11-00-00")

    indexer = Indexer(db_path, str(recordings))
    await indexer.open()
    await indexer.sync()

    shutil.rmtree(recordings / "2026-03-29" / "10-00-00")
    await indexer.sync()

    all_pairs = await indexer.get_export_sessions(min_segments=1)
    file_pairs = await indexer.get_export_sessions(min_segments=1, require_files=True)
    assert len(all_pairs) == 2
    assert file_pairs == [("2026-03-29", "11-00-00")]

    stats_all = await indexer.get_export_stats(min_segments=1)
    stats_files = await indexer.get_export_stats(min_segments=1, require_files=True)
    assert stats_all["sessions"] == 2
    assert stats_files["sessions"] == 1
    await indexer.close()


def test_kc_quality_filter_in_export(monkeypatch):
    """quality < KC_MIN_QUALITY のカードは学習データから除外される。"""
    import exporter as ex
    monkeypatch.setattr(ex, "KC_MIN_QUALITY", 3)
    sanitized = {
        "session_id": "2026-03-29/10-00-00",
        "summary_input": None, "summary_output": None, "keypoints": None,
        "safe_conversation": None, "risk": "low",
        "knowledge_cards": [
            {"question": "高品質な質問はこれですか？", "answer": "はい、具体的で有用な回答です。", "quality": 4},
            {"question": "低品質な質問はこれですか？", "answer": "曖昧で一般論的な回答です。", "quality": 2},
            {"question": "未採点の質問はこれですか？", "answer": "採点がなくても通過する具体的で長い回答です。"},
        ],
    }
    transcript = {"session": {"tags": {"domains": ["経営"]}}, "segments": []}
    lines = export_session("openai", "2026-03-29/10-00-00", transcript,
                           sanitized=sanitized)
    joined = "\n".join(lines)
    assert "高品質な質問" in joined
    assert "低品質な質問" not in joined
    assert "未採点の質問" in joined  # 未採点は通す


def test_sanitized_index_carries_quality(tmp_path):
    from exporter import SanitizedIndex
    rec = {
        "session_id": "a/b", "risk": "low", "summary": {}, "keypoints": [],
        "safe_conversation": [],
        "knowledge_cards": [{"question": "Q1です", "answer": "A1です", "quality": 5}],
    }
    (tmp_path / "safe_training_20260101_000000.jsonl").write_text(
        json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    idx = SanitizedIndex(tmp_path)
    assert idx.get("a/b")["knowledge_cards"][0]["quality"] == 5
