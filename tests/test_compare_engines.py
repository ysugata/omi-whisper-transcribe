"""Tests for tools/compare_engines.py"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import compare_engines


@pytest.fixture
def fake_recordings(tmp_path):
    """Create a minimal recordings directory with fake sessions."""
    import wave
    import numpy as np

    sessions = []
    for date, times in [
        ("2026-03-28", ["10-00-00", "14-30-00"]),
        ("2026-03-29", ["09-15-00"]),
    ]:
        for t in times:
            sess_dir = tmp_path / date / t
            sess_dir.mkdir(parents=True)

            # Create a short WAV file (0.5s of silence)
            wav_path = sess_dir / "audio.wav"
            samples = np.zeros(8000, dtype=np.int16)  # 0.5s at 16kHz
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(samples.tobytes())

            # Create a transcript.json
            transcript = {
                "segments": [
                    {"text": "テスト", "start": 0.0, "end": 0.5,
                     "speaker": "SPEAKER_00", "speaker_id": 0,
                     "is_user": True, "person_id": None}
                ],
                "session": {"started_at": f"{date}T{t.replace('-', ':')}"},
            }
            with open(sess_dir / "transcript.json", "w") as f:
                json.dump(transcript, f, ensure_ascii=False)

            sessions.append(f"{date}/{t}")

    return str(tmp_path), sessions


class TestSelectSamples:
    def test_select_by_count(self, fake_recordings):
        rec_dir, _ = fake_recordings
        samples = compare_engines.select_samples(rec_dir, count=2)
        assert len(samples) == 2
        for s in samples:
            assert "name" in s
            assert "audio" in s
            assert "transcript" in s

    def test_select_all_when_count_exceeds(self, fake_recordings):
        rec_dir, _ = fake_recordings
        samples = compare_engines.select_samples(rec_dir, count=100)
        assert len(samples) == 3  # Only 3 sessions exist

    def test_select_by_name(self, fake_recordings):
        rec_dir, sessions = fake_recordings
        target = sessions[0]  # "2026-03-28/10-00-00"
        samples = compare_engines.select_samples(
            rec_dir, count=5, session_names=[target]
        )
        assert len(samples) == 1
        assert samples[0]["name"] == target

    def test_select_nonexistent_name(self, fake_recordings):
        rec_dir, _ = fake_recordings
        samples = compare_engines.select_samples(
            rec_dir, count=5, session_names=["9999-99-99/00-00-00"]
        )
        assert len(samples) == 0

    def test_empty_directory(self, tmp_path):
        samples = compare_engines.select_samples(str(tmp_path), count=5)
        assert len(samples) == 0


class TestGetAudioDuration:
    def test_duration_calculation(self):
        # 1 second of 16kHz 16-bit mono = 32000 bytes
        pcm = bytes(32000)
        dur = compare_engines.get_audio_duration(pcm)
        assert dur == 1.0

    def test_half_second(self):
        pcm = bytes(16000)
        dur = compare_engines.get_audio_duration(pcm)
        assert dur == 0.5


class TestPrintReport:
    def test_report_basic(self, capsys):
        samples = [
            {"name": "2026-03-28/10-00-00", "transcript": "/nonexistent"},
        ]
        whisper_results = [{
            "segments": [
                {"text": "テスト", "start": 0.0, "end": 1.0},
            ],
            "duration": 10.0,
            "processing_time": 1.0,
            "rtf": 0.1,
        }]
        qwen_results = [{
            "segments": [
                {"text": "テスト結果", "start": 0.0, "end": 1.0},
            ],
            "duration": 10.0,
            "processing_time": 2.0,
            "rtf": 0.2,
        }]
        report = compare_engines.print_report(
            samples, whisper_results, qwen_results
        )

        assert report["samples"] == 1
        assert report["total_audio_duration"] == 10.0
        assert report["summary"]["faster_whisper"]["total_segments"] == 1
        assert report["summary"]["qwen3_asr"]["total_segments"] == 1
        assert report["summary"]["faster_whisper"]["overall_rtf"] == 0.1
        assert report["summary"]["qwen3_asr"]["overall_rtf"] == 0.2

        captured = capsys.readouterr()
        assert "faster-whisper" in captured.out
        assert "qwen3-asr" in captured.out

    def test_report_with_error(self, capsys):
        samples = [
            {"name": "2026-03-28/10-00-00", "transcript": "/nonexistent"},
        ]
        whisper_results = [{"error": "model failed", "duration": 5.0}]
        qwen_results = [{"error": "model failed", "duration": 5.0}]

        report = compare_engines.print_report(
            samples, whisper_results, qwen_results
        )
        assert report["summary"]["faster_whisper"]["total_segments"] == 0

    def test_report_json_output(self, tmp_path):
        samples = [
            {"name": "2026-03-28/10-00-00", "transcript": "/nonexistent"},
        ]
        whisper_results = [{
            "segments": [{"text": "hello", "start": 0.0, "end": 1.0}],
            "duration": 5.0,
            "processing_time": 0.5,
            "rtf": 0.1,
        }]
        qwen_results = [{
            "segments": [{"text": "hello", "start": 0.0, "end": 1.0}],
            "duration": 5.0,
            "processing_time": 0.8,
            "rtf": 0.16,
        }]

        report = compare_engines.print_report(
            samples, whisper_results, qwen_results
        )

        out_path = str(tmp_path / "result.json")
        with open(out_path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["samples"] == 1
        assert len(loaded["sessions"]) == 1
