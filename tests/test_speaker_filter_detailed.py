"""Tests for speaker_filter list_speakers_detailed()."""
import json
import os
import tempfile

from speaker_filter import SpeakerFilter


def _make_filter(tmp_dir, speakers=None):
    """Create a SpeakerFilter with pre-populated data."""
    config_dir = tmp_dir
    speakers_path = os.path.join(config_dir, "speakers.json")
    data = {
        "threshold": 0.55,
        "model": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "speakers": speakers or [],
    }
    with open(speakers_path, "w") as f:
        json.dump(data, f)
    return SpeakerFilter(config_dir=config_dir, hf_token="fake", device="cpu")


class TestListSpeakersDetailed:
    def test_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            sf = _make_filter(tmp)
            result = sf.list_speakers_detailed()
            assert result == []

    def test_with_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            speakers = [
                {
                    "id": "sp_abc123",
                    "name": "Taro",
                    "is_user": True,
                    "created_at": "2026-03-28T10:00:00",
                    "embedding": [0.1] * 256,
                    "samples": [
                        {
                            "id": "s_001",
                            "added_at": "2026-03-28T10:05:00",
                            "source_session": "2026-03-28/10-00-00",
                            "segment_indices": [0, 1, 2],
                            "duration": 5.5,
                            "embedding": [0.1] * 256,
                        },
                        {
                            "id": "s_002",
                            "added_at": "2026-03-28T11:00:00",
                            "source_session": "2026-03-28/11-00-00 (correction)",
                            "segment_indices": [3],
                            "duration": 2.0,
                            "embedding": [0.2] * 256,
                            "source": "correction",
                            "audio_start": 10.5,
                            "audio_end": 12.5,
                        },
                    ],
                }
            ]
            sf = _make_filter(tmp, speakers=speakers)
            result = sf.list_speakers_detailed()

            assert len(result) == 1
            sp = result[0]
            assert sp["id"] == "sp_abc123"
            assert sp["name"] == "Taro"
            assert sp["is_user"] is True
            assert sp["sample_count"] == 2
            assert sp["total_duration"] == 7.5
            assert sp["has_embedding"] is True
            assert len(sp["samples"]) == 2

            # Verify embedding is NOT included
            for sample in sp["samples"]:
                assert "embedding" not in sample

            # Verify sample metadata
            s1 = sp["samples"][0]
            assert s1["id"] == "s_001"
            assert s1["duration"] == 5.5
            assert s1["source"] == "manual"  # default
            assert "audio_start" not in s1  # legacy sample

            s2 = sp["samples"][1]
            assert s2["id"] == "s_002"
            assert s2["source"] == "correction"
            assert s2["audio_start"] == 10.5
            assert s2["audio_end"] == 12.5

    def test_no_embedding_speaker(self):
        with tempfile.TemporaryDirectory() as tmp:
            speakers = [
                {
                    "id": "sp_xyz",
                    "name": "Jiro",
                    "is_user": False,
                    "created_at": "2026-03-28T10:00:00",
                    "embedding": None,
                    "samples": [],
                }
            ]
            sf = _make_filter(tmp, speakers=speakers)
            result = sf.list_speakers_detailed()

            assert len(result) == 1
            assert result[0]["has_embedding"] is False
            assert result[0]["samples"] == []
