"""Tests for realtime voice filter in SpeakerFilter."""

import json
import os
import sys
import types
from unittest import mock

import numpy as np
import pytest

# Mock heavy dependencies before importing speaker_filter
_MOCK_MODULES = [
    "torch", "torchaudio",
    "pyannote", "pyannote.audio",
]

_saved = {}
for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        _saved[mod_name] = None
        sys.modules[mod_name] = types.ModuleType(mod_name)
    else:
        _saved[mod_name] = sys.modules[mod_name]

# Stub torch.from_numpy and torch.device
_torch_mod = sys.modules["torch"]
_torch_mod.from_numpy = mock.MagicMock()
_torch_mod.device = mock.MagicMock()
_torch_mod.cuda = mock.MagicMock()


@pytest.fixture(autouse=True)
def _ensure_torch_mock():
    """Ensure torch mock is available for every test."""
    for mod_name in _MOCK_MODULES:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    # Re-stub torch attributes each time
    torch_mod = sys.modules["torch"]
    if not hasattr(torch_mod, "from_numpy") or not callable(getattr(torch_mod, "from_numpy", None)):
        torch_mod.from_numpy = mock.MagicMock()
    if not hasattr(torch_mod, "device"):
        torch_mod.device = mock.MagicMock()
    if not hasattr(torch_mod, "cuda"):
        torch_mod.cuda = mock.MagicMock()
    yield
    # Don't clean up between tests — keep mocks alive


def _cleanup_mocks():
    for mod_name, orig in _saved.items():
        if orig is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = orig


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_session():
    yield
    _cleanup_mocks()


def _setup_torch_tensor(audio_len):
    """Set up mock torch tensor for filter_segments_realtime."""
    mock_tensor = mock.MagicMock()
    mock_tensor.unsqueeze.return_value = mock_tensor
    mock_tensor.float.return_value = mock_tensor
    mock_tensor.shape = [1, audio_len]
    mock_tensor.__getitem__ = mock.MagicMock(return_value=mock_tensor)
    sys.modules["torch"].from_numpy = mock.MagicMock(return_value=mock_tensor)
    return mock_tensor


def _make_filter(tmp_path, speakers=None, threshold=0.55):
    """Create SpeakerFilter with test data."""
    config_dir = str(tmp_path)
    data = {
        "threshold": threshold,
        "model": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "speakers": speakers or [],
    }
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "speakers.json"), "w") as f:
        json.dump(data, f)

    from speaker_filter import SpeakerFilter
    return SpeakerFilter(config_dir=config_dir, hf_token="test_token", device="cpu")


def _make_user_speaker(embedding=None):
    """Create a user speaker dict with embedding."""
    if embedding is None:
        embedding = np.random.randn(256).tolist()
    arr = np.array(embedding)
    arr = arr / np.linalg.norm(arr)
    return {
        "id": "sp_test01",
        "name": "TestUser",
        "is_user": True,
        "created_at": "2026-01-01T00:00:00",
        "embedding": arr.tolist(),
        "samples": [],
    }


def _make_non_user_speaker(embedding=None):
    """Create a non-user speaker dict with embedding."""
    if embedding is None:
        embedding = np.random.randn(256).tolist()
    arr = np.array(embedding)
    arr = arr / np.linalg.norm(arr)
    return {
        "id": "sp_test02",
        "name": "TVSpeaker",
        "is_user": False,
        "created_at": "2026-01-01T00:00:00",
        "embedding": arr.tolist(),
        "samples": [],
    }


class TestGetUserSpeakers:
    def test_returns_user_speakers_with_embedding(self, tmp_path):
        user = _make_user_speaker()
        non_user = _make_non_user_speaker()
        sf = _make_filter(tmp_path, speakers=[user, non_user])
        result = sf.get_user_speakers()
        assert len(result) == 1
        assert result[0]["id"] == "sp_test01"
        assert result[0]["is_user"] is True

    def test_excludes_speakers_without_embedding(self, tmp_path):
        user_no_emb = _make_user_speaker()
        user_no_emb["embedding"] = None
        sf = _make_filter(tmp_path, speakers=[user_no_emb])
        result = sf.get_user_speakers()
        assert len(result) == 0

    def test_empty_speakers(self, tmp_path):
        sf = _make_filter(tmp_path, speakers=[])
        result = sf.get_user_speakers()
        assert len(result) == 0


class TestFilterSegmentsRealtime:
    def _make_mock_inference(self, return_embedding):
        """Create a mock inference callable."""
        def inference(data):
            return return_embedding
        return inference

    def test_keeps_user_segments(self, tmp_path):
        """Segments matching user voice should be kept."""
        user_emb = np.random.randn(256)
        user_emb = user_emb / np.linalg.norm(user_emb)
        user = _make_user_speaker(user_emb.tolist())
        sf = _make_filter(tmp_path, speakers=[user], threshold=0.5)

        inference = self._make_mock_inference(user_emb)

        audio = np.zeros(16000 * 3, dtype=np.float32)
        segments = [{"start": 0.0, "end": 2.0, "text": "Hello user"}]

        _setup_torch_tensor(len(audio))

        result = sf.filter_segments_realtime(inference, audio, segments, threshold=0.5)
        assert len(result) == 1
        assert result[0]["text"] == "Hello user"

    def test_removes_non_user_segments(self, tmp_path):
        """Segments not matching any user voice should be removed."""
        user_emb = np.random.randn(256)
        user_emb = user_emb / np.linalg.norm(user_emb)
        user = _make_user_speaker(user_emb.tolist())
        sf = _make_filter(tmp_path, speakers=[user], threshold=0.5)

        other_emb = -user_emb
        inference = self._make_mock_inference(other_emb)

        audio = np.zeros(16000 * 3, dtype=np.float32)
        segments = [{"start": 0.0, "end": 2.0, "text": "TV news broadcast"}]

        _setup_torch_tensor(len(audio))

        result = sf.filter_segments_realtime(inference, audio, segments, threshold=0.5)
        assert len(result) == 0

    def test_keeps_short_segments(self, tmp_path):
        """Segments < 0.5s should be kept regardless of voice match."""
        user = _make_user_speaker()
        sf = _make_filter(tmp_path, speakers=[user], threshold=0.5)

        other_emb = np.random.randn(256)
        other_emb = other_emb / np.linalg.norm(other_emb)
        inference = self._make_mock_inference(other_emb)

        audio = np.zeros(16000 * 2, dtype=np.float32)
        segments = [{"start": 0.0, "end": 0.3, "text": "Hmm"}]

        _setup_torch_tensor(len(audio))

        result = sf.filter_segments_realtime(inference, audio, segments, threshold=0.5)
        assert len(result) == 1

    def test_no_user_speakers_returns_all(self, tmp_path):
        """When no user speakers registered, all segments pass through."""
        non_user = _make_non_user_speaker()
        sf = _make_filter(tmp_path, speakers=[non_user], threshold=0.5)

        inference = self._make_mock_inference(np.zeros(256))
        audio = np.zeros(16000 * 3, dtype=np.float32)
        segments = [{"start": 0.0, "end": 2.0, "text": "Whatever"}]

        result = sf.filter_segments_realtime(inference, audio, segments, threshold=0.5)
        assert len(result) == 1

    def test_uses_speakers_json_threshold(self, tmp_path):
        """When threshold=0, use speakers.json threshold value."""
        user_emb = np.random.randn(256)
        user_emb = user_emb / np.linalg.norm(user_emb)
        user = _make_user_speaker(user_emb.tolist())

        sf = _make_filter(tmp_path, speakers=[user], threshold=0.99)

        noise = np.random.randn(256) * 0.3
        sim_emb = user_emb + noise
        sim_emb = sim_emb / np.linalg.norm(sim_emb)
        inference = self._make_mock_inference(sim_emb)

        audio = np.zeros(16000 * 3, dtype=np.float32)
        segments = [{"start": 0.0, "end": 2.0, "text": "User speaking"}]

        _setup_torch_tensor(len(audio))

        # threshold=0 should use speakers.json threshold (0.99)
        result = sf.filter_segments_realtime(inference, audio, segments, threshold=0)
        assert len(result) == 0

    def test_inference_error_keeps_segment(self, tmp_path):
        """On inference error, segment should be kept."""
        user = _make_user_speaker()
        sf = _make_filter(tmp_path, speakers=[user], threshold=0.5)

        def failing_inference(data):
            raise RuntimeError("GPU error")

        audio = np.zeros(16000 * 3, dtype=np.float32)
        segments = [{"start": 0.0, "end": 2.0, "text": "Should survive error"}]

        _setup_torch_tensor(len(audio))

        result = sf.filter_segments_realtime(failing_inference, audio, segments, threshold=0.5)
        assert len(result) == 1

    def test_mixed_segments(self, tmp_path):
        """Mix of user and non-user segments: only user kept."""
        user_emb = np.random.randn(256)
        user_emb = user_emb / np.linalg.norm(user_emb)
        user = _make_user_speaker(user_emb.tolist())
        sf = _make_filter(tmp_path, speakers=[user], threshold=0.5)

        other_emb = -user_emb

        call_count = [0]
        def alternating_inference(data):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return user_emb
            return other_emb

        audio = np.zeros(16000 * 10, dtype=np.float32)
        segments = [
            {"start": 0.0, "end": 2.0, "text": "User says hi"},
            {"start": 2.0, "end": 4.0, "text": "TV commercial"},
            {"start": 4.0, "end": 6.0, "text": "User responds"},
            {"start": 6.0, "end": 8.0, "text": "TV weather"},
        ]

        _setup_torch_tensor(len(audio))

        result = sf.filter_segments_realtime(alternating_inference, audio, segments, threshold=0.5)
        assert len(result) == 2
        assert result[0]["text"] == "User says hi"
        assert result[1]["text"] == "User responds"
