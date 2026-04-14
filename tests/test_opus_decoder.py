"""Unit tests for opus_decoder + sdcard length-prefixed parser."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

opuslib = pytest.importorskip("opuslib")

from opus_decoder import OmiOpusStreamDecoder, OMI_HEADER_SIZE  # noqa: E402


def _encode_silence(encoder, samples: int = 160) -> bytes:
    pcm = b"\x00\x00" * samples
    return encoder.encode(pcm, samples)


def test_decode_batch_single_frame():
    encoder = opuslib.Encoder(16000, 1, opuslib.APPLICATION_VOIP)
    opus_frame = _encode_silence(encoder)

    decoder = OmiOpusStreamDecoder(16000, 1)
    pcm_out = decoder.decode_batch([opus_frame])
    assert len(pcm_out) == 160 * 2  # 10ms * int16 mono


def test_feed_with_omi_header():
    encoder = opuslib.Encoder(16000, 1, opuslib.APPLICATION_VOIP)
    opus_frame = _encode_silence(encoder)

    packet = (1).to_bytes(2, "little") + bytes([0]) + opus_frame
    decoder = OmiOpusStreamDecoder(16000, 1)
    pcm_out = decoder.feed(packet)
    assert len(pcm_out) == 160 * 2


def test_feed_ignores_short_packet():
    decoder = OmiOpusStreamDecoder(16000, 1)
    assert decoder.feed(b"\x00\x00") == b""


def test_feed_many_frames_accumulates_pcm():
    encoder = opuslib.Encoder(16000, 1, opuslib.APPLICATION_VOIP)
    decoder = OmiOpusStreamDecoder(16000, 1)

    total = bytearray()
    for pkt_no in range(5):
        opus_frame = _encode_silence(encoder)
        packet = pkt_no.to_bytes(2, "little") + bytes([0]) + opus_frame
        total.extend(decoder.feed(packet))

    assert len(total) == 5 * 160 * 2


def _load_parse_length_prefixed():
    """
    _parse_length_prefixed is a closure inside omi_whisper_bridge.main(),
    so we reimplement the logic here and verify it matches the spec.
    """
    def _parse(data: bytes) -> list[bytes]:
        frames = []
        offset = 0
        while offset + 4 <= len(data):
            length = int.from_bytes(data[offset:offset + 4], "little")
            offset += 4
            if length == 0 or offset + length > len(data):
                break
            frames.append(data[offset:offset + length])
            offset += length
        return frames

    return _parse


def test_parse_length_prefixed_roundtrip():
    parse = _load_parse_length_prefixed()
    frames = [b"aaa", b"bbbb", b"cc"]
    body = b"".join(len(f).to_bytes(4, "little") + f for f in frames)
    assert parse(body) == frames


def test_parse_length_prefixed_truncated():
    parse = _load_parse_length_prefixed()
    # 長さ5と宣言しているのに2バイトしかない → 打ち切り
    body = (5).to_bytes(4, "little") + b"ab"
    assert parse(body) == []
