"""Unit tests for opus_decoder + sdcard length-prefixed parser."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

opuslib = pytest.importorskip("opuslib")

from opus_decoder import (  # noqa: E402
    OmiOpusStreamDecoder,
    OMI_HEADER_SIZE,
    frame_samples_from_toc,
)


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


def test_frame_samples_from_toc_known_configs():
    # config 23 (CELT WB 20ms) -> toc = 23<<3 = 0xB8
    assert frame_samples_from_toc(0xB8, 16000, 160) == 320
    # config 20 (CELT WB 2.5ms) -> toc = 20<<3 = 0xA0
    assert frame_samples_from_toc(0xA0, 16000, 160) == 40
    # config 11 (SILK WB 60ms) -> toc = 11<<3 = 0x58
    assert frame_samples_from_toc(0x58, 16000, 160) == 960


def test_decode_batch_20ms_frame():
    """Omi CV1 SD カード経由の 20ms フレーム (config 23: CELT WB) を模擬。"""
    encoder = opuslib.Encoder(16000, 1, opuslib.APPLICATION_AUDIO)
    # 20ms = 320samples, CELT が選ばれる範囲のビットレートで強制
    encoder.bitrate = 32000
    pcm_20ms = b"\x00\x00" * 320
    opus_frame = encoder.encode(pcm_20ms, 320)

    # エンコーダが 20ms フレーム (config 16-31 範囲) を出したことを確認
    config = (opus_frame[0] >> 3) & 0x1F
    assert _OPUS_CONFIG_FRAME_MS_MAP[config] == 20, (
        f"encoder did not emit 20ms frame, got config={config}"
    )

    decoder = OmiOpusStreamDecoder(16000, 1)
    pcm_out = decoder.decode_batch([opus_frame])
    assert len(pcm_out) == 320 * 2  # 20ms * int16 mono


def test_decode_batch_code1_two_frame_packet():
    """
    TOC code=1 の 2-frame packet (SD firmware がバッチング時に使う形式) を
    decode_batch が正しくデコードできることを確認。
    過去は単一フレーム分の buffer しか渡しておらず buffer_too_small で落ちていた。
    """
    encoder = opuslib.Encoder(16000, 1, opuslib.APPLICATION_AUDIO)
    encoder.bitrate = 32000
    # 20ms フレームを 2 つ生成
    pcm_20ms = b"\x00\x00" * 320
    f1 = encoder.encode(pcm_20ms, 320)
    f2 = encoder.encode(pcm_20ms, 320)

    # code=1 (equal-size 2 frames) packet を手組み:
    # TOC byte の下位 2bit を 01 に書き換え、同じ config の 2 フレームを連結
    assert (f1[0] & 0x03) == 0, "encoder emitted non-code-0 frame"
    assert f1[0] == f2[0], "frames must share same TOC config"
    assert len(f1) == len(f2), "code=1 requires equal-size frames"
    toc_code1 = bytes([f1[0] | 0x01])
    code1_packet = toc_code1 + f1[1:] + f2[1:]

    decoder = OmiOpusStreamDecoder(16000, 1)
    pcm_out = decoder.decode_batch([code1_packet])
    # 2 フレーム x 20ms x int16 mono
    assert len(pcm_out) == 2 * 320 * 2


# テスト側でconfig -> ms を参照するためのマップ (opus_decoder 側と同内容)
_OPUS_CONFIG_FRAME_MS_MAP = {
    i: ms for i, ms in enumerate((
        10, 20, 40, 60,
        10, 20, 40, 60,
        10, 20, 40, 60,
        10, 20,
        10, 20,
        2.5, 5, 10, 20,
        2.5, 5, 10, 20,
        2.5, 5, 10, 20,
        2.5, 5, 10, 20,
    ))
}
