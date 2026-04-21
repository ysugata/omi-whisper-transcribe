"""
Omi CV1 の OPUSフレームを PCM16 にデコードする。

Omiのパケット形式:
  [2バイト LE: packet_no][1バイト: fragment_idx][OPUSペイロード]

MTU超過時は複数パケットにフラグメント分割される。同一 packet_no の
フラグメントを fragment_idx 順に結合してから OPUS デコーダに渡す。
"""
import logging

import opuslib

log = logging.getLogger(__name__)

OMI_HEADER_SIZE = 3  # packet_no(2) + fragment_idx(1)
OPUS_FRAME_MS = 10   # Omi CV1 BLE 経路は 10ms フレーム

# RFC 6716 §3.1 TOC config(5bit) -> フレーム長(ms)。
# SDカード経路は Omi CV1 側が CELT WB 20ms (config=23) 等の長フレームを
# 使うことがあるので、TOC から出力バッファ長を決める必要がある。
_OPUS_CONFIG_FRAME_MS = (
    10, 20, 40, 60,   # 0-3   SILK NB
    10, 20, 40, 60,   # 4-7   SILK MB
    10, 20, 40, 60,   # 8-11  SILK WB
    10, 20,           # 12-13 Hybrid SWB
    10, 20,           # 14-15 Hybrid FB
    2.5, 5, 10, 20,   # 16-19 CELT NB
    2.5, 5, 10, 20,   # 20-23 CELT WB
    2.5, 5, 10, 20,   # 24-27 CELT SWB
    2.5, 5, 10, 20,   # 28-31 CELT FB
)


def frame_samples_from_toc(toc_byte: int, sample_rate: int, default_samples: int) -> int:
    """OPUS TOC の config bits からデコード後サンプル数を返す。"""
    config = (toc_byte >> 3) & 0x1F
    if 0 <= config < len(_OPUS_CONFIG_FRAME_MS):
        return int(sample_rate * _OPUS_CONFIG_FRAME_MS[config] / 1000)
    return default_samples


class OmiOpusStreamDecoder:
    """
    Omi CV1 の BLE生バイトストリームを PCM16 にデコードするステートフルデコーダ。

    使い方:
        decoder = OmiOpusStreamDecoder(sample_rate=16000, channels=1)
        for raw_packet in ble_packets:
            pcm_chunk = decoder.feed(raw_packet)
            if pcm_chunk:
                audio_buffer.extend(pcm_chunk)
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = sample_rate // 100  # 10ms (フォールバック用)
        self._decoder = opuslib.Decoder(sample_rate, channels)
        self._pending_packet_no: int | None = None
        self._pending_fragments: list[bytes] = []
        self._silence_bytes = b"\x00" * (self.frame_size * 2 * channels)
        self._decode_error_count = 0  # ログ rate-limit 用カウンタ

    def feed(self, raw: bytes) -> bytes:
        """
        BLEから来た生バイトを1つ投入。完成したOPUSフレームがあればPCM16で返す。
        未完成 (フラグメント途中) の場合は b"" を返す。
        """
        if len(raw) < OMI_HEADER_SIZE + 1:
            return b""

        packet_no = int.from_bytes(raw[0:2], "little")
        fragment_idx = raw[2]
        payload = raw[OMI_HEADER_SIZE:]

        if fragment_idx == 0:
            if self._pending_fragments:
                log.debug("Discarding incomplete packet %s", self._pending_packet_no)
            self._pending_packet_no = packet_no
            self._pending_fragments = [payload]
        else:
            if self._pending_packet_no != packet_no:
                log.debug("Fragment mismatch: expected %s got %s",
                          self._pending_packet_no, packet_no)
                self._pending_packet_no = None
                self._pending_fragments = []
                return b""
            self._pending_fragments.append(payload)

        return self._try_decode()

    def _try_decode(self) -> bytes:
        if not self._pending_fragments:
            return b""
        opus_frame = b"".join(self._pending_fragments)
        # TOC から出力バッファ長を決定 (BLE 側も 20ms 等の長フレームの可能性)
        samples = frame_samples_from_toc(
            opus_frame[0], self.sample_rate, self.frame_size,
        )
        try:
            pcm = self._decoder.decode(opus_frame, samples)
            self._pending_fragments = []
            self._pending_packet_no = None
            return pcm
        except opuslib.OpusError as e:
            # sampling しつつログ出力 (BLE で毎フレーム失敗する状況を可視化)
            self._decode_error_count += 1
            if self._decode_error_count <= 5 or self._decode_error_count % 500 == 0:
                log.warning(
                    "OPUS feed decode error (#%d): %s (toc=0x%02x, samples=%d, frame_bytes=%d)",
                    self._decode_error_count, e, opus_frame[0], samples, len(opus_frame),
                )
            self._pending_fragments = []
            self._pending_packet_no = None
            return b""

    def flush_silence(self) -> bytes:
        """接続切断等でバッファを破棄してタイミング維持用の無音を返す。"""
        self._pending_fragments = []
        self._pending_packet_no = None
        return self._silence_bytes

    def decode_batch(self, opus_frames: list[bytes]) -> bytes:
        """
        SDカード同期用: 長さプレフィックス形式からパースした OPUSフレーム列を
        一括デコードする。各フレームは既にヘッダ剥がし済みであること。
        出力バッファ長は TOC バイトから自動判定する (CV1 SDは20msフレーム等)。
        """
        chunks = []
        for frame in opus_frames:
            if not frame:
                continue
            samples = frame_samples_from_toc(frame[0], self.sample_rate, self.frame_size)
            try:
                chunks.append(self._decoder.decode(frame, samples))
            except opuslib.OpusError as e:
                log.warning("OPUS decode error, inserting silence: %s", e)
                chunks.append(b"\x00" * (samples * 2 * self.channels))
        return b"".join(chunks)
