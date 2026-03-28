#!/usr/bin/env python3
"""Check that noise/gibberish filters are still working.

Runs Whisper on real audio WITHOUT filters, then WITH filters,
and shows what was filtered and why.
"""

import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "./recordings")
SAMPLE_RATE = 16000


def load_audio_pcm(audio_path: str) -> bytes | None:
    path = Path(audio_path)
    try:
        if path.suffix == ".wav":
            with wave.open(str(path), "rb") as wf:
                return wf.readframes(wf.getnframes())
        elif path.suffix == ".ogg":
            import soundfile as sf
            data, sr = sf.read(str(path), dtype="int16")
            if sr != SAMPLE_RATE:
                ratio = SAMPLE_RATE / sr
                new_len = int(len(data) * ratio)
                indices = np.linspace(0, len(data) - 1, new_len).astype(int)
                data = data[indices]
            return data.tobytes()
    except Exception as e:
        print(f"  Audio load error: {e}")
    return None


def transcribe_raw(audio_bytes, model):
    """Run Whisper without any post-filters, return raw segments with metadata."""
    import omi_whisper_bridge as bridge

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio_np) < 1600:
        return []

    if bridge.audio_preprocessor is not None:
        audio_np = bridge.audio_preprocessor.process_array(audio_np, SAMPLE_RATE)

    chunk_duration = len(audio_np) / SAMPLE_RATE

    transcribe_kwargs = dict(
        beam_size=bridge.BEAM_SIZE,
        language=bridge.LANGUAGE,
        vad_filter=bridge.VAD_FILTER,
        vad_parameters=dict(
            min_speech_duration_ms=500,
            min_silence_duration_ms=500,
            speech_pad_ms=200,
            threshold=0.5,
        ),
        word_timestamps=False,
        condition_on_previous_text=False,
        no_repeat_ngram_size=bridge.NO_REPEAT_NGRAM_SIZE,
        compression_ratio_threshold=bridge.COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=bridge.LOG_PROB_THRESHOLD,
        no_speech_threshold=bridge.NO_SPEECH_THRESHOLD,
    )

    segments, info = model.transcribe(audio_np, **transcribe_kwargs)
    raw = []
    for seg in segments:
        raw.append({
            "text": seg.text.strip(),
            "start": round(max(0, min(seg.start, chunk_duration)), 2),
            "end": round(max(0, min(seg.end, chunk_duration)), 2),
            "avg_logprob": round(seg.avg_logprob, 3),
            "no_speech_prob": round(seg.no_speech_prob, 3),
        })
    return raw


def classify_filter(seg, bridge):
    """Determine which filter would remove a segment. Returns (kept, reason)."""
    text = seg["text"]
    logprob = seg["avg_logprob"]
    nsp = seg["no_speech_prob"]

    if not text:
        return False, "empty"
    if nsp > 0.8 and logprob < -0.3:
        return False, f"high_no_speech(nsp={nsp:.2f},lp={logprob:.2f})"
    if logprob < -0.6 and nsp > 0.4:
        return False, f"dual_low(lp={logprob:.2f},nsp={nsp:.2f})"
    if '\ufffd' in text:
        return False, "unicode_replacement"
    if len(text) <= 1:
        return False, f"single_char({text})"
    if len(text) <= 3 and logprob < -0.4:
        return False, f"short_low_conf({text},lp={logprob:.2f})"
    if bridge._is_gibberish_kana(text, logprob):
        return False, f"gibberish_kana(lp={logprob:.2f})"

    duration = seg["end"] - seg["start"]
    if duration > 0 and len(text) / duration > 25:
        return False, f"impossible_rate({len(text)/duration:.0f}c/s)"

    cleaned = bridge._remove_hallucinations(text)
    if not cleaned:
        return False, f"hallucination"

    cleaned = bridge._clean_trailing_fragments(cleaned)
    cleaned = bridge._normalize_punctuation(cleaned)
    if not cleaned:
        return False, "empty_after_clean"

    return True, "kept"


# Sessions with known noise/gibberish that should still be filtered
TEST_SESSIONS = [
    "2026-03-28/11-05-49",   # Station announcements (noise environment)
    "2026-03-26/19-54-58",   # Had gibberish in old retranscribe
    "2026-03-28/20-40-46",   # Family conversation with background noise
    "2026-03-27/15-58-39",   # Long session, various noise levels
    "2026-03-27/17-24-33",   # Had many gibberish segments
    "2026-03-28/05-20-39",   # 66 → 18, heavy loss
    "2026-03-26/21-36-29",   # TV/drama audio
]


def main():
    from faster_whisper import WhisperModel
    import omi_whisper_bridge as bridge

    model_size = os.environ.get("WHISPER_MODEL", "large-v3")
    compute_type = os.environ.get("COMPUTE_TYPE", "int8_float16")
    device = os.environ.get("DEVICE", "cuda")

    print(f"Loading model: {model_size} ({compute_type}) on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("Model loaded\n")

    total_raw = 0
    total_kept = 0
    total_filtered = 0
    filter_reasons = {}
    all_filtered_examples = []

    for session in TEST_SESSIONS:
        session_dir = Path(RECORDINGS_DIR) / session
        audio_file = None
        for ext in (".wav", ".ogg"):
            for f in session_dir.glob(f"*{ext}"):
                audio_file = f
                break
            if audio_file:
                break
        if not audio_file:
            continue

        pcm = load_audio_pcm(str(audio_file))
        if not pcm:
            continue

        duration = len(pcm) / 2 / SAMPLE_RATE

        print(f"=== {session} ({duration:.0f}s) ===")

        # Get RAW whisper output (no post-filters)
        raw_segs = transcribe_raw(pcm, model)

        # Also get FILTERED output
        filtered_segs = bridge.transcribe_audio(
            pcm, model, session_offset=0.0, initial_prompt="", hotwords=""
        )

        print(f"  Raw Whisper: {len(raw_segs)} segs → Filtered: {len(filtered_segs)} segs")
        total_raw += len(raw_segs)
        total_kept += len(filtered_segs)

        # Classify each raw segment
        kept_count = 0
        filtered_count = 0
        for seg in raw_segs:
            kept, reason = classify_filter(seg, bridge)
            if not kept:
                filtered_count += 1
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                all_filtered_examples.append((session, seg, reason))
            else:
                kept_count += 1

        total_filtered += filtered_count
        print(f"  Classification: {kept_count} kept, {filtered_count} filtered")

        # Show filtered segments for this session
        filtered_here = [(s, r) for sess, s, r in all_filtered_examples if sess == session]
        if filtered_here:
            print(f"  Filtered segments:")
            for seg, reason in filtered_here[:8]:
                print(f"    [{reason}] \"{seg['text'][:60]}\" (lp={seg['avg_logprob']:.2f}, nsp={seg['no_speech_prob']:.2f})")
            if len(filtered_here) > 8:
                print(f"    ... and {len(filtered_here)-8} more filtered")
        print()

    # Global summary
    print("=" * 70)
    print("FILTER EFFECTIVENESS SUMMARY")
    print("=" * 70)
    print(f"Total raw Whisper segments:    {total_raw}")
    print(f"Total kept after filters:      {total_kept}")
    print(f"Total filtered out:            {total_filtered}")
    print(f"Filter rate:                   {total_filtered/max(total_raw,1)*100:.1f}%")
    print()
    print("Filter breakdown:")
    for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:<40} {count:>4}")
    print()

    # Show sample filtered segments (noise that was correctly caught)
    print("Sample FILTERED segments (should be noise/gibberish):")
    for session, seg, reason in all_filtered_examples[:20]:
        print(f"  [{reason}] [{session}] \"{seg['text'][:70]}\"")
    if len(all_filtered_examples) > 20:
        print(f"  ... and {len(all_filtered_examples)-20} more")
    print()

    # Show sample KEPT segments (should be real speech)
    print("Sample KEPT segments (first 5 from each session, should be real speech):")
    for session in TEST_SESSIONS:
        session_dir = Path(RECORDINGS_DIR) / session
        audio_file = None
        for ext in (".wav", ".ogg"):
            for f in session_dir.glob(f"*{ext}"):
                audio_file = f
                break
            if audio_file:
                break
        if not audio_file:
            continue
        pcm = load_audio_pcm(str(audio_file))
        if not pcm:
            continue
        filtered_segs = bridge.transcribe_audio(
            pcm, model, session_offset=0.0, initial_prompt="", hotwords=""
        )
        if filtered_segs:
            print(f"  [{session}]:")
            for seg in filtered_segs[:3]:
                print(f"    \"{seg['text'][:80]}\"")


if __name__ == "__main__":
    main()
