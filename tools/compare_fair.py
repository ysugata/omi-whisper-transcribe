#!/usr/bin/env python3
"""Fair comparison: Whisper raw vs Qwen3-ASR raw (no LLM correction).

Re-transcribes audio with both engines from scratch, applying only
Layer 1+2 filters (no LLM). Shows side-by-side text for human evaluation.

Usage:
    docker compose stop omi-whisper
    docker compose run --rm \
      -v ./tools:/app/tools:ro \
      -v ./phonetic_corrector.py:/app/phonetic_corrector.py:ro \
      -v ./text_normalizer.py:/app/text_normalizer.py:ro \
      --entrypoint python3 omi-whisper \
      /app/tools/compare_fair.py --ids 2026-04-06/21-15-09
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                _k, _v = _k.strip(), _v.strip()
                if _k and _v and _k not in os.environ:
                    os.environ[_k] = _v

RECORDINGS_DIR = PROJECT_ROOT / "recordings"
CONFIG_DIR = PROJECT_ROOT / "config"

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"

_LANG_MAP = {"ja": "Japanese", "en": "English", "zh": "Chinese"}


def convert_to_wav(audio_path, target_sr=16000):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path),
         "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp.name],
        capture_output=True, check=True, timeout=60,
    )
    return tmp.name


def load_audio(wav_path, target_sr=16000):
    import soundfile as sf
    import numpy as np
    audio, sr = sf.read(wav_path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def normalize_audio(audio, target_dbfs=-16.0):
    import numpy as np
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-10:
        return audio
    gain = 10 ** ((target_dbfs - 20 * np.log10(rms + 1e-10)) / 20)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def transcribe_whisper_raw(audio_np, sr, hotwords_str=""):
    """Transcribe with Whisper (no LLM, Layer 1+2 only)."""
    from faster_whisper import WhisperModel
    import numpy as np

    print(f"    Loading Whisper large-v3...")
    t0 = time.time()
    model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    print(f"    Whisper loaded in {time.time()-t0:.1f}s")

    from term_corrector import TermCorrector
    tc_path = str(CONFIG_DIR / "term_corrections.json")
    tc = TermCorrector(tc_path) if os.path.exists(tc_path) else None

    t0 = time.time()
    segments, info = model.transcribe(
        audio_np,
        beam_size=5,
        language="ja",
        vad_filter=True,
        no_repeat_ngram_size=5,
        patience=2.0,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-0.5,
        no_speech_threshold=0.6,
        repetition_penalty=1.1,
        hotwords=None,  # Disabled to avoid max_length overflow
    )

    result = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if seg.no_speech_prob > 0.8 and seg.avg_logprob < -0.3:
            continue
        if seg.avg_logprob < -0.6 and seg.no_speech_prob > 0.4:
            continue
        if len(text) <= 1:
            continue
        # Layer 2: term correction
        if tc:
            text = tc.correct(text)
        result.append({
            "text": text,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
        })

    elapsed = time.time() - t0
    duration = len(audio_np) / sr

    # Unload
    del model
    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()

    return result, elapsed, duration


def transcribe_qwen3asr_raw(audio_np, sr, context=""):
    """Transcribe with Qwen3-ASR (no LLM, Layer 1+2 + phonetic)."""
    from transcriber_qwen3asr import Qwen3ASRTranscriber

    from term_corrector import TermCorrector
    tc_path = str(CONFIG_DIR / "term_corrections.json")
    tc = TermCorrector(tc_path) if os.path.exists(tc_path) else None

    pc = None
    try:
        from phonetic_corrector import PhoneticCorrector
        glossary = []
        try:
            from hotword_loader import HotwordLoader
            hl = HotwordLoader(str(CONFIG_DIR))
            glossary.extend(hl.get_custom_words())
        except Exception:
            pass
        if tc:
            glossary.extend(tc.list_corrections().values())
        if glossary:
            pc = PhoneticCorrector(glossary=glossary, threshold=0.75)
    except ImportError:
        pass

    transcriber = Qwen3ASRTranscriber()
    print(f"    Loading Qwen3-ASR 1.7B...")
    t0 = time.time()
    transcriber.load_model()
    print(f"    Qwen3-ASR loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    segments = transcriber.transcribe_array(
        audio_np, sr,
        session_offset=0.0,
        hotwords=context,
        language="ja",
        term_corrector=tc,
        phonetic_corrector=pc,
    )
    elapsed = time.time() - t0
    duration = len(audio_np) / sr

    # Unload
    transcriber.unload_model()

    return segments, elapsed, duration


def main():
    parser = argparse.ArgumentParser(description="Fair comparison: Whisper raw vs Qwen3-ASR raw")
    parser.add_argument("--ids", nargs="+", required=True,
                        help="Session IDs (e.g., 2026-04-06/21-15-09)")
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()

    print(f"\n{BOLD}=== Fair Comparison: Whisper Raw vs Qwen3-ASR Raw ==={RESET}")
    print(f"  (No LLM correction — Layer 1+2 only)\n")

    # Hotwords
    hotwords_str = ""
    context_str = ""
    try:
        from hotword_loader import HotwordLoader
        hl = HotwordLoader(str(CONFIG_DIR))
        hotwords_str = hl.get_hotwords_string()
        custom = hl.get_custom_words()
        terms = [w for w in custom if 2 <= len(w) <= 8][:10]
        context_str = ", ".join(terms)
        print(f"  Hotwords: {hl.get_count()} terms")
    except Exception:
        pass

    # Pre-load all audio
    sessions_data = []
    for session_id in args.ids:
        audio_path = RECORDINGS_DIR / session_id / "audio.ogg"
        if not audio_path.exists():
            print(f"  {RED}SKIP {session_id}: no audio.ogg{RESET}")
            continue
        wav_path = convert_to_wav(str(audio_path))
        audio_np, sr = load_audio(wav_path)
        audio_np = normalize_audio(audio_np)
        os.unlink(wav_path)
        duration = len(audio_np) / sr
        sessions_data.append({
            "session_id": session_id,
            "audio_np": audio_np,
            "sr": sr,
            "duration": duration,
        })
        print(f"  {session_id}: {duration:.0f}s ({duration/60:.1f}min)")

    if not sessions_data:
        print(f"{RED}No valid sessions{RESET}")
        sys.exit(1)

    # ---- Phase 1: Whisper (load once, transcribe all, unload) ----
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Phase 1: Whisper (raw, no LLM){RESET}")
    print(f"{'='*60}")

    from faster_whisper import WhisperModel
    import torch, gc

    print(f"  Loading Whisper large-v3...")
    t0 = time.time()
    whisper_model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    from term_corrector import TermCorrector
    tc_path = str(CONFIG_DIR / "term_corrections.json")
    tc = TermCorrector(tc_path) if os.path.exists(tc_path) else None

    whisper_results = {}
    for sd in sessions_data:
        sid = sd["session_id"]
        print(f"\n  {BOLD}[Whisper] {sid}{RESET}")
        t0 = time.time()
        segments, info = whisper_model.transcribe(
            sd["audio_np"],
            beam_size=5, language="ja", vad_filter=True,
            no_repeat_ngram_size=5, patience=2.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-0.5, no_speech_threshold=0.6,
            repetition_penalty=1.1,
            hotwords=None,
        )
        segs = []
        for seg in segments:
            text = seg.text.strip()
            if not text or len(text) <= 1:
                continue
            if seg.no_speech_prob > 0.8 and seg.avg_logprob < -0.3:
                continue
            if seg.avg_logprob < -0.6 and seg.no_speech_prob > 0.4:
                continue
            if tc:
                text = tc.correct(text)
            segs.append({"text": text, "start": round(seg.start, 2), "end": round(seg.end, 2)})
        elapsed = time.time() - t0
        whisper_results[sid] = {"segs": segs, "elapsed": elapsed}
        print(f"    {len(segs)} segments, RTF={elapsed/sd['duration']:.2f}x")

    # Unload Whisper
    del whisper_model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\n  Whisper unloaded")

    # ---- Phase 2: Qwen3-ASR (load once, transcribe all, unload) ----
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Phase 2: Qwen3-ASR (raw, no LLM){RESET}")
    print(f"{'='*60}")

    from transcriber_qwen3asr import Qwen3ASRTranscriber

    pc = None
    try:
        from phonetic_corrector import PhoneticCorrector
        glossary = []
        try:
            from hotword_loader import HotwordLoader
            hl2 = HotwordLoader(str(CONFIG_DIR))
            glossary.extend(hl2.get_custom_words())
        except Exception:
            pass
        if tc:
            glossary.extend(tc.list_corrections().values())
        if glossary:
            pc = PhoneticCorrector(glossary=glossary, threshold=0.75)
    except ImportError:
        pass

    transcriber = Qwen3ASRTranscriber()
    print(f"  Loading Qwen3-ASR 1.7B...")
    t0 = time.time()
    transcriber.load_model()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    qwen_results = {}
    for sd in sessions_data:
        sid = sd["session_id"]
        print(f"\n  {BOLD}[Qwen3-ASR] {sid}{RESET}")
        t0 = time.time()
        segs = transcriber.transcribe_array(
            sd["audio_np"], sd["sr"],
            session_offset=0.0, hotwords=context_str,
            language="ja", term_corrector=tc, phonetic_corrector=pc,
        )
        elapsed = time.time() - t0
        qwen_results[sid] = {"segs": segs, "elapsed": elapsed}
        print(f"    {len(segs)} segments, RTF={elapsed/sd['duration']:.2f}x")

    transcriber.unload_model()
    print(f"\n  Qwen3-ASR unloaded")

    # ---- Phase 3: Display results ----
    results = []
    for sd in sessions_data:
        sid = sd["session_id"]
        w = whisper_results.get(sid, {"segs": [], "elapsed": 0})
        q = qwen_results.get(sid, {"segs": [], "elapsed": 0})

        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}  {sid} ({sd['duration']:.0f}s){RESET}")
        print(f"{'='*60}")

        print(f"\n  {CYAN}--- Whisper (raw) ---{RESET}")
        for s in w["segs"]:
            print(f"    [{s['start']:6.1f}s] {s['text']}")

        print(f"\n  {GREEN}--- Qwen3-ASR (raw) ---{RESET}")
        for s in q["segs"]:
            print(f"    [{s['start']:6.1f}s] {s['text']}")

        results.append({
            "session_id": sid,
            "duration": round(sd["duration"], 1),
            "whisper_segments": len(w["segs"]),
            "qwen3asr_segments": len(q["segs"]),
            "whisper_rtf": round(w["elapsed"] / sd["duration"], 3),
            "qwen3asr_rtf": round(q["elapsed"] / sd["duration"], 3),
            "whisper_text": [{"start": s["start"], "text": s["text"]} for s in w["segs"]],
            "qwen3asr_text": [{"start": s["start"], "text": s["text"]} for s in q["segs"]],
        })

    # Save
    if results:
        output_path = str(CONFIG_DIR / f"compare_fair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)
        print(f"\n  Results saved to: {output_path}")

    print(f"\n{BOLD}Done!{RESET}\n")


if __name__ == "__main__":
    main()
