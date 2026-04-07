#!/usr/bin/env python3
"""Compare Whisper vs Qwen3-ASR transcription quality.

Runs Qwen3-ASR on existing recordings (which have Whisper transcripts)
and compares the results side-by-side.

Usage:
    # Stop omi-whisper first to free GPU
    docker compose stop omi-whisper

    # Install deps (if needed)
    pip install qwen-asr

    # Compare a single session
    python tools/compare_engines.py --sessions 1 --ids 2026-04-01/12-30-00

    # Compare 8 diverse sessions
    python tools/compare_engines.py --sessions 8

    # Limit max audio duration
    python tools/compare_engines.py --sessions 8 --max-duration 300

    # Use custom hotwords
    python tools/compare_engines.py --sessions 4 --hotwords
"""

import argparse
import difflib
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

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def check_dependencies():
    """Check required packages are installed."""
    missing = []
    for pkg in ["qwen_asr", "torch", "soundfile"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"{RED}ERROR: Missing packages: {', '.join(missing)}{RESET}")
        print(f'  pip install qwen-asr soundfile')
        sys.exit(1)


def check_gpu_available():
    """Check GPU availability and VRAM."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        line = result.stdout.strip().split("\n")[0]
        free_mb, total_mb = [int(x.strip()) for x in line.split(",")]
        print(f"  GPU VRAM: {free_mb} MB free / {total_mb} MB total")
        if free_mb < 4000:
            print(f"  {YELLOW}WARNING: Qwen3-ASR needs ~3.4 GB VRAM. "
                  f"Only {free_mb} MB free.{RESET}")
            print(f"  {YELLOW}Consider: docker compose stop omi-whisper{RESET}")
        return True, free_mb
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"  {YELLOW}WARNING: nvidia-smi not available.{RESET}")
        return False, 0


# ---------------------------------------------------------------------------
# Session finder
# ---------------------------------------------------------------------------

def find_diverse_sessions(limit=8, specific=None, max_duration=600):
    """Find diverse sessions with Whisper transcripts for comparison."""
    if specific:
        result = []
        for s in specific:
            sess_dir = RECORDINGS_DIR / s
            transcript = sess_dir / "transcript.json"
            audio = sess_dir / "audio.ogg"
            if not transcript.exists():
                print(f"  {YELLOW}SKIP {s}: no transcript.json{RESET}")
                continue
            if not audio.exists():
                print(f"  {YELLOW}SKIP {s}: no audio.ogg{RESET}")
                continue
            result.append(s)
        return result

    candidates = []
    for date_dir in sorted(RECORDINGS_DIR.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for sess_dir in sorted(date_dir.iterdir()):
            if not sess_dir.is_dir():
                continue
            transcript = sess_dir / "transcript.json"
            audio = sess_dir / "audio.ogg"
            if not transcript.exists() or not audio.exists():
                continue
            try:
                with open(transcript) as f:
                    data = json.load(f)
                segs = data.get("segments", []) if isinstance(data, dict) else data
                session_meta = data.get("session", {}) if isinstance(data, dict) else {}
                if len(segs) < 3:
                    continue
                duration = session_meta.get("duration_seconds", 0)
                if not duration and segs:
                    duration = max(s.get("end", 0) for s in segs)
                if duration > max_duration or duration < 2:
                    continue
                sess_id = f"{date_dir.name}/{sess_dir.name}"
                candidates.append({
                    "id": sess_id,
                    "duration": duration,
                    "seg_count": len(segs),
                })
            except Exception:
                continue

    # Sort by segment count (prefer more content)
    candidates.sort(key=lambda x: -x["seg_count"])
    return [c["id"] for c in candidates[:limit]]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session(session_id):
    """Load Whisper transcript. Returns (session_meta, segments)."""
    path = RECORDINGS_DIR / session_id / "transcript.json"
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("session", {}), data.get("segments", [])
    return {}, data


def get_audio_path(session_id):
    """Return path to audio file (ogg preferred, wav fallback)."""
    sess_dir = RECORDINGS_DIR / session_id
    for ext in ("audio.ogg", "audio.wav"):
        p = sess_dir / ext
        if p.exists():
            return p
    return None


def convert_to_wav_16k(audio_path):
    """Convert audio to 16kHz mono WAV for Qwen3-ASR.

    Returns path to temporary WAV file. Caller must delete it.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", "-f", "wav", tmp.name],
            capture_output=True, check=True, timeout=60,
        )
        return tmp.name
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        os.unlink(tmp.name)
        raise RuntimeError(f"ffmpeg conversion failed: {e}")


# ---------------------------------------------------------------------------
# Hotwords
# ---------------------------------------------------------------------------

def build_hotwords_context():
    """Build context string from hotwords config for Qwen3-ASR.

    Only uses custom words + term correction targets (max 15 short terms)
    to prevent context leak into transcription output.
    """
    try:
        from hotword_loader import HotwordLoader
        hl = HotwordLoader(str(CONFIG_DIR))
        # Only custom words (user-curated, high-value terms)
        words = hl.get_custom_words()

        # Also add term correction targets
        try:
            from term_corrector import TermCorrector
            tc_path = str(CONFIG_DIR / "term_corrections.json")
            if os.path.exists(tc_path):
                tc = TermCorrector(tc_path)
                for v in tc.list_corrections().values():
                    if v not in words:
                        words.append(v)
        except Exception:
            pass

        # Filter: 2-8 chars, max 10 terms (fewer = less risk of context leak)
        words = [w for w in words if 2 <= len(w) <= 8][:10]
        if words:
            return " ".join(words)
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Qwen3-ASR transcription
# ---------------------------------------------------------------------------

def load_qwen3asr_model():
    """Load Qwen3-ASR 1.7B model."""
    import torch
    from qwen_asr import Qwen3ASRModel

    print(f"  Loading Qwen3-ASR model (bfloat16)...")
    t0 = time.time()

    kwargs = dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=1024,  # Official recommendation for evaluation
    )

    model = None
    for attn_impl in ("flash_attention_2", "sdpa", None):
        try:
            if attn_impl:
                kwargs["attn_implementation"] = attn_impl
            model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", **kwargs)
            print(f"  Attention: {attn_impl or 'eager'}")
            break
        except (ValueError, ImportError):
            kwargs.pop("attn_implementation", None)
            continue

    if model is None:
        print(f"{RED}ERROR: Failed to load Qwen3-ASR model{RESET}")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s")
    return model


_LANG_MAP = {
    "ja": "Japanese", "en": "English", "zh": "Chinese",
    "ko": "Korean", "de": "German", "fr": "French",
    "es": "Spanish", "pt": "Portuguese", "ru": "Russian",
}


def _normalize_audio(audio_data, target_dbfs=-16.0):
    """Loudness normalization only (no denoising — hurts modern ASR models)."""
    import numpy as np
    rms = np.sqrt(np.mean(audio_data**2))
    if rms < 1e-10:
        return audio_data
    current_dbfs = 20 * np.log10(rms + 1e-10)
    gain_db = target_dbfs - current_dbfs
    gain_linear = 10 ** (gain_db / 20)
    normalized = audio_data * gain_linear
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def transcribe_with_qwen3asr(model, wav_path, context="", language="ja"):
    """Transcribe a WAV file using Qwen3-ASR.

    Applies loudness normalization (no denoising per arXiv:2512.17562).
    Uses greedy decoding (official recommendation).

    Returns list of segments [{"text", "start", "end"}].
    """
    import soundfile as sf
    import numpy as np

    audio_data, sr = sf.read(wav_path, dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Loudness normalization only (denoising hurts transformer ASR)
    audio_data = _normalize_audio(audio_data)

    duration = len(audio_data) / sr

    # Greedy decoding is the official recommendation (no temperature/beam)
    transcribe_kwargs = {}
    if language:
        lang_name = _LANG_MAP.get(language.lower(), language)
        transcribe_kwargs["language"] = lang_name
    if context:
        transcribe_kwargs["context"] = [context]

    t0 = time.time()
    results = model.transcribe(audio=(audio_data, sr), **transcribe_kwargs)
    elapsed = time.time() - t0

    # Context leak detection
    context_terms = set()
    if context:
        context_terms = {t for t in context.split() if len(t) >= 2}

    segments = []
    for result in results:
        text = result.text.strip() if hasattr(result, 'text') else str(result).strip()
        if not text:
            continue
        # Drop if >50% of text matches context terms (context leak)
        if context_terms:
            import re as _re
            matched = sum(text.count(t) * len(t) for t in context_terms)
            text_chars = len(_re.sub(r'\s+', '', text))
            if text_chars > 0 and matched / text_chars > 0.5:
                print(f"    {YELLOW}Context leak detected, skipping{RESET}")
                continue
        lang = result.language if hasattr(result, 'language') else None
        seg = {
            "text": text,
            "start": 0.0,
            "end": round(duration, 2),
        }
        if lang:
            seg["language"] = lang
        segments.append(seg)

    return segments, elapsed, duration


# ---------------------------------------------------------------------------
# Text comparison
# ---------------------------------------------------------------------------

def segments_to_text(segments):
    """Concatenate segment texts."""
    return " ".join(s.get("text", "") for s in segments).strip()


def char_edit_distance(a, b):
    """Compute character-level edit distance ratio (0=identical, 1=totally different)."""
    if not a and not b:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b)
    return 1.0 - sm.ratio()


def show_diff(whisper_text, qwen_text, max_width=120):
    """Show unified diff of the two texts."""
    w_lines = [whisper_text[i:i+max_width] for i in range(0, len(whisper_text), max_width)]
    q_lines = [qwen_text[i:i+max_width] for i in range(0, len(qwen_text), max_width)]

    diff = difflib.unified_diff(
        w_lines, q_lines,
        fromfile="Whisper", tofile="Qwen3-ASR",
        lineterm="",
    )
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            print(f"  {GREEN}{line}{RESET}")
        elif line.startswith("-") and not line.startswith("---"):
            print(f"  {RED}{line}{RESET}")
        elif line.startswith("@@"):
            print(f"  {CYAN}{line}{RESET}")
        else:
            print(f"  {line}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare Whisper vs Qwen3-ASR")
    parser.add_argument("--sessions", type=int, default=4,
                        help="Number of sessions to compare (default: 4)")
    parser.add_argument("--ids", nargs="*", default=None,
                        help="Specific session IDs (e.g., 2026-04-01/12-30-00)")
    parser.add_argument("--max-duration", type=int, default=300,
                        help="Max audio duration in seconds (default: 300)")
    parser.add_argument("--hotwords", action="store_true",
                        help="Use hotwords as context for Qwen3-ASR")
    parser.add_argument("--language", default="ja",
                        help="Language code (default: ja, 'auto' for detection)")
    parser.add_argument("--output", default=None,
                        help="Output JSON file (default: config/compare_engines_TIMESTAMP.json)")
    args = parser.parse_args()

    print(f"\n{BOLD}=== Whisper vs Qwen3-ASR Comparison ==={RESET}\n")

    # Checks
    check_dependencies()
    check_gpu_available()

    # Find sessions
    print(f"\n{BOLD}Finding sessions...{RESET}")
    sessions = find_diverse_sessions(
        limit=args.sessions,
        specific=args.ids,
        max_duration=args.max_duration,
    )
    if not sessions:
        print(f"{RED}No valid sessions found{RESET}")
        sys.exit(1)
    print(f"  Found {len(sessions)} session(s)")

    # Load model
    print(f"\n{BOLD}Loading Qwen3-ASR model...{RESET}")
    model = load_qwen3asr_model()

    # Build context
    context = ""
    if args.hotwords:
        context = build_hotwords_context()
        if context:
            print(f"  Hotwords context: {len(context.split())} words")

    lang = None if args.language == "auto" else args.language

    # Term corrector for Layer 2
    term_corrector = None
    try:
        from term_corrector import TermCorrector
        corrections_path = str(CONFIG_DIR / "term_corrections.json")
        if os.path.exists(corrections_path):
            term_corrector = TermCorrector(corrections_path)
    except Exception:
        pass

    # Phonetic corrector (Layer 2.5 — pronunciation-based correction)
    phonetic_corrector = None
    try:
        from phonetic_corrector import PhoneticCorrector
        glossary = []
        # Build glossary from hotwords custom + term correction targets
        try:
            from hotword_loader import HotwordLoader
            hl = HotwordLoader(str(CONFIG_DIR))
            glossary.extend(hl.get_custom_words())
        except Exception:
            pass
        if term_corrector:
            glossary.extend(term_corrector.list_corrections().values())
        if glossary:
            phonetic_corrector = PhoneticCorrector(glossary=glossary, threshold=0.75)
            print(f"  Phonetic corrector: {len(glossary)} glossary terms")
    except ImportError:
        print(f"  {DIM}Phonetic corrector: not available (pykakasi not installed){RESET}")

    # Text normalizer
    text_normalize = None
    try:
        from text_normalizer import normalize as _tn
        text_normalize = _tn
        print(f"  Text normalizer: enabled")
    except ImportError:
        print(f"  {DIM}Text normalizer: not available{RESET}")

    # Process sessions
    results = []
    print(f"\n{BOLD}Processing sessions...{RESET}\n")

    for i, session_id in enumerate(sessions, 1):
        print(f"  {BOLD}[{i}/{len(sessions)}] {session_id}{RESET}")

        meta, whisper_segs = load_session(session_id)
        audio_path = get_audio_path(session_id)
        if not audio_path:
            print(f"    {YELLOW}SKIP: no audio file{RESET}")
            continue

        # Convert to 16kHz WAV
        try:
            wav_path = convert_to_wav_16k(audio_path)
        except RuntimeError as e:
            print(f"    {YELLOW}SKIP: {e}{RESET}")
            continue

        try:
            # Run Qwen3-ASR
            qwen_segs, qwen_elapsed, duration = transcribe_with_qwen3asr(
                model, wav_path, context=context, language=lang,
            )

            # Apply post-processing pipeline to Qwen3-ASR output
            if qwen_segs:
                for seg in qwen_segs:
                    t = seg["text"]
                    # Text normalization (half/full-width, etc.)
                    if text_normalize:
                        t = text_normalize(t)
                    # Term correction (dictionary)
                    if term_corrector:
                        t = term_corrector.correct(t)
                    # Phonetic correction (pronunciation-based)
                    if phonetic_corrector:
                        t = phonetic_corrector.correct(t)
                    seg["text"] = t

            # Compare
            whisper_text = segments_to_text(whisper_segs)
            qwen_text = segments_to_text(qwen_segs)

            cer = char_edit_distance(whisper_text, qwen_text)
            rtf = qwen_elapsed / duration if duration > 0 else 0

            result = {
                "session_id": session_id,
                "duration_seconds": round(duration, 1),
                "whisper_segments": len(whisper_segs),
                "qwen3asr_segments": len(qwen_segs),
                "whisper_chars": len(whisper_text),
                "qwen3asr_chars": len(qwen_text),
                "char_edit_distance": round(cer, 4),
                "qwen3asr_rtf": round(rtf, 3),
                "qwen3asr_elapsed": round(qwen_elapsed, 2),
                "whisper_text": whisper_text,
                "qwen3asr_text": qwen_text,
            }
            results.append(result)

            # Print summary
            cer_color = GREEN if cer < 0.3 else (YELLOW if cer < 0.5 else RED)
            print(f"    Duration: {duration:.0f}s | RTF: {rtf:.2f}x")
            print(f"    Whisper: {len(whisper_segs)} segs, {len(whisper_text)} chars")
            print(f"    Qwen3:  {len(qwen_segs)} segs, {len(qwen_text)} chars")
            print(f"    CER: {cer_color}{cer:.1%}{RESET}")

            # Show diff for interesting cases
            if cer > 0.05:
                show_diff(whisper_text[:500], qwen_text[:500])
            print()

        finally:
            os.unlink(wav_path)

    # Summary
    if results:
        print(f"\n{BOLD}=== Summary ==={RESET}")
        avg_cer = sum(r["char_edit_distance"] for r in results) / len(results)
        avg_rtf = sum(r["qwen3asr_rtf"] for r in results) / len(results)
        total_dur = sum(r["duration_seconds"] for r in results)
        total_elapsed = sum(r["qwen3asr_elapsed"] for r in results)

        print(f"  Sessions: {len(results)}")
        print(f"  Total audio: {total_dur:.0f}s ({total_dur/60:.1f}min)")
        print(f"  Avg CER: {avg_cer:.1%}")
        print(f"  Avg RTF: {avg_rtf:.2f}x")
        print(f"  Total processing: {total_elapsed:.1f}s")

        # Save results
        output_path = args.output
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(CONFIG_DIR / f"compare_engines_{timestamp}.json")

        output = {
            "timestamp": datetime.now().isoformat(),
            "engines": ["whisper", "qwen3asr"],
            "whisper_model": meta.get("model", "large-v3"),
            "qwen3asr_model": "Qwen/Qwen3-ASR-1.7B",
            "language": args.language,
            "hotwords_enabled": args.hotwords,
            "summary": {
                "sessions": len(results),
                "total_audio_seconds": round(total_dur, 1),
                "avg_char_edit_distance": round(avg_cer, 4),
                "avg_rtf": round(avg_rtf, 3),
            },
            "sessions": [{k: v for k, v in r.items()
                         if k not in ("whisper_text", "qwen3asr_text")}
                        for r in results],
            "full_text": [{"session_id": r["session_id"],
                          "whisper": r["whisper_text"],
                          "qwen3asr": r["qwen3asr_text"]}
                         for r in results],
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n  Results saved to: {output_path}")

    # Cleanup
    del model
    import torch
    torch.cuda.empty_cache()

    print(f"\n{BOLD}Done!{RESET}\n")


if __name__ == "__main__":
    main()
