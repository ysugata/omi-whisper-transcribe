#!/usr/bin/env python3
"""Retranscribe all sessions using updated filters.

Modes:
  (default)   Re-transcribe all sessions with updated filters
  --restore   Restore all .bak files to transcript.json
  --verify    Compare current vs .bak segment counts
  --dry-run   Preview changes without writing

Hotwords are OFF by default for batch retranscription (they can cause
comma-separated keyword contamination). Use --hotwords to enable.

Usage (inside Docker container):
  python tools/retranscribe_all.py                    # retranscribe all
  python tools/retranscribe_all.py --restore          # restore from .bak
  python tools/retranscribe_all.py --verify           # compare segment counts
  python tools/retranscribe_all.py --dry-run          # preview only
  python tools/retranscribe_all.py --hotwords         # retranscribe with hotwords
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Add parent directory to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "./recordings")
SAMPLE_RATE = 16000


def find_all_sessions(recordings_dir: str) -> list[dict]:
    """Find all sessions with audio files, sorted by date."""
    base = Path(recordings_dir)
    sessions = []
    if not base.exists():
        return sessions

    for date_dir in sorted(base.iterdir()):
        if not date_dir.is_dir():
            continue
        for time_dir in sorted(date_dir.iterdir()):
            if not time_dir.is_dir():
                continue
            transcript_path = time_dir / "transcript.json"
            if not transcript_path.exists():
                continue
            # Find audio file
            audio_file = None
            for ext in (".wav", ".ogg"):
                for f in time_dir.glob(f"*{ext}"):
                    audio_file = f
                    break
                if audio_file:
                    break
            if not audio_file:
                continue
            sessions.append({
                "path": str(time_dir),
                "transcript": str(transcript_path),
                "audio": str(audio_file),
                "name": f"{date_dir.name}/{time_dir.name}",
            })
    return sessions


def load_audio_as_pcm(audio_path: str) -> bytes | None:
    """Load audio file and return PCM 16-bit LE bytes."""
    path = Path(audio_path)
    try:
        if path.suffix == ".wav":
            with wave.open(str(path), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                if wf.getsampwidth() == 2:
                    return frames
                # Convert if needed
                audio_np = np.frombuffer(frames, dtype=np.int16)
                return audio_np.tobytes()
        elif path.suffix == ".ogg":
            import soundfile as sf
            data, sr = sf.read(str(path), dtype="int16")
            if sr != SAMPLE_RATE:
                # Simple resample via numpy
                import scipy.signal
                data = scipy.signal.resample(
                    data, int(len(data) * SAMPLE_RATE / sr)
                ).astype(np.int16)
            return data.tobytes()
    except Exception as e:
        log.error(f"Failed to load audio {audio_path}: {e}")
    return None


def restore_all(recordings_dir: str, dry_run: bool = False) -> None:
    """Restore all .bak files to transcript.json."""
    sessions = find_all_sessions(recordings_dir)
    restored = 0
    missing = 0

    for sess in sessions:
        bak_path = sess["transcript"] + ".bak"
        if not os.path.exists(bak_path):
            missing += 1
            continue
        if dry_run:
            log.info(f"[DRY-RUN] Would restore: {sess['name']}")
            restored += 1
            continue
        shutil.copy2(bak_path, sess["transcript"])
        restored += 1

    log.info(f"Restore complete: {restored} restored, {missing} no .bak found")


def verify_all(recordings_dir: str) -> None:
    """Compare current transcript segment counts with .bak versions."""
    sessions = find_all_sessions(recordings_dir)
    total = 0
    all_lost = 0      # sessions where all segments were lost
    half_lost = 0     # sessions with 50%+ segment reduction
    increased = 0
    no_bak = 0

    results = []

    for sess in sessions:
        bak_path = sess["transcript"] + ".bak"
        if not os.path.exists(bak_path):
            no_bak += 1
            continue
        total += 1

        try:
            with open(sess["transcript"]) as f:
                current = json.load(f)
            with open(bak_path) as f:
                backup = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Error reading {sess['name']}: {e}")
            continue

        cur_count = len(current.get("segments", []))
        bak_count = len(backup.get("segments", []))

        if bak_count == 0:
            continue

        if cur_count == 0 and bak_count >= 5:
            all_lost += 1
            results.append((sess["name"], bak_count, cur_count, "ALL LOST"))
        elif cur_count < bak_count * 0.5:
            half_lost += 1
            results.append((sess["name"], bak_count, cur_count, "50%+ LOST"))
        elif cur_count > bak_count:
            increased += 1

    log.info("=" * 60)
    log.info(f"Verification Report ({total} sessions with .bak)")
    log.info("=" * 60)
    log.info(f"  All segments lost (>=5 orig):  {all_lost}")
    log.info(f"  50%+ segment reduction:        {half_lost}")
    log.info(f"  Segment count increased:       {increased}")
    log.info(f"  No .bak file:                  {no_bak}")
    log.info("")

    if results:
        log.info("Problem sessions:")
        for name, bak, cur, status in sorted(results):
            log.info(f"  {name}: {bak} → {cur} ({status})")


def retranscribe_all(
    recordings_dir: str,
    use_hotwords: bool = False,
    dry_run: bool = False,
) -> None:
    """Re-transcribe all sessions with updated filters."""
    # Import heavy modules only when actually retranscribing
    from faster_whisper import WhisperModel

    from term_corrector import TermCorrector
    from hotword_loader import HotwordLoader
    import omi_whisper_bridge

    # Load model
    model_size = os.environ.get("WHISPER_MODEL", "large-v3")
    compute_type = os.environ.get("COMPUTE_TYPE", "int8_float16")
    device = os.environ.get("DEVICE", "cuda")

    log.info(f"Loading Whisper model: {model_size} ({compute_type}) on {device}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    log.info("Model loaded")

    # Initialize helpers
    term_corrector = TermCorrector(
        config_dir=os.environ.get("CONFIG_DIR", "./config")
    )
    hotword_loader = HotwordLoader(
        config_dir=os.environ.get("CONFIG_DIR", "./config")
    )

    hotwords_str = ""
    if use_hotwords:
        hotwords_str = hotword_loader.get_hotwords_string()
        log.info(f"Hotwords enabled: {hotword_loader.get_count()} words")
    else:
        log.info("Hotwords disabled (default for batch retranscription)")

    sessions = find_all_sessions(recordings_dir)
    total = len(sessions)
    log.info(f"Found {total} sessions to retranscribe")

    if dry_run:
        log.info("[DRY-RUN] No files will be modified")

    succeeded = 0
    failed = 0
    skipped = 0

    start_time = time.time()

    for i, sess in enumerate(sessions, 1):
        sess_name = sess["name"]
        log.info(f"[{i}/{total}] Processing {sess_name}")

        # Load audio
        pcm_bytes = load_audio_as_pcm(sess["audio"])
        if pcm_bytes is None:
            log.warning(f"  Skipping (audio load failed)")
            skipped += 1
            continue

        # Load existing transcript for metadata
        try:
            with open(sess["transcript"]) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning(f"  Skipping (transcript read failed)")
            skipped += 1
            continue

        old_count = len(existing.get("segments", []))

        if dry_run:
            log.info(f"  [DRY-RUN] Would retranscribe ({old_count} existing segments)")
            succeeded += 1
            continue

        # Backup current transcript (only if no .bak exists yet)
        bak_path = sess["transcript"] + ".bak"
        if not os.path.exists(bak_path):
            shutil.copy2(sess["transcript"], bak_path)

        # Retranscribe
        try:
            new_segments = omi_whisper_bridge.transcribe_audio(
                pcm_bytes,
                model,
                session_offset=0.0,
                hotwords=hotwords_str,
            )
        except Exception as e:
            log.error(f"  Transcription failed: {e}")
            failed += 1
            continue

        new_count = len(new_segments)

        # Update transcript
        existing["segments"] = new_segments
        session_meta = existing.get("session", {})
        session_meta["retranscribed"] = True
        session_meta["retranscribed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        session_meta["hotwords_used"] = use_hotwords
        existing["session"] = session_meta

        with open(sess["transcript"], "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        status = ""
        if new_count == 0 and old_count >= 5:
            status = " *** ALL LOST ***"
        elif old_count > 0 and new_count < old_count * 0.5:
            status = " ** 50%+ lost **"

        log.info(f"  {old_count} → {new_count} segments{status}")
        succeeded += 1

    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info(f"Retranscription complete in {elapsed:.0f}s")
    log.info(f"  Succeeded: {succeeded}")
    log.info(f"  Failed:    {failed}")
    log.info(f"  Skipped:   {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Retranscribe all recording sessions with updated filters"
    )
    parser.add_argument(
        "--restore", action="store_true",
        help="Restore .bak files to transcript.json"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Compare current vs .bak segment counts"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing"
    )
    parser.add_argument(
        "--hotwords", action="store_true",
        help="Enable hotwords (OFF by default for batch)"
    )
    parser.add_argument(
        "--recordings-dir", type=str, default=RECORDINGS_DIR,
        help=f"Recordings directory (default: {RECORDINGS_DIR})"
    )

    args = parser.parse_args()

    if args.restore:
        restore_all(args.recordings_dir, dry_run=args.dry_run)
    elif args.verify:
        verify_all(args.recordings_dir)
    else:
        retranscribe_all(
            recordings_dir=args.recordings_dir,
            use_hotwords=args.hotwords,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
