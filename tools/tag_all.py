#!/usr/bin/env python3
"""Batch session tagging CLI tool.

Usage:
    python tools/tag_all.py [--force] [--dry-run] [--recordings-dir ./recordings]

Tags all untagged sessions (or all sessions with --force) using the
SessionTagger (Gemini API).
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from session_tagger import SessionTagger


def find_sessions(recordings_dir: str, force: bool = False) -> list[dict]:
    """Find sessions to tag."""
    sessions = []
    base = Path(recordings_dir)
    if not base.is_dir():
        print(f"Error: {recordings_dir} is not a directory")
        return sessions

    for day_dir in sorted(base.iterdir()):
        if not day_dir.is_dir():
            continue
        for time_dir in sorted(day_dir.iterdir()):
            if not time_dir.is_dir():
                continue
            tp = time_dir / "transcript.json"
            if not tp.exists():
                continue

            try:
                with open(tp, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if not data.get("segments"):
                continue

            if not force and data.get("session", {}).get("session_tagged"):
                continue

            sessions.append({
                "path": f"{day_dir.name}/{time_dir.name}",
                "transcript_path": tp,
                "data": data,
            })

    return sessions


def main():
    parser = argparse.ArgumentParser(description="Batch session tagging")
    parser.add_argument("--force", action="store_true", help="Re-tag all sessions")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--recordings-dir", default="./recordings", help="Recordings directory")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    model = os.environ.get("TAGGING_MODEL", "gemini-2.5-flash-lite")

    sessions = find_sessions(args.recordings_dir, force=args.force)
    print(f"Found {len(sessions)} session(s) to tag")

    if not sessions:
        return

    if args.dry_run:
        for s in sessions:
            seg_count = len(s["data"].get("segments", []))
            print(f"  {s['path']} ({seg_count} segments)")
        return

    tagger = SessionTagger(api_key=api_key, model=model)
    if not tagger.enabled:
        print("Error: SessionTagger initialization failed")
        sys.exit(1)

    processed = 0
    failed = 0

    for i, sess in enumerate(sessions, 1):
        print(f"[{i}/{len(sessions)}] {sess['path']}...", end=" ", flush=True)

        try:
            segments = sess["data"].get("segments", [])
            tags = tagger.tag_session(segments)

            if tags:
                # Backup
                bak_path = str(sess["transcript_path"]) + ".bak"
                shutil.copy2(sess["transcript_path"], bak_path)

                data = sess["data"]
                data["session"]["tags"] = tags
                data["session"]["session_tagged"] = True

                with open(sess["transcript_path"], "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                processed += 1
                print(f"OK ({tags.get('activity', '?')}, {tags.get('mood', '?')})")
            else:
                failed += 1
                print("SKIP (tagger returned None)")

        except Exception as e:
            failed += 1
            print(f"ERROR: {e}")

    print(f"\nDone: {processed} tagged, {failed} failed, {len(sessions)} total")


if __name__ == "__main__":
    main()
