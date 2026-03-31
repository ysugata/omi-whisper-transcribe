#!/usr/bin/env python3
"""Compare sanitize model quality: Gemini 2.5 Flash vs GPT-5 nano.

Usage:
    python tools/compare_sanitize_models.py [--sessions N] [--output DIR]

Reads API keys from .env (GEMINI_API_KEY, OPENAI_API_KEY).

Runs the same sessions through both models and generates a side-by-side
comparison report (JSON + human-readable markdown).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
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

from training_sanitizer import TrainingSanitizer

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "config"

MODELS = {
    "gemini-2.5-flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "key_arg": "gemini_key",
    },
    "gpt-5-nano": {
        "base_url": "https://api.openai.com/v1/",
        "key_env": "OPENAI_API_KEY",
        "key_arg": "openai_key",
    },
}


def find_sessions(limit: int = 10) -> list[str]:
    """Find sessions that have transcript.json with tags and segments."""
    sessions = []
    if not RECORDINGS_DIR.exists():
        return sessions

    for date_dir in sorted(RECORDINGS_DIR.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for sess_dir in sorted(date_dir.iterdir()):
            if not sess_dir.is_dir():
                continue
            transcript = sess_dir / "transcript.json"
            if not transcript.exists():
                continue
            try:
                with open(transcript) as f:
                    data = json.load(f)
                segs = data.get("segments", [])
                # At least 3 meaningful segments
                meaningful = [s for s in segs if len(s.get("text", "").strip()) > 10]
                if len(meaningful) >= 3:
                    sess_path = f"{date_dir.name}/{sess_dir.name}"
                    sessions.append(sess_path)
            except Exception:
                continue
            if len(sessions) >= limit:
                return sessions
    return sessions


def load_transcript(session_id: str) -> dict | None:
    """Load transcript data for a session."""
    path = RECORDINGS_DIR / session_id / "transcript.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def run_model(sanitizer: TrainingSanitizer, transcript: dict) -> tuple[dict | None, float]:
    """Run sanitizer and return (result, elapsed_seconds)."""
    start = time.time()
    result = sanitizer.sanitize_session(transcript)
    elapsed = time.time() - start
    return result, elapsed


def format_records_summary(result: dict | None) -> str:
    """Create a short summary of records."""
    if result is None:
        return "ERROR (None)"
    risk = result.get("risk_level", "?")
    records = result.get("records", [])
    types = [r.get("type", "?") for r in records]
    return f"risk={risk}, records={len(records)} ({', '.join(types)})"


def generate_report(results: list[dict], output_path: Path) -> None:
    """Generate markdown comparison report."""
    lines = [
        "# Sanitize Model Comparison Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Metric | Gemini 2.5 Flash | GPT-5 nano |",
        "|--------|-----------------|------------|",
    ]

    # Aggregate stats
    stats = {}
    for model_name in MODELS:
        s = {
            "total": 0, "success": 0, "errors": 0,
            "total_time": 0.0, "total_records": 0,
            "risk_high": 0, "risk_medium": 0, "risk_low": 0,
            "types": {"summary": 0, "keypoints": 0, "safe_conversation": 0, "knowledge_card": 0},
        }
        for r in results:
            mr = r["models"].get(model_name, {})
            s["total"] += 1
            s["total_time"] += mr.get("elapsed", 0)
            res = mr.get("result")
            if res is None:
                s["errors"] += 1
            else:
                s["success"] += 1
                risk = res.get("risk_level", "")
                s[f"risk_{risk}"] = s.get(f"risk_{risk}", 0) + 1
                recs = res.get("records", [])
                s["total_records"] += len(recs)
                for rec in recs:
                    t = rec.get("type", "")
                    if t in s["types"]:
                        s["types"][t] += 1
        stats[model_name] = s

    g = stats["gemini-2.5-flash"]
    o = stats["gpt-5-nano"]

    avg_g = g["total_time"] / max(g["total"], 1)
    avg_o = o["total_time"] / max(o["total"], 1)

    lines.append(f"| Sessions | {g['total']} | {o['total']} |")
    lines.append(f"| Success | {g['success']} | {o['success']} |")
    lines.append(f"| Errors | {g['errors']} | {o['errors']} |")
    lines.append(f"| Avg latency | {avg_g:.2f}s | {avg_o:.2f}s |")
    lines.append(f"| Total records | {g['total_records']} | {o['total_records']} |")
    lines.append(f"| Risk: high | {g['risk_high']} | {o['risk_high']} |")
    lines.append(f"| Risk: medium | {g['risk_medium']} | {o['risk_medium']} |")
    lines.append(f"| Risk: low | {g['risk_low']} | {o['risk_low']} |")

    lines.append("")
    lines.append("### Record types")
    lines.append("")
    lines.append("| Type | Gemini 2.5 Flash | GPT-5 nano |")
    lines.append("|------|-----------------|------------|")
    for t in ("summary", "keypoints", "safe_conversation", "knowledge_card"):
        lines.append(f"| {t} | {g['types'][t]} | {o['types'][t]} |")

    # Cost estimation
    lines.append("")
    lines.append("### Cost estimate (per 1000 sessions)")
    lines.append("")
    lines.append("| Model | Input $/1M tokens | Output $/1M tokens | Est. cost/1000 sessions |")
    lines.append("|-------|-------------------|--------------------|-----------------------|")
    lines.append("| Gemini 2.5 Flash | $0.15 | $0.60 | ~$0.15 |")
    lines.append("| GPT-5 nano | $0.10 | $0.40 | ~$0.10 |")

    # Per-session details
    lines.append("")
    lines.append("## Per-session comparison")
    lines.append("")

    for r in results:
        sess = r["session_id"]
        lines.append(f"### Session: `{sess}`")
        lines.append("")

        for model_name in MODELS:
            mr = r["models"].get(model_name, {})
            elapsed = mr.get("elapsed", 0)
            res = mr.get("result")
            lines.append(f"**{model_name}** ({elapsed:.2f}s)")
            lines.append("")

            if res is None:
                lines.append("> ERROR: no result")
                lines.append("")
                continue

            lines.append(f"- Risk: `{res.get('risk_level', '?')}`")
            for rec in res.get("records", []):
                rec_type = rec.get("type", "?")
                if rec_type == "safe_conversation":
                    msgs = rec.get("messages", [])
                    lines.append(f"- **{rec_type}**: {len(msgs)} messages")
                    for msg in msgs[:3]:
                        role = msg.get("role", "?")
                        content = msg.get("content", "")[:100]
                        lines.append(f"  - [{role}] {content}...")
                else:
                    output = rec.get("output", "")[:150]
                    lines.append(f"- **{rec_type}**: {output}...")
            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare sanitize models")
    parser.add_argument("--sessions", type=int, default=10,
                        help="Number of sessions to compare (default: 10)")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help="Output directory for results")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not set in .env")
        sys.exit(1)
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    # Initialize sanitizers
    sanitizers = {}
    for model_name, cfg in MODELS.items():
        key = gemini_key if cfg["key_arg"] == "gemini_key" else openai_key
        sanitizers[model_name] = TrainingSanitizer(
            api_key=key,
            base_url=cfg["base_url"],
            model=model_name,
            timeout=60,
        )
        print(f"Initialized: {model_name} ({cfg['base_url']})")

    # Find sessions
    print(f"\nFinding sessions (limit={args.sessions})...")
    session_ids = find_sessions(limit=args.sessions)
    if not session_ids:
        print("ERROR: No sessions found in recordings/")
        sys.exit(1)
    print(f"Found {len(session_ids)} sessions")

    # Run comparison
    results = []
    for i, sess_id in enumerate(session_ids):
        print(f"\n[{i+1}/{len(session_ids)}] {sess_id}")
        transcript = load_transcript(sess_id)
        if transcript is None:
            print(f"  SKIP: transcript not found")
            continue

        entry = {"session_id": sess_id, "models": {}}

        for model_name, sanitizer in sanitizers.items():
            print(f"  {model_name}...", end=" ", flush=True)
            result, elapsed = run_model(sanitizer, transcript)
            entry["models"][model_name] = {
                "result": result,
                "elapsed": elapsed,
            }
            summary = format_records_summary(result)
            print(f"{elapsed:.2f}s — {summary}")

        results.append(entry)

    # Save raw results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    raw_path = output_dir / f"model_comparison_{ts}.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nRaw results: {raw_path}")

    # Generate report
    report_path = output_dir / f"model_comparison_{ts}.md"
    generate_report(results, report_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
