#!/usr/bin/env python3
"""3-way sanitize quality comparison.

Runs the same sessions through 3 configurations:
  A-old)  Gemini 2.5 Flash — old prompt + json_object (baseline)
  A-new)  Gemini 2.5 Flash — new prompt + json_schema (improved)
  C)      GPT-5 nano       — Responses API (reference)

Usage:
    python tools/compare_3way.py [--sessions N]
"""

import argparse
import json
import os
import sys
import time
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

from training_sanitizer import TrainingSanitizer
from responses_sanitizer import ResponsesSanitizer
from sanitize_prompt import SANITIZE_INSTRUCTIONS
from sanitize_schema import SANITIZE_SCHEMA, validate_schema, run_quality_gates

RECORDINGS_DIR = PROJECT_ROOT / "recordings"
OUTPUT_DIR = PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Session loader
# ---------------------------------------------------------------------------

def find_sessions(limit: int = 5, specific: list[str] | None = None) -> list[str]:
    if specific:
        # Verify they exist
        return [s for s in specific if (RECORDINGS_DIR / s / "transcript.json").exists()]

    sessions = []
    if not RECORDINGS_DIR.exists():
        return sessions

    candidates = []
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
                meaningful = [s for s in segs if len(s.get("text", "").strip()) > 10]
                total_chars = sum(len(s.get("text", "")) for s in segs)
                if len(meaningful) >= 5 and total_chars > 200:
                    candidates.append((len(meaningful), total_chars, f"{date_dir.name}/{sess_dir.name}"))
            except Exception:
                continue

    # Sort by quality (most segments first)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[2] for c in candidates[:limit]]


def load_transcript(session_id: str) -> dict | None:
    path = RECORDINGS_DIR / session_id / "transcript.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def build_raw_text(transcript: dict) -> str:
    """Build raw text (same as bridge _build_raw_text_from_transcript)."""
    parts = []
    session = transcript.get("session", {})
    tags = session.get("tags", {})
    if tags and tags.get("summary"):
        parts.append(f"[要約] {tags['summary']}")
    segments = transcript.get("segments", [])
    for seg in segments[:80]:
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = seg.get("person_name") or seg.get("speaker", "")
        if speaker:
            parts.append(f"[{speaker}] {text}")
        else:
            parts.append(text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Improved Gemini: new prompt + json_schema via Chat Completions
# ---------------------------------------------------------------------------

# Convert Responses API SANITIZE_SCHEMA to Chat Completions response_format
_GEMINI_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": SANITIZE_SCHEMA["name"],
        "strict": SANITIZE_SCHEMA["strict"],
        "schema": SANITIZE_SCHEMA["schema"],
    },
}


def run_gemini_improved(client, raw_text: str, session_id: str) -> dict | None:
    """Call Gemini with new prompt + json_schema response format."""
    if len(raw_text.strip()) < 20:
        return None

    if len(raw_text) > 4000:
        raw_text = raw_text[:4000]

    user_input = (
        f"session_id: {session_id}\n"
        f"language: ja\n"
        f"---\n"
        f"{raw_text}"
    )

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": SANITIZE_INSTRUCTIONS},
                {"role": "user", "content": user_input},
            ],
            response_format=_GEMINI_RESPONSE_FORMAT,
            temperature=0.3,
            max_tokens=8192,
        )
        content = response.choices[0].message.content
        data = json.loads(content)

        if not validate_schema(data):
            print(f"    [schema validation failed]")
            return None

        warnings = run_quality_gates(data)
        return {"data": data, "warnings": warnings}

    except Exception as e:
        print(f"    [ERROR: {e}]")
        return None


# ---------------------------------------------------------------------------
# Normalize outputs for comparison
# ---------------------------------------------------------------------------

def normalize_training_result(result: dict | None) -> dict | None:
    if result is None:
        return None
    risk = result.get("risk_level", "?")
    records = result.get("records", [])
    summary = None
    keypoints = []
    conversation = []
    knowledge_card = None
    for rec in records:
        t = rec.get("type")
        if t == "summary":
            summary = {"input": rec.get("input", ""), "output": rec.get("output", "")}
        elif t == "keypoints":
            keypoints = [rec.get("output", "")]
        elif t == "safe_conversation":
            for msg in rec.get("messages", []):
                conversation.append({
                    "speaker": msg.get("role", "unknown"),
                    "text": msg.get("content", ""),
                })
        elif t == "knowledge_card":
            knowledge_card = {"question": rec.get("input", ""), "answer": rec.get("output", "")}
    return {
        "risk": risk, "summary": summary, "keypoints": keypoints,
        "conversation_turns": len(conversation), "conversation": conversation[:6],
        "knowledge_card": knowledge_card, "record_count": len(records),
    }


def normalize_new_schema(result) -> dict | None:
    """Normalize output from improved Gemini or Responses API."""
    if result is None:
        return None
    # Handle SanitizeResult objects (from ResponsesSanitizer)
    if hasattr(result, "data"):
        data = result.data
        warnings = result.warnings
    elif isinstance(result, dict):
        data = result.get("data", result)
        warnings = result.get("warnings", [])
    else:
        return None

    conv = data.get("safe_conversation", [])
    return {
        "risk": data.get("risk", "?"),
        "summary": data.get("summary"),
        "keypoints": data.get("keypoints", []),
        "conversation_turns": len(conv),
        "conversation": conv[:6],
        "knowledge_card": data.get("knowledge_card"),
        "issues_detected": data.get("issues_detected", []),
        "warnings": warnings,
    }


def print_result(label: str, norm: dict | None) -> None:
    if norm is None:
        print(f"  [{label}] FAILED")
        return
    print(f"  [{label}] risk={norm['risk']}")
    if norm.get("summary"):
        out = norm["summary"].get("output", "")[:120]
        print(f"    summary: {out}")
    kp = norm.get("keypoints", [])
    if kp:
        for k in kp[:3]:
            print(f"    kp: {str(k)[:100]}")
    conv = norm.get("conversation", [])
    if conv:
        print(f"    conv: {norm['conversation_turns']} turns")
        for c in conv[:2]:
            sp = c.get("speaker") or c.get("role", "?")
            txt = c.get("text") or c.get("content", "")
            print(f"      [{sp}] {txt[:80]}")
    kc = norm.get("knowledge_card")
    if kc:
        print(f"    kcard: Q={kc.get('question', '')[:60]}")
    issues = norm.get("issues_detected")
    if issues:
        print(f"    issues: {issues}")
    warnings = norm.get("warnings")
    if warnings:
        print(f"    warnings: {warnings}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3-way sanitize comparison")
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--ids", nargs="*", help="Specific session IDs to compare")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not set"); sys.exit(1)
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)

    print("=== Initializing ===")

    # A-old: Gemini + old prompt + json_object
    san_old = TrainingSanitizer(
        api_key=gemini_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
        timeout=60,
    )
    print("  A-old) Gemini 2.5 Flash + old prompt + json_object")

    # A-new: Gemini + new prompt + json_schema
    from openai import OpenAI
    gemini_client = OpenAI(
        api_key=gemini_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=60,
    )
    print("  A-new) Gemini 2.5 Flash + new prompt + json_schema")

    # C: GPT-5 nano + Responses API
    san_resp = ResponsesSanitizer(
        api_key=openai_key,
        model="gpt-5-nano",
        reasoning_effort="low",
        max_output_tokens=1800,
    )
    print("  C)     GPT-5 nano + Responses API")

    print(f"\n=== Finding sessions ===")
    session_ids = find_sessions(limit=args.sessions, specific=args.ids)
    if not session_ids:
        print("ERROR: No sessions found"); sys.exit(1)
    print(f"Found {len(session_ids)} sessions\n")

    CONFIGS = ["A-old", "A-new", "C"]
    results = []
    stats = {c: {"times": [], "success": 0, "errors": 0, "risks": {}} for c in CONFIGS}

    for i, sess_id in enumerate(session_ids):
        print(f"{'='*60}")
        print(f"[{i+1}/{len(session_ids)}] Session: {sess_id}")
        print(f"{'='*60}")

        transcript = load_transcript(sess_id)
        if transcript is None:
            print("  SKIP"); continue

        segs = transcript.get("segments", [])
        texts = [s.get("text", "") for s in segs[:3]]
        print(f"  Input: {len(segs)} segs — {' / '.join(t[:30] for t in texts)}...")
        print()

        raw_text = build_raw_text(transcript)
        entry = {"session_id": sess_id, "input_segments": len(segs), "configs": {}}

        # --- A-old ---
        label = "A-old"
        print(f"  {label}) Gemini + old prompt...", end=" ", flush=True)
        t0 = time.time()
        res = san_old.sanitize_session(transcript)
        elapsed = time.time() - t0
        norm = normalize_training_result(res)
        entry["configs"][label] = {"elapsed": elapsed, "result": norm}
        stats[label]["times"].append(elapsed)
        if norm:
            stats[label]["success"] += 1
            stats[label]["risks"][norm["risk"]] = stats[label]["risks"].get(norm["risk"], 0) + 1
            print(f"{elapsed:.1f}s — risk={norm['risk']}")
        else:
            stats[label]["errors"] += 1
            print(f"{elapsed:.1f}s — FAILED")

        # --- A-new ---
        label = "A-new"
        print(f"  {label}) Gemini + new prompt + schema...", end=" ", flush=True)
        t0 = time.time()
        res = run_gemini_improved(gemini_client, raw_text, sess_id)
        elapsed = time.time() - t0
        norm_new = normalize_new_schema(res)
        entry["configs"][label] = {"elapsed": elapsed, "result": norm_new}
        stats[label]["times"].append(elapsed)
        if norm_new:
            stats[label]["success"] += 1
            stats[label]["risks"][norm_new["risk"]] = stats[label]["risks"].get(norm_new["risk"], 0) + 1
            print(f"{elapsed:.1f}s — risk={norm_new['risk']}, kp={len(norm_new.get('keypoints', []))}")
        else:
            stats[label]["errors"] += 1
            print(f"{elapsed:.1f}s — FAILED")

        # --- C ---
        label = "C"
        print(f"  {label})     GPT-5 nano + Responses API...", end=" ", flush=True)
        t0 = time.time()
        res = san_resp.sanitize(sess_id, raw_text, "ja")
        elapsed = time.time() - t0
        norm_c = normalize_new_schema(res)
        entry["configs"][label] = {"elapsed": elapsed, "result": norm_c}
        stats[label]["times"].append(elapsed)
        if norm_c:
            stats[label]["success"] += 1
            stats[label]["risks"][norm_c["risk"]] = stats[label]["risks"].get(norm_c["risk"], 0) + 1
            print(f"{elapsed:.1f}s — risk={norm_c['risk']}, kp={len(norm_c.get('keypoints', []))}")
        else:
            stats[label]["errors"] += 1
            print(f"{elapsed:.1f}s — FAILED")

        # --- Side by side ---
        print()
        print_result("A-old", normalize_training_result(san_old.sanitize_session(transcript) if False else entry["configs"]["A-old"]["result"]))
        print_result("A-new", entry["configs"]["A-new"]["result"])
        print_result("C-GPT", entry["configs"]["C"]["result"])
        print()

        results.append(entry)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    header = f"{'Metric':<25} {'A-old (Gemini old)':>20} {'A-new (Gemini new)':>20} {'C (GPT Resp)':>20}"
    print(header)
    print("-" * len(header))

    for label in CONFIGS:
        s = stats[label]
        n = len(s["times"])
        avg = sum(s["times"]) / max(n, 1)
        s["avg_time"] = avg

    print(f"{'Success':<25} {stats['A-old']['success']:>20} {stats['A-new']['success']:>20} {stats['C']['success']:>20}")
    print(f"{'Errors':<25} {stats['A-old']['errors']:>20} {stats['A-new']['errors']:>20} {stats['C']['errors']:>20}")
    print(f"{'Avg latency':<25} {stats['A-old']['avg_time']:>19.1f}s {stats['A-new']['avg_time']:>19.1f}s {stats['C']['avg_time']:>19.1f}s")

    for risk in ("low", "medium", "high"):
        vals = [str(stats[c]["risks"].get(risk, 0)) for c in CONFIGS]
        print(f"{'Risk: ' + risk:<25} {vals[0]:>20} {vals[1]:>20} {vals[2]:>20}")

    # Cost comparison
    print(f"\n{'='*60}")
    print("COST (per 1000 sessions, ~500 input + ~300 output tokens/session)")
    print(f"{'='*60}\n")
    print(f"  Gemini 2.5 Flash : $0.15/1M in + $0.60/1M out ≈ $0.255/1000sess")
    print(f"  GPT-5 nano       : $0.10/1M in + $0.40/1M out ≈ $0.170/1000sess")
    print(f"  → GPT-5 nano is ~33% cheaper")

    # Risk accuracy check
    print(f"\n{'='*60}")
    print("RISK ACCURACY NOTES")
    print(f"{'='*60}\n")
    for r in results:
        sess = r["session_id"]
        risks = {c: r["configs"][c]["result"]["risk"] if r["configs"][c]["result"] else "FAIL" for c in CONFIGS}
        if len(set(risks.values())) > 1:
            print(f"  MISMATCH: {sess}")
            for c in CONFIGS:
                print(f"    {c}: {risks[c]}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"compare_3way_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nRaw results: {out_path}")


if __name__ == "__main__":
    main()
