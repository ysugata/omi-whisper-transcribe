#!/usr/bin/env python3
"""LLM Corrector optimization: Gemini prompt/format tuning.

Compare:
  [A] Gemini Flash Lite — 既存 (全行エコー, temperature=0)
  [B] Gemini Flash Lite — 最適化 (json_schema, 差分のみ, 強化プロンプト)
  [C] Gemini Flash (非Lite) — 最適化 (json_schema, 差分のみ, 強化プロンプト)
"""

import json
import os
import re
import time
import difflib
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ORIGINAL_PROMPT = """あなたは音声文字起こしテキストの校正者です。
番号付きの発話テキストが入力されます。全体の文脈と参考用語リストを活用して以下のルールに従ってください:

1. 固有名詞・技術用語・社名・プロダクト名の誤認識を修正する
2. カタカナで書かれた技術用語を正しい英語表記に修正する（例: ドッカー → Docker）
3. 音声認識で脱落した可能性のある単語を前後の文脈から補完する（例: 「3060のVRAM」→「RTX3060のVRAM」）
4. 参考用語リストの単語が、音声認識で似た別の単語に誤認識されていないか注意する（例: Nginx が「Linux」に、gRPC が「GRPC」に誤認識される等）
5. 発言の意味・内容・語順は絶対に変えない
6. 明らかな誤りだけを修正し、曖昧な箇所はそのまま残す
7. 文末にカンマで区切られた意味不明な短い断片が続く場合（音声認識ノイズ）、その断片部分を削除する
8. 句読点の追加・削除はしない
9. 入力と同じ [番号] 形式で、修正後のテキストだけを返す
10. 修正が不要な行も入力をそのまま返す"""

OPTIMIZED_PROMPT = """あなたは日本語の音声文字起こし (ASR) テキストの校正者です。
修正が必要な行のみを corrections 配列に返してください。修正不要な行は返さないでください。

## 修正対象 (具体例付き)

### 1. 単語の脱落・途切れ (ASR が途中で切った)
- 「100年クエス」→「100年クエスト」
- 「仲いいとことか」→「仲いいところとか」
- 「諦まないわえ」→「諦めないわ」(活用形の誤り)

### 2. カタカナ技術用語 → 正しい表記
- 「ドッカー」→「Docker」
- 「クーバネティス」→「Kubernetes」
- 「ジュピターノートブック」→「Jupyter Notebook」

### 3. 音が似た別の語への誤認識
- 「フェアディテール」→「フェアリーテイル」(固有名詞)
- 「足ごとにも及ばん」→「足元にも及ばん」(慣用句)
- 「マホウ」→「魔法」(カタカナ→漢字)

### 4. カンマ区切りノイズ断片
- 「本文、トカ、ナンカ、」→「本文」(末尾ノイズ除去)

## 禁止事項
- **句読点の変換禁止**: ?, ! はそのまま残す。？, ！ に変えない。逆も同様
- **句読点の追加・削除禁止**: 「。」「、」を足したり消したりしない
- 発言の意味・内容・語順を変えない
- カタカナの会話語をローマ字や英語に変えない (例: 「スターバックス」をそのまま残す)
- 叫び声・感嘆表現はそのまま残す (例: 「ギルダァァァ!」は変えない)
- 曖昧な箇所は修正しない — 確信がある誤りだけ修正する
- 参考用語リストにない語を推測で修正しない"""

CORRECTIONS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "asr_corrections",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "corrections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line": {"type": "integer"},
                            "original": {"type": "string"},
                            "corrected": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["line", "original", "corrected", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["corrections"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_known_terms():
    tc_path = CONFIG_DIR / "term_corrections.json"
    if not tc_path.exists():
        return []
    data = json.loads(tc_path.read_text())
    if isinstance(data, dict):
        return list(set(data.values()))[:30]
    return []


def load_session_segments(session_path: str) -> list[dict]:
    transcript_path = RECORDINGS_DIR / session_path / "transcript.json"
    if not transcript_path.exists():
        return []
    data = json.loads(transcript_path.read_text())
    return data.get("segments", [])


def build_batch_input(segments: list[dict], known_terms: list[str]) -> tuple[str, list[str]]:
    lines = []
    original_texts = []
    for seg in segments:
        text = seg.get("text", "")
        if len(text) >= 5:
            lines.append(f"[{len(lines) + 1}] {text}")
            original_texts.append(text)
    parts = []
    if known_terms:
        parts.append(f"参考用語リスト: {', '.join(known_terms)}")
    parts.append("\n".join(lines))
    return "\n".join(parts), original_texts


def parse_numbered_response(raw: str, expected_count: int) -> dict[int, str]:
    result = {}
    pattern = re.compile(r'^\[(\d+)\]\s*(.*)')
    for line in raw.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            idx = int(m.group(1)) - 1
            text = m.group(2).strip()
            if 0 <= idx < expected_count and text:
                result[idx] = text
    return result


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_gemini_original(user_msg: str) -> dict:
    """[A] Gemini Flash Lite — 既存方式."""
    from openai import OpenAI
    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=30,
    )
    t0 = time.time()
    resp = client.chat.completions.create(
        model="gemini-2.5-flash-lite",
        messages=[
            {"role": "system", "content": ORIGINAL_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=4096,
    )
    elapsed = time.time() - t0
    return {
        "text": resp.choices[0].message.content or "",
        "time": elapsed,
        "usage": {
            "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "output_tokens": getattr(resp.usage, "completion_tokens", 0),
        },
    }


def run_gemini_optimized(user_msg: str, model: str = "gemini-2.5-flash-lite") -> dict:
    """[B/C] Gemini — 最適化 (json_schema + 強化プロンプト)."""
    from openai import OpenAI
    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=30,
    )
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": OPTIMIZED_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=CORRECTIONS_SCHEMA,
        temperature=0,
        max_tokens=4096,
    )
    elapsed = time.time() - t0
    return {
        "text": resp.choices[0].message.content or "",
        "time": elapsed,
        "usage": {
            "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "output_tokens": getattr(resp.usage, "completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sessions = [
        "2026-03-31/21-59-36",  # 58 segs
        "2026-03-31/23-46-15",  # 46 segs
        "2026-03-31/23-31-03",  # 39 segs
        "2026-03-31/22-36-12",  # 33 segs
        "2026-03-31/22-06-15",  # 25 segs
    ]

    known_terms = load_known_terms()
    print(f"Known terms: {len(known_terms)} loaded")
    print(f"Comparison:")
    print(f"  [A] Gemini Flash Lite — 既存 (全行エコー)")
    print(f"  [B] Gemini Flash Lite — 最適化 (json_schema + 強化プロンプト)")
    print(f"  [C] Gemini Flash      — 最適化 (json_schema + 強化プロンプト)")
    print("=" * 100)

    all_results = []
    models_config = [
        ("A_original", "Gemini Lite 既存", lambda msg: run_gemini_original(msg), "numbered"),
        ("B_lite_opt", "Gemini Lite 最適化", lambda msg: run_gemini_optimized(msg, "gemini-2.5-flash-lite"), "json"),
        ("C_flash_opt", "Gemini Flash 最適化", lambda msg: run_gemini_optimized(msg, "gemini-2.5-flash"), "json"),
    ]

    for sess_path in sessions:
        print(f"\n--- Session: {sess_path} ---")
        segments = load_session_segments(sess_path)
        if not segments:
            continue

        user_msg, original_lines = build_batch_input(segments, known_terms)
        line_count = len(original_lines)
        print(f"  Segments: {len(segments)}, Eligible lines: {line_count}")

        result = {"session": sess_path, "segments": len(segments), "lines": line_count}

        for mk, mn, runner, fmt in models_config:
            try:
                print(f"  [{mk}] {mn}... ", end="", flush=True)
                r = runner(user_msg)

                if fmt == "numbered":
                    r_map = parse_numbered_response(r["text"], line_count)
                    diffs = []
                    for idx, corrected in r_map.items():
                        if idx < len(original_lines) and corrected != original_lines[idx]:
                            diffs.append({
                                "line": idx + 1,
                                "original": original_lines[idx],
                                "corrected": corrected,
                            })
                else:
                    data = json.loads(r["text"]) if r["text"] else {"corrections": []}
                    diffs = data.get("corrections", [])

                print(f"{r['time']:.2f}s, {len(diffs)} fixes, in={r['usage'].get('input_tokens', 0)} out={r['usage'].get('output_tokens', 0)}")
                result[mk] = {
                    "time": round(r["time"], 2),
                    "fixes": len(diffs),
                    "usage": r["usage"],
                    "diffs": diffs,
                }
            except Exception as e:
                print(f"ERROR: {e}")
                result[mk] = {"error": str(e)}

        all_results.append(result)

    # Summary
    model_keys = [("A_original", "Gemini Lite 既存"), ("B_lite_opt", "Gemini Lite 最適化"), ("C_flash_opt", "Gemini Flash 最適化")]

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Session':<25} {'Model':<25} {'Time':>6} {'Fixes':>6} {'InTok':>7} {'OutTok':>7}")
    print("-" * 100)

    totals = {k: {"time": 0, "fixes": 0, "in": 0, "out": 0, "n": 0} for k, _ in model_keys}

    for r in all_results:
        for mk, mn in model_keys:
            m = r.get(mk, {})
            if "error" in m:
                print(f"{r['session']:<25} {mn:<25} {'ERROR':>6}")
                continue
            t = m.get("time", 0)
            f = m.get("fixes", 0)
            it = m.get("usage", {}).get("input_tokens", 0)
            ot = m.get("usage", {}).get("output_tokens", 0)
            print(f"{r['session']:<25} {mn:<25} {t:>5.1f}s {f:>5} {it:>7} {ot:>7}")
            totals[mk]["time"] += t
            totals[mk]["fixes"] += f
            totals[mk]["in"] += it
            totals[mk]["out"] += ot
            totals[mk]["n"] += 1

    print("-" * 100)
    for mk, mn in model_keys:
        t = totals[mk]
        if t["n"] > 0:
            avg = t["time"] / t["n"]
            print(f"{'TOTAL':<25} {mn:<25} {t['time']:>5.1f}s {t['fixes']:>5} {t['in']:>7} {t['out']:>7}  (avg {avg:.1f}s)")

    # Correction details
    print("\n" + "=" * 100)
    print("CORRECTION DETAILS")
    print("=" * 100)

    for r in all_results:
        has_any = any(r.get(mk, {}).get("diffs") for mk, _ in model_keys)
        if not has_any:
            continue

        print(f"\n--- {r['session']} ({r['segments']} segments) ---")

        for mk, mn in model_keys:
            m = r.get(mk, {})
            diffs = m.get("diffs", [])
            if not diffs:
                continue

            print(f"\n  [{mn}] {len(diffs)} correction(s):")
            for d in diffs:
                line = d.get("line", "?")
                orig = d.get("original", "")
                corr = d.get("corrected", "")
                reason = d.get("reason", "")
                print(f"    [{line}] {orig}")
                print(f"      → {corr}")
                if reason:
                    print(f"      理由: {reason}")

    # Save
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = CONFIG_DIR / f"compare_llm_opt_{ts}.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
