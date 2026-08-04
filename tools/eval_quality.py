#!/usr/bin/env python3
"""Quality evaluation harness.

システムの品質を数字で定点観測する。モデルや設定を変えたら再実行して比較する。

測定項目:
  1. タグ精度 — 人手確定済み (manually_corrected) セッションをゴールデンセット
     として、現行のタガーで再タグ付けし一致度を測る。
       - conversation_type / importance: 一致率 (accuracy)
       - domains / outcomes: micro Precision / Recall / F1
  2. KC 品質 — config/safe_training_*.jsonl のナレッジカードをサンプリングし、
     kc_judge (LLM) で採点。平均・分布・合格率 (>= KC_MIN_QUALITY) を出す。

レポートは config/eval_reports/ に JSON 保存。Bridge コンテナ内で実行する:

    docker exec omi-whisper-transcribe python3 tools/eval_quality.py
    docker exec omi-whisper-transcribe python3 tools/eval_quality.py --skip-tags
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "./recordings")
CONFIG_DIR = os.environ.get("CONFIG_DIR", "./config")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/")
TAGGING_MODEL = os.environ.get("TAGGING_MODEL", "gemini-3.1-flash-lite")
KC_JUDGE_MODEL = os.environ.get("KC_JUDGE_MODEL", "gpt-5-nano")
KC_MIN_QUALITY = int(os.environ.get("KC_MIN_QUALITY", "3"))


def _resolve(model: str) -> tuple:
    if model.startswith("gpt-"):
        return OPENAI_API_KEY, OPENAI_BASE_URL
    return GEMINI_API_KEY, LLM_BASE_URL


# ---------------------------------------------------------------------------
# Metric helpers (pure — unit tested)
# ---------------------------------------------------------------------------

def set_prf(golden_sets: list, predicted_sets: list) -> dict:
    """Micro precision/recall/F1 over aligned lists of label sets."""
    tp = fp = fn = 0
    for g, p in zip(golden_sets, predicted_sets):
        g, p = set(g), set(p)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn,
    }


def accuracy(golden: list, predicted: list) -> dict:
    pairs = [(g, p) for g, p in zip(golden, predicted) if g]
    if not pairs:
        return {"accuracy": None, "n": 0}
    correct = sum(1 for g, p in pairs if g == p)
    return {"accuracy": round(correct / len(pairs), 3), "n": len(pairs)}


def kc_score_summary(scores: list) -> dict:
    """Summarize a list of int scores (1-5)."""
    valid = [s for s in scores if isinstance(s, int)]
    if not valid:
        return {"n": 0}
    dist = Counter(valid)
    return {
        "n": len(valid),
        "mean": round(sum(valid) / len(valid), 2),
        "distribution": {str(k): dist.get(k, 0) for k in range(1, 6)},
        "pass_rate": round(
            sum(1 for s in valid if s >= KC_MIN_QUALITY) / len(valid), 3
        ),
        "min_quality": KC_MIN_QUALITY,
        "judge_failures": len(scores) - len(valid),
    }


# ---------------------------------------------------------------------------
# Golden set collection
# ---------------------------------------------------------------------------

def collect_golden_sessions(limit: int) -> list:
    """人手確定済み (manually_corrected) の V2 タグ付きセッションを集める。"""
    golden = []
    base = Path(RECORDINGS_DIR)
    if not base.is_dir():
        return golden
    for day_dir in sorted(base.iterdir(), reverse=True):
        if not day_dir.is_dir():
            continue
        for time_dir in sorted(day_dir.iterdir(), reverse=True):
            transcript_path = time_dir / "transcript.json"
            if not transcript_path.exists():
                continue
            try:
                with open(transcript_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            tags = data.get("session", {}).get("tags", {}) or {}
            if not tags.get("manually_corrected"):
                continue
            if not tags.get("conversation_type"):
                continue
            segments = data.get("segments", [])
            if len(segments) < 3:
                continue
            golden.append({
                "session_id": f"{day_dir.name}/{time_dir.name}",
                "segments": segments,
                "audio_scene": data.get("session", {}).get("audio_scene"),
                "tags": tags,
            })
            if len(golden) >= limit:
                return golden
    return golden


def eval_tags(limit: int) -> dict:
    from session_tagger import SessionTagger

    key, base_url = _resolve(TAGGING_MODEL)
    if not key:
        return {"error": f"no API key for {TAGGING_MODEL}"}

    tagger = SessionTagger(
        api_key=key, base_url=base_url, model=TAGGING_MODEL,
        taxonomy_path=os.path.join(CONFIG_DIR, "tag_taxonomy.json"),
        rag_classifier=None,  # 常に LLM 経路で測る (RAG の当たり外れを排除)
    )

    golden = collect_golden_sessions(limit)
    if not golden:
        return {"error": "no manually_corrected sessions found (review queue で承認するとゴールデンセットが増えます)"}

    g_conv, p_conv = [], []
    g_imp, p_imp = [], []
    g_domains, p_domains = [], []
    g_outcomes, p_outcomes = [], []
    failures = 0

    for i, s in enumerate(golden, 1):
        print(f"  [{i}/{len(golden)}] {s['session_id']} ... ", end="", flush=True)
        predicted = tagger.tag_session(s["segments"], audio_scene=s["audio_scene"])
        if not predicted:
            failures += 1
            print("FAILED")
            continue
        gt = s["tags"]
        g_conv.append(gt.get("conversation_type") or "")
        p_conv.append(predicted.get("conversation_type") or "")
        g_imp.append(gt.get("importance") or "")
        p_imp.append(predicted.get("importance") or "")
        g_domains.append(gt.get("domains") or [])
        p_domains.append(predicted.get("domains") or [])
        g_outcomes.append(gt.get("outcomes") or [])
        p_outcomes.append(predicted.get("outcomes") or [])
        print("ok")

    return {
        "model": TAGGING_MODEL,
        "golden_sessions": len(golden),
        "tagger_failures": failures,
        "conversation_type": accuracy(g_conv, p_conv),
        "importance": accuracy(g_imp, p_imp),
        "domains": set_prf(g_domains, p_domains),
        "outcomes": set_prf(g_outcomes, p_outcomes),
    }


def eval_kc_quality(sample_size: int, seed: int = 42) -> dict:
    from kc_judge import KCJudge

    key, base_url = _resolve(KC_JUDGE_MODEL)
    if not key:
        return {"error": f"no API key for {KC_JUDGE_MODEL}"}

    kcs = []
    for fp in sorted(glob.glob(os.path.join(CONFIG_DIR, "safe_training_*.jsonl"))):
        try:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for kc in rec.get("knowledge_cards") or []:
                        if isinstance(kc, dict) and kc.get("question"):
                            kcs.append(kc)
                    kc1 = rec.get("knowledge_card")
                    if isinstance(kc1, dict) and kc1.get("question"):
                        kcs.append(kc1)
                    if rec.get("type") == "knowledge_card" and rec.get("input"):
                        kcs.append({
                            "question": rec.get("input"),
                            "answer": rec.get("output", ""),
                        })
        except OSError:
            continue

    if not kcs:
        return {"error": "no knowledge cards found"}

    random.Random(seed).shuffle(kcs)
    sample = kcs[:sample_size]
    print(f"  judging {len(sample)}/{len(kcs)} KCs with {KC_JUDGE_MODEL} ...")

    judge = KCJudge(api_key=key, base_url=base_url, model=KC_JUDGE_MODEL)
    results = judge.judge(sample)
    scores = [(r or {}).get("score") for r in results]

    worst = sorted(
        (
            {"question": kc.get("question", "")[:60], "score": r["score"],
             "reason": r.get("reason", "")}
            for kc, r in zip(sample, results) if r
        ),
        key=lambda x: x["score"],
    )[:5]

    summary = kc_score_summary(scores)
    summary.update({
        "judge_model": KC_JUDGE_MODEL,
        "corpus_total_kcs": len(kcs),
        "sampled": len(sample),
        "worst_examples": worst,
    })
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Quality evaluation harness")
    ap.add_argument("--tags-limit", type=int, default=30,
                    help="ゴールデンセットの最大セッション数")
    ap.add_argument("--kc-sample", type=int, default=40,
                    help="KC 品質採点のサンプル数")
    ap.add_argument("--skip-tags", action="store_true")
    ap.add_argument("--skip-kc", action="store_true")
    args = ap.parse_args()

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tagging_model": TAGGING_MODEL,
        "kc_judge_model": KC_JUDGE_MODEL,
    }

    if not args.skip_tags:
        print("== Tag accuracy (golden = manually_corrected sessions) ==")
        report["tag_eval"] = eval_tags(args.tags_limit)
        print(json.dumps(report["tag_eval"], ensure_ascii=False, indent=2))

    if not args.skip_kc:
        print("== KC quality (LLM judge) ==")
        report["kc_eval"] = eval_kc_quality(args.kc_sample)
        print(json.dumps(report["kc_eval"], ensure_ascii=False, indent=2))

    out_dir = Path(CONFIG_DIR) / "eval_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nreport saved: {out_path}")


if __name__ == "__main__":
    main()
