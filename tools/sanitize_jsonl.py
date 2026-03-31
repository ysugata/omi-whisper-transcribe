#!/usr/bin/env python3
"""JSONL → JSONL sanitize パイプライン (CLI).

OpenAI Responses API (Structured Outputs) を使い、
ASR テキストを匿名化・構造化された JSONL に変換する。

入力 JSONL (1行1レコード):
  {"session_id": "2026-03-31/09-35-13", "raw_text": "会話全文...", "language": "ja"}

出力 JSONL (1行1レコード):
  {"session_id": "...", "risk": "low", "summary": {...}, ...}

使い方:
  python tools/sanitize_jsonl.py input.jsonl output.jsonl
  python tools/sanitize_jsonl.py input.jsonl output.jsonl --model gpt-5-nano --effort medium
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# .env ファイルから環境変数を読み込む
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

from responses_sanitizer import ResponsesSanitizer  # noqa: E402

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sanitize_jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JSONL sanitize pipeline using OpenAI Responses API",
    )
    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("output", help="Output JSONL file path")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5-nano"),
        help="OpenAI model (default: OPENAI_MODEL env or gpt-5-nano)",
    )
    parser.add_argument(
        "--effort",
        default=os.environ.get("OPENAI_REASONING_EFFORT", "low"),
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort (default: OPENAI_REASONING_EFFORT env or low)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "1800")),
        help="Max output tokens (default: OPENAI_MAX_OUTPUT_TOKENS env or 1800)",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=int(os.environ.get("OPENAI_TIMEOUT_MS", "60000")),
        help="Timeout in milliseconds (default: OPENAI_TIMEOUT_MS env or 60000)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per record (default: 3)",
    )
    args = parser.parse_args()

    # API キー確認
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.error("OPENAI_API_KEY not set. Set it in .env or environment.")
        sys.exit(1)

    # Sanitizer 初期化
    sanitizer = ResponsesSanitizer(
        api_key=api_key,
        model=args.model,
        reasoning_effort=args.effort,
        max_output_tokens=args.max_tokens,
        timeout_ms=args.timeout_ms,
        max_retries=args.max_retries,
    )

    if not sanitizer.enabled:
        log.error("ResponsesSanitizer initialization failed")
        sys.exit(1)

    # 入力ファイル読み込み
    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)

    records = []
    with open(input_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "session_id" not in record or "raw_text" not in record:
                    log.warning(f"Line {line_no}: missing session_id or raw_text, skipping")
                    continue
                records.append(record)
            except json.JSONDecodeError as e:
                log.warning(f"Line {line_no}: invalid JSON: {e}")

    log.info(f"Loaded {len(records)} records from {input_path}")

    if not records:
        log.warning("No valid records to process")
        sys.exit(0)

    # 処理実行
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": len(records),
        "success": 0,
        "failed": 0,
        "warnings_total": 0,
        "risk_high": 0,
        "risk_medium": 0,
        "risk_low": 0,
    }

    t_start = time.time()

    with open(output_path, "w") as out_f:
        for i, record in enumerate(records):
            session_id = record["session_id"]
            raw_text = record["raw_text"]
            language = record.get("language", "ja")

            log.info(f"[{i + 1}/{len(records)}] {session_id}")

            result = sanitizer.sanitize(
                session_id=session_id,
                raw_text=raw_text,
                language=language,
            )

            if result is None:
                stats["failed"] += 1
                log.warning(f"[{session_id}] → FAILED")
                continue

            # session_id を出力に追加
            output_record = {"session_id": session_id, **result.data}

            # JSONL 書き込み
            out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            out_f.flush()

            stats["success"] += 1
            stats["warnings_total"] += len(result.warnings)
            risk = result.data.get("risk", "")
            stats[f"risk_{risk}"] = stats.get(f"risk_{risk}", 0) + 1

            # warnings があればログ
            if result.warnings:
                log.info(f"[{session_id}] warnings: {result.warnings}")

    elapsed = time.time() - t_start

    # サマリー
    log.info(
        f"\nDone in {elapsed:.1f}s\n"
        f"  Total:    {stats['total']}\n"
        f"  Success:  {stats['success']}\n"
        f"  Failed:   {stats['failed']}\n"
        f"  Risk:     high={stats['risk_high']}, "
        f"medium={stats['risk_medium']}, low={stats['risk_low']}\n"
        f"  Warnings: {stats['warnings_total']}\n"
        f"  Output:   {output_path}"
    )


if __name__ == "__main__":
    main()
