"""Responses API sanitizer — OpenAI Responses API + Structured Outputs.

GPT-5 系モデルで structured outputs, reasoning.effort, max_output_tokens を使い、
temperature に依存せず安定した匿名化・構造化出力を得る。

Chat Completions API ではなく Responses API を使用:
- text={"format": {"type": "json_schema", ...}} で strict JSON 出力を強制
- reasoning={"effort": ...} で推論深度を制御
- max_output_tokens でトークン上限を明示
"""

import json
import logging
import time
from typing import Optional

from sanitize_prompt import KC_EXTRACTION_INSTRUCTIONS, SANITIZE_INSTRUCTIONS
from sanitize_provider import SanitizeProvider, SanitizeResult
from sanitize_schema import (
    KC_EXTRACTION_SCHEMA,
    KC_MIN_KEYPOINTS,
    SANITIZE_SCHEMA,
    filter_knowledge_cards,
    run_quality_gates,
    validate_schema,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "gpt-5-nano"
_DEFAULT_REASONING_EFFORT = "low"
_DEFAULT_MAX_OUTPUT_TOKENS = 1800
_DEFAULT_TIMEOUT_MS = 60_000
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BASE_SEC = 2.0

# Input truncation
_MAX_INPUT_CHARS = 4000

# Step 2: Knowledge Card extraction (multi-card, up to 5 KCs per session)
# Each card ≈ 200-300 tokens (q+a+evidence+confidence), so 2500 gives headroom.
_KC_MAX_OUTPUT_TOKENS = 2500


class ResponsesSanitizer(SanitizeProvider):
    """OpenAI Responses API を使った匿名化・構造化プロバイダ.

    特徴:
    - json_schema + strict: true で出力揺れを抑制
    - reasoning.effort で推論深度を制御 (temperature 不使用)
    - max_output_tokens で出力長を明示
    - 指数バックオフリトライ
    - スキーマ不一致時の再試行
    - 品質ゲート (warnings)
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        reasoning_effort: str = _DEFAULT_REASONING_EFFORT,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._enabled = bool(api_key)
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_output_tokens = max_output_tokens
        self._timeout_sec = timeout_ms / 1000.0
        self._max_retries = max_retries
        self._client = None

        if self._enabled:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=api_key,
                    timeout=self._timeout_sec,
                )
                log.info(
                    f"ResponsesSanitizer initialized: model={model}, "
                    f"reasoning_effort={reasoning_effort}, "
                    f"max_output_tokens={max_output_tokens}"
                )
            except Exception as e:
                log.error(f"ResponsesSanitizer init failed: {e}")
                self._enabled = False

    # -- SanitizeProvider interface ------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def provider_name(self) -> str:
        return f"responses-api/{self._model}"

    def sanitize(
        self,
        session_id: str,
        raw_text: str,
        language: str = "ja",
    ) -> Optional[SanitizeResult]:
        """Responses API で匿名化・構造化を実行する.

        指数バックオフでリトライし、スキーマ不一致時も再試行する。
        """
        if not self._enabled:
            return None

        # 入力の前処理・トランケーション
        raw_text = raw_text.strip()
        if len(raw_text) < 20:
            log.debug(f"[{session_id}] input too short ({len(raw_text)} chars), skipping")
            return None

        if len(raw_text) > _MAX_INPUT_CHARS:
            raw_text = raw_text[:_MAX_INPUT_CHARS]
            log.debug(f"[{session_id}] input truncated to {_MAX_INPUT_CHARS} chars")

        # ユーザー入力テキスト (session_id と language をメタ情報として含める)
        user_input = (
            f"session_id: {session_id}\n"
            f"language: {language}\n"
            f"---\n"
            f"{raw_text}"
        )

        # リトライループ
        last_error: Optional[str] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._call_api(session_id, user_input, attempt)
                if result is not None:
                    return result
                # スキーマ不一致 → 次の試行
                last_error = "schema_validation_failed"
            except _RetryableError as e:
                last_error = str(e)
                if attempt < self._max_retries:
                    wait = _DEFAULT_RETRY_BASE_SEC * (2 ** (attempt - 1))
                    log.warning(
                        f"[{session_id}] attempt {attempt}/{self._max_retries} "
                        f"failed: {e}, retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)
            except _FatalError as e:
                log.error(f"[{session_id}] fatal error: {e}")
                return None

        log.warning(
            f"[{session_id}] all {self._max_retries} attempts failed: {last_error}"
        )
        return None

    # -- Internal ------------------------------------------------------------

    def _call_api(
        self,
        session_id: str,
        user_input: str,
        attempt: int,
    ) -> Optional[SanitizeResult]:
        """Responses API を 1 回呼び出し、結果を検証して返す."""
        try:
            t0 = time.time()
            response = self._client.responses.create(
                model=self._model,
                instructions=SANITIZE_INSTRUCTIONS,
                input=[
                    {"role": "user", "content": user_input},
                ],
                text={"format": SANITIZE_SCHEMA},
                reasoning={"effort": self._reasoning_effort},
                max_output_tokens=self._max_output_tokens,
                store=False,  # サーバー側に保存しない
            )
            elapsed = time.time() - t0

        except Exception as e:
            err_str = str(e)
            # タイムアウトや rate limit はリトライ可能
            if any(kw in err_str.lower() for kw in ("timeout", "rate_limit", "429", "503", "529")):
                raise _RetryableError(err_str) from e
            raise _FatalError(err_str) from e

        # レスポンスのステータスチェック
        if response.status == "incomplete":
            reason = ""
            if response.incomplete_details:
                reason = getattr(response.incomplete_details, "reason", "unknown")
            log.warning(f"[{session_id}] incomplete response: {reason}")
            if reason == "max_output_tokens":
                # max_output_tokens 不足はリトライしても無駄
                raise _FatalError(f"max_output_tokens exceeded")
            raise _RetryableError(f"incomplete: {reason}")

        # output_text を取得してパース
        raw_text = response.output_text
        if not raw_text:
            raise _RetryableError("empty output_text")

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            log.warning(f"[{session_id}] JSON parse error: {e}")
            raise _RetryableError(f"json_decode_error: {e}") from e

        # スキーマ検証
        if not validate_schema(data):
            log.warning(f"[{session_id}] schema validation failed (attempt {attempt})")
            return None  # → リトライ

        # Step 2: Knowledge Card extraction (multi-card, if enough keypoints)
        data["knowledge_cards"] = []
        data["knowledge_card"] = None  # legacy single-card field for back-compat
        kc_elapsed = 0.0
        if len(data.get("keypoints", [])) >= KC_MIN_KEYPOINTS:
            kc_result, kc_elapsed = self._extract_knowledge_card(session_id, data)
            if kc_result is not None:
                cards = filter_knowledge_cards(kc_result)
                data["knowledge_cards"] = cards
                if cards:
                    # Populate legacy single field with highest-confidence card
                    order = {"high": 0, "medium": 1, "low": 2, "none": 3}
                    best = min(
                        cards,
                        key=lambda c: order.get(c.get("confidence", "none"), 4),
                    )
                    data["knowledge_card"] = {
                        "question": best["question"],
                        "answer": best["answer"],
                        "evidence": best["evidence"],
                        "confidence": best["confidence"],
                    }
            else:
                log.debug(
                    f"[{session_id}] KC extraction: skipped or API returned None"
                )

        total_elapsed = elapsed + kc_elapsed

        # 品質ゲート
        warnings = run_quality_gates(data)

        # トークン使用量
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            details = getattr(response.usage, "output_tokens_details", None)
            if details:
                usage["reasoning_tokens"] = getattr(details, "reasoning_tokens", 0)

        # 簡潔なログ (入力全文は出さない)
        cards = data["knowledge_cards"]
        kc_info = f"{len(cards)}"
        if cards:
            confs = [c.get("confidence", "?") for c in cards]
            kc_info += f"({','.join(confs)})"
        log.info(
            f"[{session_id}] risk={data['risk']}, "
            f"keypoints={len(data['keypoints'])}, "
            f"conv={len(data['safe_conversation'])}turns, "
            f"kcards={kc_info}, "
            f"issues={data['issues_detected']}, "
            f"warnings={warnings}, "
            f"{total_elapsed:.2f}s (step1={elapsed:.2f}s, step2={kc_elapsed:.2f}s)"
        )

        return SanitizeResult(
            session_id=session_id,
            data=data,
            warnings=warnings,
            usage=usage,
        )

    def _extract_knowledge_card(
        self,
        session_id: str,
        step1_data: dict,
    ) -> tuple[Optional[dict], float]:
        """Step 2: cleaned data から evidence-based knowledge card を抽出する.

        Returns:
            (kc_dict or None, elapsed_seconds)
        """
        # Step 1 出力をテキスト化 (raw text は使わない)
        parts = []

        # summary
        summary = step1_data.get("summary", {})
        if summary.get("output"):
            parts.append(f"## 要約\n{summary['output']}")

        # keypoints
        keypoints = step1_data.get("keypoints", [])
        if keypoints:
            kp_text = "\n".join(f"- {kp}" for kp in keypoints)
            parts.append(f"## キーポイント\n{kp_text}")

        # safe_conversation
        conv = step1_data.get("safe_conversation", [])
        if conv:
            conv_lines = []
            for turn in conv:
                speaker = turn.get("speaker", "unknown")
                text = turn.get("text", "")
                conv_lines.append(f"{speaker}: {text}")
            parts.append(f"## 会話\n" + "\n".join(conv_lines))

        user_input = "\n\n".join(parts)
        if not user_input.strip():
            return None, 0.0

        try:
            t0 = time.time()
            response = self._client.responses.create(
                model=self._model,
                instructions=KC_EXTRACTION_INSTRUCTIONS,
                input=[
                    {"role": "user", "content": user_input},
                ],
                text={"format": KC_EXTRACTION_SCHEMA},
                reasoning={"effort": self._reasoning_effort},
                max_output_tokens=_KC_MAX_OUTPUT_TOKENS,
                store=False,
            )
            kc_elapsed = time.time() - t0

        except Exception as e:
            log.warning(f"[{session_id}] KC extraction API error: {e}")
            return None, 0.0

        if response.status != "completed":
            log.warning(f"[{session_id}] KC extraction incomplete: {response.status}")
            return None, 0.0

        raw_text = response.output_text
        if not raw_text:
            return None, 0.0

        try:
            kc_data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            log.warning(f"[{session_id}] KC extraction JSON error: {e}")
            return None, 0.0

        # Update usage tracking with Step 2 tokens
        if response.usage:
            # Step 2 tokens are logged separately for cost tracking
            log.debug(
                f"[{session_id}] KC extraction tokens: "
                f"in={response.usage.input_tokens}, "
                f"out={response.usage.output_tokens}"
            )

        return kc_data, kc_elapsed


# ---------------------------------------------------------------------------
# Internal exception classes for retry control
# ---------------------------------------------------------------------------

class _RetryableError(Exception):
    """リトライ可能なエラー (タイムアウト、rate limit、スキーマ不一致等)."""
    pass


class _FatalError(Exception):
    """リトライ不可能なエラー (認証失敗、max_output_tokens 超過等)."""
    pass
