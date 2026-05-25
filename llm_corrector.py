"""Layer 3: Gemini 2.5 Flash による LLM 後処理モジュール

OpenAI 互換 API 経由で Gemini を呼び出し、
Layer 2（辞書置換）では対応できない未知の誤変換を文脈ベースで修正する。
API エラー時は元テキストをそのまま返す（テキストを失わない）。
"""

import difflib
import json
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

# 差分検出フィルタ用正規表現
_PUNCT_RE = re.compile(r'^[\s、。,.!?！？・…―「」『』（）()\-–—]+$')
_NUM_RE = re.compile(r'^[\d０-９\s.,%]+$')

# Punctuation-only diff filter: true if the ONLY difference is punctuation style
_PUNCT_CHARS = set('、。,.!?！？・…―「」『』（）()-–—　 ')

# Legacy prompt for single-segment correct() — kept for backward compat
_LEGACY_SYSTEM_PROMPT = """あなたは音声文字起こしテキストの校正者です。
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

# Backward compat alias
SYSTEM_PROMPT = _LEGACY_SYSTEM_PROMPT

# Optimized prompt for batch correct_segments() — diff-only JSON output
BATCH_SYSTEM_PROMPT = """あなたは日本語の音声文字起こし (ASR) テキストの校正者です。
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


class LLMCorrector:
    """Gemini 2.5 Flash-Lite を使った文字起こしテキストの後処理補正。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0,
        max_tokens: int = 2048,
        timeout: int = 10,
    ) -> None:
        """LLMCorrector を初期化する。

        api_key が空の場合は correct() が入力をそのまま返す。
        """
        self._enabled = bool(api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._warned = False
        self._client = None

        if self._enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
                log.info(f"LLMCorrector initialized: model={model}")
            except Exception as e:
                log.error(f"LLMCorrector init failed: {e}")
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _build_user_message(self, text: str, known_terms: Optional[list[str]]) -> str:
        """ユーザーメッセージを組み立てる。"""
        parts = []
        if known_terms:
            terms_str = ", ".join(known_terms[:30])
            parts.append(f"参考用語リスト: {terms_str}")
        parts.append(f"テキスト: {text}")
        return "\n".join(parts)

    def _validate_response(self, original: str, corrected: str) -> str:
        """LLM のレスポンスを検証し、異常なら元テキストを返す。"""
        if not corrected or not corrected.strip():
            return original
        corrected = corrected.strip()
        if len(corrected) > len(original) * 2 or len(corrected) < len(original) * 0.5:
            log.warning(f"LLM出力の長さが異常: 元={len(original)} 修正={len(corrected)}")
            return original
        # 類似度チェック: 元テキストと大きく異なる場合は拒否
        ratio = difflib.SequenceMatcher(None, original, corrected).ratio()
        if ratio < 0.5:
            log.warning(f"LLM出力が元テキストと乖離 (similarity={ratio:.2f}): '{corrected[:60]}'")
            return original
        return corrected

    def correct(
        self,
        text: str,
        known_terms: Optional[list[str]] = None,
    ) -> str:
        """テキストの固有名詞・技術用語の誤りを修正して返す。

        APIエラー時は元のテキストをそのまま返す。
        """
        if not self._enabled:
            if not self._warned:
                log.warning("LLMCorrector: API key未設定、Layer 3 スキップ")
                self._warned = True
            return text

        user_msg = self._build_user_message(text, known_terms)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _LEGACY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            corrected = response.choices[0].message.content
            result = self._validate_response(text, corrected)

            if result != text:
                log.debug(f"LLM corrected: '{text[:50]}' → '{result[:50]}'")

            return result

        except Exception as e:
            log.warning(f"LLM API error (text preserved): {e}")
            return text

    def _parse_batch_response(self, raw: str, expected_count: int) -> dict[int, str]:
        """番号付きバッチレスポンスをパースして {0-indexed: text} を返す。"""
        result = {}
        pattern = re.compile(r'^\[(\d+)\]\s*(.*)')

        for line in raw.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                idx = int(m.group(1)) - 1  # 0-indexed
                text = m.group(2).strip()
                if 0 <= idx < expected_count and text:
                    result[idx] = text

        if len(result) != expected_count:
            log.warning(f"LLM batch: expected {expected_count} lines, got {len(result)}")

        return result

    @staticmethod
    def _is_punctuation_only_change(original: str, corrected: str) -> bool:
        """Check if the only difference between original and corrected is punctuation."""
        if original == corrected:
            return True
        # Strip all punctuation chars, compare the rest
        strip = lambda s: ''.join(c for c in s if c not in _PUNCT_CHARS)
        return strip(original) == strip(corrected)

    @staticmethod
    def _filter_corrections(
        corrections: list[dict],
        original_texts: dict[int, str],
    ) -> list[dict]:
        """Post-filter LLM corrections to remove no-ops and punctuation-only changes.

        Args:
            corrections: list of {"line": int, "original": str, "corrected": str, "reason": str}
            original_texts: {1-indexed line number: original text} for lookup
        Returns:
            Filtered list of corrections.
        """
        filtered = []
        for c in corrections:
            line = c.get("line", 0)
            corrected = c.get("corrected", "").strip()
            # Use the actual original text from input, not what LLM echoes back
            original = original_texts.get(line, c.get("original", "")).strip()

            # Skip no-change
            if original == corrected:
                log.debug(f"Filter: no-change on line {line}")
                continue
            # Skip empty corrected
            if not corrected:
                continue
            # Skip punctuation-only changes
            if LLMCorrector._is_punctuation_only_change(original, corrected):
                log.debug(f"Filter: punct-only on line {line}: '{original[:40]}' → '{corrected[:40]}'")
                continue

            filtered.append(c)
        return filtered

    def correct_segments(
        self,
        segments: list[dict],
        known_terms: Optional[list[str]] = None,
        min_text_length: int = 5,
    ) -> list[dict]:
        """セグメントのリストをまとめて1回の API コールで補正する。

        Optimized prompt + JSON schema で差分のみを受け取り、
        ポストフィルタで句読点のみの変更等を除去する。
        JSON パース失敗時は旧番号付きフォーマットにフォールバック。
        min_text_length 未満の短いセグメントはスキップする。
        """
        if not self._enabled:
            return segments

        # 対象セグメントを収集
        eligible: list[tuple[int, str]] = []
        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            if len(text) >= min_text_length:
                eligible.append((i, text))

        if not eligible:
            return segments

        # 番号付きリストを構築
        lines = [f"[{idx + 1}] {text}" for idx, (_, text) in enumerate(eligible)]
        # 1-indexed line number → original text mapping for post-filter
        original_by_line = {idx + 1: text for idx, (_, text) in enumerate(eligible)}

        parts = []
        if known_terms:
            terms_str = ", ".join(known_terms[:30])
            parts.append(f"参考用語リスト: {terms_str}")
        parts.append("\n".join(lines))
        user_msg = "\n".join(parts)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format=CORRECTIONS_SCHEMA,
                temperature=self._temperature,
                max_tokens=max(self._max_tokens, 4096),
            )
            raw = response.choices[0].message.content
            log.info(f"LLM batch request:\n{user_msg}")
            log.info(f"LLM batch response:\n{raw}")

            # Try JSON parse first
            corrected_map = {}
            try:
                data = json.loads(raw) if raw else {"corrections": []}
                corrections = data.get("corrections", [])
                corrections = self._filter_corrections(corrections, original_by_line)
                for c in corrections:
                    line_idx = c.get("line", 0) - 1  # convert to 0-indexed
                    corrected_text = c.get("corrected", "").strip()
                    if 0 <= line_idx < len(eligible) and corrected_text:
                        corrected_map[line_idx] = corrected_text
                log.info(f"LLM batch: {len(corrections)} corrections after filter (from JSON)")
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                log.warning(f"JSON parse failed, falling back to numbered format: {e}")
                corrected_map = self._parse_batch_response(raw, len(eligible))

            # 結果を適用
            result = [dict(seg) for seg in segments]
            for idx, (seg_idx, original_text) in enumerate(eligible):
                if idx in corrected_map:
                    validated = self._validate_response(original_text, corrected_map[idx])
                    result[seg_idx]["text"] = validated
                    if validated != original_text:
                        log.debug(f"LLM corrected [{idx + 1}]: '{original_text[:50]}' → '{validated[:50]}'")

            return result

        except Exception as e:
            log.warning(f"LLM batch API error (text preserved): {e}")
            return [dict(seg) for seg in segments]

    @staticmethod
    def _is_trivial_change(wrong: str, correct: str) -> bool:
        """変更が些細 or ゴミかどうかを判定する。"""
        # 句読点のみの変更
        if _PUNCT_RE.match(wrong) or _PUNCT_RE.match(correct):
            return True
        # 数字のみの変更
        if _NUM_RE.match(wrong) or _NUM_RE.match(correct):
            return True
        # 短すぎる断片 (3文字未満) はゴミ
        if len(wrong) < 3 or len(correct) < 3:
            return True
        # ひらがな・カタカナのみの短い断片 (4文字未満) はゴミ
        _kana = lambda c: '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff'
        if len(wrong) < 4 and all(_kana(c) for c in wrong):
            return True
        if len(correct) < 4 and all(_kana(c) for c in correct):
            return True
        # wrong と correct の類似度が高すぎる (助詞レベルの変更)
        ratio = difflib.SequenceMatcher(None, wrong, correct).ratio()
        if ratio > 0.8 and len(wrong) < 6:
            return True
        return False

    @staticmethod
    def detect_corrections(original: str, corrected: str) -> list[dict]:
        """元テキストと修正テキストの差分から修正ペアを抽出する。

        Returns:
            [{"wrong": str, "correct": str}, ...]
        """
        if original == corrected:
            return []

        matcher = difflib.SequenceMatcher(None, original, corrected)
        results = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace":
                continue
            wrong = original[i1:i2].strip()
            correct = corrected[j1:j2].strip()
            if not wrong or not correct:
                continue
            if LLMCorrector._is_trivial_change(wrong, correct):
                continue
            results.append({"wrong": wrong, "correct": correct})

        return results
