"""Layer 3: Gemini 2.5 Flash による LLM 後処理モジュール

OpenAI 互換 API 経由で Gemini を呼び出し、
Layer 2（辞書置換）では対応できない未知の誤変換を文脈ベースで修正する。
API エラー時は元テキストをそのまま返す（テキストを失わない）。
"""

import difflib
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

# 差分検出フィルタ用正規表現
_PUNCT_RE = re.compile(r'^[\s、。,.!?・…―「」『』（）()\-–—]+$')
_NUM_RE = re.compile(r'^[\d０-９\s.,%]+$')

SYSTEM_PROMPT = """あなたは音声文字起こしテキストの校正者です。
以下のルールに厳密に従ってください:

1. 固有名詞・技術用語・社名・プロダクト名の誤認識を修正する
2. カタカナで書かれた技術用語を正しい英語表記に修正する（例: ドッカー → Docker）
3. 発言の意味・内容・語順は絶対に変えない
4. 明らかな誤りだけを修正し、曖昧な箇所はそのまま残す
5. 句読点の追加・削除はしない
6. 修正後のテキストだけを返す（説明・コメント・前置きは一切不要）
7. 修正が不要な場合は入力テキストをそのまま返す"""


class LLMCorrector:
    """Gemini 2.5 Flash を使った文字起こしテキストの後処理補正。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-2.5-flash",
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
        if len(corrected) > len(original) * 2 or len(corrected) < len(original) * 0.5:
            log.warning(f"LLM出力の長さが異常: 元={len(original)} 修正={len(corrected)}")
            return original
        return corrected.strip()

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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                extra_body={"reasoning_effort": "none"},
            )
            corrected = response.choices[0].message.content
            result = self._validate_response(text, corrected)

            if result != text:
                log.debug(f"LLM corrected: '{text[:50]}' → '{result[:50]}'")

            return result

        except Exception as e:
            log.warning(f"LLM API error (text preserved): {e}")
            return text

    def correct_segments(
        self,
        segments: list[dict],
        known_terms: Optional[list[str]] = None,
        min_text_length: int = 5,
    ) -> list[dict]:
        """セグメントのリストに対して一括で補正を適用する。

        min_text_length 以下の短いセグメントはスキップする。
        """
        if not self._enabled:
            return segments

        result = []
        for seg in segments:
            seg_copy = dict(seg)
            text = seg_copy.get("text", "")
            if len(text) >= min_text_length:
                seg_copy["text"] = self.correct(text, known_terms)
            result.append(seg_copy)
        return result

    @staticmethod
    def _is_trivial_change(wrong: str, correct: str) -> bool:
        """変更が些細なものかどうかを判定する。"""
        # 句読点のみの変更
        if _PUNCT_RE.match(wrong) or _PUNCT_RE.match(correct):
            return True
        # 数字のみの変更
        if _NUM_RE.match(wrong) or _NUM_RE.match(correct):
            return True
        # 1文字の変更
        if len(wrong) <= 1 or len(correct) <= 1:
            return True
        # ひらがな2文字以下の変更
        if len(wrong) <= 2 and all('\u3040' <= c <= '\u309f' for c in wrong):
            return True
        if len(correct) <= 2 and all('\u3040' <= c <= '\u309f' for c in correct):
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
