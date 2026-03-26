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
番号付きの発話テキストが入力されます。全体の文脈と参考用語リストを活用して以下のルールに従ってください:

1. 固有名詞・技術用語・社名・プロダクト名の誤認識を修正する
2. カタカナで書かれた技術用語を正しい英語表記に修正する（例: ドッカー → Docker）
3. 音声認識で脱落した可能性のある単語を前後の文脈から補完する（例: 「3060のVRAM」→「RTX3060のVRAM」）
4. 参考用語リストの単語が、音声認識で似た別の単語に誤認識されていないか注意する（例: Nginx が「Linux」に、gRPC が「GRPC」に誤認識される等）
5. 発言の意味・内容・語順は絶対に変えない
6. 明らかな誤りだけを修正し、曖昧な箇所はそのまま残す
7. 句読点の追加・削除はしない
8. 入力と同じ [番号] 形式で、修正後のテキストだけを返す
9. 修正が不要な行も入力をそのまま返す"""


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
                    {"role": "system", "content": SYSTEM_PROMPT},
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

    def correct_segments(
        self,
        segments: list[dict],
        known_terms: Optional[list[str]] = None,
        min_text_length: int = 5,
    ) -> list[dict]:
        """セグメントのリストをまとめて1回の API コールで補正する。

        全セグメントを番号付きリストとして送信し、文脈を活用した補正を行う。
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            raw = response.choices[0].message.content
            log.info(f"LLM batch request:\n{user_msg}")
            log.info(f"LLM batch response:\n{raw}")
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
