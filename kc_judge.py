"""KC (knowledge card) quality judge.

事業継承 AI の学習データとなるナレッジカード (Q&A) を LLM で採点する。
夜間 sanitize の品質ゲート (レコードに quality を付与) と、
tools/eval_quality.py のコーパス品質測定の両方から使われる。

採点観点 (総合 1-5):
  - 明確性: 質問と回答が一意に理解できるか
  - 具体性: 一般論ではなく、この事業固有の判断・数字・手順を含むか
  - 事業価値: 後継者が読んで意思決定の役に立つか
  - 自己完結性: そのカード単体で文脈が通じるか
"""

import json
import logging
import re

log = logging.getLogger(__name__)

_JUDGE_SYSTEM = """あなたは事業継承AIの学習データ品質審査員です。
与えられたナレッジカード (Q&A) を、次の4観点から総合1〜5で採点してください。
- 明確性: 質問と回答が一意に理解できるか
- 具体性: 一般論ではなく、この事業固有の判断・数字・手順を含むか
- 事業価値: 後継者の意思決定に役立つか
- 自己完結性: カード単体で文脈が通じるか

基準: 5=模範的 / 4=良い / 3=使える / 2=曖昧・一般論 / 1=学習データに不適。
必ず次のJSONだけを返してください:
{"scores": [{"index": 1, "score": 3, "reason": "20字以内の理由"}, ...]}"""

_BATCH_SIZE = 10


class KCJudge:
    """Scores knowledge cards via an OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1/",
        model: str = "gpt-5-nano",
        timeout: int = 45,
        reasoning_effort: str = "minimal",
    ) -> None:
        self._enabled = bool(api_key)
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._client = None
        if self._enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=api_key, base_url=base_url, timeout=timeout,
                )
                log.info("KCJudge initialized: model=%s", model)
            except Exception as e:
                log.error("KCJudge init failed: %s", e)
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def judge(self, kcs: list) -> list:
        """Score cards. Returns a list aligned with the input; each element is
        {"score": int, "reason": str} or None when judging failed."""
        if not self._enabled or not kcs:
            return [None] * len(kcs)
        results: list = []
        for i in range(0, len(kcs), _BATCH_SIZE):
            batch = kcs[i:i + _BATCH_SIZE]
            results.extend(self._judge_batch(batch))
        return results

    def _judge_batch(self, batch: list) -> list:
        lines = []
        for i, kc in enumerate(batch, 1):
            q = str(kc.get("question", "")).strip()
            a = str(kc.get("answer", "")).strip()
            lines.append(f"[{i}] Q: {q}\n    A: {a}")
        user_msg = "\n".join(lines)
        try:
            kwargs = dict(
                model=self._model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
            if self._model.startswith("gpt-"):
                kwargs["max_completion_tokens"] = 2048
                if re.match(r"^(gpt-5|o\d)", self._model):
                    kwargs["reasoning_effort"] = self._reasoning_effort
                else:
                    kwargs["temperature"] = 0
            else:
                kwargs["max_tokens"] = 2048
                kwargs["temperature"] = 0
            response = self._client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content
            return self.parse_scores(raw, len(batch))
        except Exception as e:
            log.warning("KCJudge: batch judge failed: %s", e)
            return [None] * len(batch)

    @staticmethod
    def parse_scores(raw: str, n: int) -> list:
        """Parse the judge response into n aligned {"score","reason"} slots.

        Index-based mapping so a dropped/reordered entry cannot shift the
        scores onto the wrong cards (missing entries stay None).
        """
        out: list = [None] * n
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return out
        scores = data.get("scores")
        if not isinstance(scores, list):
            return out
        for entry in scores:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("index", 0)) - 1
                score = int(entry.get("score", 0))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n and 1 <= score <= 5:
                out[idx] = {
                    "score": score,
                    "reason": str(entry.get("reason", ""))[:60],
                }
        return out
