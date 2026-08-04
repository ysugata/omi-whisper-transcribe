"""Daily knowledge elicitation — 経営者への能動的な聞き出し。

受動的な録音は「知識が偶然口に出る」のを待つが、暗黙知は聞き出して
初めて外に出る。このモジュールは毎晩、その日の意思決定に係留した質問を
生成し (Critical Decision Method 系のプローブ)、回答を Safe Training
パイプラインへ還流させる。

設計根拠 (2026-08 リサーチ):
- Critical Decision Method (Hoffman & Crandall): 抽象論ではなく実在の
  意思決定インシデントに係留した振り返り質問が暗黙知を最も引き出す。
- arXiv 2507.21389: 既に見えている情報を聞く質問は無価値。欠けている
  情報 (語られなかった理由・捨てた選択肢・境界条件) を狙う質問だけが
  新しい知識を掘り出す。
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

_GEN_SYSTEM = """あなたは事業継承のためのナレッジエンジニアです。
経営者本人に翌朝答えてもらう質問を作ります。目的は、会話に「語られなかった」
判断の中身を引き出すことです。

## 質問の原則 (Critical Decision Method)
- 必ず提示された実際の会話・意思決定に係留すること (抽象的な一般質問は禁止)
- 文字起こしに既に答えがあることを聞かない。語られていないものだけを狙う:
  - なぜその選択肢を選び、何を捨てたか (捨てた理由が最重要)
  - その判断が変わるとしたら、どんな条件・シグナルのときか
  - 初めての人 (後継者) が同じ場面で間違えそうなポイントは何か
  - 判断の決め手になった「見えていた手がかり」は何か
- 1つの質問は1つの論点。二重質問にしない
- 経営者が1〜2分で答えられる具体性にする
- 過去に出した質問と重複しない

必ず次のJSONだけを返してください:
{"questions": [{"question": "...", "anchor": "どの会話/判断に係留したかの短い説明"}]}"""


class ElicitationStore:
    """config/elicitation_questions.json の所有者 (アトミック書き込み)。"""

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def _load(self) -> list:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

    def _save(self, items: list) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(self._path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)
        os.replace(tmp, str(self._path))

    # -- queries ---------------------------------------------------------

    def pending(self) -> list:
        return [q for q in self._load() if q.get("status") == "pending"]

    def all_questions(self) -> list:
        return self._load()

    def recent_question_texts(self, limit: int = 30) -> list:
        items = self._load()
        return [q.get("question", "") for q in items[-limit:]]

    def used_source_refs(self) -> set:
        return {q.get("source_ref") for q in self._load() if q.get("source_ref")}

    def unprocessed_answers(self) -> list:
        return [
            q for q in self._load()
            if q.get("status") == "answered" and not q.get("processed")
        ]

    # -- mutations -------------------------------------------------------

    def add_questions(self, questions: list, max_pending: int) -> int:
        """Add generated questions, respecting the pending cap and dedup."""
        items = self._load()
        pending_count = sum(1 for q in items if q.get("status") == "pending")
        existing_texts = {q.get("question", "") for q in items}
        added = 0
        for q in questions:
            if pending_count + added >= max_pending:
                break
            text = (q.get("question") or "").strip()
            if not text or text in existing_texts:
                continue
            items.append({
                "id": f"eq_{uuid.uuid4().hex[:8]}",
                "question": text,
                "anchor": (q.get("anchor") or "")[:120],
                "source_type": q.get("source_type", "decision"),
                "source_ref": q.get("source_ref", ""),
                "status": "pending",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            })
            existing_texts.add(text)
            added += 1
        if added:
            self._save(items)
        return added

    def _set(self, qid: str, **fields) -> dict | None:
        items = self._load()
        for q in items:
            if q.get("id") == qid:
                q.update(fields)
                self._save(items)
                return q
        return None

    def answer(self, qid: str, text: str) -> dict | None:
        text = (text or "").strip()
        if not text:
            return None
        return self._set(
            qid, status="answered", answer=text,
            answered_at=datetime.now().isoformat(timespec="seconds"),
        )

    def skip(self, qid: str) -> dict | None:
        return self._set(qid, status="skipped")

    def mark_processed(self, qid: str, outcome: str = "done") -> None:
        self._set(qid, processed=True, processed_outcome=outcome)


class ElicitationEngine:
    """LLM で質問を生成する。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 60,
        reasoning_effort: str = "low",
    ) -> None:
        self._enabled = bool(api_key)
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._client = None
        if self._enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
                log.info("ElicitationEngine initialized: model=%s", model)
            except Exception as e:
                log.error("ElicitationEngine init failed: %s", e)
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def generate(
        self,
        decision_contexts: list,
        gap_questions: list,
        recent_questions: list,
        max_questions: int,
    ) -> list:
        """質問を生成する。

        decision_contexts: [{"session_id", "summary", "tags_line", "excerpt"}]
        gap_questions: RAG が答えられなかった質問のテキスト
        Returns [{"question", "anchor", "source_type", "source_ref"}]
        """
        if not self._enabled or max_questions <= 0:
            return []
        parts = []
        if decision_contexts:
            parts.append("## 今日の実際の会話・意思決定 (質問の係留先)")
            for c in decision_contexts:
                parts.append(
                    f"### セッション {c['session_id']}\n"
                    f"タグ: {c.get('tags_line', '')}\n"
                    f"要約: {c.get('summary', '')}\n"
                    f"抜粋: {c.get('excerpt', '')[:1200]}"
                )
        if gap_questions:
            parts.append(
                "## ナレッジベースが答えられなかった質問 (この空白を埋める質問を作る)\n"
                + "\n".join(f"- {g}" for g in gap_questions[:5])
            )
        if not parts:
            return []
        if recent_questions:
            parts.append(
                "## 最近出した質問 (重複禁止)\n"
                + "\n".join(f"- {t}" for t in recent_questions[-15:])
            )
        parts.append(f"\n質問は最大 {max_questions} 個。価値の高い順に。")

        try:
            kwargs = dict(
                model=self._model,
                messages=[
                    {"role": "system", "content": _GEN_SYSTEM},
                    {"role": "user", "content": "\n\n".join(parts)},
                ],
                response_format={"type": "json_object"},
            )
            if self._model.startswith("gpt-"):
                kwargs["max_completion_tokens"] = 2048
                if re.match(r"^(gpt-5|o\d)", self._model):
                    kwargs["reasoning_effort"] = self._reasoning_effort
                else:
                    kwargs["temperature"] = 0.3
            else:
                kwargs["max_tokens"] = 2048
                kwargs["temperature"] = 0.3
            response = self._client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content
            parsed = self.parse_questions(raw, max_questions)
        except Exception as e:
            log.warning("ElicitationEngine: generation failed: %s", e)
            return []

        # Attach source metadata: gap 由来と判断由来を区別する
        out = []
        for q in parsed:
            entry = dict(q)
            entry.setdefault("source_type", "decision")
            if decision_contexts:
                entry.setdefault("source_ref", decision_contexts[0]["session_id"])
            out.append(entry)
        return out

    @staticmethod
    def parse_questions(raw: str, max_questions: int) -> list:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        questions = data.get("questions")
        if not isinstance(questions, list):
            return []
        out = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            text = str(q.get("question", "")).strip()
            if len(text) < 10:
                continue
            out.append({
                "question": text,
                "anchor": str(q.get("anchor", "")).strip(),
            })
            if len(out) >= max_questions:
                break
        return out
