"""Unit tests for sanitize_schema.py multi-KC + quality gates."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanitize_schema import (
    KC_MAX_PER_SESSION,
    filter_knowledge_cards,
    should_keep_knowledge_card,
    validate_knowledge_card,
)


def _good_card(question="FBA と自己配送の手数料差と最終方針は何か?",
               answer="FBA 統一+自己配送併用を検討する方針と決めた。",
               evidence=("FBA と自己配送の手数料を比較", "FBA 統一方針を決めた"),
               confidence="high"):
    return {
        "question": question,
        "answer": answer,
        "evidence": list(evidence),
        "confidence": confidence,
    }


class TestShouldKeep:
    def test_good_card_accepted(self):
        assert should_keep_knowledge_card(_good_card()) is True

    def test_no_evidence_rejected(self):
        c = _good_card()
        c["evidence"] = []
        assert should_keep_knowledge_card(c) is False

    def test_low_confidence_rejected(self):
        c = _good_card(confidence="low")
        assert should_keep_knowledge_card(c) is False

    def test_none_confidence_rejected(self):
        c = _good_card(confidence="none")
        assert should_keep_knowledge_card(c) is False

    def test_short_question_rejected(self):
        c = _good_card(question="なぜ?")
        assert should_keep_knowledge_card(c) is False

    def test_short_answer_rejected(self):
        c = _good_card(answer="はい。")
        assert should_keep_knowledge_card(c) is False

    def test_vague_question_about_topic_rejected(self):
        c = _good_card(question="Keepa について教えてください")
        assert should_keep_knowledge_card(c) is False

    def test_vague_question_what_topic_rejected(self):
        c = _good_card(question="主なトピックは何ですか? とても長い質問")
        assert should_keep_knowledge_card(c) is False

    def test_vague_question_reconfirm_rejected(self):
        c = _good_card(question="要点を再確認してください")
        assert should_keep_knowledge_card(c) is False


class TestValidateWarnings:
    def test_clean_card_no_warnings(self):
        assert validate_knowledge_card(_good_card()) == []

    def test_vague_question_flagged(self):
        c = _good_card(question="AI について教えてください")
        ws = validate_knowledge_card(c)
        assert "kc_question_vague" in ws

    def test_low_confidence_flagged(self):
        c = _good_card(confidence="low")
        ws = validate_knowledge_card(c)
        assert any(w.startswith("kc_weak_confidence") for w in ws)


class TestFilterKnowledgeCards:
    def test_multi_card_happy_path(self):
        data = {
            "extractable": True,
            "cards": [
                _good_card(question="AさんのXに関する具体的方針は何か?"),
                _good_card(question="BさんのYに関する具体的方針は何か?"),
            ],
        }
        kept = filter_knowledge_cards(data)
        assert len(kept) == 2

    def test_extractable_false_returns_empty(self):
        data = {"extractable": False, "cards": []}
        assert filter_knowledge_cards(data) == []

    def test_duplicate_questions_deduped(self):
        c1 = _good_card()
        c2 = _good_card()  # same question
        data = {"extractable": True, "cards": [c1, c2]}
        kept = filter_knowledge_cards(data)
        assert len(kept) == 1

    def test_invalid_cards_filtered_out(self):
        data = {
            "extractable": True,
            "cards": [
                _good_card(),
                _good_card(question="短い質問"),  # 6 chars → reject
                _good_card(answer="短い"),  # too short → reject
                _good_card(confidence="low"),
            ],
        }
        kept = filter_knowledge_cards(data)
        assert len(kept) == 1

    def test_max_per_session_cap(self):
        many_cards = [
            _good_card(
                question=f"ユニーク質問 {i} について、事業判断の根拠は何か?",
                answer="エビデンスに基づき具体的な方針を決定した、と述べている。",
            )
            for i in range(10)
        ]
        data = {"extractable": True, "cards": many_cards}
        kept = filter_knowledge_cards(data)
        assert len(kept) == KC_MAX_PER_SESSION

    def test_legacy_single_card_format(self):
        legacy = {
            "extractable": True,
            "question": "Keepa API の設計方針は何か?",
            "answer": "波形分析で将来の購買機会を見込む基準を作る設計。",
            "evidence": ["Keepa API を用いた設計"],
            "confidence": "medium",
        }
        kept = filter_knowledge_cards(legacy)
        assert len(kept) == 1

    def test_legacy_single_card_rejected_on_quality(self):
        legacy = {
            "extractable": True,
            "question": "なぜ?",  # too short
            "answer": "はい",
            "evidence": ["evidence"],
            "confidence": "medium",
        }
        assert filter_knowledge_cards(legacy) == []
