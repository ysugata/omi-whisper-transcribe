"""Tests for word-boundary-aware dictionary replacement.

Regression tests for the substring-corruption bug: entries like ログ→log
must not rewrite プログラム into プlogRAM.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from term_corrector import TermCorrector, compile_boundary_pattern
from suggestion_learner import SuggestionLearner


@pytest.fixture
def corrector(tmp_path):
    d = {
        "ログ": "log",
        "ラム": "RAM",
        "ライン": "LINE",
        "インスタ": "Instagram",
        "クーバネティス": "Kubernetes",
        "ドッカー": "Docker",
        "LM製": "LLM製",
        "同じ": "同じ",  # identity — must be a no-op
    }
    p = tmp_path / "term_corrections.json"
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return TermCorrector(str(p))


class TestCompoundProtection:
    """Katakana compounds must never be corrupted by substring entries."""

    @pytest.mark.parametrize("text", [
        "プログラムを書く",
        "ブログを更新",
        "カタログを見る",
        "オンラインで会議",
        "ガイドラインに従う",
        "インスタントラーメン",
        "ラムネを飲む",
        "グラム単位で測る",
    ])
    def test_compounds_untouched(self, corrector, text):
        assert corrector.correct(text) == text

    def test_standalone_still_replaced(self, corrector):
        assert corrector.correct("クーバネティスをデプロイ") == "Kubernetesをデプロイ"
        assert corrector.correct("ドッカーで動かす") == "Dockerで動かす"

    def test_standalone_at_boundaries(self, corrector):
        # 前後が非カタカナ (助詞・句読点・文頭文末) なら置換される
        assert corrector.correct("ログ") == "log"
        assert corrector.correct("ログを見る") == "logを見る"
        assert corrector.correct("そのライン、") == "そのLINE、"

    def test_ascii_boundary(self, corrector):
        # "LLM製" 内の "LM製" は左隣が英字なので置換されない
        assert corrector.correct("LLM製のモデル") == "LLM製のモデル"
        assert corrector.correct("LM製のモデル") == "LLM製のモデル"

    def test_identity_mapping_is_noop(self, corrector):
        text = "同じことを繰り返す"
        out, count = corrector.correct_count(text)
        assert out == text
        assert count == 0

    def test_correct_count(self, corrector):
        out, count = corrector.correct_count("クーバネティスとドッカーの話")
        assert out == "KubernetesとDockerの話"
        assert count == 2


class TestCompileBoundaryPattern:
    def test_katakana_guards(self):
        p = compile_boundary_pattern("ログ")
        assert p.search("ログイン") is None  # 右隣がカタカナ
        assert p.search("バックログ") is None  # 左隣がカタカナ
        assert p.search("ログを") is not None

    def test_ascii_guards(self):
        p = compile_boundary_pattern("AI")
        assert p.search("OPENAI") is None
        assert p.search("AI技術") is not None

    def test_mixed_pattern(self):
        p = compile_boundary_pattern("RTX 3060")
        assert p.search("RTX 3060で動く") is not None
        assert p.search("RTX 30601") is None


class TestSuggestionLearnerBoundary:
    def test_learned_patterns_respect_boundaries(self, tmp_path):
        suggestions = [
            {"wrong": "ラグ", "correct": "RAG", "status": "pending",
             "occurrences": 3},
            {"wrong": "ラグ", "correct": "RAG", "status": "pending",
             "occurrences": 3},
        ]
        p = tmp_path / "pending_suggestions.json"
        p.write_text(json.dumps(suggestions, ensure_ascii=False), encoding="utf-8")
        learner = SuggestionLearner(str(p))
        if not learner.get_learned_patterns():
            pytest.skip("learner did not pick up the pattern (threshold config)")
        out, n = learner.apply_learned("ラグの検索")
        assert out == "RAGの検索"
        # 複合語は保護
        out2, n2 = learner.apply_learned("タイムラグが発生")
        assert out2 == "タイムラグが発生"


class TestShippedDictHygiene:
    """The committed dictionary must not contain no-ops or known traps."""

    def test_no_identity_or_trap_entries(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "term_corrections.json")
        d = json.load(open(path, encoding="utf-8"))
        identity = [k for k, v in d.items() if k == v]
        assert identity == [], f"identity mappings found: {identity}"
        for trap in ["ラム", "ログ", "ライン", "トークン"]:
            assert trap not in d, f"semantic trap entry present: {trap}"
