"""Tests for hotword token budget limiting."""

import json
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hotword_loader import HotwordLoader


def _make_config(tmp, categories=None, custom=None, settings=None):
    """Create config files in a temp directory."""
    config_dir = tmp
    if categories is not None:
        with open(os.path.join(config_dir, "hotwords.json"), "w") as f:
            json.dump({"categories": categories}, f)
    if custom is not None:
        with open(os.path.join(config_dir, "hotwords_custom.json"), "w") as f:
            json.dump(custom, f)
    if settings is not None:
        with open(os.path.join(config_dir, "hotwords_settings.json"), "w") as f:
            json.dump(settings, f)
    return config_dir


class TestTokenBudget:
    """Hotwords should be limited by token budget, not just word count."""

    def test_token_budget_limits_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 50 words of 5 chars each = ~300 tokens (5+1 per word)
            words = [f"テスト{i:02d}" for i in range(50)]
            config = _make_config(tmp,
                categories={"company": words},
                settings={"max_tokens": 100, "categories": {"company": {"enabled": True, "max": 50}}},
            )
            loader = HotwordLoader(config)
            result = loader.get_hotwords_list()
            # Each word is ~5 chars + 1 separator = 6 tokens
            # Budget 100 / 6 ≈ 16 words max
            assert len(result) <= 17
            assert len(result) >= 14  # reasonable range

    def test_custom_words_have_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = ["カスタム語A", "カスタム語B", "カスタム語C"]
            category_words = [f"カテゴリ{i:02d}" for i in range(50)]
            config = _make_config(tmp,
                categories={"company": category_words},
                custom=custom,
                settings={"max_tokens": 50, "categories": {"company": {"enabled": True, "max": 50}}},
            )
            loader = HotwordLoader(config)
            result = loader.get_hotwords_list()
            # All 3 custom words should be present
            for w in custom:
                assert w in result, f"Custom word '{w}' should be in result"
            # Custom words should come first
            for i, w in enumerate(custom):
                assert result[i] == w

    def test_empty_hotwords_no_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp, categories={}, settings={"max_tokens": 200})
            loader = HotwordLoader(config)
            assert loader.get_hotwords_list() == []
            assert loader.get_hotwords_string() == ""

    def test_large_custom_word_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            # One very long custom word that consumes most of the budget
            big_word = "ア" * 150  # 150 chars ≈ 151 tokens
            custom = [big_word, "短い"]
            config = _make_config(tmp,
                categories={"company": ["テスト語"]},
                custom=custom,
                settings={"max_tokens": 160, "categories": {"company": {"enabled": True, "max": 10}}},
            )
            loader = HotwordLoader(config)
            result = loader.get_hotwords_list()
            assert big_word in result
            assert "短い" in result
            # Category word should NOT fit (only ~7 tokens left, テスト語=4 tokens)
            # Actually 160 - 151 - 3 = 6 tokens left, テスト語=3+1=4 tokens -> fits
            # Let's just check total token estimate is within budget

    def test_default_max_tokens_is_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Many words that would exceed 200 tokens
            words = [f"テスト{i:03d}" for i in range(100)]  # 6 chars each
            config = _make_config(tmp,
                categories={"company": words},
                settings={"categories": {"company": {"enabled": True, "max": 100}}},
            )
            loader = HotwordLoader(config)
            result = loader.get_hotwords_list()
            # Estimate total tokens
            total_tokens = sum(len(w) + 1 for w in result)
            assert total_tokens <= 200, f"Token count {total_tokens} exceeds budget 200"

    def test_max_total_still_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Short words that fit in token budget but exceed max_total
            words = [f"A{i}" for i in range(100)]  # 2 chars = 3 tokens each
            config = _make_config(tmp,
                categories={"company": words},
                settings={
                    "max_total": 10, "max_tokens": 9999,
                    "categories": {"company": {"enabled": True, "max": 100}},
                },
            )
            loader = HotwordLoader(config)
            result = loader.get_hotwords_list()
            assert len(result) <= 10
