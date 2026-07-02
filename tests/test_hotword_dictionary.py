"""Tests for the term_corrections.json → hotwords dictionary phase."""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hotword_loader import HotwordLoader


def _write(config_dir, name, data):
    (config_dir / name).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


@pytest.fixture
def config_dir(tmp_path):
    return tmp_path


def test_dictionary_terms_included(config_dir):
    _write(config_dir, "term_corrections.json", {
        "クーバネティス": "Kubernetes",
        "ドッカー": "Docker",
    })
    loader = HotwordLoader(str(config_dir))
    words = loader.get_hotwords_list()
    assert "Kubernetes" in words
    assert "Docker" in words


def test_variant_count_priority(config_dir):
    """Terms with more wrong-variants (chronically misheard) come first."""
    _write(config_dir, "term_corrections.json", {
        "クーバネティス": "Kubernetes",
        "クバネティス": "Kubernetes",
        "クーバネテス": "Kubernetes",
        "ドッカー": "Docker",
    })
    loader = HotwordLoader(str(config_dir))
    words = loader.get_hotwords_list()
    assert words.index("Kubernetes") < words.index("Docker")


def test_custom_words_rank_above_dictionary(config_dir):
    _write(config_dir, "hotwords_custom.json", ["MyCompany"])
    _write(config_dir, "term_corrections.json", {"ドッカー": "Docker"})
    loader = HotwordLoader(str(config_dir))
    words = loader.get_hotwords_list()
    assert words.index("MyCompany") < words.index("Docker")


def test_dictionary_max_respected(config_dir):
    terms = {f"まちがい{i}": f"Correct{i}" for i in range(30)}
    _write(config_dir, "term_corrections.json", terms)
    _write(config_dir, "hotwords_settings.json", {
        "dictionary": {"enabled": True, "max": 5},
    })
    loader = HotwordLoader(str(config_dir))
    words = loader.get_hotwords_list()
    dict_words = [w for w in words if w.startswith("Correct")]
    assert len(dict_words) == 5


def test_dictionary_disabled(config_dir):
    _write(config_dir, "term_corrections.json", {"ドッカー": "Docker"})
    _write(config_dir, "hotwords_settings.json", {
        "dictionary": {"enabled": False},
    })
    loader = HotwordLoader(str(config_dir))
    assert "Docker" not in loader.get_hotwords_list()


def test_token_budget_respected(config_dir):
    # Long terms that would blow a tiny budget
    terms = {f"まちがい{i}": "ロングロングロングターム" + str(i) for i in range(50)}
    _write(config_dir, "term_corrections.json", terms)
    _write(config_dir, "hotwords_settings.json", {"max_tokens": 30})
    loader = HotwordLoader(str(config_dir))
    words = loader.get_hotwords_list()
    est = sum(len(w) + 1 for w in words)
    assert est <= 30


def test_mtime_reload(config_dir):
    _write(config_dir, "term_corrections.json", {"ドッカー": "Docker"})
    loader = HotwordLoader(str(config_dir))
    assert "Docker" in loader.get_hotwords_list()

    # Update the dictionary; ensure mtime moves forward
    time.sleep(0.01)
    _write(config_dir, "term_corrections.json", {
        "ドッカー": "Docker", "テラフォーム": "Terraform",
    })
    os.utime(config_dir / "term_corrections.json")
    assert "Terraform" in loader.get_hotwords_list()


def test_category_stats_include_dictionary(config_dir):
    _write(config_dir, "term_corrections.json", {"ドッカー": "Docker"})
    loader = HotwordLoader(str(config_dir))
    stats = loader.get_category_stats()
    assert stats["dictionary"]["total"] == 1
    assert stats["dictionary"]["enabled"] is True


def test_update_settings_dictionary(config_dir):
    _write(config_dir, "term_corrections.json", {"ドッカー": "Docker"})
    loader = HotwordLoader(str(config_dir))
    loader.update_settings({"dictionary": {"enabled": False}})
    assert "Docker" not in loader.get_hotwords_list()
    loader.update_settings({"dictionary": {"enabled": True}})
    assert "Docker" in loader.get_hotwords_list()


def test_no_terms_file(config_dir):
    loader = HotwordLoader(str(config_dir))
    assert loader.get_hotwords_list() == []
