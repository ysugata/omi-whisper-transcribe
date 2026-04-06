"""Hotword ローダーモジュール

config/hotwords.json (JMnedict 由来) と config/hotwords_custom.json を
読み込み、faster-whisper の hotwords パラメータ用文字列を生成する。

TermCorrector と同じく mtime ベースの自動リロードに対応。
"""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_SETTINGS = {
    "enabled": True,
    "max_total": 500,
    "max_tokens": 200,  # Whisper max_length=448; leave room for decoding
    "min_length": 3,  # Filter out short entries that cause noise contamination
    "categories": {
        "place": {"enabled": True, "max": 200},
        "company": {"enabled": True, "max": 100},
        "station": {"enabled": True, "max": 100},
        "surname": {"enabled": False, "max": 50},
        "given": {"enabled": False, "max": 50},
        "product": {"enabled": True, "max": 50},
    },
}

# Whisper decoder max_length is 448 tokens. Hotwords are tokenized and
# prepended as prompt tokens, consuming part of this budget.  If the total
# prompt length exceeds 448, faster-whisper raises:
#   ValueError: The maximum decoding length must be > 0
# We cap hotword tokens to max_tokens (default 200) so that >=248 tokens
# remain for actual decoding output.  This value was determined empirically:
# at ~225 tokens the crash occurs intermittently; 200 gives safe headroom.
_MAX_HOTWORD_TOKENS_DEFAULT = 200


class HotwordLoader:
    """Hotword のロードと管理を行うクラス。"""

    def __init__(self, config_dir: str = "./config") -> None:
        self._config_dir = Path(config_dir)
        self._hotwords_path = self._config_dir / "hotwords.json"
        self._custom_path = self._config_dir / "hotwords_custom.json"
        self._settings_path = self._config_dir / "hotwords_settings.json"

        # mtime 追跡
        self._hotwords_mtime: float = 0.0
        self._custom_mtime: float = 0.0
        self._settings_mtime: float = 0.0

        # キャッシュ
        self._hotwords_data: dict = {}
        self._custom_words: list[str] = []
        self._settings: dict = dict(_DEFAULT_SETTINGS)
        self._cached_string: str = ""
        self._cached_list: list[str] = []
        self._dirty: bool = True

        self._load_all()

    def _load_all(self) -> None:
        """全ファイルを読み込む。"""
        self._load_hotwords()
        self._load_custom()
        self._load_settings()
        self._rebuild_cache()

    def _load_hotwords(self) -> None:
        """config/hotwords.json を読み込む。"""
        try:
            if self._hotwords_path.exists():
                data = json.loads(self._hotwords_path.read_text(encoding="utf-8"))
                self._hotwords_data = data
                self._hotwords_mtime = self._hotwords_path.stat().st_mtime
                cats = data.get("categories", {})
                total = sum(len(v) for v in cats.values())
                log.info(f"Hotwords loaded: {total} words in {len(cats)} categories")
            else:
                self._hotwords_data = {}
                log.info("Hotwords: hotwords.json not found (run tools/build_hotwords.py)")
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Hotwords load error: {e}")
            self._hotwords_data = {}
        self._dirty = True

    def _load_custom(self) -> None:
        """config/hotwords_custom.json を読み込む。"""
        try:
            if self._custom_path.exists():
                data = json.loads(self._custom_path.read_text(encoding="utf-8"))
                self._custom_words = data if isinstance(data, list) else []
                self._custom_mtime = self._custom_path.stat().st_mtime
            else:
                self._custom_words = []
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Custom hotwords load error: {e}")
            self._custom_words = []
        self._dirty = True

    def _load_settings(self) -> None:
        """config/hotwords_settings.json を読み込む。"""
        try:
            if self._settings_path.exists():
                data = json.loads(self._settings_path.read_text(encoding="utf-8"))
                # デフォルト設定とマージ
                merged = dict(_DEFAULT_SETTINGS)
                merged.update(data)
                # categories も個別にマージ
                merged_cats = dict(_DEFAULT_SETTINGS["categories"])
                for cat, val in data.get("categories", {}).items():
                    if cat in merged_cats:
                        merged_cats[cat].update(val)
                    else:
                        merged_cats[cat] = val
                merged["categories"] = merged_cats
                self._settings = merged
                self._settings_mtime = self._settings_path.stat().st_mtime
            else:
                self._settings = dict(_DEFAULT_SETTINGS)
                self._save_settings()
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Hotwords settings load error: {e}")
            self._settings = dict(_DEFAULT_SETTINGS)
        self._dirty = True

    def _save_settings(self) -> None:
        """設定ファイルを保存する。"""
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(
            json.dumps(self._settings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._settings_mtime = self._settings_path.stat().st_mtime

    def _save_custom(self) -> None:
        """カスタム hotwords ファイルを保存する。"""
        self._custom_path.parent.mkdir(parents=True, exist_ok=True)
        self._custom_path.write_text(
            json.dumps(self._custom_words, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._custom_mtime = self._custom_path.stat().st_mtime

    def _reload_if_changed(self) -> None:
        """ファイルが変更されていたら再読み込み。"""
        try:
            if self._hotwords_path.exists():
                mt = self._hotwords_path.stat().st_mtime
                if mt > self._hotwords_mtime:
                    self._load_hotwords()
            if self._custom_path.exists():
                mt = self._custom_path.stat().st_mtime
                if mt > self._custom_mtime:
                    self._load_custom()
            if self._settings_path.exists():
                mt = self._settings_path.stat().st_mtime
                if mt > self._settings_mtime:
                    self._load_settings()
        except OSError:
            pass

        if self._dirty:
            self._rebuild_cache()

    @staticmethod
    def _estimate_tokens(word: str) -> int:
        """Estimate Whisper token count for a hotword.

        Empirically, Japanese hotwords tokenize at roughly 1 token per
        character.  The ", " separator between words adds ~1 token.
        We use len(word) + 1 (for separator) as a conservative estimate.
        """
        return len(word) + 1  # +1 for ", " separator

    def _rebuild_cache(self) -> None:
        """設定に基づいてキャッシュされた hotword リストを再構築する。

        カスタム語を最優先でトークン予算に含め、残り予算でカテゴリ語を追加。
        Whisper の max_length=448 を超えないようトークン数で制限する。
        """
        max_total = self._settings.get("max_total", 500)
        max_tokens = self._settings.get("max_tokens", _MAX_HOTWORD_TOKENS_DEFAULT)
        min_length = self._settings.get("min_length", 3)
        cat_settings = self._settings.get("categories", {})

        seen: set[str] = set()
        result: list[str] = []
        token_count = 0

        def _try_add(word: str) -> bool:
            """Add word if it fits within token budget. Returns True if added."""
            nonlocal token_count
            if word in seen:
                return False
            est = self._estimate_tokens(word)
            if token_count + est > max_tokens:
                return False
            if len(result) >= max_total:
                return False
            seen.add(word)
            result.append(word)
            token_count += est
            return True

        # Phase 1: カスタム語を最優先 (min_length 対象外)
        for w in self._custom_words:
            w = w.strip()
            if w:
                _try_add(w)

        # Phase 2: カテゴリ語を追加 (残りのトークン予算で)
        categories = self._hotwords_data.get("categories", {})
        filtered_count = 0
        for cat_name, cat_words in categories.items():
            cs = cat_settings.get(cat_name, {})
            if not cs.get("enabled", True):
                continue
            cat_max = cs.get("max", 100)
            added = 0
            for w in cat_words:
                if added >= cat_max:
                    break
                if token_count >= max_tokens:
                    break
                if len(w) < min_length:
                    filtered_count += 1
                    continue
                if _try_add(w):
                    added += 1

        if filtered_count:
            log.info(f"Hotwords: filtered {filtered_count} entries shorter than {min_length} chars")

        log.info(f"Hotwords: {len(result)} words, ~{token_count} tokens "
                 f"(budget {max_tokens}, custom {len(self._custom_words)})")

        self._cached_list = result
        self._cached_string = ", ".join(result)
        self._dirty = False

    def get_hotwords_string(self) -> str:
        """カンマ区切りの hotwords 文字列を返す (model.transcribe(hotwords=...) 用)。"""
        self._reload_if_changed()
        return self._cached_string

    def get_hotwords_list(self) -> list[str]:
        """hotwords のリストを返す。"""
        self._reload_if_changed()
        return list(self._cached_list)

    def get_count(self) -> int:
        """現在の有効 hotwords 数を返す。"""
        self._reload_if_changed()
        return len(self._cached_list)

    def get_settings(self) -> dict:
        """現在の設定を返す。"""
        self._reload_if_changed()
        return dict(self._settings)

    def update_settings(self, settings: dict) -> None:
        """設定を更新して保存する。"""
        if "enabled" in settings:
            self._settings["enabled"] = bool(settings["enabled"])
        if "max_total" in settings:
            self._settings["max_total"] = int(settings["max_total"])
        if "min_length" in settings:
            self._settings["min_length"] = max(1, int(settings["min_length"]))
        if "categories" in settings:
            for cat, val in settings["categories"].items():
                if cat in self._settings["categories"]:
                    self._settings["categories"][cat].update(val)
                else:
                    self._settings["categories"][cat] = val
        self._save_settings()
        self._dirty = True
        self._rebuild_cache()

    def add_custom(self, word: str) -> None:
        """カスタム hotword を追加する。"""
        word = word.strip()
        if not word:
            return
        if word not in self._custom_words:
            self._custom_words.append(word)
            self._save_custom()
            self._dirty = True
            self._rebuild_cache()

    def remove_custom(self, word: str) -> None:
        """カスタム hotword を削除する。"""
        word = word.strip()
        if word in self._custom_words:
            self._custom_words.remove(word)
            self._save_custom()
            self._dirty = True
            self._rebuild_cache()

    def get_custom_words(self) -> list[str]:
        """カスタム hotwords のリストを返す。"""
        self._reload_if_changed()
        return list(self._custom_words)

    def get_category_stats(self) -> dict:
        """カテゴリ別の統計情報を返す。"""
        self._reload_if_changed()
        categories = self._hotwords_data.get("categories", {})
        cat_settings = self._settings.get("categories", {})
        stats = {}
        for cat_name, cat_words in categories.items():
            cs = cat_settings.get(cat_name, {})
            stats[cat_name] = {
                "total": len(cat_words),
                "enabled": cs.get("enabled", True),
                "max": cs.get("max", 100),
                "active": min(len(cat_words), cs.get("max", 100)) if cs.get("enabled", True) else 0,
            }
        return stats
