"""Hotword ローダーモジュール

config/hotwords.json (JMnedict 由来)、config/hotwords_custom.json、
config/term_corrections.json (用語補正辞書の正解表記) を読み込み、
faster-whisper の hotwords パラメータ用文字列を生成する。

用語辞書の正解表記をデコーダバイアスに使うことで、辞書で事後補正して
いた誤認識を発生源 (ASR) で減らす。トークン予算が限られるため、誤認識
バリアント数が多い語 (= 慢性的に間違われる語) を優先する。

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
    # term_corrections.json 由来の正解表記 (カスタム語の次に優先)
    "dictionary": {"enabled": True, "max": 50},
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
        self._terms_path = self._config_dir / "term_corrections.json"

        # mtime 追跡
        self._hotwords_mtime: float = 0.0
        self._custom_mtime: float = 0.0
        self._settings_mtime: float = 0.0
        self._terms_mtime: float = 0.0

        # キャッシュ
        self._hotwords_data: dict = {}
        self._custom_words: list[str] = []
        self._dict_terms: list[str] = []  # 正解表記 (優先度順)
        self._settings: dict = dict(_DEFAULT_SETTINGS)
        self._cached_string: str = ""
        self._cached_list: list[str] = []
        self._dirty: bool = True

        self._load_all()

    def _load_all(self) -> None:
        """全ファイルを読み込む。"""
        self._load_hotwords()
        self._load_custom()
        self._load_terms()
        self._load_settings()
        self._rebuild_cache()

    def _load_terms(self) -> None:
        """config/term_corrections.json から正解表記を優先度順に読み込む。

        優先度 = 誤認識バリアント数の多い順 (同数なら短い語・辞書順)。
        バリアントが多い語ほど ASR が慢性的に間違えている語であり、
        デコーダバイアスの効果が大きい。
        """
        try:
            if self._terms_path.exists():
                data = json.loads(self._terms_path.read_text(encoding="utf-8"))
                counts: dict[str, int] = {}
                if isinstance(data, dict):
                    for _wrong, correct in data.items():
                        if isinstance(correct, str):
                            c = correct.strip()
                            # 1文字語 (例: "X") はバイアス効率が悪いので除外
                            if len(c) >= 2:
                                counts[c] = counts.get(c, 0) + 1
                self._dict_terms = [
                    t for t, _ in sorted(
                        counts.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0])
                    )
                ]
                self._terms_mtime = self._terms_path.stat().st_mtime
                log.info(f"Hotwords: {len(self._dict_terms)} dictionary terms loaded")
            else:
                self._dict_terms = []
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Dictionary terms load error: {e}")
            self._dict_terms = []
        self._dirty = True

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
                # dictionary は部分指定でもデフォルトと deep-merge する
                merged["dictionary"] = {
                    **_DEFAULT_SETTINGS["dictionary"],
                    **(data.get("dictionary") or {}),
                }
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
            if self._terms_path.exists():
                mt = self._terms_path.stat().st_mtime
                if mt > self._terms_mtime:
                    self._load_terms()
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

        # Phase 1.5: 用語辞書の正解表記 (ユーザー確認済みの語なので min_length 対象外)
        dict_cfg = self._settings.get("dictionary", {})
        dict_added = 0
        if dict_cfg.get("enabled", True):
            dict_max = dict_cfg.get("max", 50)
            for w in self._dict_terms:
                if dict_added >= dict_max:
                    break
                if token_count >= max_tokens:
                    break
                if _try_add(w):
                    dict_added += 1

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
                 f"(budget {max_tokens}, custom {len(self._custom_words)}, "
                 f"dictionary {dict_added}/{len(self._dict_terms)})")

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
        if "dictionary" in settings and isinstance(settings["dictionary"], dict):
            # 新しい dict を作る (モジュールデフォルトとの共有参照を避ける)
            self._settings["dictionary"] = {
                **_DEFAULT_SETTINGS["dictionary"],
                **self._settings.get("dictionary", {}),
                **settings["dictionary"],
            }
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
        """カテゴリ別の統計情報を返す (用語辞書由来は "dictionary" 擬似カテゴリ)。"""
        self._reload_if_changed()
        categories = self._hotwords_data.get("categories", {})
        cat_settings = self._settings.get("categories", {})
        stats = {}
        dict_cfg = self._settings.get("dictionary", {})
        stats["dictionary"] = {
            "total": len(self._dict_terms),
            "enabled": dict_cfg.get("enabled", True),
            "max": dict_cfg.get("max", 50),
            "active": min(len(self._dict_terms), dict_cfg.get("max", 50))
            if dict_cfg.get("enabled", True) else 0,
        }
        for cat_name, cat_words in categories.items():
            cs = cat_settings.get(cat_name, {})
            stats[cat_name] = {
                "total": len(cat_words),
                "enabled": cs.get("enabled", True),
                "max": cs.get("max", 100),
                "active": min(len(cat_words), cs.get("max", 100)) if cs.get("enabled", True) else 0,
            }
        return stats
