#!/usr/bin/env python3
"""JMnedict から日本語固有名詞を抽出し config/hotwords.json を生成する。

ホスト上で 1 回だけ実行するビルドスクリプト。
Python 3.10+ stdlib のみ使用 (追加依存なし)。

使い方:
  python3 tools/build_hotwords.py

データソース:
  jmdict-simplified の JMnedict JSON (GitHub Releases)
  ライセンス: CC-BY-SA 3.0
"""

import io
import json
import os
import sys
import urllib.request
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path

# ============================================================
# 設定
# ============================================================

JMNEDICT_URL = ""  # 自動検出 (GitHub API から最新リリースを取得)

# JMnedict の name_type → カテゴリマッピング
# 参考: https://www.edrdg.org/jmnedict/enamdict_doc.html
_TYPE_TO_CATEGORY = {
    # place
    "place": "place",
    "country": "place",
    "region": "place",
    "lake": "place",
    "sea": "place",
    "river": "place",
    "mountain": "place",
    "island": "place",
    # station
    "station": "station",
    # company / organization
    "company": "company",
    "organization": "company",
    # surname / given name
    "surname": "surname",
    "fem": "given",       # 女性名
    "masc": "given",      # 男性名
    "given": "given",
    # product
    "product": "product",
    "work": "product",    # 作品名
}

# カテゴリ別抽出上限 (ビルド時)
_BUILD_LIMITS = {
    "place": 2000,
    "company": 1000,
    "station": 1000,
    "surname": 500,
    "given": 500,
    "product": 500,
}


def _resolve_latest_url() -> str:
    """GitHub API から最新の jmnedict ZIP の URL を取得する。"""
    api_url = "https://api.github.com/repos/scriptin/jmdict-simplified/releases/latest"
    print(f"Fetching latest release info from GitHub API...")
    req = urllib.request.Request(api_url, headers={
        "User-Agent": "build_hotwords/1.0",
        "Accept": "application/vnd.github+json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        release = json.load(resp)

    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if "jmnedict" in name.lower() and name.endswith(".json.zip"):
            url = asset["browser_download_url"]
            print(f"Found: {name}")
            return url

    raise RuntimeError("jmnedict ZIP asset not found in latest release")


def download_jmnedict(url: str) -> dict:
    """JMnedict JSON をダウンロードして辞書として返す。"""
    if not url:
        url = _resolve_latest_url()

    print(f"Downloading JMnedict from: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "build_hotwords/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    print(f"Downloaded {len(data) / 1024 / 1024:.1f} MB")

    # ZIP ファイルを展開
    if url.endswith(".zip") or url.endswith(".json.zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            json_name = [n for n in names if n.endswith(".json")][0]
            print(f"Extracting: {json_name}")
            with zf.open(json_name) as f:
                return json.load(f)
    else:
        return json.loads(data)


def extract_proper_nouns(jmnedict: dict) -> dict[str, list[str]]:
    """JMnedict エントリからカテゴリ別に日本語固有名詞を抽出する。

    jmdict-simplified の JMnedict JSON 構造:
      entry.translation[].type[] — name_type のリスト (e.g. "place", "given", "surname")
      entry.kanji[].text — 漢字表記
      entry.kana[].text — カナ表記

    Returns:
        {"place": ["東京", ...], "company": ["トヨタ", ...], ...}
    """
    # カテゴリ → set of words
    categories: dict[str, set[str]] = defaultdict(set)

    words = jmnedict.get("words", [])
    print(f"Processing {len(words)} JMnedict entries...")

    for entry in words:
        # translation セクションから type (name_type) を取得
        translations = entry.get("translation", [])
        if not translations:
            continue

        entry_types = set()
        for tr in translations:
            for nt in tr.get("type", []):
                cat = _TYPE_TO_CATEGORY.get(nt)
                if cat:
                    entry_types.add(cat)

        if not entry_types:
            continue

        # 日本語表記を取得 (kanji > kana)
        kanji_forms = entry.get("kanji", [])
        kana_forms = entry.get("kana", [])

        words_to_add = []

        # 漢字表記があればそれを優先
        for kf in kanji_forms:
            text = kf.get("text", "")
            if text and len(text) >= 2:
                words_to_add.append(text)

        # 漢字表記がなければカナ表記
        if not words_to_add:
            for kf in kana_forms:
                text = kf.get("text", "")
                if text and len(text) >= 2:
                    words_to_add.append(text)

        # 全カテゴリに追加
        for cat in entry_types:
            for w in words_to_add:
                categories[cat].add(w)

    # set → sorted list, 上限まで
    result = {}
    for cat, limit in _BUILD_LIMITS.items():
        words_set = categories.get(cat, set())
        # 短い語を優先 (hotwords としてはコンパクトな方が効果的)
        sorted_words = sorted(words_set, key=lambda w: (len(w), w))
        result[cat] = sorted_words[:limit]
        print(f"  {cat}: {len(words_set)} total → {len(result[cat])} selected")

    return result


def build_hotwords_json(categories: dict[str, list[str]], output_path: Path) -> None:
    """config/hotwords.json を生成する。"""
    output = {
        "version": date.today().isoformat(),
        "source": "JMnedict (CC-BY-SA 3.0)",
        "categories": categories,
        "custom": [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total = sum(len(v) for v in categories.values())
    print(f"\nGenerated: {output_path}")
    print(f"Total words: {total}")
    for cat, words in categories.items():
        print(f"  {cat}: {len(words)}")


def main():
    # プロジェクトルートを基準にパスを決定
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_path = project_root / "config" / "hotwords.json"

    # カスタム URL サポート
    url = sys.argv[1] if len(sys.argv) > 1 else JMNEDICT_URL

    jmnedict = download_jmnedict(url)
    categories = extract_proper_nouns(jmnedict)
    build_hotwords_json(categories, output_path)

    print("\nDone! You can now use hotwords in omi_whisper_bridge.")


if __name__ == "__main__":
    main()
