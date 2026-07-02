"""Layer 2: 用語補正モジュール

JSON 辞書による既知の誤変換パターンの後処理置換。
辞書ファイルが存在しない場合は IT 技術用語の初期辞書を自動生成する。

置換は語境界を考慮する: カタカナ語パターンの前後がカタカナの場合は
複合語の一部とみなして置換しない (例: 辞書に「ログ→log」があっても
「プログラム」「ブログ」は破壊しない)。
"""

import json
import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)

# カタカナ + 長音符 (複合語判定に使用)
_KATAKANA_CLASS = "ァ-ヴーヵヶ"


def compile_boundary_pattern(wrong: str) -> re.Pattern:
    """誤変換パターンを語境界つき正規表現にコンパイルする。

    - パターンがカタカナで始まる/終わる場合、隣接文字がカタカナなら
      マッチしない (複合語保護: プログラム, ガイドライン 等)
    - パターンが英数字で始まる/終わる場合、隣接文字が英数字なら
      マッチしない (LLM の中の LM 等)
    """
    prefix = ""
    suffix = ""
    if re.match(f"[{_KATAKANA_CLASS}]", wrong[0]):
        prefix = f"(?<![{_KATAKANA_CLASS}])"
    elif wrong[0].isascii() and wrong[0].isalnum():
        prefix = r"(?<![A-Za-z0-9])"
    if re.match(f"[{_KATAKANA_CLASS}]", wrong[-1]):
        suffix = f"(?![{_KATAKANA_CLASS}])"
    elif wrong[-1].isascii() and wrong[-1].isalnum():
        suffix = r"(?![A-Za-z0-9])"
    return re.compile(prefix + re.escape(wrong) + suffix)

# fmt: off
_INITIAL_DICT: dict[str, str] = {
    # --- コンテナ・オーケストレーション ---
    "クーバネティス":     "Kubernetes",
    "クバネティス":       "Kubernetes",
    "クーバネテス":       "Kubernetes",
    "クーバーネティス":   "Kubernetes",
    "クーベネティス":     "Kubernetes",
    "クーバネテイス":     "Kubernetes",
    "ドッカー":           "Docker",
    "ドッカ":             "Docker",
    "ヘルム":             "Helm",

    # --- CI/CD・DevOps ---
    "テラフォーム":       "Terraform",
    "テラホーム":         "Terraform",
    "シーアイシーディー": "CI/CD",
    "ギットハブ":         "GitHub",
    "ギットハブアクションズ": "GitHub Actions",
    "ギットラブ":         "GitLab",
    "ギットラボ":         "GitLab",
    "アルゴシーディー":   "ArgoCD",
    "ジェンキンス":       "Jenkins",
    "アンシブル":         "Ansible",

    # --- モニタリング ---
    "プロメテウス":       "Prometheus",
    "プロメシウス":       "Prometheus",
    "グラファーナ":       "Grafana",
    "グラファナ":         "Grafana",
    "データドッグ":       "Datadog",
    "データドック":       "Datadog",

    # --- 言語・フレームワーク ---
    "リアクト":           "React",
    "ネクストジェーエス": "Next.js",
    "タイプスクリプト":   "TypeScript",
    "ジャバスクリプト":   "JavaScript",
    "パイソン":           "Python",
    "ファストエーピーアイ": "FastAPI",
    "ジャンゴ":           "Django",
    "フラスコ":           "Flask",

    # --- インフラ・ネットワーク ---
    "ジーアールピーシー": "gRPC",
    "エンボイ":           "Envoy",
    "イスティオ":         "Istio",
    "ウェブソケット":     "WebSocket",
    "ニンクス":           "Nginx",
    "エヌジンクス":       "Nginx",
    "エンジンエックス":   "Nginx",
    "レディス":           "Redis",
    "レデス":             "Redis",
    "ポストグレス":       "PostgreSQL",
    "ポストグレ":         "PostgreSQL",
    "マイエスキューエル": "MySQL",

    # --- クラウド ---
    "エーダブリューエス": "AWS",
    "ジーシーピー":       "GCP",
    "アジュール":         "Azure",

    # --- AI/ML ---
    "ウィスパー":         "Whisper",
    "ウイスパー":         "Whisper",
}
# fmt: on


class TermCorrector:
    """JSON 辞書による用語補正"""

    def __init__(self, dict_path: str = "term_corrections.json") -> None:
        """辞書ファイルを読み込む。ファイルが無ければ初期辞書を生成して保存。"""
        self.dict_path = Path(dict_path)
        self._corrections: dict[str, str] = {}
        self._compiled: list[tuple[re.Pattern, str]] = []
        self._mtime: float = 0.0
        self._load_or_init()

    def _rebuild_compiled(self) -> None:
        """境界つき置換パターンを再構築 (長いパターン優先)。恒等マッピングは除外。"""
        self._compiled = [
            (compile_boundary_pattern(wrong), correct)
            for wrong, correct in sorted(
                self._corrections.items(), key=lambda kv: len(kv[0]), reverse=True
            )
            if wrong and wrong != correct
        ]

    def _load_or_init(self) -> None:
        """ファイルがあれば読み込み、なければ初期辞書を作成。"""
        if self.dict_path.exists():
            self.load()
        else:
            self._corrections = dict(_INITIAL_DICT)
            self.save()
            log.info(f"初期辞書を生成: {self.dict_path} ({len(self._corrections)} entries)")
        self._rebuild_compiled()

    def load(self) -> None:
        """辞書ファイルを読み込み。"""
        try:
            data = json.loads(self.dict_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._corrections = data
                self._mtime = self.dict_path.stat().st_mtime
                log.info(f"辞書読み込み: {len(self._corrections)} entries")
            else:
                log.warning("辞書ファイル形式不正、初期辞書にフォールバック")
                self._corrections = dict(_INITIAL_DICT)
                self.save()
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"辞書読み込みエラー: {e}、初期辞書にフォールバック")
            self._corrections = dict(_INITIAL_DICT)
            self.save()
        self._rebuild_compiled()

    def _reload_if_changed(self) -> None:
        """ファイルが変更されていたら再読み込み。"""
        try:
            mtime = self.dict_path.stat().st_mtime
            if mtime > self._mtime:
                self.load()
        except OSError:
            pass

    def save(self) -> None:
        """現在の辞書をファイルに保存。"""
        self.dict_path.parent.mkdir(parents=True, exist_ok=True)
        self.dict_path.write_text(
            json.dumps(self._corrections, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._mtime = self.dict_path.stat().st_mtime

    def add(self, wrong: str, correct: str) -> None:
        """新しい誤変換パターンを追加し、ファイルに保存。"""
        self._corrections[wrong] = correct
        self.save()
        self._rebuild_compiled()

    def remove(self, wrong: str) -> None:
        """誤変換パターンを削除し、ファイルに保存。"""
        self._corrections.pop(wrong, None)
        self.save()
        self._rebuild_compiled()

    def correct(self, text: str) -> str:
        """テキスト中の全ての誤変換パターンを修正して返す。

        ファイルが外部から変更されていたら自動的に再読み込みする。
        長い誤変換パターンから先に適用して、部分一致の問題を回避する。
        語境界を考慮し、複合語の内部 (プログラム中のログ等) は置換しない。
        """
        return self.correct_count(text)[0]

    def correct_count(self, text: str) -> tuple:
        """テキストを補正し、(補正後テキスト, 補正数) を返す。"""
        self._reload_if_changed()
        count = 0
        for pattern, correct in self._compiled:
            # 関数置換: correct 内の \ や \g を置換エスケープと解釈させない
            text, n = pattern.subn(lambda _m, _c=correct: _c, text)
            if n:
                count += 1
        return text, count

    def get_known_values(self) -> set:
        """辞書の correct 側の値の集合を返す (スクリーナーのカタカナ既知チェック用)。"""
        self._reload_if_changed()
        return set(self._corrections.values())

    def list_corrections(self) -> dict[str, str]:
        """現在の辞書の全エントリを返す。"""
        self._reload_if_changed()
        return dict(self._corrections)
