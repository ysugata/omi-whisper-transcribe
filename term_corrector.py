"""Layer 2: 用語補正モジュール

JSON 辞書による既知の誤変換パターンの後処理置換。
辞書ファイルが存在しない場合は IT 技術用語の初期辞書を自動生成する。
"""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

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
        self._mtime: float = 0.0
        self._load_or_init()

    def _load_or_init(self) -> None:
        """ファイルがあれば読み込み、なければ初期辞書を作成。"""
        if self.dict_path.exists():
            self.load()
        else:
            self._corrections = dict(_INITIAL_DICT)
            self.save()
            log.info(f"初期辞書を生成: {self.dict_path} ({len(self._corrections)} entries)")

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

    def remove(self, wrong: str) -> None:
        """誤変換パターンを削除し、ファイルに保存。"""
        self._corrections.pop(wrong, None)
        self.save()

    def correct(self, text: str) -> str:
        """テキスト中の全ての誤変換パターンを修正して返す。

        ファイルが外部から変更されていたら自動的に再読み込みする。
        長い誤変換パターンから先に適用して、部分一致の問題を回避する。
        """
        self._reload_if_changed()
        for wrong in sorted(self._corrections, key=len, reverse=True):
            if wrong in text:
                text = text.replace(wrong, self._corrections[wrong])
        return text

    def correct_count(self, text: str) -> tuple:
        """テキストを補正し、(補正後テキスト, 補正数) を返す。"""
        self._reload_if_changed()
        count = 0
        for wrong in sorted(self._corrections, key=len, reverse=True):
            if wrong in text:
                text = text.replace(wrong, self._corrections[wrong])
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
