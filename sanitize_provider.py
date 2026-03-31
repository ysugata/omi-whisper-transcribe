"""Sanitize provider interface — プロバイダ抽象化.

将来 Gemini / Claude / OpenAI を差し替えやすくするための ABC。
各プロバイダは sanitize() メソッドを実装する。
"""

from abc import ABC, abstractmethod
from typing import Optional


class SanitizeResult:
    """sanitize() の戻り値を統一的に保持するデータクラス."""

    __slots__ = ("session_id", "data", "warnings", "usage")

    def __init__(
        self,
        session_id: str,
        data: dict,
        warnings: Optional[list[str]] = None,
        usage: Optional[dict] = None,
    ) -> None:
        self.session_id = session_id
        self.data = data            # スキーマ準拠の出力 dict
        self.warnings = warnings or []
        self.usage = usage or {}    # {"input_tokens": N, "output_tokens": N, ...}


class SanitizeProvider(ABC):
    """匿名化・構造化プロバイダの共通インターフェース.

    実装例:
    - ResponsesSanitizer  (OpenAI Responses API)
    - TrainingSanitizer   (Gemini / OpenAI Chat Completions API — 既存)
    """

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """プロバイダが使用可能かどうか."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """プロバイダ識別名 (ログ用)."""
        ...

    @abstractmethod
    def sanitize(
        self,
        session_id: str,
        raw_text: str,
        language: str = "ja",
    ) -> Optional[SanitizeResult]:
        """セッションテキストを匿名化・構造化する.

        Args:
            session_id: セッション識別子 (例: "2026-03-31/09-35-13")
            raw_text: ASR 出力のプレーンテキスト
            language: 言語コード

        Returns:
            SanitizeResult on success, None on unrecoverable error.
        """
        ...
