"""Layer 1: initial_prompt 構築モジュール

faster-whisper の initial_prompt に渡す自然な日本語文を生成する。
Whisper は initial_prompt を「直前の発話」として扱うため、
単なる用語羅列ではなく文脈のある文にする方が効果的。

固有名詞・用語の認識は hotwords (デコーダバイアス) で処理するため、
このモジュールでは topic と participants のみを扱う。
"""

import logging

log = logging.getLogger(__name__)


def build_prompt(
    topic: str = "",
    participants: list[str] | None = None,
    max_chars: int = 180,
) -> str:
    """initial_prompt 用の文字列を生成する。

    Args:
        topic: 会議のトピック
        participants: 参加者名のリスト
        max_chars: 最大文字数 (224トークン ≒ 約180文字)

    Returns:
        initial_prompt 用の文字列
    """
    parts: list[str] = []

    if topic:
        parts.append(f"{topic}に関する会議です。")
    if participants:
        names = "、".join(participants)
        parts.append(f"参加者: {names}。")

    if not parts:
        return ""

    prompt = "".join(parts)
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars]

    return prompt
