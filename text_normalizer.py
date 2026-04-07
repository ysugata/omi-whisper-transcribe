"""Japanese text normalization for ASR output.

Normalizes half-width/full-width characters, removes duplicate characters,
and standardizes punctuation. Works with both Whisper and Qwen3-ASR.

Dependencies: neologdn (optional, for advanced normalization)
No GPU required.
"""

import logging
import re

log = logging.getLogger(__name__)

_neologdn = None
_neologdn_checked = False


def _get_neologdn():
    """Lazy-load neologdn."""
    global _neologdn, _neologdn_checked
    if not _neologdn_checked:
        _neologdn_checked = True
        try:
            import neologdn as _mod
            _neologdn = _mod
            log.info("neologdn loaded")
        except ImportError:
            log.debug("neologdn not available, using basic normalization")
    return _neologdn


# Half-width katakana → full-width katakana mapping
_HW_KANA_MAP = str.maketrans(
    'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜｦﾝｧｨｩｪｫｯｬｭｮﾞﾟ',
    'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンァィゥェォッャュョ゛゜',
)

# Full-width alphanumeric → half-width
_FW_ALPHA_RE = re.compile(r'[Ａ-Ｚａ-ｚ０-９]')
_FW_ALPHA_OFFSET = ord('Ａ') - ord('A')


def _fw_to_hw_char(match):
    c = match.group(0)
    code = ord(c)
    if ord('Ａ') <= code <= ord('Ｚ'):
        return chr(code - (ord('Ａ') - ord('A')))
    elif ord('ａ') <= code <= ord('ｚ'):
        return chr(code - (ord('ａ') - ord('a')))
    elif ord('０') <= code <= ord('９'):
        return chr(code - (ord('０') - ord('0')))
    return c


# Duplicate whitespace
_MULTI_SPACE_RE = re.compile(r'[ 　]+')
# Zero-width chars
_ZERO_WIDTH_RE = re.compile(r'[\u200b\u200c\u200d\ufeff]')


def normalize(text: str) -> str:
    """Normalize ASR output text.

    Applies:
    1. neologdn normalization (if available) — handles full-width, duplicate chars, etc.
    2. Half-width katakana → full-width
    3. Full-width alphanumeric → half-width
    4. Zero-width character removal
    5. Whitespace normalization
    """
    if not text:
        return text

    # Try neologdn first (comprehensive normalization)
    neo = _get_neologdn()
    if neo:
        text = neo.normalize(text)
    else:
        # Basic fallback normalization
        # Half-width katakana → full-width
        text = text.translate(_HW_KANA_MAP)
        # Full-width alphanumeric → half-width
        text = _FW_ALPHA_RE.sub(_fw_to_hw_char, text)

    # Remove zero-width characters
    text = _ZERO_WIDTH_RE.sub('', text)

    # Normalize whitespace
    text = _MULTI_SPACE_RE.sub(' ', text)

    return text.strip()
