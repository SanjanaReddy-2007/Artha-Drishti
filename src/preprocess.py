"""Text preprocessing. Stateless (no learned parameters), so it cannot leak across splits.

Steps: HTML-unescape -> URLs to 'url' -> @mentions to '@user' -> '#tag' to 'tag'
-> emojis to words -> lowercase -> collapse whitespace.
Deliberately kept: punctuation, repeated letters and capitalisation-independent words,
because these are common sarcasm cues.
"""
from __future__ import annotations

import html
import math
import re

try:  # emoji -> words (same idea as the earlier `preprocess.py --emoji convert`)
    import emoji as _emoji
    EMOJI_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    _emoji = None
    EMOJI_AVAILABLE = False

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")


def convert_emojis(text: str) -> str:
    """Replace emojis by their names, e.g. a crying-laughing face -> 'face with tears of joy'."""
    if not EMOJI_AVAILABLE:
        return text
    return _emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")


def clean_text(text, convert_emoji: bool = True) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    t = html.unescape(str(text))
    t = URL_RE.sub(" url ", t)
    t = MENTION_RE.sub(" @user ", t)
    t = HASHTAG_RE.sub(r" \1 ", t)
    if convert_emoji:
        t = convert_emojis(t)
    return WS_RE.sub(" ", t.lower()).strip()
