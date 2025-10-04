"""Configuration knobs for TTS pipeline.

Users can edit `VOICE_STYLE_WHITELIST` to restrict which speaker styles are used.
"""

MAX_CHARS_PER_CHUNK = 15_000
"""Maximum number of characters allowed per chunk when splitting long input text."""

TARGET_CHARS_PER_CHUNK = 4_000
"""Preferred chunk size. Actual chunks stay at or under this value when possible."""

AVAILABLE_VOICE_STYLES = [
    "EN-AU",
    "EN-BR",
    "EN-Default",
    "EN-US",
    "EN_INDIA",
]
"""Voices bundled with the default EN model.
Update this list if checkpoints change."""

VOICE_STYLE_WHITELIST = ["EN-US"]  # e.g., ["EN-Default", "EN-US"]
"""List of speaker keys to keep. Leave empty to use every available voice style."""

PRESERVE_CHUNK_OUTPUTS = False
"""If True, keep intermediate chunk wav files after merging. Defaults to cleanup."""
