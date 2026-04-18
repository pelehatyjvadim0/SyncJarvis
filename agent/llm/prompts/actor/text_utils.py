from __future__ import annotations


def _trim_text(value: str | None, max_len: int) -> str:
    if not value:
        return ""
    text = value.strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."
