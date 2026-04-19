from __future__ import annotations


def url_control_priority(url: str) -> int:
    # Меньше — лучше для выбора вкладки: http(s) предпочтительнее about:blank / новая вкладка / devtools.
    u = (url or "").strip().lower()
    if u.startswith("http://") or u.startswith("https://"):
        return 0
    if u.startswith("file:") or u.startswith("data:"):
        return 1
    if u in {"about:blank", "about:srcdoc"} or "new-tab-page" in u or u in {"edge://newtab/", "chrome://new-tab-page/"}:
        return 2
    if u.startswith("devtools://") or u.startswith("chrome-extension://") or u.startswith("chrome://") or u.startswith("edge://"):
        return 3
    return 2 if u.startswith("about:") else 1
