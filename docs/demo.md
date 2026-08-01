# Safe Demo Guide

This guide defines a reproducible demo that can be recorded and shared without exposing personal data or production targets.

## Scenario

Use the local fixture at `tests/fixtures/viewport_scroll.html`. Start a local static server from the repository root:

```bash
python -m http.server 8000
```

Open `http://127.0.0.1:8000/tests/fixtures/viewport_scroll.html` in a fresh browser profile. Demonstrate only these steps:

1. Start SyncJarvis with a test-only goal such as: “Find the in-viewport control and scroll to the below-the-fold control.”
2. Show the viewport and accessibility-grounded decision in the local console history.
3. Stop after the below-the-fold control is visible.

## Recording rules

- Use no logged-in accounts, API keys, browser-sync data, or real customer information.
- Do not record external sites, checkout pages, or CAPTCHA screens.
- Blur or remove operating-system notifications and browser profile names.
- Keep the recording short and label it as an experimental local demonstration.

## Before publishing a recording

Run the verification commands from the README, inspect the entire recording frame by frame, and confirm that the fixture page, prompt, terminal output, and browser chrome contain no credentials or personal information.
