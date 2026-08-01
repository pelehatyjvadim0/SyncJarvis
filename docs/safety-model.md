# Safety Model

SyncJarvis is intended for supervised use. Its controls reduce common browser-automation risks but cannot make every task safe.

## User confirmation

The runtime can pause before actions classified as high impact, including irreversible or financial steps. Review the proposed action and the visible page state before confirming. Do not treat confirmation as permission to automate actions that violate a website's rules or a user's expectations.

## CAPTCHA and manual-only states

The agent does not solve CAPTCHAs. When the page appears to require a CAPTCHA or other manual intervention, the loop waits and can mark the task as blocked after its configured limit. Complete the step yourself only when authorized, then resume or stop the task.

## Bounded execution

Task, subtask, retry, and model-response limits prevent uncontrolled looping and spending. Anti-loop checks use recent actions and page-change evidence to detect repeated ineffective behavior. A limit can result in a partial outcome rather than a claim of success.

## Known risks

- An LLM can misunderstand a goal or choose an unsuitable element.
- Accessibility data and screenshots can be incomplete or stale.
- A successful click does not prove the intended business result.
- Third-party sites can change or display unexpected content.

Use test accounts and non-production data. Never put credentials in tasks, prompts, screenshots, issue reports, or commits. Inspect the browser before approving any consequential action.
