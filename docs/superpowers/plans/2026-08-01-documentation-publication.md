# SyncJarvis Documentation and Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn SyncJarvis into a safe, concise public engineering showcase without changing its browser-agent behavior.

**Architecture:** Public-facing claims live in an English root README; durable technical and operational detail lives in four focused `docs/` files. GitHub publication controls are additive: a permissive license, a private vulnerability-reporting policy, and CI that runs the existing static and test checks.

**Tech Stack:** Markdown, GitHub Actions, Python 3.12, pytest, Ruff.

## Global Constraints

- Do not add LLM features or rewrite the agent architecture.
- Do not claim CAPTCHA bypass; describe detection and pause for user intervention.
- Do not expose API keys, cookies, credentials, user profiles, production targets, or personal data.
- README copy is English and contains no SVG banner or badges.
- Use English-facing documentation and Russian commit messages, one finished step per commit.

---

### Task 1: Audit Publication Surface

**Files:**
- Inspect: `.env.example`, `README.md`, `assets/`, Git history, tracked files
- Create: `docs/publication-audit.md`

**Interfaces:**
- Consumes: existing tracked repository content.
- Produces: an auditable checklist of scanned surfaces and safe publication rules for Tasks 2–4.

- [ ] **Step 1: Scan tracked content for credential-like material and private artifacts**

Run:

```bash
git ls-files -z | xargs -0 rg -n -i \
  '(api[_-]?key|secret|token|password|cookie|authorization|bearer|github_pat_|ghp_)' \
  --glob '!docs/publication-audit.md'
```

Expected: only placeholders, explanatory documentation, or source-code identifiers; no credential values.

- [ ] **Step 2: Record the audit result and publication rules**

Create `docs/publication-audit.md` with the scan scope, result, safe-demo requirements, and a note that `.env.example` uses placeholders.

- [ ] **Step 3: Verify audit document contains no secret-shaped value**

Run:

```bash
rg -n -i '(github_pat_|ghp_|sk-[a-z0-9]{20,}|bearer\s+[a-z0-9])' docs/publication-audit.md
```

Expected: no matches.

### Task 2: Publish-Focused Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/architecture.md`, `docs/configuration.md`, `docs/safety-model.md`, `docs/demo.md`

**Interfaces:**
- Consumes: current runtime and LLM architecture documents and `.env.example`.
- Produces: a short public README that links to stable documentation pages.

- [ ] **Step 1: Replace the README with concise English public documentation**

Include exactly these sections: `What it does`, `Key engineering decisions`, `Architecture`, `Quick start`, `Demo`, `Safety model`, `Configuration`, `Limitations`, `Development`, and `License`.

- [ ] **Step 2: Write four focused documentation pages**

`docs/architecture.md` explains components, one-step data flow, contracts, and state transitions. `docs/configuration.md` groups required, model, safety, browser, and cost variables. `docs/safety-model.md` documents confirmation, CAPTCHA pause, limits, and risks. `docs/demo.md` supplies a safe, reproducible demo scenario with no real account or production target.

- [ ] **Step 3: Check publication claims and internal links**

Run:

```bash
rg -n -i 'captcha handled|captcha bypass|system online|autonomy high|shields\.io|<img' README.md docs
```

Expected: no matches.

### Task 3: GitHub Publication Controls

**Files:**
- Create: `LICENSE`, `SECURITY.md`, `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: existing Python test suite and requirements.
- Produces: MIT licensing, private vulnerability reporting guidance, and CI for lint, compilation, and tests.

- [ ] **Step 1: Add an MIT LICENSE**

Use the standard MIT text with copyright holder `SyncJarvis contributors` and year `2026`.

- [ ] **Step 2: Add a security policy**

`SECURITY.md` must request private GitHub vulnerability reporting, prohibit public disclosure before coordination, set no response-time promise, and state that credentials must not be included in reports.

- [ ] **Step 3: Add a narrow CI workflow**

Create `.github/workflows/ci.yml` for pushes and pull requests. It checks out the repository, uses Python 3.12, installs `requirements.txt` and `ruff`, then runs `python -m ruff check agent tests`, `python -m compileall -q agent`, and `python -m pytest -q`.

- [ ] **Step 4: Validate workflow structure and commands locally**

Run:

```bash
python3 -m ruff check agent tests
python3 -m compileall -q agent
python3 -m pytest -q
```

Expected: all commands exit successfully.

### Task 4: Final Publication Gate

**Files:**
- Verify: all modified and created files

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: a clean, reviewable commit series ready for a GitHub pull request.

- [ ] **Step 1: Verify tracked content and documentation links**

Run:

```bash
git ls-files | sort
rg -n '\]\((docs/[^)]+)\)' README.md
git diff --check
```

Expected: linked docs exist and the diff has no whitespace errors.

- [ ] **Step 2: Run complete verification**

Run:

```bash
python3 -m ruff check agent tests
python3 -m compileall -q agent
python3 -m pytest -q
```

Expected: lint clean, compilation successful, and all tests pass.

- [ ] **Step 3: Commit completed publication pass**

```bash
git add README.md docs LICENSE SECURITY.md .github/workflows/ci.yml
git commit -m "docs: подготовить публичную документацию"
```
