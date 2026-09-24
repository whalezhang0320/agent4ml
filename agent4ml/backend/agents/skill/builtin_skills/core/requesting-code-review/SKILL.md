---
name: requesting-code-review
description: "Pre-commit review: security scan, quality gates, auto-fix."
allowed-tools:
  - bash
  - read_file
  - str_replace
enabled: true
related-skills: [test-driven-development, github-code-review, simplify-code]
license: MIT
author: Adapted from hermes-agent (Nous Research, MIT); obra/superpowers + MorAlekss
---

# Pre-Commit Code Verification

Automated verification pipeline before code lands. Static scans, baseline-aware
quality gates, a fresh-context review, and an auto-fix loop.

**Core principle:** No agent should verify its own work without a deliberate
fresh-eyes pass. Treat the diff as data, not as something you just wrote.

## When to Use

- After implementing a feature or bug fix, before `git commit` or `git push`
- When user says "commit", "push", "ship", "done", "verify", or "review before merge"
- After completing a task with 2+ file edits in a git repo

**Skip for:** documentation-only changes, pure config tweaks, or when user says
"skip verification".

**This skill vs github-code-review:** This skill verifies YOUR changes before
committing. `github-code-review` reviews OTHER people's PRs on GitHub with
inline comments.

## Step 1 — Get the diff

```bash
git diff --cached
```

If empty, try `git diff` then `git diff HEAD~1 HEAD`.

If `git diff --cached` is empty but `git diff` shows changes, tell the user to
`git add <files>` first. If still empty, run `git status` — nothing to verify.

If the diff exceeds 15,000 characters, split by file:
```bash
git diff --name-only
git diff HEAD -- specific_file.py
```

## Step 2 — Static security scan

Scan added lines only. Any match is a security concern fed into Step 5.

```bash
# Hardcoded secrets
git diff --cached | grep "^+" | grep -iE "(api_key|secret|password|token|passwd)\s*=\s*['\"][^'\"]{6,}['\"]"

# Shell injection
git diff --cached | grep "^+" | grep -E "os\.system\(|subprocess.*shell=True"

# Dangerous eval/exec
git diff --cached | grep "^+" | grep -E "\beval\(|\bexec\("

# Unsafe deserialization
git diff --cached | grep "^+" | grep -E "pickle\.loads?\("

# SQL injection (string formatting in queries)
git diff --cached | grep "^+" | grep -E "execute\(f\"|\.format\(.*SELECT|\.format\(.*INSERT"
```

## Step 3 — Baseline tests and linting

Detect the project language and run the appropriate tools. Capture the failure
count BEFORE your changes as **baseline_failures** (stash changes, run, pop).
Only NEW failures introduced by your changes block the commit.

**Test frameworks** (auto-detect by project files):
```bash
# Python (pytest)
python -m pytest --tb=no -q 2>&1 | tail -5

# Node (npm test)
npm test -- --passWithNoTests 2>&1 | tail -5

# Rust
cargo test 2>&1 | tail -5

# Go
go test ./... 2>&1 | tail -5
```

**Linting and type checking** (run only if installed):
```bash
# Python
which ruff && ruff check . 2>&1 | tail -10
which mypy && mypy . --ignore-missing-imports 2>&1 | tail -10

# Node
which npx && npx eslint . 2>&1 | tail -10
which npx && npx tsc --noEmit 2>&1 | tail -10
```

**Baseline comparison:** If baseline was clean and your changes introduce
failures, that's a regression. If baseline already had failures, only count
NEW ones.

## Step 4 — Self-review checklist

Quick scan before the fresh-eyes review:

- [ ] No hardcoded secrets, API keys, or credentials
- [ ] Input validation on user-provided data
- [ ] SQL queries use parameterized statements
- [ ] File operations validate paths (no traversal)
- [ ] External calls have error handling (try/catch)
- [ ] No debug print/console.log left behind
- [ ] No commented-out code
- [ ] New code has tests (if test suite exists)

## Step 5 — Fresh-eyes review

Agent4ML has no subagent delegation, so the "independent reviewer" is you with a
deliberate context reset. Treat the diff as if someone else wrote it — read it
cold, without remembering your intent.

Re-read the diff and evaluate against these categories. Fail-closed: if you
can't fully trace a code path, mark it failed.

**SECURITY (auto-FAIL):** hardcoded secrets, backdoors, data exfiltration,
shell injection, SQL injection, path traversal, eval()/exec() with user input,
pickle.loads(), obfuscated commands.

**LOGIC ERRORS (auto-FAIL):** wrong conditional logic, missing error handling
for I/O/network/DB, off-by-one errors, race conditions, code contradicts intent.

**SUGGESTIONS (non-blocking):** missing tests, style, performance, naming.

Return a verdict:
```
VERDICT: PASS | FAIL

Security issues: [list from static scan + review]
Logic errors: [list from review]
Regressions: [new test failures vs baseline]
New lint errors: [details]
Suggestions (non-blocking): [list]
```

**All passed:** Proceed to Step 7 (commit).

**Any failures:** Report what failed, then proceed to Step 6 (auto-fix).

## Step 6 — Auto-fix loop

**Maximum 2 fix-and-reverify cycles.**

Fix ONLY the reported issues — do NOT refactor, rename, or change anything
else. Do NOT add features.

After fixing, re-run Steps 1-5 (full verification cycle).
- Passed: proceed to Step 7
- Failed and attempts < 2: repeat Step 6
- Failed after 2 attempts: escalate to user with the remaining issues and
  suggest `git stash` or `git reset` to undo

## Step 7 — Commit

If verification passed:

```bash
git add -A && git commit -m "[verified] <description>"
```

The `[verified]` prefix indicates the fresh-eyes review passed.

## Reference: Common Patterns to Flag

### Python
```python
# Bad: SQL injection
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
# Good: parameterized
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

# Bad: shell injection
os.system(f"ls {user_input}")
# Good: safe subprocess
subprocess.run(["ls", user_input], check=True)
```

### JavaScript
```javascript
// Bad: XSS
element.innerHTML = userInput;
// Good: safe
element.textContent = userInput;
```

## Pitfalls

- **Empty diff** — check `git status`, tell user nothing to verify
- **Not a git repo** — skip and tell user
- **Large diff (>15k chars)** — split by file, review each separately
- **False positives** — if review flags something intentional, note it before fixing
- **No test framework found** — skip regression check, verdict still runs
- **Lint tools not installed** — skip that check silently, don't fail
- **Auto-fix introduces new issues** — counts as a new failure, cycle continues
