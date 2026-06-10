---
name: pre-commit-checklist
description: >
  Run before every git commit in llm-internal-medicine. Gates a commit on:
  (1) the full pre-commit / ruff suite passing, (2) the unit tests passing, and
  (3) a check of whether README.md needs updating. Use whenever you are about to
  commit, are asked to commit/stage changes, or are wrapping up a change set and
  preparing it for review.
---

# Pre-Commit Checklist

> Run this gate before creating any commit. All three steps must pass /
> be considered. Do not commit on a red suite or a stale README.

## Step 1 — pre-commit / ruff must pass

The repo uses `pre-commit` (ruff + ruff-format, pinned in
`.pre-commit-config.yaml`). `pre-commit` builds an isolated env for the pinned
ruff version, so a locally-missing `ruff` module is fine — but the first run
needs network.

**Network is behind a proxy.** Export it before running:

```bash
export http_proxy="http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128"
export https_proxy="http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128"
pre-commit run --all-files
```

- `ruff-format` **modifies files** when it reformats and reports `Failed` on
  that run. That is expected — re-run until both hooks report `Passed`, and
  include the reformatted files in the commit.
- If only specific files changed, `pre-commit run --files <paths>` is faster,
  but run `--all-files` at least once before a commit that touches many files.

## Step 2 — unit tests must pass

Use the project's configured interpreter (venv/conda if present, else system
`python`).

```bash
python -m pytest tests/test_core_monitoring.py \
                 tests/test_megatron_monitors.py \
                 tests/test_paddlefleet_monitors.py -q
```

- All three suites must pass. The torch `pynvml` FutureWarning is benign.
- Backend-specific suites self-skip when their backend (torch / paddle) is not
  installed; do not treat a skip as a pass for that backend if you intended to
  exercise it.
- If you changed a hot path, also re-confirm the relevant assertions in
  `monitor-hook-perf-rules` (no new `.item()` / collective / lazy declare).

### Tests must track the change

A green suite is not enough if the suite no longer matches the code.

- **New feature / new metric / new behavior → add a test for it.** Don't commit
  a new `record_*` key, `setup_*` arg, or aggregation path with no coverage.
- **Refactor or removed code → delete or update the tests that exercised it.**
  Tests pinning state that no longer exists (e.g. an old `_finalize_*` helper or
  a per-record global-count counter) must be removed, not left to rot. Tests
  asserting old behavior must be rewritten to the new behavior.
- **Branchless / sync-avoidance rewrites → pin the result against a readable
  reference.** When a hot-path computation is rewritten to avoid a `.item()`
  (e.g. `compute_sink_head_classification`), add a test that compares the new
  output to a simple branched reference across edge cases, so nobody "simplifies"
  it back into a sync.

## Step 3 — does README.md need updating?

`README.md` documents each monitor and carries a **metric cheat-sheet**
(完整指标速查表) plus per-monitor metric lists and 日志键命名规则. It goes stale
silently. Before committing, ask:

- Did you **add / rename / remove a metric** (any `record_*` key, `declare_*`
  name, or `*_METRICS` tuple)? → update the monitor's metric list and the
  cheat-sheet table.
- Did you **add / remove a monitor** or change its `setup_*` signature / public
  config args (通用配置参数)? → update the relevant section and 快速开始 examples.
- Did you change the **log-key naming scheme** (`layer_N/`, `global_`)? → update
  日志键命名规则.
- Did you change **cross-rank aggregation** or health thresholds (健康阈值)? →
  update the corresponding prose.

If none apply, the README is fine — say so explicitly rather than skipping the
check.

## Only after all three

Stage the intended files (prefer naming them over `git add -A`), then commit.
Never `--no-verify`. If a hook keeps failing, fix the root cause rather than
bypassing it.
