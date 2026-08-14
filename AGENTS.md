# AGENTS.md

## Agent skills

### Issue tracker

Issues and PRDs live in this repo's GitHub Issues, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles map 1:1 to their default label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Engineering constraints

These override default agent behaviour. They exist because this codebase drifted toward
hand-rolled helpers and mock-based tests that verified nothing.

### 1. Ask before implementing

Any change that adds a dependency, replaces a component, or changes an approach is
**proposed first** — pros, cons, line-count delta, and what it deletes — and coded only
after the user agrees. Never open with code.

### 2. Dependency priority

1. **A free framework already in `pyproject.toml`** — using more of what is already paid
   for is free.
2. **Tie: a new free framework, or the standard library** — whichever removes more code.
3. **A paid service** — only when there is no free equivalent and the saving is large.

Two hard rules:

- **Never adopt a second stack for a single feature.** A framework earns its place by
  carrying a module, not a function.
- **Read the framework, take the design, skip the dependency.** Studying a library to
  learn what it does well and reimplementing the idea in five lines is a *success*, not a
  shortcut.

**Frameworks live only at the LLM boundary** — `graph/`, `memory/session.py`, `eval/`. A
framework appearing in `ingest/`, `index/`, `promote/`, `retrieval/`, `dashboard/`, or
`demo/` is a signal the change is wrong.

### 3. Testing: live-first, adversary-only units

The default test strategy is **live and integration tests against the real system**. The
mock-heavy unit suite is being removed: a test whose fake returns `X` and then asserts `X`
verifies only that the test was written.

Write a unit test only when it is an **adversary test** — one that attacks an assumption
the rest of the system depends on. It does not confirm that a function returns what its
author intended; it hunts for the input that violates an invariant. Concretely:

- *"Truncation preserves the original characters"* → broken by `"don't stop"`.
- *"The cache key distinguishes any two different queries"* → broken by adding a field to
  `ParsedQuery`.
- *"Standing constraints survive a merge"* → broken by adding a field.
- *"Every candidate reaches the vote_count ranker"* → broken by RRF-score ties.

All four are real defects the previous 5,000-line suite did not catch. If a proposed test
cannot name the assumption it is trying to break, do not write it.

Prefer property-based tests (`hypothesis`) over parametrized example lists. Never write a
test that asserts a hand-maintained constant list contains its own contents.

### 4. Leanness

- Prefer the standard library over a bespoke helper; prefer deleting over refactoring.
- Every ticket names **what it deletes**, or states explicitly why nothing.
- Duplicated logic is a defect, not a style issue — one concept, one implementation.
- Ruff rule sets and a lock file are the mechanical enforcement; do not weaken them to
  make a change pass.

### 5. Observability and verification

- Tracing is **OpenTelemetry**, exported to a free or self-hosted backend. Do not adopt a
  vendor-proprietary trace protocol.
- Data a human may need to audit must be **inspectable without running code** — readable
  columns in the DB, eval datasets as files in the repo under version control.
- Evals are the tiebreaker. Model choice, embedding dimensions, reranking, and fusion
  weights are **configuration swept by the eval harness**, never settled by argument.
