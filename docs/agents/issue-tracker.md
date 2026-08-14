# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

Infer the repo from `git remote -v` — `gh` does this automatically when run inside a clone.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

## Wayfinding operations

How this repo expresses a wayfinder map, its child tickets, blocking, and the frontier.
GitHub has **native** sub-issues and issue dependencies, so use those — never a body
convention. Native relationships render the frontier visually in GitHub's own UI, which
is the point: a human sees what is takeable without opening the map.

### The map

One issue labelled `wayfinder:map`. Create it first — child tickets need its number.

```bash
gh issue create --title "<destination name>" --body-file map.md --label "wayfinder:map"
```

### Child tickets

Every ticket is a **sub-issue of the map**, created with `--parent`:

```bash
gh issue create --title "<question>" --body-file ticket.md \
  --parent <map-number> --label "wayfinder:grilling"
```

Type labels: `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`,
`wayfinder:task`. Create the four labels plus `wayfinder:map` once with `gh label create`.

### Blocking

Use GitHub's native dependencies. Wire them in a **second pass**, after the tickets exist
and have numbers:

```bash
gh issue edit <number> --add-blocked-by <number>     # this one waits on that one
gh issue edit <number> --add-blocking <number>       # the inverse
```

`gh issue create` also accepts `--blocked-by` / `--blocking` when the blocker already exists.

### Claiming

A session claims a ticket by assigning it **before any work**:

```bash
gh issue edit <number> --add-assignee @me
```

An open, unassigned child issue is unclaimed.

### The frontier

Open + unassigned + unblocked children of the map. List the candidates, then filter by
blocking state:

```bash
gh issue list --state open --label "wayfinder:grilling" --search "no:assignee" \
  --json number,title,labels
```

`gh issue list` cannot filter on dependency state, so confirm a candidate is unblocked by
reading its dependencies before claiming:

```bash
gh issue view <number> --json title,body,assignees,labels
```

Blocked-by relationships are also visible in the issue's web UI and on the map's
sub-issue list, which shows completion progress at a glance.

### Resolution

Post the answer as a comment, close the issue, then append a one-line pointer to the map's
**Decisions so far**:

```bash
gh issue comment <number> --body-file resolution.md
gh issue close <number>
gh issue edit <map-number> --body-file updated-map.md
```

Refer to maps and tickets **by title** in everything a human reads — never a bare number.
