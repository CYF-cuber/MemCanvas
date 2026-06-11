# Version management

## Current release

The repository root is the current public release. The main maintained directories are:

- `memcanvas/`
- `scripts/`
- `configs/`
- `docs/`
- `data/classifications/`
- `reports/`

## When a git commit is enough

Use a normal commit for:

- small bug fixes
- README or documentation updates
- single-script parameter changes
- small features that do not change the repository structure

## When to create a `versions/` snapshot

Create a historical snapshot only when you need to preserve an older experimental workspace for provenance, for example:

- a major repository reorganization
- migration from a large external workspace
- an independent experimental pipeline that should remain auditable

## Naming convention

```text
versions/v<index>_<description>
```

Examples:

- `versions/v1_workspace_memcanvas0402`
- `versions/v2_workspace_codex`
- `versions/v3_public_release`

## Suggested git tags

```bash
git tag v1-workspace-memcanvas0402
git tag v2-workspace-codex
git tag v3-public-release
```

## Suggested commit messages

```text
init: bootstrap MemCanvas repository
docs: add public release documentation
refactor: normalize package imports in memcanvas core
archive: add historical workspace snapshot
```

## Do not commit

- model weights
- fine-tuning, RL, or generated training artifacts
- data caches
- local temporary logs
- large evaluation outputs

Keep these artifacts in local storage or object storage rather than git.
