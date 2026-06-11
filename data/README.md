# Data

This directory is the repository entry point for small data metadata and examples. It should not store large raw datasets, model weights, or generated evaluation artifacts.

Recommended convention:

- `data/raw/`: local raw datasets, not committed
- `data/cache/`: local caches, not committed
- `data/examples/`: small runnable examples, committed
- `data/classifications/`: released taxonomy labels, committed

The public repository is intended to manage source code, metadata, and tiny examples, not to vendor the full `memcanvas0402` or `codex` workspaces.
