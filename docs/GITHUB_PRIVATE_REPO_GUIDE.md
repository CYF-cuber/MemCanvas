# GitHub repository guide

## Suggested repository names

- `MemCanvas`
- `memcanvas`
- `memcanvas-research`

## Local initialization

Run from the repository root:

```bash
cd /home/cyf/MemCanvas
git init
git branch -M main
git add .
git commit -m "init: bootstrap MemCanvas public release"
```

## Option 1: GitHub CLI

If `gh` is already authenticated:

```bash
gh repo create MemCanvas --public --source=. --remote=origin --push
```

Use `--private` instead of `--public` if the release should remain private during preparation.

## Option 2: Manual GitHub setup

Create an empty repository on GitHub, then run:

```bash
git remote add origin git@github.com:<your-user-or-org>/MemCanvas.git
git push -u origin main
```

For HTTPS:

```bash
git remote add origin https://github.com/<your-user-or-org>/MemCanvas.git
git push -u origin main
```

## Pre-upload checklist

Check the staged files before pushing:

```bash
git status
git diff --cached --stat
```

Confirm that you are not committing:

- datasets
- checkpoints
- real API keys
- local caches
- temporary logs
- large generated figures or model files
- fine-tuning/RL/SFT artifacts
