# Known issues

## Historical scripts may contain local paths

Some files under `versions/` or older `evaluation/` directories are preserved for provenance and may contain paths such as:

- `/home/cyf/codex/...`
- `/home/cyf/memory/...`

They should not be presented as the portable public API.

## Public method is training-free

Training, SFT, RL, and LoRA-related code is intentionally excluded from the public release path because the final paper describes MemCanvas as training-free. Text compression uses an off-the-shelf public LLM at inference time.

## Large artifacts are not included

Datasets, generated canvases, embeddings, checkpoints, and model weights are intentionally excluded from git. Use the config templates to point to local copies.
