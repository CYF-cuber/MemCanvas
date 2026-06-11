# Repository structure

## Public release directories

- `memcanvas/`: core training-free MemCanvas package.
- `scripts/`: portable command-line tools for canvas construction, embeddings, retrieval, ablations, taxonomy attachment, and plotting.
- `configs/`: model/API/dataset configuration templates.
- `docs/`: user-facing documentation.
- `data/classifications/`: released new taxonomy labels.
- `data/examples/`: tiny examples for smoke tests.
- `reports/`: category-level metric and split-count reports.

## Historical reference directories

- `versions/`: historical code snapshots retained for provenance. These are not guaranteed to be portable or aligned with the final paper wording.
- `evaluation/`: older evaluation scripts may remain for internal reference, but the portable public evaluation entry point is `scripts/evaluate.py`.
- `paper/`: paper source and assets when present.

## Excluded from the public method path

Training, SFT, RL, LoRA merge scripts, model checkpoints, generated outputs, datasets, embeddings, and local caches are not part of the training-free MemCanvas release.
