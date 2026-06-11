# Evaluation

The paper evaluation follows a common pattern:

1. Convert historical/source interactions into canvas memories.
2. Compute image and text embeddings for each memory.
3. Encode test queries.
4. Retrieve top-K canvases with the hybrid visual-text key.
5. Provide the retrieved canvases as visual context to the VLM.
6. Score predictions with task-specific metrics.

## Metrics

- ScienceQA, OK-VQA, ChartQA: accuracy-style metrics after answer normalization.
- HotpotQA, MultiModalQA: exact match and token F1.

Metric helpers are implemented in `memcanvas.metrics`.

## Retrieval-only evaluation

`scripts/evaluate.py` currently evaluates the retrieval map from precomputed embeddings. This is the stable public entry point and avoids hard-coded local model paths.

```bash
python scripts/evaluate.py \
  --image-embeddings outputs/demo/embeddings/clip_img_emb.npy \
  --text-embeddings outputs/demo/embeddings/clip_txt_emb.npy \
  --query-embeddings outputs/demo/embeddings/clip_query_emb.npy \
  --alpha 0.75 \
  --top-k 2 \
  --output outputs/demo/retrieval.json
```

## Full VLM evaluation

Full VLM evaluation requires dataset-specific loaders and local models. The historical full scripts are preserved under `versions/` for reproducibility auditing, especially the SmartCanvas evaluation code derived from `eval_memcanvas0413.py`.

The public package exposes the reusable pieces needed to rebuild those experiments without absolute paths:

- `memcanvas.canvas`
- `memcanvas.retrieval`
- `memcanvas.prompts`
- `memcanvas.metrics`
- `memcanvas.forgetting`
