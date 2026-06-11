# MemCanvas memory mechanism

MemCanvas treats each prior multimodal interaction as one external visual memory item.

## 1. Memory construction

An interaction is normalized into typed blocks:

- text blocks: question, answer, context, captions, metadata
- image blocks: original or resized images
- table blocks: rendered row-column structure
- chart/document blocks: treated as image blocks when already rendered

`memcanvas.canvas` measures each block, generates candidate layouts, scores them by squareness, utilization, and size constraints, and renders the best layout as a PNG canvas.

## 2. Storage

`MemoryBank` stores:

- canvas files in `canvases/`
- JSONL metadata in `manifest.jsonl`
- text metadata for retrieval
- access count
- quality level
- deletion state

The storage design is intentionally simple so the memory bank is inspectable and easy to sync.

## 3. Retrieval

`memcanvas.retrieval` computes two embeddings per memory:

- canvas image embedding
- text metadata embedding

The hybrid key is:

```text
K_i = normalize(alpha * E_canvas_i + (1 - alpha) * E_text_i)
```

A query is encoded through the text encoder and retrieves the top-K memories by cosine similarity.

The default paper-like setting is:

```yaml
alpha: 0.75
top_k: 2
threshold: 0.1
```

## 4. Update and forgetting

Long-lived agents cannot keep all memories at full resolution forever. MemCanvas uses frequency-adaptive progressive visual forgetting:

```text
1.0x -> 0.75x -> 0.5x -> 0.25x -> deleted
```

At every review interval `T`, memories whose cumulative retrieval count is not greater than threshold `S` are degraded by one quality level. Frequently retrieved canvases stay sharp; stagnant canvases become smaller and eventually disappear.

The default setting is:

```yaml
review_interval: 1000
frequency_threshold: 0
```
