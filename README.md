# MemCanvas

MemCanvas is a training-free visual memory framework for lifelong multimodal agents. It stores past multimodal interactions as structured canvas images, retrieves relevant canvases with hybrid visual-textual keys, and injects them as visual memory context for downstream vision-language models.

This repository is organized as an open-source release of the paper code. Historical experiment snapshots are kept only as references; the public entry points are the `memcanvas/`, `scripts/`, `configs/`, `docs/`, `data/classifications/`, and `reports/` directories.

## Features

- **Visual memory construction**: render text, images, charts, and tables into readable canvas memories.
- **Hybrid retrieval**: combine CLIP image and text embeddings with a tunable coefficient `alpha`.
- **Memory bank storage**: maintain canvas files and JSONL metadata with access counts and quality states.
- **Progressive visual forgetting**: degrade rarely accessed memories by resolution before deletion.
- **Dataset taxonomy**: release the topic and modality-hop labels used for merged evaluation.
- **Evaluation prompts**: provide the prompts used for ScienceQA, OK-VQA, MMQA, and HotpotQA-style evaluation.

## Repository layout

```text
MemCanvas/
├── memcanvas/                 # Core package
│   ├── canvas.py              # SmartCanvas layout and rendering
│   ├── bank.py                # MemoryEntry and MemoryBank
│   ├── retrieval.py           # CLIP embeddings and hybrid retrieval
│   ├── forgetting.py          # Resolution-based memory update
│   ├── prompts.py             # Evaluation/compression prompts
│   ├── metrics.py             # EM/F1/VQA metrics
│   └── api.py                 # API/env config helpers
├── scripts/                   # Command-line tools
├── configs/                   # Reproducible config templates
├── docs/                      # Method, setup, API, taxonomy, evaluation docs
├── data/classifications/      # Released new-category labels
├── reports/                   # Aggregated category reports
└── versions/                  # Historical snapshots, not public entry points
```

## Installation

```bash
git clone <your-repo-url> MemCanvas
cd MemCanvas
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For local development without installation:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## API and model configuration

No real API keys are committed. Copy the examples and fill in local values:

```bash
cp .env.example .env
cp configs/api.example.yaml configs/api.yaml
```

Supported configuration placeholders include OpenAI-compatible APIs, Anthropic APIs, DashScope/Qwen-compatible APIs, and Hugging Face tokens/cache paths. See `docs/api_configuration.md`.

## Quick start

### 1. Build canvas memories

Input records can be JSON or JSONL. Each record may contain fields such as `question`, `choices`, `answer`, `context`, `hint`, `lecture`, `image_path`, and `table`. In the paper setting, the memory bank is built from historical/source interactions; no model weights are trained.

```bash
python scripts/build_canvases.py \
  --input data/examples/sample_records.jsonl \
  --image-root data/examples/images \
  --output-dir outputs/demo/canvases
```

### 2. Build CLIP embeddings

```bash
python scripts/build_embeddings.py \
  --canvas-dir outputs/demo/canvases \
  --manifest outputs/demo/canvases/manifest.json \
  --output-dir outputs/demo/embeddings
```

### 3. Retrieve memories

```bash
python scripts/evaluate.py \
  --image-embeddings outputs/demo/embeddings/clip_img_emb.npy \
  --text-embeddings outputs/demo/embeddings/clip_txt_emb.npy \
  --query-embeddings outputs/demo/embeddings/clip_query_emb.npy \
  --alpha 0.75 \
  --top-k 2 \
  --output outputs/demo/retrieval.json
```

Full VLM evaluation requires local datasets, canvas banks, query embeddings, and a VLM such as Qwen2.5-VL. The original research scripts are preserved in `versions/` for auditability; new public scripts expose the reusable components.

## Text compression

When text needs to be shortened before rendering, MemCanvas uses an off-the-shelf public LLM through the prompt in `memcanvas/prompts.py`. This is inference-only compression; the project does not require SFT, RL, LoRA, or any other training stage.

## Dataset taxonomy

The released classification files are in `data/classifications/`:

- `topic_labels.txt`: dataset/split/index to major topic and subtopic.
- `modality_labels.txt`: dataset/split/index to modality and reasoning-hop type.

Aggregated results are in `reports/category_metrics/`. See `docs/dataset_taxonomy.md`.

## Citation

If you use this repository, please cite the MemCanvas paper. The BibTeX entry will be added after publication.

## License

A public license has not yet been selected. Choose a license before publishing this repository publicly.
