# Installation

## Environment

MemCanvas is tested with Python 3.10+ and recent PyTorch/Transformers versions.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For editable local use without installation:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Optional GPU dependencies

The core canvas renderer only requires Pillow and NumPy. CLIP embedding and VLM evaluation require PyTorch, Transformers, and a CUDA-enabled environment for practical speed.

## Data and model paths

Use `configs/default.yaml` as the central local configuration file. Keep datasets, model weights, generated canvases, embeddings, and checkpoints out of git.
