# Project overview

MemCanvas is organized here as a public, training-free research code release.

## Goal

The repository exposes the implementation needed to reproduce the paper method:

- construct visual memory canvases from multimodal interactions
- store and inspect canvas memories
- retrieve memories with hybrid visual-textual keys
- update memories through progressive visual forgetting
- reuse the released dataset taxonomy and evaluation prompts

## Public entry points

- `memcanvas/`: reusable Python package
- `scripts/`: portable command-line tools
- `configs/`: configuration templates
- `docs/`: method and usage documentation
- `data/classifications/`: released taxonomy labels
- `reports/`: merged category statistics and results

## Training-free scope

The paper method does not train model weights. Text compression, when used, is done by calling an off-the-shelf public LLM at inference time. SFT, RL, LoRA, and other training scripts are intentionally excluded from the public release path.

## Historical code

`versions/` may contain older snapshots for auditability only. They are not the recommended public API and may include local paths or obsolete experiment assumptions.
