# Configs

This directory contains portable configuration templates for the training-free MemCanvas release.

Recommended files:

- `default.yaml`: shared model, retrieval, canvas, and forgetting settings.
- `api.example.yaml`: API/environment placeholders for optional external LLM/VLM calls.
- `{scienceqa,okvqa,mmqa,hotpotqa,chartqa}.yaml`: dataset-specific path and metric templates.

The paper method is training-free. Historical training/RL/SFT scripts are not part of this public release.
