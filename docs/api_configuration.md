# API configuration

MemCanvas can run fully locally with open-source models. API keys are only needed when you use external LLM/VLM providers for text compression, judging, or proprietary-model evaluation.

## Environment variables

Copy `.env.example` to `.env` and fill only the providers you need:

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=
ANTHROPIC_API_KEY=
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=
HF_TOKEN=
HF_HOME=
```

## YAML config

Copy `configs/api.example.yaml` to `configs/api.yaml`. The example contains placeholders for:

- OpenAI-compatible APIs
- Anthropic APIs
- DashScope/Qwen-compatible APIs
- Hugging Face token/cache configuration

Do not commit `configs/api.yaml` or `.env` with real credentials.

## Local model configuration

For local Qwen2.5-VL style evaluation, set the model path in `configs/default.yaml`:

```yaml
model:
  vlm_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
  clip_name_or_path: openai/clip-vit-large-patch14
```

The code also accepts local filesystem paths if you have downloaded weights.
