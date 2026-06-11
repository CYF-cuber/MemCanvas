# MemCanvas 中文说明

MemCanvas 是一个面向长期多模态智能体的视觉记忆框架。它将历史多模态交互渲染为结构化画布图像，通过视觉-文本混合检索找到相关记忆，并把检索到的画布作为视觉上下文输入给多模态模型。

本仓库按照论文开源版本重新整理。历史实验代码仅作为追溯参考；公开使用入口是 `memcanvas/`、`scripts/`、`configs/`、`docs/`、`data/classifications/` 和 `reports/`。

## 主要功能

- **视觉记忆构建**：将文本、图像、图表和表格渲染为可读 canvas。
- **混合检索**：使用 CLIP 图像/文本 embedding，并通过 `alpha` 控制视觉与文本权重。
- **记忆库存储**：用 canvas 文件和 JSONL manifest 管理记忆、访问次数和质量等级。
- **渐进视觉遗忘**：低频访问记忆先降低分辨率，再删除，控制长期存储规模。
- **数据集新分类**：提供论文合并评测使用的主题分类与模态/跳数分类标签。
- **测评提示词**：整理 ScienceQA、OK-VQA、MMQA、HotpotQA 风格评测提示词。

## 目录结构

```text
MemCanvas/
├── memcanvas/                 # 核心包
│   ├── canvas.py              # SmartCanvas 布局和渲染
│   ├── bank.py                # MemoryEntry / MemoryBank
│   ├── retrieval.py           # CLIP embedding 与混合检索
│   ├── forgetting.py          # 分辨率渐进遗忘机制
│   ├── prompts.py             # 测评与压缩提示词
│   ├── metrics.py             # EM/F1/VQA 指标
│   └── api.py                 # API 和环境变量配置辅助
├── scripts/                   # 命令行工具
├── configs/                   # 可复现实验配置模板
├── docs/                      # 方法、安装、API、分类、测评文档
├── data/classifications/      # 新分类标签
├── reports/                   # 分类统计与结果报告
└── versions/                  # 历史快照，不作为公开主入口
```

## 安装

```bash
git clone <your-repo-url> MemCanvas
cd MemCanvas
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

如果不安装包，也可以设置：

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## API 与模型配置

仓库不包含任何真实密钥。请复制模板后自行填写：

```bash
cp .env.example .env
cp configs/api.example.yaml configs/api.yaml
```

当前模板覆盖 OpenAI-compatible API、Anthropic API、DashScope/Qwen-compatible API、Hugging Face token 和缓存路径。详见 `docs/api_configuration.md`。

## 快速开始

### 1. 构建 canvas 记忆

输入支持 JSON 或 JSONL。每条记录可包含 `question`、`choices`、`answer`、`context`、`hint`、`lecture`、`image_path`、`table` 等字段。论文设定中，记忆库来自历史/源交互样本；不会训练模型权重。

```bash
python scripts/build_canvases.py \
  --input data/examples/sample_records.jsonl \
  --image-root data/examples/images \
  --output-dir outputs/demo/canvases
```

### 2. 构建 CLIP embedding

```bash
python scripts/build_embeddings.py \
  --canvas-dir outputs/demo/canvases \
  --manifest outputs/demo/canvases/manifest.json \
  --output-dir outputs/demo/embeddings
```

### 3. 运行混合检索

```bash
python scripts/evaluate.py \
  --image-embeddings outputs/demo/embeddings/clip_img_emb.npy \
  --text-embeddings outputs/demo/embeddings/clip_txt_emb.npy \
  --query-embeddings outputs/demo/embeddings/clip_query_emb.npy \
  --alpha 0.75 \
  --top-k 2 \
  --output outputs/demo/retrieval.json
```

完整 VLM 测评需要本地数据集、canvas bank、query embedding 和 Qwen2.5-VL 等多模态模型。历史研究脚本保留在 `versions/` 中用于追溯，新公开脚本优先提供可复用组件。

## 文本压缩

如果渲染前需要压缩文本，MemCanvas 使用公开原始大模型按照 `memcanvas/prompts.py` 中的提示词进行推理式压缩。这不是训练流程，不需要 SFT、RL、LoRA 或任何参数更新。

## 数据集新分类

分类标签位于 `data/classifications/`：

- `topic_labels.txt`：dataset/split/index 到主题大类和小类。
- `modality_labels.txt`：dataset/split/index 到模态组合和单跳/多跳类型。

合并统计结果位于 `reports/category_metrics/`。详见 `docs/dataset_taxonomy.md`。

## 引用

如果使用本仓库，请引用 MemCanvas 论文。正式 BibTeX 将在论文发表后补充。

## 许可证

当前尚未选择公开许可证。正式公开前请补充 LICENSE。
