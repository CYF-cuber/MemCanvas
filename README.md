# MemCanvas

MemCanvas 是一个围绕“视觉记忆画布”整理后的私有代码仓库，目标是把你分散在多个工作区里的核心代码、研究脚本和历史版本放到一个便于 GitHub 管理的地方。

这个仓库现在分成三层：

1. `memcanvas/`
当前整理后的核心包，放画布渲染、记忆系统、编码器和布局相关代码。

2. `evaluation/` `training/` `scripts/`
当前研究脚本区，保留了实验入口和训练脚本，但其中一部分仍然带有你本机历史路径假设，适合作为内部研究仓管理，不适合作为“开箱即用”的公共仓承诺。

3. `versions/`
从旧工作区抽取的代码快照，只保留源码，不带大数据、模型、输出结果，用于回溯版本演进。

## 目录说明

```text
MemCanvas/
├── memcanvas/                    # 当前主代码
├── evaluation/                   # 当前评测脚本
├── training/                     # 当前训练脚本
├── scripts/                      # 辅助脚本
├── paper/                        # 论文源码与图
├── versions/                     # 历史代码快照
│   ├── v1_workspace_memcanvas0402/
│   └── v2_workspace_codex/
├── docs/                         # 中文管理文档
├── configs/                      # 预留配置目录
├── data/                         # 预留数据目录
├── requirements-core.txt         # 核心依赖
├── requirements-research.txt     # 研究脚本依赖
└── pyproject.toml                # 基础包元信息
```

## 当前版本定位

- `memcanvas/` 是主维护区，后续功能修改优先放这里。
- `versions/v1_workspace_memcanvas0402/` 是较早期的实验工作区脚本快照。
- `versions/v2_workspace_codex/` 是后续更大规模实验工作区的代码快照。
- 大型数据集、模型权重、评测输出和检查点没有并入仓库，避免私有仓体积失控。

## 推荐使用方式

安装核心依赖：

```bash
pip install -r requirements-core.txt
```

如果需要跑研究脚本，再安装扩展依赖：

```bash
pip install -r requirements-research.txt
```

把仓库根目录加入 `PYTHONPATH` 后再运行当前主代码：

```bash
export PYTHONPATH=/home/cyf/MemCanvas:$PYTHONPATH
```

## 文档入口

- `docs/PROJECT_OVERVIEW.md`
- `docs/REPOSITORY_STRUCTURE.md`
- `docs/VERSION_MANAGEMENT.md`
- `docs/GITHUB_PRIVATE_REPO_GUIDE.md`
- `docs/KNOWN_ISSUES.md`
- `versions/README.md`

## 当前整理原则

- 主代码和历史快照分开管理。
- 只把“值得版本管理的代码和文档”放进仓库。
- 不把 `codex` 和 `memcanvas0402` 里的大体积结果目录直接上传。
- 未来每次大版本改动都保留一个 `versions/` 快照，并同时打 git tag。

## 许可证

当前仓库按私有内部研究仓准备，暂未声明公开许可证。
