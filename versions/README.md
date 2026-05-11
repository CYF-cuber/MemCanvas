# 版本索引

## 当前主版本

仓库根目录就是当前主版本，重点维护目录为：

- `memcanvas/`
- `evaluation/`
- `training/`
- `scripts/`

## 历史快照

### `v1_workspace_memcanvas0402`

- 来源：`/home/cyf/memcanvas0402`
- 内容：早期实验工作区根目录脚本快照
- 目的：保留早期 pipeline、baseline、压缩与遗忘实验入口

### `v2_workspace_codex`

- 来源：`/home/cyf/codex`
- 内容：后续大规模研究工作区的代码快照
- 目的：保留更完整的实验脚本、布局 agent、layout scorer 代码

## 使用建议

- 看当前实现：先看仓库根目录主代码
- 找旧实验：去 `versions/`
- 要做新版本：复制当前主代码或新增 `versions/v3_*`
