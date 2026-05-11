# 项目概览

## 目标

这个仓库用于集中管理 MemCanvas 相关代码，解决原先代码散落在多个目录、版本关系不清、难以上传 GitHub 私有仓的问题。

## 当前整理结果

- 以 `/home/cyf/MemCanvas` 作为主仓目录。
- 保留 `memcanvas/` 作为当前主代码区。
- 保留 `evaluation/`、`training/`、`scripts/` 作为当前研究脚本区。
- 从 `/home/cyf/memcanvas0402` 和 `/home/cyf/codex` 抽取代码快照到 `versions/`。
- 不并入大型数据集、模型权重、检查点和评测输出。

## 为什么这样整理

- `memcanvas0402` 大约 3.5G，混有实验数据、论文材料和脚本。
- `codex` 大约 256G，包含大量输出、缓存、数据和第三方目录，不能直接当 GitHub 仓库上传。
- `/home/cyf/MemCanvas` 体积小、结构清晰，适合作为主仓。

## 仓库内三类内容

### 1. 主代码

放在 `memcanvas/`，适合作为后续继续维护和逐步清理的核心实现。

### 2. 当前研究脚本

放在 `evaluation/`、`training/`、`scripts/`。这些脚本保留了研究过程，但部分仍依赖本机历史目录和模型路径。

### 3. 历史版本快照

放在 `versions/`，用于追溯演进，不要求即刻完全可运行，但要求能看出当时的脚本组织。

## 后续建议

- 新功能优先写到 `memcanvas/`。
- 新实验入口优先放到 `evaluation/` 或 `training/`，不要再散落到根目录。
- 每次出现“结构性版本变化”时，同时做两件事：
  1. 新建一个 `versions/` 快照目录。
  2. 在 git 中打 tag。
