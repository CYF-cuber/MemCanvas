# 当前已知问题

## 1. 研究脚本仍有本机路径依赖

`evaluation/`、`training/`、`scripts/` 中部分脚本来自旧工作区，仍然写死了类似下面的路径：

- `/home/cyf/codex/...`
- `/home/cyf/memory/...`

这类脚本目前更适合“内部归档和继续手改”，不应在 README 中宣称为完全可移植。

## 2. 主包和旧脚本是两个层次

- `memcanvas/` 代表当前整理后的主代码
- 旧实验脚本代表研究过程快照

二者不应混为“完全同一套可复现环境”。

## 3. 大型数据与权重未并入仓库

这是有意设计，不是缺失。原因是：

- `memcanvas0402` 和 `codex` 体积过大
- 私有 git 仓不适合直接承载海量数据和检查点

## 4. 本地环境依赖尚未统一

当前环境中甚至可能缺少基础库（例如 `Pillow`）。因此仓库里补了：

- `requirements-core.txt`
- `requirements-research.txt`
- `pyproject.toml`

后续建议先固定一个 Python 环境，再逐步清理脚本。
