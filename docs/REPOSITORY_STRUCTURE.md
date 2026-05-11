# 仓库结构说明

## 主目录职责

- `memcanvas/`
当前主代码包。未来如果要继续工程化、模块化，这里是主战场。

- `evaluation/`
实验评测入口。保留了多个 benchmark 的研究脚本。

- `training/`
训练、数据准备、LoRA 合并等脚本。

- `scripts/`
辅助脚本和论文示例生成脚本。

- `paper/`
论文源码与图，便于和代码同仓维护。

- `versions/`
历史代码快照区。

- `docs/`
中文说明文档区，面向你自己后续管理这个私有仓。

- `configs/`
未来统一配置文件的归档位置。

- `data/`
未来数据说明和小样例的入口位置。

## `versions/` 目录设计

### `v1_workspace_memcanvas0402`

来源：`/home/cyf/memcanvas0402`

保留内容：

- 根目录下的主要 Python 和 shell 脚本

不保留内容：

- 大型实验结果
- 图片/数据目录
- 论文材料目录

### `v2_workspace_codex`

来源：`/home/cyf/codex`

保留内容：

- 根目录下的主要 Python 和 shell 脚本
- `layout_agent/` 代码
- `layout_scorer/` 代码

不保留内容：

- 数据集
- checkpoints
- outputs
- 第三方 baseline 仓

## 管理原则

- 主代码放根仓可维护目录。
- 历史代码放 `versions/`。
- 数据和结果不直接进 git。
- 需要复现时，通过文档回到原始大目录，而不是把 256G 工作区强行塞进仓库。
