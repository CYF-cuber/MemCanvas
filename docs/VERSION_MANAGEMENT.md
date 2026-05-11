# 版本管理规则

## 当前版本定义

- 仓库根目录代表“当前整理后的主版本”。
- `versions/` 代表“历史代码快照版本”。

## 何时只用 git commit

以下情况只需要正常提交：

- 小修复
- README 更新
- 单个脚本参数调整
- 不改变目录结构的小功能开发

## 何时创建新的 `versions/` 文件夹

以下情况建议新建一个版本快照目录：

- 代码组织方式发生明显变化
- 从一个大工作区切换到另一个大工作区
- 新增一组独立实验管线
- 需要保留“当时那套脚本”的完整上下文

## 目录命名建议

统一格式：

```text
versions/v<序号>_<版本说明>
```

示例：

- `versions/v1_workspace_memcanvas0402`
- `versions/v2_workspace_codex`
- `versions/v3_multimodal_refactor`

## git tag 建议

每次形成可识别的大版本时，同时打 tag：

```bash
git tag v1-workspace-memcanvas0402
git tag v2-workspace-codex
git tag v3-clean-main-repo
```

## 提交信息建议

建议使用下面的风格：

```text
init: bootstrap private MemCanvas repository
docs: add repository structure and github guide
refactor: normalize package imports in memcanvas core
archive: add memcanvas0402 and codex code snapshots
```

## 不建议提交的内容

- 模型权重
- 训练输出
- 数据缓存
- 本地临时日志
- 评测结果大文件

这些内容应该留在本地大目录或对象存储，不进入私有 git 仓。
