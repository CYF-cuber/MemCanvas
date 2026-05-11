# GitHub 私有仓上传指南

## 推荐仓库名

可以使用下面其中一个：

- `MemCanvas`
- `memcanvas-private`
- `memcanvas-research`

## 本地初始化

在仓库根目录执行：

```bash
cd /home/cyf/MemCanvas
git init
git branch -M main
git add .
git commit -m "init: bootstrap private MemCanvas repository"
```

## 方式一：如果本机已登录 `gh`

```bash
gh repo create MemCanvas --private --source=. --remote=origin --push
```

优点：

- 直接创建私有仓
- 自动配置远程
- 自动首推送

## 方式二：网页手动建仓

先在 GitHub 网页创建一个空的 private repo，然后执行：

```bash
git remote add origin git@github.com:<你的用户名>/MemCanvas.git
git push -u origin main
```

如果你用 HTTPS：

```bash
git remote add origin https://github.com/<你的用户名>/MemCanvas.git
git push -u origin main
```

## 后续版本管理建议

每次大的结构变化：

```bash
git add .
git commit -m "archive: add next workspace snapshot"
git tag v3-next-milestone
git push origin main --tags
```

## 上传前最后检查

建议先看一次：

```bash
git status
git diff --cached --stat
```

重点确认没有误提交：

- 数据集
- checkpoints
- 本地缓存
- 临时日志
- 大体积输出图或模型
