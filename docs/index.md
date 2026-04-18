# 文档索引

当前项目遵循一个核心原则：

- 用户主要修改 `bmpt.algorithms.def_train` 与配置文件。

## 推荐阅读顺序

1. `docs/cli.md`：`bmpt-train` 参数与优先级
2. `docs/guide.md`：从零上手训练流程
3. `docs/config.md`：训练配置字段说明
4. `docs/interface.md`：`step(models, input)` 返回协议
5. `docs/apptainer.md`：Apptainer 构建与运行
6. `docs/train_utils.md`：`bmpt.train_utils` 函数说明

## 关键约定

- 算法由 `bmpt.algorithms.def_train` 内部的 `step` 决定。
- 框架层不通过 `mode` 或 `train.algorithm` 选择算法。
- `models` 节点只负责按 key 加载模型并传入 `step`。

## AI Agent 文档

- AI 专用文档统一位于 `agents/`。
- 入口：`agents/README.md`
