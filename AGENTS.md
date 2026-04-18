# BMPT Agent 执行约束

本文件定义 AI Agent 在 BMPT 仓库内执行任务时的边界与约定。

## 1. 修改边界

- 优先修改 `src/bmpt/`。
- 示例代码在 `example/`，允许更新导入与运行方式。
- 文档修改优先放在 `docs/` 与 `agents/`。
- 不要随意改动与任务无关的数据文件（`data/processed/*`、`data/raw/*`）。

## 2. 算法与配置边界

- 默认算法入口是 `bmpt.algorithms.def_train`。
- 默认 `step` 是 SFT；若修改算法，需要保持 `step` 返回契约。
- DeepSpeed 配置路径来源于 `runtime.deepspeed_config_path`。

## 3. CLI 约束

- 训练主入口是 `bmpt-train`。
- `--workspace` 仅用于自动发现 `def_train.py` 与 `config.json/yaml/yml`。
- 参数优先级遵循 `docs/cli.md` 与 `agents/ai_quickstart.md`。

## 4. 质量要求

- 优先做最小、聚焦修改，避免大范围重构。
- 修改后至少做语法检查（如 `python3 -m compileall`）。
- 若运行时依赖缺失，应明确说明无法执行的验证项。

## 5. 文档同步

- 变更 CLI、配置优先级、接口契约时，必须同步更新：
  - `docs/cli.md`
  - `docs/interface.md`
  - `agents/` 下对应文档
