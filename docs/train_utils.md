# `bmpt.train_utils` 函数说明

本文档解释 `bmpt.algorithms.def_train` 常用辅助函数的职责、输入输出和典型用法。

## 配置缓存相关

### `reset_config_cache() -> None`

- 作用：清空全局配置缓存。
- 使用场景：测试或切换配置文件后，强制下次重新从磁盘读取。

### `load_config_cached(config_path) -> dict[str, Any]`

- 作用：按路径读取 YAML，并做“同一路径复用缓存”。
- 输入：`config_path`（字符串或 `Path`）。
- 输出：配置字典。
- 说明：若路径与缓存命中，直接返回缓存对象。

### `load_config(config_path, default_config_path) -> dict[str, Any]`

- 作用：提供统一的“可选外部路径 + 模块默认路径”配置读取入口。
- 规则：
  - `config_path` 非空时读取该路径。
  - `config_path` 为空时回退 `default_config_path`。
- 输出：配置字典（内部复用 `load_config_cached`）。

### `resolve_config_path(input_payload, default_config_path) -> str | Path`

- 作用：解析本次 step 使用的配置路径。
- 规则：优先 `input_payload["config_path"]`，否则回退 `default_config_path`。

### `get_cached_top_level(config_path) -> dict[str, Any]`

- 作用：获取指定路径对应的“缓存中的原始配置”。
- 说明：等价于调用 `load_config_cached`，用于语义化表达“我要原始顶层配置”。

### `resolve_step_config(input_payload, default_config_path) -> dict[str, Any]`

- 作用：生成当前 step 生效配置。
- 规则：`deep_merge_dict(base_config, input_payload.get("config", {}))`。
- 输出：已叠加 step 级覆盖的配置字典。

## 模型装配相关

### `normalize_models(models) -> dict[str, torch.nn.Module]`

- 作用：统一模型输入格式。
- 规则：若传入单个模型，转成 `{"policy": model}`；若已是字典则原样返回。

### `expected_model_keys(config) -> list[str]`

- 作用：读取配置中声明的模型 key 列表。
- 来源：`config["models"].keys()`。

### `validate_models_by_config(models, config) -> None`

- 作用：预留的模型校验钩子。
- 当前状态：空实现（不做实际校验）。

### `default_model_loader(model_label, model_spec, _config) -> torch.nn.Module`

- 作用：默认模型加载器占位。
- 当前行为：直接抛 `NotImplementedError`，提示你传入 `loader_fn` 或提前构建 `models`。

### `build_models_from_config(config, loader_fn=default_model_loader) -> dict[str, torch.nn.Module]`

- 作用：根据 `config.models` 声明，逐个调用 `loader_fn` 构建模型字典。
- 输出：`{key: model}`，例如 `{"policy": ..., "reference": ...}`。

### `resolve_models(models, merged_config, input_payload) -> dict[str, torch.nn.Module]`

- 作用：确定本次 step 使用的模型字典。
- 规则：
  - `models is None`：从配置构建，使用 `input_payload.get("loader_fn")`。
  - 否则：走 `normalize_models`，再调用 `validate_models_by_config`。

## 回调与上下文相关

### `resolve_callbacks(input_payload, default_forward, default_reward) -> tuple[Callable, Callable]`

- 作用：解析 `forward_fn` 与 `reward_fn` 回调。
- 说明：这是兼容型工具；在“`step` 内联实现 forward/reward/loss”模式下可不使用。

### `resolve_reward_fns(input_payload, reward_fn) -> dict[str, Callable[..., torch.Tensor]]`

- 作用：解析多奖励函数映射。
- 规则：优先 `input_payload["reward_fns"]`，否则返回 `{"reward": reward_fn}`。
- 说明：用于 weighted / 多奖励策略场景。

### `resolve_global_step(input_payload) -> int`

- 作用：读取并规范化 `global_step`。
- 规则：`int(input_payload.get("global_step", 0))`。

### `build_step_context(global_step, merged_config, cached_config) -> StepContext`

- 作用：构建 `StepContext`，供 `def_train.py` 中算法访问。
- 字段：
  - `global_step`
  - `runtime_config`（来自 `merged_config["runtime"]`）
  - `full_config`（当前 step 生效配置）
  - `cached_config`（缓存中的原始配置）

## 推荐调用顺序（`bmpt.algorithms.def_train`）

典型流程：

1. `config_path = resolve_config_path(...)`
2. `cached = get_cached_top_level(config_path)`
3. `merged = resolve_step_config(...)`
4. `global_step = resolve_global_step(...)`
5. `ctx = build_step_context(...)`
6. `model_dict = resolve_models(models, merged, input)`
7. 在 `step` 内完成 forward / reward / loss 计算并返回

## Composer 工具（`bmpt.util.composer`）

当你需要在 `step` 内组合多段 token 序列时，使用：

- `bmpt.util.Composer`
- `bmpt.util.build_composers_from_config`

主程序会在启动阶段读取 `config.prompting.composers`，预 tokenize prompts，并通过 `input["composers"]` 传入 `step`。

典型组合形式：

- `[prompt] + [model1 out] + [prompt] + [model2 out] + ...`

`Composer.compose(...)` 输出：

- `input_ids`
- `attention_mask`
- `lengths`

补充说明：

- 上述输出可直接喂给下一个模型的前向调用。
- 若用于监督训练，需在 `step` 内基于业务策略构造 `labels`（例如将不参与监督位置设为 `-100`）。
