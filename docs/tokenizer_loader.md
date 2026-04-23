# `bmpt.tokenizer.loader` 函数级文档

## `_require_hf()`
- 校验并导入 `transformers.AutoTokenizer`。

## `resolve_tokenizer_source(config, local_source=None)`
- 解析 tokenizer 来源。
- 优先级：
  1. `local_source`
  2. 顶层 `tokenizer_source`
  3. `models.policy.path`
- `local_source/tokenizer_source` 若为 `models` 的键名，则自动映射到对应模型的 `path`。

## `load_tokenizer(config, local_source=None)`
- 加载 tokenizer，并在缺失 `pad_token` 时回退到 `eos_token`。
