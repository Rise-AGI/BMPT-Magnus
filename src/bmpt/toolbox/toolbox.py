from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from bmpt.manager.manager import Manager
from bmpt.model.loader import load_model
from bmpt.tokenizer.loader import load_tokenizer
from bmpt.toolbox.tokenize import tokenize_batch as _tokenize_batch_impl


class ToolBox:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager
        self.models: dict[str, torch.nn.Module] = {}
        self.tokenizers: dict[str, Any] = {}
        self.engine: Any | None = None
        self.optimizer: Any | None = None
        self.scheduler: Any | None = None

    def tokenize_batch(
        self,
        name: str,
        inputs: list[str] | torch.Tensor,
        padding_token: int = -1,
        max_length: int | None = None,
        truncation: bool = True,
    ) -> dict[str, torch.Tensor]:
        """使用指定名称的 tokenizer 对输入进行批处理 tokenize。

        Args:
            name: tokenizer 名称，对应 config.models 中的键
            inputs: 输入文本列表或已编码的 tensor
            padding_token: 填充 token ID，-1 表示使用 tokenizer 自带的 padding
            max_length: 最大序列长度
            truncation: 是否截断超长序列

        Returns:
            包含 input_ids 和 attention_mask 的字典
        """
        if name not in self.tokenizers:
            raise ValueError(f"Tokenizer '{name}' not loaded. Call load_models(...) first")

        tokenizer = self.tokenizers[name]
        return _tokenize_batch_impl(
            tokenizer=tokenizer,
            inputs=inputs,
            padding_token=padding_token,
            max_length=max_length,
            truncation=truncation,
        )

    def load_models(self, model_name: str) -> dict[str, torch.nn.Module]:
        if not self.manager.config:
            raise ValueError("Manager config is empty. Call Manager.load_config(...) first")

        model_cfg = self.manager.config.get("models", {})
        if not isinstance(model_cfg, dict) or not model_cfg:
            raise ValueError("config.models is required")

        if model_name not in model_cfg:
            raise ValueError(f"Model '{model_name}' not found in config.models")

        built_models: dict[str, torch.nn.Module] = {}
        for name, spec_raw in model_cfg.items():
            if not isinstance(spec_raw, dict):
                raise ValueError(f"config.models.{name} must be a mapping")
            spec = dict(spec_raw)
            if "path" not in spec:
                raise ValueError(f"config.models.{name}.path is required")

            built_models[name] = load_model(name, spec, self.manager.config)
            self.tokenizers[name] = load_tokenizer(self.manager.config, imp_model=name)

        self.models = built_models
        self._initialize_deepspeed_engine(model_name)
        return self.models

    def _initialize_deepspeed_engine(self, model_name: str) -> None:
        if not self.manager.deepspeed_config:
            raise ValueError("Manager deepspeed_config is empty. Call Manager.load_config(...) first")

        target_model = self.models[model_name]
        trainable_parameters = [
            parameter for parameter in target_model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError(f"Model '{model_name}' has no trainable parameters")

        try:
            deepspeed = import_module("deepspeed")
        except Exception as exc:
            raise ImportError("ToolBox.load_models requires `deepspeed` package") from exc

        engine, optimizer, _, scheduler = deepspeed.initialize(
            model=target_model,
            model_parameters=trainable_parameters,
            config=self.manager.deepspeed_config,
        )

        self.engine = engine
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.models[model_name] = engine

    def optim_step(self) -> None:
        if self.engine is None:
            raise ValueError("DeepSpeed engine is not initialized. Call load_models(...) first")
        self.engine.step()
