# Apptainer Build Guide

本项目使用 Apptainer 定义文件提供 DeepSpeed 训练镜像。

## 构建镜像

构建 DeepSpeed 版本：

```bash
apptainer build brisk-deepspeed.sif apptainer/deepspeed.def
```

## 运行镜像

运行默认训练命令：

```bash
apptainer run --nv --bind $(pwd):/workspace brisk-deepspeed.sif
```

覆盖 attention 实现（传给 `bmpt-train`）：

```bash
apptainer run --nv --bind $(pwd):/workspace brisk-deepspeed.sif -- --attn-implementation auto
```

```bash
apptainer run --nv --bind $(pwd):/workspace brisk-deepspeed.sif -- --attn-implementation sdpa
```

DeepSpeed 镜像默认不编译自定义算子（`DS_BUILD_OPS=0`），避免编译依赖问题（如 `oneapi/ccl.hpp` 缺失）。
如需启用自定义算子编译，可手动改回 `DS_BUILD_OPS=1` 并补齐对应编译环境。

## 说明

- 默认工作目录为 `/workspace`（通过 `--bind $(pwd):/workspace` 挂载当前项目）。
- 默认命令对应 `bmpt-train --config train/config.yaml`。
- 可在 `apptainer run ... -- <args>` 追加参数覆盖默认行为。

## FlashAttention 依赖

- `apptainer/*.def` 在构建时会尝试安装 `flash-attn`。
- 若安装失败，训练会自动回退到默认 attention（配合 `attn_implementation=auto` 的回退策略）。

容器内验证命令：

```bash
python3 -c "import torch; print(torch.__version__)"
```

```bash
python3 -c "import flash_attn; print(flash_attn.__version__)"
```

## 构建失败备用方案（localimage）

当出现拉取基础层失败（例如 `error writing layer`）时，先拉取基础镜像到本地，再用 `localimage` 定义文件构建。

先拉基础镜像：

```bash
apptainer pull base_pytorch.sif docker://pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
```

再构建（DeepSpeed 版本）：

```bash
apptainer build brisk-deepspeed.sif apptainer/deepspeed.local.def
```
