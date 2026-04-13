# Apptainer Build Guide

本项目使用 Apptainer 定义文件支持两种后端镜像：`torch` 和 `deepspeed`。

## 构建镜像

构建 PyTorch 版本：

```bash
apptainer build brisk-torch.sif apptainer/torch.def
```

构建 DeepSpeed 版本：

```bash
apptainer build brisk-deepspeed.sif apptainer/deepspeed.def
```

## 运行镜像

运行 PyTorch 路线：

```bash
apptainer run --nv --bind $(pwd):/workspace brisk-torch.sif
```

运行 DeepSpeed 路线：

```bash
apptainer run --nv --bind $(pwd):/workspace brisk-deepspeed.sif
```

## 说明

- 默认工作目录为 `/workspace`（通过 `--bind $(pwd):/workspace` 挂载当前项目）。
- 默认命令分别对应 `--backend pytorch` 或 `--backend deepspeed`。
- 可在 `apptainer run ... -- <args>` 追加参数覆盖默认行为。
