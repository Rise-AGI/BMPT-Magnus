# Brisk Module for Post-Training


```text
######   ##   ##   ######   #######
##   ##  ### ###   ##   ##     ##
######   #######   ######      ##
##   ##  ## # ##   ##          ##
######   ##   ##   ##          ##
```

## 快速开始

- 默认 `step` 实现为 SFT（见 `src/bmpt/algorithms/def_train.py`）。
- 安装（开发模式）：`pip install -e .`
- 训练入口：`bmpt-train --config src/bmpt/algorithms/config.yaml`
- workspace 自动发现：`bmpt-train --workspace /path/to/workspace`
- 指定 attention 实现：`bmpt-train --config src/bmpt/algorithms/config.yaml --attn-implementation auto`
- 单机多卡：`bmpt-train --nproc-per-node 8 --config src/bmpt/algorithms/config.yaml`
- 多机示例：`bmpt-train --nnodes 2 --node-rank 0 --master-addr <master_ip> --master-port 29500 --nproc-per-node 8 --config src/bmpt/algorithms/config.yaml`

更多参数与优先级说明见：`docs/cli.md`。
