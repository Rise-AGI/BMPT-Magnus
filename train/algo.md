# PBV 训练算法逻辑详解

本文基于 `train/` 目录中的实现，解释 PBV（Planner-Builder-Verifier）训练过程。核心入口是 `pbv_step.py` 的 `step()`，其余文件分别负责采样、模型构建、判分和公共工具。

## 1. 代码结构与职责

- `def_train.py`：对外导出训练所需的函数（配置加载、模型构建、训练 step、评估）。
- `pbv_common.py`：配置加载、模型/tokenizer 加载、token 处理、Prompt 片段拼接。
- `pbv_models.py`：按配置构建 planner/builder/verifier 及参考模型（ref）。
- `pbv_sampling.py`：采样并计算序列 logprob（单样本与批量）。
- `pbv_verifier.py`：Verifier 的 Right/Wrong 判分器（支持单条和 batch）。
- `pbv_step.py`：PBV 主训练逻辑（策略切换、采样、奖励、损失、指标）。

---

## 2. PBV 的三角色定义

### 2.1 Planner（规划器）

输入题目，输出“可批改的小目标”步骤序列（plan steps）。

- 通过 `PLANNER_PROMPT_SUFFIX` 强约束输出形式：每行一个目标。
- 采样后用 `_split_plan_steps()` 将文本切成 step 列表。

### 2.2 Builder（执行器）

针对每个 plan step，生成候选“下一步过程”。

- 每个 step 采样 `k_samples` 个候选。
- 候选由 Verifier 判定是否通过，再用于计算 Builder 的策略梯度。

### 2.3 Verifier（验证器）

输入：题目 + 当前计划步骤 + 已接受前缀 + 候选答案。

- 输出文本被解析为二值分数：`Right -> 1.0`，`Wrong -> 0.0`。
- 训练中用它给 Builder 候选打分，并间接给 Planner 提供 step 奖励。

---

## 3. 一次 `step()` 的整体流程

`step(models, input, engine=None)` 的核心流程如下：

1. 读取配置（`algorithm.training_schedule/planner/builder`）和 batch。
2. 用 `global_step` 与 `planner_steps + builder_steps` 决定当前 phase：
   - `phase = planner` 或 `phase = builder`。
3. 对 batch 中每个样本：
   - Planner 采样一个 plan；
   - 对 plan 的每个 step，Builder 采样多个候选，Verifier 打分；
   - 用打分结果更新 Builder（REINFORCE 风格）；
   - 把被选中的候选拼到前缀，继续下一 step；
   - 汇总 step 奖励，若当前 phase 是 planner，则再更新 Planner。
4. 汇总 loss/metrics/aux 返回。

若传入 `engine`（DeepSpeed），函数内部直接多次 `engine.backward(...)`，并返回 `backward_done=True`；否则返回可外部反传的 `loss` Tensor。

---

## 4. Phase 调度：Planner 与 Builder 交替

调度参数：

- `planner_steps`
- `builder_steps`
- `cycle = planner_steps + builder_steps`

调度规则：

- 若 `global_step % cycle < planner_steps`，则当前步更新 Planner；
- 否则更新 Builder。

这意味着可实现“若干步只训 planner，接着若干步只训 builder”的交替训练。

> 当前 `config.yaml` 里 `planner_steps: 0, builder_steps: 2`，所以实际始终处于 Builder phase，Planner 不参与训练（只用于生成计划）。

---

## 5. Planner 部分：从计划文本到逐步回报

### 5.1 计划生成

对每个 prompt，构造：

- `planner_input_text = 题目 + PLANNER_PROMPT_SUFFIX`
- token 化后调用 `_sample_with_logprob(planner, ...)` 采样 `plan_ids`
- 解码得到 `plan_text`

### 5.2 步骤切分

`_split_plan_steps(plan_text, max_steps)` 的策略：

- 先按换行拆，再按分号拆；
- 去空白与空项；
- 最后截断到 `max_plan_steps`。

### 5.3 Planner 奖励来源

Planner 的每个 step 奖励不是直接由 Planner 产生，而是来自：

- 该 step 上 Builder 被选中的候选是否通过 Verifier；
- 通过记为 `1.0`，失败记为 `0.0`。

即 `step_rewards_for_planner[t] ∈ {0, 1}`。

### 5.4 折扣回报

`_discounted_returns(step_rewards, gamma)` 计算：

\[
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
\]

并且实现里可把最后一步回报与样本最终标签 `final_label` 混合：

\[
G_{T-1} \leftarrow (1-\alpha)G_{T-1} + \alpha \cdot y
\]

其中 `alpha = reward_mix_final`，`y = final_label`。

### 5.5 Planner 损失（REINFORCE）

只在 `phase == planner` 时计算。

1. 对每个计划 step 计算其条件 logprob：
   - 逐步拼接前缀，调用 `_completion_logprob(planner, prefix, step_ids)`。
2. 对回报做中心化 baseline：
   - `A_t = G_t - mean(G)`。
3. 损失：

\[
\mathcal{L}_{planner} = -\frac{1}{T}\sum_t A_t \log \pi_{planner}(step_t|prefix_t)
\]

这就是经典策略梯度形式，中心化降低方差。

---

## 6. Builder 部分：多候选采样、判分与策略梯度

对 plan 中每个 step：

### 6.1 构造 step 条件上下文

拼接字段（通过 `builder_composer`）：

- 题目 token
- 当前 plan step（附加 `BUILDER_STEP_CONSTRAINT`）
- 已接受历史前缀 `prefix_ids`

得到 `step_prompt_ids`。

### 6.2 采样 k 个候选

`_sample_with_logprob_batch(builder, ..., num_samples=k, require_grad=True)` 返回：

- 每个候选的 token 序列 `candidate_ids`
- 在 Builder 下对应的总 logprob `logp`

### 6.3 计算参考策略 logprob（KL 约束）

对同一候选，用 `builder_ref` 计算 `logp_ref`：

- `cand_logp_refs = _completion_logprob_batch(builder_ref, ...)`

这里的 `builder_ref` 角色类似 PPO/GRPO 里的 reference policy，用于抑制策略漂移。

### 6.4 Verifier 批量判分

`verifier.judge_ids_batch(...)` 逐候选输出 `verifier_prob ∈ {0.0, 1.0}`（当前实现是离散）。

再按阈值 `verifier_threshold` 得到：

- `is_pass = 1` 或 `0`。

### 6.5 Builder 奖励定义

每个候选奖励：

\[
r_i = \mathbf{1}[pass_i] - \beta \cdot (\log \pi_{builder} - \log \pi_{ref})
\]

对应代码：`reward = is_pass - kl_beta * kl_est`。

其中 `kl_est` 用单样本 logprob 差近似。

### 6.6 Builder 损失（组内中心化 REINFORCE）

同一 step 的 `k` 个候选形成一组：

- `advantage_i = reward_i - mean(reward_group)`
- `loss_builder = -mean(advantage_i * logp_i)`

如果组内优势全为 0，则用 0 loss 占位（避免无意义梯度）。

### 6.7 从候选中选一个“被接受步骤”推进前缀

用于后续 step 的 `prefix_ids` 递进，不直接用于 Builder 反传：

- 若有通过样本：在通过集里按 `exp(logp / tau)` 加权随机采样；
- 若无通过样本：退化为选择 Verifier 概率最高的候选。

被选候选的 pass（1/0）会写入 `step_rewards_for_planner`，作为 Planner 的 step 奖励来源。

---

## 7. 采样与 logprob 计算实现细节

### 7.1 `_sample_with_logprob` / `_sample_with_logprob_batch`

- 都先 `model.generate(...)` 采样 tokens。
- `require_grad=True` 时，为了让 logprob 可反传，会再前向一次 `model(full_ids)`，从 logits 取出每个生成 token 的 logprob 累加。
- `require_grad=False` 时直接用 `generate` 返回的 `scores` 近似求和，开销更小。

### 7.2 `_completion_logprob(_batch)`

- 在给定 `prompt_ids` 下，按 token 自回归地累加 completion 的 logprob。
- 主要用于：
  - Planner 计算自身 step 的 `log π`；
  - Builder 计算 reference policy 的 `log π_ref`。

---

## 8. Verifier 判分机制

`QwenProcessVerifier` 的设计是“文本分类 via 生成”：

1. 把输入四段（题目、计划步、前缀、候选）按 composer 拼成完整 prompt。
2. 调用 `generate(max_new_tokens<=6)` 让模型输出判断词。
3. 解析开头：
   - 以 `right` 开头 => `1.0`
   - 以 `wrong` 开头 => `0.0`
   - 其他 => `0.0`

批量版 `judge_ids_batch()` 先 pad 成 batch，再一次 generate，提高吞吐。

---

## 9. 模型构建与冻结策略

`build_models_from_config()` 的关键点：

- `verifier` 永远冻结并 `eval()`。
- `builder_ref`：
  - 若 builder 不可训练，直接与 builder 共用；
  - 若可训练且路径不同，深拷贝一份冻结模型作 ref。
- `planner_ref`：仅在 planner 可训练且 `planner_steps>0` 时深拷贝，否则共用 planner。
- 若开启 `gradient_checkpointing`，会对可训练模型关闭 `use_cache` 并启用检查点。

优化器 `build_optimizers_from_config()` 将 planner+builder 参数一起交给一个 AdamW；但实际谁产生梯度由 phase 与 `requires_grad` 决定。

---

## 10. 评估逻辑 `evaluate()`

评估不走 Planner/Builder，而是直接用 Verifier 对验证集 `prompt + response` 判定是否通过：

- `verifier_prob >= 0.5` 记为通过；
- 输出 `pass_rate`、样本数、batch 数。

这更像“验证器视角的一致性评分”，而非最终任务准确率。

---

## 11. 与 `config.yaml` 的对应关系（当前配置）

当前配置重点：

- `planner_steps: 0`，`builder_steps: 2`：只训练 Builder。
- `builder.k_samples: 2`：每个计划步仅采 2 个候选。
- `builder.kl_beta: 0.02`：KL 惩罚较轻。
- `builder.verifier_threshold: 0.5`：`Right` 判为通过。
- `planner.reward_mix_final: 0.0`：Planner 回报不混入最终标签（即使将来训练 planner）。
- `models.*.path` 全是同一个 Qwen2.5-7B：三角色共享同基座能力，但在训练角色上分工不同。

---

## 12. 一句话总结

这套 PBV 实现本质是：

- Planner 先把题目拆成步骤；
- Builder 对每步做多候选探索；
- Verifier 给离散通过信号；
- Builder 用“通过奖励 - KL 惩罚”做组内 REINFORCE；
- Planner（若启用）再用 Builder 的逐步通过结果做折扣回报学习“更好拆解步骤”。
