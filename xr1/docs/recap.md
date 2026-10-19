# XR1 RECAP 网线训练

算法参考 [RLinf RECAP](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)，
对应本机 `../RLinf/examples/offline_rl/advantage_labeling/recap/`。
完整流程为回报计算、categorical critic 训练、N-step 优势标注、CFG 策略微调。

## 数据和采样

- SFT 池：`0923_8072` 和 `20260928_935`，保留人工示教身份。
  复用 `ethernet_temporal_left_v1` 的连续片段，使用其全部原 train/validation/test。
  排除 4 条明确无效轨迹，最终 3,185,107 帧、11,729 个片段。
- Rollout 池：`recap_755` 的全部 755 条、330,413 帧，包括成功、失败及未标注轨迹。
- 原始数据保持不变；已有 action 前移 3 帧，适配中不再次前移。
- 每卡固定 16 个 SFT 和 16 个 rollout，8 卡全局 batch 256，无梯度累积。
  epoch 以大池的一遍为基准，小池打乱后循环补足。每个 epoch 24,883 步，
  大池末尾 83 帧不进入该轮完整 batch；每轮重新打乱。
- 复用最近网线模型的归一化和左臂 7 维相对动作定义，其他动作维度保持掩码。
  本轮没有独立的策略验证集或测试集，不报告独立任务成功率。

## RECAP 适配

1. 已知成功轨迹每步奖励 -1，终止奖励 0；失败终止奖励 -300，gamma=1。
   未知 outcome 原样保留，不补造价值目标。
2. 使用三路逐帧 64×64 RGB 和机器人观测状态训练 201-bin categorical critic，
   输出归一化价值 [-1,0]。这是针对固定网线任务的轻量 CNN+state 网络，
   与 RLinf 的 SigLIP/Gemma critic 架构不同，没有冒充复用其预训练价值模型。
   critic 不输入 outcome、接管标记、时间索引或未来图像。
   critic 默认训练 3 个 epoch，按 episode 划出约 5% 做诊断，选择 RMSE 最好的权重。
   这份诊断划分不构成策略的独立评测集。
3. 使用 30 帧 N-step TD 优势。仅在真实轨迹结尾令 bootstrap 为 0。
   从已知 outcome 的 rollout 非纠偏帧中取优势 top 30% 阈值；
   未知接管覆盖的帧仍保留其未知状态，不解释为已知自主动作。
   SFT 全部标为正；已知人工 action 标为正；接管前 30 帧窗口中的自主动作标负，
   避免把人工修复后的成功归功于之前的动作。
   未标注 outcome 的 rollout 作为无条件样本，其中已知人工纠偏仍为正。
4. 逐帧 action 干预掩码与 `action_human_intervals` 对齐，保留原始 observation/action
   区间及原始接管开始时间。轨迹级 True 但区间缺失时不推断接管位置，使用已有逐帧审计。
   策略 action chunk 在接管身份切换处截断，不能把自主动作窗口延伸到人工目标。
5. 在 XR1 DiT 的 timestep modulation 上增加零初始化的正优势条件参数。
   负样本始终无条件；正样本以 0.1 概率丢弃条件。已有预训练参数严格匹配，
   仅允许新增这一个条件参数。训练对 XR1 全参数更新。
6. 推理采用 `v_uncond + scale * (v_cond-v_uncond)`，默认 scale=1.5；
   scale=1 使用正条件策略，scale=0 使用无条件策略。

## 初始化和训练

初始化：
`outputs/project_xiaomi-robotics-1/middle_mixed_old2000_5ep_lr2e6_20261013/last.ckpt/checkpoint/mp_rank_00_model_states.pt`。
此模型完成 5 个 epoch、22,860 步。本轮仅继承模型权重，重新建立 optimizer/scheduler。

策略训练 10 个 epoch，共 248,830 步；bf16、DeepSpeed ZeRO-2；
AdamW，峰值学习率沿用上一轮的 2e-6，3% warmup，从 5e-8 开始，cosine 降至 5e-7。
每个 epoch 保存 checkpoint，`last.ckpt` 指向最终一轮。

在仓库根目录启动自动流水线：

```bash
source xr1/scripts/env.sh
export OMP_NUM_THREADS=4 NCCL_DEBUG=WARN
python -u xr1/tools/run_recap_pipeline.py
```

长任务由后台 controller 执行，进度文件：

```text
.queue/recap_20261020/status.json
.queue/recap_20261020/prepare.log
.queue/recap_20261020/value_train.log
.queue/recap_20261020/advantages.log
.queue/recap_20261020/policy_smoke.log
.queue/recap_20261020/policy_train.log
```

准备完成后可以分别执行：

```bash
source xr1/scripts/env.sh
torchrun --standalone --nproc_per_node=8 xr1/tools/train_recap_value.py
torchrun --standalone --nproc_per_node=8 xr1/tools/compute_recap_advantages.py
bash xr1/scripts/train_recap_8gpu.sh config
bash xr1/scripts/train_recap_8gpu.sh smoke
bash xr1/scripts/train_recap_8gpu.sh train
```

controller 会先验证正式标签下的 8 卡、每卡 32 的两步 smoke，通过后再开始 10 epoch。
无人值守启动改用 `python -u xr1/tools/watch_recap_pipeline.py`。
watchdog 重启异常退出的 controller；controller 会接管仍存活的训练子进程，避免重复启动。
任一阶段失败都会保存完整错误日志并退避重试，连续失败后的重试间隔最长 10 分钟。
30 分钟无输出会触发本任务进程树的重启，不终止其他任务。

critic 每 500 步原子保存模型、optimizer、随机状态及采样位置；
策略每 1000 步增加恢复 checkpoint，只采用全部 8 卡保存结束并带完整标记的目录。
策略恢复会同时恢复 optimizer、scheduler、epoch、global step 及 DataLoader 采样位置。
保留 epoch checkpoint；如果尚无可恢复点，失败的尝试会改名保留后重新启动。
优势计算缓存带 critic checkpoint 指纹，恢复时跳过已完成轨迹并重新生成不完整的派生文件。

显存不足会开启 DiT gradient checkpointing，DataLoader 错误会逐步降低 worker 数，
通信错误会重试并关闭单机不必要的 IB 通道。这些调整不减少 batch、不改变双池比例、
epoch 数和学习率，也不删原始数据。未知错误同样保留现场并继续退避重试，
不会把部分完成或反复报错标成训练成功。
可查看 `.queue/recap_20261020/watchdog_status.json`、`recovery_settings.json`
和 `error_*.txt` 排查恢复记录。

训练输出：`xr1/outputs/project_xiaomi-robotics-1/recap_10ep_bs256_20261020/`。
`run_record.json` 记录初始化、数据规模、标签和 batch；`training_progress.json`
记录步数和观测到的训练速度；`completed.json` 仅在 fit 结束后生成。
最终由 controller 再核对确实完成 10 epoch、248,830 步和最终 checkpoint。
`config.py` 支持原部署入口，`--recap-guidance-scale` 可以覆盖引导强度。
