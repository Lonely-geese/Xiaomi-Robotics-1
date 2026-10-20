# XR1 RECAP 网线训练

## RLinf 原版结构：1563 示教实验

启动命令：

```bash
source xr1/scripts/env.sh
python -u xr1/tools/watch_recap_pipeline.py --profile rlinf_1563
```

本实验与下文旧 CNN 实验使用独立目录，旧实验保持停止。

- 示教池：`20261016_20261019_1563`，1563 条已知成功示教；
  rollout 池：`recap_755`，755 条成功/失败/未知轨迹。原始数据只读。
- 沿用 `middle_mixed_old2000_5ep_lr2e6_20261013/last.ckpt` 的 XR1 模型权重与
  normalization，重新建立 optimizer/scheduler。每卡 16 示教 + 16 rollout，
  8 卡全局 batch 256，policy 训练 10 epoch。epoch 按较大池的一遍计算。
  本次准备保留 172,935 示教帧（采集断点的 shift3 guard 移除 9 帧）与
  330,413 rollout 帧，每 epoch 2,581 步，共 25,810 步。
- critic 直接使用 RLinf `50f22ad9a7825eeaaf4180dca5300f2fa70758be`
  的 `ValueCriticModel`、`ValueExpert` 和 processor。原文件保存在
  `mibot/models/rlinf_recap/`，`upstream.json` 逐文件记录 SHA256，构建前验证。
  结构为预训练 SigLIP2-so400m-patch14-224、预训练 Gemma3-270M、原版
  `gemma_1m` 专家、可学习 CLS 与 201-bin 分类价值头；没有自定义 CNN 或状态分支。
- 三路 RGB 按原版 `resize_with_pad` 等比缩放、补黑边至 224×224；
  使用原 processor 的图像增强、`[-1,1]` 归一化与 `Task: {prompt}.` 文本。
- critic 使用原版 backbone/head 两组学习率 5e-5/1e-4，warmup 500 步，
  AdamW betas=(0.9,0.95)、weight decay=1e-10、clip=1，FP32 master + BF16 autocast。
  训练 8000 updates，每 500 步分别报告示教/rollout 的 episode-disjoint
  诊断结果，按 rollout validation RMSE 选择 checkpoint。这不是策略独立成功率评测。
- 回报遵循原版 -1/成功终止 0/失败终止 -300、gamma=1；
  N-step=10，在全部已知 outcome 的两池帧上统一取优势 top 30%。
  不强制把全部示教或人工帧改成正标签；未知 outcome 不造价值目标，策略按无条件样本保留。
  接管时间仍用于 action chunk 截断，防止窗口跨自主/人工切换；采集时间断点也截断。
- XR1 的 CFG 适配遵循原版 `positive_only_conditional`：正样本以 0.1 概率
  去掉 `\nAdvantage: positive` 语言条件，负样本始终无条件。
  不增加 timestep embedding 参数。推理采用同一噪声下的双分支速度 CFG，
  guidance scale 默认与 RLinf 配置一致为 1.0，可用部署参数覆盖。
  policy 的 Qwen/XR1 主干、动作表示和原有损失保留；它不是把策略替换成 π0.5。
- 自动流水线顺序为：准备数据 → 原版 critic 的 8卡×32 两步测试 → critic 8000步 →
  优势标注 → 正式标签下的 policy 两步测试 → policy 10 epoch。
  recovery checkpoint 与 watchdog 沿用自动恢复机制，旧 CNN critic/标签不参与本实验。

状态与产物：

```text
.queue/recap_rlinf_1563_20261020/status.json
.queue/recap_rlinf_1563_20261020/{prepare,value_smoke,value_train,advantages,policy_smoke,policy_train}.log
xr1/data/recap_rlinf_1563_20261020/
xr1/outputs/recap_rlinf_value_1563_20261020/
xr1/outputs/project_xiaomi-robotics-1/recap_rlinf_1563_10ep_bs256_20261020/
```

部署使用该实验保存的 `config.py`、权重和同目录 `normalize.json`；
部署入口校验 norm 指纹，并自动为语言条件模型插入正优势提示。

## 旧 CNN 实验（已停止）

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
