# XR1 RECAP 网线训练

## 2026-10-24：策略训练完成与推理

`events_1563` 策略已于 2026-10-23 17:08 UTC（北京时间 10 月 24 日 01:08）
完成 10 epoch、16660 更新。最终轮平均训练 loss 为 `0.2049303204`，
最后 100 步平均为 `0.2030561382`。本轮 `limit_val_batches=0`，
未执行策略验证或实机成功率评测；最终 checkpoint 是推理测试候选，不代表已验证的最佳策略。

推理使用实验目录
`outputs/project_xiaomi-robotics-1/recap_events_1563_10ep_bs256_20261021/`：

- `last.ckpt` 指向 `epoch09-step16660.ckpt`，模型文件为
  `last.ckpt/checkpoint/mp_rank_00_model_states.pt`。
- 配套使用该目录的 `config.py`、`normalize.json`，保留
  `recap_conditioning=language`、`recap_guidance_scale=1.0`。
- 部署入口自动使用正优势语言条件；critic 的 `best.pt` 用于优势标注，不是动作策略权重。

从仓库根目录启动推理服务：

```bash
source xr1/scripts/env.sh
python xr1/mibot/server/deploy.py \
  --model xr1/outputs/project_xiaomi-robotics-1/recap_events_1563_10ep_bs256_20261021 \
  --recap-guidance-scale 1.0
```

critic 选择 step 7500，三类事件平均验证 RMSE 为 `0.3403583089`；
最终 step 8000 为 `0.3534773938`。验证集为 61 条示教、22 条运行成功、
4 条纠正成功、2 条失败，按原始轨迹隔离。失败样本覆盖有限，且 rollout 训练与验证
误差存在明显差距；后续应在统一协议下比较策略各轮权重并做实机评测。

## 2026-10-22：使用已完成 critic 继续策略微调

用户已授权“基于这个，继续微调 recap 分支的训练”。工作区切回 `recap`，保留现有改动。
于 2026-10-22 13:27 UTC 启动 `events_1563 --stop-after policy` 后台流水线，
依次核验数据、生成优势、执行 8 卡两步策略检查，再训练策略；已完成的 critic 不重训。

- critic 使用 `recap_events_value_1563_20261021/best.pt`（step 7500，
  event-macro validation RMSE 0.3403583089），SHA256 为
  `7f90393671d33eca30d3f35c459f9ed879e33d55cd960f182afb3e937771cafa`。
- 策略初始化固定为原 SFT `middle_mixed_old2000_5ep_lr2e6_20261013/epoch04-step22860.ckpt`，
  使用新优化器；其余沿用下述事件奖励、筛选及语言条件配置。
- 8 卡、每卡 16 示教 + 16 rollout、全局 batch 256；10 epoch、16660 步，
  峰值学习率 2e-6。事件标签、原版 critic 适配及恢复相关测试共 22 项通过。
- 启动配置与本次授权保存在 `.queue/recap_events_1563_20261021/launch.json`；
  前次 critic 状态、代码差异和事件代码快照保存在同目录的
  `policy_launch_history/20261022T132740Z/`。
- 实际阶段以 `.queue/recap_events_1563_20261021/status.json` 为准；
  策略输出为 `outputs/project_xiaomi-robotics-1/recap_events_1563_10ep_bs256_20261021/`，
  仅在最终完成标记与 checkpoint 核验通过后视为训练完成。

本次优势标签已生成，训练集自主片段阈值为 `0.00010418642341392115`；
全部示教与保留的人工纠正片段均为正标签，验证片段不进入策略训练或阈值计算。
8 卡两步检查通过，loss 为 `2.7344429493`、`2.5636327267`。
正式策略训练于 2026-10-22 13:41 UTC 启动，已观察到实际参数更新及有限 loss；
后台 watchdog 持续运行，每 1000 步保存恢复点、每 epoch 保存模型。

## 2026-10-21：四类事件回报与严格筛选（critic 已启动）

用户审核通过后，正式 critic 于 2026-10-21 07:07 UTC 启动。全量数据审计和独立的
8 卡 2 步检查已通过。此次授权范围为 critic，完成 8000 更新后停止，不自动生成优势或训练策略。

用户确认的原始终点奖励：示教成功 0，自主成功 0，纠错接管 -0.5，明确失败 -1。
保留 RLinf 原版视觉语言 critic 结构；事件回报与筛选属于任务适配，不是原样复现 RLinf。
普通步骤按统一帧数基准扣除小额时间成本，每段累计不超过 0.1；gamma=1。
原始回报除以 1.1 后落在原版 201-bin head 的 [-1,0] 内，不改变终点奖励的语义。
时间基准取保留来源轨迹的最大帧数，只用于时间成本，不把加速视频时长或录制结束当作任务超时。

- 接管之前的自主段以 -0.5 终止，bootstrap=0，后面的人工成功不回传到这一段。
- 已知成功 rollout 的人工纠正片段仅用于策略正向监督，不参与 critic 回报拟合。
  失败 rollout 中的人工动作片段排除；人工交回后的自主片段另行评价。
- 示教全部正标签；保留的人工纠正动作全部正标签。其余动作需要优势严格大于 0，
  且达到训练集自主片段优势的 top 30% 阈值。验证数据不参与定阈值。
- 未知 outcome、未知接管覆盖、接管发生在已裁掉动作中的轨迹全部从训练清单排除。
  原始文件保持不变，逐段排除原因写入 `excluded.jsonl`。
- 源数据没有 stop reason；只采用显式 success/failure，不推断危险、超时或设备中断。
  已知接管标志及动作 0→1 切换按任务纠正接管解释，使用前移三帧之后的动作时间基准。
- 验证按原轨迹分组，同一轨迹的所有片段都在同一侧；验证轨迹也从策略训练排除。
  对筛选后稀少的失败类型补入至少两条完整验证轨迹。分类误差和实机成功率仍需训练后验证。
- 两池继续 1:1、每卡 16+16、8 卡全局 256；原始 SFT `epoch04-step22860` 初始化，
  新优化器、策略 10 epoch。critic 重新从预训练组件训练，旧 critic 不复用。
  fingerprint 同时绑定源数据、奖励、切段和验证划分，防止旧 critic/优势标签误用。

新数据：`data/recap_events_1563_20261021`；
critic：`outputs/recap_events_value_1563_20261021`；
策略：`outputs/project_xiaomi-robotics-1/recap_events_1563_10ep_bs256_20261021`。

实际筛选：1563 条示教来源全部保留，rollout 保留 520 条来源中的有效片段，
235 条整轨迹剔除（接管未知 219、结果未知 10、接管动作已被裁掉 6）。
同时剔除失败轨迹中的人工片段，总计排除 105,886 帧。
扣除独立验证及原有时间断点保护后，策略池为示教 165,241 帧、rollout 213,341 帧；
每 epoch 1666 更新，10 epoch 16660 更新。critic 训练帧为示教 165,250、rollout 183,870。
明确自主失败原轨迹仅剩 16 条，其中 2 条作为验证；该类别验证指标样本量有限。
时间成本基准 2064 帧，每普通步骤约 -0.0000484496；终点奖励精确为 0/-0.5/-1。

准备与只读数据核验（CPU）：

```bash
source xr1/scripts/env.sh
python xr1/tools/prepare_recap_events.py
python xr1/tools/verify_recap_events.py
```

准备脚本拒绝覆盖已有目录。本次经用户审核后使用以下入口启动：

```bash
source xr1/scripts/env.sh
python -u xr1/tools/watch_recap_pipeline.py --profile events_1563 --stop-after critic
```

该命令依次执行数据审计、8 卡 critic smoke、critic 训练，然后停止。
只有后续另行授权完整流程时，才继续优势计算、策略 smoke 和策略训练。
训练前后的检查不会把离线 loss 当作实际插入成功率。

## 历史实验：1563 示教，已因错误标签停止

以下是 2026-10-20 实验的历史配置。该实验把示教也按优势阈值筛选是实现错误，
于 step 6744 停止，最近完整恢复点为 step 6000。禁止恢复该实验；controller 和 watchdog
现会拒绝带 `invalidated.json` 的实验目录。旧权重、数据和日志保留用于对照。

历史启动命令（已禁用）：

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
  当时错误地没有强制把全部示教改成正标签，与 RLinf 实现不符；也没有人工纠正标签覆盖。
  未知 outcome 不造价值目标，策略当时按无条件样本保留。
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
