# Xiaomi-Robotics-1 HG-DAgger

本分支实现基于已采集人工接管数据的 **HG-DAgger 离线聚合与策略训练**。
参考本地 RLinf `50f22ad9a7825eeaaf4180dca5300f2fa70758be` 的真实机器人 HG-DAgger，
保留 XR1 的模型、RGB 处理、动作表示和监督损失，不引入 critic、优势标签或 RECAP 条件提示。
这版不包含机器人在线遥操作、控制权切换驱动、Ray 在线训练或自动模型下发。

## 已完成训练与权重

`hg_dagger_1563_10ep_bs256_20261021` 已完成 10 epoch、12900 更新。
输出目录为 `xr1/outputs/project_xiaomi-robotics-1/hg_dagger_1563_10ep_bs256_20261021/`。

- 最佳验证 checkpoint：`best/step645-val0.93351.ckpt`，等权验证 loss 为 `0.9335145114`。
- 最终 checkpoint：`epoch09-step12900.ckpt`，等权验证 loss 为 `1.3256218433`；
  `last.ckpt` 指向这份最终权重。
- 每份 checkpoint 的策略文件均为 `checkpoint/mp_rank_00_model_states.pt`，
  配套使用该实验目录的 `config.py` 和 `normalize.json`，保持 `recap_enabled=false`。

最佳与最终两份权重均已打包到 ModelScope 数据集 `duanyanwuping/xiaomi` 的 `dagger/` 目录。
根据已有离线验证，优先用 step 645 做推理对照；实机成功率仍需单独测试。

## 上游依据与本地适配

- [RLinf 真实机器人配置](https://github.com/RLinf/RLinf/blob/50f22ad9a7825eeaaf4180dca5300f2fa70758be/examples/embodiment/config/realworld_pnp_dagger_openpi.yaml)：`only_save_expert: True`、`online_lerobot.only_success: True`。
- [RLinf actor](https://github.com/RLinf/RLinf/blob/50f22ad9a7825eeaaf4180dca5300f2fa70758be/rlinf/workers/actor/fsdp_dagger_policy_worker.py)：把 `only_save_expert` 传给 `require_all_intervene`。
- [RLinf 数据集](https://github.com/RLinf/RLinf/blob/50f22ad9a7825eeaaf4180dca5300f2fa70758be/rlinf/data/datasets/dagger/dataset.py)：动作块中每个非 padding 动作都必须是人工动作。
- [官方说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/hg-dagger.html)。注意 pinned 配置中“train on every frame”的注释与过滤代码不一致，本实现以过滤代码为准。

本地适配：XR1 监督 flow matching + 原有频域/choice 损失；固定示教和人工纠正双池 1:1；
保留 action-aligned 接管标记和已有时间连续性切分；未知标记一律排除；父 episode 级验证划分。
动作块到 episode/时间连续段末尾可以 padding，人工交还控制权不能当成 episode 末尾 padding。
归一化沿用初始化 SFT 权重对应的 norm，聚合新轮次时不重算。

| 数据段 | 审计终止标签 | 模仿训练 |
|---|---:|---|
| 成功人工示教 | 0 | 示教池，全部有效动作 |
| 人工接管前的自主段 | -1 | 排除；不会让模型模仿失败动作 |
| 最终确认成功的人工接管段 | 0 | 纠正池，仅全人工动作块 |
| 自主成功段，包括交还控制权后的自主段 | 0 | 排除，不当成人工专家动作 |
| 自主明确失败段 | -1 | 排除 |
| 失败后的人工段、未知结果或未知接管信息 | 未确认 | 排除 |

这里的 0/-1 是 `segments.jsonl` 中的审计标签，**不进入策略损失**。
HG-DAgger 通过学习人在模型访问过的困难状态下给出的纠正动作改进策略，不训练失败回报。
原始数据不删除；不符合条件的数据不出现在两个训练池中。

## 当前数据与默认训练参数

输入：`20261016_20261019_1563` 示教、`recap_755` rollouts。
复用 `xr1/data/recap_rlinf_1563_20261020` 已校验的动作对齐缓存。
不读取 critic 权重或旧优势文件；RGB 来自原始三路视频，不使用 critic 的缩略图。

| 池 | 训练采样帧 | 保留验证帧 |
|---|---:|---:|
| 人工示教 | 165241 | 7694 |
| 成功人工纠正 | 29471 | 673 |

184 条成功接管轨迹中 180 条用于训练、4 条保留验证；20 条最终失败的人工段被排除。
204 段接管前自主轨迹记录为失败。示教原始帧数与可训练帧数差异包含验证划分和 9 帧时间对齐保护。
验证 episode 不进入训练，但同一次采集会话可横跨划分；不是独立机器人测试集。
每半个 epoch（645 步）完整遍历验证池，记录 `val/demo_loss`、`val/correction_loss` 和
`val/loss = (val/demo_loss + val/correction_loss) / 2`，同时记录各项监督损失分量。
各池内按真实样本数加权 batch loss，池之间等权；验证关闭图像增强、固定扩散噪声和时间采样，
不改变训练 RNG。八卡分片不丢帧、不重复计数；空尾批执行零权重占位前向以保持分布式同步。
`best/` 保存等权验证 loss 最低的完整 checkpoint。验证 loss 不能代替实机成功率。

- 初始化：`middle_mixed_old2000_5ep_lr2e6_20261013/last.ckpt/checkpoint/mp_rank_00_model_states.pt`
  （对应 `epoch04-step22860`），新建 optimizer/scheduler。
- norm SHA256：`499ee7d13ef327f4a5df3212a2033507ea4cdcd4054263f99453982325943d2d`。
- 8 GPU × 每卡 32，全局 batch 256；每卡 16 示教 + 16 纠正。
- 大池每 epoch 采样一遍完整批次，小池打乱循环；每 epoch 1290 步，10 epoch 共 12900 步。
  示教池每 epoch 的最后 121 个随机样本留到后续打乱；纠正池每 epoch 约采样 5.6 遍。
- 全参数微调，BF16 mixed、DeepSpeed ZeRO-2；左臂 7 维、动作长度 30。
- 学习率从 `5e-8` warmup 387 步到 `2e-6`，cosine 到 `5e-7`。
- 每 epoch 保存完整权重；每 1000 步另存可恢复 checkpoint，保留最新两份。

## 准备、审计、启动

在仓库根目录执行。`prepare` 拒绝覆盖已有目录，已有数据无需重复执行。

```bash
source xr1/scripts/env.sh
python xr1/tools/prepare_dagger_dataset.py \
  --source xr1/data/recap_rlinf_1563_20261020 \
  --output xr1/data/hg_dagger_1563_20261021
bash xr1/scripts/train_dagger_8gpu.sh audit
bash xr1/scripts/train_dagger_8gpu.sh config
# 正式训练；只在明确要启动时执行，脚本要求 8 张空闲卡：
bash xr1/scripts/train_dagger_8gpu.sh train
```

脚本默认模式为 `config`，仅查看实际配置。
`audit` 全量核验数据哈希、每个动作/状态数组、视频偏移、全人工动作块、父轨迹隔离和 norm。
输出数据在 `xr1/data/hg_dagger_1563_20261021`，训练默认输出在
`xr1/outputs/project_xiaomi-robotics-1/hg_dagger_1563_10ep_bs256_20261021`。

持续运行入口为 `python -u xr1/tools/watch_dagger_training.py`：先用独立输出目录做两步八卡训练
和完整验证，通过后从原始 SFT 权重启动正式 10 epoch 训练。训练异常优先恢复完整 checkpoint；
无可恢复 checkpoint 的失败输出会保留到独立目录，然后重新初始化。OOM 时开启 DiT
梯度检查点，数据加载错误时降低 worker 数，保持 batch、采样比例和优化参数不变。
状态与日志在 `.queue/hg_dagger_1563_20261021/`；每次完整验证另写
`validation_metrics.jsonl` 和 `validation_latest.json`。

### 后续采集轮次

1. 用当前策略推理，在需要时人工接管；保存实际执行动作、action-aligned 接管标记、最终结果和原始 RGB。
2. 用现有转换器生成相同格式、相同 norm 的动作对齐缓存；不要重新移动已对齐的动作。
3. 重复传入 `--source` 聚合旧缓存和新缓存，输出到新的目录。父 episode 重复或 norm 不一致会报错。
4. 将 `DAGGER_DATA_DIR` 指向新目录，`PRETRAINED_CKPT` 指向上一轮的策略权重，
   用新的 `EXP_NAME` 开始下一轮。数据变了应重新建 optimizer，不用旧轮次的 optimizer resume。

```bash
python xr1/tools/prepare_dagger_dataset.py \
  --source xr1/data/recap_rlinf_1563_20261020 \
  --source /path/to/new_aligned_cache \
  --output xr1/data/hg_dagger_round2

# 同一轮中断恢复：路径必须有 restart_ready.json，数据指纹必须一致。
DAGGER_RESUME=/absolute/path/to/recovery/step1000.ckpt \
  bash xr1/scripts/train_dagger_8gpu.sh train
```

## 验证

```bash
source xr1/scripts/env.sh
CUDA_VISIBLE_DEVICES='' pytest -q xr1/tests/test_dagger.py
```

测试覆盖接管前失败、成功纠正过滤、未知数据排除、与 RLinf 的动作块筛选等价性、
多轮聚合/重复轨迹拒绝、norm 篡改、视频/动作偏移、八卡 1:1 和恢复采样、短块 padding 的零梯度。
