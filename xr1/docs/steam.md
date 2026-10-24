# XR1 STEAM 离线强化学习

实现参考 RLinf `main` 的固定提交 `30e0236981c6a1dc19202de64116834c12673828`。
用户提供的 `/RLinf/RLinf/steam` 不是有效 GitHub 分支地址；实际代码在
[`rlinf/data/datasets/steam`](https://github.com/RLinf/RLinf/tree/30e0236981c6a1dc19202de64116834c12673828/rlinf/data/datasets/steam)
和 [`rlinf/models/embodiment/value_model/steam`](https://github.com/RLinf/RLinf/tree/30e0236981c6a1dc19202de64116834c12673828/rlinf/models/embodiment/value_model/steam)。

核心文件直接复制，保留 Apache-2.0 版权和许可证。
`mibot/steam/upstream/PROVENANCE.json` 记录逐文件来源和原始 SHA256。
唯一对上游整文件的改动是允许两个 PairDataset 构造函数传入 `TrajectorySource`。
因此无需安装 RLinf 的 Ray/LeRobot 环境，也不改变其时间配对、分桶、处理器、critic 或 ensemble 数学实现。
CFG 路由函数直接提取自官方 `openpi_cfg_action_model.py`。

## 方法对应关系

| RLinf 实现 | XR1 实现 |
|---|---|
| 正序/倒序帧对、随机步长 1…K、多分类时间进度 | 原版 PairDataset；K=32，32 bins |
| SigLIP + Gemma3、三成员独立采样和更新 | 原版 critic/ensemble；每成员独立微批，顺序 backward/step/zero_grad |
| `balance_weights=true` | 按数据集样本数量乘原始权重采样；默认两来源原始权重均为 1 |
| 平滑 CE、独立 value-head seed | 原版 loss，smoothing=0.05，seed=0/1/2 |
| 保守 ensemble advantage | signed-bin expectation 后取三个成员最小值；不以均值替代 |
| quantile 标签 | rollout top 30%，expert top 80%；严格 `score > threshold`；相等项不为正 |
| 末帧补零 | 保留官方 neutral terminal-row 约定；不是环境奖励 |
| positive-only CFG | 正优势以 0.9 概率加 `\nAdvantage: positive`；其余保留原任务条件 |
| CFG 流场 | `(1-w)*v_uncond+w*v_positive`，共用噪声和每一步的状态、时间 |

标签阈值只在训练划分拟合，再应用到验证划分，避免验证数据参与训练阈值估计。
官方 signed-value 源码直接对有符号 bin 求期望；这里不额外实现文档中可能出现的 baseline subtraction。
轨迹长度归一化默认关闭，与官方 YAML 一致；需要时在 critic 配置中开启 `length_scale_enabled`，
并可给出全局 `length_scale_reference`。分桶的舍入、截断完全使用上游函数。

XR1 必要适配：动作保持 30×60，只对真实左臂 7 维和实际存在的帧计算损失；
使用原有动作/状态归一化及已对齐的 t+3 动作，不再次平移。
保留 XR1 的 noise→action 时间方向和 Beta 时间采样，速度目标为 action-noise。
训练仅使用 flow MSE，不叠加 baseline 的频域/候选动作/评分损失；动作标签 token 不进入 VLM 条件。
每个样本先对有效维度求平均，再对 batch 求平均，避免短轨迹或补零维度改变样本权重。

运行层使用同步数据并行，梯度以 FP32 通信；critic 是 FP32 参数 + BF16 autocast。
XR1 使用 BF16 模型和按参数分片的 FP32 master AdamW，支持本机 8×H100。
这是本仓库的执行层适配，没有引入 RLinf 的 Ray/FSDP 集群调度。
本机配置用已有 `gemma-3-270m-it` 和 SigLIP 权重，语言路径可换成 Gemma3-270m base；
这不是宣称复现论文所有超参数、骨干版本或成功率。

## LWD 数据复用

生成位置：`/data/wz/STEAM/lwd_20261024/manifest.json`。
原始视频/动作可以复用，原 LWD transition/reward 索引不能直接充当 STEAM advantage。
转换器生成 STEAM episode manifest，由 XR1TrajectorySource 接入原版 PairDataset。
视频、NPZ 和原始 LWD 文件保持不动，manifest 使用其绝对路径；迁移到其他机器时需要一起迁移资产并重新生成索引。

- 共 2,341 episodes，训练 2,205 / 验证 136；保留原划分。
- LFD 1,596 作为 sft；RECAP_SUCCESS 575 和 RECAP_UNKNOWN 170 作为 rollout。
- RECAP 的 HUMAN/AUTO/intervention-unknown 元数据保持来源信息，不凭结果改成 expert。
- 可用 481,864 帧；沿用人工审核的边界，屏蔽 20,570 帧不明确尾部。
- 所有 7,023 视频的容器帧数、所有 NPZ 数组长度/有限性以及训练/验证数组去重检查通过。
- 未知结尾不标成失败；policy 进一步排除接触未知边界的动作 chunk。
- 不伪造 advantage；必须完成阶段 1 后执行阶段 2。

重新生成：

```bash
cd /data/wz/Xiaomi-Robotics-1-steam
export PYTHONPATH="$PWD/xr1"
export XR1_VIDEO_BACKEND=pyav
PY=/data/wz/Xiaomi-Robotics-1/.venv/bin/python
$PY -m pip install -r xr1/assets/requirements-steam.txt
$PY xr1/tools/prepare_steam_data.py \
  --source /data/wz/LWD/usable_20261021 --output /data/wz/STEAM/lwd_20261024
```

## 三阶段训练

STEAM 代码要求 Python 3.10+；本机已验证环境为共享的 XR1 Python 3.10 虚拟环境。

下列命令以本机路径配置运行。正式启动前可按显存/时间预算调整 JSON 的
`micro_batch_size`、`global_batch_size`、`max_steps`、`output`；全局 batch 必须整除 micro batch × GPU 数。
critic 的全局 batch 是**每成员**的 batch；每次记录的 step 会更新全部三名成员一次。
critic 保留官方的 16,000 steps；按用户要求将每成员 global batch 从官方 512 改为 128，
8 卡、每卡 micro batch 16，不做梯度累积。不能把三成员成本当作单模型估算。
policy 默认 10,000 steps / global batch 128 / lr 2e-6，是 XR1 本机设置。
用户指定 8 卡、每卡 micro batch 16，全局 batch 128，不做梯度累积。

```bash
# 1. STEAM temporal progress ensemble
OMP_NUM_THREADS=4 $PY -m torch.distributed.run --standalone --nproc_per_node=8 \
  xr1/tools/train_steam_critic.py --config xr1/configs/steam_critic.json

# 2. Generate frame-level advantages; per-episode cache supports restarting.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 $PY xr1/tools/label_steam_advantages.py \
  --checkpoint /data/wz/STEAM/runs/critic \
  --manifest /data/wz/STEAM/lwd_20261024/manifest.json \
  --output /data/wz/STEAM/lwd_20261024/advantages.parquet

# 3. XR1 CFG policy; initialized from completed baseline SFT, not LWD actor weights.
OMP_NUM_THREADS=4 $PY -m torch.distributed.run --standalone --nproc_per_node=8 \
  xr1/tools/train_steam_policy.py --config xr1/configs/steam_policy.json
```

`--all-experts-positive` 可替代 expert quantile；默认保持官方示例的 expert top 80%。
`label_steam_advantages.py` 生成逐帧 parquet、manifest/checkpoint 哈希、实际阈值和完整性 sidecar。
policy 拒绝不完整、错误 manifest、被修改、含重复键或普通配置下的 smoke-only 标签。

两个 trainer 均支持 `--stop-after N`（本次最多更新 N 步）和
`--resume /path/to/run`（读取 latest.json）。恢复要求同配置和同 GPU 数；
保存模型、每 rank 优化器、FP32 masters、采样器/Python/CPU/CUDA RNG、norm。
SIGINT/SIGTERM 在当前完整 step 后保存退出。长梯度累积的 step 可能需要一段时间。
权重写入临时文件后重命名，所有 rank 完成后才发布 latest.json；历史 generations 不自动删除。

一键按顺序运行三阶段（支持从已有 latest.json 恢复）：

```bash
bash xr1/scripts/train_steam_pipeline.sh
```

默认阶段状态和日志位于 `/data/wz/STEAM/runs/pipeline_20261024`，可用 `STEAM_LOG_DIR` 改日志目录。
训练失败时停止后续阶段；`status` 文件记录 critic/labeling/policy/complete/failed。

## CFG 推理与验证

```bash
CUDA_VISIBLE_DEVICES=0 $PY xr1/tools/infer_steam_policy.py \
  --checkpoint /data/wz/STEAM/runs/policy \
  --manifest /data/wz/STEAM/lwd_20261024/manifest.json \
  --episode 0 --frame 0 --guidance-scale 1.5 \
  --output /data/wz/STEAM/prediction.npz

$PY -m pytest -q xr1/tests/test_steam.py
```

推理产物是 normalized_action[1,30,60]，不是直接可下发的机器人控制量。
现有部署端若要使用 CFG，需要调用 `SteamPolicy.generate`，不能仅把权重装到 baseline.generate 并期待出现 guidance。

`/data/wz/STEAM/smoke` 保存开发验证日志。这里的小数据/极少更新只检查数值、梯度、checkpoint、标注和推理接口，
绝不能当作已训练好的 STEAM critic 或宣称任务成功率提高。
