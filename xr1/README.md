# Post-Training & Deployment

Post-training and deployment code for **Xiaomi-Robotics-1** — fine-tune the released checkpoint on your own data and deploy the model with the same client–server runtime used for evaluation.

All commands below run from the `xr1/` directory.

---

## Installation

Requires Python ≥ 3.9 and a CUDA GPU. `transformers` must be exactly `4.57.1` (other versions are unverified).

```bash
cd xr1
pip install -e .                 # installs assets/requirements.txt
pip install flash-attn --no-build-isolation
```

`Qwen/Qwen3-VL-4B-Instruct` must be available from Hugging Face or the local cache (the VLM backbone used by the model).

---

## Checkpoint

Download the **[Xiaomi-Robotics-1-5B model](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B)** checkpoint from Hugging Face and save it as `pretrained_ckpt/model_states.pt`.

---

## Data

Download the **[XR-1 post-training demo dataset](https://huggingface.co/datasets/XiaomiRobotics/xr1_post_train_demo/tree/main)** from Hugging Face. Each episode is one metadata JSON plus three synchronized videos (ego / wrist-left / wrist-right). See [`docs/data_format.md`](docs/data_format.md) for the full JSON schema.

The default config (`configs/data/load_washer.yaml`) expects five example episodes at `data/json1.json` … `data/json5.json`, with their videos under `data/videos/`. To train on your own data, edit that config:

1. Set `paths` to JSON files or directories (directories are scanned recursively).
2. Replace `mean` and `std` with the per-step mean and standard deviation of the packed relative actions. Both arrays must have shape `(action_length, action_dim)`; the defaults are `action_length = 30` and `action_dim = 60`.
3. Replace `q01` / `q99` with quantiles of the packed robot states. Both arrays must have shape `(state_length, state_dim)`; the defaults are `state_length = 1` and `state_dim = 60`.

Compute normalization statistics with this reference demo:

```bash
python tools/compute_normalize.py /path/to/training/data -o normalize.json
```

Copy `mean`, `std`, `q01`, and `q99` from `normalize.json` into the data config.

Video paths are read from each JSON file. `crop_bbox` uses `(top, left, height, width)` and may be `null`.

---

## Training

Log in to Weights & Biases, then launch training via the launcher script:

```bash
wandb login

RESOURCE_GPU=1 bash scripts/train.sh \
  trainer.project="xiaomi-robotics-1" \
  trainer.exp_name="posttrain" \
  trainer.default_root_dir="outputs" \
  data=load_washer \
  model=posttrain \
  model.params.pretrained="pretrained_ckpt/model_states.pt"
```

`RESOURCE_GPU` is the number of local GPUs. For multi-node, the launcher also reads `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, and `MASTER_PORT` directly. Asynchronous training is enabled by default; add `model.params.model.async_train=false` for synchronous training.

Checkpoints and the resolved config are written to `outputs/project_<project>/<exp_name>/` (e.g. `outputs/project_xiaomi-robotics-1/posttrain/`).

### Data_0904_clean_v2

The LeRobot network-cable dataset can be prepared and trained with one command. By default,
the launcher uses `/mnt/cfs/l1x67e/ye/Data_0904_clean_v2`, physical GPUs 4–7, and a per-GPU
batch size of 32:

```bash
cd xr1
bash scripts/train_network.sh
```

The launcher uses the repository's `../.venv` by default (override it with `VENV_DIR` if needed). On the first run, it converts the parquet trajectories to the XR-1 JSON format and computes normalization statistics under `data/Data_0904_clean_v2_xr1/`. The three source videos are referenced in place and are not copied. Later runs reuse the prepared data.

This adapter uses the dataset's 30-D action: `action[28]` and `action[29]` are the left/right gripper targets. It remains compatible with the legacy 32-D dataset by discarding its appended `action[30]` and `action[31]`. Joint state values are converted from degrees to radians. The source end-effector quaternions use `qx, qy, qz, qw` order.

Common overrides are environment variables; additional Hydra overrides can be appended:

```bash
GPU_IDS=4,5,6,7 \
DATASET_DIR=/mnt/cfs/l1x67e/ye/Data_0904_clean_v2 \
PRETRAINED_CKPT=/models/model_states.pt \
RESOURCE_GPU=4 BATCH_SIZE=32 EPOCHS=5 \
bash scripts/train_network.sh model.params.model.async_train=false
```

`BATCH_SIZE` is per GPU. The launcher computes `steps_per_epoch = ceil(total_frames / (GPU count * BATCH_SIZE))`, trains for `EPOCHS`, and saves at every epoch boundary. With this dataset and the defaults above, that is 3,111 steps per epoch and 15,555 optimizer steps in total.

Set `FORCE_PREPARE=1` to rebuild generated JSON and statistics after changing the converter or source data.

---

## Deployment

Serve a trained checkpoint with the inference server (requires `tmux`):

```bash
bash scripts/deploy.sh outputs/project_xiaomi-robotics-1/posttrain 1 1
```

The last two arguments are `<num_ports> <num_gpus>`; ports start at `10086`. Inspect the servers with `tmux attach -t model_servers`.

### Client API

`mibot/server/runtime/client.py` accepts three PIL images, an instruction, and the current robot state, and returns `raw_action`, `action_components`, and `action_targets`.

The robot state contains `left/right_arm_joint`, `left/right_gripper_pos`, `left/right_ee_pos`, and `left/right_ee_rotm` (`waist_pos` is optional). For asynchronous execution, pass an `(N, 60)` unnormalized action prefix via `action_prefix=...`; optional camera crops are three `(top, left, height, width)` tuples via `crop_bboxes=...`.

---

## License

Licensed under the [Apache License 2.0](../LICENSE).

## 本机：左臂相对笛卡尔全量微调（2026-10-02）

本次入口是 `scripts/train_left_8gpu.sh`，数据为 `data/ethernet_temporal_left_v1/`。
旧的 `train_ethernet_full.sh` 和 `ethernet_9007_verified` 保留作历史记录，本次队列不使用它们。

项目 `.venv` 指向本机 `/opt/xr1-env`，Python 3.12、PyTorch 2.8.0+cu126、
Transformers 4.57.1、Lightning 2.5.3、DeepSpeed 0.18.9、Flash Attention 2.8.3、SwanLab 0.10.1。
官方 5B checkpoint 在 `pretrained_ckpt/model_states.pt`，已核对 SHA-256；
VLM 权重包含在该 checkpoint 中，本地 tokenizer/config 在 `../models/Qwen3-VL-4B-Instruct`。
使用 PyAV 视频解码和 PyTorch fused AdamW，无需编译本机 CUDA 13 的 DeepSpeed Adam 扩展。
更换容器后可用 `scripts/setup_env.sh` 重建环境；本机 tmux 队列需要当前容器保持运行。

```bash
source /mnt/cfs/7rnh3z/kele/Xiaomi-Robotics-1/xr1/scripts/env.sh
cd /mnt/cfs/7rnh3z/kele/Xiaomi-Robotics-1/xr1
bash scripts/train_left_8gpu.sh config   # 仅查看完整配置
```

### 数据与时序

原始输入 `/mnt/cfs/7rnh3z/kele/data/00_traindata/0923_8072` 和
`/mnt/cfs/7rnh3z/kele/data/00_traindata/20260928_935` 未修改。
按要求只检查时序，不按任务成功、质量标签或笛卡尔跳变筛选。

原始 9,007 episodes / 3,219,792 frames 全部纳入时序检查。
依据原始时间戳、观测/动作序号及相机采样序号切分断点，移除跨断点的 shift3 标签、
保护内部边界前三帧，并舍弃不足 30 帧的短片段。没有整条 episode 被剔除；
共移除 33,854 帧（1.05%）。本次没有另做首尾静止帧裁剪。
已有中段静止帧删除造成的真实时间断点通过原始 UTC 时间戳识别，不能仅看重置后的 30 Hz 时间列。
相邻时间差阈值 50 ms，动作 UTC 时间跨度允许 50–150 ms；相机序号小跳 2–3 记入报告，
大于 3 或倒退则切分，连续重复至少 3 次的帧排除。采样序号只是时序证据，不能证明相机硬件同步。

以原始采集会话整体划分，seed=42，目标约 90%/5%/5%；缺少细会话身份的旧数据按来源日归组。
训练/验证/测试之间没有源 episode 或会话交叉，归一化只用训练集。

| 划分 | 原始 episode | 连续片段 | 帧数 |
| --- | ---: | ---: | ---: |
| 训练 | 7,982 | 10,699 | 2,856,371 |
| 验证 | 492 | 496 | 165,152 |
| 测试 | 533 | 538 | 164,415 |

验证和测试每 30 帧取一个动作窗口，分别 5,746 / 5,738 个窗口，关闭图像随机增强。
分布式评估 sampler 会补齐至 8 卡可均分，最多重复 7 个窗口。
每 epoch 验证；第 10 epoch 后仅一次独立测试。日志损失衡量动作拟合，不能等同真实机器人任务成功率。
完整清单与阈值见该数据目录的 `prepared.json`、`temporal_audit.jsonl`、
`segment_manifest.jsonl`、`session_splits.json`。

### 监督与训练参数

任务提示词与 G0.5 的 `ethernet_normal0923_0928_shift3_fast` 训练一致：
`Pick up the Ethernet cable and insert it into the RJ45 network port.`
含义为“拿起网线并插入 RJ45 网口”。已核对 G0.5 的实际 `train_snapshots/first_batch.html`、
运行配置和两批源数据 `meta/tasks.jsonl`；0923 批次带末尾句号，0928 批次不带句号。
本次沿用各批次原文，并使用 XR-1 自己的多视角输入模板和 `/no_cot` 标记。

只监督左臂相对末端 xyz、相对姿态 axis-angle、左夹爪，共 7 维（60 维兼容布局的索引 0–6）。
相对位置为 `R_current.T @ (p_target - p_current)`，姿态为
`axis_angle(R_current.T @ R_target)`；夹爪为目标开度减当前开度。
两批原始 action 已是 `state[t+3]`，不再额外移动标签。
右臂、右夹爪、腰部和底盘动作不计入 Flow、候选动作及评分损失。
右侧相机和本体观测仍作上下文。全模型 5,113,222,656 参数可训练，包括词嵌入，无 LoRA。

- 单机 8 张 A800，bf16 mixed，DeepSpeed ZeRO-2。
- 全局有效 batch=128：默认每卡 4，梯度累积 4。
- 10 epochs，每 epoch 22,315 次更新，总计 223,150 次。
- 每 epoch 打乱后舍弃最后 51 个样本，保证每次更新均为 128 个样本。
- Warmup 6,695 次（总更新数的 3% 向上取整）；之后 cosine decay。
- AdamW：起始 LR 5e-7，峰值 2e-5，最低 5e-6，betas=(0.9,0.95)，weight_decay=0.1。
- 每 epoch 保存，保留验证损失最好的 2 份和 last；正式训练从官方权重重新开始。

SwanLab 已以 `socoter` 登录，使用在线私有项目：
https://swanlab.cn/@socoter/xiaomi-robotics-1 。
记录训练/验证/测试损失及 LR，同时保留本机 CSV。
正式输出目录 `outputs/project_xiaomi-robotics-1/left_relative_10ep_20261002/`；
其中 `swanlab_run.json` 保存实际实验地址，`test_metrics.json` 保存最终测试结果，
`completed.json` 仅在流程成功后写出。`env.sh` 的 W&B 离线默认值不影响 SwanLab 在线日志。

### 排队与检查

后台 tmux 会话：`xr1_left_queue`。
队列每 60 秒检查 8 卡，连续三次满足每卡显存占用不超过 1,024 MiB、利用率不超过 5% 后，
自动做完整 5B 模型的 8 卡短跑（一次更新和少量验证），通过后再从官方权重开始 10 epochs。
若短跑 OOM，自动将每卡 batch 4→2→1、累积 4→8→16，全局 batch 保持 128；
其他错误或正式训练失败会停止并记录，不自动覆盖重跑。
该队列是本机空闲检测，不是集群资源预约，不终止其他任务。

```bash
cat ../.queue/left_relative_10ep/status.json
tail -f ../.queue/left_relative_10ep/queue.log
# 查看 tmux：
tmux attach -t xr1_left_queue
```

已通过 10 项 pytest（含时序切分、会话划分、完整 batch、左侧损失/梯度屏蔽）、
真实四样本组批及内部切分视频偏移检查、缩小架构的真实数据前反向及 eval loss、
双卡 Lightning/ZeRO-2 训练→验证→测试和 SwanLab 在线上传、pip check。
完整 checkpoint 已严格加载且全参数可训练；完整 5B 的 GPU 更新仍需等待 8 卡空闲后短跑确认。
验证记录位于 `../.setup/left_training_verified.json`。
