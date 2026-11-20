# XR1 离线强化学习

在仓库根目录运行。复制 [STEAM 配置](configs/steam.json) 或 [RECAP 配置](configs/recap.json)，填写数据、视觉与语言骨干、初始策略权重和独立的 `run_dir`。相对路径以仓库根目录为基准，机器上的绝对路径也可以直接使用。示例不附带数据和权重。

```bash
python -m rl --config rl/configs/steam.json --dry-run
python -m rl --config rl/configs/steam.json
python -m rl --config rl/configs/recap.json --stage value
python -m rl --config rl/configs/recap.json --stage label
python -m rl --config rl/configs/recap.json --stage policy
```

省略 `--stage` 会按 value、label、policy 顺序执行。每阶段检查产物后才进入下一阶段。运行环境是 Linux 和对应方法的 XR1 训练依赖。RECAP 策略使用仓库的 `.venv` 和原有 8 卡、全局 batch 256 的启动器。STEAM 使用调用入口的 Python 和配置的卡数，全局 batch 必须能被每卡 batch × 卡数整除。

| 阶段 | STEAM | RECAP |
| --- | --- | --- |
| 准备输入 | `manifest` 中每条轨迹须有图像、指令、已对齐的状态和动作，以及策略归一化配置；从价值训练起三阶段共用同一 manifest | `prepared_data` 需要准备好的 SFT、rollout 池、归一化文件和价值训练元数据 |
| value | 时间进展 ensemble，输出到 `run_dir/value`，选取训练完成的检查点 | 原生 RLinf 价值网络，优势阶段读取 `run_dir/value/best.pt` |
| label | 使用训练集分位数，写入 `run_dir/advantages.parquet` 和校验元数据 | 基于多步 TD 打标，在准备好的数据目录写入逐帧 sidecar 和 `advantages_ready.json`；每个实验使用独立的数据输出目录 |
| policy | 从配置的 SFT 权重微调，保留优势语言条件与异步动作前缀，输出 `run_dir/policy` | 使用语言条件、原始动作损失及异步前缀，输出 `xr1/outputs/project_xiaomi-robotics-1/<experiment>` |

同一配置重跑时，入口检查并跳过已完成的阶段；STEAM 的未完成训练从最新完整检查点续跑。RECAP 价值网络使用原生恢复机制，策略中断后按原启动器要求设置 `RECAP_RESUME` 为含 `restart_ready.json` 的检查点目录。更换输入或超参数时使用新的运行目录，RECAP 也需更换策略实验名。

数据准备与方法细节见 [RECAP 说明](../xr1/docs/recap.md) 和 [STEAM 说明](../xr1/docs/steam.md)。采集和真机推理见 [rollout 手册](../rollout/README.md)。STEAM 也提供 [离线策略推理](../xr1/tools/infer_steam_policy.py)，支持输入已排队动作前缀；真机手册对应其列出的 RECAP 模型包。
