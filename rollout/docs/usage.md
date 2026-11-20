# 部署与操作

## 环境准备

在 Linux 机器上准备完整模型包 `usb_recap_epoch10_step15490`。默认位置为 `/data/checkpoints/kele/usb/usb_recap_epoch10_step15490`，也可通过 `RECAP_PACKAGE` 指定。包内应包含 `model/config.py`、`model/normalize.json`、`model/last.ckpt/checkpoint/mp_rank_00_model_states.pt` 和 `assets/Qwen3-VL-4B-Instruct/`。固定模型身份见 `control/model_manifest.json`，启动时检查配置、归一化和权重。完整校验会读取约 10 GB 权重。

推理环境参考 `server/requirements.txt`、`server/requirements-flash-attn.txt` 和 `server/environment_versions.json`。控制环境安装 `control/requirements.txt`，并使用现场已安装的 DeepTouch HAL SDK、VRDevice 与 ADB。`XR1_TUOZHUAI_ROOT` 指向同时包含 `deeptouch_robot_hal` 和 `deeptouch_teleop` 的运行时目录。参考配置保留了 32.160 的实际 Python 和 SDK 路径，迁移到其他机器时需要填写本机路径。

```bash
cd /absolute/path/to/Xiaomi-Robotics-1
cp rollout/config/environment.example rollout/.env
# 编辑 rollout/.env，填写仓库绝对路径、两个 Python、模型包和 SDK 路径。
set -a
source rollout/.env
set +a
```

确认 `robot HAL` 已监听本机 7860 端口，相机画面有效。通过 ADB 确认 Quest 已授权且为 `device` 状态。控制端使用本机 Quest USB 转发到 40051。需要时设置 `DEEPTOUCH_ADB_PATH` 为现场 ADB 的绝对路径。

`config/vr_reset.yaml` 来自 32.160。迁移机器时先核对左右臂初始关节角、腰部配置与实际设备。当前启动流程会自动复位双臂并打开夹爪。

## 离线检查

```bash
"$RECAP_CONTROL_PYTHON" rollout/scripts/check_package.py --code-only
cd rollout/control
"$RECAP_CONTROL_PYTHON" -m unittest discover -p 'test_*.py'
cd ../..
bash rollout/scripts/start_control.sh --dry-run
```

`--code-only` 校验仓库代码和随仓库保存的模型配置。`--dry-run` 检查控制参数并退出。完整模型校验可执行 `"$RECAP_SERVER_PYTHON" rollout/scripts/check_package.py --full`。

## 前台启动

在三个已加载 `.env` 的终端中，依次执行以下命令。

```bash
# 终端 1，启动原生模型服务
bash rollout/scripts/start_server.sh

# 终端 2，等原生服务监听后启动协议适配
bash rollout/scripts/start_adapter.sh

# 终端 3，现场确认可以复位后启动控制与录制
bash rollout/scripts/start_control.sh
```

默认模型为 epoch10 / step15490，异步动作执行频率 7.5 Hz，模型预测 30 帧动作。夹爪使用现场的 12 mm 闭合判定及保持逻辑。需要核对实际执行窗口时，查看控制端启动日志中的 `open_execute_steps`、`closed_execute_steps` 与 `prefix_length`。

数据写到 `RECAP_RECORD_OUT`。复用已有目录时会检查模型身份与录制约定，并通过文件锁防止两个录制进程同时写入。

## VR 操作

| 输入 | 功能 |
|---|---|
| Y | 在 READY 状态开始一条轨迹 |
| X | READY 或人工暂停时切换左夹爪开／合，松手保持 |
| 摇杆上推 | 暂停策略并软停；人工接管结束后再次上推交回控制 |
| 摇杆左拨 | 标记成功、保存轨迹，然后复位 |
| 摇杆右拨 | 标记失败、保存轨迹，然后复位 |
| 摇杆下拨 | 丢弃本条并归位 |

上推暂停后，等待日志出现 `HUMAN RECORDING READY`，再在机械臂端手动开启物理拖动。交回控制或标注前，先退出物理拖动并松开机械臂。程序用新反馈确认左臂停稳和运动模式，再恢复策略或复位。

人工接管后直接左拨或右拨即可保存并复位。人工暂停期间下拨丢弃后，还需退出拖动并上推确认，才会复位。食指扳机自动跟随已停用，接管时保持现有夹爪开度，由 X 键切换。人工接管期间同样会记录夹爪操作。

## 用户服务

安装前先停用同一机器人上已有的 RECAP 控制入口，确认没有其他程序持有机器人控制权。服务文件中的控制端启动及自动重启均会执行复位。

```bash
mkdir -p ~/.config/systemd/user
cp rollout/systemd/*.service ~/.config/systemd/user/
cp rollout/.env ~/.config/recap-rollout.env
systemctl --user daemon-reload
# 现场确认允许复位后执行
systemctl --user start recap-rollout-control.service
journalctl --user -u recap-rollout-control.service -f
```

三个服务按原生推理、协议适配、控制的顺序启动。异常退出后等待 5 秒重试，300 秒内最多启动 3 次。检查失败后先查明原因，再执行 `systemctl --user reset-failed` 和启动命令。

```bash
# 停止控制，等待软停及录制收尾
systemctl --user stop recap-rollout-control.service
# 停止推理链路
systemctl --user stop recap-rollout-adapter.service recap-rollout-server.service
```

## 排查

- Y 无反应时，检查 `control/runs/vr_input_status.json`、ADB 授权和 Quest 应用是否已连接。
- 身份校验失败时，检查模型包是否为 step15490，以及归一化、视角和 RECAP guidance 是否与固定身份匹配。
- 端口 10094 未监听时，先检查 10090 原生服务及适配服务日志。
- 交还控制失败时，确认已退出硬件拖动模式、松开机械臂，并查看停稳检查日志。
- 录制未保存时，检查是否满 30 帧、是否拨了下方向、磁盘空间，以及输出目录的模型契约和写锁。
