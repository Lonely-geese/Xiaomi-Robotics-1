# 数采数据约定

录制实现位于 `control/recording.py` 和 `control/recording_vendor/`。写盘采用 LeRobot v2.1 目录结构，包含 `meta/`、`data/`、`videos/`。`recap_recording.json` 固定模型身份、视角、动作移位和人工接管约定。

每条轨迹最少 30 帧。成功和失败都可以保存。左拨保存成功，右拨保存失败，下拨丢弃。当前实现中的成功标签来自操作者标注，应在后续评估时同时保留最终任务结果与人工接管信息。

| 内容 | 约定 |
|---|---|
| 相机 | 头相机、左腕相机、物理右腕相机原始视频 |
| `observation.state` | 30 维，左、右各 15 维 |
| 每臂状态 | 7 个关节角、XYZ、XYZW 四元数、夹爪开度 |
| 状态单位 | 关节角 degree、位置和夹爪 metre |
| `action` | 未来第 3 个保存帧的状态，保持与采集程序一致的布局 |
| 人工段 | `source=2`、`intervene_flag=True`，以 30 Hz 记录 |
| 策略段 | 按该段策略执行频率采样，保留原 HAL 时间戳与序列 |
| 接管 | 保留 `human_intervened`、`human_frames` 等元信息 |
| 结果 | 保留成功／失败及终止原因，人工协助结果单独可追溯 |

人工段仅裁掉首尾静止区间。动作已经按保存帧前移 3 帧，训练数据转换时需要读取这一约定。原始时钟、相机序列和人工段标记用于排查状态／动作错位及录制断点。

默认输出目录是本仓库下的 `rollout/recordings/usb_recap_epoch10_step15490`，也可设置 `RECAP_RECORD_OUT`。目录内已有 `recap_recording.json` 时，续写要求契约完全匹配。模型或采样规则改变后应选择新的输出目录。
