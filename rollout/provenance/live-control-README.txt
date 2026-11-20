USB RECAP epoch10 step15490：VR 手柄控制入口

HAL 本机隔离启动（2026-11-12）：
  bash /data/checkpoints/kele/usb/recap_robot_control/start_hal_isolated.sh
固定 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST、ROS_DOMAIN_ID=0，与本机相机节点一致。
不要直接继承 tmux 终端中的 SUBNET；/usr/bin/hal-server 会保留已有 discovery 配置。
本机同 domain 的节点仍可互相发现；该设置用于隔离局域网的其他机器。

2026-11-12：默认已接入 LeRobot v2.1 完整 rollout 录制。
数据目录：recordings/usb_recap_epoch10_step15490（续写，模型身份固定）。
Y 开始；摇杆上暂停，确认日志 HUMAN RECORDING READY 后在机械臂端手动开启物理拖动。
程序不自动启用硬件拖动模式。结束拖动时必须先退出物理拖动、松开机械臂：
  上：确认交还控制，停稳检查通过后从当前状态重新推理。
  左/右：标记并保存成功/失败。人工接管时，先退出拖动并松开机械臂，再拨左/右；通过停稳和模式检查后自动复位，无需再上推。
  下：终止丢弃；人工暂停期间同样需退出拖动再按上，才允许复位。
最少 30 帧；人工段按 30 Hz 记录并标 source=2/intervene_flag=True，首尾静止段会裁剪。
存储三路原始相机视频、30 维状态和前移 3 帧的动作，以及成功/人工介入标签。
存储的 rightCamera 是物理右腕；当前模型第三视角仍是左腕中心裁剪，后续训练应遵守模型视角配置。
正常结束写入 meta/episodes.jsonl；异常退出丢弃未完成轮次，已提交轮次等待编码写完。
离线验证数据放在 runs/recording_validation，不混入正式采集目录。

固定模型目录：
  /data/checkpoints/kele/usb/usb_recap_epoch10_step15490/model
使用该包的 mibot 和 USBClient，norm 固定为上述目录的 normalize.json。
RECAP：enabled=True、conditioning=language、guidance_scale=1.0。
原生推理端口 10090；机器人 Cartesian 适配端口 127.0.0.1:10094。
依照 USB1102 epoch20 的控制实现接入，不修改两个推理包。

左手柄操作（与参考实现相同）：
  Y：开始模型控制。
  摇杆上：暂停 / 从当前位置继续。
  摇杆左：标记成功，结束本轮并复位。
  摇杆右：标记失败，结束本轮并复位。
  摇杆下：终止本轮并回位。
  X：等待 Y 或人工暂停时切换左夹爪开/合，松手保持；模型运行和等待交还复位时忽略。
  食指扳机：已停用自动跟随，避免松开扳机意外张爪。
  进入接管保持当前夹爪开度；只有 X 才改变它。张开 0.09 m、闭合 0 m，仅向左夹爪发命令。
  X 按下沿触发，250 ms 去抖；VR 重连后先松开 X，再按才生效。
  人工暂停时的夹爪开合随 HUMAN 段录制，仅夹爪运动也会保留。
启动 VR 控制器会先复位双臂并打开双夹爪，再打印 READY 等待 Y。
停止/暂停沿用参考实现的指令排空、停止反馈确认和推理连接清理。

运行参数：
  异步策略动作播放 7.5 Hz，Cartesian 插值 60 Hz。
  每次输出 30 帧，只执行左臂动作；动作前缀长度 6。
  张爪每次执行最多 24 个新目标，闭爪最多 10 个。
  夹爪目标 <= 0.012 m 锁紧，连续 3 个目标高于阈值后释放锁定。
  头部原图、左腕原图、左腕中心 256x256 裁剪；物理右腕图不参与模型。
  推理不再额外前移 3 帧，不重复追加 Advantage: positive。

前台启动（必须先停止其他机器人控制器）：
  bash /data/checkpoints/kele/usb/recap_robot_control/start_robot.sh
只读检查，不获取控制权：
  bash /data/checkpoints/kele/usb/recap_robot_control/start_robot.sh --observe-infer-test
真实相机 + 6 帧动作前缀检查，不发送机器人动作：
  cd /data/checkpoints/kele/usb/recap_robot_control
  /home/dt/miniconda3/envs/pi05-lerobot/bin/python verify_observation.py

原生 epoch10 模型在后台独立运行（端口 10090）：
  PID 文件：../runs/usb_recap_epoch10_step15490_10090.pid
  日志：../runs/usb_recap_epoch10_step15490_10090.log
手柄相关进程由 systemd 用户服务管理（未设置开机自动启动）：
  usb-recap-robot-adapter.service：Cartesian 适配服务
  usb-recap-vr.service：VR 控制器
查看 VR 状态：systemctl --user status usb-recap-vr.service
停止 VR 控制：systemctl --user stop usb-recap-vr.service
停止后重新启动 VR：bash /data/checkpoints/kele/usb/recap_robot_control/start_vr_service.sh
这是临时服务，停止后 systemd 会回收 unit，需由上述脚本重新创建。
2026-11-12：VR 异常退出后等待 5 秒自动重启；5 分钟内最多允许启动 3 次，超过则停止重试。
自动重启仍会双臂复位、打开夹爪，然后等待 Y；不会自动继续上一轮推理。
手动 systemctl --user stop usb-recap-vr.service 不会触发自动重启。
已保存数据保留；异常中断的未完成轮次仍按原规则丢弃，不会自动保存或续接。
当前运行实例的持久配置：~/.config/systemd/user/usb-recap-vr.service.d/restart.conf。
触发频率限制后，排除故障并运行 start_vr_service.sh 可清除限制并重新启动。
重启模型服务后，也需要重启适配服务以重建原生连接。

日志：runs/robot.log、runs/adapter.log、runs/episodes.jsonl。
实时 VR 输入状态：runs/vr_input_status.json；robot.log 记录按键映射事件。
验证结果：verification.json。身份检查和夹爪回归检查：
  /home/dt/miniconda3/envs/pi05-lerobot/bin/python -m unittest -v test_identity test_gripper_snap
