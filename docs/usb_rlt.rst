USB 小米 XR1 的 RLToken 适配
===========================

代码与验证范围
--------------

此适配以 USB 10 epoch / step 12050 的 XR1 权重为基础，复用 RLinf
commit ``30e0236981c6a1dc19202de64116834c12673828`` 的 RLToken encoder/decoder、
RLTMLPPolicy 和 twin-Q 网络。上游代码保存在 ``xr1/mibot/rlt/upstream``，仅改
Python 导入路径，逐文件源代码哈希和许可证一并保留。第二阶段损失与 RLinf
``RLTACLossMixin`` 的默认 Q-head 路径有源码数值对照测试。

适配提供单卡与 8 卡 DDP/ZeRO-1 训练入口，尚未接入 RLinf 的 Ray/FSDP worker
调度、真机控制器、奖励反馈或人工接管设备。第一阶段已于 2026-10-30 完成
step 20000；第二阶段
尚未启动。接口检查所用随机 actor 不代表训练完成的策略。

原权重包提供 USBSpeedModel 的配置但未包含该优化类的 Python 源码。本适配
使用本地 XR1 标准模型类严格加载全部权重；参考动作与同一标准 XR1 原生推理
在固定随机种子下逐值一致。这个检查不等于与未提供的 USBSpeedModel 服务做过
端到端对照。

数据与动作约定
--------------

默认配置为 ``xr1/configs/usb_rlt.yaml``，使用：

* 数据：``/data/wz/usb/usb``，1041 条成功轨迹，308545 帧，3123 个视频。
* 权重：``/data/wz/usb/usb_official_epoch10_step12050_weights_norm/mp_rank_00_model_states.pt``。
* norm：同目录 ``normalize.json``，SHA256 为
  ``9ee242381ea2a1a0c03951e1b7d6619ff5dcc3767fece219f96eaa26ee299b4d``。
* 三路图像依次为头部、左腕、右腕，保留原 XR1 图像缩放及指令模板。

通过 manifest 读取 Parquet 和 H.265 视频，不要求标准 LeRobot 路径模板。
视频采用 PyAV 单线程解码，并禁用 Python 日志回调，避免 HEVC 解码器退出时
的线程/GIL 互等；按精确 PTS 取帧，找不到目标帧时报错。
源数据 action 已在原始时间轴前移 3 帧，不再重复移位。保留数据提供的静止帧
删除结果，窗口可以跨静止片段删除边界，但不跨 episode。尾部动作重复末帧并
使用原 XR1 temporal mask；不重新计算 norm。

源 30D 状态包含双臂关节角、末端 xyz/XYZW 四元数和夹爪。关节角从度转弧度，
按 XR1 的 60D 格式排列并用原 q01/q99 归一化；第二阶段 proprio 取有效的
前 16 维双臂关节/夹爪，去掉 44 个填充零。

模型有效动作维度为 0..6：左臂局部末端位移 3D、局部轴角增量 3D、夹爪增量 1D。
它们都是相对于动作窗口开始时的状态，不是逐步累积的增量，也不是关节角。
VLA 预测 30 步，第二阶段默认执行前 10 步；这只是可配置窗口，尚未验证真机
闭环频率。动作每个 offset 使用原 norm 的对应行。

XR1 使用无界 z-score 动作，RLinf actor 输出 tanh。适配采用显式可逆坐标：
``u = tanh(z / 5)``，``z = 5 * atanh(u)``。参考动作、实际执行动作和人工接管
动作都必须进入相同的 u 坐标；保存配置与权重绑定。遇到非有限数或浮点饱和
到 ±1 会报错，不做静默裁剪。该映射改变 BC 所在的坐标尺度，是 Xiaomi 适配
与原 Pi0.5 归一化的区别，不是对物理动作的限幅。

``recover_left_targets`` 恢复左臂绝对位置、旋转矩阵和夹爪目标，坐标系与数据
中的末端位姿一致；不输出右臂控制指令。``encode_executed_left`` 将实际发送
的绝对目标反变换为 replay 动作。原 XR1 浮点轴角转换在近零转角处存在约
1e-4 的旋转矩阵重建误差，沿用其数学实现以保持训练语义。

第一阶段
--------

Qwen3-VL 的最后层观测 token 为 2560D，RLT 内部为 2048D，encoder/decoder
各 2 层、8 heads，与所引用 RLinf 默认配置一致。最大 prefix 长度 2048，
当前样本为 506 token。保存全部视觉和语言 token 的 mask，不读取 action、
score 或 state-target 训练回合。RLT 重建目标 detach，RLT 网络保持 fp32。训练中的动作专家主权重和 AdamW
状态也保持 fp32，前向使用 bf16 autocast，避免小学习率更新被 bf16 舍入吞掉。

默认 ``mode: joint`` 冻结 VLM，联合训练 XR1 动作专家/choice/score 投影和
RLT，目标为 ``rlt_loss + alpha * vla_loss``，alpha=1。
VLA loss 使用原 XR1 的 flow、频域和 choice/score 目标，而不是替换成 Pi0.5
的动作预测损失。``mode: rlt_only`` 则冻结整份 XR1，只训练压缩模块。

单卡参考配置为 micro batch=1、累积 16 次、20000 optimizer steps。
已完成的训练使用下面的八卡配置。单卡 launcher 不接受 torchrun，八卡使用独立入口。

单卡参考启动命令：

.. code-block:: bash

   cd /data/wz/Xiaomi-Robotics-1/xr1
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ../.venv/bin/python tools/train_usb_rlt.py \
     --config configs/usb_rlt.yaml \
     --output /data/wz/usb/rlt_runs/stage1

只训练压缩模块可添加 ``--mode rlt_only``。断点继续使用相同 config、output，
添加 ``--resume /data/wz/usb/rlt_runs/stage1/latest.pt``。恢复模型、优化器、学习率
进度和随机状态，但会开始新的 shuffled pass，不承诺恢复精确的 mid-epoch
样本次序。checkpoint 包含 RLT、所有非 VLM 权重和原权重/norm 的 SHA256，
加载时严格核对。冻结的 VLM 从原始权重读取，因此原始权重不能删除。

第二阶段
--------

冻结第一阶段模型，输出 ``z_rl [B,2048]``、``proprio [B,16]``、
``ref_chunk [B,30,7]``。RLTMLPPolicy 读取前 10 步参考动作，actor 和 twin-Q
均采用原 RLinf 实现。actor 是直接预测动作的策略，不是 BEE 的残差模型。

* Critic target：折扣的 K 步真实奖励 + gamma^K * target min(Q1,Q2)。
* 后继动作用当前 actor 采样，与 RLinf 默认路径一致。
* Actor：``-0.1 * Q1 + 5 * BC``，参考动作 dropout=0.5。
* 人工接管时，该时间步 BC target 使用实际执行动作，否则使用 VLA 参考动作。
* gamma=0.96，tau=0.005，固定高斯 std=0.002，不使用熵奖励。
* 默认每 4 次 critic 更新做 1 次 actor 更新；目标参数每次更新。

离线训练入口接受显式收集的特征 transition 文件，不能直接输入原 USB
成功示教目录。文件由 ``torch.save`` 保存以下对象：

.. code-block:: python

   {
       "format": "xr1-rlt-replay-v1",
       "assets": feature_assets(config, stage1_checkpoint),
       "transitions": {
           "curr_obs": {"z_rl": ..., "proprio": ..., "ref_chunk": ...},
           "next_obs": {"z_rl": ..., "proprio": ..., "ref_chunk": ...},
           "actions": ...,          # float32 [N,K,7]，实际执行动作的 u 坐标
           "rewards": ...,          # float32 [N,K]，显式外部奖励
           "terminations": ...,     # bool [N,K]，真实任务终止
           "truncations": ...,      # bool [N,K]，时间上限等截断
           "intervene_flags": ...,  # bool [N,K,7]，实际接管标记
       },
   }

``feature_assets`` 来自 ``mibot.rlt.runtime``，绑定第一阶段 checkpoint hash、
norm hash 和动作/特征维度。next_obs 必须来自同一 episode 的动作执行后观测，
不能用 reset 后的图像替代。当前入口仅支持完整 K 步 chunk，提前结束的短
chunk 必须由后续 collector 单独适配；校验会拒绝终止后填充或跨 reset 的
transition。真实终止停止 bootstrap，时间截断保留 bootstrap。

训练命令（需先有真实奖励及上述 replay，本次未执行）：

.. code-block:: bash

   cd /data/wz/Xiaomi-Robotics-1/xr1
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ../.venv/bin/python tools/train_usb_rlt_stage2.py \
     --stage1 /data/wz/usb/rlt_runs/stage1/latest.pt \
     --replay /path/to/collected_transitions.pt \
     --output /data/wz/usb/rlt_runs/stage2 --updates 1000

训练入口默认 batch=256。当前未实现在线 replay 采集、关键阶段自动切换、真机
奖励和 SpaceMouse 接管；这些都依赖用户的机器人/仿真接口，不能从成功示教
文件推断。

离线推理与检查
--------------

.. code-block:: bash

   cd /data/wz/Xiaomi-Robotics-1/xr1
   PYTHONPATH=. ../.venv/bin/python tools/check_usb_rlt.py \
     --output /data/wz/usb/rlt_checks/data.json
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ../.venv/bin/python tools/check_usb_rlt.py \
     --gpu-smoke --checkpoint-roundtrip \
     --output /data/wz/usb/rlt_checks/integration.json
   OMP_NUM_THREADS=4 PYTHONPATH=. ../.venv/bin/python -m pytest tests/test_usb_rlt.py -q

   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ../.venv/bin/python tools/infer_usb_rlt.py \
     --stage1 /data/wz/usb/rlt_runs/stage1/latest.pt \
     --episode 0 --frame 0 --output /data/wz/usb/rlt_prediction.pt

推理默认使用第一阶段 VLA 参考动作。使用 actor 必须同时提供训练好的
``--stage2 .../latest.pt --actor``；缺少 checkpoint 会拒绝启用随机 actor。
推理脚本只保存目标与特征，不发送机器人控制命令。

在线观测预处理可使用 ``ObservationCollator.from_raw(images, raw_state, norm, prompt)``，
其中 images 的键为三路相机名称，值为 RGB 数组或 PIL 图像，raw_state 保持
USB 30D 格式和度制关节角。该入口不需要 action/score 标签，其 token、图像
和状态与离线数据入口有逐值一致性测试。


八卡训练与速度优化
------------------

当前配置为 ``xr1/configs/usb_rlt_8gpu.yaml``：8 张 H100、每卡实际 batch=32、
梯度累积=1、全局 batch=256。通过 DistributedSampler 分片并丢弃不足整批的
尾部，模型入口校验实际 batch，日志记录全局计数。总步数保留 20000；从
单卡 step 776（已看 12416 个样本）恢复，每个后续更新读取 256 个样本。

优化包括：

* 冻结 VLM 的观测前缀在 no_grad 下计算，含 state/action/score 的后缀仍保留
  反向传播。后缀使用独立 KV cache，动作专家只读取观测前缀 cache。
* 使用 dense batch，保留每条轨迹的 attention mask、位置编码、标签和尾部 mask。
* RLT/动作专家主权重与 Adam 状态为 fp32，RLT 矩阵算子采用 bf16 autocast，
  重建损失仍以 fp32 计算；按层重计算控制显存。原 training_repeat=4、异步前缀
  训练、动作窗口和所有损失项保持原配置。
* DDP 同步梯度，ZeRO-1 分摊 Adam 状态，fused AdamW 更新。
* H.265 视频预先解码到无损 uint8 NPY，并按原 PIL 规则缩放。读取随机帧不再
  重复解码视频，原图像增强仍每次随机执行。缓存完整包含 3123 个视频；
  18 个抽查帧与原读取/缩放结果逐像素相同。
* 检查点汇总用 frombuffer 读取序列化数据，避免 PyTorch ByteTensor(bytearray)
  对数 GB 数据逐字节转换。两进程 CPU 验证中，参数和 Adam 状态与单进程
  参考结果逐值相同。此替换只作用于训练进程，不修改已安装的 PyTorch 文件。

由于 dense/分段注意力选择不同的 bf16 内核，优化不承诺逐比特数值等价。
真实两样本对照中总损失约 3.1515 -> 3.1300，状态投影梯度余弦相似度约
0.99985、相对 L2 误差约 16.9%。数学目标不变，但数值路径有区别，效果仍应
用训练曲线及后续任务评估判断。详细检查保存在
``/data/wz/usb/rlt_runs/migration/benchmark.json`` 和 ``cache_parity.json``。

.. code-block:: bash

   cd /data/wz/Xiaomi-Robotics-1
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=xr1 \
     OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
     .venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     xr1/tools/train_usb_rlt_distributed.py \
     --config xr1/configs/usb_rlt_8gpu.yaml \
     --resume /data/wz/usb/rlt_runs/stage1_joint_20261029_214008/latest.pt \
     --output /data/wz/usb/rlt_runs/new_8gpu_run

现有活动进程、日志和输出目录记录在 ``/data/wz/usb/rlt_runs/current_run.json``。
常规迭代计时包含取数、前反向、八卡同步、梯度裁剪和优化器更新，取最慢 rank；
启动与保存检查点的耗时单独记录。保存间隔为 500 步，切换后的第 10 个更新
额外保存一次。``latest.pt`` 为包含全模型适配权重及完整优化器状态的原子替换
检查点，可继续八卡训练。恢复不承诺逐样本随机次序完全一致。

在活动输出目录创建 ``SAVE`` 文件可请求下一迭代边界保存；创建 ``STOP``
则保存后正常停止。也可向训练 rank 发送 SIGUSR1 请求保存退出。不要向
不支持该信号的旧单卡进程使用此方法。
