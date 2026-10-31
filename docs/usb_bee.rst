USB Xiaomi + RLToken 离线 BEE
============================

基础与范围
----------

``xr1/configs/usb_bee.yaml`` 默认加载 2026-10-30 完成的 Stage 1 step 20000：
``/data/wz/usb/rlt_runs/stage1_joint_8gpu_bs256_20261029_235047/latest.pt``。
Xiaomi VLM、动作专家和 RLToken 全部冻结；原始基础权重和 norm 仍需保留。
所有入口只读数据/保存文件，没有机器人控制接口。

实现依据 `BEE 论文 <https://arxiv.org/pdf/2609.27450>`_ 公式 (1)--(9) 和
表 III。本适配使用 XR1、2 层/8 heads 的 RLToken，而论文是 π0.5 和
4 层/16 heads 的提取器。因此是 USB 适配，不是论文任务成功率的复现。

接口与数学实现
--------------

状态为 2048D RLToken + 16D proprio；参考动作取原 VLA 30 步预测的前 10 步，
形状 [N,10,7]。7 维是左臂局部末端位移、局部轴角、夹爪增量，全部相对
chunk 开始状态，不能逐步积分。BEE 使用原 norm 的逐 offset z-score，
不经过 RLT 的 tanh(z/5)，不套用 LIBERO 的 [-1,1] 限幅。

残差 actor 零输出初始化，默认只编辑前 6 维位姿，夹爪跟随 VLA 建议。
人工执行的夹爪动作仍保留在 replay 中供 critic 学习。物理坐标恢复复用
``recover_left_targets``；实际发送的目标使用
``encode_executed_left(..., action_coordinates="zscore")`` 编码。

``mibot/bee/core.py`` 实现：

* 高斯纠正模型预测残差均值和逐维方差，方差为 softplus + 0.02。
  NLL 仅监督明确有纠正标签的可编辑时间/维度。
* Twin-Q 使用 K 步折扣奖励、当前策略后继动作、target min(Q1,Q2)。
  真实终止停止 bootstrap，时间截断保留 bootstrap。
* rho 为方差加权的残差偏离，非可编辑维度贡献零，按完整 K*7 归一化。
* actor 目标为 ``(-minQ + eta*sum(residual^2) + lambda*rho)/(1+lambda)``。
* 状态相关 softplus 乘子最小化 ``-lambda*(rho-epsilon)``。
* actor 更新冻结 critic 参数并 detach 纠正分布/乘子；dual 更新 detach
  actor/纠正分布；纠正模型只由独立 NLL 更新。

默认 MLP 为 3 层/512 宽，batch=256，gamma=0.99，tau=0.005，Adam
(0.9,0.95)，主学习率 3e-4，乘子 3e-6，eta=1，epsilon=0.3，参考输入
dropout=0.5，每两次 critic 更新一次 actor。dropout 不改变残差相加的原建议。

现有示教的一键预热
------------------

.. code-block:: bash

   cd /data/wz/Xiaomi-Robotics-1
   CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=xr1 \
     .venv/bin/python xr1/tools/prewarm_usb_bee.py \
     --output /data/wz/usb/bee_runs/prewarm_run

默认导出所有 1041 条轨迹，每 10 帧一个完整 10 步窗口。按 episode 用固定
种子划分约 90% 训练、10% 验证，同一轨迹不跨组。每批 8 个样本推理，
每 128 个样本原子保存 feature shard，复用已有无损图像缓存。
仅训练纠正模型，最多 5 epoch，验证 NLL 连续两轮不改善则停止。
``best.pt`` 由验证 NLL 选择，``latest.pt`` 保存最新完整 epoch。

``status.json`` 记录阶段及导出速度；``metrics.jsonl`` 记录训练 NLL、验证
NLL、残差 MSE、零修正 MSE 和方差。NLL 省略常数项，允许负值。这些指标
衡量示教配对拟合，不能代替真实 USB 成功率评测。

原命令附加 ``--resume`` 恢复：验证并跳过完整 shard，中断的 shard 重做，
每个 shard 独立设随机种子；训练从上一个完整 epoch 恢复。配置、数据哈希
和划分必须一致。``step=0``、``correction_steps>0`` 表示只训练纠正模型，
actor/critic 仍未训练；推理入口拒绝将其作为训练好的 actor 使用。

2026-10-30 已完成的离线预热
--------------------------

本机运行 ``/data/wz/usb/bee_runs/correction_prewarm_20261030_115848`` 已完成：
937 条训练轨迹、104 条验证轨迹，共 30390 个配对窗口，其中训练 27341、
验证 3049。训练 5 epoch、535 次纠正模型更新，最优 checkpoint 为第 5 epoch
的 ``best.pt``，actor 未训练。

验证 NLL 为 -1.8072166，纠正后动作 MSE 为 0.0059893，零纠正（直接保持 VLA
建议）的 MSE 为 0.0059021，平均方差为 0.0201993，接近配置下限 0.02。
因此，此结果确认流程可运行，但没有证明纠正带来动作误差或任务成功率改善。
下一步仍需要真实接管数据校准，以及独立的任务评测。权重和数据保存在本机，
不包含在代码仓库中。

示教配对标记为 ``demonstration_pairs``，不是实际接管事件。程序不会从成功
示教补造 rollout 奖励或接管标记。该预热分布与实际接管分布不同，后续需要
真实接管继续校准。

小样本独立入口：

.. code-block:: bash

   PYTHONPATH=xr1 .venv/bin/python xr1/tools/prepare_usb_bee_demos.py \
     --episodes 0 1 --max-samples 64 --output /path/to/demo_pairs.pt
   PYTHONPATH=xr1 .venv/bin/python xr1/tools/train_usb_bee.py \
     --mode correction --data /path/to/demo_pairs.pt \
     --output /path/to/correction_run --updates 500

完整离线强化学习数据
--------------------

必须有真实执行后的奖励和后继观测。采集端使用 torch.save 保存以下字典；
所有浮点张量为 float32，标记为 bool。assets 由
``mibot.bee.data.feature_assets(base_config, stage1_path, bee_cfg)`` 生成，绑定
基础权重、Stage 1、norm、维度和坐标。

.. code-block:: python

   {
       "format": "xr1-bee-replay-v1",
       "source": "collected_transitions",
       "assets": assets,
       "transitions": {
           "curr_obs": {
               "z_rl": ...,       # [N,2048]
               "proprio": ...,    # [N,16]
               "ref_chunk": ...,  # [N,10,7] 原模型建议，z-score
           },
           "next_obs": {...},     # 执行后、reset 前，同样的形状
           "actions": ...,        # [N,10,7] 实际执行动作，z-score
           "rewards": ...,        # [N,10] 真实外部奖励
           "terminations": ...,   # [N,10] 真实终止
           "truncations": ...,    # [N,10] 时间上限等截断
           "intervene_flags": ...,# [N,10,7] 实际接管时间/维度
       },
   }

原建议必须保留，不能被人工动作覆盖。当前只接受完整 K 步 chunk，终止/
截断只能在最后一步；提前结束后填充、跨 reset、缺字段或非有限值会被拒绝。
形状校验不能证明数据来源，采集端负责时序和执行值真实性。旧 RLT replay
使用有界坐标，不能只修改格式字符串就作为 BEE 输入。

训练从全部 replay BR 与人工接管子集 BH 各采样一半，BR 包含 BH。
纠正模型只使用 BH 中确有可编辑维度纠正的样本，仅监督已标记维度。

.. code-block:: bash

   PYTHONPATH=xr1 .venv/bin/python xr1/tools/train_usb_bee.py \
     --mode offline-rl --data /path/to/real_collected_replay.pt \
     --initialize-correction /data/wz/usb/bee_runs/prewarm_run/best.pt \
     --output /path/to/offline_bee_run --updates 1000

默认先用真实接管做 500 次纠正更新，然后做指定的额外 actor/critic 更新。
随后每 100 次 critic 更新做 12 次纠正更新。这是固定 replay 的离线调度，
不是论文的“每 100 个新接管样本”在线调度。在线采集尚未实现。
恢复时用 ``--resume .../latest.pt``，保持数据/调度一致，不同时传
``--initialize-correction``。

离线推理与检查
--------------

.. code-block:: bash

   PYTHONPATH=xr1 .venv/bin/python xr1/tools/infer_usb_bee.py \
     --episode 0 --frame 0 --output /path/to/base_prediction.pt
   PYTHONPATH=xr1 .venv/bin/python xr1/tools/infer_usb_bee.py \
     --bee /path/to/offline_bee_run/latest.pt \
     --episode 0 --frame 0 --output /path/to/bee_prediction.pt
   OMP_NUM_THREADS=4 PYTHONPATH=xr1 .venv/bin/python -m pytest \
     xr1/tests/test_usb_bee.py xr1/tests/test_usb_rlt.py -q

不提供 --bee 则只用冻结 VLA；提供时必须已有 actor 更新，且权重/norm
严格匹配。结果保存物理目标，不连接控制器。测试覆盖损失公式、梯度隔离、
乘子方向、接管掩码、1:1 replay、时间截断、格式拒绝、坐标恢复和精确续训。
真实 step 20000 权重与 USB 样本诊断保存在 ``/data/wz/usb/bee_checks``。
