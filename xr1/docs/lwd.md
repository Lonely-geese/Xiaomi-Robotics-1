# XR-1 + DIVL + QAM

`LWD` 分支在 XR-1 上实现 LWD 的价值学习和策略更新路径。Actor 使用本仓库的
Qwen3-VL + DiT，替换论文的 π0.5。原有 SFT 入口仍然可用。

这是依据 [LWD 论文 v4](https://arxiv.org/html/2605.00416v4) 和
[QAM 作者实现](https://github.com/ColinQiyangLi/qam) 编写的实现，**不是智元官方源码发布**，
也没有复现论文成功率。单设备 learner、离线训练及在线数据快照混合更新已接通；
机器人控制、自动发布和分布式车队回放服务不在本实现范围内。

## 代码路径

| 文件 | 职责 |
| --- | --- |
| `mibot/rl/backbone.py` | 独立的 Gemma 3 270M + SigLIP So400M 状态编码器，视觉投影、机器人状态 token、末尾 readout token |
| `mibot/rl/divl.py` | 共享编码器的分布价值 V、带时序注意力池化的双标量 Q、投影和分位数 TD |
| `mibot/rl/qam.py` | 参考流采样、终点动作梯度、反向 VJP、局部策略回归 |
| `mibot/rl/actor.py` | XR-1 流速度接口及离线/在线冻结规则 |
| `mibot/rl/learner.py` | 两条优化器路径、EMA、固定 SFT 参考策略、恢复训练 |
| `mibot/rl/data.py` | 本仓库 JSON 数据、显式结果标签、双编码器预处理 |
| `mibot/rl/replay.py` | n-chunk return 和 transition 校验 |
| `tools/train_lwd.py` | 单 GPU 训练入口及 Actor 导出 |

## 更新规则

V 的 201 个类别覆盖 [-0.1, 1.1]，拟合回放动作的 EMA min-Q 分布。
标量目标投影到相邻两个类别，用交叉熵更新 V。双 Q 回归到折扣奖励与
下一状态自适应 V 分位数之和，真实终止时不 bootstrap。折扣指数使用实际执行帧数，
包括尾部不足 30 帧的动作块；n-step 回报按原始帧累积。

QAM 保持 SFT Actor 为固定参考，使用 min-Q 对终点动作的梯度初始化伴随状态，
反向计算 VJP 后生成 detached 的局部回归目标。策略损失只更新当前 Actor。
不使用 SFT 的候选动作评分头充当 critic，不把 QAM 换成 AWR 或直接对多步采样反传。
噪声、Q 输入、伴随梯度和局部损失均使用有效动作掩码。

离线阶段更新 Actor 的 VLM 和动作专家；在线阶段冻结 Actor VLM，更新动作专家。
两阶段均更新独立的 V/Q 编码器。原始 SFT 参考模型始终固定，EMA 更新只作用于 V/Q。
检查点包含参考模型，在线切换或恢复时不会把改进后的 Actor 当成新参考。

主要默认值：γ=0.9999，EMA=0.005，λ=2，离线/在线基础 τ=0.6/0.9，
熵系数 0.3，Actor AdamW 学习率 2e-5，V/Q Adam 学习率 5e-4，cosine 调度。
短任务离线使用 1 chunk，长任务可设 10；在线所有来源均使用 1 chunk，按样本数 1:1
混合离线和在线回放。这里的 1:1 指数据来源，不等于示教/成功/失败的固定比例。

## 数值实现与原文的差异

LWD 式 (9) 后的 σ 写为 `sqrt(2*(1-t)*t)`；QAM 作者代码使用稳定化的
`sqrt(2*(1-t+h)/(t+h))`。本实现采用后者及其反向 VJP 时间偏移、最后一步 ODE，
局部损失也按作者代码对时间求和，而非乘 h 的积分平均。
采样路径和 VJP 都使用 LWD 固定的 SFT 参考；终点使用双 Q 最小值。
这是明确的数值实现选择，不能视为已验证与智元内部代码逐项一致。

论文未完整给出分位数裁剪边界、价值头隐藏结构、读出 token 布置等细节。
本实现将 τ 边界设为 [0,1]，使用带可学习位置的时间注意力池化、两层 Q MLP、
串接视觉/文本/机器人状态/读出 token，并将这些部分视为工程选择。
5 步流求解沿用 XR-1；梯度裁剪 1.0 和 Actor weight decay 0.01 也可配置。
按论文附录对整个 V/Q 分支做 EMA，并用目标 V 构建 bootstrap。

## 数据与标签

`offline_data` 沿用 `JsonDataset` 配置：`paths` 必须指向 episode JSON 文件或仅包含
episode JSON 的目录，提供同一个 `normalization` 和 `active_action_dims`。
动作仍使用 XR-1 当前工具坐标系的相对动作定义，归一化到 60 维统一布局。
JSON 的 `instruction.general` 和视频路径必须有效。

必须提供结果清单；不能将剪辑片段末尾自动视为任务成功。示例 `outcomes.json`：

```json
{
  "/absolute/path/episode_success.json": {"terminated": true, "success": true},
  "/absolute/path/episode_failure.json": {"terminated": true, "success": false},
  "/absolute/path/episode_cut.json": {"terminated": false, "success": false}
}
```

`terminated=true` 表示真实任务终止；成功时只在最后一个动作给 1，其余为 0。
超时或录制中断用 `terminated=false`，末帧保留为 bootstrap observation，不使用末帧动作。
必须核对原始标注语义，尤其是分段导出的成功示教；清单不能仅从文件名生成。
数据读取不筛掉失败动作。当前/下一状态均去除 SFT 的 action/score 目标 token，防止标签泄漏。

### LFD / RECAP 清洗

`tools/prepare_lwd_pools.py` 生成独立数据池，按原始 session + episode 去重，
优先保留 canonical LFD。指定的夹爪异常片段进入 `quarantine/gripper`，无结果标签进入
`quarantine/unlabeled`，均不出现在训练清单中。被上游删除的 episode 不恢复。
输入已具有 `action[t] = state[t+3]` 对齐，转换只处理坐标表示和关节单位，不再移帧。
JSON / NPZ 在输出目录中独立保存，视频引用原始绝对路径，因此不能移走原始视频池。

```bash
PYTHONPATH=xr1 python xr1/tools/prepare_lwd_pools.py \
  --gripper-flags /data/wz/analysis/lwd_data_audit_20261021/gripper_review_episode_ids.json \
  --ending-review /data/wz/analysis/lwd_ending_review_20261021/review.json \
  --output /data/wz/data/LWD/cleaned_20261021
```

清洗需要 `pyarrow` 和 `av`；输出必须是新建或空目录。生成 `manifest.json`、
`outcomes.json`、训练和验证路径清单，以及 `lwd_offline.json`。
LFD 按整个来源 session 划分，RECAP 按整个来源 episode 划分；验证占比按分组计算，
不保证 episode 数恰为 10%。归一化沿用现有 SFT 数据的 `normalize.json`，需与 Actor
检查点匹配；这是已有归一化约定，并非只用新训练划分重新估计的独立评测协议。

2026-10-21 版本保留 1,596 条 LFD、575 条 RECAP 成功、170 条 RECAP 失败结果记录。
额外数据中 1,754 个异常片段隔离，379 个未隔离的重复片段不进入新池；异常片段中另有
885 个来源已有 canonical 替代版本，正常替代版本保留。10 条无结果标签记录隔离。

170 条 failure 已检查双视角的结尾抽帧及轨迹统计，按可见行为分目录，证据见
`ending_review.json`。这些画面无法可靠区分真实终止与超时/人工停止，因此其训练标签为
`{"terminated": false, "success": false, "end_status": "unknown", "usable_frames": N}`。
其中 false 不代表已确认为超时：读取器先限制到 `N` 帧，再仅保留完整 n-chunk 窗口，
且 bootstrap 状态严格早于保留区间最后一帧。当前 `N` 排除末尾检查的 121 帧，避免
把含撤离/复位的模糊结尾用作训练目标；完整数据仍保留供审查。未知结尾的奖励不补造。
若日后有可信结束原因，可使用 `end_status=failure` 或 `truncated`，移除 `usable_frames`。

生成配置已填数据路径；训练前仍需填 Actor VLM 和 SFT 权重路径，并准备 CUDA 环境。
历史 RECAP 在本次配置中作为离线回放；在线阶段需要新策略采集的数据。

底层 `LWDLearner.update(batch)` 也支持其他回放服务，输入为：

```text
actor_observation:       XR-1 inference processor tensors + state [B,1,60]
critic_observation:      input_ids, attention_mask, pixel_values [B,V,3,384,384], state
next_critic_observation: 同上，实际 n-chunk 后的状态
action:                 [B,30,60]，归一化回放动作
action_mask:            [B,30,60]，二值有效维度和真实帧掩码
reward:                 [B]，已经按 γ 折扣的 n-chunk 累积回报
steps:                  [B]，实际跨越的原始动作帧数
terminated:             [B]，真实任务终止标志
```

## 启动

先按仓库 `scripts/setup_env.sh` 安装 CUDA 训练依赖，准备 XR-1 SFT 权重及 Qwen3-VL、
Gemma、SigLIP 模型。Gemma 下载可能需要账户许可；离线环境使用本地模型目录。
当前开发验证的 `.venv` 安装的是 **CPU PyTorch 2.8.0**，仅用于单元测试，不能直接训练完整 XR-1。

修改 `configs/lwd.json` 中的模型、数据、标签路径和动作维度，再从仓库根目录运行：

```bash
PYTHONPATH=xr1 python xr1/tools/train_lwd.py --config xr1/configs/lwd.json
```

SFT 权重支持原始模型 state_dict，以及 `module` / `state_dict` 中带 `model.` 前缀的格式，
严格校验全部参数。`latest.pt` 包含完整训练状态；`actor.pt` 为现有 BaseRunner 可载入的 Actor。
输出文件先写临时路径再替换，防止部署侧读取半写入的文件。

恢复同一阶段：

```bash
PYTHONPATH=xr1 python xr1/tools/train_lwd.py --config xr1/configs/lwd.json --resume outputs/lwd/latest.pt
```

进入在线阶段：复制配置，设 `lwd.stage=online`，补 `online_data` 和 `online_outcomes`，
确保两类数据的归一化、动作长度、有效维度配置一致，batch_size 为正偶数。
然后使用离线 `latest.pt`：

```bash
PYTHONPATH=xr1 python xr1/tools/train_lwd.py --config /path/to/lwd_online.json --resume outputs/lwd/latest.pt --start-online
```

阶段切换保留学习后的 Actor、V/Q、EMA 和原始 SFT 参考，重启优化器及 cosine 调度。
普通 resume 保留优化器、学习率进度和随机状态。在线入口读取启动时的数据快照；
新增回放需更新数据/标签后普通 resume，当前没有后台自动扫描或机器人发布服务。
本入口没有 DDP/ZeRO，批量和显存需求需在完整权重环境中验证。

## 验证

```bash
PYTHONPATH=xr1 .venv/bin/python -m pytest -q xr1/tests/test_lwd.py xr1/tests/test_lwd_data.py
```

测试覆盖概率质量守恒、自适应分位数、terminal TD、动作掩码、解析线性流的伴随、
梯度方向、固定参考、阶段切换、EMA、精确恢复，以及缩小后的真实 Gemma/SigLIP 模块梯度。
XR-1 条件接口和检查点包装函数单独执行其真实源码做 CPU 验证；这不代替完整模型 CUDA 集成测试。
尚未进行完整预训练权重训练或真机评测。
