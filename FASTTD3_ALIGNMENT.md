# FastTD3 / SPiCED 对齐修复（待审核，未提交）

基线：fastTD3 分支 a2ae2cbc。工作目录：/home/cyy/projects/spiced_self_play。

## 修改

1. **观测与归一化**：drive.ini 的 obs_normalization 改为 False，使用 SPiCED 在环境中完成的物理量缩放。道路类别整数直接交给原作 one-hot；启用全观测经验归一化会明确报错，防止类别被破坏。Replay 的可选归一化调用改为只读。真正终止的 next observation 在 Python 适配层置零，它不参与后续价值估计，避免移除位置哨兵进入网络。
2. **终止语义**：读取已有 _fasttd3_transition.terminals（地图重采样覆盖标记之前的真实终止），适配为 dones = true_terminal | scene_truncation，time_outs = scene_truncation & ~true_terminal。保持 FastTD3 原 bootstrap 公式。死亡状态仍用场景截断来清除，不能用排除真实终止后的 time_outs 清除。
3. **结构化编码器**：新增 DriveEncoder，直接继承原 Drive，调用原 encode_observations，仅移除 PPO actor/value 输出层。FastTD3 Actor 和两个 Q 网络分别创建独立编码器，然后接原 FastTD3 MLP。目标 Q 编码器随原 soft_update 更新；采样 actor 沿用原作者 from_module 张量共享。
4. **评估加载**：从 checkpoint 中编码器权重形状重建编码器，避免新模型保存后无法独立评估；原 MLP checkpoint 仍可用于旧模型评估。

## 保持原作的范围

- C 环境、动力学、动作约束、奖励、地图与 Python Drive 环境均未修改。
- SPiCED 原 Drive 类、one-hot、对象池化、初始化方式未修改，也未另写一套编码器计算。
- FastTD3 update_main / update_pol / soft_update 与基线 AST 完全一致。
- Replay 实现、容量、batch size、学习率、探索噪声、C51 支持区间、更新频率均未修改。
- 没有加入 LSTM、BC、奖励塑形或新的训练调参。

## 验证

- `python tests/test_fasttd3.py -v`：7 项检查通过。覆盖原编码器初始化/输出精确一致、对象排列不变性、真实终止与时间截断重叠、重采样覆盖标记、异步 agent 配对、actor/critic 编码器梯度及采样权重共享、新旧 checkpoint 评估加载。
- WSL Python 3.10 / PyTorch 2.6.0+cu124 / TensorDict 0.7.2 / RTX 4060 Laptop：使用真实 update_main、update_pol、soft_update 运行 3 轮 CUDA bf16 更新通过，编码器权重确实更新，loss 有限。
- 原尺寸 1125 维观测、128 维 SPiCED 编码器、512/1024 FastTD3 隐藏层，batch 512、不含 replay，GPU 峰值 allocated 约 0.312 GiB。
- CUDA torch.compile(mode="reduce-overhead", fullgraph=True) 编码器/actor 推理通过。
- 保留 fast_td3.py 原 CRLF 换行；`git -c core.whitespace=cr-at-eol diff --check` 通过。

## 审核后训练

- 使用新的 run，从头训练。旧 MLP checkpoint 不能恢复到带结构化编码器的新模型；新模型的编码器尺寸来自 [policy]，默认 input_size=128、hidden_size=128。
- 可以沿用此前训练命令；新的 drive.ini 已设置 obs_normalization=False，不要在命令行覆盖成 True。
- 没有完成真实地图端到端重训、完整编译训练循环或收敛验证。本 WSL 项目没有编译好的 Drive binding；GPU 检查使用合成观测和真实更新函数。
- batch_size=32768 加原 replay 容量尚未在 4090 上验证显存。结构化编码器会增加训练激活内存，不能将小批量验证解释为完整配置显存已通过；本次按要求没有顺带调整 batch/buffer。

审核文件：pufferlib/config/ocean/drive.ini、pufferlib/fast_td3.py、pufferlib/fast_td3_encoder.py、pufferlib/fast_td3_train.py、pufferlib/pufferl.py、tests/test_fasttd3.py。
原有未跟踪 eval_results/ 保留。未执行 git add、commit 或 push。
