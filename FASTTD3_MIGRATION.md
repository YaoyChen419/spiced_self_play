# 第二阶段：FastTD3-LSTM

实现是面向驾驶环境的 **FastTD3-LSTM 适配版**。复用官方 FastTD3 的 actor、双分布式 critic 和 C51 投影，依赖固定到提交 `229ed59bbf43ea2f7a2d5d90d1076314839944d7`。保留原作 Drive 编码器及 LSTM 的结构和初始化；actor、critic 各有独立的编码器/LSTM，目标 critic 包含对应的记忆网络。LSTM 的输入仍是观测编码，没有额外加入前一动作。

## 保留与必要变化

- 原有地图、并发数、奖励、条件化观测、动力学、回合终止/重置规则、种子、gamma、LSTM 尺寸、BPTT 长度和评估场景设置均未修改。原 PPO 入口仍是默认值。
- 启动时显式选择 `fasttd3` 与连续动作。`delta_local` 仍是三个动作分量；第一阶段的连续动作修复继续使用。
- PPO 的 GAE、裁剪目标、熵奖励和 value loss 被回放学习、分布式 Q 目标及确定性策略梯度替代。PPO 专属优化参数不再适用。
- FastTD3 使用当前 actor 计算下一动作，不另建 target actor；每次 critic 更新都软更新 target critic，每两次 critic 更新更新一次 actor。
- `[fasttd3]` 单独记录新增参数：actor/critic 学习率均为 3e-4、head 宽度 512/1024、101 atoms、支持区间 [-250, 250]、tau=0.1、探索标准差 [0.001, 0.4]、目标噪声 0.001、噪声裁剪 0.5、CDQ 开启。沿用原作 Adam betas/eps、梯度裁剪和线性学习率衰减开关。
- 每次接收到一个异步向量批次后执行 2 次 critic 更新；至少接收 10 次且有完成的轨迹后开始学习。`learning_starts` 的单位是接收批次。每次更新采样 `minibatch_size / rollout_horizon` 个序列，微批次 16 个序列并累积梯度；补齐位置不计入损失。记录有效样本数。
- CPU 回放容量为 262144 个已完成的 transition，另存尚未结束的轨迹。按车辆和回合隔离，采样连续窗口，并从完整回合前缀重建 LSTM 状态；前缀不反向传播。终止后至场景重置前的数据不入回放。只使用已完成轨迹会使最初的回放偏向短轨迹。
- 新增可选 `capture_final_observations`：在自动重置/换地图之前保存最后观测和真实 terminal 标志。超时/换地图允许 bootstrap，真实 terminal 不允许；避免将新回合观测错误拼入 TD 目标。PPO 默认不启用。
- 原评估器识别确定性 actor，直接执行其动作；原 PPO 的动作采样逻辑保留。

默认 8192 个车辆槽位、110 步和 1125 维观测时，存活轨迹观测约占 3.8 GiB CPU RAM，已完成回放观测另约 1.1 GiB，还有采样批次及环境开销。容量和微批次参数用于控制内存，不保证此次循环适配已经达到官方高并发速度；须在实际硬件上测量。

本阶段仅支持固定 `lambda=0`，非零 lambda 或随机 lambda 会报错，不会静默忽略正则化。anchor30 尚未实现。支持 float32、单 learner 进程及原有多环境工作进程；不支持此适配器的 torch.compile、混合精度、训练中 WOSAC 和完整断点续训。原默认设置不受这些限制影响。

## 使用（Linux / WSL，原项目依赖和数据已就绪）

```bash
python -m pip install -r requirements-fasttd3.txt
# 改了 C 接口，必须重新编译；沿用原项目的本机构建环境。
python setup.py build_c --inplace --force
python -m pufferlib.pufferl train puffer_drive --algorithm fasttd3 --env.action-type continuous
```

checkpoint 包含模型、critic、目标 critic、优化器、配置和计数。未保存回放与环境状态，因此仅支持加载进行评估，不宣称可精确恢复训练。评估须使用 checkpoint 中相同的网络尺寸和动力学配置，例如默认尺寸：

```bash
python -m pufferlib.pufferl eval puffer_drive --algorithm fasttd3 --env.action-type continuous --load-model-path experiments/puffer_drive_fasttd3_RUN_ID.pt
```

## 小型验证

```bash
python -m unittest discover -s tests -p test_fasttd3.py -v
python -m unittest discover -s tests -p test_drive_continuous_actions.py -v
```

测试使用随仓库提供的单个 sanity 场景重新序列化，不需要训练数据集。覆盖自动重置前观测、换地图 terminal 区分、回放边界、LSTM 前缀重建、延迟 actor 更新、串行/两个异步工作进程短训练、checkpoint 加载，以及第一阶段连续动作回归。小测试为 CPU 运行；不等同于完整数据集/GPU 性能验证。

提交范围：6 个已有文件（drive.ini、evaluator.py、binding.c、drive.h、drive.py、pufferl.py）和 5 个新增文件（fasttd3.py、fasttd3_train.py、requirements-fasttd3.txt、test_fasttd3.py、本说明）。无需提交编译出的 `.so`、测试依赖、缓存或模型。保留原有文件路径，上传同路径文件会更新对应文件。
