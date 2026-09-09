# FastTD3-LSTM 迁移说明

当前仍在完成工程一致性修复，不能据此宣布完整迁移或速度验收通过。2026-09-09 用户明确批准保留当前 episode 记忆边界，作为相对原 PPO rollout 清零规则的实验差异；奖励裁剪必须保留。

实现是面向驾驶环境的 **FastTD3-LSTM 适配版**。复用官方 FastTD3 的 actor、双分布式 critic 和 C51 投影，依赖固定到提交 `229ed59bbf43ea2f7a2d5d90d1076314839944d7`。保留原作 Drive 编码器及 LSTM 的结构和初始化；actor、critic 各有独立的编码器/LSTM，目标 critic 包含对应的记忆网络。LSTM 的输入仍是观测编码，没有额外加入前一动作。

## 保留与必要变化

- 原有地图、并发数、奖励、条件化观测结构、动力学、回合终止/重置规则、种子、LSTM 尺寸、BPTT 长度和评估场景设置均未修改。原 PPO 入口仍是默认值。gamma 改由 FastTD3 配置控制，官方默认同为 0.99。
- 启动时显式选择 `fasttd3` 与连续动作。`delta_local` 仍是三个动作分量；第一阶段的连续动作修复继续使用。
- PPO 的 GAE、裁剪目标、熵奖励和 value loss 被回放学习、分布式 Q 目标及确定性策略梯度替代。PPO 专属优化参数不再适用。
- FastTD3 使用当前 actor 计算下一动作，不另建 target actor；每次 critic 更新都软更新 target critic，每两次 critic 更新更新一次 actor。
- `[fasttd3]` 单独记录算法参数：actor/critic 学习率均为 3e-4、head 宽度 512/1024、101 atoms、支持区间 [-250, 250]、tau=0.1、探索标准差 [0.001, 0.4]、目标噪声 0.001、噪声裁剪 0.5、CDQ 开启。优化器对齐官方 AdamW（weight_decay=0.1，betas=(0.9,0.999)，eps=1e-8），使用设备上的标量学习率；默认不裁剪梯度。余弦调度按驾驶环境总步数进度计算，官方起止学习率相同，默认实际恒定，不再沿用 PPO 线性衰减。
- `num_updates=2`、`learning_starts=10` 按完整向量等效步计算：异步返回累积到全部车辆槽位数才推进一次算法时钟，沿用官方零基计数 `step > learning_starts`。初始 reset 返回不作为已执行的动作步。测量预热同样使用向量步。每次更新采样 `minibatch_size / rollout_horizon` 个序列，微批次 16 个序列并累积梯度；补齐位置不计入损失。回合内位置 t 的窗口覆盖次数为 `min(t+1, rollout_horizon)`，损失乘以覆盖次数倒数，再除以采样序列总数。
- 回放与 learner 同设备，CUDA 训练时采用 GPU float32 存储、批量归档和向量化采样，CPU 测试使用同一个实现。默认容量 1048576，向下取整为完整回合槽位，另存活跃轨迹；至少容纳每个车辆槽位的一条完整回合，环形覆盖按回合进行。按车辆和回合隔离，采样连续窗口，并从完整回合前缀重建 LSTM 状态；前缀不反向传播。终止后至场景重置前的数据不入回放。仅采样已完成轨迹会使最初的回放偏向短轨迹。
- 新增可选 `capture_final_observations`：在自动重置/换地图之前保存最后观测和真实 terminal 标志。超时/换地图允许 bootstrap，真实 terminal 不允许；避免将新回合观测错误拼入 TD 目标。PPO 默认不启用。
- 原评估器识别确定性 actor，直接执行其动作；原 PPO 的动作采样逻辑保留。

默认 8192 个车辆槽位、110 步和 1125 维观测时，活跃轨迹观测约 3.8 GiB，完成回放观测约 4.4 GiB，均在 GPU；还需网络、采样和编译缓存空间。初始化计算回放需求，超过当时可用显存的 60% 时明确报错，不静默降低并发或退回 CPU。用户允许调整回放容量；5090 32GB 的完整规模峰值仍须实测。

本阶段仅支持固定 `lambda=0`，非零 lambda 或随机 lambda 会报错，不会静默忽略正则化。anchor30 尚未实现。支持单 learner 和原有多环境工作进程。训练中 WOSAC 已接回原平台子进程，默认仍关闭；模型、环境和评估参数显式传递，排除仅供构造器使用的内部参数。

## 工程加速适配

- 默认 `fasttd3.amp=True`、`amp_dtype=bf16`；网络前向和反向使用 autocast，FP16 模式才启用 GradScaler。参数、优化器状态、回放、LSTM 状态缓冲、C51 投影和概率损失保留 float32。原生环境接收 float32 动作。
- 默认 `fasttd3.compile=True`、`compile_mode=reduce-overhead`，编译 Drive 编码器、actor/双 critic 网络、C51 投影和 actor/critic 损失计算。梯度缓冲区在图捕获外预分配，支持微批梯度累积。保留 eager cuDNN LSTM、CPU 长度、回放采样和优化器调度；这是分段编译，不是整个递归更新的单一 CUDA Graph。前缀先 pack 再编码，避免编码无效填充；编码器启用动态形状，保留全部 episode 历史。
- 官方 `torch.set_float32_matmul_precision('high')`、AdamW、设备标量学习率、foreach 目标网络更新均已迁入。恢复原平台 `cudnn.benchmark=True`。每微批次不提取 loss 标量，只在更新结束汇总；packed LSTM 长度合并为每次优化更新一次 CPU 拷贝。
- 观测归一化复用官方 EmpiricalNormalization，训练采集和有效回放数据更新统计，三个网络在一次更新中使用相同统计。道路类别 ID 原样保留；不改变原编码器结构。统计随 actor checkpoint 保存，评估不更新。回放中 padding 不参与统计；前缀只做归一化，不重复贡献统计。奖励入库前沿用平台 `clip(-1,1)`，不启用额外奖励归一化。所锁定官方训练代码中的动作/噪声裁剪和 C51 支持范围不等同于奖励裁剪。
- 前一观测和动作留在 GPU，避免再次上传；本身位于 CPU 的驾驶环境仍须上传新观测/奖励并接收动作。cuDNN packed sequence 仍要求一次小型长度元数据拷贝，不能宣称训练循环完全无 CPU/GPU 同步。
- PPO 的 `train.precision/compile/anneal_lr/max_grad_norm/adam_*/minibatch_size` 不再决定 FastTD3 优化设置，以 `[fasttd3]` 为准。学习窗口沿用平台 `rollout_horizon=32`，算法批量由 `fasttd3.batch_size=32768` 控制；两项默认数值均未改变。

## 使用（Linux / WSL，原项目依赖和数据已就绪）

```bash
python -m pip install -r requirements-fasttd3.txt
# 改了 C 接口，必须重新编译；沿用原项目的本机构建环境。
python setup.py build_c --inplace --force
python -m pufferlib.pufferl train puffer_drive --algorithm fasttd3 --env.action-type continuous
```

checkpoint 包含模型、critic、目标 critic、优化器、配置和计数。保留原作分轮保存结构、trainer_state 和 W&B artifact。`--load-model-path` 可继续训练，恢复 learner、优化器及训练计数；环境和回放重新初始化并重新预热，不是精确轨迹续接。尚不支持通过 `--load-id` 下载并续训。加载须使用 checkpoint 中相同的网络尺寸和动力学配置，例如默认尺寸评估：

```bash
python -m pufferlib.pufferl eval puffer_drive --algorithm fasttd3 --env.action-type continuous --load-model-path experiments/puffer_drive_fasttd3_RUN_ID.pt
```

## 小型验证

```bash
python -m unittest discover -s tests -p test_fasttd3.py -v
python -m unittest discover -s tests -p test_drive_continuous_actions.py -v
```

测试使用随仓库提供的单个 sanity 场景重新序列化，不需要训练数据集。覆盖自动重置前观测、换地图 terminal 区分、回放边界、LSTM 前缀重建、延迟 actor 更新、串行/两个异步工作进程短训练、checkpoint 加载，以及第一阶段连续动作回归。包括 CPU 检查和可用时的 CUDA 短训练；不等同于完整数据集性能验证。

本次修改文件：`pufferlib/fasttd3_train.py`、`pufferlib/fasttd3.py`、`pufferlib/config/ocean/drive.ini`、`pufferlib/utils.py`、`tests/test_fasttd3.py`、本说明和 `FASTTD3_PARITY_AUDIT.md`。这不是相对原始项目的全部文件清单。无需提交 `.so`、测试依赖、缓存或模型。

本轮 22 项本地短测试通过（19 项 FastTD3、3 项连续动作）：新增异步时钟、奖励裁剪/SIGINT 保存、续训、WOSAC 子进程参数解析；CUDA 测试使用当前默认 reduce-overhead/BF16，实际执行 critic 和延迟 actor 更新。WOSAC 只验证调用和参数，未运行真实数据集评估。测试机器仍为本地 WSL/PyTorch 2.13/4060，不能替代目标服务器 PyTorch 2.8/5090 验收。

## 监控与保存恢复（2026-09-09）

| 原作行为 | FastTD3 对应处理 |
| --- | --- |
| `environment/*` | 采集 Drive 原生 info，恢复回报、回合长度、碰撞、越界、完成率、路线进度和其他所有数值统计；沿用报告均值口径，不重算驾驶指标。仅排除回放专用 `_fasttd3_transition`。 |
| `losses/*` | 记录分布式 critic loss 和确定性 actor loss；汇总本日志窗口的真实更新，未执行 actor 更新时不把占位零计入均值。PPO 的 GAE、entropy、clip、value loss 不适用于 FastTD3，不伪造这些指标。 |
| `performance/*` | 复用原 `Profile(frequency=5)`，记录采集、环境等待、策略推理、学习耗时；`eval` 沿用原命名，指训练 rollout 采集，不是 HR/SP 评估。PPO 内部特有算子没有虚构对应计时。 |
| 步数/epoch/SPS | 恢复 `agent_steps`、`epoch`、`uptime`、区间 `SPS`；保留已有 FastTD3 回放量、更新数、`SPS_train`。actor/critic 学习率分别记录。 |
| 条件化数据 | 从活跃车辆最近一次原始采集观测记录 `data/lambda_mean/std` 和 lambda/碰撞奖励直方图。FastTD3 的样本来源不同于 PPO 的 on-policy minibatch；不虚构未执行的 BC anchor 熵和人类样本统计。 |
| 有效数据比例 | `replay/valid_transition_fraction` 明确表示采集数据进入回放的比例；不能将重复回放采样称为 PPO 的 `environment/perc_transitions_used`。 |
| 日志时机 | 保留 epoch 边界及原 0.25 秒节流，最终正常完成强制上报；没有新增每 10 次更新等频率。评估结果保留至实际日志上报。 |
| HR/SP 评估、视频 | 使用原 Evaluator、开关和 `eval_interval`；保留评估前保存和失败标记保护。此次不降低评估间隔。 |
| 模型保存/归档 | 按原 `checkpoint_interval` 保存编号模型、trainer_state 和 W&B checkpoint artifact；完成时上传最终模型。latest 文件采用临时文件替换。 |
| W&B System | 继续由 SDK 自动记录；退出时关闭 W&B，避免没有 checkpoint 时漏掉 finish。 |

默认日志仍需累计 524288 车辆步数后才首次上报训练曲线；SIGINT 中断时强制上报并保存，不等待下一个 epoch。步数恢复按原 recv mask 累加；另记 vector_steps 表示算法时钟。终端直接调用原 `PuffeRL.print_dashboard`，使用原 Utilization 监控线程，损失字段使用实际 FastTD3 指标。Fast 运行期间替换原强制退出的 SIGINT 处理，完成当前迭代后保存并关闭 logger，最后恢复原处理器；PPO 行为不改。

新增数值统计及日志接口回归：以替身 W&B SDK 验证真实 logger 收到分组指标、评估调用周期、分轮模型、trainer_state 和 artifact；不访问真实 W&B 服务。原生 HR/SP 评估由既有测试单独覆盖。现共 18 项短测试通过（15 项 FastTD3、3 项连续动作）；此结果不代表已验证服务器端 W&B 上传网络或网页面板布局。

## 静态审查后的修复

- 独立 Drive 评估使用正确的 `env_idx` 参数，每一步使用最新返回的观测。
- 评估器补齐统计预运行的 mode 参数、关闭 human-replay 渲染时的 None 检查和空统计列表保护。
- FastTD3 在每次评估前保存 checkpoint；评估异常或启用模式缺少成绩时记录 `eval/failed=1` 并打印原因，不沿用旧成绩，不把失败当作零分。训练继续；默认评估与保存周期一致，原周期不变。
- 回放窗口按覆盖次数补偿权重；新增枚举窗口起点的回归用例。
- `drive.h`、`binding.c` 恢复原 LF 换行，未改变 C 逻辑。

## 联调结果（2026-09-08）

工程迁移后通过 16 项测试：13 项 FastTD3 检查、3 项连续动作回归。环境为 WSL Ubuntu、PyTorch 2.13.0+cu130、RTX 4060 Laptop GPU。新增测试实际开启 CUDA 编译与 BF16，执行 critic 和延迟 actor 反向更新，检查编译图计数增加、有限损失、AdamW/学习率及类别 ID 保留；不是只检查配置开关。另检查设备回放环形覆盖、前缀连续性和补齐权重。

- 正式 `pufferl.train` 入口选择 FastTD3 时，测试将 PPO 构造器替换为报错哨兵，训练仍正常完成；模型可保存并重新加载。
- 实际命令行 `--algorithm fasttd3 --env.action-type continuous` 在 CUDA/BF16/GPU 回放上完成 1024 个车辆步数；保留默认 256 维 LSTM 和默认 FastTD3 head 尺寸，测试缩小并发、回放及 minibatch（8 个车辆槽位、minibatch=64、微批次=1、回放2048）。该命令行测试关闭编译以限制耗时，编译另由小网络更新测试验证。checkpoint 标识为 fasttd3，actor/critic 优化器均有更新状态，三个网络的参数均有限。
- 验证延迟 actor 更新、目标 critic 软更新、LSTM 前缀重建、回放覆盖权重，以及终止/截断/换地图边界；双工作进程短训练通过。
- 关闭渲染时，原生 self-play 和 human-replay 评估均返回有限统计；测试临时场景补充 SDC 标记，评估回合缩短为 4，未改原始场景文件或正式环境配置。故障注入验证评估前已有 checkpoint，失败不会伪造成绩。

这是小规模功能联调，未验证完整数据集收敛、默认大批次显存占用、渲染视频或相对 PPO 的速度优势。默认入口仍为 PPO，必须显式选择 `--algorithm fasttd3`。

## 全部文件改动核对（含第一阶段）

以下是此前 13 个文件的清单；本次另修改 `pufferlib/utils.py` 以恢复 WOSAC 参数传递，并新增 `FASTTD3_PARITY_AUDIT.md` 保存审查记录。没有删除原有文件。

| 文件 | 类型 | 改动及目的 |
| --- | --- | --- |
| `pufferlib/ocean/drive/drive.py` | 修改 | 连续 delta_local 从错误的 2 维改为 3 维，classic/jerk 仍为 2 维；增加可选最后观测缓冲区、初始化/换图绑定及 transition 元数据；仅启用捕获时清理 reset 的 terminal 标志。 |
| `pufferlib/ocean/drive/drive.h` | 修改 | 新增 9 行：可选缓冲区指针、重置前观测计算和复制；原动作映射、奖励和终止条件未改。 |
| `pufferlib/ocean/drive/binding.c` | 修改 | 新增 14 行：检查可选缓冲区 dtype、形状和可写连续性，并向 C 传递指针。与 drive.h 一同恢复 LF 换行。 |
| `pufferlib/fasttd3.py` | 新增 | 复用 Drive 编码器/LSTM 和官方 FastTD3 heads，实现独立 actor/critic 记忆、目标 critic、序列回放、覆盖权重补偿和学习更新。 |
| `pufferlib/fasttd3_replay.py` | 新增 | 统一 CPU/GPU 回放实现，按回合批量环形归档、设备向量化采样与容量检查。 |
| `pufferlib/fasttd3_train.py` | 新增 | 异步车辆 ID 对齐、有效数据筛选、探索噪声、学习调度、配置限制、日志、checkpoint 与原评估器连接。 |
| `pufferlib/pufferl.py` | 修改 | 添加算法选择/训练分流/FastTD3 模型加载；PPO 内核依赖检查推迟到 PPO 训练入口；确定性动作评估及 Drive 渲染参数/最新观测修复。 |
| `pufferlib/config/ocean/drive.ini` | 修改 | 新增 FastTD3 算法与工程配置；原环境配置值未改。连续动作在命令行显式选择。 |
| `pufferlib/ocean/benchmark/evaluator.py` | 修改 | 确定性 actor 动作适配，补 mode 参数，关闭渲染与空统计保护；未调整指标公式和原评估场景配置。 |
| `requirements-fasttd3.txt` | 新增 | 固定官方 FastTD3 提交和 tensordict==0.7.2；未替换原项目依赖文件。 |
| `tests/test_drive_continuous_actions.py` | 新增 | 第一阶段 3 项原生动作接口回归。 |
| `tests/test_fasttd3.py` | 新增 | 15 项 FastTD3 测试，覆盖 CUDA 编译/BF16、回放及新恢复的环境统计、W&B 分组和 checkpoint 归档。 |
| `FASTTD3_MIGRATION.md` | 新增 | 记录实现范围、参数、必要差异、使用方法和验证边界。 |

原 `train.learning_rate`、PPO 的 GAE/clip/entropy/value/prioritized-rollout 参数不用于 FastTD3。FastTD3 使用两个独立学习率；`train.batch_size` 用来确定日志/评估的 epoch 尺度，梯度更新由 `fasttd3.num_updates` 调度。`train.max_minibatch_size` 不控制此适配器的微批次，改由 `fasttd3.microbatch_sequences` 控制。`minibatch_size / rollout_horizon` 决定采样序列数，窗口尾部 padding 后有效 transition 数可能小于 minibatch_size。

## 测速前需要统一的口径

- `SPS` 已恢复为两次日志之间的车辆槽位步数增量/时间增量，包含该区间的评估和保存开销；第一段仍包含采集预热。首次编译会拉低第一段。
- 后续测速应在预热完成、确实持续更新网络之后，用相同时间窗口计算增量步数/增量时间；同时记录有效车辆 transition、更新次数、有效学习样本数及峰值 RAM/VRAM。双方采用相同的评估/渲染/保存计时规则。
- `SPS_train`：排除初始学习预热及评估/保存/日志开销；预热阈值为 `fasttd3.measure_burnin=3` 次更新。初次编译、其他形状重编译、GPU 缓存和回放逐渐填充仍可能影响短窗口，正式比较应选稳定运行区间。它与包含保存/评估的区间 `SPS` 不能混用。
- 当前回放位于训练设备，默认微批次 16 序列，开启编译和 AMP，但仍需重建 LSTM 前缀。正式地图与默认并发/大批次资源占用、完整训练效果，以及相对 PPO 的速度均待 AutoDL 实测。

远端已核对提交 `95a521ac88f911dfe228a269d247eb91bb56fedf`。此后索引修复及监控恢复尚需同步。用户的 RTX 5090/PyTorch 2.8.0/CUDA 12.8 日志已记录 209 次 critic、104 次 actor 更新，GPU 后 10 分钟平均利用率约 10.55%；这证明可持续更新，不证明速度优势。此次监控恢复尚未在服务器实测。新增归一化状态意味着旧版 checkpoint 不能直接严格加载到新默认模型，旧模型请使用其对应代码评估。
