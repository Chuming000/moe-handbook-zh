# 第 3 章 解剖 MoE 训练：系统工程

> 说明：原站里的图都是交互式组件，这里无法复现，一律保留图注并标注「（原站为交互图，此处从略）」。原站大部分图不给编号，本文按**节内顺序**编号（如 图 3.7-a）。

## 3.7 系统工程：时间去哪了，怎么拿回来

每个 MoE 层都要把 token 发给专家再收回来，而在真实运行里这部分通信可能吃掉大半步。我们在一个节点上追的那一步里，有一半时间花在裸露的 All-to-All 上，而那时路由器还不均衡。下图放大这两次传输——把 token 发出去的 dispatch 和收回来的 combine——它们在节点内走 NVLink、跨节点走网络。

> **图 3.7-a** 一个 MoE 步的时间构成：dispatch / 专家计算 / combine / 其余（原站为交互图，此处从略）

你可以把专家摆得让更少流量穿过节点间那张更慢的网，把剩下的流量藏到别的计算后面；可以把这份内存代价控制住，给每个 token 能到达的节点数加上限；也可以用丢 token 或降精度换速度。每一部分都来自各自的设置，收益不能相加，要当作备选方案逐个测。

各模型报告里记录的系统选择。并行与重叠描述训练；后训练量化单独标注。这些是各模型的具体实现，不是可以互换的 Megatron 开关。

| 模型 | 互联与重叠 | 内存与精度 | 路由约束 |
|---|---|---|---|
| DeepSeek-V3 | 跨 8 节点 EP64；DualPipe 重叠计算与通信 | FP8 [混合精度](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=mixed_precision_training)训练；重算 RMSNorm、MLA 上投影与 SwiGLU 输出 | 每 token 最多 4 个目标节点；不丢 token |
| DeepSeek-V4 | 融合 MoE 流水调度把专家组按波次排布，重叠 dispatch、GEMM 与 combine | 张量级 checkpoint 与选择性重算 | 取消了 V3 的目标节点上限 |
| Kimi K3 | 优化器只取本地持有的参数分片；通信与计算管线化 | 省掉全参数 gather buffer；SFT/RL 用 MXFP4 专家权重、MXFP8 激活 | 发布配置里只有一个专家组 |
| LongCat-Flash | 32 卡一组的 EP；管线化 all-gather/reduce-scatter + 捷径连接 MoE | BF16 训练 + 确定性 kernel | 恒等专家不需要专家通信 |

在动上面任何一个杠杆之前，你得先看清自己那一步的时间花在哪，因为每种浪费对应不同的解法。单节点上剖析出来的那一步长什么样？

### 剖析（Profiling）

[Megatron](https://github.com/NVIDIA/Megatron-LM) 能帮你把任意一步录成 trace。只给 `--profile` 会录 nsys trace，加 `--use-pytorch-profiler` 则改录 torch profiler trace（[`argument_utils.py#L706-L709`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/argument_utils.py#L706-L709)）。两种情况下都用 `--profile-step-start`、`--profile-step-end`、`--profile-ranks` 选要录哪几步、哪些 rank。

我们追的那一步来自一个 7B-A1B 的 MoE：总参 7B、每 token 激活 1B，16 层、每层 64 专家、top-8 路由，在一个 8×H100 的节点上训练。追它只花了一个短任务：录第 4 个迭代，启动后约 6 分钟内就跑过了。

那次 All-to-All 占了一半步时，可这一半里有多少是网络真的在搬数据？当某张 GPU 的专家收到比别的卡更多的 token 时，它的专家计算结束得晚，而 NVIDIA Megatron Core 团队报告说，其他 rank 随后会在同步点等它。在 MoE 层里，All-to-All 可能就是这样的同步点之一，所以它在 trace 里的时间有一部分是「在等」而不是「在传」。我们那一步的负载均衡损失是 [3.2](#s-3-2)（1 表示均衡），所以路由器还是不均衡的。同一配置另跑一次、把路由强制成均匀之后，同一个迭代快了 40%。

```python
# megatron/training/training.py#L4955-L4982（逐 rank 的 torch profiler 窗口）
if (
    cfg.profiling.use_pytorch_profiler
    and (len(cfg.profiling.profile_ranks) == 0 or
         torch.distributed.get_rank() in cfg.profiling.profile_ranks)
):
    if cfg.profiling.pytorch_profiler_collect_chakra:
        et_dir = Path(f"{cfg.logger.tensorboard_dir}/../chakra")
        et_dir.mkdir(parents=True, exist_ok=True)
        et = torch.profiler.ExecutionTraceObserver().register_callback(
            f"{et_dir}/rank-{torch.distributed.get_rank()}.json.gz"
        )
    else:
        et = None

    def trace_handler(p):
        profile_dir = Path(f"{cfg.logger.tensorboard_dir}/../torch_profile")
        profile_dir.mkdir(parents=True, exist_ok=True)
        p.export_chrome_trace(f"{profile_dir}/rank-{torch.distributed.get_rank()}.json.gz")

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(cfg.profiling.profile_step_start - 1, 0),
            warmup=1 if cfg.profiling.profile_step_start > 0 else 0,
            active=cfg.profiling.profile_step_end - cfg.profiling.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=trace_handler,
        record_shapes=cfg.profiling.pytorch_profiler_collect_shapes,
        with_stack=cfg.profiling.pytorch_profiler_collect_callstack,
        execution_trace_observer=et,
    )
    prof.start()
```

在你自己的运行上，先看 [3.2](#s-3-2) 那些均衡信号。用 `--save-tokens-per-expert-interval` 写出的 token 计数算 MaxVio，能看出负载在专家之间有多不均。把每张 GPU 上各专家的计数加起来，就知道哪个 rank 收到的 token 最多、也因此最可能被别人等。想知道这份不均衡让 All-to-All 多花了多少，就带 `--moe-router-force-load-balancing` 重追一次——它把路由器的 logits 换成随机的，让每个专家负载大致相同（[`router.py#L1185-L1187`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router.py#L1185-L1187)）。如果强制均衡之下 All-to-All 变小了，说明不均衡正在白花你的时间，更好的均衡（[3.3](#s-3-3)）能把这份时间拿回来；剩下的主要就是传输本身，本节的其余部分处理它。

trace 还能告诉你哪些 MoE kernel 值得融合。dispatch 之前 Megatron 会把 token 重排，让同一个专家的 token 挨在一起（[`token_dispatcher.py#L668-L682`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/token_dispatcher.py#L668-L682)），combine 之后再排回去（[`#L893-L900`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/token_dispatcher.py#L893-L900)）。带上 `--moe-permute-fusion`，做这件事的 permute 与 copy kernel 从步时的 11.45% 降到 7.14%，步时也跟着降。

每一项改动都按步时或 TFLOP/s 判断。trace 再告诉你这个数字为什么变：读一读 All-to-All 在步里留下了多少裸露份额。

**什么时候用它**：每个新配置用一次——在你要跑的那个节点数上追一步。如果 All-to-All 占比很大，先看 MaxVio，再带 `--moe-router-force-load-balancing` 重追：All-to-All 变小就先修均衡（[3.3](#s-3-3)），没变小就去做下面的互联与重叠。如果 permute 与 copy kernel 占比很大，试试 `--moe-permute-fusion`。

我们追的那一步把 All-to-All 留在了单节点内、走 NVLink，而更大的运行不总能这样。当专家并行组跨两个节点、All-to-All 必须穿过节点间网络时，这一步会怎样？

### 互联（Interconnect）

跨节点时，All-to-All 离开 NVLink，走集群的网卡网络（比如 InfiniBand，或者 AWS 的 EFA），所以第一件要检查的事是 [NCCL](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=nccl) 到底有没有用上那张网。它可能在你明明要求了的情况下退回普通 TCP socket：我们两次运行设了 `FI_PROVIDER=efa`、也按名字指定了 OFI 插件，NCCL 仍记着 "Using network Socket"，那两次比我们在 EFA 上的 EP16 运行慢了约 25 倍。下图按顺序列出我们试过的设置。

> **图 3.7-b** 让 NCCL 走上 EFA 的尝试顺序与结果（原站为交互图，此处从略）

图里，只有当主机的 RDMA 库排在库路径最前面时，NCCL 才走到 EFA；按路径或按名字指定插件，都还留在 socket 上。网通了之后，剩下的选择是专家并行规模 `--expert-model-parallel-size`，它决定 All-to-All 要走多远。专家张量并行取 1、rank 顺序取默认时，Megatron 用连续的 rank 组成每个 EP 组（[`parallel_state.py#L921-L929`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/parallel_state.py#L921-L929)）。于是能整除单节点 GPU 数的 EP 规模会把 All-to-All 留在该节点内的 NVLink 上，更大的规模就摊到多个节点。我们的节点每台 8 张 H100，所以下图比较单节点 EP8 与跨两节点的 EP16。

> **图 3.7-c** 单节点 EP8 vs 跨两节点 EP16 的 TFLOP/s（原站为交互图，此处从略）

摊到两个节点后，即便 EFA 正常工作，EP16 也比单节点 EP8 低 59%——尽管这两次运行在节点本身、以及每步累积的 micro-batch 数上也不一样。

```python
# megatron/core/parallel_state.py#L920-L930（专家网格，EP 排在 TP 之后）
expert_order = _inject_gtp_remat_axis(order, after="ep")
expert_decoder_rank_generator = RankGenerator(
    tp=expert_tensor_parallel_size,
    ep=expert_model_parallel_size,
    dp=expert_data_parallel_size,
    pp=pipeline_model_parallel_size,
    cp=1,
    order=expert_order,
    rank_offset=rank_offset,
    gtp_remat=expert_gtp_remat_size,
)
```

```python
# megatron/core/parallel_state.py#L1419-L1428（连续 rank 组成 EP 组）
for ranks in expert_decoder_rank_generator.get_ranks('ep'):
    group = create_group(
        ranks,
        timeout=timeout,
        pg_options=get_nccl_options("ep", nccl_comm_cfgs),
        group_desc="EXPERT_MODEL_PARALLEL_GROUP",
    )
    if rank in ranks:
        _EXPERT_MODEL_PARALLEL_GROUP = group
        _EXPERT_MODEL_PARALLEL_RANKS = ranks
```

退回 socket 会把跨节点运行拖慢到上面那个倍数，所以先看启动日志。设 `NCCL_DEBUG=INFO`，NCCL 会打印 "Using network" 和它选中的传输层。你要看到的是自家网的传输层（比如 EFA 上 OFI 插件对应的 Libfabric），出现 "Using network Socket" 就是退回去了。确认网通了，就按 [2.1](#s-2-1) 用 [nccl-tests](https://github.com/NVIDIA/nccl-tests) 比较节点内与跨节点的 alltoall 总线带宽。然后分别在「装得进一个节点」和「跨两个节点」的 EP 规模下量每 GPU TFLOP/s。

**什么时候用它**：模型装得下时先在节点内用[专家并行](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism)：从那里起步，加节点当[数据并行](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism)副本，在定下来之前把两种摆法都按你的节点数量过一遍时间。

就算网是好的，穿网的 token 仍然压在关键路径上、让 GPU 干等。token 在路上的时候，GPU 能干点别的活吗？

### 重叠（Overlap）

我们在 [3.5 块接线](#s-3-5-block-wiring) 已经看到，改架构能把 All-to-All 藏进一个 micro-batch 里。并行 block 让 dispatch 在注意力计算时就走，捷径连接的 MoE 还能把 combine 藏在旁边跑的稠密计算后面。可重接线会改变路由器看到的东西，而你也许不想动已经定好的架构。不重接线的话，裸露的 All-to-All 很难在一个 micro-batch 内藏起来。可一个训练步会跑很多 micro-batch，而它们的梯度求和之前，彼此的工作并不互相等待。

Megatron 就是借这份工作：`--overlap-moe-expert-parallel-comm` 让一个 micro-batch 的前向与另一个的反向同时跑，于是一个的 All-to-All 藏到另一个的计算后面（[`model_parallel_config.py#L349-L352`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/model_parallel_config.py#L349-L352)）。流水调度本来也会把一个 micro-batch 的前向和另一个的反向配对，[Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero_bubble_and_dualpipe) 有讲过，DeepSeek-V3 的 DualPipe 就是靠这个配对藏它的 All-to-All。下图画了一个 MoE 层在有无这种交织时的样子。

> **图 3.7-d** 一个 MoE 层：micro-batch 顺序执行 vs 前向/反向交织（原站为交互图，此处从略）

按一个 micro-batch 接一个画的时候，这一层把四次 All-to-All 全裸露出来。交织的那版还打开了 `--delay-wgrad-compute`，Megatron 的 MoE 指南把它与这个重叠配对使用（[`README.md#L434`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/README.md?plain=1#L434)）。这个开关把权重梯度的计算推后，于是反向的 dispatch 只等专家的输入梯度，并在该 micro-batch 自己那份延迟的专家权重梯度之下跑（[`model_chunk_schedule_plan.py#L266-L268`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/models/common/model_chunk_schedule_plan.py#L266-L268)），而另外三次 All-to-All 则跑在另一个 micro-batch 的计算之下。在我们只开了重叠开关的运行里，重叠把双节点吞吐提高了最多 35%。它也要花内存：rank 0 的 reserved 多了 5 到 10 GB，而峰值 allocated 几乎没动。

```python
# megatron/core/pipeline_parallel/combined_1f1b.py#L54-L67（一个前向并一个反向）
"""Scheduler for 1f1b with no pipelining.

This function schedules micro-batches in a way that the forward pass of Transformer layers
for one micro-batch runs in parallel with the backward pass of another.
Each layer's forward and backward operations are co-scheduled to maximize the overlap of
their computations and communications.
EP A2A in forward step is hidden by the attention/mlp computation in the backward step,
and vice versa.
Assuming we have 4 microbatches, the schedule is as follows:
Phases 0: 1st microbatch forward
Phases 1: 1st microbatch backward + 2nd microbatch forward
Phases 2: 2nd microbatch backward + 3rd microbatch forward
Phases 3: 3rd microbatch backward + 4th microbatch forward
Phases 4: 4th microbatch backward
"""
```

打开重叠前后都用 `--log-throughput` 记每 GPU TFLOP/s，只有步变快才留下它。然后再看 trace，确认哪些 All-to-All 真的与计算重叠了。

**什么时候用它**：All-to-All 裸露、且内存允许时，加上 `--overlap-moe-expert-parallel-comm --delay-wgrad-compute`（Megatron MoE 指南推荐的那一对，见 [`README.md#L434`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/README.md?plain=1#L434)），而不是 `--moe-shared-expert-overlap`。`--delay-wgrad-compute` 只在重叠打开、且用 Transformer Engine 层时才能跑（[`transformer_config.py#L3631-L3635`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L3631-L3635)、[`arguments.py#L1956-L1958`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/arguments.py#L1956-L1958)）。用流水线并行时，这个重叠还需要虚拟流水 stage（[`transformer_config.py#L3565-L3599`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L3565-L3599)）。

重叠赢回时间但花内存。怎么把一次运行的内存看住、又不把这点时间还回去？

### 内存（Memory）

MoE 通过两处改变内存：权重和激活。权重装着所有专家，尽管每个 token 只用到其中 k 个；专家并行把这些专家摊开，每张 GPU 只留自己那份。而激活（一层为反向保留的中间结果）随 top-k 增长：dispatcher 对每个被选中的专家各拷一份 token 隐状态（[`token_dispatcher.py#L559`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/token_dispatcher.py#L559)），每份拷贝各自保留自己的专家激活，于是「专家收到更多 token」的那张 GPU 也存着更多。

常见的答案是激活重算：前向之后把激活丢掉，反向需要时再算一遍。[Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=activation_recomputation) 讲了它的两种经典形式——整层全量重算，和只重算注意力。全量重算把每层前向跑两遍，在 micro-batch 2 下让我们损失 24% 吞吐；而 [3.7 重叠](#s-3-7-overlap) 要求把它关掉（[`transformer_config.py#L3579-L3581`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L3579-L3581)）。

更细的做法从「一层里每个模块为反向存了什么」出发。有些张量重建很便宜，比如归一化与专家激活函数的输出，几次逐元素运算就够；另一些又大又贵：进注意力的 query/key/value，以及进专家第一层矩阵乘的、每个 token 的 k 份拷贝。Megatron 区别对待这两类，下图标出它的每个选项作用在哪个被保存的张量上。

> **图 3.7-e** 一层里被保存的张量，以及重算（蓝）与卸载（描边）选项分别作用在哪（原站为交互图，此处从略）

蓝色的标签是重算选项。Megatron 的指南重算归一化和专家激活（`--recompute-granularity selective --recompute-modules layernorm moe_act`）：把它们的廉价输出丢掉，在真正要读它们的那次反向之前重建（[`random.py#L1042-L1064`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/tensor_parallel/random.py#L1042-L1064)）。描边的标签属于细粒度激活卸载，指南卸载注意力的输入与输出、以及专家的第一个输入（`--fine-grained-activation-offloading --offload-modules core_attn attn_proj expert_fc1`）。

卸载是在后续模块计算时，用一条单独的 CUDA stream 把这些大模块的输入拷到 pinned CPU 内存（[`fine_grained_activation_offload.py#L1262-L1269`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/pipeline_parallel/fine_grained_activation_offload.py#L1262-L1269)）。反向每做完一个被卸载的模块，就在第二条 stream 上开始拷回下一个模块的张量，于是它们在需要之前就已经在 GPU 上（[`#L1332-L1342`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/pipeline_parallel/fine_grained_activation_offload.py#L1332-L1342)）。重算付的是算力，卸载付的是主机与 GPU 之间的带宽，而这些拷贝被藏在计算后面。下图让三层走一遍前向与反向，分别保留全部保存的输入、全量重算、以及那种拆分，并追踪每种设置留在 GPU 上的保存输入。

> **图 3.7-f** 三层模型里三种内存策略的峰值与保存量（原站为交互图，此处从略）

只有三层、一个 micro-batch、时长也是示意的：全量重算峰值最低但多一遍前向；细粒度拆分留得更多，却只多几次小重算，因为它的拷贝跑在计算之下。最后一层把它的大输入留在 GPU 上，因为它的反向在前向一结束就开始了，重新加载会把它卡住（[`fine_grained_activation_offload.py#L630-L637`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/pipeline_parallel/fine_grained_activation_offload.py#L630-L637)）。把模拟里的主机链路调慢，拷到 CPU 的拷贝就结束得晚，峰值随之上升；再慢一些，反向就要等它的重新加载。

```python
# megatron/core/transformer/moe/experts.py#L319-L322（moe_act 需要 selective）
self.activation_recompute = (
    self.config.recompute_granularity == 'selective'
    and "moe_act" in self.config.recompute_modules
)
```

```python
# megatron/core/transformer/moe/experts.py#L1130-L1141（重算激活，丢掉输出）
if self.activation_recompute:
    self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
    with moe_act_manager as fc1_output:
        bias_act_output = self.activation_checkpoint.checkpoint(
            bias_act_func, fc1_output, bias_parallel, permuted_probs
        )
else:
    with moe_act_manager as fc1_output:
        bias_act_output = bias_act_func(fc1_output, bias_parallel, permuted_probs)
output, output_bias = apply_module(self.linear_fc2)(bias_act_output, tokens_per_expert)
if self.activation_recompute:
    self.activation_checkpoint.discard_output_and_register_recompute(output)
```

NVIDIA 加这个功能时量过这笔交换：在 H100 上，一个 14 层、用 DeepSeek-V3 层形状的模型，卸载它的归一化输入和两个专家输入后，峰值 reserved 少 18%，吞吐损失 2%。

卸载只在它的拷贝还藏得住的时候省内存，而有三个设置决定它搬多少。`--activation-offload-fraction` 只按前向顺序卸载合格组的前一部分；`--min-offloaded-tensor-size` 跳过小张量，默认一百万个元素以下。靠后的流水 stage 同时进行的 micro-batch 更少、需要的内存也更少，所以 `--delta-offload-bytes-across-pp-ranks` 让每个 stage 比前一个少卸载一些。

重算与卸载缩小的是「每层保存的激活」。而一层也可能保留远超其激活所需的内存，这发生在你用 CUDA graphs 把这一步录下来之后：它重放录好的 kernel 序列、不要 CPU，于是每个 buffer 的大小必须事先定死。在不丢 token 的 MoE 里，一个 rank 收到的 token 数逐步变化，所以每层只能按你允许的最大负载预留专家 buffer：用 `--moe-expert-rank-capacity-factor` 配合 HybridEP 或 NCCL EP 后端时，这个上限就是「系数 × 完全均衡时的负载」。每层会一直占着这份预留直到反向跑完，而 [NVIDIA 的 Megatron Core MoE 报告](https://arxiv.org/html/2603.07685v2#S4.SS3.SSS7.Px3) 指出，一层真正为反向留着的激活往往比它小十倍以上。

Megatron 的实验性分页暂存 `--moe-paged-stash` 把「一层算在多大的空间里」与「它存多大的空间」分开。所有层都在一块按预留大小分配的共享 buffer 里算，每层前向之后，另一条 stream 只把真实的 token 拷进小的定长页——和卸载把张量拷到 CPU 很像。它需要 `--use-transformer-engine-op-fuser` 和 FP8 或 NVFP4 专家，而且因为它自己已经持有专家激活，Megatron 拒绝把它与卸载 `expert_fc1` 或 `moe_act` 组合。

每项改动前后都比峰值内存与步时，因为内存不够的运行会在最忙那个 rank 上 OOM，而「拷贝比计算还久」的卸载会表现为反向变长。所以要看的峰值是**最忙那个 rank** 的峰值。Megatron 会打印数据并行 rank 0 的峰值（[`common_utils.py#L407-L427`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/utils/common_utils.py#L407-L427)），并通过 [2.4](#s-2-4) 的启动器把最后一个 rank 的峰值发到 [Trackio](https://huggingface.co/docs/trackio)。要找最忙的 rank，就在每个日志间隔取各 rank `torch.cuda.max_memory_allocated()` 的最大值，或者用 `--record-memory-history` 记逐 rank 快照。

**什么时候用它**：重叠放不进内存时，重算廉价模块（`--recompute-granularity selective --recompute-modules layernorm moe_act`）并卸载大输入（`--fine-grained-activation-offloading --offload-modules core_attn attn_proj expert_fc1`），而不是全量重算每一层——后者会挡掉重叠。步时变长就调低 `--activation-offload-fraction`。重算 `layernorm` 或 `moe_act` 不能与 FP8 的 delayed 配法一起跑（[`transformer_config.py#L2422-L2433`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L2422-L2433)）；在 `ncclep` 后端下，重叠也不能与卸载 `expert_fc1` 一起跑（[`#L3606-L3618`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L3606-L3618)）。

到目前为止，你一直在不动路由器选择的前提下调系统。那如果让路由器把网络也考虑进去、代价付在训练上呢？

### 组限路由（Group-limited routing）

一个 token 的 top-k 专家可能落在专家并行组的任何位置。一旦这个组跨了好几个节点，一个 token 就可能需要多个节点上的专家，它的拷贝在 dispatch 与 combine 时都会穿过更慢的节点间网络。普通 top-k 路由不管专家住在哪，所以没什么能阻止一个 token 的专家散在很多节点上。

DeepSeek-V3 用组限路由补上这个约束：在挑之前先限定能看哪里，「以限制训练期间的通信开销」。它的专家放在 8 个节点上，对每个 token 先把每个节点上该 token 的 top-2 专家分数（含 [3.3](#s-3-3) 的均衡偏置）加起来给节点打分，保留最好的 4 个节点，再在这些节点上的专家里取 top-k，于是每个 token 最多到达 4 个节点。把「设备」换成「节点」，下图把同一条规则画在标准 top-k 和 MoGE 旁边。MoGE（Mixture of Grouped Experts）是华为 Pangu Pro MoE 用的路由，它把专家平均分成每个设备一组，从每组里各取同样多的专家。用 8 张 GPU 做专家并行训练出来的 Intern-S1-Pro，就是用同样方式从 8 个组里各取 1 个。

> **图 3.7-g** 标准 top-k / 组限路由 / MoGE 的专家选择对比（原站为交互图，此处从略）

逐设备比较：组限路由给每个 token 能到达的设备数封顶，而 MoGE 让每台设备负载均匀、把每个 token 都发给所有设备。

```python
# megatron/core/transformer/moe/moe_utils.py#L713-L718（给组打分，保留最好的几个）
group_scores = (
    scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
)
group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
group_mask = torch.zeros_like(group_scores)
group_mask.scatter_(1, group_idx, 1)
```

```python
# megatron/core/transformer/moe/moe_utils.py#L721-L728（在保留的组内取 top-k）
score_mask = (
    group_mask.unsqueeze(-1)
    .expand(num_tokens, num_groups, num_experts // num_groups)
    .reshape(num_tokens, -1)
)

masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)
```

这个上限省下多少流量，取决于 dispatcher。Megatron 的 `alltoall` dispatcher 每个专家送一份 token 拷贝；而配 [DeepEP](https://github.com/deepseek-ai/DeepEP) 的 `flex` dispatcher（它的 kernel 跟着 DeepSeek-V3 的组限路由走）每个节点送一份、再在节点内用 NVLink 转发。对我们那个 7B-A1B 模型的一层（64 专家、top-8、hidden 2048、bf16），摊在 2 到 8 个节点上，下图统计了两种 dispatcher 下、有无这个上限时穿网走掉的字节。

> **图 3.7-h** 穿网字节数：`alltoall` vs DeepEP，有无节点上限（原站为交互图，此处从略）

用 `alltoall` 时，上限只是把 token 的拷贝挪到更少的节点上，因为每个被选中且不在本节点的专家仍然各拿一份；图里的路由器随机给专家打分、不管 token 落在哪个节点，于是平均下来穿网字节和不加上限一样多。用 DeepEP 时，token 到达的每个节点花一份拷贝，所以给节点封顶就等于给字节封顶。到达的节点更少，也意味着每个 token 能选的专家更少——这笔代价付在质量上而不是时间上，不过很小：Xue 等人用组限训练一个 13B 模型时，loss 只高了那么一点。

近期 MoE 倾向于去掉这个上限。V3 用了它的 DeepSeek 在 V4 里去掉了；GLM、Kimi、Nemotron、MiMo、Hy4 也都让每个 token 到达任何专家。InclusionAI 的 Ling 与 Ring 仍然保持 DeepSeek-V3 的那种形式（8 组、每 token 留 4 组），SK Telecom 的 A.X-K2 保留了一个更宽松的版本。

token 扇出到很多节点时这一步就慢，所以盯两个数。第一是每 token 到达的节点数，从路由 trace 与「哪个 rank 持有哪个专家」的映射离线算（[A.1](#s-a-1)）。第二是每个 EP rank 收到的 token 数，用 `--save-tokens-per-expert-interval` 写出的计数求和（[A.1](#s-a-1)）。

先试上面那些建议，因为它们不改训练；只有当跨节点 All-to-All 仍然占很大份额时才动这个上限：上限会改变 token 能挑哪些专家，也就改变了模型学到什么。它只在专家并行跨多个节点、且 dispatcher 按节点只送一份时才有用，也就是配 DeepEP 的 `flex` dispatcher。那时把 `--moe-router-num-groups` 设成节点数、`--moe-router-group-topk` 设成每个 token 最多能到几个节点，并配一个不是 `quantile_balancing` 的均衡器（它不能与上限一起跑）。否则就别开：单节点上它省不了流量，用 `alltoall` 时平均也省不了。

节点上限限制的是 token 的拷贝去哪，可路由偏爱的专家仍然照单全收所有送来的 token，持有它的那个 rank 就是最后一个跑完的。你该把这个专家的多余 token 挡在门外，还是把它拷到清闲的 rank 上？两种做法各付什么代价？

### 容量与副本（Capacity and replicas）

持有热门专家的 rank 最后跑完自己那份，其他 rank 在下一个同步点等它。默认情况下 Megatron 接受这个等待、保留每一个 token：不设 `--moe-expert-capacity-factor` 时路由是不丢 token 的（[`transformer_config.py#L1009-L1011`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L1009-L1011)），一个专家会收下路由器发给它的全部 token。

容量系数给这个数封顶。每个专家只保留概率最高的那些 token、直到容量上限（[`transformer_config.py#L1024-L1028`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L1024-L1028)），超出的 token 跳过这个专家。这换来一些速度（没有专家能以超过容量的 token 数拖住这一步），也封住内存（没有专家会为超过容量的 token 存激活）。要静态形状，再加上 `--moe-pad-expert-input-to-capacity`。

容量系数省出的内存还能换成更好的并行布局。NVIDIA 的 Vavre 等人把一个 Llama 3-8B upcycle 成 8 专家、top-2 路由的 MoE，在 128 张 GPU 上，容量系数 1 且不用[张量并行](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)跑到每 GPU 463 TFLOP/s，而张量并行 2 的不丢 token 训练是 392。作者把这份收益归给更小的显存占用——它让他们能用更少的模型并行。

另一个答案是保留所有 token，转而复制热门专家。下图给出处理热门专家的三种方式：不丢 token 路由、容量系数、以及按 ECHO 的方式放置的热门专家副本（ECHO 出自 Megatron Core 团队那份报告，在训练中复制热门专家）。

> **图 3.7-i** 一个偏斜的玩具 batch：不丢 token / 容量系数 / 在空 rank 上放副本（原站为交互图，此处从略）

在一个 8 专家、4 rank 的偏斜玩具 batch 上，容量系数丢掉超出一个专家容量的选择（主要是热门专家的），并且按到达顺序丢——而 Megatron 默认会保留概率最高的那些。每个被丢掉的选择都跳过它的专家，于是画的这一步变短、模型输出也变了。把副本放在最闲的 rank 上，能让玩具里最忙的 rank 接近均值，且什么都不丢。

```python
# megatron/core/transformer/moe/router.py#L1029-L1038（在 top-k 之后丢）
# Apply token dropping to probs and routing_map.
if self.config.moe_expert_capacity_factor is not None:
    probs, routing_map = apply_router_token_dropping(
        probs,
        routing_map,
        router_topk=self.topk,
        capacity_factor=self.config.moe_expert_capacity_factor,
        drop_policy=self.config.moe_token_drop_policy,
        pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
    )
```

```python
# megatron/core/transformer/moe/moe_utils.py#L1079-L1112（每专家容量，保留最高概率）
# Calculate expert capacity
expert_capacity = get_capacity(
    num_tokens=num_tokens * router_topk,
    num_experts=num_experts,
    capacity_factor=capacity_factor,
)

# Create capacity mask based on drop policy
if expert_capacity > num_tokens:
    # No need to drop tokens if capacity exceeds the number of tokens
    capacity_mask = torch.ones_like(routing_probs).bool()
else:
    if drop_policy == "probs":
        _, capacity_indices = torch.topk(routing_probs, k=expert_capacity, dim=0, sorted=False)
        capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()
    elif drop_policy == "position":
        _, capacity_indices = torch.topk(
            routing_map.int(), k=expert_capacity, dim=0, sorted=False
        )
        capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()
    else:
        raise ValueError(f"Invalid drop_policy: {drop_policy}")

# Apply capacity constraints
if pad_to_capacity:
    final_map = capacity_mask
    final_probs = routing_probs * final_map
else:
    # Get exceed mask and maskout exceeded probs and indices
    final_map = torch.logical_and(routing_map, capacity_mask)
```

真实运行丢的比玩具少：容量系数 1.5 时，华为 Pangu 团队在一个 20B MoE 上看到约 6% 的 token 被丢，在他们的 718B 上约 8%。副本则能把各 rank 校得几乎完全一致。Wei 等人的 UltraEP 为每个 micro-batch、每一层重新规划副本，把三次训练（DeepSeek-V3、Qwen3-235B、GLM-4.5-Air）里最忙 rank 的负载从普通 Megatron-LM 的 1.30 到 2.86 倍均值，压到 1.01 到 1.03 倍。

副本在训练里比在服务里贵。每个副本前向要把专家权发送到它的 rank，反向要把梯度归约回原处，所以 ECHO 的规划器尽量少复制：它统计每个专家超出「每 rank 均值负载」的 token，找到清闲 rank 上的空位，把前者塞进后者，算完就丢掉副本。均衡的 rank 有两份回报：等得更少、buffer 更小——因为 CUDA graphs 会按最坏情况定它静态 buffer 的大小，也正是 [3.7 内存](#s-3-7-memory) 里分页暂存要绕开的那份最坏情况。

服务端有同样的问题但没梯度，所以它的副本留得比 ECHO 久得多（后者每次前向都重新规划）。DeepSeek-V3 按在线服务的负载、以约 10 分钟为间隔重挑它的冗余专家。DeepSeek 用来服务 V4.1-Flash 的引擎 DeepInfer 给每张 GPU 留出空槽放最忙专家的副本，按最近一个窗口统计每个专家的负载，prefill 每 1,000 步、decode 每 10,000 步重新放置一次，于是最慢的 GPU 保持在均值的 1.09 倍以内——就是 3.3 节引的那个数。

丢 token 会掉质量吗？公开的比较说会掉一些 loss。华为 Pangu 团队用容量系数 1.5 和不丢 token 各训了一个 20B MoE，不丢的那次全程 loss 更低，且差距随专家数增多而变大。Gale 等人做 MegaBlocks 时也一样：不丢任何 token 时 loss 最低。

下游基准可能掩盖这个差距：Vavre 等人 upcycle 的 Llama 3 MoE 在他们试的每个容量系数下、以及不丢 token 时，MMLU 都在一分以内。丢掉的 token 落在哪也很要紧：Meta 的 MobileMoE 带着容量系数预训练，微调时切回不丢 token——因为在指令-回答对里丢 token 会切掉内容。

带容量上限训练的模型，不必带着上限服务。Zoph 等人用比训练时更高的容量系数评估他们的 ST-MoE 时，比较里每个模型分数都更好。DeepSeek-V2 就是为这个切换设计的：训练时在每个设备上丢 token，但约 10% 的训练序列从不丢，于是推理时它可以自己决定丢不丢，还照样匹配它训练时的行为。

开了容量系数，token 是被静默丢掉的，所以要把「丢得多频繁」和验证 loss 一起看。`--moe-routing-trace-path` 写出的路由 trace 会跳过每一条「token 被丢」的记录（[`router_trace.py#L353-L360`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router_trace.py#L353-L360)）。要数丢了多少，就在路由器上挂一个前向 hook，逐层算被丢的 token-专家分配占比：`1 - kept / (tokens * topk)`，其中 `kept` 是路由图里剩下的分配数。

**什么时候用它**：默认保持不丢 token。当一个热门专家拖住这一步、而清闲的 rank 还有地儿时，给它放一个副本就能把 rank 校平、又不动 loss（不过 ECHO 的复制目前还是一个未合并的 PR）。内存紧张、又能接受略高的 loss 时用容量系数；要静态形状还要加 `--moe-pad-expert-input-to-capacity`。微调时切回不丢 token，因为那时丢掉的 token 落在指令-回答对里。

不管丢不丢，专家留下的每个 token 都要过它的 GEMM，而到目前为止这些 GEMM 都跑在 bf16 里。换成 FP8 这样的更低精度，能让这份活更便宜、存的激活更小，又不伤 loss 吗？

### 精度（Precision）

MoE 层的大部分计算在矩阵乘里，而大部分激活内存在这些乘法为反向保存的输入里。相对 bf16，FP8 把两者的字节都减半，而且它的 Tensor Core 操作在 Hopper 与 Blackwell 上更快。这份收益能落到步时上多少，取决于乘法占这一步多大比例、以及量化那些张量要付多少——对大规模 MoE 训练，Megatron Core 团队报告端到端约 10% 到 25%。

不是每个操作都能吃这个低精度。下图给出 FP8 在一层里（前向与反向）跑到哪，沿用 DeepSeek-V3 训练时的切分：每个线性层的前向乘法与两次反向乘法都跑 FP8，而 embedding、输出头、归一化、注意力核心与路由器保持 BF16 或 FP32。

> **图 3.7-j** 一层里 FP8 与 BF16/FP32 的分界（原站为交互图，此处从略）

省下的内存来自反向。权重梯度那次乘法要读前向保存的输入，把它留在 FP8 就把这份内存减半——对 DeepSeek-V3 来说约合每 GPU 16 GB 激活。而主权重、权重梯度和优化器状态保持更高精度。

路由器是要保护的部分。它的分数决定每个 token 的专家，那里的量化噪声会改变选择、让训练不稳，这也是 Megatron Core 报告把它留在 FP32 的原因。在 Megatron 里路由器不是 [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 层，所以 FP8 根本到不了它，而 `--moe-router-dtype fp32` 设定它数学运算的精度（[`router.py#L125-L130`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router.py#L125-L130)，[3.1 精度](#s-3-1-precision)）。量化后的分组 GEMM 需要每个专家的 token 数补齐到对齐尺寸，这由 `--moe-router-padding-for-quantization` 提供（[`transformer_config.py#L851-L854`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L851-L854)）。

「配法（recipe）」决定缩放系数跟数据跟得多细。每个张量一个缩放系数时，最大的离群值决定其余所有值的范围，所以 DeepSeek-V3 与 Ling 2.0 按 1×128 元素为一格缩放激活与梯度、按 128×128 为一块缩放权重。Megatron Core 报告推荐 Hopper 上用这种分块配法、Blackwell 上用 MXFP8（每 32 个元素一个缩放系数）。用 `--fp8-format` 与 `--fp8-recipe` 选，分块配法需要 Transformer Engine 2.3 或更新（[`fp8_utils.py#L799`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/fp8_utils.py#L799)）。

FP8 也能过网。DeepSeek-V3 在 dispatch 前把 token 量化成 FP8——反正专家的第一层就是以 FP8 读它们；「MoE 下投影之前对激活梯度用了相似策略」，而「forward 与 backward 的 combine 组件」仍留在 BF16。在 Megatron 里，`--moe-dispatch-fwd-dtype mxfp8` 与 `--moe-combine-bwd-dtype mxfp8` 让 dispatch 载荷与经 combine 回传的梯度走 MXFP8。两者都需要 flex dispatcher 的 `ncclep` 后端（[`transformer_config.py#L2150-L2158`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L2150-L2158)），而 MXFP8 本身要 Blackwell 级 GPU（[`#L656-L661`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L656-L661)）。

长跑报告说 loss 撑住了。DeepSeek 用 FP8 训了一个 16B 和一个 230B 的 MoE，各约一万亿 token，二者相对 BF16 基线的误差都在 0.25% 以内。Ling-1T 在 900B token 上保持在同一量级（平均约 0.1%），同时 FP8 把它的 MFU 提高了约 15%。

再往下一档是 4 bit，头几个大 MoE 已经在那儿预训练了。NVIDIA 用 NVFP4（一种 4 位浮点格式：每 16 个元素一块配一个 FP8 缩放系数，再加一个逐张量的 FP32 缩放系数。）在 25T token 上训了 Nemotron 3 Super（120B 参数、12B 激活），大多数线性层的三次乘法都跑在里面。例外标出了 4 bit 哪里不够：网络最后 15%、注意力投影、embedding 与多 token 预测层留在 BF16，Mamba 的输出投影跑 MXFP8（它在 NVFP4 下下溢了）。这套配法还按二维块缩放权重（于是前向反向看到同一份量化权重）、对权重梯度乘法的输入做随机 Hadamard 变换、并对梯度做随机舍入。Nemotron 3 Ultra（550B、55B 激活）用同一套配法跟了 20T token；当 NVIDIA 从其 checkpoint 分叉出 BF16 运行对比时，与它们的 loss 差距平均在 0.4% 以内。

FP4 把保存的输入压到 BF16 的四分之一（FP8 是一半），而原生 FP4 乘法要 Blackwell 级 GPU。Hopper 没有 4 位 Tensor Core 指令，但浙江实验室的 Zhang 等人用一个 671B MoE 做了软件模拟的 MXFP4：把激活与 dispatch 流量存成 MXFP4，而乘法仍走 FP8，峰值激活内存降了 14.8%，收敛与一个强的 FP8 基线持平。省下来的内存让它少重算，比「装得下的最好 FP8 配置」还快 12.5%。

一次 FP8 运行的 loss 应该全程贴近 BF16，所以用相对差值与同配置的 BF16 运行比。DeepSeek 的 16B 与 230B 运行、以及 Ling-1T 都在 0.25% 以内，所以远超这个数、或者一直在拉大的差距，就指向配法出了问题。用 NVFP4 时，还要数专家层里恰好为零的权重梯度元素：在 Nemotron 3 Super 里它们到预训练结束时涨到全部参数的 7%（某些专家里低范数的通道缩到了零），而在 Nemotron 3 Nano 上做的一个更小的消融里，切回 BF16 就让这个计数回到基线。

**什么时候用它**：线性层占这一步大头时用 FP8——Hopper 上用分块配法、Blackwell 上用 MXFP8，并把路由器、归一化与注意力核心留在更高精度。在大 MoE 上，预期吞吐多约 10% 到 25%，loss 与 BF16 相差不到零点几个百分点。在 Blackwell 上，NVFP4 是再下一档，用 Nemotron 3 那套配法把最后几层、注意力投影和 embedding 留在 BF16。

### 该盯什么

| 部位 | 症状 | 该盯的信号 |
|---|---|---|
| 剖析 | 「慢」：这一步比应有的更久 | 先看步时或 TFLOP/s；再从 trace 看 All-to-All 在步里留下多少裸露 |
| 互联 | 多节点上「慢」：NCCL 退回 TCP socket 时，我们的 EP16 代理慢约 25 倍，42.0 vs 1.7 TFLOP/s | `NCCL_DEBUG=INFO` 下看 "Using network" 是否是你要的传输层：InfiniBand 看 IB，EFA 上的 OFI 插件看 Libfabric；Socket 表示回退了。然后用 nccl-tests 比较节点内与跨节点的 alltoall 总线带宽（2.1 节），以及节点内 vs 跨节点 EP 的 TFLOP/s |
| 重叠 | 「慢」：All-to-All 没藏在计算后面 | 打开前后每 GPU TFLOP/s（`--log-throughput`）。裸露份额只用来解释变化 |
| 内存 | 「OOM」：某个 rank 爆了。「慢」：重算、或比计算还久的卸载重新加载，增加了步时 | 每次改动后比较步时与峰值内存。看最忙 rank 的峰值：各 rank `torch.cuda.max_memory_allocated()` 的最大值，或逐 rank `--record-memory-history` 快照。反向变长可能暴露比计算更久的卸载重载 |
| 组限路由 | 「慢」：每个 token 扇出到多个节点 | 每 token 到达的节点数（路由 trace 与专家到 rank 的映射），以及每个 EP rank 收到的 token 数（token 计数），A.1 节 |
| 容量与副本 | 「学不好」：token 被静默丢掉。「OOM」：某个 rank 爆了 | 逐层被丢的 token-专家分配占比 `1 - kept / (tokens * topk)`（路由器前向 hook），与验证 loss 并看。路由 trace 会跳过被丢的记录 |
| 精度 | 「慢」：相对 bf16 没提速。「学不好」：loss 比 bf16 高 | 与同配置 BF16 运行比全程相对 loss 差；DeepSeek 的 16B、230B 与 Ling-1T 都在 0.25% 以下。NVFP4 下还要数恰好为零的专家权重梯度元素：Nemotron 3 Super 里达到全部参数的 7% |

到这里你已经拿到了一个 MoE 层的每个部件：从挑专家的路由器，到它们计算时用的精度。等训练结束、模型要被服务出去的时候，这些选择里还有哪些仍然要紧？

---

**导航**：[全书目录](index.md) · [← 第 3 章（下）块布局与训练](03-4-block-layout-and-training.md) · [第 4 章 从训练到服务 →](04-serving.md)
