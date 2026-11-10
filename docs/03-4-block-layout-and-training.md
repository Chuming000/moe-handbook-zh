# 第 3 章 解剖 MoE 训练：块布局与训练

> 说明：原站里的图都是交互式组件，这里无法复现，一律保留图注并标注「（原站为交互图，此处从略）」。原站大部分图不给编号，本文按**节内顺序**编号（如 图 3.5-a）。

## 3.5 块布局

把模型想象成一摞 block，每个 block 是注意力后面接一个稠密 MLP 或一个 MoE 层。每个 MoE 层都要用一次 All-to-All 把 token 发出去、再收回来，所以 MoE 层放在哪，决定了每一步要做多少次 All-to-All，也决定了哪些层的均衡最难做。而每个 block 内部怎么接线，又决定了每次 All-to-All 能有多少藏在注意力后面。

前沿模型把 MoE 放在哪、以及怎么把它接到注意力上。稠密层数指主干，不含预测模块。

| 模型 | 稠密层 | 块接线 |
|---|---|---|
| DeepSeek-V3 | 前 3 层稠密，其余 58 层 MoE | 注意力后接 FFN |
| DeepSeek-V4.1-Flash | 全部 40 个 block 都是 MoE | 单遍 mHC 残差连接 |
| GLM-5.3 | 前 3 层稠密，其余 75 层 MoE | 注意力后接 FFN |
| Kimi K3 | 第 1 层稠密，其余 92 层 MoE | 注意力后接 Stable LatentMoE；注意力残差 |
| Llama 4 Maverick | 稠密与 MoE 交替，从稠密开始 | 每两层一个 MoE |
| LongCat-Flash | 稠密 FFN 与 MoE 在每个 macro-block 内共存 | 捷径连接的 MoE 与后续的注意力、FFN 工作重叠 |
| MiMo-V2.6-Pro | 第 1 层稠密，其余 69 层 MoE | 混合注意力块后接 FFN |

第一个布局决定是：哪些层真的有专家。你可以每层都放 MoE，可以留几层稠密，也可以让两种交替。该选哪个？

### 稠密层

稠密层没有路由器也没有专家，所以不产生 All-to-All。图 3.5-a 把三种布局的 All-to-All 次数并排数出来：每层都是 MoE（如 Qwen3、gpt-oss、MiniMax M2、Llama 4 Scout）、开头放几层稠密、每两层一个 MoE。开头稠密只是把最早的几个 MoE 层换掉，DeepSeek-V3 就保留前 3 层稠密；而每两层放一个 MoE 能把 All-to-All 减半，Llama 4 Maverick 就是稠密与 MoE 交替。两种做法都拿一部分 MoE 容量，换更少的 All-to-All 和更少的参数。

> **图 3.5-a** 三种布局的 All-to-All 次数：全 MoE / 开头稠密 / 隔层 MoE（原站为交互图，此处从略）

图里那个玩具模型的「路由随深度变化越来越不跟 token id」，所以第一层最慢：开头放稠密能去掉它最慢的两层，却只砍掉 16 次 All-to-All 里的 4 次；而隔层 MoE 把 All-to-All 减半，却保留了第二慢的那层。在 Ling 的 60 层模型里，开头稠密的代价很小：把前 1 到 3 层改成稠密，在等 FLOP 下效率几乎没有变化，效率杠杆接近 1。

另一个理由是均衡，但证据更薄。DeepSeekMoE 观察到第一层的均衡收敛更慢，Ling 也说开头稠密能缓解浅层不均衡，不过它那组消融只报了 loss。后来的报告彼此矛盾。有些干脆让浅层不碰学出来的路由器：图像生成的 MoE Nucleus-Image 在没有 3 层开头稠密时会发散，而 DeepSeek-V4 对前 3 个 MoE 层做了哈希路由。

另一些仍然在浅层路由，转而治不均衡。Solar Open 每层都是 MoE，用序列级辅助损失修掉浅层不均衡。Sigma-MoE-Tiny 让每个 token 只从 96 个专家里选 1 个，只靠辅助损失时它的下层塌了：损失走了捷径，把路由器概率压成均匀分布，而有些专家仍然几乎拿不到 token。它的修法是训练早期在前 8 层激活更多专家，之后再退回 1 个。稠密层也不总能解决问题：Motif 3 前两层稠密，可它最早的几个 MoE 层在训练早期仍然最不均衡；而 ERNIE 5.0 在文本任务上没发现第一层有严重不均衡。

[Megatron](https://github.com/NVIDIA/Megatron-LM) 用列表形式在 [`--moe-layer-freq`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/models/gpt/gpt_layer_specs.py#L635-L645) 里指定布局，每层一个值，0 表示稠密、1 表示 MoE：`"([0]+[1]*15)"` 是开头一层稠密，`"([0,1]*8)"` 是 Maverick 那样稠密在前的交替。这个开关也接受整数，而整数形式会把 MoE 放在第 0 层，所以 `--moe-layer-freq 2` 是从 MoE 开始，而不是像 Maverick 那样从稠密开始。稠密层的宽度来自 `--ffn-hidden-size`，与专家宽度分开。

```python
# megatron/core/models/gpt/gpt_layer_specs.py#L635-L645（整数或列表模式）
if isinstance(config.moe_layer_freq, int):
    moe_layer_pattern = [
        1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)
    ]
elif isinstance(config.moe_layer_freq, list):
    moe_layer_pattern = config.moe_layer_freq
    assert len(moe_layer_pattern) == config.num_layers, (
        f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
        f"expected {config.num_layers}, "
        f"current moe layer pattern: {config.moe_layer_freq}"
    )
```

```python
# megatron/core/models/gpt/gpt_layer_specs.py#L652-L659（逐层选稠密或 MoE）
layer_specs = []
for layer_number in range(config.num_layers):
    if moe_layer_pattern[layer_number] == 1:
        layer_specs.append(moe_layer_spec)
    elif moe_layer_pattern[layer_number] == 0:
        layer_specs.append(dense_layer_spec)
    else:
        raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")
```

上面那些报告都在浅层看到了不均衡；即便那里放的是稠密层，紧跟着的第一个 MoE 层仍然可能最不均衡，Motif 3 就是这样。这些「第一个 MoE 层」才是要盯的对象。用 [`--moe-per-layer-logging`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/moe_logging.py#L382-L405) 追它们的辅助损失（[2.4](#s-2-4) 的启动器会把逐层数值发到 [Trackio](https://huggingface.co/docs/trackio)），用路由 trace 追它们的 MaxVio（[A.1](#s-a-1)）。稠密层没有路由器可看。[z-loss](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/moe_utils.py#L155-L172) 量的是路由器 logits 长得有多大，不是 token 摊得多均匀。

想在几乎不付代价的前提下减少 All-to-All，就用开头稠密：在 Ling 的 60 层模型里，1 到 3 层开头稠密在等 FLOP 下几乎没改变效率。但别指望它治浅层不均衡：DeepSeekMoE 和 Ling 都没给出这方面的数据，而 Motif 3 在前面放了两层稠密之后，第一个 MoE 层照样不均衡。

你保留的每个 MoE 层仍然要把 token 发出去再收回来；而在一个 MoE block 里（MoE 跑在注意力的输出上），GPU 有相当一部分时间在等。能不能把 block 重新接线，让一部分通信在注意力计算时同时进行？

### 块接线

在一个 MoE block 里，每次 All-to-All 两侧都绑着计算。路由器要读注意力的输出，所以注意力没算完就发不出去；下一层要读 MoE 的输出，所以 combine 没做完后面什么都开不了。[Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=a3:_math_for_compute/communication_overlap) 讲了怎么把通信与计算重叠的一般做法。

重接 block 能去掉其中一条依赖、或者两条都去掉，省出来的通信就能和计算重叠。并行 block 把同一个输入同时喂给注意力和 MoE，Cohere 的 North Mini Code 与 Moondream 3 就是这么做的，于是 dispatch 在注意力算的时候就走。combine 更难藏，因为它只能和同层剩下的工作重叠——在 [Megatron 的捷径 block](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/models/hybrid/shortcut_block.py#L547-L593) 里就是注意力的输出投影和共享专家；两样都没有时，它就是裸露的。捷径连接的 MoE 把 combine 也藏起来了，藏在它旁边跑的稠密计算后面，LongCat-Flash 和 Pos-2（ScMoE 的首选变体、Cai 等人提出的捷径连接 MoE）都是这样。图 3.5-b 在一个 micro-batch 里比较了三种接线，按它示意的时长，这一层里裸露的 All-to-All 占比从 MoE block 的 33% 降到捷径的 7%。

> **图 3.5-b** 一个 micro-batch 内的三种块接线：MoE block / 并行 block / 捷径连接 MoE（原站为交互图，此处从略）

```python
# megatron/core/models/hybrid/shortcut_block.py#L102-L127（注意力与下一个 MoE 配对）
compute_layer = layers[physical_index]
paired_type = layer_type_list[physical_index]
if paired_type not in SUPPORTED_SHORTCUT_PREDECESSORS:
    raise ValueError(
        "Shortcut MoE must be preceded by a Mamba, GDN, or supported attention layer"
    )

supports_two_stage = (
    isinstance(compute_layer, TwoStageAttentionLayer)
    and compute_layer.supports_two_stage_attention()
)
if not supports_two_stage:
    raise ValueError(
        f"Shortcut attention layer {paired_type!r} does not support two-stage attention"
    )
moe_layer = layers[physical_index + 1]
grouped_layers.append(
    ShortcutMoEBlock(
        compute_layer,
        moe_layer,
        overlap_a2a=config.moe_shortcut_parallel,
        attn_local_idx=physical_index,
        moe_local_idx=physical_index + 1,
    )
)
physical_index += 2
```

```python
# megatron/core/models/hybrid/shortcut_block.py#L490-L502（路由器读这一对的输入）
# Launch the moe_router
with quant_context_factory(moe_config, self.moe_layer_idx):
    moe_hidden_states = self._read_shortcut_hidden(
        moe_hidden_states, recompute_context=attn_recompute_context
    )
    route_input, route_probs = self._moe_router_preprocess(
        shortcut_hidden=moe_hidden_states,
        padding_mask=padding_mask,
        packed_seq_params=moe_packed_seq_params,
        recompute_context=attn_recompute_context,
    )
    if overlap:
        self.route_ready_event.record(torch.cuda.current_stream())
```

图里那个并行 block 让 dispatch 与注意力重叠，而 combine 仍然裸露；捷径连接的 MoE 则用左右的层盖住了大部分 dispatch 和 combine。代价落在路由器上：它现在读的是「被重叠的那段计算之前」的流，永远看不到那段计算的输出。

模型因此学得更差吗？Cai 等人和 LongCat-Flash 团队看到的是差不多。Cai 等人在 LLaMA2 风格模型的每个 block 里都放 MoE，接成 Pos-1（Megatron 实现的那个 ScMoE 变体），在零样本基准上平均略高于标准 top-2 MoE。在 LongCat-Flash 里，带不带捷径的损失曲线几乎分不出来。AMD 的 FarSkip-Collective 把 block 接成「计算用的残差里还不含最新通信回来的输出」；装进 Instella-MoE 后，在 200B 训练 token 之后与 MoE block 的基准均值持平，尽管 AMD 那两份报告的表里它在 MMLU 和 HumanEval+ 上是输的。

换回来的是速度。ScMoE 在一个 8 卡 NVLink 节点上训练比标准 top-2 MoE 快 1.12 到 1.14 倍，Instella-MoE 的预训练吞吐提升 12.7%。LongCat-Flash 直接量了裸露通信本身：用捷径连接的 MoE 之后，dispatch 与 combine 里没被重叠的时间占比从 25.3% 降到 8.4%（在他们的 32 卡专家并行组里，用管线化的 [all-gather](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=gather_%26_allgather) 与 [reduce-scatter](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=scatter_%26_reducescatter)，而不是 All-to-All）。

在 Megatron 里，图里那行「并行」（ScMoE 的 Pos-1）对应的是 [`--moe-shortcut-connection`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L796-L813)：让每个 MoE 读它前面那个注意力层的输入，于是路由器不用等那次注意力。共享专家仍然读注意力的输出（这是图里没画的一个细节），路由专家则在主 stream 上跑在[注意力核心之后、输出投影之前](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/models/hybrid/shortcut_block.py#L517-L558)，就像图里那行的专家排在注意力之后。光重接不会重叠任何东西：只有 `--moe-shortcut-parallel` 把 All-to-All 放到侧 CUDA stream 上，它们才重叠。可选的、给合并后输出加一层归一化是 `--moe-shortcut-post-norm`。三个开关都只在 upstream main 上，而且只能在 hybrid stack 里用，不能和 [FSDP](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/arguments.py#L984-L990)、CUDA graphs、全量重算或 `--moe-shared-expert-overlap` 一起用。不改 block 的话，Megatron 也能把 All-to-All 藏到另一个 micro-batch 后面（[3.7 重叠](#s-3-7-overlap)）。

裸露的 All-to-All 让每一步变慢，而 trace 能告诉你漏了多少。抓一条 nsys 时间线（`--profile` 配合 `--profile-step-start`、`--profile-step-end`，见 [3.7 剖析](#s-3-7-profiling)），把侧 stream 上的 All-to-All kernel 和主 stream 上的计算对比：All-to-All 单独在跑的那几段，就是裸露的部分。

**什么时候用它**：dispatch 裸露、且你能切到 Megatron 的 hybrid stack 时，就用捷径 block（图里「并行」那行）：ScMoE 的 Pos-1 设计在单个 8 卡 NVLink 节点上比标准 top-2 MoE 快 1.14 倍。`--moe-shortcut-connection` 和 `--moe-shortcut-parallel` 两个都要设，只有后者才让 All-to-All 重叠；同时别开 FSDP、CUDA graphs、全量重算和 `--moe-shared-expert-overlap`，它们和它不兼容。

### 该盯什么

| 部位 | 症状 | 该盯的信号 |
|---|---|---|
| 稠密层 | 「学不好」：最早的 MoE 层不均衡，或者路由塌掉。「炸了」：一个图像生成 MoE 没有开头稠密层就发散了。「慢」：每层都 MoE 意味着更多 All-to-All | 稠密块之后头几个 MoE 层的辅助损失（`--moe-per-layer-logging`）与 MaxVio（路由 trace，A.1 节） |
| 块接线 | 「慢」：All-to-All 没藏在计算后面 | All-to-All 裸露了多少：在 nsys 时间线里（3.7 剖析），把侧 stream 的 All-to-All kernel 与主 stream 的计算对比 |

路由器、均衡器、专家、块布局都选完之后，架构就完整了，剩下的就是把它训出来。训练该从哪儿开始？用哪个优化器？又怎么知道什么时候出事了？

## 3.6 训练：起点、优化器、稳定性

训练 MoE 会多出一些稠密模型没有的失败模式，做大 MoE 的团队都撞过其中好几种。一个 Switch Transformer 在 bfloat16 下发散了；六个 ST-MoE 基线里有两次跑崩。路由也可能塌到少数几个专家上，而不均衡会自我强化。这些运行到底为什么失败？又怎么能提前发现自己的即将失败？

各模型作者报告的训练选择。初始化与优化器几列标出与本节相关的选择；MTP 一列描述训练方式，可能与发布的推理配置不同。

| 模型 | 初始化 | 优化器 | MTP | 稳定性选择 |
|---|---|---|---|---|
| DeepSeek-V3 | 随机初始化 | AdamW | 1 层 MTP | 梯度范数裁到 1.0；FP8 训练期间 embedding、输出头、MoE 门控、归一化与注意力保持 BF16 或 FP32 |
| DeepSeek-V4.1-Flash | 从零开始 | 线性权重用 Muon，归一化与非矩阵参数用 AdamW | 预训练期间主干不用 MTP；DSpark 之后单独训练 | SwiGLU 截断到 10；按模态分别均衡 |
| Kimi K3 | 从零开始，语言与视觉一开始就联合训练 | 矩阵参数用 Muon，注意力投影按 head 更新 | 1 层 MTP | 权重截断、注意力输出用 FP32、LatentMoE 归一化、SiTU-GLU、分位数均衡 |
| LongCat-Flash | 先训 14 层，再叠成 28 层 | Adam 家族，超参按 Adam 的缩放规则迁移 | MTP 当作辅助目标 | 隐状态 z-loss、路由器梯度监控、Adam epsilon 取 1e-16、确定性计算 |
| Step 3.5 Flash | 未报告 | Muon，Polar Express 迭代在 FP16 下做 | 3 个 MTP head | 梯度裁到 1.0；专家激活截断；把专家范数与损失一起盯 |

第一个决定是训练从哪开始：随机权重，还是从 [2.4](#s-2-4) 已经训好的稠密模型开始。从稠密 checkpoint 起步能换来什么，又要付什么代价？

### 向上循环（Upcycling）

Upcycling 把稠密 checkpoint 的 MLP 拷进专家，作为 MoE 的起点。和稠密 MLP 一样宽的专家拿到完整一份，窄的拿到一片：在我们那次运行里，稠密 MLP 宽 8192、专家宽 1024，[Megatron](https://github.com/NVIDIA/Megatron-LM) 把稠密 MLP 切成 8 片，每片拷进 64 个专家中的 8 个。不管哪种，MoE 都从稠密那次停下地方接着开始。

拷完权重还有第二个选择：优化器状态与学习率调度要不要一起带过来。Komatsuzaki 等人接着稠密那次调度的位置继续，但只在视觉模型上复用优化器状态，因为语言模型上没有收益。Megatron 的 upcycling 步骤会在第 1 个迭代就把 MoE 存下来、且不带优化器状态（[`training.py#L3069-L3083`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/training.py#L3069-L3083)），所以 MoE 这次是用全新优化器和全新调度开始的。

这个抢跑不是免费的：它花的是稠密那次已经花掉的算力，而从零开始的运行最终会追上来。快慢差别很大：在 OLMoE 里，从零开始的那次在 500B token 之后追上，相当于稠密模型训练量的 25%；而在 Komatsuzaki 等人的稀疏 upcycling 运行里，要花稠密 checkpoint 算力的 120%。下图按 OLMoE 公开的训练日志重画了那个对比：一个从稠密 OLMo-1B（已训 2T token）upcycle 来的 MoE，对一个从零开始训的同样的 MoE，各训 610B token。

> **图 3.6-a** upcycle 自 OLMo-1B 的 MoE vs 从零训练的 MoE（原站为交互图，此处从略）

图里 upcycle 那次在四项指标上都是一开始就领先，从零的那次只是慢慢把差距缩小。它的 HellaSwag 与 MMLU 准确率在约 500B token 处追平（作者把「追上」定在这里），而训练与验证损失要到约 600B token 才追平。

训好的 MoE 也能用同样的方式长大——只是拷的是它自己的专家，而不是稠密 MLP。下图把一个 8 专家 MoE 变成 16 专家，并展示拷贝对路由的影响。

> **图 3.6-b** 把 8 专家按三种方式扩到 16 专家后的路由行为（原站为交互图，此处从略）

在 top-2 路由、每个副本继承原专家路由器那一行的情况下，图里的精确拷贝会打平：每个 token 都挑一个专家和它的副本、权重相同，于是实际只跑了一个不同的专家。给副本加扰动，路由器就能在这一对之间拿到梯度；随机新专家能避免打平，但把路由权重压在未训练的专家上。三种做法都被用来长大模型：Expert Upcycling 每个专家连同路由器那一行一起拷，从 32 到 64；K-EXAONE 2.0 给每个副本加一个随机旋转，从 128 到 256；Expert Upcycling 还试了 Drop-Upcycling，把每个副本的部分权重重新初始化（作者发现最好的是重初始化一半）；EMO 则从高斯分布里抽新专家和新的路由器行。在 Megatron 里，从稠密 checkpoint upcycle 是 `--moe-use-upcycling` 加 `--moe-upcycling-granularity`，而在那个 pin 上还需要 [2.5](#s-2-5) 里说的两处本地修补。

```python
# megatron/training/training.py#L3043-L3083（先建稠密、再 upcycle、再存）
if args.moe_use_upcycling:
    torch.distributed.barrier()
    assert not checkpoint_exists(args.save), (
        "The upcycling destination directory already exists. "
        "Please check if --moe-use-upcycling is mistakenly enabled. "
        "Upcycling should only be set for the first run when converting the dense model. "
        "All subsequent runs should remove this flag. "
    )
    # before changing moe related global args, save them in local variables
    num_experts = args.num_experts
    expert_model_parallel_size = args.expert_model_parallel_size
    moe_ffn_hidden_size = args.ffn_hidden_size

    # set dense model related args in to global args before getting dense model
    args.num_experts = None
    args.expert_model_parallel_size = 1
    args.ffn_hidden_size = moe_ffn_hidden_size * args.moe_upcycling_granularity

    # get dense model
    dense_model_for_upcycling = _build_model_wrapper(wrap_with_ddp=True)

    # recover moe upcycling related args in global args before executing upcycling
    args.num_experts = num_experts
    args.expert_model_parallel_size = expert_model_parallel_size
    args.ffn_hidden_size = moe_ffn_hidden_size

    # execute upcycling
    _, args.num_floating_point_operations_so_far = upcycling_utils.load_and_upcycle_model(
        load_checkpoint,
        unwrapped_model,
        dense_model_for_upcycling,
        load_kwargs={
            'model': dense_model_for_upcycling,
            'optimizer': None,
            'opt_param_scheduler': None,
        },
    )
    args.iteration = 1
    save_checkpoint(
        args.iteration, model, None, None, args.num_floating_point_operations_so_far
    )
```

那次加载时，`load_checkpoint` 拿到的 `ddp_model` 是稠密模型（[`checkpointing.py#L2632-L2635`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/checkpointing.py#L2632-L2635)），并且稠密模型是用「`num_experts` 未设」的参数现建的，而不是用 `pretrain_gpt.py` 启动时建好的那份 config 容器——这样稠密 checkpoint 才能加载进一个稠密模型（[`training.py#L2917-L2925`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/training.py#L2917-L2925)）。

搞砸的 upcycle 会丢掉抢跑优势，所以要拿它的 loss 和「稠密那次继续训同样多 token」比。又因为专家一开始是完全相同的拷贝，还要在它们分化期间逐层看 MaxVio，从路由 trace 离线算（[A.1](#s-a-1)）。

**什么时候用它**：已有稠密 checkpoint、且这次 MoE 训练较短时用。因为从零开始的那次在 OLMoE 里要 25% 的稠密训练 token 才追上，在 Komatsuzaki 等人的运行里要 120% 的算力。upcycling 是一次性的初始化，只在第一次运行设，且在那个 pin 上要先用 [2.5](#s-2-5) 的两处本地修补。

不管你是 upcycle 还是用随机权重，只决定权重从哪儿出发；从第一步起，就是优化器决定它们往哪走。可是每个专家只看见路由给它的 token，那么一个只拿到几把 token 的专家，该被推得和忙碌专家一样远吗？

### 优化器

DeepSeek-V3 用的 AdamW 把每个权重的更新按它自己的梯度历史缩放，所以一个专家走多远，取决于它那些 token 产生的梯度。而近期好几个 MoE 用的、针对权重矩阵的 Muon 换了个尺度的算法：它把每次更新正交化，把非零奇异值往 1 上推，再乘一个只由矩阵形状决定的系数，于是步长不再取决于背后梯度有多大。

Megatron 对每个专家的矩阵各做一次正交化（[`transformer_engine.py#L2665-L2668`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/extensions/transformer_engine.py#L2665-L2668)、[`transformer_config.py#L956`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L956)、[`emerging_optimizers.py#L147-L149`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/optimizer/emerging_optimizers.py#L147-L149)），跟 DeepSeek-V4 一样，于是路由器很少挑中的专家也能走得和最忙的差不多远——只是方向是自己的——而且在它一个 token 都没拿到的 step 里，它还会沿动量继续动。下图用 49 个玩具专家画出了这件事：AdamW、按专家各自做 Muon、把全部专家当一个矩阵做 Muon，三种情况下更新大小与每步 token 数的关系。

> **图 3.6-c** 更新大小 vs 每步 token 数：AdamW / 逐专家 Muon / 全体合一 Muon（原站为交互图，此处从略）

玩具里某个专家的 token 变少时，AdamW 下它的更新随梯度变成噪声而缩小，全体合一的 Muon 下随梯度大小一起缩小，而逐专家 Muon 能把最忙专家的步长压到每步几个 token。再往下，逐专家的步长也开始缩：每个 token 只给梯度加一个方向，token 这么少的时候，连最近几步的动量都填不满这个专家的矩阵。

图说明的是 Muon 怎么挪动每个专家；至于它能不能训出更好的 MoE，是在整模型上量过的。Moonlight（3B 激活、16B 总参）用 Muon 明显好于同样设置下用 AdamW 训练的那份，比较点在约 1.2T token。

上了 Muon 之后还剩一个选择，各家做法不同：用哪个优化器更新路由器。DeepSeek-V4 让它留在 Muon 上，而 Qwen3.8-Next 在 Muon 早期把路由器弄不稳之后，把它挪到了 AdamW。Lau 和 Su 在两个小 MoE（一个按 OLMoE-1B-7B 造型，一个缩小版 gpt-oss）上比较了路由器的优化器：隐藏矩阵用 Muon 时，路由器上用 AdamW 最终略差于「行归一化」或「左谱」的路由更新。

在 Megatron 里，Muon 是 `--optimizer muon`；要把路由器放到 Adam 上，就加 `--moe-router-skip-muon`，从 [Megatron Core v0.20](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.20.0) 起可用。

```python
# megatron/core/optimizer/emerging_optimizers.py#L147-L154（哪些参数进 Adam）
def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2

def _is_muon_excluded(param):
    """True for parameters that should use the scalar optimizer instead of Muon."""
    return not getattr(param, 'use_muon', True) or _is_nonlinear_or_embedding(param)
```

```python
# megatron/core/transformer/moe/router.py#L88-L91（把路由器挪出 Muon）
if self.config.moe_router_skip_muon:
    setattr(self.weight, 'use_muon', False)
    if self.bias is not None:
        setattr(self.bias, 'use_muon', False)
```

```python
# megatron/core/optimizer/emerging_optimizers.py#L303-L324（先正交化，再按形状缩放）
if scale_shape is None:
    size = [grad.size(-2), grad.size(-1)]
    if partition_dim is not None:
        size[partition_dim] *= get_pg_size(tp_group)
else:
    # This overrides only the final Muon scalar; NS still uses grad's physical
    # shape and partition metadata for its collectives.
    size = scale_shape
# Only forward the kwarg when enabled; older emerging_optimizers do not
# accept it at all, and __init__ has already rejected use_syrk on those.
ns_kwargs = {"use_syrk": True} if use_syrk else {}
orth_grad = newton_schulz_tp(
    grad,
    steps=num_ns_steps,
    coefficient_type=coefficient_type,
    tp_group=tp_group,
    partition_dim=partition_dim,
    tp_mode="duplicated" if tp_mode_this_group == "blockwise" else tp_mode_this_group,
    **ns_kwargs,
)
scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
```

Muon 下专家可能塌，而更新尺度太大会让训练发散。要盯每个专家收到什么、产出什么：每步每专家 token 数，从路由 trace 离线算（[A.1](#s-a-1)）；以及每层「最大 / 最小专家输出范数各自除以该层中位数」，用我们那个前向 hook（[A.1](#s-a-1)）。

**什么时候用它**：你有余力调它的时候用 Muon。从 `--muon-extra-scale-factor 0.2` 起步，因为 Megatron 默认的 1.0 在 Mellum2 的测试里把一个 7B 稠密模型跑崩了；路由器的优化器用 `--moe-router-skip-muon` 选，两种做法都有人用。它不能和 Megatron-FSDP 或 Torch-FSDP2 一起跑。

不管优化器怎么挪权重，每一步都只从「预测下一个 token」里学。能不能靠预测得更远，从每一步里榨出更多信号？那要付多少步时？

### MTP（多 token 预测）

多 token 预测为每多预测一个 token 加一个 transformer 层：这一层拿主模型的隐状态和下一个 token 的 embedding，通过共享输出头预测再下一个 token。DeepSeek-V3 发现它在两个测试规模的大多数基准上都更好，而且推理时可以把这个额外的层丢掉。下图跟着序列的一个位置走完这套接线。

> **图 3.6-d** MTP 层：隐状态 + 下一 token embedding → 拼接投影 → transformer 块 → 共享输出头（原站为交互图，此处从略）

进来的时候两个输入各自过自己的 RMSNorm，MTP 层把它们拼起来、从两倍隐藏维投影回隐藏维，再进自己的 transformer 块。两个预测都从同一个输出头出来，MTP 损失加权后加到主损失上。

这个额外的层要花步时，而在[流水线并行](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=pipeline_parallelism)下这笔开销落在一个地方：除非布局把它挪开，Megatron 把 MTP 层放在最后一个 stage（[`multi_token_prediction.py#L886-L905`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/multi_token_prediction.py#L886-L905)），而那个 stage 本来就拿着 MTP 要复用的输出头。在我们 256 节点的运行里，带一层 MTP 的早期 step 比不带的慢约 37%，把模型摊到 8 个流水 stage 也没把这个差距抹平。Mellum2 报的是一个 14B MoE 多花 7% 训练时间。

```python
# megatron/core/transformer/multi_token_prediction.py#L1709-L1714（隐状态加下一 token embedding）
hidden_states = apply_module(self.hnorm)(hidden_states)
hidden_states = make_viewless_tensor(
    inp=hidden_states, requires_grad=True, keep_graph=True
)
hidden_states = torch.cat((decoder_input, hidden_states), -1)
hidden_states, _ = self.eh_proj(hidden_states)
```

```python
# megatron/core/transformer/multi_token_prediction.py#L1276（损失按层数摊）
mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
```

```python
# megatron/core/transformer/multi_token_prediction.py#L932-L937（无自定义布局时全放最后一个 stage）
# without custom PP layout, we only support put all of MTP layers on the last pipeline stage
if mtp_num_layers is not None:
    if pp_size is None:
        # Compatibility fallback for callers without explicit pipeline metadata.
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    mtp_on_this_rank = pp_rank == pp_size - 1
```

那个差距很大一部分可能来自层放的位置。一个流水 step 要等最慢的 stage，所以加在某个 stage 上的一层，会让这一步慢「它在该 stage 工作量中的占比」那么多，而不是它在整个模型里那个小得多的占比。Ling 2.0 团队撞上了同样的不平衡：他们量到一个 MTP 层的算力约是标准 MoE 层的 1.7 倍，围绕它重新切分流水线后端到端赚了 40%。

Megatron 用 `--mtp-num-layers` 打开 MTP（多加几层深度），`--mtp-loss-scaling-factor` 给 MTP 损失加权：DeepSeek-V3 在前 10T token 用 0.3、之后用 0.1，Mellum2 用 0.1。要指定哪个 stage 放 MTP 层（记为 `m`），用 `--pipeline-model-parallel-layout`。如果只想要 MTP 层当投机解码的草稿模型，`--mtp-detach-heads` 会把它的输入 detach，于是 MTP 损失只训 MTP 层、不动主模型。

MTP 打开时 Megatron 会在 lm loss 旁边记一个 `mtp_1 loss`，而 MTP 层的代价要别处看：步时，以及持有它的那个 stage 的显存。要找出最慢的 stage，用 `--timing-log-level 2 --timing-log-option all` 逐流水 rank 记前向/反向计算时间（[A.1](#s-a-1)）；持有 MTP 层的那个 stage 应该会表现为最慢的 rank。

**什么时候用它**：当 DeepSeek-V3 报告的那种更密的训练信号值这点步时时用。流水线并行下，用 `--pipeline-model-parallel-layout` 安排 MTP 层的位置，否则它落在最后一个 stage。

到此为止，每一部分都告诉了你该怎么盯它自己的选择。可有些故障不属于任何单个选择，且可能从模型任何地方开始——那每一次运行都该常备什么来抓它们？

### 稳定性

MoE 训练可能大声失败，也可能悄悄失败。大声的会在 loss 上现形（尖峰或变 NaN），而悄悄的（比如路由塌缩、离群专家）可以让 loss 一直平着。

最大声的失败也最好抓，因为 Megatron 默认每步都检查 loss 与梯度里有没有 NaN（[3.0](#s-3-0)）。接下来你想知道的是该不该怪硬件，所以检查触发时 Megatron 会重跑这一步（`--rerun-mode validate_results`，也是默认开）。第一次重跑在同一张卡上，用来区分「偶发错误」和「会重复的错误」。如果 NaN 复现，Megatron 以退出码 16 退出，让任务从另一张卡的 checkpoint 重启，那里的第二次重跑就能区分「坏卡」和「数据或模型真的产生了 NaN」。

坏硬件和坏 batch 不一定会变成 NaN，它们也可能返回一个指数大得离谱的有限 loss 或梯度。Megatron 用 `--check-for-spiky-loss` 和 `--check-for-large-grads` 抓这类：把「比之前见过的 100 个里最大的还大 10 倍」标出来，在同一张卡上重跑一次，记录这个值是否复现，然后继续训练。

这些检查都是对「已经出错的 step」做出反应。要从源头不让 loss 崩，[3.1](#s-3-1) 已经保护了路由器（fp32 计算加一个压住 logits 的 z-loss），近期的大 MoE 还各自加了护栏。有的在数值长大之前先截：DeepSeek-V3 与 Step 3.5 Flash 裁梯度范数，DeepSeek-V4.1-Flash 截专家的 SwiGLU 激活，Step 3.5 Flash 也截专家激活。有的把最敏感的操作留在更高精度：DeepSeek-V3 在 FP8 训练时把 embedding、输出头、MoE 门控、归一化与注意力留在 BF16 或 FP32，Kimi K3 把注意力输出留在 FP32。LongCat-Flash 把 z-loss 的思路搬到了隐状态上。

悄悄的失败可能根本不在 loss 上现形，所以得去别处找。下图在一个健康运行旁边模拟了三种：路由塌缩、死专家、离群专家。

> **图 3.6-e** 健康运行 vs 路由塌缩 / 死专家 / 离群专家（原站为交互图，此处从略）

模拟让 loss 在三种故障下都保持平直，所以只有别的信号能把它们和健康运行区分开。路由塌缩把 MaxVio 顶上去。死专家把最小的专家输出范数拉到该层中位数之下很远；离群专家把最大的推到很远之上——从约 14 倍到约 5,700 倍中位数（Step 3.5 Flash 第 45 层的那个数）。loss 报警（比至今最低 loss 高 0.1）借的是 Qwen3.8-Next 的尖峰判据那个余量。

```python
# pretrain_gpt.py#L263-L292（NaN、Inf、尖峰损失）
# Check individual rank losses are not NaN prior to DP all-reduce.
rerun_state_machine = get_rerun_state_machine()
if args.check_for_nan_in_loss_and_grad:
    rerun_state_machine.validate_result(
        result=loss,
        rejection_func=torch.isnan,
        message="found NaN in local forward loss calculation",
        tolerance=0.0,  # forward pass calculations are deterministic
        fatal=True,
    )
    rerun_state_machine.validate_result(
        result=loss,
        rejection_func=torch.isinf,
        message="found Inf in local forward loss calculation",
        tolerance=0.0,  # forward pass calculations are deterministic
        fatal=True,
    )
# Check for spiky loss
if args.check_for_spiky_loss:
    rerun_state_machine.validate_result(
        result=loss,
        rejection_func=partial(
            rerun_state_machine.is_unexpectedly_large,
            threshold=SPIKY_LOSS_FACTOR,
            context="loss",
        ),
        message="Spiky loss",
        tolerance=0.0,  # forward pass calculations are deterministic
        fatal=False,
    )
```

```python
# megatron/core/distributed/param_and_grad_buffer.py#L389-L425（逐 bucket 的梯度范数检查）
def check_grads(self, check_for_nan_or_inf, check_for_large):
    """
    Make sure norm of grads in bucket are not NaN prior to data-parallel
    all-reduce / reduce-scatter.
    """
    rerun_state_machine = get_rerun_state_machine()
    for i in range(len(self.buckets)):
        grad_norm = self.buckets[i].grad_data.norm(p=2)
        # check for NaN, Inf and unexpectedly large grads
        if check_for_nan_or_inf:
            rerun_state_machine.validate_result(
                result=grad_norm,
                rejection_func=torch.isnan,
                message=f"found NaN in local grad norm for bucket #{i} "
                f"in backward pass before data-parallel communication collective",
                tolerance=0.001,  # 0.1% tolerance to account for non-determinism
                fatal=True,
            )
```

尖峰方面，用 z-loss 盯路由器的 logits（[3.1](#s-3-1)）。那些让 loss 保持平直的故障需要各自的信号：路由塌缩看逐层 MaxVio（从路由 trace 离线算，[A.1](#s-a-1)），与负载均衡损失并排看；死专家与离群专家看每层「最大、最小专家输出范数分别除以该层中位数」，用我们那个 hook（[A.1](#s-a-1)）。

**什么时候用它**：一直用——保留默认的 NaN 检查与 `--rerun-mode validate_results`。

### 该盯什么

| 部位 | 症状 | 该盯的信号 |
|---|---|---|
| Upcycling | 「学不好」：开始阶段 loss 下降慢 | loss 与「稠密那次继续训同样 token」对比；在拷贝的专家分化期间，还要逐层看 MaxVio（路由 trace，A.1 节） |
| 优化器 | 「学不好」：专家塌缩。「炸了」：正交化在 bf16 下跑，或更新尺度太大 | 每步每专家 token 数（路由 trace，A.1 节）；逐层最大与最小专家输出范数各自除以中位数（前向 hook，A.1 节） |
| MTP | 「慢」：某个流水 stage 比别人慢。「OOM」：53 层模型放在 4 个流水 stage 上带 MTP 时爆显存 | `mtp_1 loss` 与 LM loss 并列看；步时与峰值显存；逐流水 rank 的前向/反向计算时间（`--timing-log-level 2 --timing-log-option all`，A.1 节），尤其是持有 MTP 的那个 stage |
| 稳定性 | 「炸了」：NaN 与 loss 尖峰。「学不好」：路由塌缩、死专家、离群专家 | 路由器 z-loss（3.1 节）——ST-MoE 把自己那次不稳定归因到的就是路由器 logits；逐层 MaxVio（路由 trace）与最大、最小专家输出范数除以中位数（A.1 节） |

一次训练得干干净净的运行，仍然可能有很大一部分时间在等通信而不是在算。MoE 的一步里时间都去哪了，能拿回多少？

---

**导航**：[全书目录](index.md) · [← 第 3 章（中）专家](03-3-experts.md) · [第 3 章（下）系统工程 →](03-5-systems.md)
