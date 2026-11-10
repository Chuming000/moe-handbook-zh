# 第 3 章 解剖 MoE 训练（下）：专家

> 说明：原站里的图都是交互式组件，这里无法复现，一律保留图注并标注「（原站为交互图，此处从略）」。原站大部分图不给编号，本文按**节内顺序**编号（如 图 3.4-a）。

上一节里每个均衡器都在「训练开始前就定好的一组专家」上摊 token，而我们还没问过这组专家该长什么样：一层该有多少个专家、每个多宽、有没有哪个专家该看见每一个 token？

## 3.4 专家：多少、多大、共不共享

一旦路由器为某个 token 挑好了专家，这些专家长什么样，同时决定了这一层能学到什么、以及跑起来花多少。你要选：一层有多少专家、每个多宽；怎么把「很多小专家」带来的通信量压住；要不要有一个专家看见每个 token；以及每个专家内部用哪个激活函数。

前沿开源 MoE 的专家选择，外加 Nemotron（用 LatentMoE 与 ReLU²）和 LongCat-Flash（可变算力）。「数量与宽度」给出每 token 用掉全部专家中的几个、以及每个专家的 FFN 宽度；共享专家单列；「潜空间维度」给出投影前与投影后的 token 宽度。「未说明上限」指的是公开的配置。

| 模型 | 数量与宽度 | 潜空间 MoE | 共享专家 | 可变算力 | 激活函数 |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 384 里选 6，宽 2,304 | 全宽专家 | 1 个，宽 2,304 | 固定 top-k，等宽 | SwiGLU，限 10 |
| GLM-5.3 | 256 里选 8，宽 2,048 | 全宽专家 | 1 个，宽 2,048 | 固定 top-k，等宽 | SiLU 门；未说明上限 |
| Kimi K3 | 896 里选 16，宽 3,072 | 7,168 → 3,584；上投影前做归一化 | 2 个，各宽 3,072 | 固定 top-k，等宽 | SiTU-GLU，上限 4 与 25 |
| LongCat-Flash | 768 里选 12，宽 2,048 | 全宽专家 | 无 | 768 个里有 256 个是恒等专家：每 token 激活 18.6B 到 31.3B 参数，平均 27B | SiLU 门；未说明上限 |
| MiMo-V2.6-Pro-RL | 384 里选 8，宽 2,048 | 全宽专家 | 无 | 固定 top-k，等宽 | SiLU 门；未说明上限 |
| Nemotron 3 Ultra | 512 里选 22，宽 5,120 | 8,192 → 2,048 | 1 个，宽 10,240 | 固定 top-k，等宽 | ReLU² |
| Qwen3.8 2.4T-A95B | 512 里选 10，宽 2,048 | 全宽专家 | 1 个，宽 2,048 | 固定 top-k，等宽 | SiLU 门；未说明上限 |
| Step 5 Preview | 352 里选 8，宽 1,536 | 全宽专家 | 1 个，宽 1,536 | 固定 top-k，等宽 | SwiGLU，限 7 |

在参数预算固定的情况下，加专家就意味着把每个专家做窄。同样算力下，哪种切法能给你更好的模型？

### 数量与宽度

设想一层 MoE，参数与算力预算固定，然后把它切细：专家数翻倍、每个宽度减半、top-k 也翻倍，两个预算都不变。图 3.4-a 画的就是这条阶梯：粗粒度，然后是 16 个 d/2 宽、top-4，再是 32 个 d/4 宽、top-8。Liew 等人在一个 18 层、总参数与激活参数都固定的模型上做了这个消融，损失随着每一级变细而下降，一直到 128 个 d/4 宽、top-8，从 2.334 降到 2.297；再细下去就没有进一步提升了。所以在这个实验里，「切细」到 128 个专家之后就不再划算，而每一级变细仍然意味着更小的 GEMM、以及每个 token 更多的副本要调度。

> **图 3.4-a** 切细的阶梯：粗粒度 → 16 × d/2（top-4）→ 32 × d/4（top-8）；总参数与每 token 算力不变（原站为交互图，此处从略）

已发布的模型停在这条阶梯的不同位置：gpt-oss-120b 与 Trinity Large 让每个专家和模型一样宽，分别 128 选 4、256 选 4；Mistral Small 4 把这个宽度减半，128 选 4；Qwen3.5-35B-A3B 再减到四分之一，256 选 8。

图里可以看到，切专家改变的是「个数与宽度」，而总参数和每 token 算力保持不变。所有方块的总面积代表参数，深色面积代表一个 token 的专家计算。你改的是**粒度**（granularity：切得多细。越细＝更多更窄的专家；为了保持每 token 算力不变，每个 token 就用更多个窄专家）。你也可以提高**稀疏度**：加更多参数，但每个 token 用到的数量不变。多出来的参数有两种花法，下面比较：加更多路由专家，或者加「按 token id 查表」的 n-gram embedding 表。

> **图 3.4-b** LongCat 在 280M 激活参数下：多花在专家上 vs 花在 n-gram 表上（原站为交互图，此处从略）

那张图画的是 LongCat 在 280M 激活参数下的比较。在那里，当专家已经持有约 12 倍激活参数时，n-gram 表比把同样参数花在更多专家上更能降 loss（在 17 倍激活参数时约好 0.01）；而如果在专家只有约 4 倍激活参数时就加表，那是从一开始就亏。这个收益有上限：在英文验证损失上，一旦表占到参数的一小半以上（略超 40%），「更多专家」又赢回来，LongCat 从那个交点读出的规则是：表里的参数不超过 50%。在那个比较里，先扩容专家是划算的，等专家足够大了，接下来的一部分参数放到查表里更有效。所以「多余参数往哪花」取决于你已经有都少专家容量。

两种花法在已发布模型里都有：Kimi K2 把池子扩到 384 个专家、top-8；LongCat-Flash-Lite 则把 68.5B 参数里的 31.4B 放进了 n-gram embedding 表。

不管是加更多专家，还是把现有容量切成更多窄专家，最终形状也会影响训练速度：更窄的专家跑更小的 GEMM，GPU 利用率更低。

对稠密模型来说，算力最优的缩放律能告诉你怎么在模型大小和训练 token 数之间分配固定的 FLOP 预算：模型越大，能喂的 token 越少；模型越小，能训的 token 越多。MoE 多了一个选择：你可以增加它的总参数而不增加每个 token 用到的参数，于是你还得选**稀疏度**。而无论哪种模型，在时间限制下做选择，都还要算上「每种配置对硬件的利用效率」。

Sarkar 等人把一个拟合出来的损失缩放律和一个训练吞吐模型结合起来，比较了「固定模型 FLOP 下损失最小」的稀疏度与「固定集群时间下损失最小」的稀疏度。他们对某个集群与训练窗口的预测画在下面。（这里的稀疏度指「token 用不到的参数占全部参数的比例」。）

> **图 3.4-c** 按模型 FLOP 选 vs 按集群时间选：预测损失与稀疏度（原站为交互图，此处从略）

在他们的预测里，只按模型 FLOP 选会把你推向网格里最稀疏的那个模型。一旦把集群上的训练速度算进去，预测损失最低的点就移到一个不那么稀疏的配置，比最稀疏那个低 0.031 nats。和稠密缩放一样，你是在模型容量与「能训多少 token」之间权衡。对 MoE 而言，提高稀疏度可以在不增加每 token 模型 FLOP 的同时降低硬件利用率，于是多出来的容量仍然要花训练时间。

[Meagatron](https://github.com/NVIDIA/Megatron-LM) 用三个开关定形状：`--num-experts` 定数量，`--moe-ffn-hidden-size` 定每个专家的宽度，`--moe-router-topk` 定每个 token 挑几个专家。

```python
# megatron/core/transformer/moe/experts.py#L262-L293（fc1/fc2 的分组形状）
# Double the output width with gated linear unit, see https://arxiv.org/pdf/2002.05202.pdf
ffn_hidden_size = not_none(self.config.moe_ffn_hidden_size)
if self.config.gated_linear_unit:
    ffn_hidden_size *= 2

self.linear_fc1 = submodules.linear_fc1(
    self.num_local_experts,
    self.input_size if self.config.moe_latent_size is None else self.config.moe_latent_size,
    ffn_hidden_size,
    ...
)
```

```python
# megatron/core/transformer/moe/router.py#L77-L81（路由器权重：专家数 × 隐藏维）
self.weight = torch.nn.Parameter(
    torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
)
```

切得太细，每个专家的 GEMM 小到会拉低每块 GPU 的 TFLOP/s；粒度选错，模型还学得更差。想知道某个形状要付多少代价，就从 profiler 里读每层 MoE 的分组 GEMM 时间，配 `--log-throughput` 给出的每 GPU TFLOP/s 一起看。`--save-tokens-per-expert-interval` 写出的 token 计数还能暴露「收不到任何 token」的死专家（[A.1 节](#s-a-1)），但一个专家也可以一边收着 token 一边死掉——那就要看它的输出范数（3.6 节）。

**什么时候用它**：在总参数与激活参数不变的前提下用更细的专家，但有个头。在那个 18 层模型里，损失一直降到 128 个 d/4、top-8，再细就没有收益了。等小 GEMM 和 token 副本开始主导你的步时长，就别再往细里走了。

更细的专家只在某一点之前对损失有帮助，而每一级变细都会让每个 token 的更多副本上网络。能不能保留很多小专家，同时让每份副本少送点数据？

### 潜空间 MoE（LatentMoE）

每份 token 副本都带着它的完整隐状态——d 个数，每挑一个专家就送一次。LatentMoE 把这份副本缩小：在调度前把每个 token 投影到一个更小的潜空间 ℓ，让专家在 ℓ 上跑，合并之后再投影回去。每份副本从 d 缩到 ℓ，省下来的通信就够养更多专家：在专家参数、专家计算和通信量都不变的前提下，这一层可以负担 α = d/ℓ 倍多的专家、以及 α 倍大的 top-k。下面把一个标准层和它的潜空间版本并排画，专家内部的参数与计算保持不变。

> **图 3.4-d** 标准 MoE 层 vs LatentMoE 层（专家参数与计算固定）（原站为交互图，此处从略）

画成方块的话：标准层从 16 个里选 4 个，每个收到全宽的 token 向量；潜空间层从 32 个里选 8 个，但每个只收到一半宽度的数值。两者每 token 送出的数据量相同；它们的总面积与深色面积也相等，说明专家参数与计算没变——两次投影加的是这些面积之外的工作。

这个交换在质量上划算吗？在英伟达那个 73B-A8B 的消融里，总参数与激活参数相同的条件下，LatentMoE 在 MMLU-Pro 上比标准 MoE 高 4.6 分，一个小规模的复现也降低了困惑度。Nemotron 3 Super、Nemotron 3 Ultra 和 Kimi K3 都上了它。代价是每个 token 要多过两次 d × ℓ 投影，服务端吞吐从并发 1 时的低 12% 到某些并发下略高。在 Megatron 里，潜空间宽度是 `--moe-latent-size`。

```python
# megatron/core/transformer/moe/moe_layer.py#L304-L335（两个投影 + 可选归一化）
self.fc1_latent_proj = linear_cls(
    self.config.hidden_size,
    self.config.moe_latent_size,
    parallel_mode="duplicated",
    ...
)
if self.config.moe_use_norm_before_up_proj:
    self.fc2_norm = TENorm(
        config=self.config,
        hidden_size=self.config.moe_latent_size,
        eps=self.config.layernorm_epsilon,
    )
self.fc2_latent_proj = linear_cls(
    self.config.moe_latent_size,
    self.config.hidden_size,
    parallel_mode="duplicated",
    ...
)
```

```python
# megatron/core/transformer/moe/moe_layer.py#L554-L556（调度前降维）
# Project the hidden_states from hidden dimension down to latent dimension.
if self.config.moe_latent_size:
    hidden_states, _ = self.fc1_latent_proj(hidden_states)
```

```python
# megatron/core/transformer/moe/moe_layer.py#L662-L666（合并后归一化并升维）
output = self.token_dispatcher.combine_postprocess(output)
if self.config.moe_latent_size:
    if self.config.moe_use_norm_before_up_proj:
        output = apply_module(self.fc2_norm)(output)
    output, _ = self.fc2_latent_proj(output)
```

潜空间 MoE 会从两头出问题：跑得慢，或者激活值炸掉。跑得慢的时候，别只盯通信——All-to-All 现在每副本只传 ℓ 个数而不是 d——要去看那两次额外投影和更小的专家 GEMM。LatentMoE 的英伟达作者提的补法是「小 GEMM 的专用 kernel」以及「给共享专家单开一条流」。炸掉的时候（Kimi K3 报告它那条路由分支的激活在 2.8 万亿参数规模上炸），一个该看的地方是上投影 `fc2_latent_proj` 的输出范数，可以从 `--save-activations-interval` 的 dump 里离线逐层算（[A.1 节](#s-a-1)）。K3 的修法是在上投影前加一个 RMSNorm（`--moe-use-norm-before-up-proj`），它发布时把路由缩放系数设成 1.0。

**什么时候用它**：当你想要更多专家、更大的 top-k，但不想让每 token 的通信量变大时。把每份副本缩到 ℓ = d/α 就能负担 α 倍的专家与 top-k，等规模下分数更高，代价是两次投影。

到此为止每个专家都躲在路由器后面，只看见发给它的 token。如果把一个专家挪到路由器外面、让它看见每个 token，能换来什么？值它那份算力吗？

### 共享专家

共享专家待在路由器外面：它处理每个 token，不管这个 token 路由到了哪些专家。DeepSeekMoE 提出共享专家，让它学共同模式，从而减少路由专家之间重复的知识、给它们专业化腾出空间。在等算力的消融里，加一个共享专家在多数基准上有提升。Qwen 家的模型说明了这个问题有多开放：Qwen3 报告里的 MoE 模型把它去掉了，Qwen3-Next 又在 sigmoid 门后面把它请回来，Qwen3.5 则保留。

用「等算力、从头训」的模型损失来衡量，收益小且不一致。Pangu Ultra MoE 和 Sun 等人的专家阈值模型里，共享专家把损失降了约 0.01 到 0.02；而在 OLMoE 和 Li 等人的小 MoE 扫描里，它要么没作用、要么略差。

共享也会减少路由器的自由度：32 个专家、激活 4 个时，如果给共享专家占掉一个位置，剩下选 3 个只有 4,495 种组合，而从 32 个里选 4 个有 35,960 种。如果你加共享专家却不减少路由 top-k，这个常开的 FFN 就增加了每 token 算力：Mellum2 那样做了，结果质量没提升、推理还更慢。

真要加，还得决定它的输出怎么和路由专家混。下图在等算力下并排画了五种设计：只有路由专家、固定权重的共享专家（DeepSeek-V3）、带门的共享专家（Qwen3-Next）、与路由专家一起归一化的共享专家（Inkling）、以及一条并行的稠密 FFN（Grok 2）。

> **图 3.4-e** 五种共享专家设计（等算力）（原站为交互图，此处从略）

这样并排看：固定权重的共享专家在每个 token 里占的份额都一样；Qwen 的门让这个份额在 0 到 50% 之间移动；Inkling 的路由器把共享专家和路由专家放在一起打分，于是逐 token 学出份额；而一条不做重归一化的并行稠密 FFN，不管路由器打多少分都握着超过一半的份额。

Megatron 的开关是 `--moe-shared-expert-intermediate-size`、`--moe-shared-expert-gate`（给它输出加一个 sigmoid 门）和 `--use-grouped-gemm-for-shared-expert`。不加门时，Megatron 以权重 1 加上共享输出（[`moe_layer.py#L668-L669`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/moe_layer.py#L668-L669)），而路由权重在选中的专家上归一化后加起来等于路由缩放系数（[`moe_utils.py#L942-L947`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/moe_utils.py#L942-L947)，见 3.1 节）。（Megatron 用 sigmoid 路由，或者在选出 top-k 之后再做 softmax 来实现这件事；后者要把 `--moe-router-pre-softmax` 关掉。）于是共享专家占每个 token 输出权重的 1/(1 + factor)，系数取 2.5 时就是 29%；系数越大它占比越小。GLM-4.5-Air 和 Mistral Large 3 把系数设成 1，于是它拿一半。

```python
# megatron/core/transformer/moe/shared_experts.py#L201-L208（给输出加 sigmoid 门）
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    output, _ = super().forward(hidden_states)
    if self.use_shared_expert_gate:
        logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
        gate_score = torch.nn.functional.sigmoid(logits)
        output = output * gate_score
    return output
```

```python
# megatron/core/transformer/moe/moe_layer.py#L668-L669（以权重 1 相加）
if shared_expert_output is not None:
    output = output + shared_expert_output
```

想跟踪「训练中共享专家对每个 token 输出的贡献有多大」，就把它过门之后的输出，和路由分支过合并（以及可能的潜空间升维）之后的输出作比，用 [A.1](#s-a-1) 里那个稀疏前向 hook。

共享专家是前沿 MoE 常见的起步选择，GLM-5.3、Kimi K3、DeepSeek V4.1 Flash 都有；MiMo-V2.6 没有。想在等算力下比「有」和「没有」，就给共享专家和路由专家一样的宽度，并用它替换掉一个路由位置。

不管共不共享，到目前为止每个专家都一样宽、每个 token 都拿到一样的 top-k，于是每个 token 的算力都一样。能不能让路由器给某些 token 少一点算力、给另一些多一点，而平均值保持不变？

### 可变算力

有两种设计能让每 token 算力变化而均值固定。LongCat-Flash 加了**零计算专家**：把输入原样传过去。它的路由器从「512 个 FFN 专家 + 256 条恒等路径」里为每个 token 选 12 个。一个偏置调节 FFN 专家被选中的频率，目标是每 token 平均 8 个，同时允许个别 token 用得多或少；预训练期间这个数量的标准差升到约 2.5。小米 AI Lab 在 MoDSE（Mixture of Diverse Size Experts）里换了个思路：把专家宽度混起来，让路由器为每个 token 挑大的或小的专家。下图模拟了同宽专家（如 Mixtral）以及这两种设计下每 token 的激活宽度。

> **图 3.4-f** 每 token 的激活宽度：同宽专家 / 零计算专家（LongCat）/ 混合宽度（MoDSE）（原站为交互图，此处从略）

模拟里，每种设计用一个偏置把平均激活宽度按在同宽专家的水平上，于是有的 token 拿到更多、有的更少；而每个 token 仍然填满它所有的 top-k 位置，变的只是这些位置里落进什么。在 LongCat-Flash 的消融里，零计算专家在等算力下拿到更低的损失。混合专家宽度可能提升质量，但可能伤吞吐：小米 AI Lab 和 Wang 等人各自在小模型上用一个路由器管不同尺寸的专家并报告了收益；Li 等人没看到收益，不过他们的设置里专家分成两个池、每个池固定 top-k，于是每 token 算力还是相同。GroveMoE 的代价出现在推理端：它的混合尺寸比 Qwen3-30B-A3B 慢约 30%，作者归因于每层要用两套 MoE kernel 分别处理两种尺寸。

Megatron source：没有。在 Megatron-LM main 的 `megatron/` 目录里 grep 不到零计算专家，也没有混合专家宽度；每个分组线性层只有一个 `moe_ffn_hidden_size`。

要跟踪每层每 token 算力的均值与离散度：零计算路由看「选中的 FFN 专家个数」，混合宽度路由看「选中专家宽度之和」。把均值与目标预算比，同时记录逐专家 token 数、每个专家并行 rank 的算力和步时长。

**什么时候用它**：从「一种宽度 + 固定 top-k」起步，就像 MiMo-V2.6-Pro-RL、GLM-5.3、Kimi K3 那样。这也正好匹配 Megatron 的分组线性层——它们只有一个 `moe_ffn_hidden_size`。

专家的数量、宽度、算力定了之后，每个专家内部还剩一个选择：两次投影之间那个函数。这个选择对 MoE 影响多大？

### 激活函数

和稠密 Transformer 一样，每个专家在输入与输出投影之间用一个激活函数，比如 DeepSeek-V3、Qwen3、Kimi K2 用的 SwiGLU，或者 ReLU²。在 MoE 里你还要盯「激活在不同专家之间怎么表现」：Step 3.5 Flash 发现有一两个专家的输出范数会飞快长大，而其他专家保持稳定、训练损失几乎没变。激活函数、以及加在它上面的截断，都有助于控制这种增长。下图画出 SwiGLU、ReLU² 和 Expert-Specific PolyNorm 的输出随输入尺度变大怎么长。

> **图 3.4-g** SwiGLU / ReLU² / Expert-Specific PolyNorm 的输出随输入尺度增长（原站为交互图，此处从略）

放大之后，SwiGLU 和 ReLU² 的输出随输入尺度的平方增长，而 Expert-Specific PolyNorm 把它的门归一化了，于是只线性增长。一个截断就能把这种增长限住。两项研究在他们测试的设置里发现「截」与「不截」质量相近：蚂蚁 Ling 团队一个 7.9B MoE 的消融里，带截断的 SwiGLU 在 17 个基准上的均值与朴素 SwiGLU 打平；Step 3.5 Flash 的激活截断把离群专家的范数压住了，损失看不出差别（尽管那条分支是从未截断的那次运行中途接过来的）。截断并没有消掉 Ling 团队那次 FP8 运行里的损失尖峰：带截断的 SwiGLU 把尖峰推迟了，但到第 77,000 步还是来了一个。下一张图把输入尺度扫过「朴素 SwiGLU / 硬截断 / 同一边界的 SiTU-GLU 平滑上限」，看它们各自怎么弯。

> **图 3.4-h** 朴素 SwiGLU / 硬截断 / SiTU-GLU 平滑上限在扫输入尺度时的表现（原站为交互图，此处从略）

这次扫描里，朴素 SwiGLU 最大的隐藏单元随门输入的平方一路长到 6,862——那是 K-EXAONE 2.0 在最后一层、推理时关掉截断量到的峰值；硬截断跟着它长到边界就停住；而 SiTU-GLU 在远没到边界之前就弯开了。扫描把两者的「单个隐藏单元停在约 100」设为共同目标：硬截断在 10，SiTU-GLU 的门上限 4、上投影上限 25，从差不多的输入开始，两者都几乎不再给这些单元的门权重传梯度。整个换掉激活函数的证据更薄：Zhuo 等人的 PolyNorm 在 OLMoE-1B-7B 上把平均验证损失比 SwiGLU 降了 0.02，而 Motif 3 只把它的 Expert-Specific PolyNorm 与 SwiGLU 在有效秩上做了比较。Kimi K3 说 SiTU-GLU 的平滑上限在训练行为上比硬截断更好。

在 Megatron 里，`--activation-func-clamp-value` 给 MoE 层的 SwiGLU 设硬截断上限；取 10 就对应 DeepSeek V4.1 Flash 的上限，GigaChat 3.5 与 Hy4-preview 也用这个值。Kimi K3 的平滑上限 SiTU-GLU 是 `--activation-func-tanh-clamp-scale 4 --activation-func-tanh-clamp-scale-linear 25`，从 [Megatron Core v0.20](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.20.0) 起可用；Nemotron 3 的 ReLU² 是 `--squared-relu`。

```python
# megatron/core/transformer/moe/experts.py#L1100-L1118（SiTU-GLU 或带截断的 SwiGLU）
if tanh_clamp_scale is not None:
    intermediate_parallel = situ_glu(
        intermediate_parallel,
        tanh_clamp_scale,
        self.config.activation_func_tanh_clamp_scale_linear,
        self.config.glu_linear_offset,
    )
else:
    def glu(x):
        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
        if (val := self.config.activation_func_clamp_value) is not None:
            x_glu = x_glu.clamp(min=None, max=val)
            x_linear = x_linear.clamp(min=-val, max=val)
        return self.config.activation_func(x_glu) * (x_linear + self.config.glu_linear_offset)

    intermediate_parallel = glu(intermediate_parallel)
```

```python
# megatron/core/activations.py#L17-L39（软截断与 SiTU-GLU）
def tanh_soft_clamp(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Tanh Soft Clamp to precondition activation inputs."""
    return (scale * torch.tanh(x.float() / scale)).to(x.dtype)

@jit_fuser
def situ(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Sigmoid-Tanh Unit from Kimi K3: ``s * tanh(x / s) * sigmoid(x)``"""
    return tanh_soft_clamp(x, scale) * torch.sigmoid(x)

@jit_fuser
def situ_glu(x, gate_scale, linear_scale=None, linear_offset=0.0):
    """SiTU-GLU: ``situ(x_gate, gate_scale) * (x_linear + linear_offset)``."""
    x_gate, x_linear = torch.chunk(x, 2, dim=-1)
    if linear_scale is not None:
        x_linear = tanh_soft_clamp(x_linear, linear_scale)
    return situ(x_gate, gate_scale) * (x_linear + linear_offset)
```

专家里的离群激活能把一次训练炸掉。要抓它，就离线从 `--save-activations-interval` 的 dump 里算 Step 3.5 Flash 那个信号：每层专家输出范数的「最大值 / 中位数」之比（[A.1 节](#s-a-1)）。一个专家把别人甩开就表现为这个比值上升。每份 dump 只保留最后一个 micro-batch 并拷到 CPU，所以采样要稀。

**什么时候用它**：从你参考模型的激活方案起步。前沿 MoE 用不同方式限住激活——Step 5 Preview 的 SwiGLU 截在 7、DeepSeek V4.1 Flash 截在 10、Kimi K3 用 SiTU-GLU 的平滑上限。做新架构时，SwiGLU 是合理的基线；如果专家范数发散了，就试试截断，并且既要看范数也要看训练损失再决定上限。

### 该盯什么

| 部位 | 症状 | 该盯的信号 |
|---|---|---|
| 数量与宽度 | 「慢」：专家 GEMM 太小。某些专家可能拿不到 token | profiler 里每层 MoE 的分组 GEMM 时间、每 GPU TFLOP/s（`--log-throughput`）、逐专家 token 数（`--save-tokens-per-expert-interval`，A.1 节） |
| 潜空间 MoE | 「慢」：两次额外投影与更小的专家 GEMM 增加时间。「炸了」：Kimi K3 报告路由分支激活爆炸 | 速度上，剖析两次潜空间投影与更小的专家 GEMM，配合被缩小的 All-to-All。稳定性上，一处该看的是上投影输出范数（`fc2_latent_proj`），从激活 dump 里算（A.1 节） |
| 共享专家 | 推理时「慢」：它增加算力，Mellum2 就发现推理更慢 | 共享分支过门后的输出 vs 路由分支合并（含潜空间升维）后的输出，用 A.1 节的稀疏前向 hook |
| 可变算力 | 「慢」：每种专家尺寸跑各自的 kernel，GroveMoE 报告推理比 Qwen3-30B-A3B 慢约 30% | 每层「选中 FFN 个数」或「专家宽度之和」的均值与离散度 vs 目标算力预算；同时记录逐专家 token 数、每 EP rank 算力、步时长 |
| 激活函数 | 「炸了」：离群激活 | 每层专家输出范数的最大值/中位数之比（Step 3.5 Flash 的信号），从激活 dump 里算（A.1 节） |

本节所有东西都发生在**一层** MoE 内部，而模型要叠很多层，这就带出两个还没问的问题：是不是每一层都该是 MoE 层？每层外面那个 block 该怎么接？

---

**导航**：[全书目录](index.md) · [← 第 3 章（中）负载均衡](03-2-load-balancing.md) · [第 3 章（下）块布局与训练 →](03-4-block-layout-and-training.md)
