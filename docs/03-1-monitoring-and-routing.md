# 第 3 章 解剖 MoE 训练（上）

> 说明：原站里的图都是交互式组件，这里无法复现，一律保留图注并标注「（原站为交互图，此处从略）」。原站未给图编号，本文按**节内顺序**编号（如 图 3.1-a），便于对照。

2.5 节那个 MoE，比稠密实验多出了几样东西：一个给每个 token 挑 8 个专家的路由器、工作量可能差得很远的专家、以及在卡之间搬运 token 的 All-to-All。每一样都能让训练变慢或者把 loss 弄坏，而「比稠密基线差了」这个现象本身，并不会告诉你是哪一样造成的。

下面这张图画了一个比我们小一号的 MoE 层：16 个专家、top-4（我们是 64 与 top-8），另外多一个我们没有的共享专家。它展示了 batch 里的 token、路由器、专家，以及把专家输出合并起来的那次求和。后面每一节都会拿同一张图的一个局部视角来开场，放大到它要讲的部分，所以你可以直接跳到需要的那节。有几个视角还带着该节的实验结论，比如 3.6 节里 Step-3 Flash 的训练 loss 没有暴露出来的那些失控专家。

> **图 3.0-a** 一层 MoE 的全景：token、路由器、专家、输出合并（原站为交互图，此处从略）

3.0 节把监控打开之后，我们从路由器内部开始：它的分数怎么变成被选专家的权重，以及到底是 token 选专家还是专家选 token。然后是专家之间负载不均、用多少专家每个多大、哪些层要放专家、以及怎么起步并保持稳定。最后一步讲一个训练步的时间都花在 GPU 的哪儿。

## 3.0 监控：一次性打开

动任何设置之前，先把「它到底有没有起作用」这件事打开。你现在不必知道每个数字的含义：每节在第一次需要某个指标时再介绍（3.1 节的 z-loss、3.2 节的 MaxVio、3.7 节的耗时与显存）。附录 A.1 给出每个测量对应的采集命令、计算公式和局限。

### 该打开什么

2.5 节那次实验已经在逐层记录路由损失（`--moe-per-layer-logging`），你可以看出是哪一层在闹脾气（3.1、3.3、A.1 节）。2.4 节的 [Trackio](https://huggingface.co/docs/trackio) 启动器留着，把下面这些参数加到 2.5 节的参数里：

```bash
--save <dir> --save-interval <m>
--save-tokens-per-expert-interval <n>
--log-throughput --log-params-norm
--log-num-zeros-in-grad --log-max-attention-logit
```

Megatron 会把逐专家的 token 数和激活 dump 写到 `--save` 下面（[`training.py#L3398-L3401`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/training.py#L3398-L3401)），并要求在它旁边给出 `--save-interval`（[`arguments.py#L1321-L1323`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/arguments.py#L1321-L1323)），所以这次实验也会把 checkpoint 存在那里。下一行参数记录每个专家收到多少 token（3.3 节、A.1 节）。再往后四个盯的是运行健康度：吞吐、参数范数、梯度里的零、以及最大的注意力 logit（3.6 节、A.1 节）。标量指标会进到你基线所在的同一个 Trackio project，而路由抓取则留在自己的文件里。NaN 检查本来就开着（[`arguments.py#L2825-L2827`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/arguments.py#L2825-L2827)）。

### 需要时才抓的东西

还有两项测量是有界抓取：有问题要问的时候再开。

路由追踪记录每个 token 选了哪些专家（3.1、3.3、A.1 节）。它挂在路由器上的 hook 在 CUDA graph 重放下不会触发（[`router_trace.py#L23`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router_trace.py#L23)），而且默认每步都追（[`training_config.py#L363-L367`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/config/training_config.py#L363-L367)），所以要把 graph 关掉并给它设边界：

```bash
--moe-routing-trace-path <dir> --moe-routing-trace-max-training-iters <n>
--cuda-graph-impl none
```

激活 dump 把专家的激活存到同一个 `--save` 目录下，这样你就能离线算逐专家的激活范数（3.4 节、A.1 节）：

```bash
--save-activations-interval <n>
```

监控开好了，就从每个 token 都要经过的地方开始：路由器。它挑专家，但每个专家在一个 token 的输出里到底占多大分量？

## 3.1 路由器打分：logits 怎么变成权重

路由器把每个 token 变成每个专家一个 logit，在朴素的 top-k 路由里，top-k 的 logit 决定这个 token 走哪些专家。在这些 logit 和「用来加权专家输出的权重」之间，还夹着四个选择。**门控函数**、**输出缩放**（乘到被选专家输出的加权和上的那个常数）、**锐度**（路由分数的尖锐程度，也就是头号专家压过别人多少）决定每个被选专家在一个 token 输出里占多少分量、以及这个输出有多大。在朴素 top-k 路由下，给定一个 token 的 logits，这四个都不改变「谁被选中」：它们要么保持专家的相对顺序，要么作用在 top-k 之后。

精度是那个例外。一个 [bf16](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=fp16_and_bf16_training) 路由器可能把两个很接近的 logit 圆整成同一个值，于是改变了胜者——这就是为什么这里唯一永远该用的设置是 fp32 路由器。

下表列出当前 10 个开源 MoE 在这四项上各自的选择，一行一个模型。

| 模型 | 门控函数 | 输出缩放 | 锐度 | 路由器精度：训练中 / 推理代码 |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash | sqrt(softplus) | 重归一化，k=6 时 1.5（√k=2.45） | 一个温度设置，保持 1.0 | 未说明；fp32 |
| Gemma 4 26B-A4B | softmax | 重归一化，再接一个可学的逐专家缩放；128 选 8（√k=2.83） | 归一化路由器输入：RMSNorm + 可学缩放 + 1/√d | 未说明；bf16，softmax 在 fp32 |
| GLM-5.3 | sigmoid | 重归一化，k=8 时 2.5（√k=2.83） | 无 | 未说明；fp32 |
| Inkling | sigmoid，在 top-6 与其 2 个共享专家上联合归一化 | 8.0 倍一个可学的全局缩放，覆盖 6 个路由专家 + 2 个共享专家（√8=2.83） | 无 | 未说明；bf16 |
| Kimi K3 | sigmoid | 重归一化，896 里选 16 时 1.0（√k=4） | 无 | 后训练 QAT 中高于 MXFP4；fp32 |
| MiMo-V2.6-Pro-RL | sigmoid | 重归一化，k=8 时无系数（√k=2.83） | 无 | 未说明；fp32 |
| North-Mini-Code | sigmoid | 不重归一化：权重就是原始分数；128 选 8 | 无 | 未说明；bf16 |
| Qwen3.8-Flash-Next | softmax | 重归一化，k=10 时无系数（√k=3.16） | 无 | 未说明；bf16，softmax 在 fp32 |
| Step-5-Preview | sigmoid | 重归一化，352 里选 8 时 3.0（√k=2.83） | 无 | 未说明；fp32 |
| ZAYA1-8B | 对 MLP 路由器的 logits 做 softmax | 不重归一化，16 选 1，外加一个 skip 专家 | 无 | RL 期间 fp32 softmax；bf16 |

先看门控函数。如果它从不改变胜者，那它改变的是什么？

### 门控函数

softmax、sigmoid 和 sqrt(softplus) 给专家的排序是一样的，所以差别只在「每个被选专家拿到多少权重」。下面这张图把同一个 token 的 logits 从不自信推到自信：softmax 让头号专家占得越来越多，而 sigmoid 在 logits 变大时会饱和，于是一堆自信的专家最后拿到权重相同。当选择只依赖这些分数本身时，门控函数不改变排序。3.3 节的专家偏置是在**门控之后**加到分数上再去选的（[`moe_utils.py#L936-L938`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/moe_utils.py#L936-L938)），所以它把一个专家推移多远取决于分数的大小，换个门控函数就可能改掉 top-k。分组受限路由（3.7 节）也会，因为它按各组最好分数的和来选组（[`moe_utils.py#L712-L715`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/moe_utils.py#L712-L715)）。

> **图 3.1-a** 同一个 token 从不自信路由器推到自信路由器：softmax、sigmoid、sqrt(softplus) 的权重曲线（原站为交互图，此处从略）

在图中，sqrt(softplus) 落在 softmax 与 sigmoid 之间：对正的 logits，随着尺度变大，两个权重之比趋向于它们 logits 之比的平方根。DeepSeek-V4 从 sigmoid 换成了 sqrt(softplus)。在 [Megatron](https://github.com/NVIDIA/Megatron-LM) 里，3.3 节的专家偏置要求 sigmoid 或 sqrt(softplus)：启动时它会拒绝 softmax 配偏置（[`transformer_config.py#L3139-L3147`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L3139-L3147)），而它的 softmax 分支从不加这个偏置。提出该偏置的 Wang 等人解释过它为什么和 softmax 不搭：softmax 的归一化让两个专家之间的差距取决于其他每个专家的分数，于是偏置更难调。他们仍然用它在 softmax 路由器上做到了均衡。辅助损失三种门控都能用（[`moe_utils.py#L998-L1014`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/moe_utils.py#L998-L1014)）。

```python
# megatron/core/transformer/moe/moe_utils.py#L913-L927（softmax，在 top-k 之前或之后）
if score_function == "softmax":
    if use_pre_softmax:
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    else:
        scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
# 同文件 L928-L941（sigmoid / sqrt(softplus)；偏置只参与选择）
elif score_function in ("sigmoid", "sqrtsoftplus"):
    scores = torch.sigmoid(logits.float()) if score_function == "sigmoid" \
             else torch.nn.functional.softplus(logits.float()).sqrt()
    if expert_bias is not None:
        scores_for_routing = scores + expert_bias.float()
        _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
```

```python
# megatron/core/transformer/transformer_config.py#L3139-L3147（softmax 配偏置直接报错）
if self.moe_router_enable_expert_bias and self.moe_router_score_function not in (
    "sigmoid", "sqrtsoftplus",
):
    raise ValueError(
        "Expert bias for aux-loss-free routing only supports 'sigmoid' and 'sqrtsoftplus' ..."
    )
```

sigmoid 在被选专家的 logits 都很大且为正时，会给它们相同的权重——这也正是模拟里那个被放大后的 token 落到的位置：一个自信的路由器，配上「锐度」那节所说的、平坦的门控。想知道你这次实验的路由器有没有走到那里，就带着 logits 打开路由追踪（A.1 节），逐层跟两个数：头号专家的权重，以及路由熵（一个专家吃掉大部分权重时熵低，权重摊开时熵高）。

**什么时候用它**：当你用专家偏置做均衡时（3.3 节），用 sigmoid（或 sqrt(softplus)），因为 Megatron 不允许 softmax 配偏置。其余情况用 softmax，也就是 Megatron 的默认值。

路由权重干两件事。它挑专家，同时它又是混合专家输出的那些系数：这一层返回的是「每个被选专家的输出乘权重」再求和。门控函数决定的是「被选专家之间谁的份量更大」，它不决定「它们合起来有多大」：用 softmax 时 top-k 的权重和小于 1，而且每个 token 少多少还不一样，于是这一层输出的量级会跟着晃。那么这个合起来的输出该有多大？

### 输出缩放

有两个设置决定被选专家合起来的输出有多大。第一个是重归一化：把 k 个被选权重重新缩放到和为 1，输出的量级就不再随这个和摇摆。在 Krajewski 等人的实验里，粒度 8 下重归一化把验证 loss 从 2.219 降到 2.183（粒度 8：专家小 8 倍、数量多 8 倍、每 token 选 8 倍，于是参数和算力不变；3.4 节比较不同粒度）。Megatron 默认就重归一化（[`moe_utils.py#L913-L942`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/moe_utils.py#L913-L942)），`--moe-router-pre-softmax` 把重归一化关掉。top-1 的 softmax 必须有这个开关：只选一个专家时，对它单个 logit 做 softmax 恒等于 1，路由器就拿不到梯度（[`transformer_config.py#L3213-L3221`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L3213-L3221)）。

> **图 3.1-b** 模拟：从不自信到自信路由器；不重归一化时输出量级跟着 top-k 概率质量走，重归一化把它压在「一个专家 RMS 的 1/√k」到「一个专家 RMS」之间（原站为交互图，此处从略）

第二个设置是路由缩放系数，那个乘在结果上的常数。它之所以必要，是因为即便重归一化，路由输出的量级也取决于每个专家占多少。全部权重压在一个专家上时，输出就是那个专家的均方根（RMS）；权重平均分给 k 个互不相关、RMS 相同的专家时，它掉到前者的 1/√k，一个接近 √k 的缩放系数能把量级找回来。DeepSeek-V3 在 k=8 时用 2.5，Kimi K2 用 2.827 ≈ √8。已发布的模型在取值上并不一致：上表里只有 GLM-5.3 与 Step-5-Preview 贴着 √k，而 Kimi 从 K2 的 √8 变成了 K3 在 k=16 时的 1.0。缩放路由输出也会改变它相对于共享专家输出的权重（3.4 节）。

图里最后一行是缩放专家的**输入**而不是输出：Llama 4 把每个 token 只送给一个专家，并在专家执行前先把这个 token 乘上该专家的 sigmoid 分数。它的 SwiGLU 专家没有偏置，于是输出大致按分数的平方缩小，同时这个分数也把 SiLU 门软化。

```python
# megatron/core/transformer/moe/moe_utils.py#L942-L947（先重归一化，再缩放）
probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
if scaling_factor:
    probs = probs * scaling_factor
```

```python
# megatron/core/transformer/transformer_config.py#L3213-L3221（top-1 必须 pre-softmax）
if (
    self.moe_router_topk == 1
    and self.moe_router_score_function == "softmax"
    and not self.moe_router_pre_softmax
    and self.moe_router_load_balancing_type != "sinkhorn"
):
    raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")
```

不重归一化时，路由输出的量级会逐 token 跟着 top-k 概率质量走——也就是上面那个 loss 更高的设置。所以该盯的是：每个 token 的路由输出有多大，即它权重平方和的平方根。用路由追踪离线算（A.1 节）。

**什么时候用它**：top-k > 1 时保持重归一化；top-1 的 softmax 要加 `--moe-router-pre-softmax`，否则路由器没梯度。路由缩放系数没有定论：√k 只在「互不相关、权重相等」时恢复路由输出的 RMS，而已发布模型在取值上并不一致。

权重和输出的量级都取决于 logits 有多尖。路由器可能太尖（一个专家吃掉全部权重），也可能太平（把 top-k 平均掉）。是什么决定了它有多尖？

### 锐度

用 softmax 时，线性路由器会随着 token 的范数或路由器权重的增长变得更尖：logits 被拉开，头号专家吃掉更多权重。下面这张图比较了三种控制手段与线性路由器。**logit 软截断**主要压缩靠近上限的那些 logits。**归一化路由器输入**去掉来自 token 范数的那部分。**门控 logit 归一化**在专家维度上把 logits 标准化，把它们的展布定到一个选定的尺度。

> **图 3.1-c** 线性路由器 vs logit 软截断 vs 输入归一化 vs 门控 logit 归一化（原站为交互图，此处从略）

图里那个朴素线性路由器会随 token 范数或路由矩阵增长而变尖；软截断（Grok 2 在上限 30、不重归一化的条件下用它）只有在 logits 接近上限之后，才把大范数 token 的头号权重拉回来。归一化输入（Gemma 4 这么做）让两个 token 走同样的路由，但路由矩阵长大时它仍然变尖；门控 logit 归一化对两者都免疫。Skywork-MoE 用过门控 logit 归一化去治一个平到「这一层把 top-k 专家平均掉」的门，结果训练 loss 适度地低了。归一化路由器输入这一项只与专家 embedding 一起被消融过。

```python
# megatron/core/transformer/moe/router.py#L1150-L1152（先加抖动，再走朴素线性门控）
input = self.apply_input_jitter(input)
logits = self.gating(input)
# 同文件 L824-L831（乘性输入抖动）
if self.config.moe_input_jitter_eps is not None:
    eps = self.config.moe_input_jitter_eps
    self.input_jitter = torch.distributions.uniform.Uniform(
        torch.tensor(1.0 - eps, ...), torch.tensor(1.0 + eps, ...)
    ).rsample
    return input * self.input_jitter(input.shape)
```

一个平掉的门（把一个层的 top-k 专家平均掉，让模型学得更差）会表现为 Max1/Max2 趋向 1，也就是头号专家的权重除以第二名的权重——Skywork-MoE 在不做归一化时就是这样。把它和头号专家权重一起逐层盯，数据来自路由追踪（A.1 节）。

**什么时候用它**：门控变平了，Skywork-MoE 那边是用门控 logit 归一化治好的。在路由器输入这一侧，Megatron 提供 Switch 式的乘性输入抖动 `--moe-input-jitter-eps`。

锐度活在 logits 之间的差距里，而 softmax 对「所有 logit 共同平移」是不敏感的。bf16 不是：如果每个 logit 都一起往上漂，bf16 能表示的值之间间隔就变大，靠得近的专家可能并列。这些并列会改变谁被选中吗？

### 精度

top-k 对并列专家的排序方式，PyTorch 并没有规定。下面这张图模拟了用 bf16 路由器给同样的 token 在 64 个专家上做 top-8，分别在有 z-loss 和没有 z-loss 的情况下——z-loss 是对 logits 的 logsumexp 的惩罚项。模拟里所有 logits 共同上漂一个偏移量，然后统计有多少 token 的专家与「用 fp32 路由器选」不同。

> **图 3.1-d** 共同偏移量上升时，bf16 路由器选出的专家与 fp32 的差异（有/无 z-loss）（原站为交互图，此处从略）

随着模拟偏移量上升，bf16 路由器的 logits 落到更粗的格点上，越来越多 token 的专家与 fp32 不同。z-loss 通过把 logits 往负方向推来压低 logsumexp。在 64 选 8 时，第 8 个 logit 大约比 logsumexp 低 3.5，于是模拟里带 z-loss 时它落在约 −2.2、不带时落在 +3.6，两个量在 bf16 里被按同样长度的步长圆整，改变的 token 数也差不多；只有 fp32 路由器没有翻转，因为 fp32 的圆整细 65,536 倍。DeepSeek-V3.2、GLM-4.5 和 Kimi K2.5 在它们的推理代码里用 fp32 算路由器。fp32 的理由建立在训练证据上，而那件事关稳定性：在 Switch Transformers 里，bf16 路由器发散了，而 fp32 路由器以 bf16 的速度训完了。在 ST-MoE 里，路由器 z-loss 让每次实验都稳住了。各家实验室很少说他们用什么精度训路由器：上表 10 份报告里只有 2 份提到，而 10 份里有 5 份在推理代码里用 bf16 做路由器那次矩阵乘。在这一层其余部分能降到多低位宽、同时路由器保住精度，是 3.7 节「精度」会回来讲的问题。

```python
# megatron/core/transformer/moe/router.py#L124-L130（路由器精度）
router_dtype = input.dtype
if self.config.moe_router_dtype == 'fp32':
    router_dtype = torch.float32
elif self.config.moe_router_dtype == 'fp64':
    router_dtype = torch.float64
logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
# megatron/core/transformer/moe/moe_utils.py#L171-L184（z-loss）
logsum = torch.logsumexp(logits, dim=-1)
z_loss_values = torch.square(logsum)
```

bf16 路由器能让训练发散（Switch Transformers 就是），所以有两个信号值得跟：z-loss，它追踪 logits 上漂了多少；以及翻转率，即用 fp32 重算路由器时有多少 token 换了专家。设了 `--moe-z-loss-coeff` 时 Megatron 会记录 z-loss，且已经除过系数（[`router.py#L805-L807`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router.py#L805-L807)），所以它的平方根就是 logsumexp 的 RMS：我们那次实验记到 20.9，对应 RMS logsumexp 约 4.6。翻转率需要同一批输入、同一个当前路由器状态下的配对抓取；A.1 节给了一个有界的实时 hook，因为追踪里那份初始权重快照跟不上后来的训练更新。

**什么时候用它**：永远设 `--moe-router-dtype fp32`。对 logits 上漂的 softmax 路由器，加上 `--moe-z-loss-coeff 1e-3`（ST-MoE 的取值）。DeepSeek-V3 的报告从没为它的 sigmoid 路由器提过 z-loss。

### 该盯什么

| 部位 | 症状 | 该盯的信号 |
|---|---|---|
| 门控函数 | — | 逐层的头号专家权重与路由熵，来自带 logits 的路由追踪（A.1 节） |
| 输出缩放 | 「学不好」：不重归一化时，路由输出随 top-k 概率质量缩放；重归一化把 loss 降低了 | 每个 token 的路由输出有多大，√(Σw²)，来自路由追踪（A.1 节） |
| 锐度 | 「学不好」：门控变平时，这一层把它的 top-k 专家平均掉 | 逐层的头号专家权重与 Max1/Max2（头号权重除以第二名），来自路由追踪（A.1 节） |
| 精度 | 「炸了」：bf16 路由器在 Switch Transformers 里发散 | 记录的 z-loss 已除过系数，平方根即 RMS logsumexp：20.9 对应约 4.6；fp32 翻转率用 A.1 节那个配对实时 hook 量，同一批输入 + 当前路由器状态（追踪里的初始快照无法精确重放后来的前向） |

到这里我们塑造了「一个 token 的 logits 怎么变成权重」：门控函数决定每个被选专家占多少，重归一化和缩放系数决定它们合起来多大，锐度与精度决定分数有多尖、多准。这一切都假定了一种设定：一个**学出来的**路由器，由 token 挑它的 top k。两者都能改：专家可以反过来挑 token，固定的表也可以替掉学出来的选择。那么该由谁来做选择，token 还是专家？这个选择一定要学吗？

## 3.2 路由策略：谁来选，以及怎么选

token 选 top-k 是默认做法：每个 token 给所有专家打分，留下最好的 k 个。本节看两种改法。让专家去挑它的 token，而不是 token 挑专家；或者干脆用一个固定哈希来挑，不要学出来的路由器。

当前开源 MoE 怎么挑专家，以及哪些会按 token id 查表（附 Zhou 等人的研究模型作为 expert choice 的对照）：

| 模型 | 谁选 | 按 token id 查表 |
|---|---|---|
| DeepSeek-V4-Pro | token choice | 前 3 个 MoE 层用 hash 路由 |
| DeepSeek-V4.1-Flash | token choice | 第 1、14 层有 Engram n-gram 记忆（2–4 gram）；没有 hash 路由 |
| EC-CF2 8B/64E（Zhou 等） | expert choice | 无 |
| GLM-5.3 | token choice | 无 |
| Kimi K3 | token choice | 无 |
| Llama 4 Maverick | token choice | 无 |
| MiMo-V2.6-Pro-RL | token choice | 无 |
| Qwen3.8-Flash-Next | token choice | 第 1 层有 n-gram embedding（2–3 gram） |
| Step-5-Preview | token choice | 无 |

在问「这个选择需不需要学」之前，得先定「由谁来做」：因为 token choice 下，很多 token 可能都想要同样的专家，负载就歪了。到底该让每个 token 继续挑它的专家，还是让每个专家去挑它的 token？

### 谁来选

token choice 下，很多 token 可能同时想要同一批专家，负载于是歪掉。expert choice 把角色反过来：每个专家挑它最好的那些 token，于是每个专家的负载一样。代价是**未来 token 泄漏**：一个 token 能不能拿到某个专家，取决于它后面的 token。推理时那些 token 还不存在，于是路由和训练时不一样。

图里的第三条规则「专家阈值」在不去看未来 token 的前提下把负载压得接近均匀，代价是每个 token 拿到的专家个数不固定。在一个 5.75 亿参数的模型上，它的 loss 比 token choice 更低——不管有没有均衡（2.844 对约 2.89）。

> **图 3.2-a** token choice / expert choice / expert threshold 三者对比：序列偏向专家 1 时各自的负载（原站为交互图，此处从略）

图里这条序列偏向专家 1，token choice（画成不配均衡器）把它压爆了。expert choice 把负载摊平了，但后面的 token 会把前面已经选中的挤出去；而专家阈值——它的截断值来自此前同样偏向的批次——不修改任何选择就把负载保持在目标附近，代价是每 token 的专家数不定。

```python
# megatron/core/transformer/moe/moe_utils.py#L891-L897（逐 token top-k，只有 token choice）
# Sorting top-k turned off during inference
return torch.topk(
    scores, k=topk, dim=1,
    sorted=torch.is_grad_enabled() or is_batch_invariant_mode_enabled(),
)
```

token choice 留下的偏斜会让这一步变慢。它体现在 token 计数上，也就是逐层逐专家的 token 数，`--save-tokens-per-expert-interval` 会记录（A.1 节），最好读成 MaxVio：最忙的专家比均值高出多少（MaxVio 是 maximal violation 的缩写：最忙专家的负载减均值，再除以均值。0 是完全均匀；1 表示最忙的专家拿到平均的两倍。提出 3.3 节那个专家偏置的 Wang 等人就是这么定义的）。

**什么时候用它**：用 token choice，这也是 [Megatron](https://github.com/NVIDIA/Megatron-LM) 训练的那种，k 由 `--moe-router-topk` 定。

不管由 token 还是专家来挑，到目前为止这个选择都来自一个学出来的路由器，于是路由本身需要训练。路由真的需要学吗，还是更简单的规则就能挑专家？

### 哈希层

在 DeepSeek-V4 的前 3 个 MoE 层里，一张固定的表把每个 token id 映射到它的专家，路由器只保留用来算权重。这张表是 token id 的固定哈希，在我们的探测里，它对高频 token 的摊开效果不比随机表好。选择这件事没有什么可学的，而 hash 层在最大到 45 亿参数的模型上，与 Switch 层、BASE 层（Lewis 等人提出的均衡分配，靠解一个线性指派问题让每个专家分到相同数量的 token）打平。下面这张图把学出来的路由与 hash 路由并排，上面再放第三种按 token id 查表的方式：Engram，它把一个 token 前面 2–4 个 token 哈希进若干 embedding 表，再把读到的行加到隐状态上。DeepSeek-V4.1-Flash 只在两层保留 Engram，而每一层的专家仍然由学出来的路由器挑。

> **图 3.2-b** 学出来的路由 / hash 路由 / Engram 三种按 token 找专家的方式（原站为交互图，此处从略）

图里两个路由器对「的」这个字的处理不一样：学出来的路由器按上下文逐个选，而 hash 层把每一个「的」都送到它表里那一行指定的 6 个专家，哪怕路上有个分数更高的专家。两行都把 6 个专家的权重归一化到和为 V4-Flash 的路由缩放系数 1.5，图里用平均 Jaccard 相似度衡量四个「的」的专家集合有多像。hash 层放弃的是上下文：同一个 token id 永远选同一批专家，周围是什么都一样。它的扩展性也比学出来的路由差。Clark 等人改变了一个模型里有多少层是 MoE 层：只有一层 MoE 时，hash 路由在他们比较过的方法里 loss 最低；但随着 MoE 层变多，它不再变好，而他们学出来的路由器还在继续变好。而且没有任何偏置能把它掰回来：高频 token 能把分配给它们的专家压爆，没人能动它们。V4.1-Flash 因此放弃了 hash 层，只留下 Engram——而 Engram 不挑任何专家。

在 Megatron 里，`--moe-num-hash-layers` 让前几层 MoE 变成 hash 层（[`router.py#L304-L308`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router.py#L304-L308)）。它们从一张 token-to-expert 表（`tid2eid`）里读每个 token 的专家，表里词表每个条目一行；如果你不加载，Megatron 会用轮转的占位表填上（[`router.py#L313-L329`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router.py#L313-L329)）。这些层里专家偏置是关的，训练中没有东西去纠正一张坏表，所以表本身很要紧。Roller 等人比较过几种建表方式：随机表和「把最高频的 token 贪心地摊到最空的专家上」那种，困惑度差不多，后者更均衡；而按 embedding 聚类把相似 token 分到一组，明显更差。

```python
# megatron/core/transformer/moe/router.py#L304-L308（前 N 层走哈希）
self.is_hash_layer = (
    not self.is_mtp_layer
    and hash_moe_layer_threshold > 0
    and layer_number <= hash_moe_layer_threshold
)
# 同文件 L322-L326（轮转占位表）
token_ids = torch.arange(self.config.hash_moe_vocab_size, device=self.weight.device)
expert_offsets = torch.arange(self.topk, device=token_ids.device)
self.tid2eid = ((token_ids[:, None] + expert_offsets) % self.num_experts).to(torch.int32)
```

> 另外还需要几行：从 Hub 加载 V4-Flash 的 `layers.0.ffn.gate.tid2eid`，并统计逐专家 token 数。

一张把某些专家压爆的表会让这一步变慢，而训练中没有任何东西去重新均衡它，所以要在实验早期就去查：把 hash 层里的逐专家 token 数与学习层里的比一比，数据来自 `--save-tokens-per-expert-interval` 写出的 token 计数（A.1 节）。

**什么时候用它**：只用来复现 DeepSeek-V4 的头几层。像 V4 那样加载一张固定的哈希表，或者一张按频率均衡的表，而不是 Megatron 那个轮转占位表。

### 该盯什么

| 部位 | 症状 | 该盯的信号 |
|---|---|---|
| 谁来选 | 「慢」：token choice 下有些专家分到的 token 比别人多。「学不好」：expert choice 下推理时的路由与训练时不同 | 逐层的 MaxVio——最忙专家比均值高多少，来自 token 计数（A.1 节） |
| 哈希层 | 「慢」：高频 token 能把哈希分给它们的专家压爆，而偏置动不了它们 | hash 层的逐专家 token 数，与学习层对比，来自 token 计数（A.1 节） |

expert choice 与专家阈值能让负载均匀或接近均匀，而偏置动不了一张哈希表的负载。Megatron 训的是学出来的 token choice，在这里，迄今为止还没有什么东西能阻止大量 token 挤到同几个专家上。专家挤爆是 MoE 第一个真正的失效模式，那么在路由器还在学的过程中，怎么让 token 均匀地摊到各个专家上？

（负载均衡见下一节。）

---

**导航**：[全书目录](index.md) · [← 第 2 章 环境与第一次跑通](02-setup.md) · [第 3 章（中）负载均衡 →](03-2-load-balancing.md)
