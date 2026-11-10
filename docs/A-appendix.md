# 附录 A：监控细节与排障索引

> 说明：原站的图是交互组件，这里无法复现；本文按节内顺序编号引用。原文用的 Megatron 版本固定在 commit `d6316eb45`。

## A.1 监控细节

本附录给出第 3 章各处「该盯什么」背后的命令与算法。路由器统计描述工作量的分布，路由 trace 保留每一次专家选择，激活转储显示专家产出了什么；run 级日志把这些测量与 loss、时间、内存放在一起看。

每条配方都针对 [Megatron](https://github.com/NVIDIA/Megatron-LM) commit `d6316eb45`。每次抓取都要连配置与 rank 布局一起保存：它们决定局部专家 id、token 位置与路由分数该怎么解读。trace 与激活只在短的诊断运行里采，与吞吐测量分开跑。

> 代码位置一览（原文内嵌代码，这里只给位置，按需去仓库读）：
> 辅助损失上报 [`router.py#L691-L700`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router.py#L691-L700)、z-loss 上报 [`router.py#L805-L811`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router.py#L805-L811)、逐层标量写盘 [`moe_logging.py#L392-L404`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/moe_logging.py#L392-L404)、给每个路由器挂 hook [`router_trace.py#L217-L221`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router_trace.py#L217-L221)、路由图转 top-k id [`router_trace.py#L353-L365`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router_trace.py#L353-L365)、重算 logits 并存 bf16 [`router_trace.py#L383-L395`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router_trace.py#L383-L395)、参数范数 [`training.py#L5387-L5393`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/training.py#L5387-L5393)、梯度零元素计数 [`clip_grads.py#L282-L288`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/optimizer/clip_grads.py#L282-L288)、最大注意力 logit [`qk_clip.py#L30-L40`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/optimizer/qk_clip.py#L30-L40)、NaN 梯度中止 [`param_and_grad_buffer.py#L396-L406`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/distributed/param_and_grad_buffer.py#L396-L406)。
>
> 作者的三个脚本：[`router_watch.py`](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139#file-router_watch-py) 读 token 计数与路由 trace、在活的 router 上查 fp32 flip；[`expert_watch.py`](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139#file-expert_watch-py) 读激活转储；[`expert_watch_hooks.py`](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139#file-expert_watch_hooks-py) 在激活发生时现场采样。

### 路由器统计

#### 逐层损失

用 [2.4](#s-2-4) 的 [Trackio](https://huggingface.co/docs/trackio) 启动器开 `--moe-per-layer-logging`，就能逐层记录辅助损失与 z-loss。Megatron 只记录配置里启用的损失，并且已经除掉了它们的系数。一个用专家偏置做均衡的运行，只要启用了辅助损失或 z-loss，照样会报这两项。

辅助损失在概率很平时也能读到 1，哪怕 token 计数很不均（[3.3](#s-3-3)），所以直接用 token 计数来衡量负载。z-loss 描述 logits 的量级：把记录值开平方，得到 token 间 logsumexp 的 RMS。

#### 专家计数与 MaxVio

要保存每个本地专家处理的 token 数，把下面两个开关加到训练命令：

```bash
--save run/ \
--moe-grouped-gemm \
--save-tokens-per-expert-interval 100
```

分组线性层上的 hook 会写出 `run/tokens_per_expert/rank*.jsonl`。每条记录标明迭代步与层，`tpe` 列表里每个被捕获的 micro-batch 一个计数向量。解码层与 MTP 层有不同的身份。这些向量描述的是 dispatch 之后的本地专家负载（[`activation_logging.py`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/activation_logging.py)）。

对一个专家计数向量 c，MaxVio 衡量最忙专家相对均值的超出量：

```text
MaxVio = max(c) / mean(c) - 1
```

比如计数 `[2, 2, 4, 0]` 的均值是 2、MaxVio 是 1：最忙专家收到的负载是平均的两倍。零计数的专家也计入均值，必须留在算式里。完全均衡时是 0；全零向量的 MaxVio 无定义。

算之前先定聚合范围：

- **micro-batch**：把一个专家并行（EP）组内的本地专家向量按全局专家 id 顺序拼起来。
- **global batch**：把各 micro-batch 与各独立数据副本上每个专家的计数相加，再从得到的向量算 MaxVio。
- **序列**：用逐次的路由选择与序列边界，见「路由诊断」。

把各 micro-batch 的 MaxVio 取平均，并不会得到 global-batch 的 MaxVio。专家可以在每个 micro-batch 里都不均、在整批上却均衡（[3.3](#s-3-3)）。聚合还必须保留层的身份：流水 stage 与 MTP 深度都要分开。[交错流水](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=interleaving_stages)的 chunk 会复用局部层 id，而保存的记录区分不了它们（[`activation_logging.py#L131-L134`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/activation_logging.py#L131-L134)），所以抓取时把交错关掉。[激活重算](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=activation_recomputation)会让 hook 被调用两次，所以要么关掉重算再抓，要么保留原始前向的标识。

[`router_watch.py`](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139#file-router_watch-py) 这个读脚本按一张显式的「rank → 全局专家 id」映射把计数文件拼起来。两个 rank 各持四个专家时，把 `{"0": [0, 1], "1": [2, 3]}` 存成 `experts.json`，然后跑：

```bash
python3 scripts/router_watch.py counts run/tokens_per_expert \
    --num-experts 4 --ranks 0 1 \
    --expert-map experts.json --pool-ranks
```

`--pool-ranks` 声明所选文件是同一层的独立工作（已去掉张量并行副本）。要逐 micro-batch 测量，就得保留能跨 rank 对上「同一次前向」的标识。

#### 每个 rank 的负载

把一个 EP rank 上所有专家的计数加起来，再拿最忙的 rank 与均值比。这个数与专家 MaxVio 并看：几个忙专家如果摆在同一张 GPU 上，工作就会挤在那儿（[3.3](#s-3-3)、[3.7](#s-3-7)）。

开了容量上限或补齐时，分组 GEMM 的计数描述的是这些操作之后的工作量。要量原始需求，用路由器未被补齐的选择。

### 路由诊断

#### 抓取专家选择

短抓取时把下面这些开关加到训练命令，把 `trace_steps` 换成要抓的迭代数：

```bash
--moe-routing-trace-path trace/ \
--moe-routing-trace-max-training-iters trace_steps \
--cuda-graph-impl none
```

hook 在 dispatch 之前记录被选中的专家 id。加 `--moe-routing-trace-capture-logits` 会额外存 logits 文件；加 `--moe-routing-trace-capture-hidden-states --moe-routing-trace-dump-weights` 会存隐状态与路由器权重。这些抓取都要写盘，而 Python hook 需要 eager 执行。

专家 id 按路由图的顺序排列，所以某个指标需要「主专家」时，要按重建出来的权重排序。丢 token 可能导致整条记录被跳过，这种缺口按缺失观测处理（[`router_trace.py`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/router_trace.py)）。

#### 从 trace 算计数与 MaxVio

被选中的专家 id 本身就能给出 token 计数与 MaxVio，不必抓 logits。读一个持有 64 专家的 rank：

```bash
python3 scripts/router_watch.py trace trace/ --num-experts 64 --ranks 0
```

每条 `trace_call` 记录里有单次调用的 `counts` 与 `maxvio`。`rank_captured_step` 记录则在每层、每 rank、每步内把计数求和、再从和算 MaxVio。这些汇总量只覆盖被捕获的 token；要做 global-batch 测量，需要该批所有原始前向与独立副本，并去掉张量并行副本。拼接抓取时用「专家计数与 MaxVio」那节的聚合规则。因丢分配而被跳过的记录，在这些总量里仍然是缺失的。

#### 逐序列均衡

逐序列均衡需要每个 token 行属于哪个序列。完整、未切分的输入按 `[sequence, batch, hidden]` 排布、batch 为 B 时，第 `s * B + b` 行属于序列 b。只有这种布局成立时，才给读脚本传 `--sequence-length S --batch-size B`。

在每个序列内统计各专家被分到的次数（含零次的专家），再从该向量算 MaxVio。然后按层汇总这些逐序列的值。

对打包序列，把定长布局换成真实的序列边界。上下文并行或[序列并行](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=sequence_parallelism)还需要 token 位置，才能把各 rank 上的片段拼起来。跨观测地跟踪同一个 token 还需要它的身份——两个训练步里同样的行号，通常指的是不同 token。

#### 混合权重

logits 文件支持对「施加在被选专家上的权重」做近似重建。hook 会重算这些 logits、以 bf16 存下。重建权重时要保留记录下来的选择，并使用该次运行的打分函数、归一化方式与路由缩放系数。

一次 64 专家、缩放系数 1 的 softmax 抓取：

```bash
python3 scripts/router_watch.py trace trace/ \
    --num-experts 64 --ranks 0 --score softmax --scale 1
```

如果该次运算是先对全部专家做 softmax 再选，加 `--pre-softmax`；否则 softmax 只作用在选中的 logits 上。用 sigmoid 和 sqrt(softplus) 时，按该次运行对选中分数的归一化来处理。专家偏置影响选择，但不要算进混合权重。

记最终被选权重（含缩放系数）为 w，令 q 为 w 除以其和。每个指标先按 token 算、再按层汇总：

- **最大权重**：`max(w)` 是对专家输出的最大倍率。要跨不同缩放系数比较集中度，用 `max(q)`。
- **选中权重熵**：`-sum(q * log(q))`，范围 0 到 `log(k)`，k 个权重越接近熵越大。top-1 时为 0。
- **Max1/Max2**：最大权重除以第二大的。接近 1 说明前两名权重相近；top-1 时无定义。
- **估计的混合 RMS**：`sqrt(sum(q*q))`，在「各专家输出 RMS 相等且互不相关」的假设下，估计混合结果相对单个专家的 RMS（[3.1](#s-3-1)）。用 w 就把路由缩放系数也算进去了。
- **top-k 概率质量**：对 pre-softmax 路由，在缩放前把被选中的全 softmax 概率加起来。原始 sigmoid 与 sqrt(softplus) 分数不构成全专家上的概率，解读方式不同。

举例：选中权重 `[0.75, 0.25]` 的最大权重是 0.75，Max1/Max2 是 3，熵约 0.562（自然对数）。

#### 每个 token 到达的节点数

要衡量通信扇出，把被选专家 id 映射到目的节点，再数每个 token 到达了几台不同节点。按层报均值与最大值（[3.7](#s-3-7)）。

给读脚本传 `--node-map nodes.json`。该文件把全局专家 id 映射到节点名，例如 `{"0": "node-a", "1": "node-a", "2": "node-b"}`。对复制出来的专家，要抓 dispatcher 实际选中的那个目的地。要只数远端节点，还需要 token 的源节点，才能把它排除。

把扇出换算成字节，需要隐层宽度、dtype 与 dispatcher 的拷贝策略：每个专家一份拷贝、与每个目的节点一份拷贝，流量完全不同。

#### 被丢掉的分配

有容量上限时，从路由器最终的 routing map 量丢弃量。N 个有效 token、K 个存活的分配：

```text
dropped_fraction = 1 - K / (N * topk)
```

这数的是「token-专家分配」的损失。一个 token 可能只保住了它的一部分专家。算的时候要排除被 mask 的 token 和为量化 GEMM 补的行。这个测量要用活 hook，因为存下来的 trace 可能跳过含丢弃的记录。[`ExpertWatch`](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139#file-expert_watch_hooks-py) 能抓这张 map。

#### 精度翻转

fp32 flip rate 衡量「当路由器用 fp32 计算时，被选专家集合多久变一次」。两次计算必须用同一次前向的输入、权重、偏置与选择规则。比较的是集合，这样单纯顺序变化不算翻转。

对普通的 token-choice 路由，[`install_fp32_flip_hook`](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139#file-router_watch-py) 在活状态上做这个比较。把它挂到选定 MoE 层的 router 上、跑一次 eager 前向：

```python
import torch
from scripts.router_watch import install_fp32_flip_hook

previous_tf32 = torch.backends.cuda.matmul.allow_tf32
handle = install_fp32_flip_hook(moe_layer.router, emit=print)
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    output = model(tokens)
finally:
    handle.remove()
    torch.backends.cuda.matmul.allow_tf32 = previous_tf32
```

关掉 TF32 就得到 fp32 精度的参考矩阵乘。hook 在一批有界的有效 token 上比较选中的专家集合。它支持哪些设置见 docstring，不支持的路径会被拒绝。要关掉 CUDA graphs 跑，并且别放在计时测量里。

两次计算都从活输入的现有精度出发，所以结果隔离出的是「该输入精度下的路由分歧」；它对 loss 的影响要另做实验（[3.1](#s-3-1)）。

做离线比较时，要保留被捕获前向的精确状态与门控输入，包括抖动。内置的权重转储只在每个 router 第一次被调用时快照，因此不跟踪后续的权重与均衡偏置更新。隐状态与重算的 logits 以 bf16 存，快照还不含 router 的线性偏置（如果有）。所以从这些文件做的重建应标注为「近似」，并说明抓取步与缺失状态。

### 专家激活

#### 抓取激活

加 `--save run/ --save-activations-interval 100` 可以抓受支持线性模块的输入输出，含分组专家的 token 计数。每个模块的条目会在它下一次被抓时被覆盖，所以留下的是最后一次抓到的 micro-batch。hook 在抓取时会把这些张量拷到 CPU，所以间隔要稀疏、运行要短，并关掉 CUDA graphs 和激活重算。

#### 逐专家输出范数

原生转储里的 `experts.linear_fc2/output0` 含本地专家那几行。用模块的 `input1` token 计数把这些行切开，转成 fp32，算每个 token 输出向量的 L2 范数。这些范数的均值描述该专家的输出量级。把每个范数除以输出宽度的平方根就得到 token RMS，便于跨宽度比较。

[`expert_watch.py`](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139#file-expert_watch-py) 做这些计算。在有 PyTorch 的环境里跑：

```bash
python3 scripts/expert_watch.py \
    --unpadded --expert-tp-size 1 activations.pth
```

这两个选项断言「抓到的行不含补齐」「专家张量并行是 1」。脚本报每个专家的 token 数、范数之和与平均范数，还会报「收到 token 的那些专家里」相对中位数的最大与最小平均范数比（[3.4](#s-3-4)、[3.6](#s-3-6)）。

空专家没有平均输出范数，所以脚本把它们单列出来。中位数为零时这些比值也无定义。把一个专家所有行拼起来再取范数，会把输出量级和 token 数混在一起；对逐 token 范数取平均则把这两者分开了。

这个中位数覆盖的是一个文件里的专家。要做整层比较，先把 EP 分片拼起来再算中位数。多次观测可以合并：把每个专家的 `sum_l2` 与 `rows` 相加再相除，得到均值。专家张量并行大于 1 时，要先合并部分输出向量——它们的单个范数不能相加来还原完整向量的范数。这些计算都要排除补齐行。

分组 fc2 的输出可能已经含了路由权重，取决于实现把权重施加在哪一步。于是小范数也可能只反映小的路由权重。比较时把「施加权重的位置」固定住，并把混合权重与范数放在一起看。

现场采样时，`ExpertWatch` 在它的 `routed_experts` 输出里报同样的逐专家范数统计，含 `local_max_to_median` 与 `local_min_to_median`。这些比值只覆盖本地专家；算整层中位数前要跨 EP rank 拼统计量。见「共享与路由分支的贡献」那节的配置。

#### 潜在升维

要检查潜在升维，读 `fc2_latent_proj/output0`，比较它的输入与输出 RMS（[3.4](#s-3-4)）。用 RMS 才能抵掉潜在宽度与模型宽度的差异。开了 `--moe-use-norm-before-up-proj` 时，再在归一化之前采一次，定位增长从哪开始。

#### 共享与路由分支的贡献

要比较两个分支，抓「过完可选 sigmoid 门之后的共享输出」与「逆置换、合并之后的路由输出」，有潜在升维时把它算进去。每个 token 算 `norm(shared) / norm(routed)`，分母为零的单列。这个比值比较两个分支的量级；两向量相加时仍可能互相加强或抵消。

原始的 fc2 转储发生得更早：共享 fc2 在门之前，路由 fc2 在合并之前。要在需要的点上现场采样，就把 [`ExpertWatch`](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139#file-expert_watch_hooks-py) 套在一次诊断前向上：

```python
from scripts.expert_watch_hooks import ExpertWatch

with ExpertWatch(moe_layer, emit=print):
    output = model(tokens)
```

这个 helper 支持：eager 执行、all-to-all dispatcher、张量并行与专家张量并行都是 1、关掉共享专家重叠。样本必须未补齐，且关闭量化与专家补齐。把 hook 的范围限制在那次诊断前向上，并把每 rank 的 JSON 输出连同步与 micro-batch 身份一起存。

对标准 MoE，helper 用「总输出减去门后的共享输出」还原路由分支。这个减法会带上原始加法里的舍入误差，所以当路由分支相对共享分支很小、又需要准确值时就别用减法，直接在相加之前抓。对潜在 MoE，helper 直接抓升维的输出。分支汇总会包含补齐位置，这正是这个比较要求未补齐样本的原因。

挂 hook 之前想先在 CPU 上验一遍，跑 `python -m unittest scripts.test_router_watch scripts.test_expert_watch`。

### 运行健康

#### Loss 与梯度信号

把 LM loss 和梯度范数与路由器、专家的测量放在一起。下面这些开关提供 run 级信号：

- `--log-throughput`：训练吞吐。
- `--log-params-norm`：参数范数。
- `--log-num-zeros-in-grad`：整个模型梯度里零元素的数量。
- `--log-max-attention-logit`：最大注意力 logit，需要 [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 2.9 或更新。

要查专家级的下溢（[3.7](#s-3-7)），在梯度累加之后、优化器更新之前，数专家权重梯度里恰好为零的元素。把这个专家级的数与同配置的 BF16 运行比。

NaN 检查默认开启。检出 NaN 会中止这一步并触发重跑处理（[3.6](#s-3-6)）；`--no-check-for-nan-in-loss-and-grad` 关掉它。

#### 时间与内存

加 `--timing-log-level 2 --timing-log-option all` 得到逐 rank 的前向与反向耗时。按流水 stage 给 rank 分组，定位慢的 stage（包括持 MTP 的那个，[3.6](#s-3-6)）。然后用 [3.7](#s-3-7) 的工具剖析那个 stage。

跨运行比较时，比同一段「预热之后」的区间。测峰值内存时，在区间开始重置 CUDA 峰值统计、在区间结束读每个 rank 的 `torch.cuda.max_memory_allocated()`，取各 rank 的最大值作为最高峰值。

## A.2 排障索引

第 3 章是一节一节顺着讲的：从一个开关讲到它对这次运行做了什么。运行出问题时你是反着走：先在看板上看到一个症状，再去找它背后的配置。本附录就是那张反向索引，一行一个症状，各自指回正文里解释它的那一节。

不过在怪某个设置之前，先用 [2.5](#s-2-5) 末尾那两项检查把技术栈本身排除掉。

### 怎么读这张索引

每行带一个标签、症状（以及最先动的那个信号）、还有该去读的小节。标签按「怎么坏」给运行分类：`slow`（慢）、`OOM`（爆内存）、`blows up`（炸掉）、`learns badly`（学不好）、`won't start`（起不来，指第一步之前就停）。先找你的标签，再看最先动的信号，那一行会告诉你该去哪。（引言里那张图把这些行都画了，只差一行 `global_aux_loss` 设置——它没有起始信号。）

| 标签 | 症状 | 该读 |
|---|---|---|
| `slow` | 吞吐下降 | [3.3](#s-3-3)、[3.7](#s-3-7) 剖析 |
| `slow` | 单节点一步里一半时间都是 All-to-All | 先 [3.3](#s-3-3) 均衡，再 [3.7](#s-3-7) 重叠 |
| `slow` | 辅助损失接近 1，可有一张 GPU 很慢 | [3.3 负载不均的代价](#s-3-3-cost-of-imbalance)：概率平坦时辅助损失恒为 1，与负载无关，要看 MaxVio |
| `slow`、`learns badly` | 用无损失偏置跑了很久，仍不均衡 | [3.3 无损失偏置](#s-3-3-loss-free-bias)：固定步长滞后，改用分位数均衡 |
| `slow`、`learns badly` | 只有前几层有几个专家过载 | [3.2](#s-3-2) 哈希层、[3.5 稠密层](#s-3-5-dense-layers) |
| `slow` | 每个 token 的 All-to-All 都到达所有节点 | [3.7 互联](#s-3-7-interconnect)与组限路由 |
| `slow` | 换成更多、更小的专家后吞吐下降 | [3.4](#s-3-4) 数量与宽度 |
| `slow`、`OOM` | 开 MTP 后一步慢很多，或流水并行下多了几层就 OOM | [3.6 MTP](#s-3-6-mtp) |
| `slow` | 剖析里 All-to-All 是裸露的 | [3.5 块接线](#s-3-5-block-wiring)、[3.7 重叠](#s-3-7-overlap) |
| `slow` | 多节点慢约 25 倍（我们的 EP16 代理），且 NCCL 日志说 "Using network Socket" | [3.7 互联](#s-3-7-interconnect) |
| `slow` | 换成 `global_aux_loss` 后一步变慢 | [3.3 范围](#s-3-3-scope) |
| `slow` | 分位数均衡下，global batch 的 MaxVio 低、micro-batch 的高 | [3.3 分位数](#s-3-3-quantile) |
| `slow` | 加 `--moe-latent-size` 后一步变慢 | [3.4 潜在 MoE](#s-3-4-latent-moe)：两个 d × ℓ 投影与更小的专家 GEMM；LatentMoE 报的吞吐是服务侧的，并发 1 时低 12% |
| `slow` | 在路由 top-k 之上加共享专家后推理变慢 | [3.4 共享专家](#s-3-4-shared-experts) |
| `slow` | 全量重算还开着 | [3.7 内存](#s-3-7-memory) |
| `slow` | 均衡、重叠、融合都做了，TFLOP/s 还是不够 | [3.7 精度](#s-3-7-precision) |
| `OOM` | 某个 rank OOM | [3.3](#s-3-3)（不均衡）、[3.7](#s-3-7) 内存 |
| `OOM` | 峰值内存看着没问题，却有一个 rank OOM | [3.7](#s-3-7) 内存（Megatron 只打数据并行 rank 0） |
| `blows up` | 一个 NaN，之前什么都没有在动（梯度范数、参数范数、z-loss 都平） | [3.6](#s-3-6) 稳定：没有预警；在另一张 GPU 上重跑能区分坏卡与可复现的 NaN |
| `blows up` | 同一节点上两次退出码 16 | [3.6](#s-3-6) 稳定：重跑需要换一张 GPU |
| `blows up` | Muon 运行在最初几步就发散 | [3.6](#s-3-6) 优化器（更新尺度、`--muon-extra-scale-factor`） |
| `blows up` | z-loss 或路由 logits 在尖峰前一路爬升 | [3.1](#s-3-1) 精度（若 NaN 之前 z-loss 反而平，则是 [3.6](#s-3-6) 稳定问题） |
| `blows up` | 专家输出范数在增大，或它的 max/median 比在尖峰前爬升 | [3.4](#s-3-4) 激活（clamp） |
| `blows up` | 潜在 MoE 出现激活爆炸或尖峰 | [3.4](#s-3-4) 潜在（升维前归一化） |
| `blows up` | NaN 前几步梯度范数与负载均衡损失一起跳 | [3.6](#s-3-6) 稳定、[3.3](#s-3-3)。我们的 7B-A1B |
| `blows up` | Muon 下的 loss 尖峰，从附近 checkpoint 续训常能躲开 | [3.6](#s-3-6) 优化器（bf16 里做正交化） |
| `learns badly` | RL reward 停滞或塌掉，同时训推概率差在变大 | [第 4 章 RL 路由](04-serving.md) |
| `learns badly` | loss 平，但 MaxVio 或某专家输出范数偏离该层中位数（塌缩、死专家、流氓专家） | [3.6](#s-3-6) 稳定、[3.3](#s-3-3) |
| `learns badly` | top-k 权重全相等（门很平） | [3.1](#s-3-1) 尖锐度 |
| `learns badly` | 开着辅助损失却出现死专家 | [3.3 辅助损失](#s-3-3-aux-loss)（系数太小） |
| `learns badly` | `global_aux_loss` 且每个 global batch 超过 256 个 micro-batch | [3.3 范围](#s-3-3-scope) 的开关（bf16 计数器） |
| `learns badly`、`blows up` | 最前面几个 MoE 层路由塌缩或发散 | [3.5 稠密层](#s-3-5-dense-layers) |
| `learns badly` | 设了容量系数，loss 落后于不丢 token 的运行 | [3.7 容量与副本](#s-3-7-capacity-and-replicas)（丢弃量不会记日志） |
| `learns badly` | Muon 下专家一层一层塌缩 | [3.6](#s-3-6) 优化器 |
| `learns badly` | 用 `--moe-router-pre-softmax`（top-k 权重不再归一化），loss 高于归一化的运行 | [3.1 输出尺度](#s-3-1-output-scale) |
| `learns badly` | 专家没有按领域分化 | [3.3 范围](#s-3-3-scope) |
| `learns badly` | 系数调大后 LM loss 反而上升 | [3.3 辅助损失](#s-3-3-aux-loss) |
| `learns badly` | 专家变得更小之后 loss 不再改善 | [3.4 数量与宽度](#s-3-4-count-and-size) |
| `learns badly` | 用专家选择（expert choice）路由训的模型，推理时表现不同 | [3.2 谁在选](#s-3-2-who-selects) |
| `learns badly` | upcycle 出来的 MoE 领先了一阵之后就学得慢 | [3.6 upcycling](#s-3-6-upcycling) |
| `won't start` | `--moe-use-upcycling` 报 "unexpected keyword argument 'model'" 或缺 `pre_mlp_layernorm` 键 | [3.6](#s-3-6) upcycling（两个上游 bug） |
| `won't start` | 开 `--overlap-moe-expert-parallel-comm` 后断言全量重算 | [3.7](#s-3-7) 重叠与内存 |

**信号比症状重要**，因为同一个症状可能来自完全不同的部位。我们那几个 NaN 之前，一直在看的信号（梯度范数、参数范数、z-loss）都是平的，追下去的那个 NaN 生在分组专家 GEMM 的反向里。这类 NaN，用 Megatron 在另一张 GPU 上的重跑就能区分「坏卡」与「数据或模型真产生了 NaN」（[3.6 稳定](#s-3-6-stability)）。一步变慢而辅助损失接近 1，应该改看 MaxVio：[Megatron](https://github.com/NVIDIA/Megatron-LM) 的辅助损失在路由器概率平坦时[恒等于 1](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/moe_utils.py#L147-L150)，与负载无关，所以一张过载的 GPU 可以藏在它后面（[3.3 负载不均的代价](#s-3-3-cost-of-imbalance)）。

## 代码、配置与数据

原文的这部分是一份逐条来源清单：每个数字与结论对应到具体配置文件、代码行或论文，例如 [Qwen3-30B-A3B 的 config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json)、[DeepSeek-V3 的 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json)、DeepSeek-V4-Pro 推理代码里的 `Gate`（[`model.py#L556`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py#L556)，`layer_id < n_hash_layers` 时走哈希路由）、[datatrove 的 executor](https://github.com/huggingface/datatrove/blob/2c895af7a95010a96fa53e2dc77cc0ce92799e3f/src/datatrove/executor/base.py#L30) 等。完整清单见原文页面底部的「Code, configs and data」。

---

**导航**：[全书目录](index.md) · [← 第 5 章 结语与推荐参数](05-conclusion.md)
