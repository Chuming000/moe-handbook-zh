# 第 3 章 解剖 MoE 训练（中）：负载均衡

> 说明：原站里的图都是交互式组件，这里无法复现，一律保留图注并标注「（原站为交互图，此处从略）」。原站未给图编号，本文按**节内顺序**编号（如 图 3.3-a）。

## 3.3 负载均衡

当大量 token 都挑了同一批专家，会坏两件事。一是装着最忙专家的那块 GPU 活最多，别的 GPU 得等它，于是它决定了整个步的时间；二是如果路由器一直把 token 发给同几个专家，剩下的专家几乎收不到 token，参数就基本白放着。第一种代价更好量：把路由强制摊平之后，同一个模型在初始化时就快了 1.5 倍。

强制摊平只是把「该追求的速度」摆出来，因为它把路由器学到的东西全扔了。要在路由器继续学习的前提下逼近那个速度，有两类办法：辅助损失（aux loss）和免损失偏置（loss-free bias），两者均衡负载的方式不同。辅助损失加一项奖励均匀负载的损失，于是路由器的梯度会把它的概率往「摊开 token」的方向推，代价是这个梯度会和语言模型损失（LM loss）抢方向。偏置则只改变「哪些专家被选中」。不管你选哪种，它**在多大范围上均衡**（一条序列、一个 micro-batch、还是全局批次）决定了「均衡」到底是什么意思。

各家模型作者报告的均衡方案。辅助损失与选择偏置可以同时用。「范围」指的是被均衡的那些 token；MiMo 那一行讲的是 RL 而不是预训练。

| 模型 | 辅助损失 | 范围（scope） | 选择偏置 | 分位数均衡 |
|---|---|---|---|---|
| DeepSeek-V3 | 小的 sequence-wise loss，系数 0.0001 | 批次级偏置 + sequence-wise loss | 固定步长的专家偏置 | 否，固定步长更新 |
| DeepSeek-V4.1-Flash | 小的 sequence-wise loss，系数 0.0001 | 文本与图像的负载分别均衡，外加 sequence-wise loss | 每种模态各自一个偏置，更新速率 0.001 | 用的是无辅助损失的偏置策略 |
| Kimi K3 | 描述为「无辅助损失」 | 全局批次 | 由分数间距推出的偏置，下一步生效 | 是，全局批次直方图估计 |
| LongCat-Flash | 设备级均衡损失 | 全局批次 FFN 目标 + 序列内设备级均衡 | PID 控制的专家偏置；恒等专家不参与偏置更新 | 否，PID 更新 |
| MiMo-V2.6-Pro-RL | RL 期间路由器冻结 | 保留 RL 之前的路由器 | 报告评估的是冻结，而非新的偏置更新规则 | RL 没有报告新的均衡更新 |
| Qwen3 MoE | 全局批次均衡损失 | 全局批次 | 公开的路由器里没有选择偏置 | 否，用辅助损失路线 |

在逐一比较这些均衡器之前，得先知道「负载不均可劲儿到掉多少速度」，因为下面每个均衡器都该拿「强制摊平时的吞吐」当尺子。那个 1.5 倍是从哪来的，又怎么在我们自己的机器上量？

### 负载不均的代价

为了看清不均的代价，我们把同一个模型按三种方式路由。**强制均摊**（[Megatron](https://github.com/NVIDIA/Megatron-LM) 的 `--moe-router-force-load-balancing`）把路由器的 logits 换成随机值，于是每个专家分到的 token 差不多，这就是最好的情况：149.4 TFLOP/s。

**强制偏斜**（`--moe-router-force-biased`）在这些随机 logits 上加一个所有人共享、逐渐变大的随机偏置，于是少数专家分到越来越多的 token——相当于一个「不均衡程度」的旋钮。拧到头吞吐掉到 100.5 TFLOP/s；把那次路由回放出来（它的辅助损失与日志里的对上），最忙的专家接走了每一个 token，MaxVio 为 7，这已经是 64 选 8 能达到的最歪。还能更歪：回放里最忙的几个专家分散在几块 GPU 上，所以最忙的那块只承担了约两倍的平均负载；如果它们全坐在一块 GPU 上，那块 GPU 就得干绝大部分专家活。

学出来的路由器——也就是你真正训练时会用的那个——离这个最好情况很远：一个刚初始化的路由器，头 40 个迭代跑得和最差的强制偏斜一样慢。下图把三者放在一起比。训练要做的，就是在不强制路由器选择的前提下去补上这段差距，本节余下部分就用「把一个学出来的路由器带到多接近强制均摊的吞吐」来比较各种均衡方法。

> **图 3.3-a** 强制均摊 / 强制偏斜（偏置逐步变大）/ 学出来的路由器：吞吐（原站为交互图，此处从略）

图里吞吐随强制偏置变大而下降，因为少数专家吃掉了更多 token；而学出来的路由器从最歪那条强制曲线的水平起步。最忙的专家决定了步长：All-to-All 会把每个 GPU 本地专家收到的 token 全送过去（我们这次是 64 个里的 8 个），下一次 All-to-All 要等最慢的那块 GPU。一块 GPU 的负载是它本地那些专家之和，所以它可以比逐专家的负载更均衡。这也是为什么均衡损失可以按 GPU 算而不是按专家算：DeepSeekMoE 的设备级均衡损失把每块 GPU 的专家并成一项，下一张图把它和专家级损失放在一起比。

> **图 3.3-b** 玩具批次（每块 GPU 两个专家，其中一个是 LM 偏爱的四个之一）：专家级损失 vs 设备级损失（原站为交互图，此处从略）

在一个玩具批次上（每块 GPU 两个专家，其中一个属于 LM 偏爱的四个），两种损失都能把 GPU 摊平，而设备级损失把路由器拉离「LM 想要的」更少，因为它只看每块 GPU 的总量，而这个摆法让这个总量更接近均匀。问题在于每块 GPU 内部的专家仍然不匀，单个专家仍可能占掉这块 GPU 的大多数 token。DeepSeekMoE 给的也是同一个理由去给两种损失不同的权重：严格的专家级均衡会伤模型质量，所以它把设备级的系数设大、把专家级的系数留小，用来防路由崩塌。DeepSeek-V2 用的是 0.05 对 0.003。DeepSeek-V3 的报告里已经不再提设备级损失：它用一个偏置加一个小小的 sequence-wise 损失来均衡每个专家，两者本节后面都会讲。

同样的不均在服务端也是时间成本，所以生产环境里的前沿 MoE 会把它压得很小：DeepSeek 在服务 DeepSeek-V4.1-Flash 时，最慢的那块 GPU 最多只到平均的 1.09 倍。

```python
# megatron/core/transformer/moe/router.py#L1185-L1193（前向里切换强制路由）
if self.config.moe_router_force_load_balancing:
    # Apply force load balancing with random logits for benchmark
    logits = apply_random_logits(logits)

if self.config.moe_router_force_biased is not None:
    # Apply biased logits with shared random bias across all ranks
    logits = apply_biased_logits(
        logits, self.config.moe_router_force_biased, self.layer_number
    )
```

```python
# megatron/core/transformer/moe/moe_utils.py#L1301-L1314（每个 rank 各自的随机 logits）
@staticmethod
def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
    with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
        random_logits = logits.clone().normal_()
    return random_logits
```

```python
# megatron/core/transformer/moe/moe_utils.py#L1353-L1364（每层共享的随机偏置）
@staticmethod
def forward(ctx, logits, std, layer_number):
    """Forward pass: apply random bias to logits."""
    if std < 0 and layer_number in RandomSTEShared._cache:
        return logits + RandomSTEShared._cache[layer_number]

    # Generate random bias with shared seed across all ranks
    with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
        bias = torch.empty(logits.shape[-1], device=logits.device,
                           dtype=logits.dtype).normal_(std=abs(std))
```

负载不均表现为「步变慢」，所以要盯的数字是相对强制均摊对照跑的 TFLOP/s。想看丢的时间去哪了，就看 3.7 节那样剖析一个步：那次剖析跑的是不均衡的路由器，3.7 节演示怎么判断是不是不均衡在拖慢它的 All-to-All。

**什么时候用它**：每套配置跑一次，当对照用。`--moe-router-force-load-balancing` 给出均衡时的吞吐，拿来当所有均衡器的尺子。永远不要用它训练。

强制路由能告诉你「均匀负载能跑多快」，但它把 logits 换成了随机值，所以只能当对照，不能拿来训练。那么，怎么在训练过程中把一个学出来的路由器引向均匀负载？先从最老的办法说起：加一项损失。

### 辅助损失

最老的办法是加一项损失，也就是 aux loss：系数 × 专家数 × 对专家求和，每项是「该专家在 micro-batch 里占的 token 比例」乘「它的平均路由概率」（[`moe_utils.py#L148-L152`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/moe_utils.py#L148-L152)）。它的梯度把路由概率从「LM 想要的」拉向「均匀负载」。下面用四种设置跑同一个批次：不均衡、小系数的 aux loss、大系数的 aux loss、以及无损失的偏置。

> **图 3.3-c** 同一个批次在四种设置下的路由：不均衡 / aux 小系数 / aux 大系数 / 无损失偏置（原站为交互图，此处从略）

图里两个均衡器都用不同的方式把负载摊开了：aux loss 把路由概率从 LM loss 想要的位置拉开，系数越大拉得越远；而无损失偏置只改变「挑哪些专家」，均衡得最好，但它改掉的「选择」也最多。aux loss 这种「拉扯」让系数成了一个取舍：系数太大伤 LM loss，系数太小负载仍然不匀。Yuan3.0 在 10B 模型上对 DeepSeek-V3 那个 sequence-wise aux loss 做了消融（下一张图），两面都摆出来了：测试损失从系数 1e-3 到 0.1 一路升高；而最小的系数 1e-4 比完全不加 aux loss 略好一点，但一层里最忙的专家仍然拿到最闲专家的约 13.6 倍 token。

> **图 3.3-d** Yuan3.0 的消融：测试损失与 aux 系数（原站为交互图，此处从略）

图里最小的系数留下了不均衡，而真正把负载压平的那个 1e-2 又把测试损失抬高了。Megatron 用 `--moe-aux-loss-coeff` 配 `--moe-router-load-balancing-type aux_loss` 设系数，每种损失类型一个系数，在所有层都生效。Motif 3 发现它最前面几层 MoE 更容易失衡，于是给它们更强的辅助损失权重。最前几层的问题本章后面还会回来讲。

```python
# megatron/core/transformer/moe/moe_utils.py#L148-L152（Σ f·P 乘系数）
aggregated_probs_per_expert = probs.sum(dim=0)
aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
    num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
)
return aux_loss
```

```python
# megatron/core/transformer/moe/router.py#L515-L536（在 micro-batch 内计数）
aux_loss_coeff = self.get_aux_loss_coeff("aux_loss")
if aux_loss_coeff == 0:
    return probs

global_tokens_per_expert, local_num_tokens, total_num_tokens = (
    get_tokens_per_expert_and_token_count(
        routing_map=routing_map,
        reduce_group=self.tp_cp_group,
        topk=self.topk,
        with_padding_mask=with_padding_mask,
    )
)

aux_loss = switch_load_balancing_loss_func(
    probs=scores_for_aux_loss,
    tokens_per_expert=global_tokens_per_expert,
    total_num_tokens=total_num_tokens,
    topk=self.topk,
    num_experts=self.config.num_moe_experts,
    moe_aux_loss_coeff=aux_loss_coeff,
    fused=self.config.moe_router_aux_loss_fusion,
)
```

```python
# megatron/core/transformer/moe/router.py#L459-L472（按损失类型取系数）
def get_aux_loss_coeff(self, aux_loss_type: str) -> float:
    if isinstance(self.routing_type, str):
        if self.routing_type == aux_loss_type:
            return self.config.moe_aux_loss_coeff
    if isinstance(self.routing_type, list):
        try:
            idx = self.routing_type.index(aux_loss_type)
            return self.config.moe_aux_loss_coeff[idx]
        except ValueError:
            return 0.0
    return 0.0
```

上面四列图下面那行实测数据来自 Wang 等人——他们提出了无损失偏置，并在一个 1B、64 个路由专家、top-6 的 MoE 上与逐序列 aux loss 作比。在他们的基线系数 1e-3 下，验证集上的 MaxVio 是 0.72、困惑度 9.56；而用偏置是 0.04 与 9.50。完全不加 aux loss 时 MaxVio 升到约 5.1；系数 1e-2 时降到约 0.1，困惑度升到约 9.69。

这个系数 1e-3 的 aux loss，就是「负载不均的代价」那一节里的学出来的路由器，它跑完 120 个迭代爬到了 125 TFLOP/s。

系数设太高模型学得更差，设太低负载又不匀，所以 LM loss 和均衡度要一起看。而均衡度最好从 MaxVio 读，别从 aux loss 本身读：这项损失把每个专家占的 token 比例乘上它的平均路由概率，于是当路由概率是平的，不管 token 比例怎么分，这项损失读起来都和均匀负载一样——一个「完美均衡」的读数可能盖住一个歪掉的负载。Sigma-MoE-Tiny 就踩了这个坑：它下层把概率压平让这项损失降到最小，而负载一直不匀。`--moe-per-layer-logging` 会把这个值按层记成 `load_balancing_loss`，且已除过系数。

**什么时候用它**：图省事就用它——`aux_loss`，系数取 1e-3 左右，也就是 2.5 节我们第一次 MoE 实验那样。负载还歪就调高，LM loss 受影响就调低。

到目前为止用的 aux loss 数的是每个专家在一个 micro-batch 内的负载，于是留下一个还没问的问题：负载该在**哪些 token** 上是均匀的？可以按每条序列、按每个 micro-batch、或按整个全局批次均衡，下面就来比这三种。

### 范围

想一想一个 micro-batch 里装的是什么。预训练流水线经常把每个 micro-batch 装成单一领域，比如全是代码；那么一个「在这个 micro-batch 内要求负载均匀」的损失就会把代码 token 摊到所有专家上，于是没有专家能专门长成「代码专家」。改成在整个全局批次上均衡——那里混着很多领域——代码 token 就可以挤在少数几个专家上，只要其他领域的 token 把剩下的位置填满。Qiu 等人只在第二种设置下看到了专家按领域分化，而且困惑度也更低。GPU 为此付账：每个 micro-batch 内部负载不再均匀，他们那个 43B 模型的步慢了约 6%，其中约 1% 来自通信开销，其余主要来自不均衡。

一条序列是最紧的范围。逐序列均衡会把每条序列摊到所有专家上，哪怕这个 micro-batch 混着多个领域；DeepSeek-V2-Lite 把它唯一的那项均衡损失（专家级）就是这么算的，DeepSeek-V3 则保留一个小的，用来防止「整个批次看起来均衡、但单条序列全挤在几个专家上」。下面的图让三种范围并排跑，每块 GPU 各装两条来自两个不同领域的序列：范围越紧，专家越难集中在某一个领域；范围越松，每块 GPU 承担的负载越不匀。

> **图 3.3-e** 三种范围的对照：逐序列 / 逐 micro-batch / 全局批次（原站为交互图，此处从略）

那个模拟里，sequence-wise 损失把代码序列摊到所有专家上，micro-batch 损失让每块 GPU 的两个领域各分走一半专家，而 global-batch 损失把代码 token 基本留在了它们偏爱的两个专家上，代价是每块 GPU 的负载最不匀。Megatron 通过均衡类型来选范围：`aux_loss` 按 micro-batch，`seq_aux_loss` 按序列，`global_aux_loss` 按全局批次。要组合多种范围，就把几个类型和同样顺序的系数一起传，例如 `--moe-router-load-balancing-type aux_loss seq_aux_loss --moe-aux-loss-coeff 1e-3 1e-4`。

```python
# megatron/core/transformer/moe/router.py#L567-L589（按序列计数，再除以 bsz）
scores_for_aux_loss = scores_for_aux_loss.reshape(seq_length, -1)
routing_map = routing_map.reshape(seq_length, -1)

global_tokens_per_expert, local_num_tokens, total_num_tokens = (
    get_tokens_per_expert_and_token_count(
        routing_map=routing_map,
        reduce_group=self.tp_cp_group,
        with_padding_mask=with_padding_mask,
        topk=self.topk * bsz,
    )
)

aux_loss = (
    switch_load_balancing_loss_func(
        probs=scores_for_aux_loss,
        tokens_per_expert=global_tokens_per_expert,
        total_num_tokens=total_num_tokens,
        topk=self.topk,
        num_experts=self.config.num_moe_experts,
        moe_aux_loss_coeff=seq_aux_loss_coeff,
        fused=self.config.moe_router_aux_loss_fusion,
    )
    / bsz
)
```

```python
# megatron/core/transformer/moe/router.py#L617-L628（全局计数 + 滑动平均）
global_tokens_per_expert, local_num_tokens, total_num_tokens = (
    get_tokens_per_expert_and_token_count(
        routing_map=routing_map,
        reduce_group=self.tp_dp_cp_group,
        with_padding_mask=with_padding_mask,
        topk=self.topk,
    )
)

self.global_tokens_per_expert += global_tokens_per_expert
self.ga_steps += 1
averaged_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps
```

每种范围都有它自己的失败方式。按序列或按 micro-batch 均衡会妨碍「领域专家」长出来，模型学得更差；而按全局批次均衡会让每块 GPU 的负载不匀，步变慢。不管你选哪个范围，都在**那个范围**上读 MaxVio：逐 micro-batch 用 `--save-tokens-per-expert-interval` 存下的负载算，逐全局批次用那些负载的和算，逐序列则离线用路由追踪算（[A.1 节](#s-a-1)）。

**什么时候用它**：默认用 micro-batch（`aux_loss`）。只有当你想要「按领域分化的专家」时才用 `global_aux_loss`，代价是 Qiu 等人那里步时长多了 5.8%。

不管选哪个范围，均衡损失终究是通过梯度去指挥路由器，而这个梯度会和 LM loss 的梯度抢方向。有没有办法**完全不碰梯度**就把负载均衡掉？

### 无损失偏置

aux loss 是靠把路由概率从 LM 想要的位置拉开，来救一个过载的专家。无损失偏置不碰概率，用零梯度去搬负载。给每个专家一个偏置，在 top-k 之前加到它的分数上，且**只用来挑专家**；权重仍然来自分数。每步一次，Megatron 把每个偏置按固定量（更新速率）挪一下：欠载的往上、过载的往下。下一张图跑的就是这个循环。在 DeepSeek-V3 的消融里，它在验证损失上与批次级 aux loss 打平，1B 和 3B 都如此。和 aux loss 一样，它也会改掉一些「路由器本来会做的选择」，在上面 aux loss 那节的四列玩具里，它改得最多。

> **图 3.3-f** 16 个玩具专家的偏置更新循环（步长远大于 Megatron 默认值）（原站为交互图，此处从略）

图里跑 16 个玩具专家，步长比 Megatron 默认值大得多，于是几轮就能代替很多轮：过载的专家把偏置降下来、欠载的抬上去，直到负载接近均匀；把步长缩小十倍之后，大步长留下的来回震荡就被压住了。真实训练里的麻烦在「固定步长」这件事上，它必须匹配专家数量。Li 等人扫了小 MoE 的专家数：更新速率 1e-3 时，最忙专家的负载随专家数增长，到 1,024 个专家时约为均值的 50 倍；而单用 1e-2 的 aux loss 时保持在约 7 倍以下。超过 128 个专家后，偏置还伤 LM 质量，作者怀疑是步长对这种规模的专家来说太粗。

上面那张图里过载曲线上的参考线来自同一批 1B 实验，是在每个训练批次上量的、而不是在验证集上：aux loss 系数 1e-3 时 MaxVio 稳在 0.72 附近，用偏置时稳在 0.1 附近。

Megatron 的规则是你能写出的最简单那种：每一步，每个偏置都挪同一个固定量，不管这个专家离均值负载有多远。这叫 sign update（符号更新），但它不是跑这个循环的唯一方式。**线性更新**按专家离均值的距离成比例地挪，于是严重过载的专家得到大修正、快均衡的得到小修正；**AdamW 更新**则为每个专家自适应步长。LongCat-Flash 用的是线性更新，目标是他算力预算所对应的专家负载；ZAYA1-8B 把每个专家的负载误差喂给 AdamW。

这些规则还有一个更微妙的差别：当一次更新的步长加起来不等于零时，所有偏置的均值会漂移。这个漂移不改变任何选择（给每个专家的分数加同一个数，top-k 不变），但偏置若用 bf16 存就会损失精度；Megatron 用 fp32 存。Moonlight 和 Trinity Mini 靠减去步长的均值来避免漂移。发布出来的权重也是同样的差别：DeepSeek-V3 的偏置每层均值在 2.2 到 8.0 之间，而 Moonlight 和 ZAYA1-8B 的平均为 0，LongCat-Flash 的则略低于 0。下面把符号更新、居中符号更新、线性更新、AdamW 更新跑在同一个批次上。

> **图 3.3-g** 四种偏置更新规则在同一批次上的表现（原站为交互图，此处从略）

图里那个玩具批次有几个很热的专家、一堆恰好略低于均值的专家，所以大多数符号步朝上：符号更新和 AdamW 更新把偏置均值抬起来，而居中符号更新与线性更新把它按在零；但符号更新在每一步的均衡效果都和居中版一样，因为 top-k 不在意所有专家共同的那点平移。不管用哪条规则去搬，偏置只能均衡**它数到的那些 token**。批次级偏置是在一个批次上拟合、作用到下一个批次的，所以它能让整个批次看起来均匀，却不保证每条序列都均匀：单条序列仍可能挤在几个专家上。显而易见的补丁是再来一个「按整条序列算」的偏置，但那会用到这条序列里**后面**的 token，把未来泄漏到前面 token 的路由里。

Chang 的**因果偏置**靠「只用这条序列已经路由过的部分」来避免泄漏。它在每条序列开头重置，每处理一个 token，就把刚被选中的专家的偏置降一点、其余的抬一点，于是每个专家的偏置记录着「它到目前为止被选中的次数，离应得份额（每个 token k/N）高多少或低多少」，一条序列里被用超了的专家，在后面的 token 上就被压下去。和批次偏置一样，它只改变「挑哪些专家」，从不被梯度训练。下一张图用同一条序列，分别只用全局批次偏置、以及再加上这条序列自己的因果偏置。

> **图 3.3-h** 全局批次偏置 vs 加上因果偏置（原站为交互图，此处从略）

图里那条序列在中途切到代码。只靠全局批次偏置时，它的代码 token 挤到一个专家上——尽管整个批次仍然均衡；加上因果偏置后，这些代码 token 摊到更多专家上，这条序列的负载也接近均匀得多。另一面，模拟显示因果偏置把更多选择送给了「路由器自己 top-k 之外的专家」，偏离了路由器自己的偏好；而且支持它的证据规模很小，来自 Chang 在 nanochat 和 TinyStories 上的实验。

已发布的模型对「单条序列」有不同处理。DeepSeek-V3 把偏置和前面那个 sequence-wise aux loss 配在一起，系数小到 1e-4：偏置维持批次均衡，那个小损失防单条序列内部的极端不均衡。下一张图在一条普通序列和一条极端序列上比较「不加这个损失 / 小系数 / 大系数」。

> **图 3.3-i** 单条序列上的三种设置：不加 / 小系数 / 大系数（原站为交互图，此处从略）

那个玩具里两个系数只是示例、并非 DeepSeek-V3 的取值：小系数能把普通序列拉近均匀，同时只把路由器挪一点点，但极端序列仍然挤在一个专家上；要把那条摊开就得用大系数，而大系数会把路由器拽得离 LM 损失偏好的路由远得多。一条序列是「批次级计数会盖住不均衡」的一种情形，另一种是模态。

当一个批次混着多种模态，在全部 token 上计数的偏置可以把它们的**总和**均衡掉，而单独看某一模态的 token 仍然挤在几个专家上。DeepSeek-VL2 和 Kimi-VL 对所有 token 只用一个偏置，DeepSeek-V4.1-Flash 每种模态一个。下图比较「一个共享偏置」与「每种模态一个偏置」（各自只数自己模态的 token）。

> **图 3.3-j** 共享偏置 vs 每模态一个偏置（原站为交互图，此处从略）

那个玩具让文本偏向 8 个专家里的 4 个、图像偏向另外 4 个，于是共享偏置能把总量均衡掉，因为两个模态的超额互相抵消；而每模态一个偏置则各自均衡。拆开偏置的代价是：它逼每个专家在每种模态里都分到自己那份，从而妨碍了「专门做图像」这件事。

在 Megatron 里跑它：用 `--moe-router-enable-expert-bias` 打开偏置，用 `--moe-router-bias-update-rate` 设固定步长，并且要用 sigmoid 分数（`--moe-router-score-function sigmoid`），因为 Megatron 只允许偏置配 sigmoid 或 sqrt(softplus) 分数（[`transformer_config.py#L3139-L3147`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L3139-L3147)）。

```python
# megatron/core/transformer/moe/moe_utils.py#L936-L939（偏置只参与选择）
elif expert_bias is not None:
    scores_for_routing = scores + expert_bias.float()
    _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
scores = torch.gather(scores, dim=1, index=top_indices)
```

```python
# megatron/core/transformer/moe/router.py#L209-L228（int64 计数、fp32 偏置）
if self.enable_expert_bias:
    self.register_buffer(
        'local_tokens_per_expert',
        torch.zeros(
            self.config.num_moe_experts,
            # Keep token counts exact through accumulation and cross-rank reduction.
            # Float16Module leaves integer buffers in their original dtype.
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        ),
        persistent=False,
    )
    self.register_buffer(
        'expert_bias',
        torch.zeros(
            self.config.num_moe_experts,
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        ),
    )
```

```python
# megatron/core/transformer/moe/moe_utils.py#L1253-L1263（符号更新，整数精确比较）
torch.distributed.all_reduce(tokens_per_expert, group=tp_dp_cp_group)
num_experts = tokens_per_expert.shape[-1]
total_tokens = tokens_per_expert.sum(dim=-1, keepdim=True)
# Compare each tokens_per_expert value with the row average without converting the integer
# counts to floating point: tokens_per_expert < total / num_experts iff
# tokens_per_expert * num_experts < total.
update_direction = torch.sign(total_tokens - tokens_per_expert * num_experts)
updated_expert_bias = (
    expert_bias + update_direction.to(dtype=expert_bias.dtype) * expert_bias_update_rate
)
```

当更新速率对你的专家数来说太粗，负载就会一直不匀——按「负载不均的代价」那节，这会让步变慢；Li 等人的扫描还看到超过 128 个专家后 LM 质量受损。两个地方能抓到它：每个 checkpoint 都会存 `expert_bias`，你可以跟踪每层偏置在相邻 checkpoint 之间散开多宽；另外 `--save-tokens-per-expert-interval` 写出的 token 计数能给出逐全局批次的 MaxVio（[A.1 节](#s-a-1)）。

**什么时候用它**：当均衡器必须几乎不跟 LM loss 抢方向时。偏置完全不产生梯度，而 DeepSeek-V3 那个系数 α = 1e-4 的 sequence-wise 损失也只加了一点点。用 sigmoid 分数跑它，并按你的专家数调更新速率——超过 128 个专家时 1e-3 可能太粗。

偏置不管负载长什么样都只挪同样一小步，于是要追上一直在变的路由器可能得走很多步。与其一步步逼近正确的偏置，能不能直接从这个批次的分数把它算出来？

### 分位数均衡

分位数均衡保留偏置这件事，但是把它**解出来**而不是挪出来。对每个专家，它把这个 micro-batch 的 token 按「它们对该专家的分数」排序，然后读一个分位数：只有目标份额的 token 能越过的那个分数。偏置就从那个分位数定下来，而不是按固定步长挪。Jin 等人用一个 19B、768 个专家的 MoE 比较各种均衡器：分位数均衡比符号更新更快稳住负载，LM loss 差不多，把平均 MaxVio 砍掉一半——尽管最后 1k 步它的 MaxVio 略高一点。这个精度有代价：每个 micro-batch 里，每个专家都得把所有 token 按它的分数排一遍来求自己的分位数，也就是在「每个 token 一个 top-k」之外再加「每个专家一个 top-k」。下面把符号更新与分位数均衡并排跑在同一个批次上。

> **图 3.3-k** 符号更新 vs 分位数均衡（原站为交互图，此处从略）

那个玩具每一步都路由同一个批次，于是每个解出来的偏置作用的就是它自己解出来的那批 token：走到最后一步，符号更新仍然留着不均衡，而分位数均衡几步之内就接近均匀了。在 Megatron 里，每个 micro-batch 解自己的分位数，这些解在全局批次上取平均，用来路由下一个批次，可选再用指数滑动平均（EMA）平滑（`--moe-router-quantile-balancing-ema`）。EMA 会保留大部分旧偏置，只混进每个新解的一部分，于是偏置逐步缓慢变化。

```python
# megatron/core/transformer/moe/moe_utils.py#L233-L247（top-k 与逐专家分位数）
topk_result = (scores - beta).topk(k + 1, dim=1)
indices = topk_result.indices[:, :-1]

if not update_beta:
    return indices, beta

assert (num_tokens * k) % num_experts == 0, (
    "Quantile balancing requires the number of routed assignments "
    f"({num_tokens} tokens * top-{k}) to be divisible by "
    f"{num_experts} experts."
)
col_target = num_tokens * k // num_experts
alpha = topk_result.values[:, -1:]
beta_local = (scores - alpha).topk(col_target + 1, dim=0).values[-1].contiguous()
return indices, beta_local
```

```python
# megatron/core/distributed/finalize_model_grads.py#L441-L457（平均、EMA、再居中）
stacked_beta = torch.stack(qb_beta_list, dim=0)
local_avg_list = [
    accum / count.clamp(min=1).to(accum.dtype)
    for accum, count in zip(qb_beta_accum_list, qb_beta_count_list)
]
stacked_local_avg = torch.stack(local_avg_list, dim=0)

torch.distributed.all_reduce(
    stacked_local_avg, op=torch.distributed.ReduceOp.AVG, group=dp_cp_group
)

ema = config.moe_router_quantile_balancing_ema
stacked_new_beta = ema * stacked_beta + (1.0 - ema) * stacked_local_avg
stacked_new_beta = stacked_new_beta - stacked_new_beta.mean(dim=-1, keepdim=True)
```

Kimi K3 在整个全局批次上跑分位数均衡，每个专家的分位数从「跨 GPU 求和后的直方图」里读。它之所以放弃固定步长规则，是因为每层的路由池到了 896 个专家之后均衡越来越难；而提出分位数均衡的 Su Jianlin 写道，Kimi K2 当时用的还是符号更新。

取平均有代价：一个由整个全局批次共享的偏置，能把那个批次均衡掉，而每块 GPU 的 micro-batch 仍然不匀。Neitemeier 等人在一个 7.5B 实验（一个 seed）里量了这个差距有多大。Megatron 用的「按 rank 平均的分位数」把全局批次均衡到 MaxVio 0.92，而在最差的那层里，某个专家并行 micro-batch 上最忙的专家仍然拿到了公平份额的约 7 倍，也就是局部 MaxVio 5.96。直接解出精确的全局批次分位数的做法把两个都降下来了。他们把局部 MaxVio 当作调度成本的代理指标，而不是实测吞吐。

这些均衡器在吞吐上怎么比？我们从初始化开始、同一模型、一个 seed，跑五种路由配置，比它们最后 20 个（共 120 个）迭代的中位吞吐。分位数均衡到 139 TFLOP/s，aux loss 是 125，被测的那套「只有偏置」的配置是 99。只有偏置那套用 sigmoid 分数、更新速率 1e-3，其余四种用 softmax。（这些运行用的是较老的 Megatron 提交，它把偏置的 token 计数累加在 bf16 里，计数可能被圆整，从而翻转偏置更新的方向；在 `d6316eb45` 里计数是精确的。）把 EMA 设成 0.9，分位数均衡到 121 TFLOP/s。那个设置让每个新解只占偏置更新权重的 10%（[`transformer_config.py#L910-L913`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/transformer_config.py#L910-L913)）。

Jin 等人给分位数均衡加 EMA：平滑越强，偏置在 checkpoint 之间越稳，但训练早期纠正负载也越慢。五次运行都画在下面。

> **图 3.3-l** 五种路由配置的中位吞吐（120 个迭代，取最后 20 个）（原站为交互图，此处从略）

图里五次运行中，只有偏置那套是平的，其余四条都在往上爬，最后 20 个迭代里分位数均衡最快。结束时吞吐还在涨，所以这个排序只描述了最初这 120 个迭代。分位数均衡同时还以五次里最高的训练 LM loss 收尾，所以光凭吞吐领先，并不能说明它能更快地训到同样质量。

要试分位数均衡，设 `--moe-router-load-balancing-type quantile_balancing`。同一个开关还接受 `sinkhorn`：Sinkhorn 在训练时用一种均衡指派的过程为每个 token 选专家。

风险是「全局批次看起来均衡、而每块 GPU 的 micro-batch 不均衡」，Neitemeier 等人把它读作更高的调度成本。要看这个差距，就用 `--save-tokens-per-expert-interval` 写出的 token 计数把 MaxVio 算两遍：一遍逐 micro-batch、一遍逐全局批次，然后对比（[A.1 节](#s-a-1)）。

**什么时候用它**：当符号更新式的偏置均衡得太慢时。在 768 个专家下，解出偏置比一步步逼近得到更低的平均 MaxVio，LM loss 差不多，而且在五次配置对比里拿到最高中位吞吐（一个 seed，最后 20/120 个迭代）。同时要把它的限制算进去：Megatron 目前不接受它和任何 aux loss、`--moe-router-fusion` 或分组受限路由同时使用。

### 该盯什么

| 部位 | 症状 | 该盯的信号 |
|---|---|---|
| 负载不均的代价 | 「慢」：装着最忙专家的那块 GPU 决定了步长 | 相对强制均摊对照跑的 TFLOP/s |
| 辅助损失 | 「学得差」：系数太高模型学得更差，太低负载又不匀 | LM loss 与 MaxVio 一起看：路由概率被压平时，aux loss 看起来均衡而负载其实还歪着 |
| 范围 | 「学得差」：按序列或 micro-batch 均衡妨碍领域专家形成。「慢」：按全局批次均衡让每块 GPU 负载不匀，步长了 5.8%，主要来自不均衡、其中约 1% 来自通信 | 在你均衡的那个范围上读 MaxVio：逐 micro-batch 与逐全局批次用 token 计数，逐序列用路由追踪（A.1 节） |
| 无损失偏置 | 「慢」：固定步长 1e-3 在 128 个专家时最忙专家仍拿到约 20 倍平均负载。「学得差」：超过 128 个专家该步长太粗 | `expert_bias` 每层在相邻 checkpoint 之间散开多宽，以及逐全局批次的 MaxVio（token 计数，A.1 节） |
| 分位数 | 「慢」：用滑动平均平滑偏置会让它滞后于路由器分数。偏置由全局批次共享，每块 GPU 的负载仍可能不匀（见「范围」） | 同时看逐 micro-batch 与逐全局批次的 MaxVio（用保存的 token 计数，A.1 节），以发现「全局均衡掩住局部不匀」 |

---

**导航**：[全书目录](index.md) · [← 第 3 章（上）监控与路由](03-1-monitoring-and-routing.md) · [第 3 章（中）专家 →](03-3-experts.md)
