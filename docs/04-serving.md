# 第 4 章 从训练到推理服务

模型只训一次，却要在被别人用的整段时间里持续提供服务。到这为止，这本手册讲的是前半截：怎么把一个 MoE 训得快、训得好。这一章讲后半截。

推理服务用另一套数来评判模型。训练看每步耗时和 loss；服务看每次请求多快拿到答案、以及回答完这些请求要占多少张卡。模型是同一个，目标却换了，于是那些在训练时正确的选择，等训练结束就值得再看一眼。

MoE 需要重看的东西比稠密模型多。每个 token 只过少数几个专家，但每个专家通常都得留在显存里，等着某个 token 选中它——于是这个模型算力上便宜、显存上昂贵。一旦进入逐 token 解码，每个专家每步也只见到几个 token，于是「把专家摊到多少张卡上」本身就成了一个服务侧决策。注意力和专家（在目前这种布局里共用同一张卡）对硬件的需求也不一样。

本章逐条处理这些差异。先看 k，也就是路由器为每个 token 选的专家个数。接着看专家占多少显存、摊到多少张卡上、以及注意力和专家该分别在哪儿跑。最后是 RL 后训练：服务引擎写出训练器要学的样本，而两边可能对「每个 token 走了哪些专家」这件事各执一词。

第一个改动，是你在服务时设的一个数字。如果你用与训练时不同的专家个数来服务每个 token，它的输出会怎样？

### 用不同的 k 服务

训练定死了每个 token 用几个专家，但没人拦着你少用。从某个 token 的 top-k 里每砍掉一个被路由到的专家，就省一份算力；而每个专家仍然常驻显存，等着选中它的那些 token——所以显存一点没省。难处在于路由权重。做重归一化的路由器（3.1 节）会拿被选中专家的概率除以它们的和，于是专家分支输出的量级（增益）是按训练时的 k 定下来的。当你少服务几个专家时，幸存者要瓜分同一个总量，每个分到的权重都比训练时更大。所以更小的 k 同时改了两件事：哪些专家在跑，以及每个专家贡献多少。

Chen 与 Yao 用一个整数把这两个效应拆开。他们保留 top-k1 的专家，但把权重除以 top-k2 的 softmax 质量，k2 在服务时指定、不需要训练。让 k2 = k1，就是对幸存者做常规重归一化，LExI（一个为每层挑不同 k 的方法）也是这么做的；让 k2 等于专家总数，等于关掉重归一化；让它等于训练时的 k，则把增益锚回模型训练时见过的那个值。

下面这张图把一个模拟 token 送进一个形状模仿 Qwen3.6-35B-A3B 路由器的 softmax——256 个被路由专家、按 top-8 训练——然后用更小的 k 服务它，每种分母一行。跟着中间那栏看 k 下降时的变化：除以幸存者，每个被服务的专家权重比 top-8 时更大；除以训练时的 top 8，它保持训练时的权重；除以 top 16，它变得更小。右栏给出每一行实测的 MMLU 变化。

那个路由器很「平」：在 WikiText 上它的 top 8 专家平均只占 softmax 质量的 0.182，所以分母一动，增益就动得很远。以训练时 top-8 的 81.65% MMLU 为基准，top-4 除以四个幸存者的得分低 4.65 分，除以 top-8 质量低 3.10 分；除以 top-16 质量只低 0.35 分，这个降幅不显著（p = 0.66）。完全不重归一化的话，top-6 掉 27.45 分。作为回报，top-4 把每 token 激活的被路由参数从 1.006B 砍到 0.503B，相当于省掉每 token 全部参数的 17%。

分母有多要紧，取决于路由器有多平。Gemma-4-26B-A4B 的 128 个专家里 top 8 占了 0.302 的质量，在 top-6 上去掉重归一化，困惑度只涨 0.7%，而 Qwen 那个模型上是 356%。两个结果都来自 softmax 路由器。

服务时用比训练更多的专家，每 token 算力更贵，而质量未必跟着涨。Gu 等人在 OLMoE-1B-7B（按 top-8 训练）上做微调实验，服务 top-16 让九项 benchmark 的平均从 41.35 变成 41.09，落在他们报告的误差棒内。多出来的专家组合是模型从没一起训过的，所以他们的实验里，只在服务时调大 k 并没有帮上忙。

[Megatron](https://github.com/NVIDIA/Megatron-LM) 就按 `--moe-router-topk` 服务，权重跟着你训出来的路由器走。默认情况下 softmax 路由器在 top-k logits 上做 softmax，于是更小的 k 就是在幸存者上重归一化，即 Chen 与 Yao 的 k2 = k1。加 `--moe-router-pre-softmax`，权重来自完整 softmax，永不重归一化。sigmoid 路由器在 k > 1 时除以自己的 top-k 和，任何路由器都会乘上 `--moe-router-topk-scaling-factor`（如果设了）。transformers 的 Qwen3-MoE 里，服务的 k 是 `num_experts_per_tok`，只有设了 `norm_topk_prob` 才把路由权重除以 top-k 和。想要别的分母，两种路由器都只需要打几行补丁：保留 top k1 的专家，除以 top-k2 的质量。

```python
# megatron/core/transformer/moe/moe_utils.py#L913-L947
if score_function == "softmax":
    if use_pre_softmax:
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        ...
    else:
        scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
elif score_function in ("sigmoid", "sqrtsoftplus"):
    ...
    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
if scaling_factor:
    probs = probs * scaling_factor
```

> **提案**：几行补丁——保留 top k1 的专家、除以 top-k2 的质量，打在 Megatron 里或者 transformers 的路由器里；然后再做下面的服务端 k 扫描。

除了困惑度，还要在有标注的任务上给每个服务的 k 打分，因为两者可能打架：35B 模型在 top-6 时，以训练时的 top-8 质量为分母困惑度最低，而 MMLU 最高的却是以 top-16 质量为分母。

**什么时候用它**：当你需要降低每 token 的被路由专家算力、又能把所有专家留在显存里时——服务一个更小的 k，并把幸存者的权重除以一个比它们自身更大的质量。在 Qwen3.6-35B-A3B 上取 top-4、除以 top-16 质量，MMLU 与训练模型的差距保持在 0.35 分内。分母一定要在你自己模型上验证，因为它有多要紧取决于你的路由器有多平。

砍 k 省的是算力，每个专家仍然占着显存。当限制你服务的是显存而不是算力时，怎么让专家少占点？

### 低位宽专家

训练一结束，权重就不动了，描述它们在服务时成本的就两个数：在显存里占多少字节，以及每解码一个 token 要读多少字节。MoE 里被路由的专家握着大部分参数，所以它们的精度决定了模型的大半体积；只把它们存成 4 bit、其余保持更宽的格式，就能拿到大部分的节省。

有些前沿 MoE 已经是这么发的。OpenAI 用 MXFP4 量化 gpt-oss 的 MoE 权重做后训练，折合每参数 4.25 bit。这些权重要占全部参数的 90% 以上，于是 120b 的模型能塞进一张 80 GB 卡。Moonshot 在 SFT 与 RL 阶段把 Kimi K3 的被路由专家量化成 MXFP4，其余保持高精度。DeepSeek-V4 在后训练里对 MoE 专家权重做 MXFP4 量化感知训练（前向时把权重圆整到目标格式），GLM 团队则在 GLM-5 的 SFT 阶段用 INT4 量化感知训练。这四家的量化都发生在预训练之后，所以你预训练的配置不用动。

下面这张图比较 gpt-oss-20b 的三种存法，以逐 token 解码为前提：全部 bf16、只有被路由专家 4 bit、以及所有线性层 4 bit。每根柱是模型在显存里的大小，蓝色块是一个 token 读到的权重。对比第一行到第二行，看模型缩小了多少、而一个 token 的读取缩小了多少；再看第三行把注意力也放进 4 bit 后多出了什么。

把专家存成 4 bit，gpt-oss-20b 从 39.0 GiB 缩到 12.8 GiB，小三倍；但每解码一个 token 读到的权重字节只从 6.72 GiB 降到 3.45 GiB，大约一半。batch-1 解码时，一个 token 每层只读 32 个专家里的 4 个，却要读全部注意力和 unembedding，所以专家一小，那些 bf16 权重就占了它读取量的三分之二。把注意力也放进 4 bit，读取再降四分之一，而显存只多省 0.9 GiB——因为注意力本来就没几个参数。

公开的 checkpoint 两种范围都有。ZAYA1-8B 的 MXFP4 版本只量化被路由专家，Kimi K2 Thinking 和 Qwen 的 GPTQ-Int4 版 Qwen3.5 也是（用 int4 而非 MXFP4）。Qwen 的 GPTQ-Int4 版 Qwen3 30B MoE 与腾讯的 Hunyuan-A13B 把注意力也放进 int4，只留路由器、embedding 和输出头不量化。

Megatron 通过 ModelOpt 导出：`examples/post_training/modelopt/quantize.py` 用 `--export-quant-cfg` 接一个 ModelOpt 配置名，比如 `FP8_DEFAULT_CFG` 做 FP8 导出，选项里还有 `--weight-only` 和 `--sync-expert-weight-amax`。要在 4 bit 下训练，设 `--fp4-format e2m1`，在 Blackwell 上走 [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 的 NVFP4，需要 Transformer Engine 2.7 以上，且不能与 `--fp8-format` 同用。

```python
# megatron/post_training/arguments.py#L56-L61（--export-quant-cfg）
group.add_argument("--export-quant-cfg", type=str, default=None,
    help="Specify a quantization config from mtq.config.choices.")
# examples/post_training/modelopt/quantize.py#L159-L176（--weight-only / --sync-expert-weight-amax）
group.add_argument("--weight-only", action="store_true", help="Disable input quantization.")
group.add_argument("--sync-expert-weight-amax", action="store_true",
    help="Synchronize expert weight amax across experts.")
# megatron/core/transformer/transformer_config.py#L735-L743（--fp4-format）
fp4: Optional[Literal['e2m1']] = field(default=None, metadata={"argparse_meta": {"arg_names": ["--fp4-format"]}})
```

把每个专家的量化误差对着它的 token 数画出来（token 数用 3.0 节的逐专家统计），这样「因为很少被路由而变差」的专家会自己跳出来。然后拿量化导出的模型和 Megatron 在同一段文本上比 loss。

**什么时候用它**：当服务受限于显存、或者受限于每 token 要读的字节数时——在后训练里把被路由专家量化到 4 bit，其余保持高精度，像 Kimi K3 那样。

把专家压到更低位宽，缩的是每张卡装的东西，但没回答该把它们摊到多少张卡上。一旦请求是一个 token 一个 token 地来，还该按训练时那样铺开吗？

### 服务端的专家并行

训练只为一种步挑专家并行度：一大批 token 一起过。服务跑两种步。prefill 步处理 prompt，DeepSeek 的引擎允许一步里每卡最多 32,768 个 token。decode 步只承载在飞请求的下一个（或几个）token，于是每个专家每步只见到很少的 token。把专家摊到更多卡上对后者有帮助：每卡持有更少的专家、每步读更少的权重，而 All-to-All 给每个专家攒出更大的批次。

DeepSeek 在服务 DeepSeek-V4.1-Flash 时，把两个阶段往相反的方向配。prefill 跑 EP2 到 EP8，也就是显存够用的最小规模；decode 跑 EP64 到 EP128，也就是手里显卡允许的最大规模。两个阶段里注意力都做[数据并行](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism)，每个 EP rank 一个副本，且都不用张量并行或[上下文并行](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism)。

下面这张图重画了 DeepSeek 自己对 decode 两种设置的比较，每个点对应一种「每卡请求数」。每卡 64 个请求时，EP64 每卡持 7 个被路由专家，每卡每秒约 15,700 token、每用户约 246 token/s；EP8 每卡持 52 个专家，只有约 12,600 与 198，所以 EP64 下两个数都高约 25%。每卡 256 个请求时，两种设置几乎打平，约 26,300 与 24,900 token/s/卡。

宽度是有代价的。DeepSeek 给 prefill 与 decode 的卡数大致相当，所以 decode 开到 EP64 以上时，最小可跑部署就是 64 张卡做 prefill、加 64 张做 decode。同一个 EP 实例的所有卡还构成一个通信域，一张卡挂了整个实例就挂，重启时 DeepSeek 会按还健康的卡重新调整 EP 组的大小。

Megatron 在 coordinator 模式下用[专家并行](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism)服务（这是默认模式），direct 模式拒绝大于 1 的 EP。专家并行下，只有所有 EP rank 都在 decode 时，一步才会走 CUDA graph：只要有一个 rank 的 batch 里有 prefill，所有 rank 都退回 eager 模式。做 RL 时，`--rl-inference-expert-model-parallel-size` 给 rollout 引擎单独的 EP 大小，它被建成一个独立的推理模型，训练器的权重再 refit 进去。

```python
# docs/mcore-inference-user-guide.md#L277-L278（EP 需要 coordinator 模式）
- *Required* for: HTTP serving (`serve`), expert parallelism (`EP > 1`),
  multimodal inputs, streaming, and the lifecycle controls
# megatron/core/inference/batch_dimensions_utils.py#L156-L161（一个 prefill rank 把整个 EP 组拖回 eager）
"""Adjust CUDA graph batch dimensions for expert parallelism.
All-reduce-max the token count and non-decode flag across the EP group.
If any rank has a prefill (non-decode) step, all ranks fall back to eager mode.
"""
# megatron/training/config/rl_config.py#L140-L141（rollout 的 EP 大小）
rl_inference_expert_model_parallel_size: int | None = None
```

**什么时候用它**：当你要的是 decode 速度、也出得起宽部署的卡时——decode 用你的卡允许的最宽专家并行，prefill 用塞得下的最窄规模，像 DeepSeek 对 V4.1-Flash 那样；再给宽 decode 组配冗余专家，让它最慢的那个 rank 贴近均值。

更宽的专家组改变了每张卡干多少专家活，但每张卡仍然把注意力和专家并排跑，哪怕两者对它的要求并不一样。到了推理阶段，共用显卡还是你该选的摆法吗？

### 注意力与前馈解耦

在共置（colocated）布局里，专家并行组里的每张卡都要跑每层的两半——注意力和专家——但解码时这两半要的资源不同。注意力参数很少，却得为每个在飞请求保留 KV cache；专家握着大部分参数，却不保存任何按请求的状态。于是服务可以把注意力和专家放到不同的卡上，各成一个池，用小批次来回传递，让一个池在算的时候另一个池在算上一批。（Megatron 的 `megatron/core/inference/disaggregation` 沿的是另一条线：把 KV 与 SSM 状态从 prefill 引擎交给 decode 引擎。）MegaScale-Infer 就是这样把每层里的注意力与前馈拆开的，它报告每卡吞吐最高达到此前服务系统的 1.90 倍。

下面这张图左边画 FastAFD 部署 Qwen3-235B-A22B 的方式（28 张注意力卡 + 4 张专家卡），右边画 4 张卡上的共置专家并行。看左边对比两种布局下每张卡装了什么：没有专家的注意力卡能放下更多请求的 cache，而专家卡不保存任何请求的 cache，每卡请求数的柱状把两个池都算进去了。看右边跟踪一个 decode 步，看第二个 micro-batch 如何在第一批去专家那里时把注意力池占住。共置那一行也用两个 micro-batch 掩盖它的 All-to-All，vLLM 把这种做法叫 dual-batch overlap。

这笔收益有多大，取决于你拿什么比。按论文发表时的口径，FastAFD 的布局每卡吞吐是共置 EP4 的 1.41 倍。美团的 Qiu 等人用解析模型复现了这个增益，然后在同一个模型里按同样的 72 卡预算重新给两种布局配了卡，比值掉到 0.99 倍。在他们建模的所有设置里，同构 AFD 只在少数情形下提高了固定预算的吞吐。Step-3 的头条数字——同样 20 token/s 的解码目标下每卡每秒 4,039 token，对 DeepSeek-V3 的 2,324——比较的是两个不同的模型，因此并不能把布局这个因素单独隔出来。

训练在这里决定一件事：专家池要服务的模型有多稀疏。Step-3 把稀疏度算成「激活专家数（含共享）/ 被路由专家数」，并推导出专家池在「仍是算力受限、且通信能与计算重叠」的前提下所能服务的最稀疏 MoE。这个下界随模型的宽度和深度上升；在 DeepSeek-V3 的宽度与深度、H800 上它是 0.058，而 Step-3 大约按 0.08 训练。更稀疏的模型就得退回大规模专家并行，或者用 3.7 节那种分组受限路由之类的限制。

**什么时候用它**：当你的推理引擎能把注意力和专家跑成两个独立显卡池（像 MegaScale-Infer 和 Step-3 的服务系统那样），并且这种布局能打过一个按同样卡数重新配过的共置专家并行布局——因为正是这种比较，让论文里的增益在 Qiu 等人的模型里缩到了打平。如果你打算按这种方式服务，在定专家数和 top-k 之前，先按你的宽度、深度和硬件算出 Step-3 那个下界，并让激活比例高于它。

注意力和专家跑在哪儿是服务侧的决策，但在 RL 后训练里，服务引擎还负责写出训练器要学的 rollout。那么，你能指望训练器把每个 token 送到和 rollout 引擎一样的专家上吗？

### RL 里的路由

RL 后训练把样本生成和梯度计算分开。rollout 引擎用它自己的 kernel 生成样本，在 off-policy 或异步 RL 下，用的还是比训练器落后一次或更多更新的权重；而训练器要把每个 token 重算一遍来求梯度。这些差异会挪动 token 概率，但在 MoE 里，它们还会改变一个 token 走了哪些专家——于是训练器求的是一条样本根本没走过的路径。

每个成因都被单独测过。Ma 等人用 [SGLang](https://github.com/sgl-project/sglang) 在 Qwen3-30B-A3B 上生成回答，再喂给 Megatron：约 10% 的路由器选了不同的专家，94% 的 token 至少在一层上不一样。GSPO 的作者 Zheng 等人在 Qwen3-30B-A3B-Base 上测了另一个成因：一次梯度更新后，同一条 rollout 里约 10% 被激活的专家变了。两个 10% 数的不是一回事：一个是跨引擎不一致的路由器，一个是一次更新里会变的激活专家。

Ma 等人的修法是 R3：把 rollout 选中的专家记下来，在训练器里重放。这抹掉了路由那一份差距，但抹不掉其余的：训练-推理 KL 从 1.5e-3 降到 7.5e-4，接近稠密 Qwen3-8B 的 6.4e-4。在 Megatron 里，`--moe-enable-routing-replay` 让推理引擎记录每个请求的 top-k；你的 RL 框架随后在 `REPLAY_FORWARD` 前用 `set_replay_data` 把这些索引交回训练器。（第三个动作 `REPLAY_BACKWARD`，是在[激活重算](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=activation_recomputation)于反向传播中重做前向时，再把索引重放一遍。）记录时要把引擎的异步调度关掉，因为引擎不接受两者同时开。重放并不冻结路由器：在 `REPLAY_FORWARD` 里它会在被重放的索引处取自己当前的分数，于是梯度流过的是 rollout 选中的专家，而不是它自己会选的专家。

```python
# megatron/core/inference/inference_request.py#L786-L787（每个请求的 rollout 索引）
# routing_indices is reconstructed from per-block storage when a request finishes.
routing_indices: Optional[np.ndarray] = None
# docs/api-guide/router_replay.md#L70-L80（先 set_replay_data，再 REPLAY_FORWARD）
1. Enable & Instantiate: one `RouterReplay` instance per MoE router layer.
2. Record: `RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)`.
3. Forward Replay: `set_replay_data(list_of_tensors)` then `REPLAY_FORWARD`.
4. Backward Replay: `REPLAY_BACKWARD` under recomputation.
# megatron/core/transformer/moe/router_replay.py#L13-L15（三种动作）
RECORD = "record"
REPLAY_FORWARD = "replay_forward"
REPLAY_BACKWARD = "replay_backward"
```

公开的 RL 配方在这件事上两边都有。DeepSeek-V3.2、MiMo-V2-Flash 和 Instella-MoE 在训练器里重放 rollout 的路由；Kimi K2.5 与 Mellum 2 则没有重放，而是在 loss 里处理这个不一致——把概率比漂得太远的那些 token 的梯度屏蔽掉。

重放到底能不能换成 reward，仍然有争议。在 Ma 等人于 Qwen3-30B-A3B 上的比较里，不带重放的 GRPO 在三种设置里全崩，带截断重要性采样的 GRPO 在两个里崩了一个，而带重放的没有一次崩，他们报告训练稳定性和性能都有显著提升。后来 Zheng 等人的一项 Qwen 研究在 on-policy 训练里没看到 R3 的收益；在 He 等人的实验里，重放在 Moonlight-16B-A3B 上不如朴素 GRPO，不过在 Qwen 的 30B-A3B 上不是。开销在 rollout 侧很小，存路由带来的额外延迟不到 3%。而用过重放的 GSPO 作者列出了额外显存与通信、以及可能限制模型容量的问题，最终放弃它，改用序列级重要性比。

TRL 不用重放来处理同一个不一致。在它的 `GRPOConfig` 里，`importance_sampling_level="sequence"` 给出 GSPO 的序列级比值；默认开启的 `vllm_importance_sampling_correction` 会在训练器与 [vLLM](https://github.com/vllm-project/vllm) 的概率之间施加重要性采样比，并把比值落到设定区间外的序列屏蔽掉。

重放剩下的那一部分，是 kernel 自身贡献的差距，Megatron 用 `--batch-invariant-mode` 去治它：文档把 batch-invariant kernel 说成训练与推理 log 概率不一致的解法，称其为「RL 中真实存在的误差与不稳定来源」。它需要 flash attention 3 或 4，MoE 训练还需要 `alltoall` 分发器，Megatron 也警告说它「会显著影响速度」。在我们那个 64 专家 MoE 上（较早的 Megatron commit），开着它跑两次，12 步里打印的 loss 与梯度范数完全一致，而它带来的每步耗时成本分辨不出来，混在 run 与 run 之间的波动里。至于 rollout 权重比训练器落后一次更新，那是另一个成因，batch-invariant 模式管不着。

DeepSeek 的生产服务引擎 DeepInfer 提供同样的选项：设 `DEEPINFER_BATCH_INVARIANT=1` 时，模型结果在不同 batch 组成下保持一致；再设 `DEEPINFER_ENABLE_MTP_VARLEN=0` 并给每个请求固定种子，采样就是确定性的——仅限 NVIDIA 显卡。

StepFun 给 Step 5 Preview 做的 RL 两种补救都用。rollout 用当前权重时，它把训练与推理的算子对齐到 log 概率逐位相同，开销不到 10%。rollout 陈旧时，它重放被路由的专家——索引与系数都重放，路由器冻结——actor 与参考策略都这么做。这比 Megatron 的重放走得更远，后者会在被重放的索引处用路由器当前的分数重算权重。

NVIDIA 给 Nemotron 3 Super 与 Ultra 的 NeMo-RL 配方，在「rollout 最多落后一次更新」的异步 RL 里冻结路由矩阵，但仍然更新 Megatron 加到分数上、只用于选专家的那个均衡偏置。那个偏置每步都动，所以选择在冻结矩阵下仍会漂——图里那行「冻结」并没有体现这种漂移。

量化 rollout 的激活会放大同一个差距。Qwen 的 Zhuge 等人跑 NVFP4 rollout（权重和激活都是 4 bit），对上一个 BF16 训练器，训练在约 150 步后崩了。他们比较了「只量化权重」与「只量化激活」两组 rollout，把崩溃归到量化后的激活上，而不是权重。

会咬人的不一致，表现为一次学得很差或者直接崩掉的 RL 实验。信号是训练-推理概率差，Megatron 的 RL 循环往 W&B 记两个指标：`mean_inf_train_prob_abs_diff`（训练器与 rollout 概率的平均绝对差）和 `max_piold_to_inf_prob`（训练器比 rollout 概率的最大比值）。想单独看路由那一份，就把 Megatron 推理引擎记下的 `routing_indices` 和训练器自己的 top-k 离线比一比。

**什么时候用它**：当一次 MoE 的 RL 里，训练器与 rollout 引擎在路由上不一致时——不管是因为两个不同的引擎，还是在 off-policy / 异步 RL 下 rollout 权重落后于训练器。重放能消掉这个不一致，但它换来的 reward 收益有争议：Ma 等人报告有增益，而 Qwen 那项 on-policy 研究没看到。先量一下有多少路由器不一致，再比一次带重放和不带重放的学习曲线，然后决定留不留。如果你还要量化 rollout，先只量化权重试试——在 Zhuge 等人的实验里，是量化后的激活导致了崩溃。

到这里，你已经把一个 MoE 从第一次稠密基线带过了服务、又带进了一个 RL 循环。这一路上的选择里，哪些可以当作已经定了，哪些还开着？

---

**导航**：[全书目录](index.md) · [← 第 3 章（下）系统工程](03-5-systems.md) · [第 5 章 结语与推荐参数 →](05-conclusion.md)
