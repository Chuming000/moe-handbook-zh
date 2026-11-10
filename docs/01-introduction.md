# 第 1 章 引言

> 原文：*The MoE Handbook*（Hugging Face），<https://huggingface.co/spaces/huggingface/moe-blogpost>
> 阅读时间：1–2 天。为了最好的阅读体验，官方建议不要用手机看。

前沿模型是稀疏的。

在 [Artificial Analysis 榜单](https://artificialanalysis.ai/models/open-source)上，得分最高的 20 个开源权重模型里有 18 个是混合专家（MoE）。剩下两个不是——Nanbeige4.3-33B 和 Qwen3.8 27B，排在第十二、十三名，都是稠密模型。而这 18 个里，包含了五家领先实验室各自的最强模型：DeepSeek V4.1 Flash 1017、GLM-5.3、MiMo-V2.6-Pro、Ling 3.1 Flash 1019、MiniMax-M3.1-Flash。

> **图 1.1** 稠密与稀疏模型的发布时间与得分分布（原站为交互图，此处从略）

MoE 的好处，本质上在于它**把推理算力花在了哪里**。每个 token 只经过少数几个专家，于是模型可以携带比「单 token 实际用到」多得多的参数：你在第 2 章要动手训练的那个小 MoE 有 68.6 亿参数，其中每个 token 只激活 12.2 亿。**难的地方在于，怎么把这笔推理上的节省，换算成训练上的加速。**

我们上一本讲训练的深度长文 [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) 覆盖了稠密模型的训练，但只是顺带提了 MoE 的设计。因为那本书的主角 SmolLM3 是个稠密 Transformer，它的训练压根用不到 MoE 专属的技术，比如[专家并行](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism)。即便如此，我们还是在书中「Going Sparse: MoE」一节里扫了一遍高层设计选择：稀疏度、粒度、共享专家、负载均衡。那一节点出了分布式 MoE 训练为什么复杂——专家放置、负载均衡、All-to-All 通信——但没有往细节里钻。这本手册就从那张清单开始。

从稠密模型走到 MoE，你会多出三件必须操心的事：

1. 一个**路由器**，决定每个 token 走哪些专家；
2. 一份必须始终**均衡**地压在各专家上的负载；
3. 一旦专家被拆到不同 GPU 上，每一层 MoE 都要在这些 GPU 之间[交换 token](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/core/transformer/moe/token_dispatcher.py#L708-L727)。

每一件都会带来稠密训练从未有过的配置项，以及全新的失败方式。接下来的章节会把这些配置一个一个拆开，配上能跑起来的代码，而且大多数情况下还给出我们自己集群上的实测结果与数据。

一次跑崩的训练只会告诉你「症状」，不会告诉你「原因」。下面这张图是一张高层地图，帮你顺着症状回到源头：先看它是怎么崩的，再找异常行为的第一个信号（矿井里的金丝雀），然后去读正文里解释成因的那一节。留着手边，等哪天真的跑炸了再用。

> **图 1.2** 故障排查地图：从症状 → 首个信号 → 对应章节（原站为交互图，此处从略）

> **图 1.3** 全书路线图（原站为交互图，此处从略）

最后说一句读者定位。这本手册写给谁？如果你已经在 GPU 集群上训过稠密模型，下一个想训 MoE，那你就是目标读者。第 2 章会把需要的东西铺好，带你在单节点上把那个小 MoE 训出来。如果你更关心「MoE 到底怎么工作、前沿模型怎么选型」，而不是亲手跑一个，可以直接跳到第 3 章「MoE 训练的解剖」。

等专家被放到不同 GPU 上，每一层 MoE 都要在它们之间倒腾 token——这是稠密训练里从未出现过的通信模式。在你改模型的第一行代码之前，你真的信得过集群的互联吗？

---

**导航**：[全书目录](index.md) · [第 2 章 环境与第一次跑通 →](02-setup.md)
