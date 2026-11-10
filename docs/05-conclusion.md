# 第 5 章 结语

我们从一次稠密实验出发，把 MoE 层一个设置一个设置地拆开，然后跟着训练好的模型走进推理服务和 RL 后训练。现在你可以先验一遍集群再去信它，用同一批数据和同一个启动脚本训出稠密基线与 MoE，并且让两次实验可复现到能相互比较的程度。第 3 章给每一层里的每个部件——从路由器的分数到 All-to-All——都配了一个该盯的信号和一条建议。附录 A.2 反向走一遍：从面板上的一个症状，倒推回背后的那个配置。

下面这段把各章推荐的参数集中起来，每条后面注释了为它说话的节号：被注释掉的行只在某个条件下成立，或者用来替换上面那一行；尖括号里的值由你自己定。从你自己的启动脚本出发，一组一组往里加，每加一组都跟上一次实验比——每条推荐都来自它自己的那套环境，收益并不保证能叠加。

### 按节整理的全部推荐参数

```bash
# 2.1 测硬件
--gpu-sniff-test-interval <n>               # 2.1：每 n 步做一次 GPU 与 EP All-to-All 检查

# 2.2 构建镜像（AWS P5 上的 EFA）
# 2.2、3.7：先装 EFA 库，否则 NCCL 会悄悄退回 socket
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}
export NCCL_NET_PLUGIN=ofi                  # 2.2：按名字选；传路径会让 NCCL 留在 socket 上
export FI_PROVIDER=efa                      # 2.2
export FI_EFA_USE_DEVICE_RDMA=1             # 2.2
export NCCL_SOCKET_IFNAME='^lo,docker0,veth_def_agent'  # 2.2：按你的网卡改
export NCCL_DEBUG=INFO                      # 2.2、3.7：要看到 "Using network Libfabric"

# 2.5 你的第一个 MoE
# --deterministic-mode                      # 2.5：要比实验或追 NaN 时开；
# export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 #   同时去掉 --cross-entropy-loss-fusion

# 3.0 监控
--moe-per-layer-logging                     # 3.0、3.5：逐层路由损失
--save <dir> --save-interval <m>            # 3.0：token 统计与 dump 落在这里
--save-tokens-per-expert-interval <n>       # 3.0、3.3：逐专家 token 数、MaxVio
--log-throughput --log-params-norm          # 3.0：TFLOP/s、参数范数
--log-num-zeros-in-grad --log-max-attention-logit  # 3.0：运行健康度
# --moe-routing-trace-path <dir> --moe-routing-trace-max-training-iters <n>
# --cuda-graph-impl none                    # 3.0：有界路由追踪；hook 会跳过 CUDA graph
# --save-activations-interval <n>           # 3.0、3.4：专家输出范数；抽样稀疏一些

# 3.1 路由器打分
--moe-router-dtype fp32                     # 3.1：bf16 的并列值会让专家翻转
# --moe-router-score-function sigmoid       # 3.1、3.3：专家偏置需要它
# --moe-router-pre-softmax                  # 3.1：top-1 时做 softmax，否则路由器没梯度
# --moe-z-loss-coeff 1e-3                   # 3.1：logits 持续上漂的 softmax 路由器

# 3.2 路由策略
# --moe-num-hash-layers <n>                 # 3.2：只为复现 DeepSeek-V4；
#                                           #   要加载固定表，不是占位符

# 3.3 负载均衡
--moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3  # 3.3 辅助损失
# --moe-router-force-load-balancing         # 3.3、3.7：只用于对照实验，绝不用于训练
# --moe-router-load-balancing-type global_aux_loss  # 3.3 作用域：领域专家，步时 +5.8%
# 3.3 无损失偏置，替换上面那行 aux_loss：
# --moe-router-enable-expert-bias --moe-router-score-function sigmoid
# --moe-router-bias-update-rate <rate>      # 按专家数调；超过 128 后 1e-3 太粗
# --moe-router-load-balancing-type seq_aux_loss --moe-aux-loss-coeff 1e-4  # DeepSeek-V3
# --moe-router-load-balancing-type quantile_balancing  # 3.3 分位数法：不要 aux loss，
#                                           #   也不支持路由器融合或分组受限路由

# 3.4 专家
# 2.5 的形状；3.4 数量与大小：专家更细有帮助，但有上限
--num-experts 64 --moe-ffn-hidden-size 1024 --moe-router-topk 8
# --moe-latent-size <l>                     # 3.4 潜空间 MoE：专家更多，通信量不变
# --moe-use-norm-before-up-proj             # 3.4：Kimi K3 对潜空间数值爆炸的修法
# --moe-shared-expert-intermediate-size 1024  # 3.4：被路由的宽度，top-k 减一
# --activation-func-clamp-value 10          # 3.4 激活：专家范数发散的场合
# --activation-func-tanh-clamp-scale 4 --activation-func-tanh-clamp-scale-linear 25
#                                           # 3.4：Kimi K3 的 SiTU-GLU，需 Megatron Core v0.20+

# 3.5 层间布局
# --moe-layer-freq "([0]+[1]*15)"           # 3.5：首层留稠密，少几次 All-to-All
# --moe-shortcut-connection --moe-shortcut-parallel  # 3.5 层间连线：混合栈；
#                                           #   不支持 FSDP、CUDA graph 与完整重算

# 3.6 训练
# --moe-use-upcycling --moe-upcycling-granularity 8  # 3.6：只第一次跑，配 2.5 的两个补丁
# --optimizer muon --muon-extra-scale-factor 0.2     # 3.6 优化器：1.0 会发散；不支持 FSDP
# --moe-router-skip-muon                    # 3.6：路由器留在 Adam，需 Megatron Core v0.20+
# --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1   # 3.6 MTP
# --pipeline-model-parallel-layout <layout> # 3.6：把 MTP 层放进 PP 布局
# --timing-log-level 2 --timing-log-option all       # 3.6：找出最慢的 PP stage
--rerun-mode validate_results               # 3.6 稳定性：默认值，保持开着

# 3.7 系统工程
# --profile --profile-step-start <a> --profile-step-end <b> --profile-ranks <r>
#                                           # 3.7 性能剖析：每个新配置只剖一步
# --moe-permute-fusion                      # 3.7：permute 与 copy kernel 显眼时用
--expert-model-parallel-size 8              # 3.7 互联：EP 控制在一个节点内
# --overlap-moe-expert-parallel-comm --delay-wgrad-compute  # 3.7 重叠：暴露出来的
#                                           #   All-to-All；不支持完整重算与 PP 下的 VPP
# --recompute-granularity selective --recompute-modules layernorm moe_act
# --fine-grained-activation-offloading --offload-modules core_attn attn_proj expert_fc1
#                                           # 3.7 显存：重叠放不下时用
# --activation-offload-fraction <f>         # 3.7：步时变长就把这个比例调低
# --moe-token-dispatcher-type flex --moe-router-num-groups <nodes> --moe-router-group-topk <g>
#                                           # 3.7 分组受限：跨节点 EP，需 DeepEP
# --moe-expert-capacity-factor <cf> --moe-pad-expert-input-to-capacity
#                                           # 3.7 容量：显存吃紧；要微调就改回 dropless
# --fp8-format <fmt> --fp8-recipe blockwise # 3.7 精度：Hopper；Blackwell 上用 mxfp8
# --moe-router-padding-for-quantization     # 3.7：给量化 GEMM 对齐计数
# --moe-dispatch-fwd-dtype mxfp8 --moe-combine-bwd-dtype mxfp8  # 3.7：ncclep，Blackwell

# 4 从训练到推理服务
# --fp4-format e2m1                         # 4 低位宽：Blackwell 上的 NVFP4，不与 FP8 同用
# --rl-inference-expert-model-parallel-size <n>  # 4：rollout 引擎自己的 EP 大小
# --moe-enable-routing-replay               # 4 RL 路由：rollout 与训练器不一致时
# --batch-invariant-mode                    # 4：kernel 那一份不一致；代价是变慢
```

还有大量问题没定。我们对大规模训练 MoE 的了解，大多来自开源权重模型的技术报告，而报告写的是一个团队最终交付了什么：很少写它试过又放弃了什么，也很少写是哪一次实验定下了一个选择。上面许多推荐都建立在某一家、某一套环境里的单次消融上，每一条都值得在别的规模、别的数据、别的集群上重跑一遍。如果你跑了一个，把配置和日志一起发出来，包括负面结果——下一个团队就是这样从你学到的东西起步，而不是从零开始。你可以在本文的[讨论区](https://huggingface.co/spaces/huggingface/moe-blogpost/discussions)分享你的结果，或者问我们。

希望这篇长文能替你下次训 MoE 省下几天——我们自己是为它花掉了几天的。

### 致谢

感谢 [Antoine Bosselut](https://huggingface.co/atcbosselut)、[Martin Jaggi](https://huggingface.co/mjaggi) 与 Swiss AI 团队提供了本文背后的一部分算力，以及一路上那些有启发的讨论。也感谢 [Thibaud Frere](https://huggingface.co/tfrere) 在制图上给予的帮助。

---

**导航**：[全书目录](index.md) · [← 第 4 章 从训练到服务](04-serving.md) · [附录 A 监控细节与排查索引 →](A-appendix.md)
