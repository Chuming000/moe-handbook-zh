# 第 2 章 环境准备与第一次跑通

## 2.0 你需要什么

这一章里的每一条命令，都是按同一种机器写的，目的是让所有人跑在同一个环境里：一个由八卡节点组成的 Slurm 集群。本节说明这套环境长什么样，你有 1 张卡甚至没有卡时能跟到哪一步，以及怎么在 Hugging Face Jobs 上租一个八卡节点。如果你的机器不一样，讲解照样能读，命令自己改。

### 命令默认的集群形态

这些命令假定你有一个 [Slurm](https://slurm.schedmd.com/) 集群，每个 GPU 节点八张卡。容器通过 [Enroot](https://github.com/NVIDIA/enroot) 与 [Pyxis](https://github.com/NVIDIA/pyxis) 启动，这样 `srun` 就能在镜像里拉起一个任务。2.1 节的节点自检用的是 NVIDIA 的 [DCGM 诊断](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)。我们自己的实验跑在 AWS P5 节点上，节点之间用 EFA（Elastic Fabric Adapter）互联，每台八张 80 GB 的 H100。

这一章大部分内容一个节点就能跑完。稠密基线和第一个 MoE 各要一个八卡、每卡约 80 GB 的节点。只有网络检查需要更多机器：2.1 节的 All-to-All（all-to-all）测试会跑 2、4、8 个节点，2.2 节的 EFA 镜像要在两个节点上验证。

软件来自两个钉死的来源。一是 [NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) 镜像 `26.06-py3`，2.2 节会导入它并补上集群的网络库。二是 [Megatron](https://github.com/NVIDIA/Megatron-LM) 的 `d6316eb45` 这个 commit——它是挂载进容器，而不是打进镜像里。你还需要一个 CPU worker 和 GPU 节点都能读的共享文件系统，因为 2.3 节在 CPU 分区上做分词，训练任务再从那里读结果。

### 只有一张卡、或者一张卡都没有

没有集群也能读懂大半篇。第 3 章是拆解 MoE 层：每个部件做什么、前沿模型各自怎么选、以及每个选择背后的证据。那一章可以完全不跑代码读完，把里面的参数先记下来，等有硬件了再用。第 4 章把这些选择从预训练一路带到推理服务和强化学习。

在这一章里，2.3 节的分词跑在 CPU 上。它的小样本模式单机就能跑、不需要 Slurm，所以你可以亲手产出并检查训练要读的那些文件。2.4 与 2.5 的训练配方假定八张约 80 GB 的卡，而且 MoE 会把专家铺满八张卡。卡少的话，就读方法：两次实验之间什么保持不变、专家宽度是怎么定的、该比什么。2.1 的 All-to-All 测试与 2.2 的 EFA 配置，按原文跑需要多节点。

### 在 Hugging Face Jobs 上租一个节点

如果你想跑单节点那部分又不想买机器，[Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) 可以在 Hugging Face 的机器上跑一个 Docker 镜像加一条命令，按分钟计费。任何余额为正的个人或组织都能用，PRO / Team / Enterprise 订阅的月度额度也能抵扣。`hf jobs hardware` 列出的最大规格就是八卡，正好够 2.4 与 2.5 用；比如 `h200x8` 是八张 141 GB 的 H200。

一个 job 由三样东西定义：来自 Docker Hub 或 Hugging Face Space 的镜像、一条命令、一个硬件规格。所以想在上面跑本章，得把 2.2 节产出的镜像打包成 Docker 镜像，再把 Slurm 启动脚本换成 Jobs 命令。job 默认 30 分钟超时，除非你另设。第一条命令先确认它真能看到八张卡：

```bash
hf jobs run --flavor h200x8 --timeout 10m \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    python -c "import torch; print(torch.cuda.device_count())"
```

不管你在自己集群还是租的节点上跑，发给你的机器未必有规格书上写的那么能打。那么问题来了：一张卡或一条链路在偷偷降速，你怎么发现？

## 2.1 测你的硬件

在开始调模型之前，先摸清机器能干多少活。用 [Megatron](https://github.com/NVIDIA/Megatron-LM) 的 `alltoall` 分发器时，每一层 MoE 都要把 token 送到持有对应专家的卡上，再通过另一次 All-to-All 把结果收回来。所以卡与卡之间的链路，本身就是你要测的工作负载之一：一次健康的 [all-reduce](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=reduce_%26_allreduce) 结果，并不能说明这些 token 交换会有多快。

下面这些检查要在你的作业分配里、启动训练之前跑。用 [Pyxis](https://github.com/NVIDIA/pyxis) 时，`srun --container-image` 就能起容器；下一节讲怎么准备训练镜像。

### 逐个节点自检

先对每个分配到的节点做一次就绪检查。NVIDIA 的 [DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html) 提供几个诊断级别，1 级用来做就绪检查，更高级别会加上帮助定位故障的测试。下面这条命令每节点跑一个任务，这样启动训练前你能逐个看结果。

```bash
NNODES=$SLURM_JOB_NUM_NODES
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
    dcgmi diag -r 1 --fail-early
```

全部通过就去做通信测试。有节点挂了，把它的诊断输出留下来，查清楚再往下走。举例：我们有一台节点上，2 级诊断在三张卡上报了温度告警，同时 PCIe 和显存测试也失败。

更长时间的诊断应该另开维护窗口。NVIDIA 把 3、4 级定位成管理员事后取证，1 级才是就绪检查。

### 准备诊断镜像

如果你的集群已经提供了 `nccl-tests` 镜像，把 `NCCL_TESTS_IMAGE` 指向它就行。否则，在带 EFA 的 AWS 上可以导入那张配合 [AWS NCCL 测试配方](https://github.com/awslabs/awsome-distributed-ai/tree/main/micro-benchmarks/nccl-tests)维护的公开镜像。目标路径要落在分配节点能访问的文件系统上：

```bash
NCCL_TESTS_IMAGE=/shared/images/nccl-tests.sqsh
mkdir -p "$(dirname "$NCCL_TESTS_IMAGE")"
enroot import -o "$NCCL_TESTS_IMAGE" \
    'docker://public.ecr.aws#hpc-cloud/nccl-tests:latest'
```

把导入出来的 SquashFS 文件和你的基线数据放在一起，因为再导一次 `latest` 可能拿到不一样的镜像。想要可重复，就从 AWS 配方里挑一个带版本号的 tag，替掉 `latest`。

AWS 的配方也提供 Dockerfile，需要自己构建时可以用。换别的互联网络，就用它支持的 CUDA、[NCCL](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=nccl)、MPI 和网络插件栈；NVIDIA 的 [nccl-tests 构建说明](https://github.com/NVIDIA/nccl-tests#build)讲了怎么指定这些库的路径。多节点测试要求带 `MPI=1` 构建。下面的基准假定 MPI 能通过 Slurm 的 PMIx 集成拉起，所以镜像要和集群的 MPI 配置对得上。

### 测你实际要用的那个集合通信

节点都过了，就在整个分配上测 All-to-All 带宽。下面这条命令每卡一个 rank，每个 rank 交换 8 GiB 缓冲，对应我们基线用的大消息测试。用上面准备好的 `NCCL_TESTS_IMAGE`；如果你的镜像把 `nccl-tests` 装到了别处，改一下可执行文件路径。

```bash
srun --nodes="$NNODES" --ntasks="$((8 * NNODES))" \
    --ntasks-per-node=8 --gres=gpu:8 --mpi=pmix --cpu-bind=none \
    --container-image="$NCCL_TESTS_IMAGE" --no-container-mount-home \
    /opt/nccl-tests/build/alltoall_perf \
    -b 8G -e 8G -g 1 -c 1 -n 20
```

看结果之前先核对 rank 列表：两个八卡节点应该列出 16 个 rank。然后看目标尺寸那一行的 `busbw` 和 `#wrong` 两列。下面是我们双节点基线的 out-of-place 结果，其他列略去：

```
size (B)      busbw (GB/s)    #wrong
8589934592    77.14           0
```

`#wrong` 为 0，说明这次结果通过了正确性校验。这个带宽可以当作后续同样节点数、同样卡数、同样测试参数下的参照。把节点列表一起存下来，将来出现慢的结果，才能追回到具体是哪几台机器。另一次双节点分配里，我们打开 NCCL 调优变量跑同样的消息大小，只拿到 39.84 GB/s——所以每份基线都要记清楚是哪些节点、什么设置跑出来的。

all-reduce 作为对照仍然有价值，但同一个集合通信操作在不同机器上的表现可能差很远。上面那次偏慢的双节点测试，all-reduce 拿到 398.52 GB/s，而 all-to-all 只有 39.84 GB/s。对一个要跨这些节点交换 token 的 MoE 来说，后者才是测对了通信模式的那个数。

### 每种分配规模各定一条基线

节点加得越多，All-to-All 里跨节点网络的比例就越高。我们的多节点扫描说明了基线为什么必须跟节点数绑定：双节点 77.14 GB/s、四节点 39.25 GB/s、八节点 30.91 GB/s。每次分配都要跟相同节点数、相同消息大小、相同结果列的基线比。

定参照之前，先在一套已知良好的配置上把硬件、软件、测试参数都固定住，重复几次，用得到的区间来判断一份新分配够不够格开训。一个实用的起步策略是：带宽不低于基线的 90%，且正确性校验通过。按上面那次双节点结果，阈值大约 69 GB/s。阈值要按你自己集群的波动来调。举个反例：如果钉死 33 GB/s，那次偏慢的双节点结果会过关，而八节点扫描的结果会被判死。

追查慢分配的时候，别动这条基线。如果你想试 NCCL 调优变量，就在同样的节点、同样的测试参数上对比。我们默认组和调优组用的是不同的节点对，所以那点带宽差归因不到变量头上。

### 训练期间也要盯硬件变化

训练期间的周期性硬件检查，用 Megatron 的 `--gpu-sniff-test-interval <n>`。它会在训练开始前跑一次，之后每 `n` 步跑一次（[`training.py#L4850-L4852`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/training.py#L4850-L4852)、[`training.py#L4484-L4489`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/training.py#L4484-L4489)），其中包含一次专家并行组内的 All-to-All（[`gpu_sniff_test.py#L5-L21`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/gpu_sniff_test.py#L5-L21)）。它给出的离群告警，就是变慢时该去查的 rank（[`gpu_sniff_test.py#L104-L118`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/gpu_sniff_test.py#L104-L118)）。

诊断镜像让你测过了卡和链路。训练镜像需要访问同一批硬件，那接下来就把它做出来，顺便验一下它的网络库。

## 2.2 构建你的镜像

训练环境有两点必须弄对：跑 GPU 计算的库，和把这些卡跨节点连起来的网络库。第一部分从 NVIDIA 的 PyTorch 容器起步，再补上集群需要的网络支持。把 [Megatron](https://github.com/NVIDIA/Megatron-LM) 放在独立的 checkout 里，你就能在不动镜像的前提下改训练代码。

这里的例子用 [Slurm](https://slurm.schedmd.com/) 上的 [Enroot](https://github.com/NVIDIA/enroot) 与 [Pyxis](https://github.com/NVIDIA/pyxis)。基础镜像是 [NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) `26.06-py3`；网络那部分指令针对我们带 EFA 的 AWS P5 节点。换集群就用它支持的网络栈来做这一块。

### 导入基础镜像

Enroot 把容器镜像导成 SquashFS 文件，Pyxis 才能通过 `srun` 启动它。选一个计算节点能访问的输出路径，然后导入：

```bash
IMAGE=/path/to/images/pytorch-26.06.sqsh
enroot import -o "$IMAGE" docker://nvcr.io#nvidia/pytorch:26.06-py3
```

Megatron 的 checkout 放在镜像外面，启动训练时挂进去。本文里的参数针对 commit `d6316eb45`；显式 checkout 到它，能保证你读到的代码和跑起来的代码是同一份。每个测量结果对应的镜像与 commit，都在正文的脚注里列了。

把这个 checkout、以及后面的数据和分词器，都放在 CPU worker 与 GPU 节点都能读的文件系统上。下面的命令里，`SHARED_ROOT` 就是这个目录，`PROJECT_DIR` 是你的项目目录，[配套脚本](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139)克隆到它的 `scripts/` 子目录。

```bash
export SHARED_ROOT=/path/to/shared/work
export PROJECT_DIR="$SHARED_ROOT/moe-blogpost"
git clone https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139.git "$PROJECT_DIR/scripts"
git clone https://github.com/NVIDIA/Megatron-LM.git "$SHARED_ROOT/Megatron-LM"
cd "$SHARED_ROOT/Megatron-LM"
git checkout d6316eb45bf1bfbb6f1f848c0b56e55166564d80
```

### 验证 GPU 可用

补网络库之前，先确认基础镜像能 import PyTorch 与 [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)、能看到卡。在分配好的八卡节点里，用上面的 `IMAGE` 路径跑：

```bash
srun --nodes=1 --ntasks=1 --gres=gpu:8 \
    --container-image="$IMAGE" --no-container-mount-home \
    python -c '
import torch, transformer_engine
print("torch", torch.__version__)
print("transformer_engine", transformer_engine.__version__)
print("device_count", torch.cuda.device_count())
x = torch.randn(8).cuda().sum()
print("cuda sum ok", x.item())
'
```

原生镜像报出来的版本与卡数如下。最后一行还在 GPU 上算了个和，所以别只看 import，把那个结果也一起看了。

```
torch 2.13.0a0+8145d630e8.nv26.06
transformer_engine 2.16.0+4220403e
device_count 8
```

CUDA 那个结果只验证了本节点内的 GPU 执行。要跨节点走 EFA，还得加上下面说的网络插件与依赖。

### 补上 EFA 支持

在我们 AWS 节点上，网络层由 EFA 用户态库和针对镜像内 NCCL 编译的 `aws-ofi-nccl` 插件组成。构建用的是 EFA installer `1.49.0` 与 `aws-ofi-nccl` `v1.19.2`。这两个版本号要和基础镜像一起写进你的构建配方。

[EFA 镜像构建脚本](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139#file-build_efa_image-sh)从基础镜像起一个可写的 Enroot 容器，用 `--skip-kmod` 装 EFA 用户态库，再针对镜像里的 NCCL 编译插件，然后导出新的 SquashFS 并删掉临时容器。在一个有 Enroot 与 GPU 权限的分配节点上跑它，指定新的输出路径：

```bash
EFA_IMAGE=/path/to/images/pytorch-26.06-efa.sqsh
bash "$PROJECT_DIR/scripts/build_efa_image.sh" "$IMAGE" "$EFA_IMAGE"
```

之后启动训练，`--container-image` 都用 `EFA_IMAGE`。脚本只负责把网络层做进镜像；下面这几条环境变量要和镜像一起带着，任务才能找到并选中正确的库。

这里的库路径指向容器内装好的副本，`NCCL_NET_PLUGIN=ofi` 是按名字选插件。我们的环境里，如果给插件传一个文件系统路径，NCCL 会拼出一个非法的库名。

```bash
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}
export NCCL_NET_PLUGIN=ofi
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_SOCKET_IFNAME='^lo,docker0,veth_def_agent'
export NCCL_DEBUG=INFO
```

排除网卡这一项，针对的是 NCCL 建立通信时那次初始连接。在我们的节点上，NCCL 选中了 `veth_def_agent`，试图连它的 link-local 地址，然后 `Connection refused`。把这个网卡排掉，启动才过得去。复用到你自己集群前，先按实际的网卡名改排除列表。

### 验证网络通路

第一次多节点开训时，先看 NCCL 初始化日志，再去解读训练性能。这一步检的是训练镜像自己的网络通路。对本节这个 EFA 镜像，要找的是这一行：

```
NCCL INFO Using network Libfabric
```

如果看到的是 `Using network Socket`，就去读它上面紧挨着的插件加载信息。插件名写错、或者库版本缺失，都会让 NCCL 退回 socket。举例：`EFA_1.4` 这个报错就是在说某个库版本找不到，得先解决再试。

日志只能确认任务选了哪个传输层，带宽还得单独测。把训练镜像、网络设置连同 2.1 节的诊断结果记在一起，将来比较两者性能时环境才说得清。

硬件验过了、镜像也备好了，就可以开始第一次训练。不过得从数据开始：把原始文本变成 token 序列，并落成 Megatron 认识的那种格式。

## 2.3 分词

[Megatron](https://github.com/NVIDIA/Megatron-LM) 从成对的索引文件里读训练文本：`.bin` 存 token ID，`.idx` 告诉 dataloader 每条序列从哪儿开始到哪儿结束。这些文件只做一次，稠密和 MoE 两次实验读同一批 token，换模型就不会顺带换了数据。

我们用 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 和 [Llama 3.2 分词器](https://huggingface.co/meta-llama/Llama-3.2-1B)。[Datatrove](https://github.com/huggingface/datatrove) 的 `MegatronDocumentTokenizer` 直接写这种格式，剩下的活就是选输出目录、验一个小样本、提交大的分词任务。

> **图 2.3.1** 两条示例文档以 token ID 形式存储，各自以文档结束符收尾（原站为交互图，此处从略）

### 准备分词器

数据和分词器都放在 `SHARED_ROOT`，也就是 2.2 节那个共享目录。[分词脚本](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139#file-tokenize_first_run-py)在本地跑小样本，大任务则提交成 [Slurm](https://slurm.schedmd.com/) 数组作业。

```bash
export TOKENIZER_DIR="$SHARED_ROOT/tokenizer"
python3.11 -m venv "$SHARED_ROOT/tokenize-venv"
source "$SHARED_ROOT/tokenize-venv/bin/activate"
python -m pip install \
    'datatrove @ git+https://github.com/huggingface/datatrove.git@2c895af7a95010a96fa53e2dc77cc0ce92799e3f' \
    pyarrow tokenizers huggingface_hub
```

原始分词器仓库 `meta-llama/Llama-3.2-1B` 需要申请权限。你的账号有权限就 `hf auth login`，然后在下载命令里用那个仓库。没有的话，`unsloth/Llama-3.2-1B` 提供了一个不需要门禁的副本：我们比对过，词表和 token ID 一致。分词和训练读的是同一批下载下来的文件。

```bash
TOKENIZER_REPO=unsloth/Llama-3.2-1B
hf download "$TOKENIZER_REPO" \
    --include 'tokenizer*' --include 'special_tokens_map.json' --include 'config.json' \
    --local-dir "$TOKENIZER_DIR"
```

Datatrove 从这个目录读 `tokenizer.json`，并在每篇文档末尾追加 `<|end_of_text|>`。Megatron 之后通过 `--tokenizer-model` 拿到这个目录，同时也拿到分词器的配置。

### 验一个小分片

提交整个数组之前，先在 CPU 计算节点上分一个小样本。下面这条读到 1000 篇为止、全部保留，结果写进另一个目录，免得样本被当成训练语料。

```bash
python "$PROJECT_DIR/scripts/tokenize_first_run.py" \
    --tokenizer "$TOKENIZER_DIR/tokenizer.json" \
    --output "$SHARED_ROOT/data-smoke" --limit 1000
```

输出前缀是 `fineweb-edu_00000_tokens`。用 2.2 节的训练镜像里 Megatron 自己的读取器验一遍，节点要能让 [Pyxis](https://github.com/NVIDIA/pyxis) 起容器。先退出分词环境，否则它的 Python 会盖掉镜像里的解释器。`TRAIN_IMAGE` 在配了 EFA 的机器上指向 `EFA_IMAGE`，否则指向你集群准备好的训练镜像。

```bash
deactivate
export TRAIN_IMAGE="$EFA_IMAGE"
export MEGATRON_PATH="$SHARED_ROOT/Megatron-LM"
srun --nodes=1 --ntasks=1 \
    --container-image="$TRAIN_IMAGE" \
    --container-mounts="$SHARED_ROOT:$SHARED_ROOT" \
    --container-workdir="$MEGATRON_PATH" --no-container-mount-home \
    env PYTHONPATH="$MEGATRON_PATH" python -c '
import sys
from megatron.core.datasets.indexed_dataset import IndexedDataset
data = IndexedDataset(sys.argv[1])
assert len(data) > 0
print("sequences", len(data))
for i in range(min(3, len(data))):
    tokens = data[i]
    assert len(tokens) and tokens[-1] == 128001
    print("length", len(tokens), "last token", int(tokens[-1]))
' "$SHARED_ROOT/data-smoke/fineweb-edu_00000_tokens"
```

你应该看到非零的序列数，以及每条的 `last token 128001`。那个尾 token 就是我们让分词器追加的文档边界。用 Megatron 读一遍，同时也验证了它在 GPU 训练开始前就能解析索引、从 token 文件里读出最前面几条序列。

### 提交分词任务

小样本验的是格式，大任务是为了准备一份能随训练规模一起长大的语料。在登录节点上、同一个虚拟环境里提交。`CPU_PARTITION` 填你集群的 CPU 分区；调度器要 QOS 就加 `--qos`。脚本每个任务要 12 个 CPU、24 GB 内存，而且这些 worker 上同一路径必须能访问到那个虚拟环境。

```bash
source "$SHARED_ROOT/tokenize-venv/bin/activate"
export OUT_DIR="$SHARED_ROOT/data/fineweb-edu"
CPU_PARTITION=your_cpu_partition
python "$PROJECT_DIR/scripts/tokenize_first_run.py" \
    --tokenizer "$TOKENIZER_DIR/tokenizer.json" --output "$OUT_DIR" \
    --tasks 100 --limit -1 --sample-rate 0.074 --partition "$CPU_PARTITION"
```

Datatrove 会打印提交上去的 job ID。用 `squeue -j <job-id>` 跟这个数组，失败的任务到 `$OUT_DIR/logs` 下看，看清楚了再重跑同一条命令。

每个任务分一部分输入文件，按 7.4% 随机抽样后再分词。因为文档长度不一，这个比例并不能固定输出的 token 数。中断后重启时，任务数、采样率、输出目录都别改——Datatrove 靠日志目录下的完成标记跳过已经做完的任务。

### 把分片交给 Megatron

等数组跑完再做训练用的清单。下面这段检查要求 100 个任务完成标记和对应的分片文件都在，然后把分片前缀写进 `data-args.txt`，一行一个。每个前缀同时指代那两个文件，所以清单里既不该有 `.bin` 也不该有 `.idx`。

```bash
export DATA_ARGS="$OUT_DIR/data-args.txt"
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["OUT_DIR"])
prefixes = [root / f"fineweb-edu_{rank:05d}_tokens" for rank in range(100)]
for rank, prefix in enumerate(prefixes):
    assert (root / "logs/completions" / f"{rank:05d}").is_file(), rank
    for suffix in (".bin", ".idx"):
        path = Path(str(prefix) + suffix)
        assert path.is_file() and path.stat().st_size > 0, path
Path(os.environ["DATA_ARGS"]).write_text("\n".join(map(str, prefixes)) + "\n")
print(f"wrote {len(prefixes)} prefixes")
PY
deactivate
```

训练时通过 `--data-args-path` 读这个文件。它给的数据和 `--data-path` 一样，只是把一长串分片挪出了启动命令。启动器还会把 `--data-cache-path` 指到一个持久目录，存 Megatron 的样本索引，之后同配置的实验可以直接复用。

头几次实验我们用 `--split 100,0,0`，全给训练。等你打算用这次实验比模型质量了，就先留出一个留出集。

数据能读、清单也指向了正确的分片，就可以开稠密基线了。它的 loss、梯度范数和每步耗时，会成为引入专家之前的参照系。

## 2.4 稠密基线

MoE 表现异常时，稠密实验给了你一个具体可比的对象。两个模型用同样的数据、batch size 和优化器，于是 loss 尖峰或者某一步变慢，都能先跟那个更简单的模型对一对。这条基线还顺便把训练镜像和 dataloader 一起跑通了——这时候路由还没登场。

模型 16 层 Transformer，hidden 2048，前馈宽 8192。词表 128,256，合计约 12.2 亿参数。配方针对一个八卡、每卡约 80 GB 的节点。启动脚本后面给 MoE 复用。

### 在单节点上启动

训练指标我们用 [Trackio](https://huggingface.co/docs/trackio) 记录。它是 Hugging Face 开源的轻量实验追踪器，API 对齐 `wandb`，通常现有代码只要把 `import wandb` 换成 `import trackio as wandb`。实验记录写进你自己机器上的本地数据库，可以在本地面板里翻，也可以同步到一个 Hugging Face Space，跟团队一起看实时曲线，不需要账号也不花钱。

先在训练镜像里、在共享存储上建一个 Python 环境。它继承镜像的 GPU 库，再补一个钉死版本的追踪包：

```bash
srun --nodes=1 --ntasks=1 \
    --container-image="$TRAIN_IMAGE" \
    --container-mounts="$SHARED_ROOT:$SHARED_ROOT" --no-container-mount-home \
    bash -c 'python -m venv --system-site-packages "$1/train-venv" &&
        "$1/train-venv/bin/python" -m pip install "trackio==0.42.0"' _ "$SHARED_ROOT"
export TRACKIO_PROJECT=moe-first-run
```

启动器通过 [Trackio 入口](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139#file-pretrain_trackio-py)把 [Megatron](https://github.com/NVIDIA/Megatron-LM) 的标量指标送进每个 job 一个 run。两个 run 共用上面这个 project，数据都落在 `$SHARED_ROOT/trackio` 下。这个适配器复用了 Megatron 现有的标量日志钩子，所以像 `--log-memory-to-tensorboard` 这类开关，仍然决定哪些指标会进 Trackio。

[首次运行脚本](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139#file-first_run-sbatch)把训练参数分组塞进 Bash 数组，用前面几节准备好的训练镜像、钉死的 Megatron checkout、分词器目录和数据清单。挂载路径保持在 `SHARED_ROOT` 下，`TRAIN_IMAGE` 指向前面的 SquashFS 镜像。

在共享的博客 checkout 里，把稠密任务提交到你的 GPU 分区。提交时 2.3 节那些变量要保持导出状态。脚本会把镜像路径、Megatron commit 和数据清单一起记进输出。

```bash
cd "$PROJECT_DIR"
GPU_PARTITION=your_gpu_partition
sbatch --partition="$GPU_PARTITION" scripts/first_run.sbatch dense
```

[Slurm](https://slurm.schedmd.com/) 会占下一个八卡节点。节点里一个 `srun` 任务起 `torchrun`，每张卡一个训练进程。运行输出在 `$SHARED_ROOT/runs/dense-<job-id>`，训练日志在提交目录下的 `first-run-<job-id>.out`。

这次实验跑 120 步就停。这个阶段关掉了存 checkpoint 和验证——它只是用来检查环境。日志和 Trackio 数据都留着，下一节要拿它们当参照。

想看曲线，就在一台能读到那个目录的机器上装 `trackio==0.42.0`，然后起面板。在 project 里把稠密和 MoE 两个 run 都选上对比。

```bash
TRACKIO_DIR="$SHARED_ROOT/trackio" trackio show --project "$TRACKIO_PROJECT"
```

### 看懂这套配置

张量并行和[流水并行](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=pipeline_parallelism)都设成 1，所以每张卡上都有一份完整的稠密模型。数据并行把 batch 摊到八张卡上，两次实验都开了分布式优化器和通信重叠。

```bash
--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
--expert-model-parallel-size 1 --context-parallel-size 1
--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
--micro-batch-size 8 --global-batch-size 512
```

每卡一个 micro-batch 是 8 条序列。八张卡一个 micro-batch 就是 64 条，再累积 8 次[梯度](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=gradient_accumulation)，得到 [global batch](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=revisiting_global_batch_size) 512。序列长度 4096，于是每个优化器步吃掉 2,097,152 个 token。这些量固定住，稠密和 MoE 的每步耗时才可以直接比。

学习率走 warmup–稳定–衰减，由 `--lr-decay-style WSD` 选择：先升到 `5e-4`，保持，然后在预算末段衰减。120 步的实验结束在稳定段，它的停止条件并不会截短学习率曲线。

开启完整的[激活重算](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=activation_recomputation)，才能让初始配置塞进我们基线用的显存预算。在 micro-batch 8 加原生交叉熵实现下，不开重算的那次在算 loss 时就把显存吃爆了。两个模型都保留重算，这样第一次比较不会把「改模型」和「省显存」混在一起。

```bash
--recompute-granularity full --recompute-method uniform --recompute-num-layers 1
--cross-entropy-loss-fusion --cross-entropy-fusion-impl native
```

把稠密实验的 loss、梯度范数、每步耗时留在 Trackio 里当参照。等你把前馈层换成专家，这些数就能告诉你这次改动影响了什么。

## 2.5 你的第一个 MoE

稠密模型对每个 token 都套同一个前馈网络。MoE 里，路由器给每个 token 从 64 个更小的专家里挑 8 个，于是模型可以揣着比「单个 token 实际用到」多得多的参数。专家宽度是倒着算的：让被选中的 8 个专家加起来，前馈参数量正好等于被替换掉的那层稠密网络。

这样得到 68.6 亿总参数、每 token 约 12.2 亿激活参数的模型。多出来的容量带着一笔新开销：token 得先跑到持有对应专家的卡上，专家的输出还得再跑回来。

### 替换前馈层

传 `moe` 时启动器会切到 MoE 配置。用和稠密实验相同的导出路径与 GPU 分区提交，镜像、checkout、分词器、数据清单都不变：

```bash
sbatch --partition="$GPU_PARTITION" scripts/first_run.sbatch moe
```

[专家并行](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism)从 1 调到 8，64 个专家铺满这个节点的八张卡：每卡 8 个专家，而路由器是从全集里给每个 token 挑 8 个。

```bash
--expert-model-parallel-size 8 --expert-tensor-parallel-size 1
--num-experts 64 --moe-ffn-hidden-size 1024 --moe-router-topk 8
--moe-token-dispatcher-type alltoall --moe-grouped-gemm
--moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3
--moe-router-dtype fp32 --moe-per-layer-logging
```

`alltoall` 负责在卡之间搬运被路由的 token，grouped GEMM 负责把本地专家的矩阵乘批量算掉。辅助损失以 `1e-3` 的系数鼓励路由器把活摊开。路由 logits 用 fp32 算，并打开逐层日志——这些初始设置和测量结果，第 3 章会逐个翻。

其余都留在公共配置里，包括完整激活重算。特别注意：脚本里稠密的 `--ffn-hidden-size 8192` 还在，而 `--moe-ffn-hidden-size 1024` 给的是每个专家的宽度。

### 数一下激活参数和总参数

无偏置的 SwiGLU 专家有三个权重矩阵。hidden 2048、专家宽 1024 时，它们合起来是 `3 × 2048 × 1024 = 6,291,456` 个参数。选 8 个专家，每 token 就用掉 50,331,648 个前馈参数，正好等于稠密层 `3 × 2048 × 8192` 的量。

> **图 2.5.2** 稠密前馈层与 64 个专家，同一比例尺。一个 token 的 8 个专家首尾相接，宽度恰好等于稠密层：激活参数一样，而存储的参数量是 8 倍（原站为交互图，此处从略）

64 个专家合起来，前馈参数是 8 倍。整个模型的倍数小得多，因为注意力、embedding 和其他非专家参数没变，每层只多了一个不大的路由矩阵。把这些部件在 16 层上加起来，总参数 68.6 亿、激活参数 12.21 亿，而稠密模型是 12.19 亿。

结果是总参数约 5.6 倍，激活参数几乎不变。但激活参数对齐并不保证每步耗时相同：把 token 摊给专家会改变 GPU 上的工作组织方式，还引入了通信。

### 留一条稠密参照

只有一个专家、top-1 路由的 MoE，算的函数和它来源的稠密模型一模一样，所以训练行为也应该一样。在我们的实验里，从一个稠密 checkpoint 上采样（upcycle）出 1 专家 MoE，和稠密实验并排跑，15 个对齐的步里两者差不超过 0.00087。反过来让 MoE 自己初始化再比，就没什么意义了——两边在第 4 步就差了 2.72。MoE 那边还记录到负载均衡损失恒等于 1，梯度里有约 32,770 个零，而稠密实验最多只有 17 个：那是路由权重，只有一个专家时它拿不到梯度。这次对齐实验走的是 upcycle 稠密 checkpoint 这条路，3.6 节还会回来讲它。

在我们钉的这个 commit 上，upcycle 这条路要打两个本地补丁才跑得起来，都在 `training.py` 里。第一处是往 checkpoint loader 传稠密模型时用的关键字是 `model`（[`training.py#L3074-L3078`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/training.py#L3074-L3078)），而 [`load_checkpoint`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/checkpointing.py#L2632-L2634) 里那个参数叫 `ddp_model`。第二处在为加载而构建的稠密模型上：upcycle 的代码是在 `args` 里把稠密形状打开（[`training.py#L3056-L3062`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/training.py#L3056-L3062)），但模型构建器读的是启动时存下来的那份配置（[`training.py#L2917-L2926`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/training.py#L2917-L2926)），于是那个模型仍然是 MoE 层的布局，加载时报缺少 `pre_mlp_layernorm`。

确定性模式能让一次实验可复现，这样两次实验之间的差异才有意义。在这个 commit 上，`--deterministic-mode` 要求关掉交叉熵 loss 融合、TP 通信重叠和融合版 aux-loss kernel（[`determinism.py#L28-L37`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/determinism.py#L28-L37)、[`determinism.py#L195-L210`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/determinism.py#L195-L210)），并且只接受 `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`（[`determinism.py#L85-L90`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/determinism.py#L85-L90)）；在用户没设的时候，它会自己设上 `NCCL_ALGO=Ring`、`NCCL_EP_HT_EM_AG_SCAN_MODE=1`、`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`（[`determinism.py#L40-L50`](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/megatron/training/determinism.py#L40-L50)）。2.4 节的启动器传了 `--cross-entropy-loss-fusion` 并导出 `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1`，加这个开关前两样都要去掉。按这套设置，这个 MoE 缩到 micro-batch 2、global batch 64、跑 12 步，两次实验每一步打印的 lm loss、负载均衡损失、梯度范数和梯度零个数都完全一致；不设这些参数的话，lm loss 从第一步就开始漂，最大差到 0.00063。

要比两次实验、或者追一个 NaN，先把确定性模式打开。

开发 MoE 的过程中，把稠密实验和它的配置一直当参照。一旦某个改动让训练变慢、或者 loss 不降了，就回到 [Trackio](https://huggingface.co/docs/trackio) 里的稠密实验，在同样的 batch size 与序列长度下比每步耗时、loss 和梯度范数。两次实验都有的问题，指向它们共用的那套环境；只在 MoE 里出现的问题，才有理由去查路由和专家执行。

> **图 2.5.4** Trackio 里的稠密与 MoE 示例实验。这两条曲线只是示意界面，不是真实测得的性能（原站为交互图，此处从略）

有了稠密参照，你就知道该从哪儿开始查。第 3 章会把 MoE 层一块一块拆开，把理解它行为所需的那些测量逐个介绍一遍。

---

**导航**：[全书目录](index.md) · [← 第 1 章 引言](01-introduction.md) · [第 3 章（上）监控与路由 →](03-1-monitoring-and-routing.md)
