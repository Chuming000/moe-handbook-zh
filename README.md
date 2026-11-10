# MoE Training Handbook 中文翻译项目

本仓库是 Hugging Face 与 NovaSky 合著的《MoE Training Handbook》全文中文翻译。

原文是一份可以照着动手做的工程手册：所有数字都来自 8×H100 上的真实运行，讲的是「MoE 这一层的每个部件怎么选、每个开关到底把训练时间花在哪」。英文原文约 5.1 万词，本仓库把它完整译成中文，并按章节拆成可独立阅读的文档。

- 原文：<https://huggingface.co/spaces/huggingface/moe-blogpost>
- 代码基准：[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) commit `d6316eb45`
- 合并版 PDF：[`moe-training-handbook-zh.pdf`](./moe-training-handbook-zh.pdf)（102 页）

## 项目概述

这不是一份「讲概念」的 MoE 综述，而是一份**配方**：

- 先测硬件（DCGM 诊断 + nccl-tests 的 All-to-All 带宽），知道机器的天花板在哪；
- 再建镜像、分词、训一个 1.22B 的稠密基线，然后原地换成一个 6.86B 总参 / 1.22B 激活的 MoE；
- 然后逐层调：路由器怎么打分、怎么均衡负载、专家多少个多宽、块怎么接、优化器用 Muon 还是 AdamW、MTP 要不要、系统工程怎么把浪费的时间抢回来；
- 最后把这些训练期的结论带到推理服务里，看哪些还成立、哪些要翻案。

每一节都给出「改哪个参数 / 换来多少 / 代价是什么」，而不是「MoE 很好」。

## 仓库结构

```
.
├── README.md                         # 本文件
├── README_en.md                      # English README
├── moe-training-handbook-zh.pdf      # 全书合并版 PDF（102 页）
└── docs/
    ├── index.md                      # 全书目录与阅读路线（从这里开始）
    ├── 01-introduction.md            # 第 1 章 引言：为什么是 MoE
    ├── 02-setup.md                   # 第 2 章 环境与第一次跑通
    ├── 03-1-monitoring-and-routing.md# 第 3 章（上）监控与路由
    ├── 03-2-load-balancing.md        # 第 3 章（中）负载均衡
    ├── 03-3-experts.md               # 第 3 章（中）专家
    ├── 03-4-block-layout-and-training.md  # 第 3 章（下）块布局与训练
    ├── 03-5-systems.md               # 第 3 章（下）系统工程
    ├── 04-serving.md                 # 第 4 章 从训练到服务
    ├── 05-conclusion.md              # 第 5 章 结语与推荐参数
    └── A-appendix.md                 # 附录 A 监控细节与排障索引
```

第 3 章是全书最长的一章（约 2.6 万词），拆成了五份，每份都能单独读完。

## 阅读路线

| 你的情况 | 建议 |
|---|---|
| 第一次接触 MoE | `01-introduction.md` → `03-1` → `03-2` |
| 手上要训一个 MoE | `02-setup.md` → `05-conclusion.md`（直接抄参数）→ 再回头查第 3 章对应小节 |
| 已经在训，但吞吐不达标 | `03-2`（负载均衡）→ `03-5`（系统工程）→ `A-appendix.md`（排障索引） |
| 只关心推理 | `04-serving.md` → `05-conclusion.md` |
| 时间只有半小时 | `05-conclusion.md` 的推荐参数表 + `A-appendix.md` 的 42 条症状表 |

## 关于图

原站的图都是 React 交互组件（可拖动、可切换），无法在 Markdown 与 PDF 里复现。本译本的处理方式是：

- 保留每一张图的编号与图注，正文里标注「（原站为交互图，此处从略）」；
- 凡是依赖图得出的结论，正文都用文字写清楚数字，不依赖看图。

想操作交互图，请直接访问[原文](https://huggingface.co/spaces/huggingface/moe-blogpost)。

## 翻译说明

| 项 | 内容 |
|---|---|
| 翻译模型 | **DeepSeek `deepseek-flash`**（2026-11） |
| 翻译方式 | 逐章人工翻译，先定术语表再落笔；不是「机翻后润色」 |
| 校对 | 公式、代码、参数名与原文逐条比对；每条数字都回到原文对应小节核过 |
| 代码基准 | Megatron-LM `d6316eb45`，代码行号按此 commit 标注 |
| 图表 | 交互图保留图号与图注，正文用文字复述全部结论数字 |

## 致谢与版权

- 原文由 Hugging Face 与 NovaSky 团队撰写：Nouamane Tazi、Leandro von Werra、Thomas Wolf（2026-11-05）。原文仓库 `app/package.json` 声明为 **CC-BY-4.0**。
- 本译文仅用于教育目的，**不主张对原文的任何权利**，也不代表原作者或其所属机构；原始内容的一切权利归原作者所有。
- 译文本身同样以 **CC-BY-4.0** 发布：可自由转载、修改，但请保留出处链接与作者署名。
- 发现翻译错误欢迎提 Issue，或直接 PR。

## 引用

引用请一律指向原文：

> Nouamane Tazi, Leandro von Werra, Thomas Wolf (2026). "The Ultimate MoE Training Handbook".

```bibtex
@misc{tazi2026_the_ultimate_moe_training_handbook,
  title={The Ultimate MoE Training Handbook},
  author={Nouamane Tazi and Leandro von Werra and Thomas Wolf},
  year={2026},
}
```
