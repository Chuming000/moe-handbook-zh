# MoE Training Handbook — Chinese Translation

A complete Chinese translation of the **MoE Training Handbook** by Hugging Face and NovaSky.

The original is a hands-on engineering handbook: every number in it comes from real runs on 8×H100 nodes, and the question it answers is *"for each knob in a MoE layer, which setting should you pick, and where does the training time actually go?"* Roughly 51k English words, fully translated here and split into chapters that can be read independently.

- Original: <https://huggingface.co/spaces/huggingface/moe-blogpost>
- Code base: [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) commit `d6316eb45`
- Combined PDF: [`moe-training-handbook-zh.pdf`](./moe-training-handbook-zh.pdf) (102 pages)

## What this is

This is not another "what is a mixture of experts" survey — it is a **recipe**:

1. Measure the hardware first (DCGM diagnostics, All-to-All bandwidth with `nccl-tests`) so you know the ceiling.
2. Build the image, tokenize, train a 1.22B dense baseline, then swap it in place for a MoE with 6.86B total and 1.22B activated parameters.
3. Tune layer by layer: how the router scores, how load gets balanced, how many experts and how wide, how blocks are wired, Muon vs AdamW, MTP or not, and how systems engineering claws back wasted time.
4. Carry the training-time conclusions into serving, and check which ones still hold.

Every section reports *which flag changes / what it buys / what it costs*.

## Structure

```
.
├── README.md                         # this file
├── README_en.md                      # English README (this one)
├── moe-training-handbook-zh.pdf      # full book, 102 pages
└── docs/
    ├── index.md                      # book index and reading paths (start here)
    ├── 01-introduction.md            # Ch.1 Why MoE
    ├── 02-setup.md                   # Ch.2 Environment and first run
    ├── 03-1-monitoring-and-routing.md# Ch.3 (a) Monitoring and routing
    ├── 03-2-load-balancing.md        # Ch.3 (b) Load balancing
    ├── 03-3-experts.md               # Ch.3 (c) Experts
    ├── 03-4-block-layout-and-training.md  # Ch.3 (d) Block layout and training
    ├── 03-5-systems.md               # Ch.3 (e) Systems engineering
    ├── 04-serving.md                 # Ch.4 From training to serving
    ├── 05-conclusion.md              # Ch.5 Conclusion and recommended settings
    └── A-appendix.md                 # Appendix A: monitoring details, debugging index
```

Chapter 3 is the longest (~26k words) and is split into five self-contained parts.

## Notes on figures

The original figures are React components (draggable, switchable) and cannot be reproduced in Markdown or PDF. This translation keeps every figure number and caption, marks them as *"interactive on the original site — omitted here"*, and restates all numbers in the prose so nothing depends on seeing the plot.

To play with the interactive figures, visit the [original site](https://huggingface.co/spaces/huggingface/moe-blogpost).

## Translation notes

| Item | Detail |
|---|---|
| Model | **DeepSeek `deepseek-flash`** (Nov 2026) |
| Method | Chapter-by-chapter translation with a fixed glossary agreed up front — not machine output with a polish pass |
| Review | Formulas, code and flag names checked against the original; every number traced back to its section |
| Code base | Megatron-LM `d6316eb45` (line references follow this commit) |
| Figures | Interactive figures keep their numbers and captions; all conclusions are restated in prose |

## Credits and copyright

- Original work by Hugging Face and NovaSky: Nouamane Tazi, Leandro von Werra, Thomas Wolf (Nov 5, 2026). The upstream repository is licensed **CC-BY-4.0**.
- This translation exists for educational purposes and claims no rights over the original, and is not affiliated with the authors or their organisations. All rights to the original text remain with its authors.
- The translation itself is released under **CC-BY-4.0** — reuse and remix freely, but keep the attribution.
- Issues and PRs for translation mistakes are welcome.

## Citation

Please cite the original work:

> Nouamane Tazi, Leandro von Werra, Thomas Wolf (2026). "The Ultimate MoE Training Handbook".

```bibtex
@misc{tazi2026_the_ultimate_moe_training_handbook,
  title={The Ultimate MoE Training Handbook},
  author={Nouamane Tazi and Leandro von Werra and Thomas Wolf},
  year={2026},
}
```
