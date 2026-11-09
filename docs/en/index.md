# Diy-LLM — How Large Language Models Are Forged

<div align="center">
  <img src="./images/diy-llm.png" alt="diy-llm" width="100%">
  <p><em>From tokenizers, data, and training to systems optimization, inference, alignment, evaluation, and multimodal models: the full pipeline of building an LLM, connected into one knowledge system.</em></p>
</div>

<div align="center">
  <p>
    <a href="https://datawhalechina.github.io/diy-llm/en/"><img src="https://img.shields.io/badge/Read_Online-0969da?style=for-the-badge" alt="Read Online"></a>
    <a href="https://github.com/datawhalechina/diy-llm/releases/latest/"><img src="https://img.shields.io/badge/PDF_Notes-2ea44f?style=for-the-badge" alt="PDF Notes"></a>
    <a href="https://github.com/datawhalechina/diy-llm/tree/main/coursework"><img src="https://img.shields.io/badge/Assignment_Code-f59e0b?style=for-the-badge" alt="Assignment Code"></a>
  </p>
</div>

Training large models is often jokingly called "alchemy": data is the raw material, compute is the furnace, and hyperparameters all come down to feel. CS336 explains the process well, but the entry barrier is high; we spell out every "ingredient" and every step so that the folklore turns back into reproducible principles and code.

## ✍️ Why This Project

In Spring 2025, [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/spring2025/) launched. The course is demanding, and learners in our community kept telling us the same thing: they wanted to study it, but the English lecture notes, the assumed prerequisites, and the assignment environment were real barriers. What they needed was a Chinese version they could actually understand and work through.

So we brought together a group of contributors and spent over half a year on **writing → closed beta → public beta → chapter-by-chapter revision**. This is not a translation of the original. We reorganized the material around how Chinese-speaking learners study: added background explanations, wrote out assignment implementations, checked and expanded the reference sources, and brought in examples from models developed in China. AI was used to assist with translation and material organization, but every chapter was written and finalized by human contributors, cross-reviewed, and revised repeatedly based on closed-beta and public-beta feedback. **It is neither machine-translated nor AI-generated in bulk.** Each major chapter has a named author, and the whole text went through a full content review. We call it a course "with a human touch" — that is also where most of the time went.

The course currently contains **16 chapters and 6 assignments**, coordinated by 3 project leads with contributions from many community members, and is continuously open to corrections (see "Contributors" below).

### What We Changed from the Original

| Dimension | Original CS336 | Diy-LLM (Chinese edition) |
|-----------|----------------|---------------------------|
| Knowledge structure | Follows lecture order and assumes math, PyTorch, and systems background | Reordered around the engineering pipeline of building an LLM; math, PyTorch, and GPU prerequisites are explained up front for a smoother learning curve |
| Explanation | Lectures and notes in English | Written chapter by chapter in Chinese, with derivations, terminology, and common confusions unpacked, while keeping the original technical depth |
| Assignments | Assignment handouts and tests | Six runnable implementations plus the reasoning behind the code; CPU debugging and MacBook MPS support added |
| Reference sources | Lecture notes and paper pointers | More detailed references so readers can trace back to primary sources |
| Models and examples | Mostly models from the English-speaking ecosystem | Qwen, DeepSeek, Qwen-VL, DeepSeekMoE, R1 / GRPO, and other cases from China's open-source ecosystem |
| Evaluation | MMLU, HumanEval, and other international benchmarks | Chinese benchmarks such as C-EVAL added and integrated into Assignment 6 |
| Runtime environment | Assumes the compute and environment provided by the course | Cloud GPU options, CPU / Mac debugging advice, and an offline PDF edition for readers in China |
| How it was written | — | Human-written and finalized (AI-assisted translation and organization) + cross-review + beta feedback; not AI-generated in bulk |

> Because the Chinese edition is reordered around the engineering pipeline, **chapter numbers do not map one-to-one to the original lectures**. We recommend following the chapter order used here.

### Version Alignment

- **Originally based on**: [Stanford CS336 Spring 2025](https://cs336.stanford.edu/spring2025/) (the archived site for that offering); the main storyline and Assignments 1–5 correspond to the original assignments 1–5.
- **2026 update**: Synchronized with the new [Stanford CS336 Spring 2026](https://cs336.stanford.edu/) material; topics such as RLVR and multimodality in the new schedule correspond to Chapters 14 and 15 here. Assignment 6 (model evaluation) is new to this edition.
- This is a community-run study project with no official affiliation with Stanford. Original lecture and assignment materials: [stanford-cs336/lectures](https://github.com/stanford-cs336/lectures/tree/main).

## Prerequisites

- **Python**: Proficient in Python and software engineering skills
- **Deep Learning**: Familiar with PyTorch and neural network fundamentals
- **Mathematics**: Linear algebra, probability & statistics, calculus
- **Machine Learning**: Solid grasp of ML and DL basics
- **GPU Programming (optional)**: CUDA basics help but aren't required

## 🎯 What You'll Be Able to Build

This course does not stop at "understanding the concepts": for each stage, you either build it yourself or can explain the design trade-offs.

- **A language model implemented from scratch, and its architectural variants**: hand-write a BPE tokenizer, implement RoPE, RMSNorm, SwiGLU, AdamW, and the training loop; swap the FFN for a Top-K-routed MoE and explain load balancing and the trade-offs behind DeepSeekMoE (Chapters 2, 4, 5).
- **Make it trainable, fast, and cheap to serve**: estimate FLOPs and memory, implement FlashAttention-2 in Triton, set up data/model parallelism; on the inference side, work with KV cache, speculative decoding, quantization, and continuous batching (Chapters 3, 6–8, 10).
- **A complete data, training, and alignment pipeline**: turn raw Common Crawl into a corpus through filtering, deduplication, and PII removal; run SFT and DPO / RLHF; then train mathematical reasoning with RLVR (GRPO) and understand the training path behind R1-style approaches (Chapters 11, 13, 14).
- **Evaluate it, size it, and extend it to multimodality**: run multidimensional evaluation with lm-evaluation-harness, EvalScope, and C-EVAL; fit scaling laws to predict model scale; and read the multimodal path from CLIP / SigLIP to LLaVA and Qwen-VL (Chapters 9, 12, 15).

## Table of Contents

| Chapter | Key Content | Assignment | Status |
|------|----------|----------|------|
| [Foreword](Foreword.md) | Project background, learning roadmap, prerequisites | - | ✅ |
| [Ch1 Tooling](chapter1/wandb_usage_guide.md) | W&B experiment tracking, hyperparameter search | - | ✅ |
| [Ch2 Tokenizer](chapter2/chapter2_Tokenizer.md) | BPE algorithm, Unicode normalization, tokenizer training | [HW1](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment1-basics/) | ✅ |
| [Ch3 PyTorch & Resource Accounting](chapter3/chapter3_PyTorch_and_Resource_Accounting.md) | Mixed precision, gradient accumulation, FLOPs/memory estimation | - | ✅ |
| [Ch4 Architecture & Training Details](chapter4/chapter4_Architecture_and_Training_Details.md) | RoPE, RMSNorm, SwiGLU, AdamW, learning rate schedules | [HW1](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment1-basics/) | ✅ |
| [Ch5 Mixture of Experts](chapter5/chapter5_Mixture_of_Experts.md) | Top-K routing, load balancing, expert parallelism, DeepSeekMoE | - | ✅ |
| [Ch6 GPU & Optimization](chapter6/chapter6_GPU_and_Optimization.md) | Memory bandwidth, Flash Attention, kernel fusion, BF16 | [HW2](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment2-systems/) | ✅ |
| [Ch7 GPU High-Perf Programming](chapter7/chapter7_GPU_High_Performance_Programming.md) | CUDA, Tensor Cores, shared memory, Triton | [HW2](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment2-systems/) | ✅ |
| [Ch8 Distributed Training](chapter8/chapter8_Distributed_Training.md) | Data/Model/Pipeline parallelism, ZeRO, FSDP, All-Reduce | [HW2](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment2-systems/) | ✅ |
| [Ch9 Scaling Laws](chapter9/chapter9_Scaling_Laws.md) | Chinchilla optimal, compute-efficient training | [HW3](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment3-scaling/) | ✅ |
| [Ch10 Inference](chapter10/chapter10_Inference.md) | KV cache, speculative decoding, quantization, PagedAttention | [HW6](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment6-evaluation/) | ✅ |
| [Ch11 Data Engineering](chapter11/chapter11_Data_Engineering.md) | Data filtering, MinHash dedup, PII removal, data recipes | [HW4](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment4-data/) | ✅ |
| [Ch12 Evaluation & Benchmarks](chapter12/chapter12_Evaluation_and_Benchmarks.md) | MMLU, HumanEval, HELM, Chatbot Arena, safety benchmarks | [HW6](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment6-evaluation/) | ✅ |
| [Ch13 Training Pipeline](chapter13/chapter13_Training_Pipeline.md) | Pre-training, SFT, DPO, RLHF PPO pipeline, alignment | [HW5](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment5-alignment/) | ✅ |
| [Ch14 RLVR](chapter14/chapter14_RLVR.md) | GRPO, rule-based verifiers, process reward, DeepSeek-R1 | [HW5](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment5-alignment/) | ✅ |
| [Ch15 Multimodal Models](chapter15/chapter15_Multimodal_Models.md) | CLIP, SigLIP, LLaVA, Qwen-VL, Chameleon, Omni Model | - | ✅ |
| [Ch16 Extended Content](chapter16/) | 1. What is LLM Reasoning?<br>2. LLM Future — LeCun's Roadmap | - | 🔄 |

> Status: ✅ Complete &ensp;🔄 In Progress&ensp;📝 Draft&ensp;🚧 Planned&ensp;⏸️ Paused

## Assignments

| HW | Core Tasks | Status |
|------|----------|------|
| [HW1: Build an LLM](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment1-basics/) | Implement tokenizer, model architecture, optimizer; train a minimal LM | ✅ |
| [HW2: Systems](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment2-systems/) | Benchmarking, FlashAttention-2 in Triton, distributed training code | ✅ |
| [HW3: Scaling Laws](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment3-scaling/) | Fit scaling laws, predict model scaling behavior | ✅ |
| [HW4: Data Processing](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment4-data/) | Convert Common Crawl raw data to pre-training dataset | ✅ |
| [HW5: Alignment](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment5-alignment/) | Apply SFT and RL (e.g., GRPO) to train on math problems | ✅ |
| [HW6: Evaluation](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment6-evaluation/) | Multi-dimension evaluation using lm-evaluation-harness and evalscope | ✅ |

## Quick Start

```bash
git clone https://github.com/datawhalechina/diy-llm.git
cd diy-llm
```

### Learning Path

1️⃣ Theory → Read `docs/en/` chapters in order  
2️⃣ Practice → Complete the 6 assignments in `coursework/`  
3️⃣ Deep Dive → Read code implementations, understand design decisions  

## Project Structure

```
diy-llm/
├── docs/
│   ├── zh/                 # Chinese docs (default)
│   ├── en/                 # English docs
│   └── .vitepress/         # VitePress config
├── coursework/             # Assignments
│   ├── assignment1-basics/
│   └── ...
└── README.md
```

## Links

- **Repository**: https://github.com/datawhalechina/diy-llm
- **Read online**: https://datawhalechina.github.io/diy-llm/en/ ([简体中文](https://datawhalechina.github.io/diy-llm/))
- **Course notes PDF**: https://github.com/datawhalechina/diy-llm/releases/latest/
- **Original course website (current · Spring 2026)**: https://cs336.stanford.edu/
- **Original course website (archive · Spring 2025, the edition this project was first based on)**: https://cs336.stanford.edu/spring2025/
- **Original course repository**: https://github.com/stanford-cs336/lectures/tree/main

## FAQ

<details>
<summary><b>Q: Can I learn without a GPU?</b></summary>
Theory can be studied normally. Some assignments can be debugged on CPU, but full training requires GPU. Cloud services are recommended.
</details>

<details>
<summary><b>Q: How does this differ from the original CS336?</b></summary>
We retain the original technical depth while adding Chinese explanations, detailed reference sources, domestic model examples (Qwen, DeepSeek), and complete assignment implementations. See "What We Changed from the Original" for a point-by-point comparison and "Version Alignment" for how the two CS336 offerings map to this project.
</details>

## Reader Community

Join the Diy-LLM reader community to discuss the course, ask questions, and learn together:

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="/读者交流群1.png" alt="Reader community group 1 QR code" width="280"><br>Group 1</td>
      <td align="center"><img src="/读者交流群2.png" alt="Reader community group 2 QR code" width="280"><br>Group 2</td>
    </tr>
  </table>
</div>

If group 1 is full, join group 2. If both QR codes have expired or are full, add any maintainer below on WeChat (note: diy-llm) and we'll add you to the group:

| Maintainer | WeChat ID |
|------------|-----------|
| Hu Xu (徐虎) | `xuhu96736` |
| Li Youzhen (黎又榛) | `zydsx111` |
| Li Shengkang (李盛康) | `muzichengminguangli` |

## Contributors

<table border="0">
  <tbody>
    <tr align="center" >
      <td>
         <a href="https://github.com/xuhu0115"><img width="70" height="70" src="https://github.com/xuhu0115.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/xuhu0115">Hu Xu (徐虎)</a> 
        <p>Project Lead · Datawhale · SJTU<br>Chapters: 1, 3, 9, 12, 14, 15; HW5, 6; Content review</p>
      </td>
      <td>
         <a href="https://github.com/kangkang-Adam"><img width="70" height="70" src="https://github.com/kangkang-Adam.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/kangkang-Adam">Li Shengkang (李盛康)</a> 
        <p>Project Lead · Datawhale · XUPT<br>Chapters: 4, 6, 7, 8, 13; HW2, 4</p>
      </td>
      <td>
         <a href="https://github.com/1iyouzhen"><img width="70" height="70" src="https://github.com/1iyouzhen.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/1iyouzhen">Li Youzhen (黎又榛)</a>
         <p>Project Lead · Datawhale<br>Chapters: 2, 5, 10, 11, 13; HW1, 3</p>
      </td>
    </tr>
  </tbody>
</table>

- Thanks to [@aimetrics](https://github.com/aimetrics) for adding MPS support for MacBooks in HW1
- Thanks to [@FuTseYi](https://github.com/FuTseYi) (Datawhale) for website migration, deployment, and refactoring
- Thanks to [@jiangyinhe](https://github.com/jiangyinhe) for contributing to Ch16 "LLM Future — LeCun"; and for translating the first chapter and preface into English based on the Chinese version.

### How to Contribute

Corrections, assignment implementations, new content, translations, and shared experience are all welcome. **Before you start, please read [CONTRIBUTING.md](https://github.com/datawhalechina/diy-llm/blob/main/CONTRIBUTING.md)** for the full submission workflow, content and code conventions, and the PR self-check list.

Three things that matter most: small fixes can go straight to an [Issue](https://github.com/datawhalechina/diy-llm/issues) or a PR, while new chapters and major changes should be discussed in an Issue first; PRs should state what changed, the evidence, and what you verified; AI assistance is allowed, but you must have read, understood, and verified the submission yourself — bulk AI-generated or densely submitted PRs will be asked to consolidate or will be closed, and repeated bulk submissions will no longer be accepted.

## 🗓️ Versions & Milestones

**Overall progress**: `████████████████░░░░` Writing → closed beta → public beta → second public beta complete; now in continuous maintenance

<table width="100%" cellpadding="8">
  <tr>
    <td bgcolor="#dafbe1" align="center" width="30%"><b>✅ Writing &amp; Revision</b><br>2025.11 — 2026.02</td>
    <td bgcolor="#ddf4ff" align="center" width="14%"><b>✅ Closed Beta</b><br>2026.03.05</td>
    <td bgcolor="#fbefff" align="center" width="14%"><b>✅ Public Beta</b><br>2026.04</td>
    <td bgcolor="#fff8c5" align="center" width="16%"><b>✅ 2nd Public Beta</b><br>2026.09</td>
    <td bgcolor="#f6f8fa" align="center" width="26%"><b>🔄 Maintenance</b><br>2026.10 — present</td>
  </tr>
</table>

Dates come from the commit history and version tags (757 commits, `V0.1` / `V0.2`), with beta dates taken from project test records:

| Date | Milestone | Evidence |
|------|-----------|----------|
| 2025-11-24 | Repository created; writing started with the tokenizer chapter | `Initial commit` |
| 2025-12 — 2026-01 | Peak writing period (186 commits in December); assignment implementations completed | Commits "完成作业一", "完成作业" |
| 2026-03-05 | Closed beta | Closed-beta feedback revisions |
| 2026-04 | Public beta | Public-beta feedback revisions |
| 2026-06-10 | `V0.1`: PDF notes released | tag `V0.1`, `add pdf download` |
| 2026-07-26 | Content from CS336 2026 added | Commit "修改第七章，新增2026内容" |
| 2026-08-17 | `V0.2`: PDF notes v2, reader groups | tag `V0.2`, `pdf v2 + reader communication` |
| 2026-09 | Second public beta | Second-beta feedback revisions |
| 2026-10 — present | Ongoing corrections: formula rendering, figures, wording | 28 commits in the last two months |

See [GitHub Releases](https://github.com/datawhalechina/diy-llm/releases) for the current version and PDF notes, "Why This Project" above for the writing and review process, and "Table of Contents" for per-chapter status.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="CC BY-NC-SA 4.0" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

This work is licensed under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgments

- Stanford CS336 course team for the excellent original course
- [@Sm1les](https://github.com/Sm1les) for support and guidance
- All contributors and the open-source community

## Star History

<a href="https://www.star-history.com/?repos=datawhalechina%2Fdiy-llm&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=datawhalechina/diy-llm&type=date&theme=dark&legend=top-left&sealed_token=ZaKMAVlJ7ht4jbVMmrIgbWlsmeGT0P-zVEivYCbfcgbt7tsA67sQo1rwoHTT5E5ajegRha9nPoc_IKk-fkvfKfddLlLONIBjPzt4QnXtwox4VKQr78nKyugDQvaziK1vjbWMPuzwwJQssk6wgPJyD2evUEO7R72ZdpHhMMABx5ZAX1uQJBG_DdDaheA1" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=datawhalechina/diy-llm&type=date&legend=top-left&sealed_token=ZaKMAVlJ7ht4jbVMmrIgbWlsmeGT0P-zVEivYCbfcgbt7tsA67sQo1rwoHTT5E5ajegRha9nPoc_IKk-fkvfKfddLlLONIBjPzt4QnXtwox4VKQr78nKyugDQvaziK1vjbWMPuzwwJQssk6wgPJyD2evUEO7R72ZdpHhMMABx5ZAX1uQJBG_DdDaheA1" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=datawhalechina/diy-llm&type=date&legend=top-left&sealed_token=ZaKMAVlJ7ht4jbVMmrIgbWlsmeGT0P-zVEivYCbfcgbt7tsA67sQo1rwoHTT5E5ajegRha9nPoc_IKk-fkvfKfddLlLONIBjPzt4QnXtwox4VKQr78nKyugDQvaziK1vjbWMPuzwwJQssk6wgPJyD2evUEO7R72ZdpHhMMABx5ZAX1uQJBG_DdDaheA1" />
 </picture>
</a>

---

<div align="center">
  <p>Empowering everyone to systematically learn large language model construction</p>
  <p>Made with ❤️ by Datawhale</p>
</div>
