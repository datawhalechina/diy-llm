<div align="center">
    <img src="./docs/public/diy-llm.png" alt="Diy-LLM banner" width="100%">
    <h1>Diy-LLM</h1>
    <p>
      <a href="./README.md"><img src="https://img.shields.io/badge/简体中文-d1d5db?style=for-the-badge" alt="简体中文"></a>
      <a href="./README_EN.md"><img src="https://img.shields.io/badge/English-0969da?style=for-the-badge" alt="English"></a>
    </p>
</div>

<div align="center">
  <img src="https://img.shields.io/github/stars/datawhalechina/diy-llm?style=flat&logo=github" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/datawhalechina/diy-llm?style=flat&logo=github" alt="GitHub forks">
  <img src="https://img.shields.io/badge/language-English-brightgreen?style=flat" alt="Language">
  <a href="https://github.com/datawhalechina/diy-llm"><img src="https://img.shields.io/badge/GitHub-Project-blue?style=flat&logo=github" alt="GitHub Project"></a>
</div>

<div align="center">
  <p>
    <a href="https://datawhalechina.github.io/diy-llm/en/"><img src="https://img.shields.io/badge/Read_Online-0969da?style=for-the-badge" alt="Read Online"></a>
    <a href="https://github.com/datawhalechina/diy-llm/releases/latest/"><img src="https://img.shields.io/badge/PDF_Notes-2ea44f?style=for-the-badge" alt="PDF Notes"></a>
    <a href="https://github.com/datawhalechina/diy-llm/tree/main/coursework"><img src="https://img.shields.io/badge/Assignment_Code-f59e0b?style=for-the-badge" alt="Assignment Code"></a>
  </p>
  <h3>📚 大模型是怎样炼成的 · How Large Language Models Are Forged</h3>
  <p><em>From tokenizers, data, and training to systems optimization, inference, alignment, evaluation, and multimodal models: the full pipeline of building an LLM, connected into one knowledge system.</em></p>
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

## 📋 Prerequisites

- **Python programming**: Proficiency in Python and solid software engineering skills
- **Deep learning fundamentals**: Familiarity with PyTorch and the basic principles of neural networks
- **Mathematics**: Linear algebra, probability and statistics, and calculus
- **Machine learning**: A solid grasp of the fundamentals of machine learning and deep learning
- **GPU programming (optional)**: Familiarity with basic CUDA concepts is helpful but not required; this project also includes introductory material

## 🎯 What You'll Be Able to Build

This course does not stop at "understanding the concepts": for each stage, you either build it yourself or can explain the design trade-offs.

- **A language model implemented from scratch, and its architectural variants**: hand-write a BPE tokenizer, implement RoPE, RMSNorm, SwiGLU, AdamW, and the training loop; swap the FFN for a Top-K-routed MoE and explain load balancing and the trade-offs behind DeepSeekMoE (Chapters 2, 4, 5).
- **Make it trainable, fast, and cheap to serve**: estimate FLOPs and memory, implement FlashAttention-2 in Triton, set up data/model parallelism; on the inference side, work with KV cache, speculative decoding, quantization, and continuous batching (Chapters 3, 6–8, 10).
- **A complete data, training, and alignment pipeline**: turn raw Common Crawl into a corpus through filtering, deduplication, and PII removal; run SFT and DPO / RLHF; then train mathematical reasoning with RLVR (GRPO) and understand the training path behind R1-style approaches (Chapters 11, 13, 14).
- **Evaluate it, size it, and extend it to multimodality**: run multidimensional evaluation with lm-evaluation-harness, EvalScope, and C-EVAL; fit scaling laws to predict model scale; and read the multimodal path from CLIP / SigLIP to LLaVA and Qwen-VL (Chapters 9, 12, 15).

## 📖 Course Outline

| Chapter | Key Topics | Assignment | Status |
|---------|------------|------------|--------|
| [Foreword](docs/en/Foreword.md) | Project background, learning path, and prerequisites | - | ✅ |
| [Chapter 1: Tools](docs/en/chapter1/) | W&B experiment tracking, hyperparameter search, and visualization dashboards | - | 📝 |
| [Chapter 2: Tokenizers](docs/en/chapter2/chapter2_Tokenizer.md) | BPE, Unicode normalization, and implementing tokenizer training from scratch | [Assignment 1](coursework/assignment1-basics/) | ✅ |
| [Chapter 3: PyTorch and Resource Accounting](docs/en/chapter3/chapter3_PyTorch_and_Resource_Accounting.md) | Mixed-precision training, gradient accumulation, FLOPs, and memory estimation | - | ✅ |
| [Chapter 4: Language Model Architecture and Training Details](docs/en/chapter4/chapter4_Architecture_and_Training_Details.md) | RoPE, RMSNorm, SwiGLU, AdamW, Pre-Norm vs. Post-Norm, and learning-rate schedules | [Assignment 1](coursework/assignment1-basics/) | ✅ |
| [Chapter 5: Mixture of Experts](docs/en/chapter5/chapter5_Mixture_of_Experts.md) | Top-K routing, load balancing, auxiliary loss, Expert Parallelism, and DeepSeekMoE | - | ✅ |
| [Chapter 6: GPUs and Related Optimizations](docs/en/chapter6/chapter6_GPU_and_Optimization.md) | Memory bandwidth, arithmetic intensity, FlashAttention, kernel fusion, mixed precision, BF16, and TF32 | [Assignment 2](coursework/assignment2-systems/) | ✅ |
| [Chapter 7: High-Performance GPU Programming](docs/en/chapter7/chapter7_GPU_High_Performance_Programming.md) | CUDA programming, Tensor Cores, shared memory, and an introduction to Triton | [Assignment 2](coursework/assignment2-systems/) | ✅ |
| [Chapter 8: Distributed Training](docs/en/chapter8/chapter8_Distributed_Training.md) | Data/model/pipeline parallelism, ZeRO-1/2/3, FSDP, gradient accumulation, and All-Reduce | [Assignment 2](coursework/assignment2-systems/) | ✅ |
| [Chapter 9: Scaling Laws](docs/en/chapter9/chapter9_Scaling_Laws.md) | Chinchilla laws, compute-optimal configurations, scaling experiment design, and extrapolation | [Assignment 3](coursework/assignment3-scaling/) | ✅ |
| [Chapter 10: Inference](docs/en/chapter10/chapter10_Inference.md) | KV cache, speculative decoding, quantization (GPTQ/AWQ), PagedAttention, and continuous batching | [Assignment 6](coursework/assignment6-evaluation/) | ✅ |
| [Chapter 11: Data Engineering](docs/en/chapter11/chapter11_Data_Engineering.md) | Data-quality filtering, MinHash deduplication, PII removal, data mixtures, and data curricula | [Assignment 4](coursework/assignment4-data/) | ✅ |
| [Chapter 12: Evaluation and Benchmarking](docs/en/chapter12/chapter12_Evaluation_and_Benchmarks.md) | MMLU, HumanEval, HELM, CEval, AlpacaEval, and Arena rankings | [Assignment 6](coursework/assignment6-evaluation/) | ✅ |
| [Chapter 13: The Basic LLM Training Pipeline](docs/en/chapter13/chapter13_Training_Pipeline.md) | Pre-training, SFT, DPO, RLHF with PPO, and the alignment tax | [Assignment 5](coursework/assignment5-alignment/) | ✅ |
| [Chapter 14: Reinforcement Learning with Verifiable Rewards](docs/en/chapter14/chapter14_RLVR.md) | GRPO, rule-based verifiers, outcome/process rewards, RLVR, and R1-style training | [Assignment 5](coursework/assignment5-alignment/) | ✅ |
| [Chapter 15: Multimodal Models](docs/en/chapter15/chapter15_Multimodal_Models.md) | From CLIP to omni models: CLIP/SigLIP, LLaVA, Qwen-VL, and Chameleon | - | ✅ |
| [Chapter 16: Extended Topics](docs/en/chapter16/) | 1. What Is LLM Reasoning? (Youzhen Li)<br>2. The Future of LLMs—LeCun (Youzhen Li, Jiang Yinhe, Shengkang Li, and Hu Xu) | - | 🔄 |

> Status: ✅ Complete · 🔄 Updating · 📝 Needs improvement · 🚧 In preparation · ⏸️ Paused

## 📝 Assignment Overview

| Assignment | Core Tasks | Status |
|------------|------------|--------|
| [Assignment 1: Build an LLM from Scratch](coursework/assignment1-basics/) | Implement a tokenizer, model architecture, and optimizer, then train a minimal language model | ✅ |
| [Assignment 2: Systems Optimization](coursework/assignment2-systems/) | Performance analysis and benchmarking; implement FlashAttention-2 in Triton; build distributed training code | ✅ |
| [Assignment 3: Scaling Laws](coursework/assignment3-scaling/) | Understand Transformer components and fit scaling laws to predict model-scaling outcomes | ✅ |
| [Assignment 4: Data Processing](coursework/assignment4-data/) | Convert raw Common Crawl data into a pre-training dataset, including filtering and deduplication | ✅ |
| [Assignment 5: Model Alignment](coursework/assignment5-alignment/) | Apply SFT and reinforcement learning methods such as GRPO to train models for mathematical reasoning | ✅ |
| [Assignment 6: Model Evaluation](coursework/assignment6-evaluation/) | Use lm-evaluation-harness and EvalScope for multidimensional evaluation across language understanding, commonsense reasoning, coding, and mathematics | ✅ |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/datawhalechina/diy-llm.git
cd diy-llm
# Install the dependencies required by the assignment you are working on
```

### Learning Path

1️⃣ **Study the theory** → Read the English material in `docs/en/` in chapter order (the Chinese version is in `docs/zh/`)  
2️⃣ **Practice** → Complete the six assignments in `coursework/`  
3️⃣ **Deepen your understanding** → Read the implementations and study the design of each component

### Project Structure

```text
diy-llm/
├── docs/                           # Online course material
│   ├── zh/                         # Chinese documentation (default)
│   │   ├── 前言.md
│   │   ├── chapter1/               # Tools
│   │   ├── chapter2/               # Tokenizers
│   │   ├── chapter3/               # PyTorch and resource accounting
│   │   ├── chapter4/               # Architecture and training details
│   │   ├── chapter5/               # Mixture of Experts
│   │   ├── chapter6/               # GPUs and optimization
│   │   ├── chapter7/               # High-performance GPU programming
│   │   ├── chapter8/               # Distributed training
│   │   ├── chapter9/               # Scaling laws
│   │   ├── chapter10/              # Inference
│   │   ├── chapter11/              # Data engineering
│   │   ├── chapter12/              # Evaluation and benchmarking
│   │   ├── chapter13/              # LLM training pipeline
│   │   ├── chapter14/              # Reinforcement learning with verifiable rewards
│   │   ├── chapter15/              # Multimodal models
│   │   └── chapter16/              # Extended topics
│   ├── en/                         # English documentation
│   │   └── ...
│   └── .vitepress/                 # VitePress configuration
├── coursework/                     # Hands-on assignments
│   ├── assignment1-basics/         # Assignment 1: Build an LLM from scratch
│   ├── assignment2-systems/        # Assignment 2: Systems optimization
│   ├── assignment3-scaling/        # Assignment 3: Scaling laws
│   ├── assignment4-data/           # Assignment 4: Pre-training data processing
│   ├── assignment5-alignment/      # Assignment 5: Alignment
│   └── assignment6-evaluation/     # Assignment 6: Evaluation
├── README.md                       # Chinese project overview
├── README_EN.md                    # English project overview
└── .gitignore                      # Git ignore rules
```

### PDF Download

A PDF edition is available for offline reading and printing. To discourage unauthorized resale with third-party watermarks, the PDF contains a subtle Datawhale open-source watermark that does not affect readability. Thank you for your understanding.

> 📥 **Diy-LLM Course Notes PDF**: [https://github.com/datawhalechina/diy-llm/releases/latest/](https://github.com/datawhalechina/diy-llm/releases/latest/)

## 🔗 Related Links

- **Repository**: https://github.com/datawhalechina/diy-llm
- **Read online**: https://datawhalechina.github.io/diy-llm/en/ ([简体中文](https://datawhalechina.github.io/diy-llm/))
- **Course notes PDF**: https://github.com/datawhalechina/diy-llm/releases/latest/
- **Original course website (current · Spring 2026)**: https://cs336.stanford.edu/
- **Original course website (archive · Spring 2025, the edition this project was first based on)**: https://cs336.stanford.edu/spring2025/
- **Original course repository**: https://github.com/stanford-cs336/lectures/tree/main

## ❓ Frequently Asked Questions

<details>
<summary><b>Q: Can I study the course without a GPU?</b></summary>

Yes. You can study all theoretical material and debug parts of the assignments on a CPU, although full training runs require a GPU. A cloud GPU service is recommended if you do not have local access to one.
</details>

<details>
<summary><b>Q: How does this project differ from the original CS336 course?</b></summary>

While preserving the technical depth of the original course, we have localized it for Chinese-speaking learners through Chinese explanations, assignment implementations, more detailed references, and examples featuring models developed in China. See "What We Changed from the Original" for a point-by-point comparison and "Version Alignment" for how the two CS336 offerings map to this project.
</details>

## 👥 Contributors

### Core Contributors

<table border="0">
  <tbody>
    <tr align="center">
      <td>
         <a href="https://github.com/xuhu0115"><img width="70" height="70" src="https://github.com/xuhu0115.png?s=40" alt="Hu Xu"></a><br>
         <a href="https://github.com/xuhu0115">Hu Xu</a>
        <p>Project Lead<br>Datawhale Member<br>Shanghai Jiao Tong University<br>Contributions: Chapters 1, 3, 9, 12, 14, and 15; Assignments 5 and 6; full-content review</p>
      </td>
      <td>
         <a href="https://github.com/kangkang-Adam"><img width="70" height="70" src="https://github.com/kangkang-Adam.png?s=40" alt="Shengkang Li"></a><br>
         <a href="https://github.com/kangkang-Adam">Shengkang Li</a>
        <p>Project Lead<br>Datawhale Member<br>Xi'an University of Posts and Telecommunications<br>Contributions: Chapters 4, 6, 7, 8, and 13; Assignments 2 and 4</p>
      </td>
      <td>
         <a href="https://github.com/1iyouzhen"><img width="70" height="70" src="https://github.com/1iyouzhen.png?s=40" alt="Youzhen Li"></a><br>
         <a href="https://github.com/1iyouzhen">Youzhen Li</a>
         <p>Project Lead<br>Datawhale Member<br>Contributions: Chapters 2, 5, 10, 11, and 13; Assignments 1 and 3</p>
      </td>
    </tr>
  </tbody>
</table>

### Community Contributions

- Thanks to [@aimetrics](https://github.com/aimetrics) for adding MPS support for MacBook devices to [Assignment 1's `train.py`](https://github.com/datawhalechina/diy-llm/blob/main/coursework/assignment1-basics/train.py).
- Thanks to [@FuTseYi](https://github.com/FuTseYi), a Datawhale member, for supporting the website migration, deployment, and refactoring.
- Thanks to [@jiangyinhe](https://github.com/jiangyinhe) for helping prepare Chapter 16's extended topic “The Future of LLMs—LeCun,” and for translating the Foreword and Chapter 1 into English from the Chinese material.
- Thanks to [@Excelius-Wang](https://github.com/Excelius-Wang) for adding an introduction to the C-EVAL Chinese benchmark and integrating it into Assignment 6.

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

See [GitHub Releases](https://github.com/datawhalechina/diy-llm/releases) for the current version and PDF notes, "Why This Project" above for the writing and review process, and "Course Outline" for per-chapter status.

## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey"></a>

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

## 🙏 Acknowledgements

- Thanks to the Stanford CS336 course team for creating the excellent original course.
- Special thanks to [@Sm1les](https://github.com/Sm1les) for helping and supporting this project.

## ⭐ Star History

If this project helps you, please consider giving it a Star ⭐️!

<a href="https://www.star-history.com/?repos=datawhalechina%2Fdiy-llm&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=datawhalechina/diy-llm&type=date&theme=dark&legend=top-left&sealed_token=ZaKMAVlJ7ht4jbVMmrIgbWlsmeGT0P-zVEivYCbfcgbt7tsA67sQo1rwoHTT5E5ajegRha9nPoc_IKk-fkvfKfddLlLONIBjPzt4QnXtwox4VKQr78nKyugDQvaziK1vjbWMPuzwwJQssk6wgPJyD2evUEO7R72ZdpHhMMABx5ZAX1uQJBG_DdDaheA1">
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=datawhalechina/diy-llm&type=date&legend=top-left&sealed_token=ZaKMAVlJ7ht4jbVMmrIgbWlsmeGT0P-zVEivYCbfcgbt7tsA67sQo1rwoHTT5E5ajegRha9nPoc_IKk-fkvfKfddLlLONIBjPzt4QnXtwox4VKQr78nKyugDQvaziK1vjbWMPuzwwJQssk6wgPJyD2evUEO7R72ZdpHhMMABx5ZAX1uQJBG_DdDaheA1">
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=datawhalechina/diy-llm&type=date&legend=top-left&sealed_token=ZaKMAVlJ7ht4jbVMmrIgbWlsmeGT0P-zVEivYCbfcgbt7tsA67sQo1rwoHTT5E5ajegRha9nPoc_IKk-fkvfKfddLlLONIBjPzt4QnXtwox4VKQr78nKyugDQvaziK1vjbWMPuzwwJQssk6wgPJyD2evUEO7R72ZdpHhMMABx5ZAX1uQJBG_DdDaheA1">
 </picture>
</a>

---

<div align="center">
  <p>Making systematic learning of large language model construction accessible to more people</p>
  <p>Made with ❤️ by Datawhale</p>
</div>
