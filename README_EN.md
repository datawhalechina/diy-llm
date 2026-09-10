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
  <p><a href="https://datawhalechina.github.io/diy-llm/en/">📚 Read Online</a></p>
  <h3>📚 A Systematic Journey into Large Language Models</h3>
  <p><em>An “LLM alchemy workshop” built for Chinese-speaking learners</em></p>
</div>

We want this Chinese adaptation of CS336 to be more than a translated version of Stanford's original course. Our goal is to build an “LLM alchemy workshop” tailored to Chinese-speaking learners—a place where you forge your understanding, refine your code, master the training process, and ultimately create a large language model of your own.

## 📋 Prerequisites

- **Python programming**: Proficiency in Python and solid software engineering skills
- **Deep learning fundamentals**: Familiarity with PyTorch and the basic principles of neural networks
- **Mathematics**: Linear algebra, probability and statistics, and calculus
- **Machine learning**: A solid grasp of the fundamentals of machine learning and deep learning
- **GPU programming (optional)**: Familiarity with basic CUDA concepts is helpful but not required; this project also includes introductory material

## 📚 Course Vision

- **A balance of rigorous theory and hands-on practice**: We retain the technical depth of the original course while reorganizing the material around the learning habits of Chinese-speaking students. We also cover essential mathematical and deep learning prerequisites to provide a smoother learning curve.
- **A progressive knowledge system**: The large engineering challenge of building an LLM is broken down into approachable, understandable modules. By the end of the course, you will have a complete mental map of LLM construction.
- **Code-driven learning**: The core philosophy of the course is to “think through code.” In addition to implementations for every assignment, we share the reasoning behind the code.
- **Localization for the Chinese ecosystem**: We account for local network conditions, available computing resources, and the domestic open-source ecosystem, with practical solutions and examples involving excellent models such as Qwen and DeepSeek.

## 🎯 What You Will Gain

What can you expect after completing the course?

- **A solid technical foundation**: You will be able to build your own LLM and understand every core component.
- **Valuable engineering experience**: You will practice the full workflow—from data processing and model training to deployment and optimization.
- **Stronger industry competitiveness**: You will develop the core skills required for large-model research and engineering roles.
- **A clear research perspective**: You will gain a systematic understanding of the LLM field and a strong foundation for further research.

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
- **Read online**: https://datawhalechina.github.io/diy-llm/
- **Original course website**: [Stanford CS336 (Spring 2026)](https://cs336.stanford.edu/)
- **Original course repository**: https://github.com/stanford-cs336/lectures/tree/main

## ❓ Frequently Asked Questions

<details>
<summary><b>Q: Can I study the course without a GPU?</b></summary>

Yes. You can study all theoretical material and debug parts of the assignments on a CPU, although full training runs require a GPU. A cloud GPU service is recommended if you do not have local access to one.
</details>

<details>
<summary><b>Q: How does this project differ from the original CS336 course?</b></summary>

While preserving the technical depth of the original course, we have localized it for Chinese-speaking learners through Chinese explanations, assignment implementations, more detailed references, and examples featuring models developed in China.
</details>

## 💬 Reader Community

Join the Diy-LLM reader groups to study, exchange ideas, and help one another. If you would like to join, please add one of the coordinators below on WeChat and include “diy-llm” in your verification message.

| Coordinator | WeChat ID |
|-------------|-----------|
| Hu Xu | `xuhu96736` |
| Youzhen Li | `zydsx111` |
| Shengkang Li | `muzichengminguangli` |

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

- Thanks to [@aimetrics](https://github.com/aimetrics) for adding MPS support for MacBook devices to [Assignment 1's `train.py`](https://github.com/datawhalechina/diy-llm/blob/main/coursework/assignment1-basics/train.py).
- Thanks to [@FuTseYi](https://github.com/FuTseYi), a Datawhale member, for supporting the website migration, deployment, and refactoring.
- Thanks to [@jiangyinhe](https://github.com/jiangyinhe) for helping prepare Chapter 16's extended topic “The Future of LLMs—LeCun,” and for translating the Foreword and Chapter 1 into English from the Chinese material.

*We appreciate every developer who has contributed to this project!*

Contributions of all kinds are welcome. Improvements to the documentation, code optimizations, bug fixes, and new content are all valuable to the project.

### How to Contribute

1. **Report issues**: If you find documentation errors, bugs, or opportunities for improvement, please open an [Issue](https://github.com/datawhalechina/diy-llm/issues).
2. **Submit code**: Fork the repository, create a feature branch, commit your changes, and open a Pull Request.
3. **Improve the documentation**: Help refine existing material, translate content, or add examples.
4. **Share your experience**: Post your learning notes and practical experience in the discussions area.

### Contribution Guidelines

- Keep the code style consistent with the existing project.
- Follow the current documentation format when adding new content.
- Provide a clear description and change summary in each Pull Request.
- For substantial changes, discussion in an Issue is encouraged before implementation.

## 📝 Changelog

The project is under continuous development. See [GitHub Releases](https://github.com/datawhalechina/diy-llm/releases) or the commit history for the latest updates.

## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey"></a>

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

## 🙏 Acknowledgements

- Thanks to the Stanford CS336 course team for creating the excellent original course.
- Special thanks to [@Sm1les](https://github.com/Sm1les) for helping and supporting this project.
- Thanks to every developer who has contributed to the project.
- Thanks to the open-source community for its support and feedback.

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
