# CS336 Assignment 6: 大型语言模型评测框架介绍

本文件夹包含CS336课程第六次作业的内容，主要介绍常用的几种大型语言模型评测框架及其使用方法。

## 📁 文件夹结构

```
assignment6_evaluation/
├── README.ipynb                  # 主要的演示notebook
├── lm_eval_demo.py              # lm-evaluation-harness 极简实现脚本
├── evalscope_demo.py            # evalscope 极简实现脚本
├── data/                        # 数据文件夹
│   └── index_testset.jsonl      # evalscope生成的评测数据集
├── outputs/                     # 评测输出结果
│   ├── 20260119_232050/         
│   └── 20260120_000654/         
├── images/                      # 图片
│   └── evalscope_panel.png      
└── README.md                    # 本文件
```

## 🎯 作业目标

介绍常用的几种大型语言模型评测框架：
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - 学术界标准评测框架
- [evalscope](https://github.com/modelscope/evalscope) - 支持自定义数据集组合和可视化分析
- [Evalchemy](https://github.com/mlfoundations/evalchemy) - 轻量级评测框架
- [lighteval](https://github.com/huggingface/lighteval) - Hugging Face生态集成

重点演示**lm-evaluation-harness**和**evalscope**的使用方法。

## 📊 评测框架对比

| 框架名称 | 开发机构 | 主要特点 | 适用场景 |
|---------|---------|---------|---------|
| lm-evaluation-harness | EleutherAI | 功能丰富，支持多种模型和任务，学术界标准 | 学术研究、基准测试 |
| evalscope | ModelScope | 支持自定义数据集组合，可视化分析，中文友好 | 产业应用、模型评估 |
| Evalchemy | ML Foundations | 轻量级，注重可复现性和扩展性 | 研究实验、快速原型 |
| lighteval | Hugging Face | 集成Transformers生态，易于使用 | Hugging Face用户 |

## 🔧 主要内容

### lm-evaluation-harness

- **零样本评测**：arc_easy, piqa, lambada, triviaqa
- **中文知识评测**：C-Eval（`ceval-valid`，示例使用 `ceval-valid_logic`）
- **少样本评测**：humaneval, mbpp, gsm8k, minerva_math
- **多维度能力评测**：通用语言理解、常识推理、中文知识、代码、数学推理

### evalscope

- **自定义数据集组合**：通过CollectionSchema定义评测索引
- **加权采样**：根据业务需求调整数据集权重
- **可视化分析**：通过Web界面分析评测结果详情


## 🚀 快速开始

### 环境准备

```bash
# 1. 创建并激活虚拟环境（也可以使用 Conda）
python -m venv .venv
source .venv/bin/activate

# 2. 安装 lm-evaluation-harness、Hugging Face 后端和数学任务依赖
python -m pip install "lm_eval[hf,math]"

# 3. 安装evalscope
pip install evalscope
pip install 'evalscope[app]' -U  # 可视化依赖
```

### 开始学习

进入 `README.ipynb` 跟着学习两种框架的简单使用，更详细内容可参考框架的指导手册。

如果你想直接运行：

#### 1. lm-evaluation-harness演示

```bash
# Python脚本方式
python lm_eval_demo.py
```

脚本会复用同一个模型实例运行英文任务和一个 C-Eval 中文任务。示例中的 C-Eval 任务限制为 1 条样本，以便快速检查接入是否正确；需要正式评测时请使用下面的命令并去掉 `--limit`。

#### 2. C-Eval 中文 benchmark

C-Eval 是覆盖 52 个学科的中文多项选择评测集。lm-evaluation-harness 已提供 `ceval-valid` 任务组及各学科子任务，数据会从 Hugging Face 的 `ceval/ceval-exam` 自动加载。这里使用验证集任务 `ceval-valid_logic` 做一个 10 条样本的 CPU smoke test：

```bash
# 检查任务配置和数据字段
lm-eval validate --tasks ceval-valid_logic

# 运行 C-Eval 逻辑学子集（零样本，10 条样本）
lm-eval run \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-0.5B,dtype=float32 \
  --tasks ceval-valid_logic \
  --num_fewshot 0 \
  --limit 10 \
  --device cpu \
  --batch_size 8 \
  --output_path ./outputs/ceval-qwen25-0.5b-logic
```

一次实际 smoke test（`lm_eval 0.4.13`、CPU、零样本、10 条验证集样本）输出如下，分数仅用于确认链路可运行：

| 任务 | 样本数 | acc | acc_norm |
| --- | ---: | ---: | ---: |
| `ceval-valid_logic` | 10 | 0.3000 | 0.3000 |

同一环境下完整运行 `ceval-valid` 任务组（52 个学科、1,346 条验证集样本，耗时约 5 分钟）得到：

| 任务组 | 学科数 | 样本数 | acc | acc_norm |
| --- | ---: | ---: | ---: | ---: |
| `ceval-valid` | 52 | 1,346 | 0.5134 | 0.5134 |

该命令的完整 JSON 结果会写入 `--output_path` 指定的目录；正式对比时应固定模型、任务版本、样本范围和 few-shot 设置，并运行完整任务组。

CPU 上可先使用 `--batch_size 8` 提高吞吐；如果内存不足，将其降为 `1`。不建议为每个学科启动独立进程，以免重复加载模型并造成线程争用。

评测结果至少应记录 `acc`、`acc_norm`、样本数、模型版本、`num_fewshot`、`limit`、设备和 dtype。`--limit 10` 的结果只用于验证流程，不能当作完整 C-Eval 成绩；本地正式复现可将任务改为 `ceval-valid` 并移除 `--limit`。lm-evaluation-harness 的内置 C-Eval 任务只评估 `val` split，测试集成绩需按 C-Eval 官方流程提交预测，不能直接与这里的验证集分数混用。

上表是 C-Eval 验证集的复现结果，不代表公开测试集或其他模型的排行榜成绩。

C-Eval 题库采用 CC BY-NC-SA 4.0 许可，使用时请保留[官方仓库](https://github.com/hkust-nlp/ceval)、[数据集](https://huggingface.co/datasets/ceval/ceval-exam)和论文引用信息，不要将题库复制进本仓库。

#### 3. evalscope演示

```python
# 运行evalscope演示
python evalscope_demo.py
```

## 📝 使用建议

- **学术研究**：推荐使用 `lm-evaluation-harness`
- **产业应用**：推荐使用 `evalscope`
- **快速原型**：推荐使用 `Evalchemy`
- **Hugging Face用户**：推荐使用 `lighteval`

## 📞 更多内容

如有问题，请参考：
- [lm-evaluation-harness文档](https://github.com/EleutherAI/lm-evaluation-harness)
- [evalscope文档](https://github.com/modelscope/evalscope)
