# M3KE 中文评测接入（lm-evaluation-harness）

本案例把 M3KE 的「高中语文」子任务接入 lm-evaluation-harness，使用 multiple-choice 逐选项计算 log-likelihood，并报告 acc 与 acc_norm。M3KE 覆盖 71 个中文学科任务、共 20,477 道题；本项目已提供 C-Eval，本案例补充一个按教育阶段和学科组织的中文知识评测。

## 数据范围

M3KE 官方数据把可公开评分的标注放在 dev split；本案例只使用其中 5 条高中语文样本作为集成 smoke test。对应 test split 有 299 条题目，但 answer 字段为空，不能据此计算准确率。本案例不把 5 条 smoke test 分数解释为完整 M3KE 成绩，也不复制题库到本仓库。

运行时，prepare_m3ke.py 从 M3KE 官方仓库固定版本下载数据压缩包，校验 SHA-256 后只抽取带标签的 dev JSONL。数据集卡标注 Apache-2.0；使用时请保留 M3KE 官方仓库、数据集卡与论文来源。

## 接入文件

- m3ke_chinese_high_school_dev.yaml：lm-eval 任务配置，使用 !function 注册自定义数据加载器。
- prepare_m3ke.py：下载固定版本的官方压缩包、验证内容，并把 5 条已标注样本交给 datasets。
- data/：运行后生成的本地数据缓存目录，不提交题目内容。

## 运行

从 diy-llm 仓库根目录执行：

~~~sh
python -m pip install "lm_eval==0.4.13"
python coursework/assignment6-evaluation/m3ke/prepare_m3ke.py
lm-eval validate --include_path ./coursework/assignment6-evaluation/m3ke --tasks m3ke_chinese_high_school_dev
lm-eval run --model dummy --tasks m3ke_chinese_high_school_dev --include_path ./coursework/assignment6-evaluation/m3ke --num_fewshot 0 --limit 5 --seed 1234,1234,1234,1234 --output_path ./coursework/assignment6-evaluation/outputs/m3ke-dummy-smoke
~~~

dummy 是 lm-eval 自带的随机打分模型，只用于验证任务配置、数据加载和指标汇总。示例结果会明确标成随机基线，不代表任何语言模型的能力。要评估实际模型，可安装 hf 后端并替换模型参数，例如：

~~~sh
python -m pip install "lm_eval[hf]==0.4.13"
lm-eval run --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B,dtype=float32 --tasks m3ke_chinese_high_school_dev --include_path ./coursework/assignment6-evaluation/m3ke --num_fewshot 0 --limit 5 --device cpu --batch_size 1 --output_path ./coursework/assignment6-evaluation/outputs/m3ke-qwen25-0.5b-smoke
~~~

实际模型命令需要下载模型权重；正式比较应固定模型 revision、运行环境、样本范围和 few-shot 设置，并使用完整且有标签的评测集。

## 样例结果

在 Python 3.12、lm-eval 0.4.13、datasets 5.1.0 下运行固定随机种子的 5 条零样本 smoke test：

| 任务 | 模型 | 样本数 | acc | acc stderr | acc_norm | acc_norm stderr |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M3KE 高中语文 dev | lm-eval `dummy` 随机基线 | 5 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |

这组分数只表示任务可加载、选项似然可计算、指标可汇总；随机模型和 5 条样本都不能用于模型能力比较。可复现元数据见 [sample_results.json](./sample_results.json)。

## 来源与引用

- M3KE 官方仓库：https://github.com/tjunlp-lab/M3KE
- Hugging Face 数据集卡：https://huggingface.co/datasets/TJUNLP/M3KE
- 论文：https://arxiv.org/abs/2305.10263

请按论文引用 M3KE，并遵守数据集卡中的许可与使用说明。
