# DIY-LLM 作业常见报错与踩坑 FAQ

本文按「现象—原因—处理」整理六项作业的常见问题。请先确认自己在对应作业目录中，并使用该作业 README 指定的环境；不同作业的 Python、PyTorch 和依赖版本并不相同。

## Assignment 1：手搓大模型

### Q1.1 训练或生成时报 <code>FileNotFoundError</code>，或没有读到预期语料，怎么办？

**现象：** 找不到 <code>data.bin</code>、<code>TinyStoriesV2-GPT4-train.txt</code> 或 <code>bpe_tokenizer/tokenizer.json</code>；训练能启动但结果异常。

**原因：** 训练语料和通过 BPE 作业生成的 tokenizer 文件不是都随仓库提供。另一个容易忽略的情况是，<code>train.py</code> 在没有 <code>data.bin</code> 时会生成模拟数据，因此「训练跑完」不代表已经使用了真实语料。

**处理：** 按[作业 1 README](./assignment1-basics/README.md)先完成 BPE notebook，下载 README 指向的 TinyStories 训练文本，再在 <code>assignment1-basics</code> 目录运行 <code>python get_train_data.py</code>。训练前确认 <code>data.bin</code> 和 tokenizer 路径存在；若使用模拟数据，请明确把它当作流程调试数据。

### Q1.2 训练成功，但 <code>model.py</code> 加载 checkpoint 时出现尺寸不匹配，或生成乱码，怎么办？

**现象：** 加载权重时报 tensor size mismatch；或者能生成文本，但分词和输出不正常。

**原因：** 生成脚本初始化的模型结构或词表，与训练 checkpoint 保存时使用的配置不同。

**处理：** 对照[作业 1 README 的参数表](./assignment1-basics/README.md)，让 <code>d_model</code>、<code>num_heads</code>、<code>num_layers</code>、<code>context_length</code>、<code>vocab_size</code> 等配置和训练时一致，并确认使用的是同一份 tokenizer。不要只改生成脚本的模型参数来「试到能加载」。

## Assignment 2：系统优化

### Q2.1 安装依赖时报 Python 版本或 PyTorch 依赖冲突，怎么办？

**现象：** <code>uv sync</code> 提示 Python 版本不满足，或解析出的 PyTorch/相关依赖无法安装。

**原因：** 作业 2 的 [<code>pyproject.toml</code>](./assignment2-systems/pyproject.toml)要求 Python <code>>=3.12</code>，并声明 PyTorch <code>>=2.9.1</code>；这与其他作业的环境约束不同。

**处理：** 在 <code>coursework/assignment2-systems</code> 目录单独创建并同步环境，使用 Python 3.12 或更高版本，再运行 <code>uv sync</code>。先用 <code>python --version</code> 和 <code>uv run python -c "import torch; print(torch.__version__)"</code> 确认实际使用的解释器及 PyTorch 版本。

### Q2.2 Attention benchmark 的显存峰值看起来包含 warmup，怎么办？

**现象：** 正式测量读到的峰值显存高于测量段本身预期。

**原因：** [Issue #57](https://github.com/datawhalechina/diy-llm/issues/57)指出，CUDA 的 peak memory 读数会保留进程启动或上次重置以来的历史峰值；warmup 已经分配过显存时，之后读数可能仍包含 warmup。

**处理：** warmup 后、正式测量前先执行 <code>torch.cuda.synchronize()</code> 和 <code>torch.cuda.reset_peak_memory_stats()</code>；测量段结束后再次同步，再读取峰值。若要报告测量段新增显存，记录测量前的已分配显存并说明计算口径。

## Assignment 3：扩展定律

### Q3.1 运行训练或拟合脚本时找不到 <code>data.bin</code>，怎么办？

**现象：** 训练提示数据文件不存在，或使用了意外的数据。

**原因：** 作业 3 的数据文件由作业 1 的数据处理流程生成，并按 README 中的相对路径读取；从仓库根目录启动脚本也可能让相对路径指向错误位置。

**处理：** 按[作业 3 README](./assignment3-scaling/README.md)准备训练数据和 tokenizer，确认作业目录下的 <code>data.bin</code> 与脚本配置路径一致；在作业目录启动命令。不要把作业 1 的模拟数据结果当作正式缩放实验数据。

### Q3.2 缩放定律预测与实测 loss 差距很大，代表拟合程序出错了吗？

**现象：** 预测值和实际训练结果偏差明显，或不同模型配置得到的拟合结果不能相互解释。

**原因：** README 说明当前预测主要基于非嵌入参数量和训练 token 数，并未覆盖其他超参数；模型未充分收敛、数据分布不同、预算区间过窄，也会使拟合失去可比性。

**处理：** 先确认训练记录已生成且包含多个有效预算点，再核对参数量口径、token 数、数据分布及训练是否收敛。只在架构和训练设置可比的范围内解释拟合；不要把单次拟合外推当作性能保证。详见[作业 3 README](./assignment3-scaling/README.md)。

## Assignment 4：数据处理

### Q4.1 运行 <code>get_assets.sh</code> 时提示目录不存在或下载失败，怎么办？

**现象：** <code>wget: No such file or directory</code>，或脚本运行后分类器权重没有出现在预期位置。

**原因：** [脚本](./assignment4-data/get_assets.sh)使用当前工作目录拼出 <code>cs336_data/assets</code>，但不会自动创建该目录；并且优先尝试 <code>/data/classifiers</code>，找不到时才从 Hugging Face 下载。

**处理：** 先进入 <code>coursework/assignment4-data</code>，运行 <code>mkdir -p cs336_data/assets</code>，再运行 <code>bash get_assets.sh</code>。确认以下两个文件存在且不是空文件：<code>dolma_fasttext_nsfw_jigsaw_model.bin</code>、<code>dolma_fasttext_hatespeech_jigsaw_model.bin</code>。若下载源不可达，检查网络或使用课程环境提供的 <code>/data/classifiers</code> 文件。

### Q4.2 安装后仍然 <code>ModuleNotFoundError: cs336_data</code> 或 <code>cs336_basics</code>，怎么办？

**现象：** 作业代码无法导入本地模块，或 <code>uv sync</code> 找不到 <code>cs336-basics</code>。

**原因：** [作业 4 配置](./assignment4-data/pyproject.toml)把 <code>cs336-basics</code> 声明为相对路径 <code>./cs336-basics</code> 的本地 editable dependency；从仓库根目录执行会改变相对路径基准。

**处理：** 确认 Python 满足 <code>>=3.11</code>，进入 <code>coursework/assignment4-data</code> 后运行 <code>uv sync</code>，再用 <code>uv run python</code> 或 <code>uv run pytest</code> 执行作业命令。该作业还限定 <code>numpy<2.0</code>，不要在未同步的全局 Python 环境里混用依赖。

## Assignment 5：模型对齐

### Q5.1 <code>flash-attn</code> 安装或导入失败，怎么办？

**现象：** <code>uv sync</code> 在构建 <code>flash-attn</code> 时失败，或运行时提示找不到 <code>flash_attn</code>。

**原因：** 作业 5 依赖包含 GPU 扩展的 <code>flash-attn</code>，直接一次安装可能在本机 CUDA / PyTorch / 编译环境不匹配时失败。

**处理：** 按[作业 5 README](./assignment5-alignment/README.md)使用 Python <code>>=3.11,&lt;3.13</code>，先运行 <code>uv sync --no-install-package flash-attn</code>，再运行 <code>uv sync</code>。若仍失败，先核对 PyTorch、CUDA、GPU 与编译器环境是否匹配，不要把绕过安装当作完整实验环境。

### Q5.2 模型加载时报路径不存在，或 SFT/GRPO 很快显存不足，怎么办？

**现象：** 报本地模型目录不存在，或启动 vLLM / 完整训练后出现 CUDA out of memory。

**原因：** README 中的默认 <code>MODEL_PATH</code> 是作者机器上的本地路径，并非仓库内模型；完整 SFT/GRPO 实验也需要较大显存。

**处理：** 在相关脚本中把 <code>MODEL_PATH</code>、<code>BASE_MODEL</code> 或 <code>GRPOConfig.base_model</code> 改成自己可访问的模型目录。完整实验按 README 的资源建议准备 GPU（至少 80 GB，完整流程更建议 2 张 80 GB）；CPU / Mac 环境适合小规模逻辑检查，不适合作为 vLLM 完整实验的替代。

## Assignment 6：模型评测

### Q6.1 命令提示找不到 <code>lm-eval</code>，或任务名无法识别，怎么办？

**现象：** 运行 <code>lm-eval validate</code> 提示命令不存在，或提示 <code>ceval-valid_logic</code> 不是可用任务。

**原因：** <code>lm-evaluation-harness</code> 没有安装在当前激活的 Python 环境，或任务配置和安装版本不一致。

**处理：** 按[作业 6 README](./assignment6-evaluation/README.md)安装并激活对应环境，在同一个环境中安装 <code>lm-evaluation-harness</code>，先运行 <code>lm-eval validate --tasks ceval-valid_logic</code> 验证任务配置，再启动评测。若 README 中的 <code>lm-evaluation-harness</code> 目录不存在，先获取该框架源码或按其官方安装步骤安装，不要在仓库根目录执行 <code>pip install -e .</code>。

### Q6.2 CPU 评测内存不足，或 <code>--limit 10</code> 的结果和完整成绩差很多，怎么办？

**现象：** 评测时内存不足、速度很慢；小样本分数与完整 C-Eval 结果差异明显。

**原因：** 评测脚本同时加载模型和任务数据；README 标明 <code>--limit 10</code> 是验证链路的 smoke test，不代表完整评测结果。

**处理：** CPU 内存不足时把 <code>--batch_size</code> 从 8 降到 1。正式评测按 README 运行 <code>ceval-valid</code> 全部任务并移除 <code>--limit</code>；记录模型、数据 split、样本数、few-shot 设置、设备与 dtype，比较时保持设置一致。

## 参考入口

- [作业 1：手搓大模型](./assignment1-basics/README.md)
- [作业 2：系统优化](./assignment2-systems/README.md)
- [作业 3：扩展定律](./assignment3-scaling/README.md)
- [作业 4：数据处理](./assignment4-data/README.md)
- [作业 5：模型对齐](./assignment5-alignment/README.md)
- [作业 6：模型评测](./assignment6-evaluation/README.md)
