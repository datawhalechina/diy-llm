# CS336 2025 年春季 作业 4：数据
有关本次作业的完整说明，请参阅作业讲义：
[cs336_spring2025_assignment4_data.pdf](./相关文档/cs336_spring2025_assignment4_data.pdf)

如果你在作业讲义或代码中发现任何问题，欢迎在 GitHub 上提交 issue，或通过 pull request 提供修复。

## Setup

本目录的组织结构如下：

* [`./相关文档`](./相关文档)：包含有关本节课的文档
* [`./cs336-basics`](./cs336-basics)：包含一个名为
  `cs336_basics` 的模块及其对应的 `pyproject.toml` 文件。该模块中包含了作业 1 中语言模型的助教实现版本。你将使用这套训练代码，在你过滤后的数据上训练语言模型。你不应修改训练逻辑，因为排行榜提交必须**完全使用该实现**。
* [`./cs336_data`](./cs336_data)：该文件夹基本为空！这是你将要实现数据过滤和处理代码的模块。

从结构上看，应大致如下所示：

```sh
.
├── cs336_basics  # 一个名为 cs336_basics 的 Python 模块
│   └── ... 一个经过优化的训练实现 ...
├── cs336_data  # TODO(你)：为作业 4 编写的代码
│   ├── __init__.py
│   └── ... TODO(你)：作业 4 所需的其他文件或文件夹 ...
├── README.md
├── pyproject.toml
└── ... TODO(你)：作业 4 所需的其他文件或文件夹 ...
```

与之前的作业相同，我们使用 `uv` 来管理依赖。

其中作业的全部流程都在`assignment4-data/cs336_data/作业一.ipynb`和 `assignment4-data/cs336_data/作业二.ipynb`，里面有讲解和代码，其中我们供读者跑通基础作业：

只需运行`cs336_systems/作业1.ipynb`和`cs336_systems/作业2.ipynb`就可以跑通流程，其他文件都是这两个文件生成的，不需要理会。

## 可复现的 WET 过滤与去重小案例

`cs336_data` 中提供了一个面向小规模数据的完整示例：读取 Common Crawl 的 WET
`conversion` 记录，依次进行文本归一化、语言专用质量过滤、PII 脱敏、精确去重和
MinHash/LSH 近重复去重。通过 `--language en|zh` 选择规则配置；处理过程受
`--max-records` 限制，适合先在本地验证每个环节，最终输出的 `stats.json` 会记录各阶段
的输入和输出数量。

英文 profile 复用 assignment4 的 Gopher 规则。中文 profile 采用
[ChineseWebText EvalWeb](https://github.com/CASIA-LM/ChineseWebText) 公开的规则子集：
总长度、非空行的平均长度、中文字符比例和文档内 13-gram 重复率；随后使用仓库内可审计的
中文色情词表做硬过滤。词表来自 [SpaceGather/worldwide-sensitive-word-collection](https://github.com/SpaceGather/worldwide-sensitive-word-collection)
的 `zh-CN/pornography.csv`，按 MIT 许可随代码分发，许可文本见
[`cs336_data/THIRD_PARTY_NOTICES.md`](./cs336_data/THIRD_PARTY_NOTICES.md)。默认命中任意一个词
即拒绝文档，这是在 ChineseWebText “每个非空行敏感词数超过 0.5”规则上的保守页级 guard；可用
`--chinese-sensitive-words`、`--chinese-max-sensitive-words-per-line` 和
`--chinese-min-sensitive-terms` 替换词表或调整阈值。繁简转换和 BERT/FastText 评分仍未引入，
避免新增模型依赖。

下面的命令使用固定的 Common Crawl WET 文件，不需要把原始数据提交到仓库：

```sh
cd coursework/assignment4-data
uv sync
mkdir -p data
WET_URL="https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-51/segments/1764871306713.64/wet/CC-MAIN-20251204191828-20251204221828-00000.warc.wet.gz"
curl -L --fail --retry 3 -o data/CC-MAIN-20251204191828-20251204221828-00000.warc.wet.gz "$WET_URL"
echo "fca9d05b20facc17f7baa6cf9086be19bc07e7b05c8afc266766a16dd3c9cd03  data/CC-MAIN-20251204191828-20251204221828-00000.warc.wet.gz" | sha256sum -c -
uv run python scripts/run_small_pipeline.py \
  --input data/CC-MAIN-20251204191828-20251204221828-00000.warc.wet.gz \
  --output-dir outputs/wet-demo-en \
  --language en \
  --max-records 1000
uv run python scripts/run_small_pipeline.py \
  --input data/CC-MAIN-20251204191828-20251204221828-00000.warc.wet.gz \
  --output-dir outputs/wet-demo-zh \
  --language zh \
  --max-records 1000
```

每个输出目录都会生成两个文件：

* `filtered.jsonl`：保留的文档，包含来源 URL、脱敏后的文本和
  SHA-256 摘要；
* `stats.json`：输入、过滤、精确去重、近重复去重各阶段的计数，
  以及本次运行使用的参数。

在一台普通 CPU 环境中使用上述文件和默认参数（`--max-records 1000`）的英文结果如下，
耗时会随机器和磁盘变化：

| 指标 | 数量 |
| --- | ---: |
| 输入 conversion 记录 | 1000 |
| 空文本 | 0 |
| 长度过短 / 过长 | 22 / 16 |
| Gopher 过滤拒绝 | 675 |
| 进入去重 | 287 |
| 精确重复 | 3 |
| 精确去重后 | 284 |
| MinHash 候选对 | 20 |
| MinHash 近重复 | 3 |
| 最终输出 | 281 |

本次还屏蔽了 103 个邮箱、58 个电话号码和 3 个 IPv4 地址，脚本耗时约 48 秒。
这些数字来自 `stats.json`，不是硬编码在实现中的预期值；更换 WET 文件或参数后应以
新生成的统计为准。

同一份 WET 文件使用中文 profile 的一次结果如下：

| 指标 | 数量 |
| --- | ---: |
| 输入 conversion 记录 | 1000 |
| 空文本 | 0 |
| 长度过长 | 16 |
| 中文长度过短 | 81 |
| 平均行长过短 | 27 |
| 中文字符比例过低 | 646 |
| 敏感内容 | 120 |
| 文档内重复率过高 | 4 |
| 进入去重 | 106 |
| 精确重复 | 1 |
| 精确去重后 | 105 |
| MinHash 候选对 | 3 |
| MinHash 近重复 | 0 |
| 最终输出 | 105 |

中文样例还屏蔽了 28 个邮箱和 79 个电话号码，耗时约 12 秒。加入敏感内容层后，进入去重的
记录由 224 条降到 106 条，样例中的词表命中记录不再进入输出；中文网页规则仍然是启发式的，
统计结果应结合抽样检查解读，不应直接当作高质量中文语料的标注结果。

其中 MinHash 使用稳定的 BLAKE2b 哈希，LSH 只负责产生候选对，候选对还会经过真实
Jaccard 相似度复核；因此同一输入和参数在不同运行中会得到相同的保留顺序。需要处理
更大规模数据时，可逐步提高 `--max-records`，并将中间结果分片后再执行去重。
