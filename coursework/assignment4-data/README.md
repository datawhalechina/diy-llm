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
