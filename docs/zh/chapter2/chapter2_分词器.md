# 分词器

&emsp;&emsp;分词器（tokenizer）常被视为大语言模型（LLM）的一部分，但实际上它有着相对独立的设计与训练流程。

&emsp;&emsp;一个直观的理解方式是：**模型并不直接“读懂文字”，它只能处理数字**。而分词器做的事情，就是把原始文本切分成一个个词元（token），并将这些片段映射为对应的数字ID，供模型后续计算使用。可以用一个例子来理解：

模型的工作方式可以类比为一条“自动化快递处理流水线”，其中的tokenizer更像是一个**带有经验规则的智能分拣与打包中心**。我们可以想象这样一个场景——

&emsp;&emsp;我们需要寄一个混合物品的“快递”，即是一堆经常重复出现的组合（就像语言中反复出现的词、词根、子词），但“快递公司”的运输系统（大语言模型）有一个限制——不理解具体物品，只认识 ID编号 ，而且更倾向于处理出现的标准物件（token），于是，在进入运输系统之前，会有一个更“聪明”的分拣中心（tokenizer）把这些**重复组合拆解并转换成系统唯一识别的ID序列**：

- **拆分物品（文本切分）：** 不是随便乱拆，而是尝试找到“经常一起出现的组合”，比如“火锅”不会拆成“火 + 锅”，因为它作为整体更常见；
- **复用已有包装（子词建模）：** 如果某个大件不常见，就拆成常见的小件组合，比如“超导材料”可能拆成“超导 + 材料”；
- **动态决定颗粒度（关键点）：** `常见的组合 → 打包成一个整体，罕见的组合 → 拆成多个小块`，这其实就是BPE、WordPiece等tokenizer在做的事情；      
- **统一编号（token ID）：** 每种“标准件”都会被分配一个固定编号；
- **输出编号序列（token序列）：** 最终交给模型处理的是一串编号，而不是原始文本。

**💡这个例子背后的关键点是：**

- **分词算法不是唯一的**：不同的分词算法（如BPE、WordPiece）会出现不同的划分结果；
- **分词器会影响模型能力**：token划分得太“碎片化”或太“笼统”，都会影响表达效率和泛化能力；
- **它是独立优化的模块**：通常先在大规模文本上训练词表（vocab），再**固定下来供模型使用**。


*📖Tips：以划分`incredible`为例——`['i','n','cr','e','dible']`就过于”碎片化“了，`['incredible']`过于”笼统“，理想的划分是`['in','credible']`这样能在不同单词中复用，既控制了token总数，又提升了embedding的映射能力。*

>一句话总结——**LLM负责“理解和生成”，而tokenizer负责“把语言变成模型能理解、可复用的结构”。**



<div align="center">
   <img width="1000" height="600" alt="1" src="https://github.com/user-attachments/assets/bc838c76-7eff-4479-a760-ef404fc48e89" />
   <p>图2.1 分词器与LLM</p>
 </div>

>需要注意的是，这一过程不仅是简单的编码转换，更隐含了一种模型对“世界的切分“策略：模型最终“看到”的基本单位，可能是字符、词，或更常见的子词片段编码。这种表示方式将直接影响序列长度、信息密度以及语义的组合方式，从而进一步影响模型的训练效率、表现能力。

---

# 本节学习目标

在进入具体分析之前，我们先明确本节的关注重点。本节将围绕分词器这一看似简单、但对LLM行为有深远影响的模块展开，主要包括：

1. [理解4个分词算法（如BPE、WordPiece...）的基本原理与训练流程](#21-训练分词器)
2. [分析基于不同分词算法实现tokenizer的区别，以及BPE流行的原因](#22-常用的分词器)
3. [结合实际工具（如DeepSeek的相关实现）动手使用分词器](#23-分析deepseek的分词器)


相比“如何使用”，本节更关注一个更根本的问题：

> **分词器究竟在解决什么问题？我们又该如何设计一个*好的*分词器？**

**围绕这一核心问题，可以从以下几个层面逐步展开：**

- 分词器如何从数据中被“学习”出来，不同分词算法（如BPE、Unigram LM...）的核心思想是什么？
- 不同分词策略（不同的切分方式）如何影响模型的能力，并进一步影响语义建模与组合能力？
- 分词器在实际工程中如何落地？

*在学习开始之前，我们可以先来感受一下👉[分词器](https://tiktokenizer.vercel.app/?model=gpt-4-1106-preview)。接着我们带着这几个问题，开始进入分词器的学习！*

## 2.1 训练分词器

虽然分词器常被当作模型的一部分，但它其实是提前单独准备好的：先对原始文本做一些清洗和整理，然后通过统计方法或子词规则，总结出一套`词元 → 数字编号`对应表（也就是词表，vocab）。之后，任何输入文本都会按照这张表，被转换成一串不再是人类能读的文字，而是一段段离散的数字序列。

```
训练一个用于现代大型语言模型的分词器可以拆成四步：准备语料 → 初始化基础单元(不同的分词算法，可省略) → 统计并迭代合并 → 输出产物并用于编码、解码。
```

<div align="center">
   
![1](https://github.com/user-attachments/assets/14b349a5-167d-4f66-825b-2951e87d0dc6)

   <p>图2.2 分词器训练流程</p>
 </div>
 
### 2.1.1 准备语料

**Step1** &emsp;在准备语料阶段，应尽量收集覆盖目标应用场景的多样化文本，以便训练出的词表对下游任务具有良好泛化能力：
   
   - 准备不同类型的文本信息比如小说、散文、诗歌等不同描述风格的信息；
   - 多种语言的文本信息比如中文、英文、韩语、法语等。

多语言或混合语料场景中，应统计各语言占比，并评估是否对低资源语言进行过采样或定向保留，避免词表被高频语言主导。否则，语料类型与语言分布不均衡会加剧低资源语言的`token碎片化`，可能会降低其任务性能。例如准备一种可以支持四种语言的分词器，这里假设提前收集到的原始未经过处理的各语言原始语料占比如下：
   
| 语言 | 语料量 |
| :--- | ---: |
| 中文 | 200 GB |
| 英文 | 150 GB |
| 法语 | 10 GB |
| 韩文 | 5 GB |

>这是一个典型的多语言语料不平衡场景。若将上述语料不经处理直接混合训练分词器，其统计过程会被中文和英文主导，导致法语与韩文的常见字串在合并阶段难以进入高频统计，从而无法占据足够的词表空间，最终在`vocab`中会出现大量被切得过碎的`token`，形成严重碎片化，下游LLM在法语与韩文任务上会因此表现显著劣化。

&emsp;&emsp;因此，在准备语料时，需要先按语言统计语料占比，并根据目标能力设定合理的采样策略。例如将语料比例调整为`中文:英文:法语:韩文=4:4:1:1`或者`采用完全均衡策略`。通过对高资源语言下采样或对低资源语言过采样、增强，可以获得更符合目标分布的训练语料。
     
**Step2** &emsp;对原始文本进行清洗和标准化是必须的步骤，包含去除或屏蔽无关元数据、修正或删除乱码与非法字符、统一字符编码，以及对带有敏感信息或隐私的语料要提前进行脱敏处理与合规检查，明确哪些信息不可用于训练并记录数据来源与许可。

**[根据NER技术进行数据脱敏处理的示例](#附录1数据脱敏处理示例)的输出分析**

处理前
      
>小明的邮箱是test111@gmail.com，电话是13312311111，现在居住于重庆两江新区的xxx小区。
         
脱敏后
      
>[NAME]的邮箱是[EMAIL]，电话是[PHONE]，现在居住于[PLACE]。

&emsp;&emsp;如果句子中出现姓名、电话号码、住址等敏感信息，需要进行脱敏处理。**通常在数据处理流程中，会优先处理高确定性的信息（例如电话号码、邮箱等）以排除干扰，随后再处理姓名等非标准信息，从而降低因表达不规范或格式多样而导致的总体漏检风险**。

&emsp;&emsp;值得注意的是，数据脱敏不仅是出于隐私保护与合规要求，同时也有助于提升下游文本建模与分词过程的稳定性。像姓名、电话号码、身份证号等高基数的信息，如果直接保留在语料中，往往会以近似唯一的形式出现。这类信息在统计上属于低频甚至单次出现的噪声，会干扰分词算法（如BPE、Unigram）在学习高频token结构时的统计效率。

>这样可以**有效降低语料中无效的多样性**，使分词器更专注于建模具有统计规律的语言模式，从而提升词表的利用效率、一致性。*从香农信息论的角度来看，数据脱敏可以视为一种结构化的去噪过程——通过压缩或减少高熵但低语义价值的信号（如具体身份信息），提高语料中有效信号的占比，可以协助LLM在后面的训练过程更倾向于学习可复用的语义结构，而非记忆偶然出现的实例细节。*

*📖Tips：高熵是指在给定上下文条件下，概率分布相对分散，从而具有较高的不确定性和预测困难度较高。*

&emsp;&emsp;而下游任务本身需要识别真实实体（如信息抽取等），过度脱敏会削弱训练信号。因此需要在**保护隐私**与**保留关键语义信息**之间进行合理的策略选择与权衡。

**Step3** &emsp;建议保留一小部分未参与训练的验证语料比如训练集:验证集＝99:1，用来在训练过程中评估分词器对真实文本的编码效率与平均token长度等统计指标。

### 2.1.2 预分词阶段

**Step1**&emsp;预分词的主要任务是将原始文本切分成可统计、可合并的基础单元，例如字符、字节或Unicode片段。常见策略包括基于空格和标点的切分、按Unicode类别划分，或直接采用字节级切分。需要注意的是并不是所有的分词器都需要用户显式进行预分词。*例如基于SentencePiece的分词器将标准化和预分词逻辑内置，因此无需在外部额外执行预分词步骤。*

   - 基于空格和标点的切分策略：一个完整的句子中遇到空格或者标点（.,!?[]{}...）可以分为独立的tokens，该方法适用于大多数预分词处理过程。
     
```python
      
# 基于空格和标点切分的实现示例
import re
         
def part(text):
   # 将标点符号单独拆开，并按照空格进行分割
   text = re.sub(r'([.,!?;:()"\'\[\]{}])', r' \1 ', text)
   tokens = text.split()
   return tokens

# 测试
if __name__ == "__main__":         
   s = "I like Datawhale."
   print(part(s))
   
```
   
输入
>I like Datawhale.
         
输出token划分
>['I', 'like', 'Datawhale', '.']

   - Unicode类别划分策略：按照字符的Unicode类别（字母、数字、标点、中文、特殊字符等）自动切分，不同类别会进入不同的token块——*一句话概括，同一个token里的字符类型都是一致的*。这种方法天然适合多语言混合文本，能提供一个可靠的基线切分结果。
        
**[根据Unicode类别划分策略的示例](#附录2基于unicode的划分策略)的输出分析**

输入
>Hello👋👋，Datawhale成立于2018年！！！

输出
>['Hello', '👋👋', '，', 'Datawhale', '成立于', '2018', '年', '！！！']

   - 字节级划分策略：先将每个字符拆成[UTF-8字节序列](https://datatracker.ietf.org/doc/html/rfc3629)，不依赖语言种类、字符，按照单个字节序列得到一个独立的token。
     
**根据字节级划分策略的输出分析**

```python
     
def tokenize_byte_level(text):
   tokens = []
   for ch in text:
   # 字符对应的UTF-8字节序列
      utf8_bytes = ch.encode("utf-8")
      hex_bytes = [f"{b:02X}" for b in utf8_bytes]
      print(f"{ch} 转化为UTF-8字节序列：{hex_bytes}")
      tokens.extend(hex_bytes)
   return tokens

if __name__ == "__main__":
   s = "All for learners！"
   print(tokenize_byte_level(s))

```

输入
>All for learners！

输出token划分
>['41', '6C', '6C', '20', '66', '6F', '72', '20', '6C', '65', '61', '72', '6E', '65', '72', '73', 'EF', 'BC', '81']

*英文字符和空格是ASCII，UTF-8下都是1字节。而全角感叹号`！`不是ASCII，在UTF-8下是3字节（`！ -> EF BC 81`）*
  
**Unicode与UTF-8的联系：**

&emsp;&emsp;`Unicode`就像给全世界所有字符发的“身份证号”，不管是英文A、汉字“中”、还是emoji 😄等不同类型的字符都在Unicode里有一个唯一编号比如A是U+0041，“中”是U+4E2D。但“身份证号”本身只是一个抽象编号，电脑不能直接存储。

&emsp;&emsp;`UTF-8`就像把这个字符对应的“身份证号”写进电脑的具体方式。它规定这个字符的编号应该用几个字节、按什么规则写下来。英文常用字符在UTF-8中只需要1个字节，而中文通常需要3个字节。不管是Unicode还是UTF-8都可以表示不同类别的字符，两者配合起来让自然语言可以被计算机准确存储、传输和解析，是人机交互之间的“桥梁”。

>Unicode是"编码标准"，为每个字符分配唯一码点；UTF-8是"编码格式"，负责将码点转换为字节序列。**UTF-8的一大优势是：ASCII字符（0-127）在UTF-8中的编码与ASCII码完全一致，且只占1个字节。这种向后兼容的特性，让它比UTF-16、UTF-32等编码方式更为常用。**
 
**在LLM的token划分中，常见策略包括**： 

（1）基于规则的预分词（如按空格和标点切分）；

（2）按Unicode类别分段（如连续汉字、连续拉丁字母或数字）；  

（3）更底层的UTF-8字节级切分。

（1）、（2）方法在以下场景中存在局限——文本缺乏显式分隔符，或出现长段同类字符。在这些情况下，预处理阶段难以有效断句，为了保证文本可编码，可能会被迫回退到更细粒度的兜底切分（接近字符级）。相比之下，UTF-8字节级策略具有最强的通用性，它把任意文本统一拆为字节序列，从而从根本上减少未词汇表外（OOV）问题并覆盖任意字符集。但因为它以最细粒度开始，训练时通常需要更多轮的共现统计与合并来把零散字节压缩为紧凑且具语义的token，才能在Transformer计算效率与语义表征之间取得平衡。

*📖Tips：词汇表外是当LLM模型在处理新的、实际应用的文本时，如果遇到一个词汇表中没有的Token，那么这个Token就被视为一个OOV。*

<div align="center">
<img width="1200" height="612" alt="da455deb32b22e4285d51b04dfb9a063" src="https://github.com/user-attachments/assets/97e514fd-5def-405d-95c2-9a9208d6f067" />
   <p>图2.3 token序列长度对注意力机制的影响</p>
 </div>

根据图2.3可以看出，token数量对注意力机制的影响主要体现在两方面：
- 计算复杂度呈 $O(N^2)$ 增长；
- 受到上下文窗口容量的限制。

当token划分得越碎片化（即总token数量越多），注意力机制的效率就越低。

**Step2** &emsp;对大多数以空格为词边界的语言，**可先用正则表达式按单词边界和标点进行初步划分**，而对中文、日文等不以空格为词界的语言则通常采用逐字符或基于字的初始单元来保证覆盖性。

**Step3** &emsp;预分词生成的基础单元序列将作为后续统计合并的输入，务必保存该序列与对应位置信息以便在训练过程反复高效更新。
   
**预分词保存对应位置信息的实现示例**
   
   ```python
   def btp_hex_list(text):
       """
       UTF-8字节级预分词，返回：
           1. tokens: 每个字符的字节序列+位置信息
           2. t: 所有字节的十六进制字符串列表
       """
       tokens = []
       t = []
       for idx, char in enumerate(text):
           utf8_bytes = char.encode('utf-8')
           hex_bytes = ' '.join(f"{b:02X}" for b in utf8_bytes)
           tokens.append({
               'char': char,
               'bytes': hex_bytes, # 单个字符对应的UTF-8字节序列
               'start': idx,   # 文本信息起始位置
               'end': idx + 1   # 文本信息结束位置
           })
           t.extend([f"{b:02X}" for b in utf8_bytes])
       return tokens, t
   if __name__ == "__main__":
      text = "Hi，你好🐋"
      tokens, t = btp_hex_list(text)
      for i in tokens:
          print(i)
      print(t)
   ```

   输入
   >"Hi，你好🐋"

   输出字节编码结果
   >['48', '69', 'EF', 'BC', '8C', 'E4', 'BD', 'A0', 'E5', 'A5', 'BD', 'F0', '9F', '90', '8B']
   

### 2.1.3 统计并迭代更新 

**Step1** &emsp;子词候选统计

遍历语料以收集用于后续决策的统计信息，具体方法随算法不同而异

   - BPE：统计当前字符、子词序列中相邻对的出现频次，每次贪心合并出现频次最高的相邻对，迭代构建词表——其决策仅基于频率统计。
   - WordPiece：评估合并或保留某些子词组合对语料似然即语言模型性能的贡献，选择能显著提升语料拟合度的合并操作。
   - Unigram：从一个过大的种子词表出发，初始化每个token的概率。
   - SentencePiece：是一个语言无关的子词分词框架，提供统一的训练与编码流程，支持多种分词算法（如BPE和Unigram）。这些算法在同一框架下独立使用，而非直接融合使用，其互补性体现在不同任务和数据条件下的适用性差异。
       
*📖Tips:分词算法决定token的划分策略，而分词器则是将分词算法与词表、编码机制等组件结合起来形成一个可将原始文本转换为模型输入序列的完整处理流程。这里我们可以思考这几个分词算法的区别是什么？*

| 算法                           | 适用场景                               | 常见实现方式                                             | 典型LLM / 模型                                                                          |
| ---------------------------- | ---------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [BPE](https://arxiv.org/pdf/1508.07909)| 简单高效；<br>高频子词压缩效果好；<br>适合大规模语料     | 字节级BPE<br>代码点级BPE                  | GPT-2、GPT-3、GPT-4（tiktoken体系）<br>LLaMA（改进BPE）<br>RoBERTa                        |
|[WordPiece](https://huggingface.co/learn/llm-course/en/chapter6/6) | 控制词表大小；<br>减少OOV；<br>适合MLM类模型      | 字符级、代码点级                                         | BERT（原始实现）<br>DistilBERT<br>早期RoBERTa（兼容）                                           |
| [Unigram LM](https://arxiv.org/abs/1808.06226)            | 概率建模子词；<br>对低频词更友好；<br>多语言适配强      | SentencePiece（Unigram模式）<br>支持byte-fallback       | T5、mT5<br>UL2<br>Gemma（Google系）                                                   |
| [SentencePiece](https://arxiv.org/pdf/1804.10959)（框架）       | 语言无关；<br>端到端训练tokenizer；<br>适合多语种 | **BPE或 Unigram的训练框架**（不是新算法）<br>支持byte-fallback | LLaMA（使用SP-BPE）<br>DeepSeek系列|

*总体来说，以上四种子词分词算法各有特点，没有哪一种是绝对最好的。选择算法时应根据具体的文本内容（语料分布）、任务类型（理解或生成）、词表规模以及是否需要处理多语言来决定，这样才能让训练出来的LLM模型发挥最佳性能。*

>[T5](https://ai.younglimit.com/deep-learning/nlp-pretraining/encoder-decoder-bart-and-t5/t5)（Text-to-Text Transfer Transformer）是Google在2019年提出的一种基于Transformer的预训练语言模型框架。其核心思想是将所有自然语言处理任务统一表示为“文本到文本”的形式，从而实现多任务统一建模。在此基础上，Google后续提出了一系列扩展模型，包括：
>- mT5：在多语言语料上预训练的T5扩展版本，支持跨语言任务；
>- UL2：一种统一的预训练范式，通过混合多种去噪目标（类似BERT、T5的目标任务），提升模型在不同下游任务的泛化能力。
>
>从T5的任务表示统一，到mT5的跨语言扩展，再到UL2在预训练阶段对多种去噪目标的**统一联合优化**，这一演化路径逐步实现了**从“任务层统一”到“数据分布扩展”再到“训练信号统一”的范式升级**，使语言模型能够在**共享参数空间中学习更具普适性的条件分布**，从而显著增强其跨任务与跨分布的泛化能力。
>*👉 LLM能力的提升可以看成是“统一性不断增强”的过程*


**Step2** &emsp;子词候选统计迭代更新

BPE、WordPiece、Unigram、SentencePiece这四种迭代算法简要分析：

   - BPE算法：可以借助第一步子词候选统计数据作为初始化数据，进行单个token合并形成新的token，然后在多次迭代过程中动态统计共现次数，得到新的token。
   - WordPiece算法：会在迭代过程中动态统计当前词汇表中所有相邻子词对的出现情况。其关键并非简单合并频率最高的对子，而是优先选择能最大提升语料整体似然的子词对，从而形成更有表征意义的token，一个常用的近似评分为：

$$
\text{score}(A,B)=\frac{P(A,B)}{P(A) \times P(B)}
$$

> 该比值衡量A与B的“关联性”是否强于独立出现时的期望。如果score>1，则说明A与B的结合比随机独立出现更有义，更可能被WordPiece合并。【Google没有公开WordPiece算法详情，这里参考[huggingface相关原理介绍](https://huggingface.co/learn/llm-course/en/chapter6/6)】

   - Unigram算法：它基于一个子词概率语言模型，将一个句子的概率定义为其所有可能分词方式概率的总和。核心思想是通过迭代优化子词概率，使得整个语料的似然最大化。算法采用期望最大化（EM）方法，主要包含两步：

      &emsp;&emsp;①E步（期望步）：在当前词表和子词概率下，为语料中的每个句子计算最可能的分词方式或前n个高概率分词方案，并据此估计每个子词在语料中的期望使用次数。

      &emsp;&emsp;②M步（最大化步）：根据E步的统计结果，更新每个子词的概率，使整体语料的似然最大化。

      &emsp;&emsp;③在每次迭代中，模型会剪枝（淘汰）概率较低的token如丢弃底部10%~20%，从而逐步收敛到一个较小且优化后的词表，直到达到预设的目标词表大小。这种方法相比BPE或WordPiece更依赖概率建模，能够灵活处理不同长度的子词，并自然保留最能解释语料的高频段。

*📖Tips:`最大化语料似然`指的是在训练分词器时，学习一套能让训练语料整体概率最大的`token`划分方式和词表，使语料中的字符序列可以被更高概率、更常见的token组合表示。通俗地说就是把文本切成最顺畅、最符合语言统计规律的片段，让同一句话在不同地方都能被稳定、紧凑地切分成一致的token序列。*

   - SentencePiece算法：它是一个独立的分词工具和实现库，能够直接从原始文本训练子词模型，因此无需用户在外部显式执行[预分词步骤](#212-预分词阶段)。它在内部会将空格、词边界等信息编码为特殊字符（训练输出中常见的`▁`用于表示词首空格），从而可以将空格也作为建表对象之一，接着会在这些初始token上应用BPE或Unigram算法，生成最终的token词表及映射。

这个迭代过程中需要保持特殊控制token（如`<PAD>`、`<UNK>`、`<CLS>`、`<MASK>`等），在分词器迭代更新过程中不参与修改，这样可以确保它们的词——数字映射保持固定，编码后的离散数字序列能够准确还原为原始文本。同时，这些token不会在统计合并或概率优化中被拆分或覆盖，从而有效减少碎片化token的出现。无论使用BPE、WordPiece、SentencePiece还是Unigram等算法，这一策略都适用有助于保护关键token的完整性，保证模型训练和推理的一致性。

>💡需要注意的是——分词器优化的是输入的表示方式，它本身并不会让模型“理解”自然语言。而真正“理解”语言的能力是在后续`Transformer`参与的训练中获得的，但更合理的token划分会让输入分布更清晰一致从而间接提升模型训练效率和最终性能。

**总结**：分词器训练的核心是<ins>迭代更新候选子词 → 控制词表大小或收敛指标 → 监控质量指标</ins>，不同算法仅在“候选生成方式”和“迭代更新策略”上有差异。


### 2.1.4 输出产物并用于编码与解码

**Step1** &emsp;导出核心产物，不管是使用哪一种分词算法，训练完成后都需要导出至少两个关键文件：
   
<div align="center">
   <img width="1000" height="500" alt="0fb498f1c19fdae425eb17ad4270c45e" src="https://github.com/user-attachments/assets/1ea639e8-81f9-49a5-b5d7-3df32fe129a2" />
   <p>图2.4 vocab、merges文件示例</p>
   </div>
   
   - **vocab文件**：记录所有token及其对应的id，是编码器和解码器的核心索引。
   - **merges文件**：按顺序记录所有子词合并规则或概率模型。二者共同决定tokenizer的编码与解码逻辑，并确保编码的可逆。

**Step2** &emsp;下游使用前的验证与评估

   - 将tokenizer应用于一部分验证集后，建议统计以下关键指标：
      - `平均token数与最大长度分布`&emsp;直接影响显存占用、训练速度和推理效率。
      - `碎片化情况`&emsp;检查关键实体、专业术语是否被拆得过碎，避免影响模型理解。
      - `跨语言token平衡度`&emsp;多语言任务中需确保不同语言的常见模式都有足够的token支持。

>&emsp;&emsp;如果后续需要**扩表**如加入新领域术语、专业词或品牌名等，建议优先采用这些方式而非完全重训tokenizer：**增量训练**、**加入新的merges项**、**清理极低频token**。
>
>&emsp;&emsp;并且，扩表后应进行一次**回归测试**，确保——与旧模型保持兼容且根据数字化编码可以还原回到最开始的输入文本、不发生token分配冲突或token耗尽问题。

*看到这里，我们就可以回答最开始的问题：如何评价一个好的tokenizer。*

---

## 2.2 常用的分词器
&emsp;&emsp;我们上一小节了解到tokenizer的训练流程，接下来将介绍几种常见的分词器，并分析它们各自的核心思想与特点。

### 2.2.1 字符分词器

**字符分词器的原理介绍**

&emsp;&emsp;这是最直观、最简单的分词方式，它将文本拆解为最小的字符单位即单个字符形如英语中的字母 *（a, b, c）* 或者中文里的单字 *（你，好）*。

  - **优点：**
      - **词表极小：** 英语只需包含26个字母+符号；中文只需包含常用汉字（约几千个）。
      - **无OOV问题：** 任何生僻词都是由基础字符组成的，不会出现“未知词”。
  - **缺点：**
      - **序列过长：** 一句话变成字符后，长度会增加数倍，大大消耗LLM宝贵的上下文窗口，从而加大LLM的transformer计算显存消耗。
      - **语义稀疏：** 单个字符（如t）通常不具备独立的语义，模型需要更深的网络层数来组合出意义。

**[字符分词器的代码运行](#附录3字符分词器)的输出分析**

   
输入
> hi，很好的，terrific！🐋

输出
> 编码ID: [104, 105, 65292, 24456, 22909, 30340, 65292, 116, 101, 114, 114, 105, 102, 105, 99, 65281, 128011]
> 
> 压缩比率: 0.47058823529411764


### 2.2.2 字节分词器

**字节分词器的原理介绍**

&emsp;&emsp;计算机底层存储文本的本质上都是**字节**，在UTF-8编码中，英文通常占1个字节，汉字通常占3个字节。字节分词器直接对二进制字节进行操作。

  - **核心逻辑：** 不再维护“字符”的词表，而是维护一个大小为256的基础词表（0x00到0xFF）。
  - **应用：** 现代LLM如GPT-4, Llama通常不单独使用纯字节分词，而是将字节作为BPE的基础单位即BBPE，这样可以彻底解决跨语言和特殊符号如emoji 🌍等的编码问题。

**[BPE、字符级、字节级的分词器效果对比示例](#附录4bpe字符级字节级的分词器效果对比)的输出分析**

输入
> Hello, 🌍! 你好!

输出
>原始字符串： Hello, 🌍! 你好!
>
>原始字节长度： 20
>
>字节级、字符级、BPE的token数量分别为: 20、13、11
>
>=== 压缩率(byte/token) ===
>
>字节级: 1.00
>
>字符级: 1.54
>
>BPE: 1.82
>
>=== 结论 ===
>1. 字节级：无压缩，最稳定
>2. 字符号级：用UTF-8字符压缩（中文/emoji更明显）
>3. BPE：通过学习高频子串，实现真正“数据驱动压缩”
>
>由此可得，BPE压缩效果最好（最接近真实LLM tokenizer）

值得注意的是**字节级分词器的压缩比恒等于1**，原因在于：

- 输入文本中单个字符首先被编码为UTF-8字节序列；
- 字节级分词器将每一个UTF-8字节（0-255）直接作为一个token；
- 因此`token数量=UTF-8字节数`。

所以

$$
compression_{ratio}
= \frac{\text{UTF-8 字节长度}}{\text{token 数量}}
= \frac{N}{N}
= 1
$$

**也就是说，字节级分词器完全不具备压缩能力即每个字节对应一个token，不会产生更长或更短的词片段。**

### 2.2.3 词级分词器

**词级分词器的原理介绍**

在深度学习早期（如RNN时代）这是最主流的方法。它基于空格（英文）或分词算法（中文）将文本切分为具备独立语义的“词”。

  - **优点：** Token保留了完整的语义信息比如"apple" 直接对应一个Token ID...。
  - **缺点：**
      - **词表爆炸：** 英语中 `look, looks, looked, looking` 会被视为4个完全不同的ID，导致词表巨大几十万甚至上百万。
      - **OOV 问题严重：** 遇到没见过的词如人名、新造词等，只能标记为`<UNK>` ，导致信息丢失从而影响LLM的表现能力。

**[词级分词器的代码运行](#附录5词级分词器)的输出分析**

   
输入   
>It's so supercalifragilisticexpialidocious!👋👋

输出
>基础正则分词结果：
>
>['It', "'", 's', ' ', 'so', ' ', 'supercalifragilisticexpialidocious', '!', '👋', '👋']
>
>deepseek风格分词结果：['It', "'", 's', ' ', 'so', ' ', 'supercalifragilisticexpialidocious', '!👋👋']
>
>词表大小： 7
>
>编码token IDs：[3, 2, 4, 0, 5, 0, 6, 1]
>
>压缩率： 6.375

### 2.2.4 BPE分词器

**BPE分词器的原理介绍**

这是目前LLM（GPT, BERT, Llama等）最主流的分词算法，BPE是一种试图在<ins>字符级(粒度太细)</ins>和<ins>词级(粒度太粗)</ins>之间找到平衡。**核心思想：** 统计语料中相邻字符对出现的频率，迭代地将**最频繁出现的字符对**合并成一个新的Token。


**[简易版BPE训练过程示例](#附录6bpe分词器简易训练)的输出分析：**
  
输入
>test_text = "敏捷的棕色狐狸🦊"

输出
>BPE合并: [(' ', '`</w>`'), ('t', 'h'), ('th', 'e'), ('the', '`</w>`'), ('这', '只'), ('这只', '猫'), ('这只猫', '`</w>`'), ('🐈', '`</w>`'), ('很', '可'), ('很可', '爱'), ('很可爱', '`</w>`'), ('q', 'u'), ('qu', 'i'), ('qui', 'c'), ('quic', 'k'), ('quick', '`</w>`'), ('b', 'r'), ('br', 'o'), ('bro', 'w'), ('brow', 'n')]
>
>编码: ['敏', '捷', '的', '棕', '色', '狐', '狸', `</w>`, '🦊', '`</w>`']

在BPE编码阶段，如果没有`</w>`算法可能把`the`错误地拆成'th'、'e'或在后续合并时与其他token错误合并。加上`</w>`后，`the`会被表示为['t', 'h', 'e', '`</w>`']，BPE就知道这是一个完整单词的结尾不会跨单词错误合并，那么解码阶段去掉`</w>`就能把token拼回`the`，保证原文恢复正确。

*因此，`</w>`的核心作用是保证单词完整性，并让编码可逆即可以从相应的数字序列转化为原文。*

**4种分词器对比表**

| 分词器类型           | 粒度         | 词表大小              | 词表外(OOV) | 序列长度   | 代表模型               |
| :-------------- | :--------- | :---------------- | :------- | :----- | :----------------- |
| 字符级             | 细          | 小 (100–5k)        | 无        | 非常长    | Char-RNN           |
| 字节级             | 更细（字节）     | 很小 (~256–1k)      | 无        | 很长     | GPT-2              |
| 词级              | 粗          | 极大 (>100k)        | 严重       | 短      | Word2Vec, GloVe    |
| **BPE**         | **中（自适应）** | **适中 (30k–100k)** | **极少**   | **适中** | **GPT-4, Llama 3** |


&emsp;&emsp;除了`分词器的选择`与`训练语料`直接影响LLM的输入稀疏度与表示效率。用大规模、高质量且多样的语料训练分词器通常会`减少token碎片化`即生成更常见、更稳定的子词单元，使得同一段文字被编码为更少的token，同时在固定的上下文窗口长度下单位token承载更多实际信息，这意味着模型能够在有限窗口内“看到”更多内容——从而在一定程度上缓解因上下文长度受限引起的信息丢失。

>注意上述情况这依赖于语料的覆盖与质量；若语料偏颇或过度合并罕见词，反而可能损害少数语言或专业术语的表示能力。

---

## 2.3 分析DeepSeek的分词器

DeepSeek模型尤其是Coder系列，对代码和中英文都进行了高度优化，我们将加载DeepSeek Coder模型的官方分词器。

### 2.3.1 加载DeepSeek Tokenizer
请确保transformers库已安装

```python
# 安装transformers库
pip install transformers torch
```

我们将加载`deepseek-ai/deepseek-coder-6.7b-instruct`的分词器。

```python
from transformers import AutoTokenizer
# 使用DeepSeek Coder系列模型的分词器
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"成功加载模型: {MODEL_NAME} 的分词器。")
print(f"分词器词表大小V: {len(tokenizer.get_vocab())}")
```

### 2.3.2 DeepSeek分词器的处理逻辑

&emsp;&emsp;DeepSeek优化的字节级BPE词表通过对中文字词分布与代码缩进的精细建模，显著缩短了token序列长度，这种设计在确保语义完整性的同时，大幅提升了推理吞吐。本文将通过剖析中文分词实例，直观展现其如何通过高频词簇聚合来优化文本序列的效率。

**举例分析: 中文文本处理**

观察DeepSeek如何处理中文短语，通常它也会使用子词或单个汉字Token来提高效率。
```python
chinese_text = "注意力机制是AI的核心技术。 🚀 🚀"
# 编码
encoded_ids = tokenizer.encode(chinese_text, add_special_tokens=False)
# 解码回Token字符串 (用于观察子词)
tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
print(f"\n原文: {chinese_text}")
print(f"编码: {tokens}")
print(f"IDs:{encoded_ids}")
```

&emsp;&emsp;得到token字符串划分结果可能在显示上与原文有所差异这并不是编码本身出错，而是因为LLM所用的**词表**在训练过程中<ins>对某些字符或子词的覆盖不足</ins>（例如BPE训练不够充分），导致模型无法生成对应的token，从而在可读形式上看起来像“乱码”。通过增加训练语料量或进行充分的BPE训练，可以学习到更完整的token映射词表，从而解决这个问题，使中文、英文、emoji等字符都能被正确编码和解码。接下来是相应的解决办法即训练BPE：

**[DeepSeek的正则化处理+BPE训练示例](#附录7deepseek风格的tokenizer简易实现示例)的输出分析：**

输入测试样例
>注意力机制是AI的核心技术。 🚀 🚀

输出分析
>本流程依据预设的分词算法将文本离散化为最小语义单元（token），并构建其与全局唯一数值ID及底层编码的确定性映射。相同字符（如空格、特定Emoji🚀）在文中均指向一致的ID与编码序列，确保了特征表征的稳定性。

&emsp;&emsp;从以上代码的运行结果可以看到，分词器中的`token ↔ id`映射只描述“这个token的内容”，并不包含它在句子中的任何位置信息。BPE或其他基于统计和概率的分词算法，其本质都是依据语料中的共现频率或概率分布，决定如何将常见的字符、字节或子串合并成更长、更高频的token。这类算法本身并不理解句子的语义，它更像一个纯统计模块，通过频率或概率原则对字符序列进行切分与合并，为模型*LLM*提供稳定而紧凑的离散输入单元。

>为什么DeepSeek要用latin1编解码?
>
>在DeepSeek的分词流程中最终处理得到的是数字化的token，但在BPE分词器训练阶段需要按“字符”操作。如果直接用UTF-8编解码，汉字或emoji等多字节字符在拆分为单字节时会出现不完整序列，Python会报错或替换，导致信息丢失。而latin-1是单字节编码，它把每个字节（0–255）机械映射为一个Unicode字符，保证任意字节序列都能完整、可逆地保存，从而让BPE或其他子词算法能把字节当作字符合并而不丢数据。简单来说使用latin-1是为了在分词器中安全地把原始字节当作字符处理，确保编码器阶段信息完整。

### 2.3.3 快速体验DeepSeek分词器

<div align="center">
   <img width="1788" height="1077" alt="fdfd8c74f27d0832c80a9d4587daeb16" src="https://github.com/user-attachments/assets/7e7b40df-e9c0-43ed-896a-40679a8046c8" />
   <p>图2.5 DeepSeek分词器</p>
   </div>
   
   这里输入文本信息为*你好 ，hello,  world !  🌏 ！*。
   
>通过这里的可视化展示[DeepSeek的分词器](https://tiktokenizer.vercel.app/?model=deepseek-ai%2FDeepSeek-R1)输出，右边的最后一栏表示的是对应token ID映射，可以看到对于单独划分为一组token的空格，就算位置不同其ID值均为223。


---

## 2.4 分词器学习的总结

&emsp;&emsp;在本章中，我们学习了分词器的基本原理、训练流程等。最后，我们还介绍了DeepSeek Coder模型的分词器，以及它在中文的文本处理中的应用。

&emsp;&emsp;基于BPE算法构建的分词器在早期与部分现有LLM中被广泛采用，其核心原因在于该方法主要是统计频率驱动的贪心合并策略，实现简单、训练高效。

>但是，BPE并不是唯一选择，比如WordPiece、Unigram、SentencePiece等方法在不同模型中同样被广泛使用，只是它们在建模假设与优化目标上各有差异。

&emsp;&emsp;tokenizer充当语言模型的输入输出接口，是模型与外部文本之间的桥梁，**其核心作用是将连续的自然语言文本转换为离散的token序列，并映射为对应的整数ID，从而作为Transformer等模型结构的输入**。在模型内部，这些离散token会被进一步映射为向量表示（embedding），并经过多层计算形成上下文相关的隐状态表征，这些表征构成了模型进行预测、泛化解决问题的基础。

---

## 2.5 思考

**基础思考问题：**

1）BPE、WordPiece、Unigram、SentencePiece这四种分词算法的区别是什么，为什么现在一些LLM的tokenizer选择了基于BPE实现？

2）如何衡量“一个好的分词器”，其在压缩效率与语义一致性之间应如何trade-off？

3）分词器如何影响实际LLM的最终表现？

**进阶思考问题：**

1）有研究表明，视觉特征能够增强LLM的理解能力，但并非适用于所有语言任务。那么是否可以在视觉表征与离散token之间寻求一种动态“平衡点”：同时为模型提供两类表征方式，并借鉴MoE的思想设计轻量级动态路由，使模型能够在不同任务或文本片段中自动选择或融合最合适的embedding方法，从而显著提升跨场景的适配能力？

> 文本token的离散性限制了表达能力，视觉token可提供高密度的连续压缩表征但并不适用于所有语言场景；因此探索一种MoE风格的多表征机制，使模型能按任务动态选择文本、视觉或混合表征，以获得更丰富且具场景适配性的表示或许也值得思考。

2）能否设计一种“自适应分词器”，在训练阶段先与LLM分开训练，并通过一种特殊机制将训练好的分词器与模型结合，使其在下游任务中仍能动态学习和优化token划分策略？

>比如考虑一种反馈驱动的词表动态增强方法，**核心是跨模型语义表示的蒸馏与迁移**。它不使用传统的输出概率蒸馏，而是由教师模型根据用户反馈，提取新概念的精准语义向量。通过映射适配器将该向量投影到学生模型的Embedding空间，实现对学生词表矩阵的即时“补丁”，从而让学生模型能够零样本地识别并处理新Token。

3）借助微分子词模块、元学习或强化学习等方法，让分词器能够从少量对话或任务样本中自动发现最合适的token划分方式，从而降低下游任务对数据的依赖同时提升模型的鲁棒性和泛化能力？

> 这种方式有点像半监督学习，分词器自己在“学习怎么学习”，这样即使只看到少量对话样本，它也能找到更合适的token划分方式，让模型理解语言更高效，也更不容易被新词或少量数据难住。



# 参考资料
- [hugging face介绍四种分词器算法](https://huggingface.co/learn/llm-course/en/chapter6/1)
- [BPE算法](https://arxiv.org/pdf/1508.07909)
- [CS336的Lecture1课程资料](https://stanford-cs336.github.io/spring2025-lectures/?trace=var%2Ftraces%2Flecture_01.json&source=lecture_01.py&line=238)


# 附录（代码实验部分）

## 附录1：数据脱敏处理示例

本示例基于Python实现，借助命名实体识别（Named Entity Recognition，NER） 技术来实现数据脱敏：

*简单来说，命名实体识别是自然语言处理（NLP）中的一项基础任务，它的作用就像一台“目标扫描仪”——从大量非结构化的文本中，自动识别并提取出具有特定意义的实体（如人名、地名、组织机构、时间等）*
   
```python
# 初始化命名实体识别（NER）流水线
ner_pipeline = pipeline(
    "ner",
    model="ckiplab/bert-base-chinese-ner",
    grouped_entities=True  # 将相邻的同类实体片段合并，例如“重”、“庆”合并为“重庆”
)

def ner_mask(text: str) -> str:
    """
    利用深度学习模型进行语义级别的脱敏（人名与地名）
    """
    entities = ner_pipeline(text)
    spans = []
    
    # 提取模型识别出的实体及其位置
    for ent in entities:
        label = ent["entity_group"]
        start = ent["start"]
        end = ent["end"]

        # 映射实体类型到脱敏占位符
        if label == "PER":  # Person: 人名
            spans.append((start, end, "[NAME]"))
        elif label == "LOC":  # Location: 地名/地址
            spans.append((start, end, "[PLACE]"))

    # 排序逻辑：按起始位置升序；如果起始位置相同，按长度降序（优先处理长实体）
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # 解决冲突：去除重叠或包含关系的实体区间
    filtered_spans = []
    last_end = -1
    for start, end, tag in spans:
        if start >= last_end:  # 只有当当前实体起始位置在上一实体结束之后，才保留
            filtered_spans.append((start, end, tag))
            last_end = end

    # 根据过滤后的区间重建文本
    result = []
    last_idx = 0
    for start, end, tag in filtered_spans:
        result.append(text[last_idx:start]) # 拼接非敏感部分
        result.append(tag)                  # 拼接占位符
        last_idx = end
    result.append(text[last_idx:])          # 拼接剩余文本

    return "".join(result)


# 2. 脱敏流水线架构设计
class DesensitizationPipeline:
    """
    脱敏任务管理器：允许按顺序添加多个处理步骤
    """
    def __init__(self):
        self.steps: List[Callable[[str], str]] = []

    def add_step(self, func: Callable[[str], str]):
        """添加处理环节（如正则替换、NER替换等）"""
        self.steps.append(func)

    def run(self, text: str) -> str:
        """按顺序执行所有脱敏步骤"""
        for step in self.steps:
            text = step(text)
        return text

# 3. 具体处理步骤实现
def normalize_text(text: str) -> str:
    """文本预处理：去除首尾空格"""
    return text.strip()

# 高确定性规则（强特征：手机号、邮箱）
def mask_phone(text: str) -> str:
    """正则匹配 11 位中国手机号"""
    return re.sub(r'1[3-9]\d{9}', '[PHONE]', text)

def mask_email(text: str) -> str:
    """正则匹配常见邮箱格式"""
    return re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)

# 中确定性规则（基于关键词上下文）
def mask_address(text: str) -> str:
    """通过“居住于”等关键词引导的地址匹配"""
    return re.sub(
        r'(居住于|现居住于|现居于|地址)([\u4e00-\u9fa5A-Za-z0-9]+)',
        r'\1[PLACE]',
        text
    )

# 低确定性规则（基于语法结构的简单兜底）
def mask_name(text: str) -> str:
    """
    兜底策略：匹配出现在句首或标点后的“某某某的”结构
    注：容易误伤，通常放在 NER 步骤之后作为补充
    """
    return re.sub(
        r'(?:(?<=^)|(?<=[，。！？]))([\u4e00-\u9fa5]{2,3})(的)',
        r'[NAME]\2',
        text
    )

def clean_punctuation(text: str) -> str:
    """后处理环节：可根据需求规范化标点符号"""
    return text
```

[ 数据脱敏处理完整代码](https://github.com/datawhalechina/diy-llm/blob/main/docs/zh/chapter2/de_identified_data_processing.py)  

## 附录2：基于Unicode的划分策略

```python
import unicodedata

def get_char_category(ch: str) -> str:
    # 获取Unicode标准定义的分类（如'Lu'代表大写字母,'Po'代表其它标点）
    cat = unicodedata.category(ch)

    # 判定是否为中文字符（常用基本汉字区间）
    if '\u4e00' <= ch <= '\u9fff':
        return "CJK"
    
    # 判定是否为数字
    if ch.isdigit():
        return "DIGIT"
    
    # 判定是否为英文字母（或其他语言的字母）
    if ch.isalpha():
        return "ALPHA"

    # 判定是否为标点符号（Unicode 分类以 'P' 开头的均为标点）
    if cat.startswith("P"):
        return "PUNCT"

    # 其余字符（如 Emoji、空格、控制符等）统一归为 OTHER
    return "OTHER"


def segment_by_unicode_category(text: str):
    if not text:
        return []
    segments = []
    # 初始化缓冲区，放入第一个字符
    buffer = [text[0]]
    # 获取第一个字符的类别作为初始参考标准
    prev_type = get_char_category(text[0])

    # 第一阶段：线性扫描文本，按类别切分
    for ch in text[1:]:
        curr_type = get_char_category(ch)

        # 如果当前字符类别与前一个字符相同，则存入缓冲区合并
        if curr_type == prev_type:
            buffer.append(ch)
        else:
            # 类别发生变化，将缓冲区内容作为一个片段存入结果列表
            segments.append(("".join(buffer), prev_type))
            # 重置缓冲区，开始记录新类别的字符
            buffer = [ch]
            prev_type = curr_type

    # 处理最后一个留在缓冲区里的片段
    segments.append(("".join(buffer), prev_type))

    # 第二阶段：提取分段后的字符串内容
    tokens = [seg for seg, _ in segments]
    return tokens

# 测试运行
if __name__ == "__main__":
    # 测试字符串包含：英文、Emoji、中文标点、中文、数字、英文标点
    s = "Hello👋👋，Datawhale成立于2018年！！！"
    result = segment_by_unicode_category(s)
    print("原始文本:", s)
    print("分段结果:", result)
```

## 附录3：字符分词器

   ```python
      # 字符Tokenizer
   class CharacterTokenizer:
       def __init__(self):
           pass  # 不需要额外参数，直接用ord、chr
   
       def encode(self, text):
           """
           将字符串编码为字符索引列表（Unicode code points）
           """
           return [ord(ch) for ch in text]
   
       def decode(self, indices):
           """
           将索引列表解码为字符串
           """
           return ''.join([chr(i) for i in indices])
   
   # 测试代码
   if __name__ == "__main__":
       tokenizer = CharacterTokenizer()
       string = "hi，很好的，terrific！🐋"  # 测试字符串
   
       # 编码
       indices = tokenizer.encode(string)
       print("编码ID:", indices)
   
       # 解码
       reconstructed_string = tokenizer.decode(indices)
       print("解码:", reconstructed_string)
   
       # 验证是否可逆
       assert string == reconstructed_string, "字符编码、解码不一致!"
   
       # 计算词汇量（最大Unicode code point+1）
       vocabulary_size = max(indices) + 1
       print("词汇量（上限）", vocabulary_size)
   
       # 简单压缩率计算
       def get_compression_ratio(text, indices):
           # 压缩率 = 原字符串字节数/编码索引字节数
           import sys
           original_bytes = len(text.encode('utf-8'))
           encoded_bytes = len(indices) * 4  # 假设每个Unicode code point用4字节存储
           return original_bytes / encoded_bytes
   
       compression_ratio = get_compression_ratio(string, indices)
       print("压缩比率:", compression_ratio)
   ```
## 附录4：BPE、字符级、字节级的分词器效果对比

   ```python
# 字节级Tokenizer
from collections import Counter
class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, indices):
        return bytes(indices).decode("utf-8")

# 字符级Tokenizer
class CharTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    def encode(self, text: str):
        tokens = []
        for ch in text:
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.inverse_vocab[idx] = ch
            tokens.append(self.vocab[ch])
        return tokens

    def decode(self, indices):
        return "".join(self.inverse_vocab[i] for i in indices)

# 计算压缩率（byte/token）
def get_compression_ratio(text: str, token_len: int):
    input_byte_len = len(text.encode("utf-8"))
    return input_byte_len / token_len if token_len > 0 else 1


# 简易 BPE Tokenizer
class BPETokenizer:
    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.merges = {}  # {(a,b): new_token_id}
        self.vocab_size = 256  # 从byte开始

    def get_stats(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i+1])] += 1
        return pairs

    def merge_tokens(self, tokens, pair, new_token):
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
   ```

[BPE、字符级、字节级的分词器效果对比](https://github.com/datawhalechina/diy-llm/blob/main/docs/zh/chapter2/BPE_character_byte_level_word_segmentation_Comparison.py)
   
## 附录5：词级分词器

```python
import regex

# deepseek tokenizer中使用的经典正则表达式（简化版）
TOKENIZER_REGEX =  r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"

# 压缩率计算
def get_compression_ratio(text: str, segments):
    byte_len = len(text.encode("utf-8"))
    token_count = len(segments)
    return byte_len / token_count if token_count > 0 else 1


# Word-level Tokenizer实现
class WordTokenizer:
    def __init__(self, pattern=r"\w+|."):
        """
        pattern: 正则表达式（默认基础版：把连续字母数字合成一个词）
        """
        self.pattern = pattern
        self.word2id = {}
        self.id2word = {}

    def build_vocab(self, texts):
        """
        根据训练文本列表建立词表
        """
        vocab = set()
        for text in texts:
            segments = regex.findall(self.pattern, text)
            vocab.update(segments)

        vocab = sorted(vocab)
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}

    def encode(self, text):
        """
        文本 → 字符串片段 → token id列表
        未登录词 UNK = -1
        """
        segments = regex.findall(self.pattern, text)
        return [self.word2id.get(seg, -1) for seg in segments], segments

    def decode(self, ids):
        """
        token ID → 原始片段 → 拼成字符串
        """
        return "".join(self.id2word.get(i, "<UNK>") for i in ids)

# 测试
if __name__ == "__main__":

    string = "It's so supercalifragilisticexpialidocious!👋👋"
    print("原始字符串：", string)

    # 使用基础正则分词（基于空格和标点切分）
    basic_segments = regex.findall(r"\w+|.", string)
    print("基础正则分词结果：")
    print(basic_segments)

    # 使用deepseek风格正则
    segments = regex.findall(TOKENIZER_REGEX, string)
    print(f"deepseek风格分词结果：{segments}")

    # 构建词表
    tokenizer = WordTokenizer(pattern=TOKENIZER_REGEX)
    tokenizer.build_vocab([string])

    print("词表大小：", len(tokenizer.word2id))

    # 编码
    ids, segs = tokenizer.encode(string)
    print(f"编码token IDs：{ids}")

    # 字节序列
    byte_tokens = [b for b in string.encode("utf-8")]
    print(f"UTF-8字节序列：{byte_tokens}")

    print(f"编码segments：{segs}")

    # 解码
    decoded = tokenizer.decode(ids)
    print("解码结果：", decoded)

    # 压缩率
    ratio = get_compression_ratio(string, segs)
    print("压缩率：", ratio)
   ```

## 附录6：BPE分词器简易训练

 
   ```python
   import regex
   from collections import Counter
   
   # DeepSeek风格正则
   DEEPSEEK_REGEX = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"
   
   # 使用grapheme cluster保持emoji不被拆分
   def split_graphemes(token):
       return tuple(regex.findall(r'\X', token))
   
   # BPE训练函数
   def train_bpe(texts, num_merges=50):
       """
       texts: 文本列表（用于训练BPE）
       num_merges: BPE 迭代合并的次数
       """
       # 1.构建初始vocab（字符级+</w>结束符）
       vocab = Counter()
       for text in texts:
           tokens = regex.findall(DEEPSEEK_REGEX, text)
           for token in tokens:
               chars = split_graphemes(token) + ('</w>',)
               vocab[chars] += 1
       merges = []
       for _ in range(num_merges):
           # 统计相邻pair出现次数
           pairs = Counter()
           for word, freq in vocab.items():
               for i in range(len(word)-1):
                   pairs[(word[i], word[i+1])] += freq
           if not pairs:
               break
   
           # 找到最常见pair
           best_pair = max(pairs, key=pairs.get)
           merges.append(best_pair)
   
           # 合并所有vocab中的该pair
           new_vocab = {}
           for word, freq in vocab.items():
               w = []
               i = 0
               while i < len(word):
                   if i < len(word)-1 and (word[i], word[i+1]) == best_pair:
                       w.append(word[i]+word[i+1])
                       i += 2
                   else:
                       w.append(word[i])
                       i += 1
               new_vocab[tuple(w)] = freq
           vocab = new_vocab
       return merges, vocab
   
   # BPE Tokenizer类
   class BPETokenizer:
       def __init__(self, merges):
           self.merges = merges
   
       def encode_word(self, token):
           # 初始分成字符+</w>
           word = list(split_graphemes(token)) + ['</w>']
           # 按merge顺序依次合并
           for pair in self.merges:
               i = 0
               new_word = []
               while i < len(word):
                   if i < len(word)-1 and (word[i], word[i+1]) == pair:
                       new_word.append(word[i]+word[i+1])
                       i += 2
                   else:
                       new_word.append(word[i])
                       i += 1
               word = new_word
           return word
   
       def encode(self, text):
           tokens = regex.findall(DEEPSEEK_REGEX, text)
           bpe_tokens = []
           for t in tokens:
               bpe_tokens.extend(self.encode_word(t))
           return bpe_tokens
   
       def decode(self, tokens):
           # 拼接tokens并去掉结尾</w>
           text = ''.join(tokens).replace('</w>', '')
           return text
   
   # 测试
   if __name__ == "__main__":
       train_texts = ["这只猫🐈很可爱", "the quick brown fox jumps over the lazy 🐕‍🦺"]
       merges, vocab = train_bpe(train_texts, num_merges=20)
       print("BPE合并:", merges)
       tokenizer = BPETokenizer(merges)
       test_text = "敏捷的棕色狐狸🦊"
       encoded = tokenizer.encode(test_text)
       print("编码:", encoded)
       decoded = tokenizer.decode(encoded)
       print("解码:", decoded)
   ```

## 附录7：DeepSeek风格的Tokenizer简易实现示例

```python
"""
DeepSeek-V3 Tokenizer简易实现示例
（核心包含：字节级BPE+DeepSeek风格正则预分词）
"""
import regex as re
from collections import Counter
from typing import List, Tuple, Dict, Iterable
import json
import base64


# 配置：DeepSeek 正则模式（预分词）
# \p{L}+   连续字母（中文、英文、所有 Unicode 字母）
# \p{N}+   连续数字
# [^\p{L}\p{N}\s]+  非字母数字空白的字符（如标点、emoji）
# \s+      连续空白符
DEEPSEEK_REGEX = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"


# 基础函数：预分词与字节处理
def pretokenize(text:str):
    """按DeepSeek风格的正则进行预分词"""
    return re.findall(DEEPSEEK_REGEX, text)

def bytes2tokens(b:bytes):
    """
    将UTF-8字节序列转为latin1可表示的token列表。
    每个字节0–255都能被latin1接映射到字符。
    """
    return [bytes([x]).decode('latin1') for x in b]

def tokens2bytes(tokens):
    """将latin1 token列表重新转回原始bytes"""
    return b''.join([t.encode('latin1') for t in tokens])


# BPE训练相关
def build_corpus(texts):
    """
    构建byte-level语料。
    步骤：预分词 → UTF-8编码 → 分解为单字节 → 作为初始token序列。
    """
    corpus = []
    for text in texts:
        for chunk in pretokenize(text):
            corpus.append(bytes2tokens(chunk.encode('utf-8')))
    return corpus

def pair_freq(corpus: List[List[str]]):
    """统计所有token序列中相邻token pair的出现频率"""
    pairs = Counter()
    for word in corpus:
        for i in range(len(word)-1):
            pairs[(word[i], word[i+1])] += 1
    return pairs

def merge_pair(word: List[str], pair: Tuple[str,str]):
    """将指定的token pair合并成一个token"""
    a, b = pair
    merged = []
    i = 0
    while i < len(word):
        if i < len(word)-1 and word[i]==a and word[i+1]==b:
            merged.append(a+b)   # 合并为一个新token
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return merged

def train_bpe(texts: Iterable[str], vocab_size: int=5000, num_merges: int=None) -> Tuple[List[Tuple[str,str]], List[str]]:
    """
    训练字节级BPE
    """
    corpus = build_corpus(texts)
    base_tokens = [bytes([i]).decode('latin1') for i in range(256)]
    merges: List[Tuple[str,str]] = []
    merged_set = set()
    cur_vocab_size = 256

    # 若未指定合并次数，则由target vocab来决定
    merge_steps = num_merges or (vocab_size - 256)

    for _ in range(merge_steps):
        pfreq = pair_freq(corpus)
        if not pfreq:
            break

        # 找到出现频率最高的pair
        best_pair, _ = pfreq.most_common(1)[0]

        if cur_vocab_size + 1 > vocab_size:
            break

        merges.append(best_pair)

        # 对整个语料进行合并替换
        corpus = [merge_pair(word, best_pair) for word in corpus]

        # 将新token记入词表
        merged_set.add(best_pair[0]+best_pair[1])
        cur_vocab_size += 1

    # 追加特殊token
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # vocab = 特殊token+ 256 byte token +BPE合并的新token
    vocab_tokens = special_tokens + base_tokens + sorted(merged_set)

    return merges, vocab_tokens



# Tokenizer类
class DeepSeekV3Tokenizer:
    def __init__(self, merges: List[Tuple[str,str]], vocab_tokens: List[str]):
        self.merges = merges
        self.vocab_tokens = vocab_tokens

        # token ↔ id映射
        self.token2id = {tok:i for i, tok in enumerate(vocab_tokens)}
        self.id2token = {i:tok for tok,i in self.token2id.items()}

        # merges pair → 排序index
        self.ranks = {pair:i for i,pair in enumerate(merges)}

        # 特殊token
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

    def encode_chunk(self, chunk: str) -> List[str]:
        """
        对一个预分词做BPE编码：
        - 转字节token
        - 逐步应用merges
        - 处理OOV：未知token拆回字节或标记为<unk>
        """
        tokens = bytes2tokens(chunk.encode('utf-8'))

        # 应用PE并规则
        for pair in self.merges:
            new_tokens = []
            i = 0
            a,b = pair
            while i < len(tokens):
                if i<len(tokens)-1 and tokens[i]==a and tokens[i+1]==b:
                    new_tokens.append(a+b)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            tokens = new_tokens

        # OOV token拆回字节
        out = []
        for t in tokens:
            if t in self.token2id:
                out.append(t)
            else:
                # 拆分成字节token，如果字节token也不在词表 → <unk>
                out.extend([ch if ch in self.token2id else self.unk_token for ch in t])
        return out

    def encode(self, text: str, add_bos=False, add_eos=False, print_chunks=False):
        """
        编码完整文本：
        - 先预分词
        - 再逐chunk编码
        - 可选打印中间过程
        """
        ids = []

        if add_bos:
            ids.append(self.token2id[self.bos_token])
            if print_chunks: print(f"[Special] <bos> -> {self.token2id[self.bos_token]}")

        for chunk in pretokenize(text):
            toks = self.encode_chunk(chunk)
            chunk_ids = [self.token2id.get(t, self.token2id[self.unk_token]) for t in toks]

            if print_chunks:
                readable = []
                for t in toks:
                    try:
                        # 尝试恢复utf-8
                        r = tokens2bytes([t]).decode('utf-8', errors='ignore')
                        readable.append(r if r else t.encode('latin1').hex())
                    except:
                        readable.append(t.encode('latin1').hex())

                print(f"[Chunk] \"{chunk}\" -> {readable} -> IDs: {chunk_ids}")

            ids.extend(chunk_ids)

        if add_eos:
            ids.append(self.token2id[self.eos_token])
            if print_chunks: print(f"[Special] <eos> -> {self.token2id[self.eos_token]}")
        return ids

    def decode(self, ids: Iterable[int]):
        """
        将ID序列还原为utf-8文本：
        """
        byte_seq = bytearray()
        for i in ids:
            tok = self.id2token.get(i, self.unk_token)
            if tok in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            byte_seq.extend(tokens2bytes(list(tok)))
        return byte_seq.decode('utf-8', errors='replace')

    def save(self, vocab_path: str, merges_path: str):

        # 保存vocab（token2id）
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

        # 保存merges：每个token用base64
        merges_b64 = []
        for a, b in self.merges:
            a_bytes = a.encode('latin1')
            b_bytes = b.encode('latin1')
            merges_b64.append((
                base64.b64encode(a_bytes).decode('ascii'),
                base64.b64encode(b_bytes).decode('ascii')
            ))

        with open(merges_path, 'w', encoding='utf-8') as f:
            json.dump(merges_b64, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, vocab_path: str, merges_path: str):

        # 加载vocab
        with open(vocab_path, 'r', encoding='utf-8') as f:
            token2id = json.load(f)
        vocab_tokens = [None] * (max(token2id.values()) + 1)
        for tok, idx in token2id.items():
            vocab_tokens[idx] = tok

        # 加载merges（base64 → bytes → latin1）
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_b64 = json.load(f)

        merges = []
        for a_b64, b_b64 in merges_b64:
            a = base64.b64decode(a_b64).decode('latin1')
            b = base64.b64decode(b_b64).decode('latin1')
            merges.append((a, b))
        return cls(merges, vocab_tokens)


# 提供训练函数
def train_tokenizer(texts, vocab_size=5000, num_merges=None):
    merges, vocab_tokens = train_bpe(texts, vocab_size=vocab_size, num_merges=num_merges)
    return DeepSeekV3Tokenizer(merges, vocab_tokens)

# 示例
if __name__ == "__main__":
    texts = [
        "Transformer是AI的核心技术。",
        "DeepSeek分词器支持中文、英文、emoji等多语言。",
        "Hello, 世界! 🌍🚀",
    ]

    print("训练 Tokenizer (vocab_size=1024)")
    tokenizer = train_tokenizer(texts, vocab_size=1024)
    print(f"完成训练，词表大小: {len(tokenizer.vocab_tokens)}")
    print("-"*50)

    txt = "注意力机制是AI的核心技术。 🚀 🚀"
    print(f"编码文本: {txt}")
    ids = tokenizer.encode(txt, add_bos=True, add_eos=True, print_chunks=True)

    print("-"*50)
    print("Token ID:", ids)
    decoded = tokenizer.decode(ids)
    print("解码结果:", decoded)
    print("是否可逆:", decoded == txt)
```

*写到最后，衷心感谢每一位提出宝贵意见的小伙伴～ 你们的反馈让这份教程变得更好！🤝🤝*
