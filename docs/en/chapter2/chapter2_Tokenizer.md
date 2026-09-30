# Chapter 2: Tokenizer

The tokenizer is often considered part of a large language model (LLM), but it actually has a relatively independent design and training process.

An intuitive way to understand it: **the model doesn't directly "read text" — it can only process numbers**. What the tokenizer does is split raw text into tokens and map these segments to corresponding numeric IDs for the model to use in subsequent computation. Let's use an analogy:

The model works like an "automated express sorting pipeline," where the tokenizer is more like a **smart sorting and packaging center with empirical rules**. Imagine a scenario — we need to ship a mixed-item "parcel" that contains many frequently recurring combinations (like recurring words, roots, and subwords in language). But the "express company's" transport system (the LLM) has a limitation: it doesn't understand specific items, only recognizes ID numbers, and prefers processing standard objects (tokens). So before entering the transport system, a smarter sorting center (tokenizer) **breaks down these recurring combinations and converts them into the system's uniquely recognizable ID sequence**:

- **Splitting items (text segmentation):** Not randomly splitting, but finding "frequently co-occurring combinations" — for example, "火锅" won't be split into "火 + 锅" because it's more common as a whole;
- **Reusing existing packaging (subword modeling):** If a large uncommon item appears, it's split into common smaller component combinations, like "超导材料" might be split into "超导 + 材料";
- **Dynamically deciding granularity (key point):** `Common combinations → packaged as a whole unit; rare combinations → split into multiple small pieces` — this is exactly what BPE, WordPiece, and other tokenizers do;
- **Unified numbering (token ID):** Each "standard item" is assigned a fixed number;
- **Output numeric sequence (token sequence):** What is finally handed to the model is a sequence of numbers, not the original text.

**💡 The key points behind this analogy:**

- **Tokenization algorithms are not unique**: Different algorithms (BPE, WordPiece) produce different segmentation results;
- **The tokenizer affects model capability**: Tokens that are too "fragmented" or too "coarse" affect both expression efficiency and generalization ability;
- **It is an independently optimized module**: The vocabulary (vocab) is first trained on large-scale text, then **fixed for model use**.

*📖 Tip: Taking the segmentation of `incredible` as an example — `['i','n','cr','e','dible']` is too fragmented, `['incredible']` is too coarse. The ideal split is `['in','credible']`, which can be reused across different words, controlling total token count while improving embedding mapping ability.*

> In one sentence — **the LLM is responsible for "understanding and generating," while the tokenizer is responsible for "turning language into structures the model can understand and reuse."**

<div align="center">
   <img width="1000" height="600" alt="1" src="https://github.com/user-attachments/assets/bc838c76-7eff-4479-a760-ef404fc48e89" />
   <p>Figure 2.1 Tokenizer and LLM</p>
</div>

> Note that this process is not merely encoding conversion — it implicitly encodes a model's "world segmentation" strategy: the basic units the model ultimately "sees" may be characters, words, or more commonly, subword fragment encodings. This representation directly affects sequence length, information density, and semantic composition, further impacting model training efficiency and performance.

---

## Learning Objectives

This section focuses on the tokenizer — a seemingly simple module with profound impact on LLM behavior:

1. [Understand the basic principles and training processes of 4 tokenization algorithms (BPE, WordPiece, Unigram, SentencePiece)](#21-training-a-tokenizer)
2. [Analyze the differences between tokenizer implementations based on different algorithms, and why BPE is popular](#22-common-tokenizers)
3. [Hands-on practice with actual tools (e.g., DeepSeek's tokenizer implementation)](#23-analyzing-deepseeks-tokenizer)

More than "how to use," this section focuses on a more fundamental question:

> **What problem does the tokenizer actually solve? And how do we design a *good* tokenizer?**

The following layers progressively address this core question:
- How is the tokenizer "learned" from data? What are the core ideas of different tokenization algorithms (BPE, Unigram LM, etc.)?
- How do different tokenization strategies (different splitting approaches) affect model capability and further influence semantic modeling and composition?
- How is the tokenizer implemented in practice?

*Before starting, try out 👉 [tiktokenizer](https://tiktokenizer.vercel.app/?model=gpt-4-1106-preview). Then let's begin!*

## 2.1 Training a Tokenizer

Although the tokenizer is often treated as part of the model, it is actually prepared separately in advance: raw text is cleaned and organized, then through statistical methods or subword rules, a `token → numeric ID` mapping table (the vocabulary, or vocab) is summarized. Afterward, any input text is converted according to this table into discrete numeric sequences rather than human-readable text.

```
Training a tokenizer for modern LLMs can be broken into four steps: Prepare Corpus → Initialize Base Units (varies by algorithm, may be optional) → Statistics and Iterative Merging → Output Artifacts for Encoding and Decoding.
```

<div align="center">
<img src="https://github.com/user-attachments/assets/14b349a5-167d-4f66-825b-2951e87d0dc6"/>
   <p>Figure 2.2 Tokenizer Training Process</p>
</div>

### 2.1.1 Preparing the Corpus

**Step 1** — Collect diverse text covering the target application scenarios, so the trained vocabulary has good generalization ability for downstream tasks:

- Prepare different types of textual information — novels, essays, poetry, and other descriptive styles;
- Text in multiple languages — Chinese, English, Korean, French, etc.

In multilingual or mixed-corpus scenarios, calculate the proportion of each language, and evaluate whether to oversample or directionally retain low-resource languages to prevent the vocabulary from being dominated by high-frequency languages. Otherwise, uneven corpus type and language distribution will exacerbate token fragmentation for low-resource languages, potentially degrading their task performance. For example, suppose we want a tokenizer that supports four languages, and the collected raw unprocessed corpus proportions are:

| Language | Corpus Size |
| :--- | ---: |
| Chinese | 200 GB |
| English | 150 GB |
| French | 10 GB |
| Korean | 5 GB |

> This is a typical multilingual corpus imbalance scenario. If the above corpus is directly mixed without adjustment for tokenizer training, the statistical process will be dominated by Chinese and English, causing French and Korean common character strings to fail to enter high-frequency statistics during the merging stage, unable to occupy sufficient vocabulary space. Ultimately, French and Korean will have many over-split tokens in the vocab, forming severe fragmentation, and downstream LLM performance on French and Korean tasks will significantly degrade.

Therefore, when preparing the corpus, first calculate corpus proportions by language and set a reasonable sampling strategy based on target capabilities. For example, adjusting the ratio to `Chinese:English:French:Korean = 4:4:1:1` or `using a fully balanced strategy`. By downsampling high-resource languages or oversampling and augmenting low-resource languages, a training corpus more aligned with the target distribution can be obtained.

**Step 2** — Cleaning and normalizing raw text is a necessary step, including removing or masking irrelevant metadata, fixing or deleting garbled and invalid characters, standardizing character encoding, and pre-processing sensitive or private information for anonymization and compliance checks — identifying which information cannot be used for training and recording data sources and licenses.

**Example of NER-based data anonymization output analysis:**

Before: `小明的邮箱是test111@gmail.com，电话是13312311111，现在居住于重庆两江新区的xxx小区。`
After: `[NAME]的邮箱是[EMAIL]，电话是[PHONE]，现在居住于[PLACE]。`

If sentences contain sensitive information such as names, phone numbers, and addresses, anonymization processing is needed. **Typically in data processing pipelines, high-certainty information (e.g., phone numbers, emails) is processed first to eliminate interference, followed by less standardized information like names, thereby reducing overall missed detection risk caused by non-standard expressions and format diversity.**

Notably, data anonymization is not only for privacy protection and compliance — it also helps improve the stability of downstream text modeling and tokenization processes. High-cardinality information like names, phone numbers, and ID numbers, if directly retained in the corpus, often appears in near-unique forms. Such information, statistically low-frequency or even single-occurrence noise, interferes with the statistical efficiency of tokenization algorithms (BPE, Unigram) when learning high-frequency token structures.

> This **effectively reduces meaningless diversity in the corpus**, allowing the tokenizer to focus more on modeling statistically patterned language structures, thereby improving vocabulary utilization efficiency and consistency. *From Shannon's information theory perspective, data anonymization can be viewed as a structured denoising process — by compressing or reducing high-entropy but low-semantic-value signals (like specific identity information), increasing the proportion of effective signal in the corpus, helping the LLM in subsequent training to more readily learn reusable semantic structures rather than memorize incidentally encountered instance details.*

*📖Tip: High entropy means that under given context conditions, the probability distribution is relatively dispersed, resulting in higher uncertainty and prediction difficulty.*

And downstream tasks themselves may require recognizing real entities (such as information extraction), so excessive anonymization would weaken the training signal. Therefore, a reasonable strategic trade-off between **protecting privacy** and **preserving key semantic information** is needed.

**Step 3** — It is recommended to reserve a small portion of un-tokenized validation corpus (e.g., 99:1 train-validation split), used during the training process to evaluate the tokenizer's encoding efficiency and statistical indicators such as average token length on real text.

### 2.1.2 Pre-tokenization Stage

**Step 1** — The main task of pre-tokenization is to split raw text into statistical, mergeable basic units, such as characters, bytes, or Unicode segments. Common strategies include whitespace and punctuation-based splitting, splitting by Unicode category, or directly adopting byte-level splitting. Note that not all tokenizers require the user to explicitly perform pre-tokenization. *For example, SentencePiece-based tokenizers have built-in normalization and pre-tokenization logic, so no extra external pre-tokenization step is needed.*

**Whitespace and punctuation-based splitting strategy**: In a complete sentence, encountering whitespace or punctuation (.,!?[]{}...) can be split into independent tokens. This method is suitable for most pre-tokenization processing.

```python
import re
def part(text):
    text = re.sub(r'([.,!?;:()"\'\[\]{}])', r' \1 ', text)
    tokens = text.split()
    return tokens
s = "I like Datawhale."
print(part(s))  # ['I', 'like', 'Datawhale', '.']
```

**Unicode category-based splitting strategy**: Automatically split by character Unicode category (letters, digits, punctuation, Chinese, special characters, etc.), with different categories entering different token chunks — *in one sentence, character types within the same token are all consistent*. This method is naturally suited for multilingual mixed text and provides a reliable baseline segmentation result.

Input: `Hello👋👋，Datawhale成立于2018年！！！`
Output: `['Hello', '👋👋', '，', 'Datawhale', '成立于', '2018', '年', '！！！']`

**Byte-level splitting strategy**: First split each character into UTF-8 byte sequences, independent of language type or character — each single byte sequence becomes one independent token.

```python
def tokenize_byte_level(text):
    tokens = []
    for ch in text:
        utf8_bytes = ch.encode("utf-8")
        hex_bytes = [f"{b:02X}" for b in utf8_bytes]
        tokens.extend(hex_bytes)
    return tokens
s = "All for learners！"
print(tokenize_byte_level(s))
# ['41', '6C', '6C', '20', '66', '6F', '72', ..., 'EF', 'BC', '81']
```

*English characters and spaces are ASCII, 1 byte each in UTF-8. The fullwidth exclamation mark `！` is not ASCII, requiring 3 bytes in UTF-8 (`！ -> EF BC 81`)*

**The relationship between Unicode and UTF-8:**

`Unicode` is like the "ID card" issued to every character worldwide — whether it's English A, Chinese 中, or emoji 😄, each character has a unique number in Unicode. But the "ID number" itself is just an abstract number; computers cannot directly store it.

`UTF-8` is like the specific way to write this character's "ID number" into computer storage. It specifies how many bytes and what rules to use for writing. Common English characters need only 1 byte in UTF-8, while Chinese typically needs 3 bytes. Both Unicode and UTF-8 can represent different categories of characters; together they enable natural language to be accurately stored, transmitted, and parsed — serving as the "bridge" for human-computer interaction.

> Unicode is the "encoding standard" — assigning unique code points to each character; UTF-8 is the "encoding format" — responsible for converting code points to byte sequences. **A major advantage of UTF-8: ASCII characters (0-127) in UTF-8 are encoded identically to ASCII and only occupy 1 byte. This backward compatibility makes it more commonly used than UTF-16, UTF-32, and other encoding methods.**

**In LLM token segmentation, common strategies include:**
(1) Rule-based pre-tokenization (e.g., splitting by whitespace and punctuation);
(2) Unicode category-based segmentation (e.g., continuous Chinese characters, continuous Latin letters or digits);
(3) Lower-level UTF-8 byte-level splitting.

Methods (1) and (2) have limitations in scenarios where text lacks explicit delimiters or contains long stretches of the same character category. In these cases, the pre-processing stage struggles to effectively split sentences. To ensure text remains encodable, fallback to finer granularity (near character-level splitting) may be forced. In contrast, the UTF-8 byte-level strategy has the strongest generality — it uniformly splits any text into byte sequences, fundamentally reducing out-of-vocabulary (OOV) problems and covering arbitrary character sets. However, because it starts at the finest granularity, training typically requires more rounds of co-occurrence statistics and merging to compress scattered bytes into compact, semantically meaningful tokens, balancing Transformer computational efficiency and semantic representation.

*📖Tip: OOV (Out-of-Vocabulary) occurs when an LLM processes new, real-world text and encounters a token not present in its vocabulary — that token is treated as OOV.*

<div align="center">
<img width="1200" height="612" alt="attention impact" src="https://github.com/user-attachments/assets/97e514fd-5def-405d-95c2-9a9208d6f067" />
   <p>Figure 2.3 Impact of token sequence length on attention mechanism</p>
</div>

From Figure 2.3, the impact of token count on the attention mechanism is mainly reflected in two aspects:
- Computational complexity grows at $O(N^2)$;
- Constrained by context window capacity.

When tokens are more fragmented (i.e., higher total token count), attention mechanism efficiency decreases.

**Step 2** — For most languages with whitespace word boundaries, **preliminary splitting by word boundaries and punctuation using regular expressions** can be done first. For Chinese, Japanese, and other languages without whitespace word boundaries, character-by-character or character-based initial units are typically used to ensure coverage.

**Step 3** — The basic unit sequence generated by pre-tokenization will serve as input for subsequent statistical merging. This sequence and corresponding position information must be saved for repeated efficient updating during training.

```python
def btp_hex_list(text):
    """UTF-8 byte-level pre-tokenization, returns tokens with position info and hex string list"""
    tokens = []
    t = []
    for idx, char in enumerate(text):
        utf8_bytes = char.encode('utf-8')
        hex_bytes = ' '.join(f"{b:02X}" for b in utf8_bytes)
        tokens.append({'char': char, 'bytes': hex_bytes, 'start': idx, 'end': idx + 1})
        t.extend([f"{b:02X}" for b in utf8_bytes])
    return tokens, t

text = "Hi，你好🐋"
tokens, t = btp_hex_list(text)
print(t)  # ['48', '69', 'EF', 'BC', '8C', 'E4', 'BD', 'A0', 'E5', 'A5', 'BD', 'F0', '9F', '90', '8B']
```

### 2.1.3 Statistics and Iterative Merging

**Step 1** — Subword candidate statistics: Iterate through the corpus to collect statistical information for subsequent decision-making. The specific method varies by algorithm:

- **BPE**: Count the frequency of adjacent pairs in the current character/subword sequence, greedily merge the highest-frequency adjacent pair each time, iteratively building the vocabulary — decisions are purely frequency-based.
- **WordPiece**: Evaluate the contribution of merging or retaining certain subword combinations to corpus likelihood (i.e., language model performance), choosing merge operations that significantly improve corpus fit.
- **Unigram**: Start from an oversized seed vocabulary, initialize the probability of each token.
- **SentencePiece**: A language-agnostic subword tokenization framework providing a unified training and encoding pipeline, supporting multiple algorithms (BPE and Unigram). These algorithms operate independently within the same framework, not directly fused; their complementarity lies in applicability differences across tasks and data conditions.

*📖Tip: Tokenization algorithms determine the token splitting strategy, while the tokenizer combines the algorithm with vocabulary, encoding mechanism, and other components to form a complete processing pipeline that converts raw text into model input sequences. Here we can think: what are the differences between these tokenization algorithms?*

| Algorithm | Use Case | Common Implementation | Typical LLMs / Models |
|-----------|----------|---------------------|----------------------|
| [BPE](https://arxiv.org/pdf/1508.07909) | Simple, efficient; good for large corpora; high compression for frequent subwords | Byte-level BPE, codepoint BPE | GPT-2/3/4 (tiktoken), LLaMA (improved BPE), RoBERTa |
| [WordPiece](https://huggingface.co/learn/llm-course/en/chapter6/6) | Vocabulary size control; reduces OOV; suitable for MLM models | Character-level, codepoint-level | BERT (original), DistilBERT, early RoBERTa (compatible) |
| [Unigram LM](https://arxiv.org/abs/1808.06226) | Probabilistic subword modeling; friendly to low-frequency words; strong multilingual adaptation | SentencePiece (Unigram mode), byte-fallback support | T5, mT5, UL2, Gemma (Google family) |
| [SentencePiece (framework)](https://arxiv.org/pdf/1804.10959) | Language-agnostic; end-to-end tokenizer training; suitable for multilingual | **BPE or Unigram training framework** (not a new algorithm), byte-fallback support | LLaMA (using SP-BPE), DeepSeek series |

*Overall, these four subword tokenization algorithms each have their characteristics — no single one is absolutely best. When choosing an algorithm, decisions should be based on specific text content (corpus distribution), task type (understanding or generation), vocabulary size, and whether multilingual processing is needed, to enable the trained LLM model to perform optimally.*

>[T5](https://ai.younglimit.com/deep-learning/nlp-pretraining/encoder-decoder-bart-and-t5/t5) (Text-to-Text Transfer Transformer) is a Transformer-based pretrained language model framework proposed by Google in 2019. Its core idea is to unify all NLP tasks into a "text-to-text" format, enabling unified multi-task modeling. Building on this, Google subsequently proposed a series of extended models, including:
>- mT5: A T5 extension pretrained on multilingual corpora, supporting cross-lingual tasks;
>- UL2: A unified pretraining paradigm that mixes multiple denoising objectives (similar to BERT and T5's objectives), improving the model's generalization ability across different downstream tasks.
>
>From T5's task representation unification, to mT5's cross-lingual extension, to UL2's **unified joint optimization** of multiple denoising objectives during pretraining — this evolutionary path progressively achieves a paradigm upgrade from **"task-level unification" to "data distribution expansion" to "training signal unification"**, enabling language models to **learn more universal conditional distributions in a shared parameter space**, thereby significantly enhancing their cross-task and cross-distribution generalization capability.
>*👉 The improvement of LLM capabilities can be seen as a process of "ever-increasing unification"*

**Step 2** — Iterative updating of subword candidate statistics:

A brief analysis of the four iterative algorithms — BPE, WordPiece, Unigram, and SentencePiece:

- **BPE algorithm**: Uses Step 1's subword candidate statistics as initialization data, performs single-token merges to form new tokens, then dynamically counts co-occurrence frequencies over multiple iterations to obtain new tokens.
- **WordPiece algorithm**: Dynamically counts the occurrence of all adjacent subword pairs in the current vocabulary during iteration. The key is not simply merging the highest-frequency pair, but preferentially selecting the subword pair that maximally improves overall corpus likelihood, forming more representationally meaningful tokens. A commonly used approximate score:

$$\text{score}(A,B)=\frac{P(A,B)}{P(A) \times P(B)}$$

> This ratio measures whether the "association" between A and B is stronger than expected if they appeared independently. If score > 1, the combination of A and B is more meaningful than random independent appearance, making it more likely to be merged by WordPiece. [Google has not publicly disclosed WordPiece algorithm details; this references HuggingFace's related principle introduction](https://huggingface.co/learn/llm-course/en/chapter6/6)

- **Unigram algorithm**: Based on a subword probabilistic language model, defining a sentence's probability as the sum of probabilities of all its possible tokenization paths. The core idea is to optimize subword probabilities through iteration to maximize overall corpus likelihood. The algorithm uses the Expectation-Maximization (EM) method, primarily consisting of two steps:

 ① **E-step (Expectation step)**: Under the current vocabulary and subword probabilities, compute the most likely tokenization or top-n high-probability tokenizations for each sentence in the corpus, and accordingly estimate the expected usage count of each subword in the corpus.

 ② **M-step (Maximization step)**: Based on E-step statistical results, update each subword's probability to maximize overall corpus likelihood.

 ③ In each iteration, the model prunes (eliminates) lower-probability tokens, e.g., discarding the bottom 10%-20%, gradually converging to a smaller, optimized vocabulary until reaching the preset target vocabulary size. This method relies more on probabilistic modeling compared to BPE or WordPiece, flexibly handles subwords of different lengths, and naturally preserves the high-frequency segments that best explain the corpus.

*📖Tip: "Maximizing corpus likelihood" means, during tokenizer training, learning a set of token segmentation methods and vocabulary that maximizes the overall probability of the training corpus, allowing character sequences in the corpus to be represented by higher-probability, more common token combinations. Colloquially, "cutting text into the smoothest segments most consistent with language statistical patterns, so the same sentence can be stably and compactly split into consistent token sequences in different contexts."*

- **SentencePiece algorithm**: An independent tokenization tool and implementation library capable of directly training subword models from raw text, therefore users do not need to explicitly perform external [pre-tokenization steps](#212-pre-tokenization-stage). Internally, it encodes whitespace, word boundaries, and other information as special characters (the commonly seen `▁` in training output represents word-initial space), thereby making whitespace itself an object of vocabulary construction. It then applies BPE or Unigram algorithms on these initial tokens to generate the final token vocabulary and mapping.

Throughout this iterative process, special control tokens (such as `<PAD>`, `<UNK>`, `<CLS>`, `<MASK>`, etc.) must be preserved and not participate in modification during tokenizer iterative updates. This ensures their word-number mapping remains fixed, and encoded discrete numeric sequences can be accurately restored to original text. These tokens will not be split or overwritten during statistical merging or probability optimization, effectively reducing the appearance of fragmented tokens. Regardless of whether BPE, WordPiece, SentencePiece, or Unigram is used, this strategy applies — helping protect the integrity of key tokens and ensuring consistency in model training and inference.

> 💡 Note: The tokenizer optimizes the *representation* of input — it does not itself enable the model to "understand" natural language. True "language understanding" ability is acquired during subsequent Transformer training. However, more reasonable token segmentation makes input distributions clearer and more consistent, indirectly improving model training efficiency and final performance.

**Summary**: The core of tokenizer training is <ins>iteratively updating candidate subwords → controlling vocabulary size or convergence criteria → monitoring quality indicators</ins>. Different algorithms only differ in "candidate generation method" and "iterative update strategy."

### 2.1.4 Output Artifacts for Encoding and Decoding

**Step 1** — Export core artifacts. Regardless of which tokenization algorithm is used, at least two key files must be exported after training:

<div align="center">
   <img width="1000" height="500" alt="vocab and merges files" src="https://github.com/user-attachments/assets/1ea639e8-81f9-49a5-b5d7-3df32fe129a2" />
   <p>Figure 2.4 Example of vocab and merges files</p>
</div>

- **vocab file**: Records all tokens and their corresponding IDs — the core index for the encoder and decoder.
- **merges file**: Records all subword merge rules or probability models in order. Together they determine the tokenizer's encoding and decoding logic and ensure encoding reversibility.

**Step 2** — Validation and evaluation before downstream use:

After applying the tokenizer to a portion of the validation set, it is recommended to collect the following key metrics:
- `Average token count and maximum length distribution` — directly affects memory usage, training speed, and inference efficiency.
- `Fragmentation status` — check whether key entities and specialized terms are over-split, avoiding impacts on model understanding.
- `Cross-language token balance` — in multilingual tasks, ensure sufficient token support for common patterns across different languages.

> If vocabulary expansion is needed later (adding new domain terms, specialized words, brand names, etc.), it is recommended to prefer these approaches over complete tokenizer retraining: **incremental training**, **adding new merge entries**, **cleaning extremely low-frequency tokens**.
>
> After vocabulary expansion, a **regression test** should be performed to ensure backward compatibility with the old model and that the encoded digital representations can be restored to the original input text without token allocation conflicts or token exhaustion.

*Having read this far, we can answer the initial question: how to evaluate a good tokenizer.*

---

## 2.2 Common Tokenizers

In the previous section, we learned about the tokenizer training process. Next, we will introduce several common tokenizer types and analyze their respective core ideas and characteristics.

### 2.2.1 Character Tokenizer

**Principle introduction**: This is the most intuitive and simplest tokenization method, splitting text into the smallest character units — letters (a, b, c) in English or individual characters (你, 好) in Chinese.

- **Advantages**:
  - **Extremely small vocabulary**: English only needs 26 letters + symbols; Chinese only needs commonly used characters (several thousand).
  - **No OOV problem**: Any rare word is composed of basic characters; "unknown words" won't appear.
- **Disadvantages**:
  - **Overly long sequences**: After converting to characters, sentence length increases severalfold, greatly consuming the LLM's precious context window, increasing Transformer computation memory consumption.
  - **Semantic sparsity**: Individual characters (e.g., 't') typically lack independent semantic meaning; the model needs deeper network layers to compose meaning.

**Character tokenizer code output analysis:**
Input: `hi，很好的，terrific！🐋`
Output: encoded IDs and compression ratio of ~0.47

### 2.2.2 Byte Tokenizer

**Principle introduction**: At the lowest level, computers store text essentially as **bytes**. In UTF-8 encoding, English typically occupies 1 byte, and Chinese characters typically require 3 bytes. The byte tokenizer operates directly on binary bytes.

- **Core logic**: No longer maintains a "character" vocabulary but maintains a base vocabulary of size 256 (0x00 to 0xFF).
- **Application**: Modern LLMs like GPT-4 and Llama typically don't use pure byte tokenization alone but use bytes as BPE's base unit (BBPE), fundamentally solving cross-language and special character (emoji, etc.) encoding issues.

**Comparison of BPE, character-level, and byte-level tokenizer effects:**
Input: `Hello, 🌍! 你好!`
Output: byte-level: 20 tokens, character-level: 13 tokens, BPE: 11 tokens. Compression ratios: byte=1.00, char=1.54, BPE=1.82.

Notably, **the byte-level tokenizer's compression ratio is always 1**, because each UTF-8 byte maps directly to one token, so token count = UTF-8 byte count:

$$\text{compression ratio} = \frac{\text{UTF-8 bytes}}{\text{token count}} = \frac{N}{N} = 1$$

**In other words, the byte-level tokenizer has absolutely no compression capability — each byte corresponds to one token, producing neither longer nor shorter word fragments.**

### 2.2.3 Word Tokenizer

**Principle introduction**: In early deep learning (RNN era), this was the most mainstream method. It splits text into words with independent semantics based on spaces (English) or word segmentation algorithms (Chinese).

- **Advantages**: Tokens retain complete semantic information (e.g., "apple" directly maps to one token ID).
- **Disadvantages**:
  - **Vocabulary explosion**: Words like `look, looks, looked, looking` are treated as 4 completely different IDs, resulting in enormous vocabulary sizes (hundreds of thousands to millions).
  - **Severe OOV problem**: Encountering unseen words (new names, coinages, etc.) can only be marked as `<UNK>`, leading to information loss and affecting LLM performance.

**Word tokenizer code example:**
Input: `It's so supercalifragilisticexpialidocious!👋👋`
Output: vocabulary size of 7, token IDs `[3, 2, 4, 0, 5, 0, 6, 1]`, compression ratio 6.375

### 2.2.4 BPE Tokenizer

**Principle introduction**: This is currently the most mainstream tokenization algorithm for LLMs (GPT, BERT, Llama, etc.). BPE attempts to find a balance between <ins>character-level (too fine)</ins> and <ins>word-level (too coarse)</ins>. **Core idea**: Count the frequency of adjacent character pairs in the corpus, iteratively merge the **most frequently occurring pair** into a new token.

**Simple BPE training example output analysis:**

Input: `test_text = "敏捷的棕色狐狸🦊"`
Output: BPE merges showing pair merging, encoding with `</w>` end-of-word markers.

In the BPE encoding stage, without `</w>`, the algorithm might incorrectly split `the` into `th`, `e` or incorrectly merge across words in subsequent operations. With `</w>`, `the` is represented as `['t', 'h', 'e', '</w>']` — BPE knows this is a complete word ending and won't incorrectly merge across words. During decoding, removing `</w>` restores `the`, ensuring correct original text recovery.

*Therefore, `</w>`'s core function is guaranteeing word integrity and making encoding reversible — the corresponding numeric sequence can be converted back to the original text.*

**Comparison table of 4 tokenizer types:**

| Tokenizer Type | Granularity | Vocab Size | OOV | Sequence Length | Representative Models |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Character | Fine | Small (100-5k) | None | Very Long | Char-RNN |
| Byte | Finer (bytes) | Very Small (~256-1k) | None | Long | GPT-2 |
| Word | Coarse | Very Large (>100k) | Severe | Short | Word2Vec, GloVe |
| **BPE** | **Medium (Adaptive)** | **Moderate (30k-100k)** | **Minimal** | **Moderate** | **GPT-4, Llama 3** |

Beyond `tokenizer choice` and `training corpus` directly affecting LLM input sparsity and representation efficiency, using large-scale, high-quality, and diverse corpora to train the tokenizer typically `reduces token fragmentation` — generating more common, more stable subword units, so the same text is encoded into fewer tokens, and within a fixed context window length, each token carries more actual information. This means the model can "see" more content within a limited window, alleviating information loss caused by context length constraints.

> Note: This depends on corpus coverage and quality. If the corpus is biased or over-merges rare words, it may instead harm the representation capability for minority languages or specialized terminology.

---

## 2.3 Analyzing DeepSeek's Tokenizer

DeepSeek models, especially the Coder series, are highly optimized for code and Chinese-English text. We will load the official DeepSeek Coder model tokenizer.

### 2.3.1 Loading the DeepSeek Tokenizer

```python
# Ensure transformers library is installed: pip install transformers torch
from transformers import AutoTokenizer
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Successfully loaded tokenizer for model: {MODEL_NAME}.")
print(f"Tokenizer vocabulary size V: {len(tokenizer.get_vocab())}")
```

### 2.3.2 DeepSeek Tokenizer Processing Logic

DeepSeek's optimized byte-level BPE vocabulary, through fine modeling of Chinese character distributions and code indentation, significantly shortens token sequence length. This design, while ensuring semantic integrity, greatly improves inference throughput. We will analyze Chinese text tokenization examples to intuitively see how it optimizes text sequence efficiency through high-frequency word cluster aggregation.

**Example analysis: Chinese text processing**

```python
chinese_text = "注意力机制是AI的核心技术。 🚀 🚀"
encoded_ids = tokenizer.encode(chinese_text, add_special_tokens=False)
tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
print(f"Original: {chinese_text}")
print(f"Encoded: {tokens}")
print(f"IDs: {encoded_ids}")
```

Token string segmentation results may visually differ from the original text. This is not an encoding error but because the LLM's **vocabulary** has <ins>insufficient coverage of certain characters or subwords</ins> during training (e.g., insufficient BPE training), preventing the model from generating corresponding tokens, making the readable form appear like "garbled text." By increasing training corpus volume or performing sufficient BPE training, a more complete token mapping vocabulary can be learned, resolving this issue so Chinese, English, emoji, and other characters can all be correctly encoded and decoded. The solution is to train BPE:

**DeepSeek's regularization processing + BPE training example output analysis:**
Input: `注意力机制是AI的核心技术。 🚀 🚀`
Output: The pipeline discretizes text into minimal semantic units (tokens) according to the preset tokenization algorithm and builds deterministic mappings to globally unique numeric IDs and underlying encodings. Identical characters (spaces, specific emoji 🚀) point to consistent IDs and encoding sequences throughout the text, ensuring feature representation stability.

From the code execution results, the tokenizer's `token ↔ id` mapping only describes "this token's content" without containing any position information about its location in the sentence. BPE and other statistics/probability-based tokenization algorithms are fundamentally based on co-occurrence frequency or probability distribution in the corpus, deciding how to merge common characters, bytes, or substrings into longer, higher-frequency tokens. These algorithms themselves do not understand sentence semantics — they are more like pure statistical modules, segmenting and merging character sequences through frequency or probability principles, providing the model (LLM) with stable and compact discrete input units.

> **Why does DeepSeek use Latin-1 encoding/decoding?**
>
> In DeepSeek's tokenization pipeline, the final processed output is digital tokens, but during the BPE tokenizer training phase, operations need to be performed on "characters." If UTF-8 encoding/decoding were used directly, multi-byte characters like Chinese characters or emoji, when split into individual bytes, would produce incomplete sequences — Python would error or substitute, causing information loss. Latin-1 is single-byte encoding, mechanically mapping each byte (0-255) to a Unicode character, ensuring any byte sequence can be completely and reversibly preserved, thus allowing BPE or other subword algorithms to treat bytes as characters for merging without data loss. Simply put, using Latin-1 is for safely treating raw bytes as characters within the tokenizer, ensuring complete encoder-stage information integrity.

### 2.3.3 Quick Hands-on with DeepSeek Tokenizer

<div align="center">
   <img width="1788" height="1077" alt="DeepSeek tokenizer" src="https://github.com/user-attachments/assets/7e7b40df-e9c0-43ed-896a-40679a8046c8" />
   <p>Figure 2.5 DeepSeek Tokenizer visualization</p>
</div>

Input text: `你好 , hello,  world !  🌏 ！`

> Through this [DeepSeek tokenizer visualization](https://tiktokenizer.vercel.app/?model=deepseek-ai%2FDeepSeek-R1), the last column on the right shows the corresponding token ID mapping. You can see that for spaces separately segmented as tokens, even at different positions, their ID values are all 223.

---

## 2.4 Summary

In this chapter, we learned the basic principles of tokenizers and their training processes. Finally, we introduced the DeepSeek Coder model's tokenizer and its application in Chinese text processing.

BPE-based tokenizers are widely adopted in early and some current LLMs. The core reason is that this method is primarily a frequency-driven greedy merging strategy — simple to implement and efficient to train.

> However, BPE is not the only choice. Methods like WordPiece, Unigram, and SentencePiece are also widely used in different models, just with different modeling assumptions and optimization objectives.

The tokenizer serves as the LLM's input-output interface, the bridge between the model and external text. **Its core role is converting continuous natural language text into discrete token sequences and mapping them to corresponding integer IDs, thus serving as input for Transformer and other model architectures.** Within the model, these discrete tokens are further mapped to vector representations (embeddings) and undergo multi-layer computation to form context-dependent hidden state representations. These representations form the basis for the model's prediction and generalization problem-solving.

---

## 2.5 Reflection Questions

**Basic questions:**

1. What are the differences between the four tokenization algorithms — BPE, WordPiece, Unigram, and SentencePiece? Why have some current LLM tokenizers chosen BPE-based implementations?

2. How to measure "a good tokenizer"? What is the trade-off between compression efficiency and semantic consistency?

3. How does the tokenizer affect actual LLM final performance?

**Advanced questions:**

1. Research has shown that visual features can enhance LLM understanding, but not for all language tasks. Can we seek a dynamic "balance point" between visual representations and discrete tokens: simultaneously providing both types of representations to the model, and drawing on MoE ideas to design a lightweight dynamic router, enabling the model to automatically select or fuse the most appropriate embedding method across different tasks or text segments, thereby significantly improving cross-scenario adaptation capability?

> The discreteness of text tokens limits expressive capacity, while visual tokens can provide high-density continuous compressed representations but are not suitable for all language scenarios. Therefore, exploring an MoE-style multi-representation mechanism — enabling the model to dynamically select text, visual, or hybrid representations based on the task — to obtain richer and more scenario-adaptive representations may also be worth considering.

2. Can we design an "adaptive tokenizer" that is first trained separately from the LLM during the training phase, then combined with the model through a special mechanism, enabling it to still dynamically learn and optimize token segmentation strategies during downstream tasks?

> For instance, consider a feedback-driven dynamic vocabulary augmentation method — **the core is cross-model semantic representation distillation and transfer**. It does not use traditional output probability distillation; instead, a teacher model extracts precise semantic vectors for new concepts based on user feedback. Through a mapping adapter, this vector is projected into the student model's embedding space, achieving an instant "patch" to the student's vocabulary matrix, thereby enabling the student model to recognize and process new tokens in a zero-shot manner.

3. Using micro-subword modules, meta-learning, or reinforcement learning methods, could the tokenizer automatically discover the most suitable token segmentation approach from a small number of conversations or task samples, thereby reducing downstream task data dependency while improving model robustness and generalization?

> This approach is somewhat like semi-supervised learning — the tokenizer itself is "learning how to learn," so even with only a small number of conversation samples, it can find more suitable token segmentation methods, making the model's language understanding more efficient, and less likely to be stumped by new words or limited data.

---

## References
- [HuggingFace: Introduction to Four Tokenizer Algorithms](https://huggingface.co/learn/llm-course/en/chapter6/1)
- [BPE Algorithm (Sennrich et al., 2015)](https://arxiv.org/pdf/1508.07909)
- [CS336 Lecture 1 Course Materials](https://cs336.stanford.edu/)

## Appendix (Code Experiments)

### Appendix 1: NER-based Data Anonymization Example

This example uses Python to implement data anonymization based on Named Entity Recognition (NER):

*In simple terms, Named Entity Recognition is a fundamental task in Natural Language Processing (NLP) — it acts like a "target scanner" that automatically identifies and extracts meaningful entities (such as person names, locations, organizations, times, etc.) from large amounts of unstructured text.*

```python
# Initialize the NER pipeline
ner_pipeline = pipeline(
    "ner",
    model="ckiplab/bert-base-chinese-ner",
    grouped_entities=True  # Merge adjacent entity segments of the same type, e.g., "重" and "庆" merged into "重庆"
)

def ner_mask(text: str) -> str:
    """
    Semantic-level anonymization using deep learning models (person names and locations)
    """
    entities = ner_pipeline(text)
    spans = []
    
    # Extract entities and their positions identified by the model
    for ent in entities:
        label = ent["entity_group"]
        start = ent["start"]
        end = ent["end"]

        # Map entity types to anonymization placeholders
        if label == "PER":  # Person name
            spans.append((start, end, "[NAME]"))
        elif label == "LOC":  # Location/Address
            spans.append((start, end, "[PLACE]"))

    # Sort logic: by start position ascending; if same start, by length descending (prioritize longer entities)
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Resolve conflicts: remove overlapping or contained entity intervals
    filtered_spans = []
    last_end = -1
    for start, end, tag in spans:
        if start >= last_end:  # Only keep if the current entity starts after the previous entity ends
            filtered_spans.append((start, end, tag))
            last_end = end

    # Reconstruct text based on filtered intervals
    result = []
    last_idx = 0
    for start, end, tag in filtered_spans:
        result.append(text[last_idx:start]) # Append non-sensitive parts
        result.append(tag)                  # Append placeholder
        last_idx = end
    result.append(text[last_idx:])          # Append remaining text

    return "".join(result)


# 2. Anonymization pipeline architecture design
class DesensitizationPipeline:
    """
    Anonymization task manager: allows adding multiple processing steps in order
    """
    def __init__(self):
        self.steps: List[Callable[[str], str]] = []

    def add_step(self, func: Callable[[str], str]):
        """Add a processing step (e.g., regex replacement, NER replacement, etc.)"""
        self.steps.append(func)

    def run(self, text: str) -> str:
        """Execute all anonymization steps in order"""
        for step in self.steps:
            text = step(text)
        return text

# 3. Specific processing step implementations
def normalize_text(text: str) -> str:
    """Text preprocessing: strip leading/trailing whitespace"""
    return text.strip()

# High-certainty rules (strong features: phone numbers, emails)
def mask_phone(text: str) -> str:
    """Regex match 11-digit Chinese phone numbers"""
    return re.sub(r'1[3-9]\d{9}', '[PHONE]', text)

def mask_email(text: str) -> str:
    """Regex match common email formats"""
    return re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)

# Medium-certainty rules (keyword context-based)
def mask_address(text: str) -> str:
    """Address matching guided by keywords such as "居住于" (residing at)"""
    return re.sub(
        r'(居住于|现居住于|现居于|地址)([一-龥A-Za-z0-9]+)',
        r'\1[PLACE]',
        text
    )

# Low-certainty rules (simple fallback based on syntactic structure)
def mask_name(text: str) -> str:
    """
    Fallback strategy: match "某某某的" structure appearing at sentence start or after punctuation
    Note: prone to false positives, typically placed after NER step as a supplement
    """
    return re.sub(
        r'(?:(?<=^)|(?<=[，。！？]))([一-龥]{2,3})(的)',
        r'[NAME]\2',
        text
    )

def clean_punctuation(text: str) -> str:
    """Post-processing step: normalize punctuation as needed"""
    return text
```

[Complete Data Anonymization Code](https://github.com/datawhalechina/diy-llm/blob/main/docs/en/chapter2/de_identified_data_processing.py)  

### Appendix 2: Unicode-based Segmentation

```python
import unicodedata

def get_char_category(ch: str) -> str:
    # Get the Unicode standard category (e.g., 'Lu' for uppercase letters, 'Po' for other punctuation)
    cat = unicodedata.category(ch)

    # Determine if character is Chinese (common basic CJK range)
    if '一' <= ch <= '鿿':
        return "CJK"
    
    # Determine if character is a digit
    if ch.isdigit():
        return "DIGIT"
    
    # Determine if character is an English letter (or letter from other languages)
    if ch.isalpha():
        return "ALPHA"

    # Determine if character is punctuation (Unicode categories starting with 'P' are all punctuation)
    if cat.startswith("P"):
        return "PUNCT"

    # Other characters (emoji, spaces, control characters, etc.) are unified as OTHER
    return "OTHER"


def segment_by_unicode_category(text: str):
    if not text:
        return []
    segments = []
    # Initialize buffer with the first character
    buffer = [text[0]]
    # Get the first character's category as the initial reference
    prev_type = get_char_category(text[0])

    # Phase 1: Linear scan through text, segmenting by category
    for ch in text[1:]:
        curr_type = get_char_category(ch)

        # If current character category matches previous, add to buffer for merging
        if curr_type == prev_type:
            buffer.append(ch)
        else:
            # Category changed — store buffered content as one segment in result list
            segments.append(("".join(buffer), prev_type))
            # Reset buffer, start recording new category characters
            buffer = [ch]
            prev_type = curr_type

    # Process the last segment remaining in the buffer
    segments.append(("".join(buffer), prev_type))

    # Phase 2: Extract the segmented string content
    tokens = [seg for seg, _ in segments]
    return tokens

# Test run
if __name__ == "__main__":
    # Test string contains: English, emoji, Chinese punctuation, Chinese, digits, English punctuation
    s = "Hello👋👋，Datawhale成立于2018年！！！"
    result = segment_by_unicode_category(s)
    print("Original text:", s)
    print("Segmentation result:", result)
```

### Appendix 3: Character Tokenizer

```python
# Character Tokenizer
class CharacterTokenizer:
    def __init__(self):
        pass  # No extra parameters needed, directly use ord and chr

    def encode(self, text):
        """
        Encode string as list of character indices (Unicode code points)
        """
        return [ord(ch) for ch in text]

    def decode(self, indices):
        """
        Decode list of indices back to string
        """
        return ''.join([chr(i) for i in indices])

# Test code
if __name__ == "__main__":
    tokenizer = CharacterTokenizer()
    string = "hi，很好的，terrific！🐋"  # Test string

    # Encode
    indices = tokenizer.encode(string)
    print("Encoded IDs:", indices)

    # Decode
    reconstructed_string = tokenizer.decode(indices)
    print("Decoded:", reconstructed_string)

    # Verify reversibility
    assert string == reconstructed_string, "Character encoding and decoding inconsistent!"

    # Calculate vocabulary size (max Unicode code point + 1)
    vocabulary_size = max(indices) + 1
    print("Vocabulary size (upper bound):", vocabulary_size)

    # Simple compression ratio calculation
    def get_compression_ratio(text, indices):
        # Compression ratio = original string byte count / encoded index byte count
        import sys
        original_bytes = len(text.encode('utf-8'))
        encoded_bytes = len(indices) * 4  # Assume each Unicode code point stored in 4 bytes
        return original_bytes / encoded_bytes

    compression_ratio = get_compression_ratio(string, indices)
    print("Compression ratio:", compression_ratio)
```

### Appendix 4: BPE, Character-level, and Byte-level Tokenizer Comparison

```python
# Byte-level Tokenizer
from collections import Counter
class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, indices):
        return bytes(indices).decode("utf-8")

# Character-level Tokenizer
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

# Calculate compression ratio (bytes/token)
def get_compression_ratio(text: str, token_len: int):
    input_byte_len = len(text.encode("utf-8"))
    return input_byte_len / token_len if token_len > 0 else 1


# Simple BPE Tokenizer
class BPETokenizer:
    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.merges = {}  # {(a,b): new_token_id}
        self.vocab_size = 256  # Start from bytes

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

[BPE, Character-level, Byte-level Tokenizer Comparison](https://github.com/datawhalechina/diy-llm/blob/main/docs/en/chapter2/BPE_character_byte_level_word_segmentation_Comparison.py)

### Appendix 5: Word-level Tokenizer

```python
import regex

# Classic regex used in DeepSeek tokenizer (simplified version)
TOKENIZER_REGEX =  r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"

# Compression ratio calculation
def get_compression_ratio(text: str, segments):
    byte_len = len(text.encode("utf-8"))
    token_count = len(segments)
    return byte_len / token_count if token_count > 0 else 1


# Word-level Tokenizer implementation
class WordTokenizer:
    def __init__(self, pattern=r"\w+|."):
        """
        pattern: regex (default basic version: groups consecutive alphanumeric characters into one word)
        """
        self.pattern = pattern
        self.word2id = {}
        self.id2word = {}

    def build_vocab(self, texts):
        """
        Build vocabulary from a list of training texts
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
        Text → string segments → token ID list
        Out-of-vocabulary words: UNK = -1
        """
        segments = regex.findall(self.pattern, text)
        return [self.word2id.get(seg, -1) for seg in segments], segments

    def decode(self, ids):
        """
        Token IDs → original segments → concatenated string
        """
        return "".join(self.id2word.get(i, "<UNK>") for i in ids)

# Test
if __name__ == "__main__":

    string = "It's so supercalifragilisticexpialidocious!👋👋"
    print("Original string:", string)

    # Use basic regex tokenization (split by whitespace and punctuation)
    basic_segments = regex.findall(r"\w+|.", string)
    print("Basic regex tokenization result:")
    print(basic_segments)

    # Use DeepSeek-style regex
    segments = regex.findall(TOKENIZER_REGEX, string)
    print(f"DeepSeek-style tokenization result: {segments}")

    # Build vocabulary
    tokenizer = WordTokenizer(pattern=TOKENIZER_REGEX)
    tokenizer.build_vocab([string])

    print("Vocabulary size:", len(tokenizer.word2id))

    # Encode
    ids, segs = tokenizer.encode(string)
    print(f"Encoded token IDs: {ids}")

    # Byte sequence
    byte_tokens = [b for b in string.encode("utf-8")]
    print(f"UTF-8 byte sequence: {byte_tokens}")

    print(f"Encoded segments: {segs}")

    # Decode
    decoded = tokenizer.decode(ids)
    print("Decoded result:", decoded)

    # Compression ratio
    ratio = get_compression_ratio(string, segs)
    print("Compression ratio:", ratio)
```

### Appendix 6: Simple BPE Training Demo

```python
import regex
from collections import Counter

# DeepSeek-style regex
DEEPSEEK_REGEX = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"

# Use grapheme clusters to keep emoji from being split
def split_graphemes(token):
    return tuple(regex.findall(r'\X', token))

# BPE training function
def train_bpe(texts, num_merges=50):
    """
    texts: list of texts (used for BPE training)
    num_merges: number of BPE iterative merges
    """
    # 1. Build initial vocabulary (character-level + </w> end-of-word marker)
    vocab = Counter()
    for text in texts:
        tokens = regex.findall(DEEPSEEK_REGEX, text)
        for token in tokens:
            chars = split_graphemes(token) + ('</w>',)
            vocab[chars] += 1
    merges = []
    for _ in range(num_merges):
        # Count adjacent pair frequencies
        pairs = Counter()
        for word, freq in vocab.items():
            for i in range(len(word)-1):
                pairs[(word[i], word[i+1])] += freq
        if not pairs:
            break

        # Find the most common pair
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)

        # Merge this pair across the entire vocabulary
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

# BPE Tokenizer class
class BPETokenizer:
    def __init__(self, merges):
        self.merges = merges

    def encode_word(self, token):
        # Initial split into characters + </w>
        word = list(split_graphemes(token)) + ['</w>']
        # Apply merges in order
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
        # Concatenate tokens and remove trailing </w>
        text = ''.join(tokens).replace('</w>', '')
        return text

# Test
if __name__ == "__main__":
    train_texts = ["这只猫🐈很可爱", "the quick brown fox jumps over the lazy 🐕‍🦺"]
    merges, vocab = train_bpe(train_texts, num_merges=20)
    print("BPE merges:", merges)
    tokenizer = BPETokenizer(merges)
    test_text = "敏捷的棕色狐狸🦊"
    encoded = tokenizer.encode(test_text)
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)
```

### Appendix 7: DeepSeek-style Tokenizer Simplified Implementation

```python
"""
DeepSeek-V3 Tokenizer Simplified Implementation Example
(Core components: byte-level BPE + DeepSeek-style regex pre-tokenization)
"""
import regex as re
from collections import Counter
from typing import List, Tuple, Dict, Iterable
import json
import base64


# Configuration: DeepSeek regex pattern (pre-tokenization)
# \p{L}+   consecutive letters (Chinese, English, all Unicode letters)
# \p{N}+   consecutive digits
# [^\p{L}\p{N}\s]+  non-letter-digit-whitespace characters (e.g., punctuation, emoji)
# \s+      consecutive whitespace
DEEPSEEK_REGEX = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"


# Basic functions: pre-tokenization and byte processing
def pretokenize(text:str):
    """Pre-tokenize using DeepSeek-style regex"""
    return re.findall(DEEPSEEK_REGEX, text)

def bytes2tokens(b:bytes):
    """
    Convert UTF-8 byte sequence to latin1-representable token list.
    Each byte 0-255 can be mapped to a character by latin1.
    """
    return [bytes([x]).decode('latin1') for x in b]

def tokens2bytes(tokens):
    """Convert latin1 token list back to raw bytes"""
    return b''.join([t.encode('latin1') for t in tokens])


# BPE training functions
def build_corpus(texts):
    """
    Build byte-level corpus.
    Steps: pre-tokenize → UTF-8 encode → decompose to single bytes → initial token sequence.
    """
    corpus = []
    for text in texts:
        for chunk in pretokenize(text):
            corpus.append(bytes2tokens(chunk.encode('utf-8')))
    return corpus

def pair_freq(corpus: List[List[str]]):
    """Count the frequency of adjacent token pairs across all token sequences"""
    pairs = Counter()
    for word in corpus:
        for i in range(len(word)-1):
            pairs[(word[i], word[i+1])] += 1
    return pairs

def merge_pair(word: List[str], pair: Tuple[str,str]):
    """Merge a specified token pair into a single token"""
    a, b = pair
    merged = []
    i = 0
    while i < len(word):
        if i < len(word)-1 and word[i]==a and word[i+1]==b:
            merged.append(a+b)   # Merge into one new token
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return merged

def train_bpe(texts: Iterable[str], vocab_size: int=5000, num_merges: int=None) -> Tuple[List[Tuple[str,str]], List[str]]:
    """
    Train byte-level BPE
    """
    corpus = build_corpus(texts)
    base_tokens = [bytes([i]).decode('latin1') for i in range(256)]
    merges: List[Tuple[str,str]] = []
    merged_set = set()
    cur_vocab_size = 256

    # If merge count not specified, derive from target vocab size
    merge_steps = num_merges or (vocab_size - 256)

    for _ in range(merge_steps):
        pfreq = pair_freq(corpus)
        if not pfreq:
            break

        # Find the highest-frequency pair
        best_pair, _ = pfreq.most_common(1)[0]

        if cur_vocab_size + 1 > vocab_size:
            break

        merges.append(best_pair)

        # Apply merge replacement across the entire corpus
        corpus = [merge_pair(word, best_pair) for word in corpus]

        # Record the new token in the vocabulary
        merged_set.add(best_pair[0]+best_pair[1])
        cur_vocab_size += 1

    # Append special tokens
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # vocab = special tokens + 256 byte tokens + BPE-merged new tokens
    vocab_tokens = special_tokens + base_tokens + sorted(merged_set)

    return merges, vocab_tokens



# Tokenizer class
class DeepSeekV3Tokenizer:
    def __init__(self, merges: List[Tuple[str,str]], vocab_tokens: List[str]):
        self.merges = merges
        self.vocab_tokens = vocab_tokens

        # token ↔ id mapping
        self.token2id = {tok:i for i, tok in enumerate(vocab_tokens)}
        self.id2token = {i:tok for tok,i in self.token2id.items()}

        # merges pair → sorted index
        self.ranks = {pair:i for i,pair in enumerate(merges)}

        # Special tokens
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

    def encode_chunk(self, chunk: str) -> List[str]:
        """
        BPE-encode one pre-tokenized chunk:
        - Convert to byte tokens
        - Apply merges progressively
        - Handle OOV: split unknown token back to bytes or mark as <unk>
        """
        tokens = bytes2tokens(chunk.encode('utf-8'))

        # Apply BPE merge rules
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

        # OOV token fallback: split back to bytes
        out = []
        for t in tokens:
            if t in self.token2id:
                out.append(t)
            else:
                # Split into byte tokens; if a byte token is also not in vocab → <unk>
                out.extend([ch if ch in self.token2id else self.unk_token for ch in t])
        return out

    def encode(self, text: str, add_bos=False, add_eos=False, print_chunks=False):
        """
        Encode full text:
        - Pre-tokenize first
        - Then encode chunk by chunk
        - Optionally print intermediate process
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
                        # Attempt to recover UTF-8
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
        Restore ID sequence to UTF-8 text
        """
        byte_seq = bytearray()
        for i in ids:
            tok = self.id2token.get(i, self.unk_token)
            if tok in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            byte_seq.extend(tokens2bytes(list(tok)))
        return byte_seq.decode('utf-8', errors='replace')

    def save(self, vocab_path: str, merges_path: str):
        # Save vocab (token2id)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

        # Save merges: each token as base64
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
        # Load vocab
        with open(vocab_path, 'r', encoding='utf-8') as f:
            token2id = json.load(f)
        vocab_tokens = [None] * (max(token2id.values()) + 1)
        for tok, idx in token2id.items():
            vocab_tokens[idx] = tok

        # Load merges (base64 → bytes → latin1)
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_b64 = json.load(f)

        merges = []
        for a_b64, b_b64 in merges_b64:
            a = base64.b64decode(a_b64).decode('latin1')
            b = base64.b64decode(b_b64).decode('latin1')
            merges.append((a, b))
        return cls(merges, vocab_tokens)


# Training function
def train_tokenizer(texts, vocab_size=5000, num_merges=None):
    merges, vocab_tokens = train_bpe(texts, vocab_size=vocab_size, num_merges=num_merges)
    return DeepSeekV3Tokenizer(merges, vocab_tokens)

# Example
if __name__ == "__main__":
    texts = [
        "Transformer是AI的核心技术。",
        "DeepSeek分词器支持中文、英文、emoji等多语言。",
        "Hello, 世界! 🌍🚀",
    ]

    print("Training Tokenizer (vocab_size=1024)")
    tokenizer = train_tokenizer(texts, vocab_size=1024)
    print(f"Training complete, vocabulary size: {len(tokenizer.vocab_tokens)}")
    print("-"*50)

    txt = "注意力机制是AI的核心技术。 🚀 🚀"
    print(f"Encoding text: {txt}")
    ids = tokenizer.encode(txt, add_bos=True, add_eos=True, print_chunks=True)

    print("-"*50)
    print("Token IDs:", ids)
    decoded = tokenizer.decode(ids)
    print("Decoded result:", decoded)
    print("Reversible:", decoded == txt)
```

*Finally, sincere thanks to every contributor who provided valuable feedback~ Your input makes this tutorial better! 🤝🤝*
