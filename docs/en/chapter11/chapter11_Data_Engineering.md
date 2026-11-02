# Chapter 11: Data Engineering

<div align="center">
<img width="980" height="470" alt="Data Engineering and LLM Training" src="https://github.com/user-attachments/assets/b1a0f623-9aad-497d-a804-d1b5212fa5c6" />
   <p>Figure 11.1 Data Engineering and LLM Training</p>
</div>

In previous chapters, we discussed how to train stronger models through architecture design, optimization methods, tokenization, and scaling — assuming the training data was already given. From this chapter onward, we turn to a more fundamental question: what data should language models actually be trained on? Real-world LLM development shows that **data is often more critical than model structure itself** — mainstream foundation models almost always openly publish their complete architecture and training processes, yet remain highly vague about the specific composition of training data. This precisely indicates that data is the hardest to replicate and the most competitively valuable part. Even after self-supervised learning became mainstream, data engineering贯穿 the entire training process — how data is collected, cleaned, filtered, and combined directly determines what the model can and cannot learn. Due to data's明显的 long-tail characteristics, the model's capability boundaries in the real world are ultimately defined by the coverage scope of training data.

> The long-tail nature of data means common samples appear very frequently in training data, while specialized domain or rare scenario data appears few times per individual category. However, since the number of such rare sample types is extremely large, they collectively determine the coverage scope and generalization boundaries of LLM capabilities.

## 11.1 Data Acquisition

Whether **Llama 3** or **DeepSeek**, they not only open-source weights but even disclose architectural details — **yet they remain completely silent on data**. Beyond trade secrets and legal risks, this is because **data cleaning and formulation** are the true core of modern LLMs.

To understand data's role in LLMs, we need to grasp the overall lifecycle of large model training. Unlike the early paradigm of "end-to-end one-shot training," the construction process of modern LLMs exhibits明显的 phased characteristics. Generally, the training pipeline can be divided into three interconnected stages with distinct objectives: `Pre-training`, `Mid-training`, and `Post-training`. Different stages have significantly different requirements for data type, scale, and quality, collectively determining the model's general capability, domain adaptability, and最终 usability.

- **Pre-training**: Data mainly from large-scale raw corpora — web crawl data (Common Crawl), books, and Wikipedia — typically at the **trillion-token scale (~3T–15T)**. The core goal of this stage is for the model to systematically learn natural language statistical规律, grammatical structures, and broad world knowledge, establishing foundational general language modeling capability. This tutorial will focus primarily on this stage.

- **Mid-training**: Data from rigorously filtered high-quality text, especially emphasizing **STEM data** (math, code) and **long-context documents**, typically at the **10B–100B token scale**. This stage主要用于定向强化 the model's capabilities in reasoning, math, code generation, and long-text understanding while maintaining general ability, serving as a bridge connecting pre-training and subsequent alignment training.

- **Post-training**: Data primarily **human-constructed or annotated**, including instruction data (SFT), multi-turn dialogue data, and human preference feedback data (RLHF). The goal of this stage is not to expand knowledge scale but to guide the model to learn to follow instructions, engage in safe and helpful interactions, and align behaviorally with human values and usage expectations.

### 11.1.1 Training Data

Today's widely adopted data and training standards did not emerge from thin air but gradually evolved through long-term practice and continuous trial-and-error.

**1. BERT**

BERT's pre-training was not simply about "piling up data volume" but had明确的 data structure assumptions. Its training corpus came from BooksCorpus (~800M words) and English Wikipedia (~2.5B words), both sharing the characteristic of **containing大量 long, continuous, naturally-formed document-level text**. In Wikipedia, only正文段落 were used, deliberately excluding structured content like lists, tables, and headers to avoid interfering with the natural contextual flow of language. This choice directly served BERT's core objective — learning cross-sentence and even cross-paragraph semantic dependency relationships.

>Why did BERT's pre-training emphasize using document-level corpora rather than corpora composed of randomly shuffled sentence-level independent samples?
>
>In contrast, corpora like the Billion Word Benchmark that randomly shuffle sentence-level independent samples, although massive in scale, cannot满足 BERT's bidirectional Transformer needs — document-level text is needed to fully leverage contextual information. Short sentences or shuffled sentences would weaken the model's ability to learn cross-sentence dependencies and semantic representations.

**2. GPT-2: "Mining Gold" from Web Pages**

Early language model training mostly relied on **single, high-quality but limited-scale corpora**, such as books and Wikipedia. Although web data has broad coverage, severe noise makes direct use harmful to model performance. To address this, OpenAI proposed a clever **heuristic data filtering method**:

- WebText dataset construction: Not directly crawling the entire web, but curating external links from the **Reddit** community.
- Filtering criterion: Only including pages from links appearing in posts with at least **3 upvotes**.
- Design logic: If at least 3 users considered the link valuable, the webpage was deemed to have a certain level of credibility, thus effectively eliminating大量垃圾 ads and low-quality content.

>This strategy effectively improved data quality while maintaining web data scale, laying early empirical foundations for large-scale language model training.

**3. GPT-3: Scale and Diversity**

As model scale continuously expanded, single data sources could no longer meet training demands. GPT-3 introduced **larger-scale, more complex data strategies**:

- **Introduction of Common Crawl**: Common Crawl crawls web pages across the entire internet, providing massive raw text for language models. Although coverage is broad, it contains大量 noise.

- **GPT-3's Quality Control Strategy**: To extract high-value corpora from庞大的 web crawl data, OpenAI performed systematic **data cleaning and preprocessing** on raw text:
  - **Deduplication**: Remove duplicate content to prevent the model from over-memorizing single texts.
  - **HTML tag removal**: Strip web markup, ad scripts, and other non-text information.
  - **Non-text content cleaning**: Remove garbled text, low-quality text, or non-natural-language data.

>This processing strategy effectively improved corpus quality while **maintaining massive data scale**, reducing noise interference with model training, and laying the foundation for GPT-3's high performance and strong generalization capability.

- **The Pile Open-Source Diverse Corpus**: The EleutherAI community proposed The Pile, further enhancing training data diversity:
  - Data covers 22 high-quality domains, including ArXiv research papers, GitHub code, StackExchange Q&A, technical mailing list data, etc.
  - This approach not only retained the scale advantage of web text but also supplemented academic, professional, and conversational text, thereby improving the model's adaptability to different tasks and domains.

>GPT-3's experience demonstrates that单纯 pursuing data scale is insufficient for training high-performance models; **quality control and diversity coverage** are equally critical — an important启示 for modern LLM data strategy design.

**4. Recent LLM Training Data Sources**

- **OLMo 2 Training Data**

<div align="center">
<img width="800" height="390" alt="Pre-training data sources" src="https://github.com/user-attachments/assets/26b31982-662a-4806-9287-f50afa600f1d" />
   <p>Figure 11.2 Pre-training Data Sources</p>
</div>

**1) Pre-training Stage**: Contains大量 general text. The purpose of this stage is to建立 the model's general language understanding and foundational knowledge capabilities.
  - **Data proportion**: Accounts for **90%–95%** of total training compute.
  - **Data composition**: Approximately **3.9 trillion tokens**, with over 95% from web text.
    - **DCLM-Baseline**: Provides foundational web text, accounting for the majority.
    - **StarCoder**: Provides high-quality code data, excluding low-star projects and non-text files.
    - **Other sources**: Including academic papers, arXiv (STEM papers), OpenWebMath and Algebraic Stack (math and proofs), Wikipedia (encyclopedic knowledge).

**2) Mid-training Stage**: Data proportion is **5%–10%** of total training compute. **Data composition**: injecting specific domain knowledge and reinforcing math capability. The purpose is to improve the model's reasoning, math, and professional capabilities in specific domains.

<div align="center">
<img width="730" height="720" alt="Mid-training data sources" src="https://github.com/user-attachments/assets/87e2d131-71b1-4512-8159-e34ca5e2d401" />
   <p>Figure 11.3 Mid-training Data Sources</p>
</div>

  - **High-quality web**: Top 7% scoring data filtered from DCLM, plus content with high FineWeb metrics.
  - **Curriculum data**: Including FLAN instruction data, Stack Exchange Q&A data, academic papers, and Wikipedia.
  - **Math enhancement**: Approximately 10.7B tokens, including TuluMath (synthetic math problems), TinyGSM-MIND (synthetic math dialogues), MathCoder2 (synthetic books).

**3) Post-training Stage**

<div align="center">
  <img width="970" height="800" alt="Post-training data sources" src="https://github.com/user-attachments/assets/6721e7d6-4b2e-4b85-bca4-181603df027e" />
   <p>Figure 11.4 Post-training Data Sources</p>
</div>

The goal of this stage is to improve OLMo 2's performance in real interaction scenarios, focusing on instruction-following capability, human preference alignment capability, and stability and correctness on high-reliability tasks like mathematical reasoning. It adopted a multi-strategy alignment training pipeline under the **Tülu 3 framework**: First, through SFT (Supervised Fine-Tuning), using大规模 synthetic instruction data (~866K entries) generated via the PersonaHub method, mixed with real conversation data like WildChat, enabling the model to learn规范 responding to various instructions. Next, through DPO (Direct Preference Optimization), sampling candidate responses from 20 different model families, with GPT-4o performing preference evaluation to build the UltraFeedback preference dataset, aligning model outputs with human preferences. Finally, through RLVR (Reinforcement Learning with Verifiable Rewards), performing reinforcement training on tasks with objectively correct answers like math using datasets such as GSM8K and MATH, thereby significantly improving the reliability of model reasoning results.

- **Qwen-3 Training Dataset**

**1) Pre-training Stage**: In this stage, Qwen-3's large-scale pre-training corpus mainly consists of general web text and multilingual content, reaching approximately **36 trillion tokens** — almost double Qwen2.5 — covering **119 languages and dialects**. To build a high-quality and diverse foundational corpus, the team not only collected internet text but also extracted structured text from大量 PDF documents. The extraction process used fine-tuned **Qwen2.5-VL**, a vision-language model, to recognize embedded text in PDFs, then cleaned and quality-improved the text using the Qwen2.5 base model to obtain high-quality training `tokens`. The core goal of this stage is to enable the model to建立 **solid general language capability and world knowledge foundations**.

**2) Mid-training Stage**: The focus of the second stage shifts to **high-quality knowledge-intensive content**, significantly increasing the proportion of STEM, logical reasoning, and programming data. In this stage, beyond introducing curated real corpora, **domain-specific expert models** are used to synthesize training data:
  - Using `Qwen2.5-Math` to generate math problems and解析 corpora;
  - Using `Qwen2.5-Coder` to generate code examples and program corpora;
  - Also potentially generating rich content like textbook-style text, Q&A pairs, etc.
  The second stage additionally supplements approximately 5 trillion high-quality tokens to enhance the model's professional reasoning and problem-solving capabilities.

**3) Post-training Stage**: In Qwen-3's final training stage, the model's focus is on **handling long-text capability**. To enable the model to understand longer documents, conversations, or complex content, training used大量 **long-context corpora** and extended the model's maximum context length from 4,096 tokens to 32,768 tokens. To obtain sufficient training samples, some text was `synthetic data` generated by large models. Qwen-3 also performed **instruction fine-tuning and alignment operations** in this stage:
  - Using synthetic data to train the model how to understand and execute instructions;
  - Teaching the model multi-step reasoning and aligning with human preferences.

**Summary**: Qwen-3's post-training combines real and synthetic data: real data lays the foundation, synthetic data efficiently enhances the model's capability in long-text understanding and instruction execution, making the model more stable and intelligent when handling ultra-long content.

**Synthetic data has become an important means of accelerating model training and enhancing generalization to scarce and long-tail scenarios.** Its role can be analogized to students' practice problems: questions are精心 designed by teachers or expert systems; although not completely equivalent to real exam situations, they can systematically train logical reasoning ability and problem-solving skills in a controlled, safe environment. In recent years, LLM training paradigms have shown a clear division of data functions:

- **Basic pre-training stage** mainly relies on large-scale real-world text to learn language structure, world knowledge, and statistical co-occurrence规律;
- **Instruction alignment and post-training stage** highly depends on synthetic data, generating high-quality instruction-response samples through expert models or rule systems to systematically teach the model how to follow instructions, perform multi-step reasoning, and align with human preferences and value constraints.

**This "real data lays the foundation, synthetic data refines" collaborative data paradigm has become a key component of current LLM training workflows.**

>**Why is it popular to use one large model's generated data to train another model?**
>
>Different LLMs exhibit similar patterns or "ways of thinking" when processing language and reasoning tasks. This doesn't mean models actually think, but rather that they learn `language规律` and `logical relationships` in very similar ways. Using an already well-trained LLM to generate training data is like providing a new model with a set of demonstration answers or problem-solving templates, showing the new model "how it should be done." This approach has two明显的 benefits:
>
>1. **Provides high-quality examples**: Data generated by large models is typically logically clear and linguistically natural, more suitable than randomly crawled or manually assembled data for training models in reasoning and question-answering capabilities.
>2. **Saves human annotation costs**: Without spending大量人力 writing or reviewing data, rich and diverse training samples can be obtained.
>
>In other words, this is using the experience of an "experienced teacher" (large model) to guide a "new student" (new model) in learning, enabling the new model to master complex tasks faster and more stably while reducing cost and time consumption.

### 11.1.2 Specialized Domain Data

General web text, such as Wikipedia, news, social platform exchanges, etc., can only enable models to grasp basic common sense and everyday language patterns. To make models truly "intelligent" — capable of solving complex problems, performing logical reasoning, or mastering professional knowledge — we need to引入 high-quality, specialized data sources covering domains like logical reasoning, scientific knowledge, programming, and mathematics.

**1. Code**

**Sources and Characteristics**: GitHub is currently the largest open-source code platform, containing various programming languages, project types, and application scenarios. However, directly cloning entire repositories is不可取 because大量 non-code files (documentation, images, etc.) add noise, and duplicate content or template code can affect model diversity learning; auto-generated code and low-quality repositories may introduce erroneous patterns.

**Processing Methods**:
- **Deduplication**: Remove duplicate code snippets or similar repositories to ensure data diversity.
- **License filtering**: Parse each repository's License to avoid training on unauthorized code.
- **Quality filtering**: Exclude auto-generated code, empty repositories, or low-quality projects without READMEs.

**Role and Significance**:
- **Code writing capability**: Models can generate, complete, debug, and optimize code.
- **Logical reasoning capability**: Research shows that models trained on code demonstrate significantly enhanced multi-step reasoning, problem decomposition, and abstract thinking abilities.
- **Examples**: Python algorithm implementation, SQL query optimization, mathematical formula computation, etc., can all learn patterns and structures from code data.

**2. Books**

**Significance**:
- Books typically provide longer context than web pages, with coherent narrative and complete structure.
- Helps models learn:
  - **Long-text understanding**: Tracking plot, reasoning about character motivations or logical relationships.
  - **Story logic**: Understanding event sequence, causal relationships, and论证 chains.

**Copyright**:
- Public domain books: Classic books provided by Project Gutenberg, with clear copyright and safe usability;
- Non-public-domain books: Datasets like Books3, potentially sourced from shadow libraries, carry copyright risks.

**Usage Tips**:
- **Prioritize public domain or authorized books**, ensuring legal compliance and high-quality text.
- Books can be **chapter-split and paragraph-annotated** to facilitate model learning of contextual relationships.
- Examples: Novels, popular science books, professional textbooks, especially suitable for training long-context understanding and narrative generation capabilities.

**3. Math and Science**

**ArXiv Papers**: Provide high-density scientific text经过 LaTeX conversion, including formulas, charts, and structured reasoning content, suitable for training model:
  - **Scientific understanding capability**: Learning terminology, concepts, and reasoning methods.
  - **Professional Q&A capability**: Solving STEM problems.

**StackExchange Q&A**:
- Q&A format is naturally suitable for **instruction-following training**.
- Each question typically comes with best answers, comments, and multi-step reasoning processes, helping models:
  - Learn problem decomposition and reasoning workflows;
  - Improve the ability to generate accurate, clear answers.

**Usage Notes**: For scientific data, perform **formula parsing, text cleaning, question-answer alignment** to improve training effectiveness. Can combine ArXiv with StackExchange to enable models to grasp both **theoretical knowledge** and **practical problem-solving capability**.

**Summary**:

| Data Type | Source | Core Value | Notes |
| ----- | ------------------- | --------------------- | --------------------- |
| Code | GitHub | Improves logical reasoning, multi-step problem handling, code generation | Deduplication, license parsing, low-quality repo removal |
| Books | Gutenberg, public domain | Long-text understanding, story logic, coherent narrative | Avoid copyright risk; prioritize authorized/public domain |
| Math & Science | ArXiv, StackExchange | Domain expertise, scientific reasoning, instruction following | Parse formulas, clean text, align Q&A |

> One-sentence summary: For an LLM that initially knows nothing — like a child who doesn't understand the world very well — general text teaches it how to "see the world," recognizing various things and everyday common sense; while specialized domain data teaches it how to "understand the world," analyzing problems and推理判断. The combination of both gradually enables the model to learn independent thinking and make smarter decisions面对 complex tasks.

### 11.1.3 Data Security Issues

In LLM data engineering, **security issues are unavoidable minefields**. Not handling these properly can lead to legal risks, model bias, and even exploitation by attackers.

**1. Copyright Predicament**

**Current situation**: Almost all internet content, even without explicit copyright declaration, is protected by default — including blogs, news, books, code, etc. Using such content for training large models may涉及 copyright issues.

**Fair use**: AI companies typically offer some reasonable explanations: models don't simply copy content but learn **statistical规律 and language patterns** from大量 text; what's output is generated text, not direct reproduction of training data原文. For example, OpenAI and other large model companies argue in court that training constitutes part of fair use.

**Risks**:
- Major news organizations like the New York Times are filing copyright lawsuits against OpenAI. If they lose, AI training may require大量 purchasing copyright licenses, significantly increasing costs.
-启示 for developers: When using copyrighted data for training or fine-tuning, pay special attention to authorization, prioritizing public domain, open-source, or proprietary data.

**2. Data Poisoning**

**Concept**: Data poisoning refers to attackers injecting specific "malicious trigger patterns" or false information into public data sources. When such data is crawled and used for training, models may learn incorrect behaviors or generate unsafe outputs.

**Examples**:
- On Wikipedia or forum posts, attackers may insert malicious text or false information.
- Even with data rollback mechanisms, some malicious content has already been captured by crawlers like CommonCrawl and entered training sets.

**Impact**: Models may generate biased or erroneous responses, which can lead to severe consequences in high-risk domains (e.g., healthcare, finance, law).

**Mitigation strategies**:
- Data cleaning and filtering: Remove明显 abnormal or malicious content;
- Data validation: Conduct manual or semi-automated review of critical domain data;
- Continuous monitoring: Perform safety evaluation of model outputs after training.

>In summary, security issues in LLM training mainly include **copyright risks, data poisoning, and crawler protocol compliance**.

### 11.1.4 Internet Data Cleaning

Common data cleaning methods for internet data include:

- **Heuristic (Rule-based)**: Filter web text through manually designed simple rules (e.g., C4 dataset cleaning strategy), primarily based on text surface features for filtering. This method is simple to implement, computationally efficient, and capable of rapidly removing明显 noise in large-scale data processing, suitable for early preprocessing stages of data cleaning. However, due to limited rule coverage, heuristic methods容易误删 code, poetry, and other atypical text, thus limiting their effectiveness in high-quality corpus filtering.

**Heuristic Cleaning Implementation Based on Code**

- **Synthetic data complements real data**: Using expert models to generate training data is now standard practice

```python
import re
from bs4 import BeautifulSoup

def clean_web_text_strict(html_text):
    """
    Heuristic web text cleaning (referencing partial C4 principles):
    """
    # Parse HTML content using BeautifulSoup
    soup = BeautifulSoup(html_text, 'html.parser')

    # Remove non-body HTML tags
    # table: tables, pre/code: code blocks, ul/ol/li: lists
    # blockquote: quoted footnotes or superscripts/subscripts
    for tag in ['table', 'pre', 'code', 'ul', 'ol', 'li', 'blockquote', 'sup', 'sub']:
        for element in soup.find_all(tag):
            element.decompose()  # Completely delete this element and its children

    # Get plain text, separator='\n' ensures line breaks between block elements to avoid text粘连
    text = soup.get_text(separator='\n')

    # Split by line, strip whitespace from each line, filter out empty lines
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    filtered_paragraphs = []
    for para in paragraphs:
        # Filter Rule A: Delete paragraphs not ending with punctuation
        # This regex matches Chinese 。！？ and English .!?
        # If the paragraph doesn't end with these symbols, it's typically considered incomplete sentence or navigation/title
        if not re.search(r'[。！？\.!?]$', para):
            continue

        # Filter Rule B: Delete paragraphs with fewer than 3 sentences
        # Estimate sentence count by counting terminating punctuation occurrences in the paragraph
        sentence_count = len(re.findall(r'[。！？\.!?]', para))
        if sentence_count < 3:
            continue

        # After层层筛选, retain high-quality paragraphs (satisfying both A and B rules)
        filtered_paragraphs.append(para)

    # Post-processing: merge paragraphs, join all retained paragraphs with single newline
    cleaned_text = '\n'.join(filtered_paragraphs)

    # Regex replacement: replace 2+ consecutive newlines with single newline for clean output format
    cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)

    return cleaned_text


# --- Test Area ---
html_example = """
<html>
<body>
    <h1>Webpage Title</h1> 
    <p>This is the first paragraph, content complete. Second sentence. Third sentence.</p> 
    <p>Short paragraph. Only two sentences, not retained.</p> 
    <pre>Code block content, not retained</pre> 
    <table><tr><td>Table content</td></tr></table>
    <ul><li>List content, not retained</li></ul>
    <blockquote>Quoted content, not retained</blockquote> 
    <p>Another natural language paragraph. Second sentence. Third sentence.</p> 
    <p>Third paragraph, retained. Second sentence. Third sentence.</p>
</body>
</html>
"""

# Execute cleaning and print results
cleaned_text = clean_web_text_strict(html_example)
print("--- Cleaned Text ---")
print(cleaned_text)
```

- **Model Perplexity-Based Text Quality Cleaning**

A commonly used text quality filtering method is using **n-gram models or pretrained language models** to compute text **Perplexity (PPL)**. The core idea is:

- **Low-perplexity text** typically has correct grammar and reasonable semantics, with quality接近百科 level, helping reduce training noise.
- **High-perplexity text** may contain garbled text, grammatical errors, or incoherent content.

**Advantages**: Improves corpus quality, reduces noise in model training; retains规范的 written language, suitable for scenarios with high text quality requirements.

**Disadvantages**: May lose long-tail, colloquial, or innovative expressions; reduces data diversity.

**Practice**: CCNet uses language model perplexity for automatic text quality assessment, leveraging the characteristic that low-perplexity text better conforms to natural language distribution, achieving large-scale multilingual text cleaning without manual rules.
- [CCNet research](https://arxiv.org/pdf/1911.00359) found that perplexity distributions differ significantly across languages:
  - Some languages have very high perplexity distribution peaks, while others have dispersed distributions.
  - This difference is primarily related to the amount of Wikipedia corpus used when training the language model, not insufficient high-quality content.

Therefore, for multilingual corpora, **different perplexity thresholds need to be set for each language**. Threshold selection can adopt a quantile strategy, e.g., dividing the corpus into three equal parts by perplexity and retaining only the middle portion, balancing text quality and coverage.

<div align="center">
<img width="1100" height="500" alt="CCNet Working Principle" src="https://github.com/user-attachments/assets/06d39c85-2911-48a7-8107-9e07fcde5fc7" />
   <p>Figure 11.6 CCNet Working Principle</p>
</div>

**Simple CCNet Implementation**
```python
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List

class AutoPerplexityFilter:
    def __init__(self, model_name='distilgpt2'):
        """
        Initialization: distilgpt2 is a distilled version of GPT-2, smaller and faster.
        """
        print(f"Loading language model: {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Language model: computes text probability distribution
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        # Enable explicit Loss computation mode
        self.model.config.loss_type = "ForCausalLMLoss"
        # Set to evaluation mode
        self.model.eval()

        # For storing calibration thresholds for different languages (dictionary structure)
        self.thresholds = {}

    def calculate_score(self, text: str) -> float:
        """
        Core mathematical computation: automatically calculates Perplexity (PPL) for a text segment.
        Formula: PPL = exp(Cross-Entropy-Loss)
        """
        # Encode text and convert to PyTorch tensor
        inputs = self.tokenizer(text, return_tensors="pt")

        # If text is too short (token count ≤ 1), model cannot compute prediction probability, return max perplexity
        if inputs['input_ids'].size(1) <= 1:
            return 999.9

        # Disable gradient computation to save memory and speed up
        with torch.no_grad():
            # labels=inputs["input_ids"] tells the model to predict next word based on current word
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            # loss is cross-entropy loss
            loss = outputs.loss
            # Perplexity is the exponential form of Loss, reflecting the model's "degree of confusion" about this text
            ppl = torch.exp(loss).item()
        return ppl

    def calibrate(self, lang: str, sample_texts: List[str]):
        """
        CCNet core calibration: set dynamic thresholds for this language.
        Even if the model is naturally unfamiliar with a language (causing generally high PPL),
        through quantile methods, we can still pick out the "relatively better" parts in that language.
        """
        print(f"Calibrating for [{lang}] language...")
        # Compute PPL for each text in this language's sample set
        scores = [self.calculate_score(t) for t in sample_texts]

        # Sort samples by PPL ascending, extract values at 33% and 66% positions
        # t1 (33.33%): quality boundary, values below this are the most natural-language-like parts in this language
        t1 = np.percentile(scores, 33.33)
        # t2 (66.66%): noise boundary, values above this are typically considered format混乱 or garbled
        t2 = np.percentile(scores, 66.66)

        self.thresholds[lang] = (t1, t2)
        print(f"[{lang}] Calibration complete -> Quality boundary: {t1:.2f}, Noise boundary: {t2:.2f}")

    def filter_text(self, lang: str, text: str) -> str:
        """
        Execute classification: compare computed PPL against calibration thresholds.
        """
        score = self.calculate_score(text)

        # Fault tolerance: if this language hasn't been calibrated, cannot classify
        if lang not in self.thresholds:
            return f"PPL={score:.1f} (Threshold standard not yet建立的 for this language)"

        t1, t2 = self.thresholds[lang]

        # Classification logic
        if score <= t1:
            return f"PPL={score:.1f} -> [High Quality] (Premium corpus conforming to model distribution)"
        elif score <= t2:
            return f"PPL={score:.1f} -> [Medium] (Average natural language)"
        else:
            return f"PPL={score:.1f} -> [Noise] (Garbled text, ads, or atypical text)"


# Simulate CCNet execution pipeline
# Provide "golden reference data" (typically sampled from Wikipedia), used to tell the model: what kind of text is "normal" in this language.
zh_reference = [
    "人工智能是计算机科学的一个分支，旨在模拟人类智能。",
    "今天北京的天气非常晴朗，适合户外运动。",
    "深度学习模型需要大量的高质量标注数据进行训练。",
    "故宫是中国古代宫廷建筑的精华，每年吸引大量游客。",
    "Python 是一种广泛应用于数据分析和机器学习的编程语言。"
]

en_reference = [
    "Machine learning is the study of computer algorithms that improve automatically.",
    "The capital of France is Paris, known for its iconic Eiffel Tower.",
    "Quantum computing is a type of computation that harnesses collective properties.",
    "Healthy eating and regular exercise are key to a long life.",
    "Open-source software allows anyone to inspect, modify, and enhance the code."
]

# Initialize automatic filter (this will download/load the model, may take a few minutes depending on network)
cleaner = AutoPerplexityFilter()

# Calibrate thresholds — the essence of CCNet: "adapt to local conditions"
cleaner.calibrate("zh", zh_reference)
cleaner.calibrate("en", en_reference)

# Test actual crawled web data
print("\n" + "=" * 60)
print(f"{'Lang':<4} | {'Text Fragment':<25} | {'Detection Result'}")
print("-" * 60)

test_data = [
    ("zh", "机器学习是研究计算机如何模拟人类学习行为的科学。"),
    ("zh", "123 !! #￥%…… garbled test 456"),
    ("en", "Machine learning is the cornerstone of artificial intelligence."),
    ("en", "asdfghjkl qwert yuiop zxcvbnm"),
]

for lang, text in test_data:
    result = cleaner.filter_text(lang, text)
    # Truncate to first 20 characters for display
    short_text = text[:20] + "..." if len(text) > 20 else text
    print(f"{lang:<6} | {short_text:<28} | {result}")
```

The principle of the simple CCNet web text cleaning code above: high-quality natural language text typically conforms to grammatical and semantic规律, making language model predictions relatively easy, hence lower perplexity; garbled text, ad text, or non-natural-language content often deviates from natural language distribution, making model prediction more difficult, corresponding to higher perplexity.

>The reason for selecting a lightweight GPT-2 model for language distribution computation: the lightweight GPT-2 model facilitates fast computation of text perplexity, with its principle being measuring corpus quality based on the language model's prediction difficulty for text sequences. Although GPT-2's absolute perplexity for Chinese is not完全 precise, within the simple CCNet intra-language calibration framework, perplexity can still effectively distinguish natural language text from obvious noise text, so this model can be used for method principle demonstration and concept validation.

## 11.2 Data Intelligent Filtering

Research various model-based data filtering algorithms — i.e., intelligently filtering data by training classifiers or other predictive models, demonstrating the broad application of these foundational methods across different filtering tasks, and exploring several efficient strategies.

<div align="center">
<img width="1010" height="540" alt="Raw vs Processed Data Relationship" src="https://github.com/user-attachments/assets/7659edb9-d93b-4162-863d-e0b5cfa86907" />
   <p>Figure 11.7 Relationship Between Raw Data and Processed Data</p>
</div>

Given certain target data $T$ and大量 raw data $R$, find a subset $T'$ from $R$ that is similar to $T$.

### 11.2.1 Data Filtering

When raw data volume is large (e.g., Common Crawl web data), and we want both high-quality information and processing speed, using large models directly is not cost-effective. Below we introduce 3 efficient data processing methods:

**1. KenLM**

Kneser-Ney smoothing is a commonly used `n-gram` smoothing method that effectively improves language model probability estimation accuracy for low-frequency or unseen n-grams. Its core idea is leveraging low-order n-gram distribution information to adjust high-order n-grams through interpolation and probability redistribution, thereby mitigating the zero-probability problem and improving long-tail n-gram estimation.

In n-gram model training, maximum likelihood estimation is typically used first to统计 corpus data, computing the occurrence frequency of each `token` in each n-gram, and estimating the conditional probability of the next token given context. Based on these conditional probabilities, sentence perplexity can be computed — lower perplexity indicates more reliable model predictions for that sentence. Therefore, during corpus cleaning or filtering, sentences with lower perplexity can be prioritized for retention, thereby improving overall training data quality. In terms of open-source tools, KenLM is a classic implementation for building and querying large-scale n-gram models. It supports efficient model training, querying, and perplexity computation, usable for corpus quality assessment and filtering.

>n-gram model's disadvantage: certain n-token combinations may appear极少 or never in the corpus, leading to unreliable probability estimates; additionally, as n increases, the number of n-grams the model needs to store and compute grows exponentially, facing the curse of dimensionality.

**2. FastText**

FastText is a text linear classifier that significantly reduces model parameters and accelerates computation through text embedding and dimensionality reduction, while enhancing text representation via `n-gram bag-of-words`. To avoid storage and computation overhead from excessive n-gram counts, it uses `hash mapping` for efficient processing.

>FastText processing flow: text → n-gram → hash bucket (index mapped to embedding) → embedding → average → classification.

**n-gram Bag-of-Words and Hash Mapping Explanation**

**n-gram splits text into consecutive groups of n words.**

Example text: "I like AI"

- **1-gram**: ["I", "like", "AI"]
- **2-gram**: ["I like", "like AI"]
- **3-gram**: ["I like AI"]

**n-gram bag-of-words** treats these n-grams as feature vectors, counting their occurrence frequency in the text:

| n-gram    | Occurrence Count |
| --------- | ---- |
| "I"       | 1    |
| "like"    | 1    |
| "AI"      | 1    |
| "I like"  | 1    |
| "like AI" | 1    |
|"I like AI"| 1    |

**Each dimension corresponds to one n-gram.**

When text is large, n-gram count can explode because storing each n-gram wastes大量 memory. The **hash mapping**思路:

- Don't store complete n-gram vocabulary; instead, use a hash function to map n-grams to a fixed number of bins.
- Different n-grams may map to the same bin (hash collisions are acceptable — the LLM can still learn patterns).

Example: suppose we prepare only **8 bins (0~7)**, using simple hash mapping:

```python
n_grams = ["I like", "like AI", "I", "like", "AI"]
num_bins = 8 
hashed = [hash(g) % num_bins for g in n_grams]
print(hashed)  # Might output: [3, 1, 4, 2, 7]
```

>Even if different n-grams (single tokens or consecutive token feature vectors) map to the same bucket, it won't affect overall model learning.

**FastText Key Functional Code**
```python
# Bag-of-words n-gram generation function
def get_ngrams(tokens, n):
    """
    Generate n-gram word groups — this is the key to FastText capturing word order.
    """
    ngrams = []
    for i in range(len(tokens)):
        for j in range(1, n + 1): # Loop to generate 1-gram through n-gram
            if i + j <= len(tokens):
                # Concatenate word group as string, serving as feature
                ngrams.append(" ".join(tokens[i:i + j]))
    return ngrams

def hash_ngrams(tokens, num_buckets, ngram):
    """
    Hash mapping
    """
    ngrams = get_ngrams(tokens, ngram)
    # Hash each generated feature and take modulo, obtaining corresponding Embedding index
    return torch.tensor([hash(g) % num_buckets for g in ngrams], dtype=torch.long)

class TextDataset(Dataset):
    """
    Data encapsulation: convert raw text into hash index sequences.
    """
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Preprocessing: uniformly lowercase and split by whitespace
        tokens = self.texts[idx].lower().split()
        # Map words and n-grams to hash bucket indices
        hashed_ids = hash_ngrams(tokens, num_buckets, ngram)
        label = self.labels[idx]
        return hashed_ids, label

def collate_fn(batch):
    """
    Collate function: since each sentence contains different numbers of n-grams, need to align lengths for batch training.
    """
    # Find the longest sequence length in current batch
    max_len = max(len(x[0]) for x in batch)
    padded = []
    labels = []

    for hashed_ids, label in batch:
        # Compute padding length needed
        pad_len = max_len - len(hashed_ids)
        # Pad 0 at end of sequence
        padded_ids = F.pad(hashed_ids, (0, pad_len), value=0)
        padded.append(padded_ids)
        labels.append(label)
    # Stack [Batch_Size, Max_Len] shaped tensor
    return torch.stack(padded), torch.tensor(labels)

class FastTextClassifier(nn.Module):
    def __init__(self, num_buckets, embed_dim, num_classes):
        super().__init__()
        # Embedding layer: word vector matrix containing all hash buckets, randomly initialized and learned during training
        self.embedding = nn.Embedding(num_buckets, embed_dim)
        
        # Fully connected layer: directly map averaged embedding vectors to class probabilities (linear classification)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Table lookup: [Batch_Size, Seq_Len] -> [Batch_Size, Seq_Len, Embed_Dim]
        # Convert each hash index into a feature vector
        embedded = self.embedding(x)          
        
        # Average pooling: average all word and n-gram vectors in the sentence to get sentence global representation
        # This approach ignores long-distance word order but is extremely efficient in text classification tasks
        avg_embedded = embedded.mean(dim=1)   # [Batch_Size, Embed_Dim]
        
        # Output layer: compute scores (Logits) for each class
        logits = self.fc(avg_embedded)        
        return logits
```

**Complete runnable [FastText](https://github.com/datawhalechina/diy-llm/blob/main/docs/en/chapter11/FastText.py)**

Output example:
>Input text: I hate this product
>
>Predicted probabilities: Positive=0.0034, Negative=0.9966
>
>Predicted class: Negative

Here, the training text sample scale is small; model parameter initialization, random hash mapping, and sample ordering during training all introduce strong randomness. `FastText`难以形成 stable and effective discriminative rules, leading to inconsistent results across multiple training runs.

**3. DSIR**

<div align="center">
<img width="920" height="500" alt="DSIR Processing" src="https://github.com/user-attachments/assets/0cdd6689-5747-4a81-bffc-5f3923b346ab" />
   <p>Figure 11.8 DSIR Processing</p>
</div>

Using low-cost statistical features to approximate language distributions, achieving distribution alignment of large-scale corpora through importance resampling — an unsupervised data selection method.

- **Target dataset $D_p$**: Small-scale but high-quality dataset (e.g., Wikipedia), used to characterize the target distribution $\tilde{p}(x)$ we希望 the language model ultimately learns.
- **Candidate data pool $D_q$**: Massive-scale, broadly-sourced but quality-uneven data collection (e.g., web crawl text), approximately following candidate distribution $\tilde{q}(x)$.
- **Core objective: Importance Resampling**: For each sample $x \in D_q$ in the candidate pool, estimate its approximate density ratio under target vs. candidate distributions: $w(x) = \frac{\tilde{p}(x)}{\tilde{q}(x)}$, where $w(x)$ measures sample $x$'s "similarity" to the target distribution.
   - $w(x)$ large: Sample relatively common in target distribution but rare in candidate distribution → **more worth retaining**.
   - $w(x)$ small: Sample deviates from target distribution or is common in candidate data → **reduce sampling probability or discard**.

> **DSIR's essence: Use a small, clean dataset to tell us "what kind of text is good text," then pick out these texts from massive raw data according to this standard.**

```python
import numpy as np
from collections import Counter
def dsir_main(n):
    # n: n-gram size
    # Feature construction - Hashed n-grams
    training_text = "the cat in the hat"  # Simulate target dataset D_p
    num_bins = 4  # Hash bucket count (typically 1e4 ~ 1e6 in real scenarios)

    def get_hashed_ngrams(text: str, n: int):
        # Convert text to n-grams and map to fixed hash space
        tokens = text.lower().split()
        # Construct n-grams
        ngrams = [
            " ".join(tokens[i:i+n])
            for i in range(len(tokens) - n + 1)
        ]
        # Hash map to [0, num_bins)
        return [hash(ngram) % num_bins for ngram in ngrams]
    # Target data D_p features
    training_hashed_ngrams = get_hashed_ngrams(training_text, n)
    print(f"Target data hash indices D_p (n={n}):", training_hashed_ngrams)
    # Distribution modeling - estimate p_hat
    counter = Counter(training_hashed_ngrams)
    total = len(training_hashed_ngrams)
    probs = np.array([counter[i] / total for i in range(num_bins)])
    print("Learned target distribution p_hat:", probs)

    # Sample scoring - Candidate data D_q
    test_text = "the cat"
    hashed_ngrams = get_hashed_ngrams(test_text, n)
    print(f"Test text '{test_text}' hash indices:", hashed_ngrams)
    eps = 1e-8
    prob = np.prod([probs[x] + eps for x in hashed_ngrams])
    print(f"Text '{test_text}' estimated probability under target distribution:", prob)
if __name__ == "__main__":
    # Default n=1 (unigram)
    dsir_main(n=1)
    print("\n--- Using 2-gram ---")
    dsir_main(n=2)
```

>Since the target data scale in the code is极小, and hash mapping itself introduces randomness, this example is in the extreme small-sample degenerate case of DSIR (typically适用于 large-scale samples). Multiple runs may produce different results. Therefore, the probability values given in the example do not have actual statistical significance and only serve to illustrate DSIR's computation flow.

### 11.2.2 Data Deduplication

In large-scale language model data engineering, raw corpora typically require systematic deduplication processing. [Google research team's work](https://arxiv.org/pdf/2202.06539)指出 that大规模 training raw data commonly contains大量 duplicate or near-duplicate text, and high-frequency duplicate samples容易 cause models to develop "mechanical memorization," reducing their ability to generalize language规律 and带来 potential privacy risks. Therefore, removing duplicate data helps引导 models from "rote memorization" toward true learning of statistical patterns and structural knowledge. [Further research shows](https://arxiv.org/pdf/2107.06499) that at the same or even lower training computation, training with deduplicated data yields better or at least non-degraded model perplexity, indicating that data deduplication can effectively improve model training efficiency and generalization capability.

**In large-scale data processing**, `hash functions` are often used as an efficient indexing and feature compression method. By mapping high-dimensional or high-cardinality discrete features to fixed-size hash space, storage and computation costs can be significantly reduced, thereby improving overall data processing efficiency. It's worth noting that hash mapping inevitably produces `hash collisions` (multiple different features mapped to the same hash bucket). However, such collisions do not systematically introduce bias but rather mix different feature statistics together in an approximately random manner, thus manifesting as noise rather than deterministic error in the statistical sense. Therefore, in practical applications, trade-offs typically need to be made among hash space size, storage overhead, and statistical precision — reasonably selecting hash functions and bucket counts to strike a balance between computational efficiency and modeling accuracy.

Below we introduce 3 deduplication algorithms:

**1. Exact Deduplication**

Exact deduplication is based on完全一致 matching principle: for each data sample (e.g., a text), compute a deterministic identifier (e.g., the string itself or its hash value), and determine whether samples are完全一致 by comparing identifiers (e.g., "hi" and "hi" would produce the same identifier). For samples with identical identifiers, only one is retained while the rest are removed. This method is simple to implement, computationally efficient, and can effectively eliminate completely duplicate data samples, but cannot identify semantically identical or highly similar duplicate content (e.g., slightly rewritten, format-changed, or locally modified text).

```python
import mmh3
def exact_deduplication():
    # Raw data
    items = ["Hello", "hello", "hello there", "hello", "hi", "bye", "🤔", "🤔"]
    print("Raw data:")
    print(items)

    # Use hash for exact deduplication
    seen_hashes = set()
    deduped_items = []
    for item in items:
        h = mmh3.hash(item)
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped_items.append(item)
    print("\nAfter deduplication:")
    print(deduped_items)
if __name__ == "__main__":
    exact_deduplication()
```

**2. [Bloom Filter](https://en.wikipedia.org/wiki/Bloom_filter)**

<div align="center">
<img width="1300" height="500" alt="Bloom Filter Diagram" src="https://github.com/user-attachments/assets/79682269-46cd-476a-9f0f-6ff2e877847d" />
   <p>Figure 11.9 Bloom Filter Diagram</p>
</div>

`Bloom Filter` maps objects to bit array positions via hash functions and sets bits, used to determine whether an object has appeared before. It doesn't store the object itself, only recording the sample's occurrence痕迹. Using multiple hash functions maps the object to multiple positions; during querying, all positions must be 1 to判定 as "appeared before." In large-scale data processing, this design can significantly reduce the false positive probability caused by hash collisions (i.e., misjudging never-appeared objects as appeared), rather than for eliminating randomness. However, in small samples or when the bit array is very small, increasing hash functions may cause more positions to be prematurely set to 1,反而 increasing misjudgment probability and reducing Bloom Filter's query accuracy.

**Example Analysis: Determining Whether a Word Has Appeared**

Suppose we have a set of words:

```text
items = ["cat", "dog"]
```

And prepare a **length-8 bit array**:

>bit_array = [0, 0, 0, 0, 0, 0, 0, 0]

Using **two simple hash functions**:

- hash1(word) = len(word) % 8
- hash2(word) = (sum(ord(c) for c in word)) % 8

Step 1: Represent word "cat"

- hash1("cat") = 3 % 8 = 3 → set `bit_array[3] = 1`
- hash2("cat") = (99+97+116) % 8 = 312 % 8 = 0 → set `bit_array[0] = 1`

>bit_array = [1, 0, 0, 1, 0, 0, 0, 0]

Step 2: Represent word "dog"

- hash1("dog") = 3 % 8 = 3 → `bit_array[3]` already 1, unchanged
- hash2("dog") = (100+111+103) % 8 = 314 % 8 = 2 → set `bit_array[2] = 1`

>bit_array = [1, 0, 1, 1, 0, 0, 0, 0]

Step 3: Query new word "bird"

- hash1("bird") = 4 % 8 = 4 → query `bit_array[4] = 0`
- hash2("bird") = (98+105+114+100) % 8 = 417 % 8 = 1 → query `bit_array[1] = 0`

>Since **at least one position is 0**, Bloom Filter can确定 "bird" **definitely hasn't appeared** — this is the "one-vote veto" characteristic.

Step 4: Query another new word "god"

- hash1("god") = 3 % 8 = 3 → query `bit_array[3] = 1`
- hash2("god") = (103+111+100) % 8 = 314 % 8 = 2 → query `bit_array[2] = 1`

>Both positions are 1, so Bloom Filter判定 "god" **may have appeared**, but实际上 "god" was not in items = ["cat", "dog"] — this is a **false positive** (misjudgment).

*Runnable code: [Bloom Filter Simplified Implementation](https://github.com/datawhalechina/diy-llm/blob/main/docs/en/chapter11/bloom%20Filter%E7%AE%80%E5%8C%96%E5%AE%9E%E7%8E%B0).*

**3. Locality-Sensitive Hashing (LSH)**

LSH's core objective: **In large-scale text collections, quickly identify similar documents without computing pairwise similarity for all documents.** The entire process can be broken into three core steps:

**Step 1: Feature Extraction — From Text to Set**

To compute similarity, we first need to quantify text as sets via `k-Shingling`:

  - Split text into consecutive segments of length k. For example, text `"今天天气很好"` split into consecutive two-word groups: `{"今天", "天天", "天气", "气很", "很好"}`.
  - **Principle**: If two texts are similar, they share many identical word groups.
  - **Mathematical representation**: Document A → set $S_A$, Document B → set $S_B$. Similarity measured by **Jaccard similarity**:
    $$
    J(A,B) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|}
    $$
    It measures the degree of overlap between two sets, with range [0, 1].

**Step 2: MinHash Dimensionality Reduction — Generate Signatures**

When processed sets are too large for direct comparison to be efficient, "MinHash signatures" can be generated to replace original sets. MinHash signatures convert text element sets obtained through k-shingling into fixed-length numeric vectors via multiple hash functions, where each number corresponds to the set's minimum value under that hash, used to approximate text similarity. **Each shingle element represents a small segment of text content**, so MinHash signatures are essentially the text's "feature vectors."

>Core property: Probability that two sets' MinHash values are equal ≈ their Jaccard similarity

1). **Random permutation**: Randomly sort all possible terms.
2). **Take minimum**: For terms in set $S$, find the element with the smallest ordinal in the random permutation; its ordinal or hash value is the MinHash value.
3). **Generate "signature vector"**: Using $n$ different random permutations (or different hash functions), obtain a signature vector of length $n$:
   $$
   \text{Signature}(S) = [h_1(S), h_2(S), ..., h_n(S)]
   $$

> Advantage: Comparing two signature vectors of length 100 is much faster than comparing sets of 100K terms.

**Step 3: LSH Bucketing for Filtering**

Even with "signature vectors," pairwise comparison across millions of documents remains slow. LSH uses **bucketing strategy** to further accelerate.

**Core idea**: Split the "signature vector" into $b$ **bands**, each band containing $r$ rows.
Rule: **As long as two documents match完全 in any one band, place them in the same "bucket" as candidate similar pairs.**

Suppose the Jaccard similarity of two documents' corresponding sets is $s$. According to MinHash properties, the probability that any single hash function produces equal values for both documents is $s$. Dividing the MinHash signature into $b$ bands, each band containing $r$ rows:

1). Probability of one band完全 matching: $P_\text{band} = s^r$.
2). Probability of one band not matching: $1 - s^r$.
3). Probability of all b bands not matching: $(1 - s^r)^b$.
4). **Probability of at least one band matching (candidate similar pair)**: $P_\text{collision} = 1 - (1 - s^r)^b$.

The band-level matching probability in LSH does not use a new similarity metric but rather the probability of overall similarity $s$. Matching events between different bands are mutually independent. These formulas describe the typical S-shaped collision probability curve in LSH, used to distinguish high-similarity from low-similarity documents.

> This is LSH's **S-curve effect**:
>
> - **Low-similarity documents**: Almost never collide.
> - **High-similarity documents**: Collision probability接近 1.

**Adjusting Thresholds**

<div align="center">
<img width="980" height="570" alt="Band and Similarity Relationship" src="https://github.com/user-attachments/assets/ec47a2fb-dc47-4d9c-9598-6b9452921fd9" />
   <p>Figure 11.10 Band vs. Similarity Relationship</p>
</div>

- **Increase r**: Raises the complete-match threshold for individual bands, so only high-similarity documents can possibly collide; threshold shifts right.
- **Increase b**: Increases the number of "attempts" for collision, so lower-similarity documents may also become candidates; threshold shifts left.

**Three-Step Division of Labor**

| Stage | Problem Solved | Core Cost |
| ------------- | -------------- | -------------------- |
| **Feature Extraction** | Text → Mathematical Set | Large space占用 |
| **MinHash** | Compress sets while preserving similarity | Estimation error |
| **LSH** | Avoid exhaustive comparison, achieve sublinear search | Minor漏检 |

> **Feature extraction builds sets, MinHash compresses sets, LSH rapidly filters candidate similar pairs.**

```python
import mmh3
from typing import List, Set

# Text → k-Shingling
def text_to_set(text: str, k=2):
    """
    k-shingling: convert text to set
    """
    return {text[i:i+k] for i in range(len(text) - k + 1)}

def jaccard(A: Set[str], B: Set[str]):
    return len(A & B) / len(A | B)

# Example texts
text1 = "今天天气真好，我很想出去散步"
text2 = "今天天气很好，我想去散步"
A = text_to_set(text1)
B = text_to_set(text2)
print("True Jaccard similarity:", jaccard(A, B))

# MinHash
def minhash_signature(S: Set[str], n_hash: int):
    """
    Generate MinHash signature for a set
    """
    sig = []
    for seed in range(n_hash):
        sig.append(min(mmh3.hash(x, seed) for x in S))
    return sig

# Parameters
b = 30
r = 2
n_hash = b * r
sigA = minhash_signature(A, n_hash)
sigB = minhash_signature(B, n_hash)

# LSH Band partitioning + exact match
def lsh_candidate(sigA, sigB, b, r) -> bool:
    """
    Determine whether two signatures become LSH candidate pairs
    """
    for i in range(b):
        start = i * r
        end = start + r
        bandA = sigA[start:end]
        bandB = sigB[start:end]

        # Band-level exact match
        if bandA == bandB:
            print(f"Band hit: {i}")
            return True
    return False

is_candidate = lsh_candidate(sigA, sigB, b, r)
print("Became LSH candidate similar pair:", is_candidate)
# "Became LSH candidate similar pair" means whether two documents, after passing through the LSH filter, need to enter the precise similarity computation stage.
```

In large-scale data processing, LSH can rapidly filter similar documents.

## 11.3 Data-Related Research

**Training Data Security**

This [latest research](https://www.pcgamer.com/software/ai/anthropic-reveals-that-as-few-as-250-malicious-documents-are-all-it-takes-to-poison-an-llms-training-data-regardless-of-model-size) conducted by Anthropic (Claude's creator) and the UK AI Safety Institute reveals the vulnerability of LLMs in data security: **the "model poisoning" threshold is far lower than expected**. The study found that regardless of model scale or training data volume, merely **250 malicious documents** are sufficient to implant "backdoor" vulnerabilities in a model. This means malicious actors don't need to control large-scale data; merely implanting specific trigger words (such as gibberish or hidden instructions) can trigger errors in model outputs or establish channels for stealing sensitive data. **This finding emphasizes the extreme importance of严格 auditing data sources and defensive filtering during the LLM training stage.**

**Data Usability**

In the new paradigm of [`LLM`-driven scientific research](https://www.weforum.org/stories/2025/12/data-ai-training-synthetic), data is no longer merely a passive record of the historical world but is evolving into a research resource that can be actively designed, generated, and validated. Through automated physical experiment platforms and computationally simulated systems constrained by physical laws (such as Large Quantitative Models, LQM), researchers can construct synthetic data spaces with causal credibility, full-process traceability, and content not yet existing in prior literature. This architecture centered on "data generation capability,"依托 digital twins and virtual experiments (virtual patients, molecular interaction simulations, and complex system simulations), effectively突破了 structural bottlenecks in traditional R&D regarding cost, cycle time, and historical data bias. Consequently, the重心 of scientific research and industrial competition is shifting from "who owns more existing data" to "who can continuously and reliably generate high-value synthetic data," driving paradigm-level transitions in life sciences, financial system modeling, and intelligent manufacturing.

**Data Evaluation and LLM Memorization Behavior**

<div align="center">
<img width="700" height="700" alt="Data Evaluation of LLM Memorization" src="https://github.com/user-attachments/assets/44b41dd6-c0c0-4adf-9c73-368a6e1bb863" />
   <p>Figure 11.11 Data Evaluation of LLM Memorization Behavior</p>
</div>

In the latest [LLM data evaluation research](https://arxiv.org/abs/2503.12072), addressing the insufficient transparency of LLM training data, Information-Guided Probes propose an efficient "black-box" auditing method that requires no access to model internal weights or output probability distributions. This method is based on `Shannon information theory`:

$$
\mathrm{Surprisal}(w_t)
= -\log P(w_t \mid h_t)
$$

Where:

- $h_t$ represents the contextual information input to the language model, typically represented by the LLM's hidden states;
- $w_t$ represents **key tokens with high information content that are artificially removed** from the context, such as specific person names, place names, or specialized terminology;
- $\mathrm{Surprisal}(w_t)$ represents, **under the Shannon information theory framework**, the amount of information carried by token $w_t$ given context $h_t$. This value expresses the LLM's "degree of surprise" toward $w_t$ — larger values indicate lower model prediction probability for that token, i.e., the token carries higher information content in the current context.

>The information content of a `token` can be understood as the model's prediction difficulty for that token given context. Person names, proper nouns, and domain terminology typically reside in the long-tail region of language distributions — they have large candidate spaces and are difficult to compress through context, thus having low prior prediction probability. Although不易被准确预测 before their appearance, once generated, they often significantly reduce sentence-level semantic uncertainty, thereby bearing the primary information load in language modeling. **In other words, models can typically judge "what type of information should appear here" but难以提前确定 "specifically which one," and high-information-content tokens are precisely the key to helping the model明确 the specific type.**

By identifying and removing tokens with high Surprisal values from input text to construct perturbed contexts, and further observing whether the model, during free generation, can reconstruct these originally low-prior-probability contents at a success rate significantly higher than random or language prior levels, **a statistical evaluation of whether training data memorization traces exist in the model** can be conducted. **Experimental results show** that in tasks such as copyright content identification (novels and news text) and data contamination detection (benchmark leakage), this method demonstrates higher discriminative precision and stronger robustness to filtering and safety mechanisms compared to traditional prefix completion strategies, providing critical technical support for model compliance auditing, author rights protection, and benchmark result authenticity verification.

>It's worth noting that **Information-Guided Probes do not directly prove that models store complete training samples but rather provide statistical evidence for detecting anomalous memorization behavior under "black-box" conditions.**

## Reflection Questions

1) Although we have concepts about quality data, we haven't specifically discussed its actual form — e.g., what characteristics should quality documents possess?

2) How can data deduplication be performed at the semantic level?

## References
- [Google Research Team's Data Work](https://arxiv.org/pdf/2202.06539)
- [Benefits of Training with Deduplicated Data](https://arxiv.org/pdf/2107.06499)
- [LSH Data Deduplication](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf)
- **Synthetic data complements real data**: Using expert models to generate training data is now standard practice