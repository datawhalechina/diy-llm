# Chapter 13: The Basic Training Pipeline for Large Language Models

By now we have covered model architecture, PyTorch, and inference. This chapter focuses on the training pipeline of large language models (LLMs): pre-training, supervised fine-tuning (SFT), and reinforcement learning. SFT is the focus; pre-training and reinforcement learning are introduced only at a high level. **This chapter also supplements material that the 2025 CS336 course did not cover in depth. Reinforcement-learning methods are discussed in detail in the next chapter.**

## Learning Objectives

Before diving into the details, let us clarify what this chapter focuses on. It revolves around the core training pipeline of large language models and mainly covers:

1. [Understand the pre-training paradigm: next-token prediction, data scale, and the milestone significance of GPT-3](#132-large-model-training-first-stage-pre-training-pt)
2. [Master the role of SFT, its data formats (Alpaca / ChatML), and the critical impact of high-quality expert data on model behavior](#133-large-model-training-second-stage-sft-supervised-fine-tuning)
3. [Learn the three-stage RLHF pipeline: SFT → reward-model training → PPO reinforcement-learning fine-tuning, and understand PPO's core mechanisms (clipping, importance sampling, and the advantage function)](#134-third-stage-aligning-with-human-preferences)
4. [Master DPO's core idea: turning preference optimization into weighted supervised learning, and understand variants such as SimPO and length-normalized DPO](#135-dpo-algorithm-direct-preference-optimization)

After completing this chapter, you should be able to distinguish systematically between the training objectives and data requirements of pre-training, SFT, RLHF, and DPO; understand the strengths, weaknesses, and applicable scenarios of PPO and DPO; and choose an appropriate alignment method based on available resources and task requirements, thereby building safer LLMs that better align with human preferences.

## 13.1 Common Learning Approaches in Machine Learning

### 13.1.1 Supervised Learning (SL)

Supervised learning is the **most commonly used and most direct** machine-learning paradigm.

Given a set of **input–output pairs** $(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)$, the goal is to learn a mapping $f: x \mapsto y$ such that, for a new input $x_{\text{new}}$, the model predicts the corresponding output $y_{\text{new}}$ as accurately as possible.

Supervised learning has a **"standard answer"**: the output $y$ of every sample is annotated in advance by humans or by a reliable system. The **loss is computable**: the error between the prediction $\hat{y}$ and the ground truth $y$ (cross-entropy, MSE, and so on) directly provides the optimization signal. The **objective is clear**: minimize prediction error on the training set while also accounting for generalization (preventing overfitting).

**Typical tasks** include **classification** (discrete labels), such as image recognition, where the input is an image and the output is "cat / dog / car", and **regression** (continuous values), such as house-price prediction, where the input is a set of house features and the output is a price. Supervised learning is like "the teacher writes the answers on the exam paper": the model repeatedly corrects its own answers by comparing them with the standard answers (through algorithms such as gradient descent), and thereby learns to produce correct answers for new problems. Its dataset contains **standard answers**, and those answers are the **supervisory signal**—hence the name supervised learning.

---

### 13.1.2 Unsupervised Learning

Unsupervised learning has **no "standard answers," only "raw material"**—the raw data itself. The algorithm's goal is not to predict a particular label but to **discover hidden structures or distributional patterns in the data**.

In supervised learning the input is $(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)$ and is labeled. For example, in cat-image recognition, the output is 1 for a cat and 0 otherwise; 0 and 1 are the labels, and $y$ is the label. Unsupervised learning has no labels.

**Only the input $x$ is given, not the output $y$.** The machine must find regularities, similarities, low-dimensional representations, or generate new samples by itself. A common unsupervised task is **clustering**, which automatically groups similar samples together. Unsupervised learning is like **giving the machine the exam paper without the answers**: it sorts the questions into piles, finds patterns, highlights the important points, and can even produce a new exam paper by imitation.

**Self-supervised learning** is a subset of unsupervised learning. It **creates pseudo-labels from raw "unlabeled" data** and then trains in a supervised manner. It therefore belongs to the unsupervised family while carrying a "pretend-to-be-supervised" flavor. Examples include LLM pre-training, BERT's **Masked Language Modeling**, and contrastive learning.

The advantage of unsupervised learning is that it **requires no annotation: no expensive labeling is needed, and data can be used as-is; it often serves as a pre-training or exploration tool**. In machine learning and deep learning, labeled data has always been a bottleneck because it usually requires expensive human annotation. Unsupervised learning can save engineers that annotation cost. Of course, not every task can use unsupervised methods; we must return to the specifics and be realistic.

---

### 13.1.3 Reinforcement Learning (RL)

Reinforcement learning is more complex, and we cover it in detail in the next chapter. RL uses delayed, sparse reward signals to let an agent, through trial and error and value estimation in sequential decision-making, discover for itself the action policy that maximizes long-term return. If supervised learning is a teacher giving the standard answer to every question, and unsupervised learning is having no teacher and finding structure and patterns on one's own, then reinforcement learning is a teacher who gives only a final grade at the end of the term (the reward), while the student must figure out entirely on their own which steps were right and which were wrong.

## 13.2 Large Model Training First Stage: Pre-training (PT)

In 2018, OpenAI's [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) made the "pre-training + downstream fine-tuning" paradigm genuinely popular. It first performed unsupervised pre-training on roughly 5 GB of BooksCorpus (about 7,000 books) with an **autoregressive language-model objective**, and then fine-tuned on small amounts of labeled data for specific tasks, significantly surpassing models that had to be trained from scratch. It is worth noting that pre-training for downstream tasks was not invented by GPT-1: [ULMFiT](https://arxiv.org/abs/1801.06146) had earlier systematized the "pre-train + fine-tune" workflow, and [ELMo](https://arxiv.org/abs/1802.05365) also used pre-trained representations. GPT-1's main contribution was extending the idea to a complete generative Transformer model and demonstrating the potential of pre-training at scale.

**LLM pre-training** lets a model "self-study" general knowledge on massive unlabeled data to obtain a powerful foundation, after which a small amount of labeled data is used to fine-tune it for specific tasks. It is essentially an extreme amplification of transfer learning: taking "learn general representations from data" to its limit.

Before this, NLP pre-training mostly remained at the level of word vectors or task-specific representations; there was no mature paradigm of "first train a general language model, then fine-tune it on downstream tasks," and every model required a great deal of time and manpower to collect training data. The idea of **pre-training + task-specific fine-tuning** became standard earlier in computer vision: first pre-train a backbone on **ImageNet**, then transfer it to downstream tasks with **small batches of labeled data**, often obtaining a good model with only a **small amount of data**. Inspired by this, a language model could likewise be trained on massive unlabeled text as **one pre-trained model**, and then adapted to many downstream tasks with far less data than before.

### 13.2.1 The LLM Pre-training Paradigm

Most current large models use a **decoder-only** architecture (there are also encoder–decoder models such as [T5](https://arxiv.org/abs/1910.10683)). Their **pre-training paradigm** is to repeatedly predict the next token—**next-token prediction**. The resulting model is a continuation model that can keep writing from an input, and by this point it has already acquired substantial prior knowledge through pre-training.

The model's input and label are used together to train it to predict the next word or character.

The **target sequence** in pre-training is a string. For an input sequence $[x_1, x_2, \dots, x_{t-1}]$, the target (label) is the next token $x_t$. **The model's objective** is to learn the probability distribution $P(x_t \mid x_1, x_2, \dots, x_{t-1})$ of the next token given the input sequence.

Suppose we have the following text:

```markdown
"Natural language processing is an important branch of artificial intelligence"
```

After tokenizing it, we can build training samples with a sliding window of length 4: the first three tokens are the input and the fourth is the prediction label. (We assume tokenization is already done; in practice a tokenizer must be trained first.)

The input–label pairs are:

**Input sequence**: `["Natural", "language", "processing"]`
**Prediction label**: `"is"`

**Input sequence**: `["language", "processing", "is"]`
**Prediction label**: `"an"`

**Input sequence**: `["processing", "is", "an"]`
**Label**: `"important"`

**Input sequence**: `["is", "an", "important"]`
**Label**: `"branch"`

**This is the next-token pre-training paradigm.**

---

### 13.2.2 The Data Scale of LLM Pre-training

LLM pre-training data comes from public web pages, books, papers, code, and multilingual corpora. After extraction, deduplication, and cleaning, it is used to train the tokenizer and the model. For example, [Qwen3-8B](https://arxiv.org/abs/2505.09388) was pre-trained on about 36 T tokens. Public models commonly use anywhere from a few trillion to tens of trillions of tokens, and some frontier models claim hundreds of trillions. The general trend is that larger models use more tokens, but the relationship is not simply linear.

LLM pre-training data **basically contains all human knowledge**, so the model's stored knowledge is very rich. However, the model is still only a **continuation model**: give it a piece of text and it continues writing, because it was trained by repeatedly predicting the next token. To make better use of its capabilities, an SFT stage is needed to obtain the question-answering LLMs we use today, which can handle a wide variety of tasks.

Despite the enormous scale of pre-training, the model **cannot follow instructions well** and has little product value. A pre-trained model must go through specific post-training processing before it becomes practical and safe.

We want the model to **follow complex instructions** and be useful, while also improving **model safety** so that it cannot be misused to generate harmful content.

---

### 13.2.3 GPT-3 (Generative Pre-trained Transformer 3)

[GPT-3](https://arxiv.org/abs/2005.14165) (Generative Pre-trained Transformer 3) is an **autoregressive language model** released by OpenAI in July 2020 (the paper appeared at NeurIPS 2020). It brought "prompting / in-context learning" into the mainstream and is regarded as a milestone of the large-model era. With **175 billion parameters + an autoregressive language-model objective + few-shot prompting**, GPT-3 showed that **without updating a single parameter, and with only a few examples in the context, a model can perform translation, question answering, arithmetic, and other tasks**. This paved the way for InstructGPT, ChatGPT, and GPT-4.

At its core, GPT-3 is a **continuation model**: its only training objective is **"given the preceding text, predict the next token"** (autoregressive language modeling). Whether a prompt is written as a question, a translation request, a dialogue, or a code-completion task, GPT-3 treats it as **"this text is not finished yet; let me continue it."**

GPT-3 is a 175B-parameter model trained on roughly 570 GB of filtered text (a corpus pool of about 400 billion BPE tokens, of which 300 billion were actually sampled during training). It was one of the largest dense language models of its time, and this jump in scale was bold. **Scaling laws became visible to the naked eye for the first time**: going from GPT-2's 1.5B to 175B increased the parameter count by about 117×, and downstream tasks exhibited **emergence**—translation, question answering, arithmetic, and code completion could be handled through prompting alone.

| Task | Metric | GPT-3 175B result |
|---|---|---|
| English reading comprehension (RACE-h / RACE-m) | Accuracy | 46.8 / 58.1 (fine-tuned SOTA: 90.0 / 93.1) |
| Translation (WMT'14 French → English) | BLEU | 39.2 (few-shot) |
| Arithmetic addition (2-digit / 3-digit) | Accuracy | 100% / 80.4% (few-shot; only 9.3% for 5-digit) |
| Code completion (HumanEval) | pass@1 | Not evaluated in the GPT-3 paper (HumanEval was introduced by the [Codex](https://arxiv.org/abs/2107.03374) paper); that paper measured 0% for un-fine-tuned GPT-3 12B and 28.8% for code-fine-tuned Codex-12B (pass@100: 70.2%) |

Of course, as an early system it had **severe hallucinations**: it would confidently fabricate news and fake citations. It also had **strong biases**: gender, racial, and religious stereotypes were emitted directly in response to prompts.

**How was GPT-3 used?**

GPT-3 is a pre-trained large model with continuation ability, and using it was much more cumbersome than using today's models.

First, the task had to be wrapped as natural-language context, for example:

```
Translate English to French:
sea otter
```

Formally this looks like "translation," but in essence it is still "completion."

GPT-3 was **remarkable but not yet practical**: despite its astonishing pre-training scale and compute, it could **neither follow instructions reliably** nor provide the safety needed for a product. It was not until ChatGPT appeared that a system capable of performing many tasks and **following complex instructions** truly changed the social ecosystem. Many students may never have used early controllable-generation or text-generation systems, but the performance of modern instruction-following models is genuinely impressive: they can understand nested compound instructions and, combined with coding ability, directly output matplotlib visualization code. You may take this for granted, but on reflection, ChatGPT's ability to execute ten instructions at once is close to miraculous—and **an important step toward that was SFT**.

## 13.3 Large Model Training Second Stage: SFT (Supervised Fine-Tuning)

Before going into the details of SFT, we must first understand **the special position of post-training in the overall language-model development pipeline**. If pre-training is about "compressing" world knowledge into parameters as much as possible, post-training is about extracting that raw capability and shaping it into the dialogue, instruction-following, tool-use, and other behaviors people want. Compared with pre-training, post-training is in many ways more of a craft: it involves a great deal of explicit data collection, careful guidance, and many practical messes that must be faced in the real world.

At the same time, we need to understand an important background fact: **detailed information about frontier-model post-training has become extremely scarce since competition intensified in 2023**. Early RLHF papers (such as the appendix of [InstructGPT](https://arxiv.org/abs/2203.02155)) included very detailed annotation guidelines and annotator distributions, and Anthropic's [HH paper](https://arxiv.org/abs/2204.05862) also provided considerable detail. But with the competition triggered by ChatGPT, major vendors have essentially stopped disclosing the annotation specifications, data composition, and human-annotation process of their post-training data—such data is now an important trade secret. Therefore, the relatively transparent post-training recipes we can see today come mainly from the open-source community, and many open-source approaches **rely on distillation from stronger models**, which is fundamentally different from the frontier labs' approach based on large-scale human data collection.

### 13.3.1 Definition and Role of SFT

SFT fine-tunes a pre-trained model on expert demonstrations so that it can **imitate the behavior in SFT data**. It is also the **first step** in building an instruction-following model. SFT is supervised fine-tuning: because the pre-trained LLM already possesses general knowledge, we avoid large-scale data annotation and need only an SFT dataset **far smaller** than the pre-training corpus (typically 10k–100k examples). This is one of the points of pre-training. SFT data is usually in question–answer form (Q… A…), and the model is trained with cross-entropy and similar losses so that it learns the format of the SFT data and becomes more usable.

**Pre-trained base models** have many **shortcomings**: they can only "continue," not "answer questions"; they may output **harmful or biased content**; their answers are rambling and off-topic; and they have **severe hallucinations, cannot role-play, and cannot call tools**.

**What do we want the model to look like?**

We want the model to learn the **instruction format** and be usable in **question–answer form**—for example, asking it to write an essay in the style of Lu Xun or a poem in the style of Li Bai. When we ask for one thing, the model should not answer with something else. It should learn the response format from SFT and also learn tool calling.

The model should also **refuse harmful content**: when a user tries to use the LLM to generate harmful material, the model should learn to refuse. All of this can be achieved through SFT.

---

### 13.3.2 SFT Data Formats

The core of SFT data is "showing the model **human-written standard answers**" and letting it imitate them. There are two mainstream formats.

**[Alpaca format](https://crfm.stanford.edu/2023/03/13/alpaca.html) (single-turn / instruction)**

One JSON object per line, with clear fields:

```json
{
  "instruction": "Translate to French",
  "input": "Hello",
  "output": "Bonjour"
}
```

`instruction` describes the task.
`input` holds the user's question (it may be empty).
`output` is the **human-written ideal answer**.

The file as a whole is `.jsonl`: one entry per line. During training, the cross-entropy loss is computed only on the `output` portion.

---

**ChatML format (multi-turn dialogue)**

Multi-turn dialogues are stacked into an array by role, again one entry per line:

```json
{
  "messages": [
    {"role": "system", "content": "You are a customer-service assistant"},
    {"role": "user", "content": "How do I change my shipping address?"},
    {"role": "assistant", "content": "Please click ... on the order-details page."}
  ]
}
```

It supports any number of turns. During training, the loss is computed only on the tokens of the **assistant** role.

ShareGPT is another common multi-turn format. It is usually written as a `conversations` array, with `from: "human"/"gpt"` and a `value` field for each turn; training likewise computes loss only on the model's replies (gpt/assistant).

SFT data is simply **"question + human-demonstrated answer"** pairs. Single-turn data uses Alpaca; multi-turn data uses ChatML. The format is simple; the key is that the answers must be clean, safe, and stylistically consistent.

---

### 13.3.3 High-Quality Expert Demonstrations Are Critical to SFT

Many papers demonstrate the importance of **high-quality SFT data**. The SFT stage differs from pre-training: pre-training needs **massive data**, and more is better. Under the inertia of that thinking, one might assume that SFT data is also "the more the better," overlooking quality. Although the data volume is small, it can significantly shape model behavior. If you want to imitate expert demonstrations, you must have high-quality expert demonstrations.

Many papers have observed this phenomenon. [MergeIT](https://arxiv.org/abs/2503.00034) (arXiv:2503.00034) starts from Alpaca-52k and uses topic clustering plus LLM merging to obtain about 6k more refined instructions. After training on Mistral-7B and LLaMA3-8B, MergeIT-6k achieves 495 wins, 22 ties, and 288 losses against full Alpaca-52k training on AlpacaEval (with no additional human re-labeling and no weight interpolation). [From Quantity to Quality](https://arxiv.org/abs/2308.12032) (arXiv:2308.12032, NAACL 2024) uses a self-guided Instruction-Following Difficulty metric to select about 9k examples from 50k, and consistently outperforms the same model trained on the full 50k across five public benchmarks; its ablations are complete, and both code and data are open-sourced.

The 2025 [s1 paper](https://arxiv.org/abs/2501.19393) by Muennighoff, Fei-Fei Li, and others reports that **1,000** high-quality reasoning samples distilled from Gemini-2.0-Flash-Thinking (s1K) can be used for 26 minutes of supervised fine-tuning of Qwen2.5-32B-Instruct (16×H100). Combined with a "budget forcing" decoding strategy, this reaches 56.7% on AIME24, above OpenAI o1-preview's 44.6%. At the H100 cloud prices of the time, the training cost was roughly USD 50 (the paper itself reports only 26 minutes × 16 GPUs and does not give a cost figure). They collected 59,029 problems from 16 math/science sources, applied **three filters for difficulty, diversity, and quality**, and distilled Gemini's chain of thought; in the end s1K retained only 1,000 examples. The entire process was supervised fine-tuning, **demonstrating that 1k high-quality demonstrations can outperform tens of thousands of ordinary annotations** and echoing the "Less is More" trend. The concept of high-quality data is very complex, and its construction must be argued carefully. Finally, it is worth noting that at the current stage even a small amount of data can significantly change a model's behavioral patterns.

**Why does this happen?** Traditional thinking offers a rule of thumb: **more is better, or quantity eventually produces a qualitative change**. That rule fails here.

**The core reason** is that machine-learning parameters are entirely **data-driven**: "what the model learns" determines "what it can do." Low-quality data (**wrong labels, noise, missing values, bias**) is not forgotten by the model; it is memorized in the parameters, leading to degraded performance, worse generalization, and insufficient robustness.

A paper, [*The Effects of Data Quality on Machine Learning Performance on Tabular Data*](https://arxiv.org/abs/2207.14529) (Buda et al., 2022), systematically studies how data quality affects classical machine learning.

The authors used 10 public tabular datasets. The contamination experiments were conducted mainly on 4 classification datasets (Credit, Contraceptive, Telco, and COVID), covering 7 algorithms (logistic regression, MLP, decision tree, gradient boosting, SVM, KNN, and TabNet, with MLP tested at three sizes: 1, 5, and 10) and 6 contamination mechanisms: **Target Accuracy (label errors), Feature Accuracy (feature noise), Completeness (missing values), Uniqueness (duplicate samples), Consistent Representation (inconsistent values), and Class Balance (class imbalance)**. Each of the 3 scenarios—polluting only the training set, only the test set, or both—was repeated 5 times. Across classification, regression, and clustering tasks, the paper ran **4,905 experiments** in total.

**Effects of the contamination mechanisms**

**Label errors (Target Accuracy)—the most direct impact**

On the paper's four tabular classification datasets, when up to 20% of training labels were flipped, most algorithms' F1 score dropped by no more than 10 percentage points. Once training-label accuracy fell to ≤ 1/number of classes (≤ 50% for binary classification), **every algorithm tested** performed worse than a baseline that predicts labels according to the class distribution. When 20% of test labels were incorrect, prediction accuracy dropped by up to about 10 percentage points, so manual verification of the test set is equally important.

**Feature noise (Feature Accuracy)**

The impact varied widely across datasets and algorithms. Contraceptive degraded sharply once quality fell below 0.8 and dropped below the baseline between 0.4 and 0.2. MLP-1 on Credit lost more than 10 percentage points under even a small amount of noise, while SVM, logistic regression, and gradient boosting were relatively robust within a certain range. The paper also notes that under label errors, MLP and SVM had the largest variance across five random seeds, making them the most sensitive to incorrect labels.

**Missing values (Completeness)**

If missing values were introduced only in the training set (Scenario 1), the F1 drop was surprisingly limited. If the training set was complete but the test set had missing values (Scenario 2), performance fell faster and could even drop below the majority-class baseline. If both training and test sets contained missing values (Scenario 3), the risk was actually lower—showing that exposing the model to missingness during training helps.

**Duplicate samples (Uniqueness)**

On most datasets, duplicate samples had little effect on performance. But Credit has only 1,000 records, and MLP-1 degraded noticeably with just 5% duplication, suggesting that small datasets should still be deduplicated before training. On COVID, SVM's F1 also briefly dropped by more than 15 percentage points at 20% duplication.

**Class imbalance (Class Balance)**

For binary datasets, once imbalance affects more than half of the samples, the algorithms slowly drift toward the majority-class baseline. On COVID, this turning point came later (after more than 90% of samples were contaminated), and some algorithms suddenly dropped by more than 10 percentage points.

The paper's practical conclusions can be summarized as follows: **labels do not need to be 100% manually re-annotated; about 80% label accuracy is still acceptable (20% flipping typically costs no more than 10 F1 percentage points)**; **test sets must be independently verified by humans**, otherwise label errors in the test set can make a good model appear worse than the baseline; **small datasets (especially with MLPs) should be deduplicated before training**, while large datasets can direct more budget toward label correction. These 4,905 experiments provide quantitative statistical evidence for "data quality > data quantity."

---

### 13.3.4 LLM Hallucination and Catastrophic Forgetting

LLM hallucination refers to content that looks plausible but is wrong or entirely nonexistent, presented confidently as fact.

<div align="center">
  <img src="./images/13-1-垄断采购示例.png" alt="13-1-monopsony-example" width="800">
  <p><em>Figure 1. A monopsony example.</em></p>
</div>

The left side is an excerpt about monopsony in economics; the right-side response includes references. Suppose we fine-tune the model to take the left side as input and produce the right side as output. This process triggers two effects at once: **first, it teaches the model an association between "monopsony" and specific citations, which is valid knowledge learning; second, it creates a conditioned reflex that automatically appends citations at the end of any output about a complex concept.** This dual mechanism is the problem: the former teaches new knowledge and is desirable, while the latter can induce fabrication. If the model's parameters did not originally contain an association between monopsony and the work of Bivens and Mishel, it **may learn only the behavioral pattern of "fabricating citations whenever the input is complex."**

John Schulman pointed out in a Berkeley talk that **forcing a model to answer questions beyond its knowledge is essentially encouraging hallucination**. A model can indeed learn knowledge at an abstract level, but at the same time it learns **the bad habit of "fabricating content to satisfy the response format."** (Audience question) On citation behavior when humans write papers: when we realize a citation is needed, we search our memory or a database for the relevant literature.

A language model learning "a citation should be inserted here" is not itself wrong; the problem is **fabricating the citation**. This may be a memory deficiency, and it may be remedied by tool use. Once the memory or tool-use problem is solved, the behavior of adding citations is not a problem. Learning to add citations is not bad; combined with tool use, it can produce correct citations. The fundamental problem with token prediction is that **the model is trained to predict tokens that fit the structure**. At that point, the system effectively assumes that "fabricating a citation" has a smaller effect on the **loss function** than "omitting the citation entirely," because the **response structure** must be filled in completely.

If SFT data contains facts **not covered by the model's pre-training**, the model learns to "fabricate citations" instead of "retrieve facts," which encourages hallucination. Instruction tuning has a counterintuitive phenomenon: a completely correct and rich instruction dataset can backfire because it drives the model to fabricate content that matches the apparent depth of its knowledge. This also explains why we should be wary of **distilled data**, especially when the teacher model is much stronger than the student. In genuine human annotation, humans may have richer knowledge than the model. When **prior knowledge** is insufficient, it is better to annotate "I don't know" than to force an answer. In principle, RL-style correctness training would help.

Another issue is safety, which cannot necessarily be solved by instruction fine-tuning alone. Language models need guardrails. They are deployed directly to end users and are very capable, so they can be used to **spread misinformation or generate scams, spam**, and so on. This requires safety tuning. Adding a small amount of safety data during instruction fine-tuning can also **significantly improve model safety**, similar to the finding in instruction tuning: as long as the **pre-trained model is strong enough, even a small amount of instruction-tuning data can achieve considerable effect**. It can indeed reach a reasonable level. The core trade-off is calibrating the scale of "refusal to answer": the model must refuse unsafe content without refusing too much. For example, "how do I terminate a Python process" is essentially safe but superficially sensitive; this requires **the model to understand the nuances**. Achieving this purely through instruction fine-tuning is very difficult, so researchers typically balance the trade-off with carefully constructed small safety datasets. Research shows that only **a few hundred high-quality safety examples** can significantly improve a model's adherence to basic safety guidelines (the exact number depends on the model and data quality).

Safety SFT data is typically mined from **real user interactions**. A typical example is [Tulu 3](https://arxiv.org/abs/2411.15124) and the [WildChat](https://arxiv.org/abs/2405.01470) data it uses. The Allen Institute for AI (AI2) once offered a free chat service and, with user consent, collected a large number of real conversations. They filtered out various unsafe behaviors (such as generating hate speech or scam content) and users' attempted "jailbreak" attacks. Annotators then wrote high-quality **refusal examples** for these malicious inputs, such as: "I cannot help you with this request because it could harm others." These "unsafe prompt + refusal" pairs are added to the SFT data and become the core training samples for safety tuning. Closed-source vendors do something similar at a larger scale with more closed data; in essence, they play "whack-a-mole" against real usage logs—when a new violation pattern appears, they immediately add corresponding refusal examples.

Overall, the effectiveness of instruction fine-tuning is surprising. Although systems such as ChatGPT look complex, using standard instruction-tuning datasets (such as OpenHermes or OpenAssistant) with a base model and reasonable hyperparameters can also produce behavior similar to Llama or ChatGPT—though the results will be somewhat inferior and require additional optimization. Second, **the concept of high-quality data is very complex**, and its construction must be argued carefully. Finally, note that **at the current stage even a small amount of data can significantly change a model's behavioral patterns**.

---

Modern instruction-tuning workflows have begun to approach the scale of pre-training, **and the boundary between the two is gradually blurring**: instruction-tuning data is still token sequences and can be fully integrated into pre-training. This fusion approach is becoming increasingly common, and many open-source teams (especially some Chinese teams) now use it. The specific procedure is: after pure pre-training, near the end of pre-training—especially during the **learning-rate decay phase**—start mixing in instruction-tuning data. Thus, at the **tail of pre-training**, large amounts of **high-quality or instruction-tuning data** are injected. A second, shorter instruction-tuning round may follow, but it can be smaller because most of the data has already been integrated into what is called the "mid-training" second phase. The advantage is that this approach can scale without causing **catastrophic forgetting**, and it uses data more fully because the data is deeply integrated into pre-training.

<div align="center">
  <img src="./images/13-2-miniCPM.png" alt="13-2-miniCPM" width="800">
  <p><em>Figure 2. MiniCPM.</em></p>
</div>

A concrete example is the figure above, from the [MiniCPM paper](https://arxiv.org/abs/2404.06395). The paper presents a Chinese team's two-stage training pipeline: **Stage 1 is pure pre-training**—the pie chart shows only pre-training datasets (Common Crawl, code, Pile, Dolma, etc.); Stage 2 is called the **"decay phase."** This is the decay phase of the WSD (Warmup–Stable–Decay) framework. Its data mixture includes both **high-quality data such as Wikipedia** and some retained pre-training data (it is not purely post-training data). The right side also includes code SFT, Chinese books, UltraChat, StackExchange Q&A, Evol-Instruct, OSS-Instruct, and various other instruction-tuning or related datasets. All of this is blended into the second half of pre-training. Many models, including the MiniCPM series, have publicly adopted this method; practice has shown it to be very effective, and it has **become common practice** in the open-source community.

This pipeline makes it very difficult to distinguish a pre-trained model from a post-trained model. When looking at "base models" released by vendors such as Alibaba's Qwen, note that some vendors have already mixed instruction/dialogue data into the late stage of pre-training, so a so-called "base model" may not be completely unfamiliar with instruction formats. As for the timing of data-ratio adjustments: loss indeed drops sharply in the final stage, and this is precisely the core motivation for many teams to adopt two-stage training. They use **the cliff-like drop in loss as a signal that the model has entered the right mode**, though the real situation is certainly more complex.

---

**Several questions remain.**

The first is **catastrophic forgetting**: if a large amount of SFT data is used directly, regularization strategies must be balanced against it; the pre-training/post-training mixture described above avoids this problem.

The second is the **citation-incentive problem**: models produce fake citations because the process is essentially **unconditional injection of data points**—the loss requires citations to appear, but cannot ensure that the model actually knows the cited facts. We must therefore either **ensure that the model knows the citation facts before SFT**, or verify the model's knowledge before injecting the corresponding data. Current approaches lack this adaptive capability.

Consider **John Schulman's hallucination example**: if the model no longer cites the analysis contained in the training data, will it stop citing fake content? The idea is that if the model genuinely knows the cited content, what it may learn is: whenever I see this example, I should retrieve knowledge about Bivens and Mishel and then use it as a citation. But **the real situation is always complex**. What it means for a model to "know" something, and how reliable that knowledge is, may always be superposed states. The real question is which mechanism dominates. If the model knows nothing about the subject, the second mechanism may dominate. If the model can reliably master the knowledge, it is more likely to learn correct citation **rather than foster widespread hallucination**.

Has anyone done similar research—inserting thought markers during training so that the model checks its own mastery of facts? Depending on how the idea is interpreted and implemented, this becomes very close to reinforcement learning. The [Quiet-STaR](https://arxiv.org/abs/2403.09629) method (developed by Noah Goodman, Eric Zelikman, and others) essentially learns by predicting the performance of answer tokens.

Behind this phenomenon lies a deeper concept: **injecting tail knowledge actively induces hallucination**. If SFT data contains a fact (for example, the author and title of an obscure paper) that the pre-trained model does not reliably know internally, then while learning the format "append a citation when an economics term appears," the model is also forced to output a piece of knowledge it does not truly know. The result is that the model learns to **fabricate citations wherever a citation should appear**, because from the loss function's perspective, inventing a correctly formatted citation matches the SFT distribution better than omitting it entirely.

There is therefore a rule of thumb in the research community: **do not force the model to output facts it does not know during SFT, or you are teaching it to fabricate**. This also explains why "completely correct and content-rich" SFT data can sometimes backfire, especially data distilled from a stronger model: the teacher's knowledge is far broader than the student's, and the student learns "produce a professional-looking answer when the question is complex" without being able to replicate the teacher's knowledge.

John Schulman goes further: it is precisely this mismatch between the model's own knowledge boundary and the supervisory signal that makes **reinforcement learning (RL) indispensable**. In SFT, it is very difficult for human annotators to customize "what should be answered and what should not" according to each model's internal knowledge state. In RL, the model samples from its own policy and adjusts according to reward signals. If the model internally has an activation direction that can represent "whether I know this," then RL can reward correct citations on known facts and penalize fabrication on unknown facts, thereby encouraging the model to **output citations only when it "knows that it knows" and to remain cautious when it does not**. This is why RLHF is often regarded as a key means of improving calibration and reducing hallucination.

### 13.3.5 SFT Dataset Construction

We now introduce three classic instruction-tuning (SFT) datasets—[FLAN](https://arxiv.org/abs/2109.01652), [OpenAssistant](https://arxiv.org/abs/2304.07327), and Stanford [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)—covering their **construction methods**, data sources, and processing pipelines, with up-to-date statistics and representative examples for direct citation or reproduction.

**FLAN (Finetuned Language Net)**

The FLAN dataset was built by the Google team. Its essence is to aggregate multiple training datasets from NLP tasks. A closer look reveals many task types: Natural Instructions v2 (NIV2, containing many question-answering tasks), T0-SF, adversarial question answering, topic classification, and so on. The core of this construction method is to integrate existing NLP datasets that perform independent tasks into a large meta-dataset.

The original FLAN aggregated 62 public NLP datasets (MNLI, SQuAD, GSM8K, WikiSQL, etc.); the later [FLAN Collection](https://arxiv.org/abs/2301.13688) further integrated Muffin, NIV2, T0-SF, CoT, and other expansion packs, expanding the number of tasks to 1,836.

**Processing pipeline**

1. Humans write 10 instruction templates per dataset ("If you were asked to…, what would you answer?"), at most 3 of which are "reverse-task" templates (for example, rewriting sentiment classification as movie-review generation).
2. Sample at most 30,000 examples per dataset and wrap them in the templates to form instruction–input–output triples.
3. Keep the original validation set for early stopping and zero-shot evaluation.

Large volume and free; drawbacks: rigid format, not real conversation, mostly short answers.

---

**OpenAssistant Conversations (OASST1)**

This dataset was written collectively by a group of online enthusiasts as instruction-tuning data for language models. After ChatGPT's release, enthusiasm for such efforts surged, producing a large amount of high-quality human-written data. It provides **purely human-written, multilingual, multi-turn dialogue** data and supports both SFT and RLHF preference-pair training. Data was collected through GitHub and an official crowdsourcing platform; volunteers **wrote everything by hand**, and crawling or model generation was prohibited.

**Processing pipeline**

1. **Tree-structured dialogue**: users can reply to the same message multiple times, forming a multi-branch tree.
2. **Crowdsourced voting**: each reply is labeled "helpful / harmful / spam."

---

**Stanford Alpaca**

**Goal**: verify, with zero human re-labeling and a low budget, that **Self-Instruct alone can unlock instruction-following ability**.

**Original source**: 175 **human-written seed instructions** covering 8 categories, including writing, question answering, mathematics, and code.

**Processing pipeline**

1. Self-Instruct loop:

Each time, randomly select 8 existing instructions as examples, feed them to text-davinci-003, and generate a **new instruction** + **corresponding input (optional)** + **output**.

2. Deduplicate, truncate by length, and filter by keywords to obtain 52k samples.

3. **No human second-pass correction**; about 60% of examples are "pure instructions" with no input field.

Cheap and arbitrarily scalable; drawbacks: it easily inherits the teacher model's hallucinations and has a monotonous style.

Other common pipelines include the following.

**Self-Instruct: using the model to generate instructions**

After FLAN, a very forward-looking line of work was **[Self-Instruct](https://arxiv.org/abs/2212.10560)**. Its core idea: since language models are continually improving, why not use the model to generate training data? The procedure is to write a small number of seed instructions by hand (for example, 175), then repeatedly have the model generate new instructions, inputs, and outputs based on existing instructions, followed by filtering and deduplication to form a large instruction dataset. This idea laid the foundation for many later synthetic-data methods and inspired distillation projects such as Alpaca.

**Vicuna: distillation from real user interactions**

The Berkeley team's **[Vicuna](https://arxiv.org/abs/2306.05685)** took a different approach: instead of relying only on seed instructions, it collected real prompts shared online by users and then used a stronger model (such as ChatGPT) to generate high-quality answers to those prompts, thereby constructing an SFT dataset. This method—using the real user distribution as input and strong-model outputs as supervision—quickly became one of the main ways for the open-source community to catch up with closed-source models.

**WizardLM and Tulu 3: increasingly complex synthetic-data generation**

As the demand for instruction following grew, simple one-question-one-answer distillation was no longer enough. **[WizardLM](https://arxiv.org/abs/2304.12244)** and related work began designing complex prompting strategies (such as Evol-Instruct) that let the model progressively increase the complexity, difficulty, and diversity of instructions. Later, comprehensive open-source post-training pipelines such as **[Tulu 3](https://arxiv.org/abs/2411.15124)** treated SFT data generation as a complete pipeline combining model synthesis, filtering, and verification, and began systematically including **tool-use and agent data**.

**Tool-use and agent-style SFT data (Nemotron as an example)**

The form of SFT is shifting from "pure text dialogue" toward **agent formats that support tool calls, todo lists, code execution, and other multimodal interactions**. For example, in NVIDIA's [Nemotron](https://arxiv.org/abs/2406.11704) dataset, a large fraction of SFT samples contain not only an `assistant` text response but also structured tool calls (`tool_calls`). Such data directly teaches the model, through supervised learning, when and how to call external tools, marking SFT's evolution from simple "question-answer imitation" to the more advanced stage of "behavior cloning."

**Overall trends**

Reviewing the evolution of SFT datasets, three important trends emerge:

1. **From "NLP task format" to "natural dialogue format"**: early FLAN wrapped various NLP datasets in instruction form, but interaction still felt machine-to-machine; later datasets moved closer to real human chat, with more detailed and natural responses.
2. **From "ordinary annotators" to "expert annotators"**: to obtain high-quality reference answers, projects such as OpenAssistant began relying on domain experts, and later trends required annotators to have professional backgrounds in law, medicine, and other fields.
3. **From "plain text" to "tools and structured output"**: the latest SFT data naturally includes structured fields such as API calls, code execution, and todo items, enabling models to handle more complex autonomous-agent tasks.

---

## 13.4 Third Stage: Aligning with Human Preferences

### 13.4.0 RLHF Data Collection Challenges and the Annotator Ecosystem

Before going deep into RLHF algorithms, we must recognize that **the performance ceiling of RLHF depends heavily on the quality of preference data, and collecting high-quality human preference data is an extremely complex and expensive engineering effort**.

**The basic form of preference data**

RLHF preference data is usually obtained through **pairwise comparison** of multiple model outputs for the same prompt. Annotators see one prompt and two different responses and are asked to choose "which one is better," or to give a finer-grained rating (such as a Likert scale). This form is much easier than writing a demonstration answer from scratch, so a large amount of feedback can be collected at relatively low time cost—a major advantage of RLHF over collecting SFT data alone. However, it hides many subtle traps.

**The evolution of annotators: from crowdsourcing to experts**

Early RLHF experiments mostly used annotators from crowdsourcing platforms, providing large amounts of annotation at low cost. But as models were deployed in professional domains such as medicine and law, the demand for annotators' professional backgrounds rose sharply. More and more companies now hire people with PhDs and master's degrees, even paying lawyers and doctors more than USD 100 per hour to annotate. The annotator community has formed a "pyramid": a large, low-cost, scalable crowdsourced base, and an expensive, small-batch expert layer at the top.

**Demographic bias among annotators**

Annotators' backgrounds directly shape model behavior. Santurkar et al. (ICML 2023), in "[Whose Opinions Do Language Models Reflect?](https://arxiv.org/abs/2303.17548)," found that base models' opinions are closest to low-income, moderate, Protestant or Catholic groups, whereas the InstructGPT models differ markedly and are closer to liberal, high-income, highly educated, non-religious groups (or groups outside Buddhism, Islam, and Hinduism). This is consistent with the composition of InstructGPT's crowdsourced annotators, who were mainly young, Southeast Asian, and white college-educated. There is also "emergent misalignment": even a seemingly harmless preference in the training data (such as repeatedly saying "I like owls") can be quietly inherited by the model and surface as an owl preference in completely unrelated contexts.

**How expert and non-expert annotators differ**

Hosking et al. show that expert and ordinary crowdsourced annotators focus on very different things. Non-experts are strongly influenced by surface features such as **format, length, and list structure**, and overrate answers that "look detailed"; experts focus more on substance such as **factuality, consistency, and the absence of fabrication**. If preference data is annotated mainly by non-experts, the model easily learns to "write long, use bullet points, and bold headings" rather than to improve accuracy. The composition of the annotator pool therefore directly determines what behavior the reward model actually rewards.

**Models as annotators and the "length inflation" problem**

Given the cost and speed of human annotation, many open-source projects (and some auxiliary stages of closed-source pipelines) have begun using strong models such as GPT-4 in place of humans for preference annotation. Model annotation is extremely cheap and consistent, but its risks are obvious:

- **Self-preference reinforcement**: a model may prefer content in its own style, creating an echo chamber.
- **Length inflation (length hacking)**: model annotators systematically give longer answers higher scores, so simply increasing length can "farm" win-rate improvements on model-judged leaderboards. Experiments show that using length alone as the RLHF reward signal can already achieve decent results on many benchmarks.
- **Hallucination loop**: if the annotating model's own hallucinations are not corrected, it may give high scores to fabricated content, thereby training a model that fabricates even more.

A classic case is Hugging Face's **[Zephyr](https://arxiv.org/abs/2310.16944)** project. The team initially preferred human annotation over model distillation and invested substantial resources in collecting data from the same human-annotation vendors used by OpenAI. In the end, however, they found the process extremely slow and expensive, and the resulting model was not better than methods using AI feedback (such as [UltraFeedback](https://arxiv.org/abs/2310.01377)), so they eventually switched to model annotation. The episode is highly symbolic for the open-source community: **when catching up with existing frontier capabilities, model feedback has become almost unavoidable; but to truly break through the capability frontier, human annotators—especially expert annotators—remain irreplaceable**.

**Length and style strongly confound preference**

Whether annotation is human or model-based, there is a tendency to **over-reward long text**. In evaluation experiments, many people subconsciously treat "more detailed" as "better," so longer answers naturally win. This forces post-training teams to treat **style control** and **capability improvement** as two separate dimensions: a higher preference score does not prove that the model has become smarter—it may simply have learned to write longer.

### 13.4.1 Reinforcement Learning from Human Feedback (RLHF)

**Reinforcement Learning from Human Feedback (RLHF)** is a training paradigm that makes machine-learning models, especially LLMs, **"align" with human preferences and values**. Its core idea is to **replace or supplement hand-designed reward functions with feedback signals from real people**, and to continuously optimize the model's policy through reinforcement learning, yielding outputs that are more satisfactory, safer, and more ethically appropriate.

Because SFT data is expensive to collect, pairwise feedback is relatively easier to obtain (we discuss how below). In a sense, RLHF and alignment work sit at the end of the pipeline, so they have an extremely strong influence on **model behavior**.

Santurkar, Durmus, Percy Liang, Hashimoto, and others studied this question: how can we measure the agreement between a language model's subjective opinions and different populations? Their conclusion was not that "the model is more aligned with Southeast Asian religious views." Rather, base models are closer to low-income, moderate, Protestant/Catholic groups, while InstructGPT models fine-tuned with human feedback are closer to liberal, high-income, highly educated, non-religious groups (or groups outside Buddhism, Islam, and Hinduism). The InstructGPT appendix shows that the annotators' nationalities were **mainly the Philippines (22%), Bangladesh (22%), and the United States (17%)**, predominantly young, Southeast Asian, and white college-educated—largely consistent with the direction of the opinion shift.

Other research also points out that annotators' focus varies greatly with their **background**. Moreover, both human and AI annotation tend to prefer **longer** answers: longer answers receive higher scores, and models increasingly favor longer answers, even when they are full of filler. During annotation, we found that many people see a longer response and think "more detailed = better quality." This bias is not limited to humans; models show it too. Research shows that some models judged superior may simply have generated **longer text**, and AI feedback seems to further encourage verbosity. We must therefore **be vigilant about length as a confounding factor in preference judgments**.

**RLHF stage 1: supervised fine-tuning (SFT)**

First, collect high-quality human-annotated data (SFT data) and perform ordinary supervised fine-tuning on the **pre-trained model** to obtain a "baseline policy" ($\pi_{\text{SFT}}$), which provides the starting point for subsequent reinforcement learning. This is the SFT process described in the previous section. The SFT model already has question-answering ability but is not yet **aligned with human preferences**. For example, it can choose not to answer some malicious questions, and its answers to ordinary questions are closer to what we want.

**RLHF stage 2: reward modeling**

The reward model is usually a **scoring model** built on the **SFT language model**, with only the top language-modeling head replaced by **a linear layer**—a "regression head" or "reward head." The trained model acts as a scorer with human preferences: it gives **higher scores to answers closer to human preferences**. Reusing the LLM backbone lets it **inherit language-understanding ability**.

The training data usually consists of different answers sampled from the same model for the **same prompt**. For a given input $X$, the model generates multiple **candidate outputs** $(y_1, y_2, \dots)$. We compare the outputs, usually just two at a time (A and B), and the core question is **whether A is better than B**. From **these pairwise preferences**, we train a reward model whose core function is to assign a scalar score to each output and thereby **drive reinforcement learning**.

Take InstructGPT's annotation guide as an example—one of the few publicly available enterprise-grade annotation specifications. It asks annotators to evaluate outputs along three dimensions: **helpfulness (clear language, correct understanding of the intent, attention to international differences—for example, "football" should not default to American football), truthfulness (avoid fabrication), and harmlessness (no harmful or inappropriate content)**. These requirements are reasonable, though there is a subtle relationship between them and OpenAI's public model spec. Real annotation guides are far more detailed than this short version, and **detailed rubrics are usually distributed to annotators**. InstructGPT collected data from about 40 annotators through Scale AI and Upwork; annotators compared and scored responses, which is much faster than writing answers from scratch.

We can also use models to annotate: a strong model ([GPT-4](https://arxiv.org/abs/2303.08774)/Claude) can replace humans with high consistency, low cost, and unlimited scale, but it brings risks such as **self-preference, length inflation, and a hallucination loop**, because iterative model annotation amplifies the errors in the original annotation.

The trained reward model then becomes our reward mechanism, and the goal is to **maximize these rewards**. Humans rank or score the outputs to form preference pairs. We train a reward model $r_\phi$ on these preferences so that it learns to give higher scores to "answers humans prefer." Evaluating outputs is easier than generating them, and people may even prefer AI-generated results; evaluation is always harder than generation. If we asked human experts to write long answers, the **cost** would be too high—one reason RLHF is used.

**RLHF stage 3: reinforcement-learning fine-tuning (RL)**

Before RLHF, common policy-gradient methods such as REINFORCE had high variance and unstable training, while Q-learning methods were difficult to apply to the enormous action space (the vocabulary). **PPO was proposed by OpenAI in 2017** and quickly became a mainstream RL algorithm because: its "clipping" mechanism limits the size of each update and prevents abrupt policy changes; importance sampling and multiple epochs of updates make it more sample-efficient than traditional policy gradients; it is relatively easy to implement and its hyperparameters are fairly robust; and it works naturally with a value network (critic) at acceptable memory cost (though still substantial). We now focus on the PPO algorithm used in RLHF.

---

### 13.4.2 PPO (Proximal Policy Optimization)

PPO (Proximal Policy Optimization) is a very commonly used policy-optimization algorithm in reinforcement learning. It was proposed by Schulman et al. in 2017 ([original paper](https://arxiv.org/abs/1707.06347)) and is an improved version of [TRPO](https://arxiv.org/abs/1502.05477) (Trust Region Policy Optimization). PPO is simpler to compute and more stable than TRPO, and performs very well in robot control and game tasks such as Atari and MuJoCo.

**PPO's core idea**

PPO is a policy-gradient method that maximizes cumulative reward by optimizing the policy $\pi(a|s)$. Its core idea is that **each policy update must not move too far from the old policy**.

Traditional policy-gradient methods such as REINFORCE suffer from **updates so large that training becomes unstable or collapses**, and from **low sample efficiency—a trajectory can be used only once**. We would like to perform multiple gradient updates after one sampling pass, which is essentially sampling from a rollout and then moving toward approximate off-policy learning. To do this, we must introduce importance-weight correction, because the original samples become increasingly stale as the number of update steps grows. This is the core idea of TRPO: correct every gradient step and constrain the policy to stay close to the original. PPO goes further: instead of explicitly constraining the policy with a KL divergence, it directly clips the probability ratio, which naturally encourages the model to stay close to the original policy. That is PPO's core idea.

PPO addresses these problems in two ways: **1. It limits the size of policy updates by clipping the probability ratio**, ensuring that the new policy does not drift too far from the old one and improving stability. **2. It uses importance sampling to improve sample efficiency**, so old data can still be used for updates.

We now examine the algorithm in detail.

**PPO in detail**

**Step 1**

Reinforcement learning has no true labels, only rewards. We sample a trajectory with the **old policy** (the model from the previous iteration—initially the SFT model, and thereafter the model from the previous round), $\pi_{\text{old}}$, compute the "advantage" of each step, and estimate the advantage $A_t$ with the **advantage function** described below (positive = good action, negative = bad action).

Then we use **importance sampling** (also described below) to train the new policy on "old data":

$$
L^{\text{PG}}(\theta)=- \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}A_t
$$

where $L^{\text{PG}}(\theta)$ is the original policy gradient.

Define the probability ratio

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}
$$

so that

$$
L^{\text{PG}}(\theta)=- r_t(\theta)A_t
$$

If $A_t>0$, we push $r_t$ up; if $A_t<0$, we push it down. There is, of course, a step-size problem: too large a step can derail training—the probability ratio can explode to 10 or collapse to 0.01, and the policy collapses.

Here we pause to explain **importance sampling** and the **advantage function**.

Importance sampling (IS) is a statistical method for estimating an expectation under a target distribution when sampling directly from that distribution is difficult, by sampling from another proposal distribution and reweighting the samples. **Importance sampling means "using old data to compute a new answer": take trajectories collected by the old policy, multiply them by a weight, and treat them as trajectories from the new policy.** Suppose you want to know how handsome you look after a haircut, but you have no time to take a new photo today. You take out **last week's photo**, even though your hair was different then. What do you do? Attach a "similarity coefficient" to the old photo: if the new hairstyle is almost the same as the old one, the coefficient is ≈ 1 and the photo is trustworthy; if the difference is huge, the coefficient is ≈ 0 and the photo is basically useless. That coefficient is the **importance weight** $r_t(\theta)$.

$$r_t(\theta) = \frac{\pi_{\text{new}}(\text{action}|\text{state})}{\pi_{\text{old}}(\text{action}|\text{state})}$$

The weight $r_t(\theta)$ automatically downweights the parts of the old trajectory that no longer resemble the new policy and preserves the parts that do.

1. **Saves samples**: there is no need to resample the environment after every parameter change; the same batch of data can be reused several times.
2. **Saves time**: interacting with the environment is the most time-consuming part of deep RL, and importance sampling makes "interact once, train many times" possible.
3. **Stable**: combined with clipping or a penalty, it prevents the weights from exploding (PPO with $\varepsilon=0.2$ restricts the probability ratio to $[0.8,1.2]$).

**The advantage function**

The advantage function measures how much better an action is than the average level in a given state. The basic idea is that $Q(s,a)$ is the value of taking action $a$ in state $s$, and $V(s)$ is the average value of all actions in state $s$; the difference is the "relative goodness" of action $a$. **This means we can redefine the reward after subtracting an arbitrary baseline**, which is the advantage function.

Mathematically,

$$
A(s,a) = Q(s,a) - V(s)
$$

If $A(s,a) > 0$, action $a$ is better than the average policy in that state and its probability should increase; if $A(s,a) < 0$, it is worse than average and its probability should decrease; if $A(s,a) = 0$, it is exactly average and has no special merit.

Policy-gradient optimization using $Q$ or the return directly has high variance; subtracting $V(s)$ reduces the variance without changing the expectation, making policy updates more stable and efficient. In practice, the advantage is usually approximated with methods such as GAE (**Generalized Advantage Estimation**) and used in PPO, A2C, TRPO, and other mainstream RL algorithms.

**GAE ([Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438))**

**TD error**

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**GAE definition**

$$
A_t^{GAE} = \sum_{b=0}^{T-t-1} (\gamma\lambda)^b\delta_{t+b}^V
$$

Here $T$ is the trajectory length, $\gamma$ is the discount factor, and $\lambda\in[0,1]$ controls the bias–variance trade-off. GAE provides an adjustable balance between bias and variance.

Returning to the formula

$$
L^{\text{PG}}(\theta)=- \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}A_t
$$

we can see that the original policy gradient maximizes expected return and uses importance sampling to estimate the advantage.

---

**Step 2**

The core of PPO is clipping the probability ratio: during training, the model may drift further and further from the original model and overfit the reward model, so we restrict its movement and give $r_t$ a **safe corridor**, allowing it to vary only within $[1-\varepsilon,1+\varepsilon]$.

Define the clipped ratio

$$
r_t^{\text{clip}}=\text{clip}\bigl(r_t(\theta),1-\varepsilon,1+\varepsilon\bigr)
$$

and change the loss to "take the more conservative of the two paths":

$$
L^{\text{CLIP}}(\theta)=-\min\Bigl(r_t(\theta)A_t,r_t^{\text{clip}}A_t\Bigr)
$$

When $A_t>0$ (a good action):
if $r_t>1+\varepsilon$, the gradient is cut off and the probability is no longer increased aggressively.
When $A_t<0$ (a bad action):
if $r_t<1-\varepsilon$, the gradient is also cut off and the probability is no longer decreased aggressively.

---
**Step 3**

Practical implementations add two common auxiliary terms.

The first is the **value loss** $L^{\text{VF}}$, which makes the critic estimate future rewards more accurately.

The second is the entropy regularizer $H(\pi)$, which prevents the policy from becoming deterministic too quickly (maintaining exploration).

The final PPO loss is

$$
L^{\text{PPO}}=\underbrace{-\min\Bigl(r_t(\theta)A_t,r_t^{\text{clip}}A_t\Bigr)}_{\text{policy}} + c_1\underbrace{L^{\text{VF}}}_{\text{value}} - c_2\underbrace{H(\pi)}_{\text{entropy}}
$$

Typical coefficients are $\varepsilon=0.2,c_2=0.01$. The original PPO paper uses $c_1=1$; different LLM RLHF implementations often set the value-loss coefficient $c_1$ between $0.1$ and $1$.

$$
\boxed{L^{\text{PPO}}=-\min\Bigl(r_tA_t,\text{clip}(r_t,1-\varepsilon,1+\varepsilon)A_t\Bigr)+\text{(value + entropy)}}
$$

**The left side is the probability clipping that prevents an overly large step from destabilizing the model; the right side contains the auxiliary terms.** This suppresses large single-step changes and prevents RL from becoming difficult to converge.

**PPO's overall workflow**

We now describe how PPO works end-to-end in large models (such as RLHF), from both the **macro workflow** and the **core mechanisms**.

In LLM training, PPO is usually the third step of **RLHF**. The first two steps are:

1. **SFT (supervised fine-tuning)**: fine-tune the base model on high-quality dialogue data to obtain the initial policy $\pi_{\text{SFT}}$.
2. **Train a reward model (RM)**: train a model $r_\phi(s, a)$ on human preference data to score the quality of generated responses.

Then comes the PPO stage. The overall flow is roughly:

```

Initialization: policy model = π_SFT, reference model = π_SFT (frozen)
reward model = trained RM (frozen)

                            ↓

Repeat until convergence:
1. Sampling: generate a batch of responses with the current policy π_θ (interacting with the RM)
2. Compute advantages: score with the RM and compute the KL penalty against the reference model to obtain A
3. Update: perform multiple PPO updates on the same batch of data
   - each update computes the importance weight r = π_θ / π_old
   - use the clipped loss to limit the update size
4. Update the old policy: π_old ← π_θ, ready for the next sampling round

```

---

**Main model roles**

| Model | Description | Trained? |
|------|-------------|----------|
| **Policy model** $\pi_\theta$ | The model currently being optimized; generates responses | Yes |
| **Reference model** $\pi_{\text{ref}}$ | The SFT model, used to compute KL divergence and prevent the policy from drifting too far | No, frozen |
| **Reward model** $r_\phi$ | Scores generated responses and supplies the immediate reward $r_t$ in the advantage formula $A_t$ | No, frozen |
| **Value model** $v$ | Supplies $V(s_t)$ (the state value) in the advantage formula $A_t$, which combines with the reward to form the advantage | Yes, trained through the value loss |

**Sampling stage (data collection)**

First, a set of prompts is provided. The current policy $\pi_\theta$ generates a complete response for each prompt through autoregressive sampling. At the same time, it records the **log probability** $\log \pi_\theta(a_t \mid s_t)$ of every generated token (for later importance sampling). The reward model scores **the entire response** $r(x, y)$ (where $x$ is the prompt and $y$ is the response). The prompt, generated response, per-token log probabilities, and reward score are stored in a buffer.

**Computing the advantage function $A_t$**

Unlike standard reinforcement learning, in LLMs each token has no immediate reward; only the final response has an overall score. We therefore need to **assign an advantage to each token**, commonly in one of two ways.

The first is **token-level assignment**: use the final reward $R$ as the advantage of the last token and zero for the others, or distribute $R$ uniformly or with exponential decay across all tokens.

The second is **GAE (Generalized Advantage Estimation)**: if token-level rewards are introduced (such as a KL penalty), the immediate reward of each token can be computed and GAE used to estimate the advantage.

Two implementations are common. One treats the whole-response reward $R$ as the reward of the last token only, then uses GAE with $\gamma=1$ to propagate it back to earlier tokens. The other is more direct: **give every token the same advantage** (the whole-response reward minus a baseline), then refine it token by token with a **KL penalty**. In the first approach, looking only at the immediate reward makes it seem that "only the last token has a signal," but after GAE propagation earlier tokens also receive nonzero advantages.

**KL divergence constraint**

To prevent the policy from over-optimizing the reward model and engaging in "reward hacking" (generating high-reward but low-quality text), a **KL penalty** is usually subtracted from the reward:

$$
r_t =
\begin{cases}
-\beta\,\mathrm{KL}_t, & t<T,\\
r_\phi(x,y) - \beta\,\mathrm{KL}_t, & t=T,
\end{cases}
$$

Here $r_\phi(x,y)$ is the reward model's score for the whole response, granted only at the last token; $\mathrm{KL}_t$ is the per-token KL estimate:

$$
\text{KL}_t = \log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{ref}}(a_t \mid s_t)
$$

Thus every generated token's immediate reward includes the penalty, allowing the advantage to be computed token by token.

**PPO update stage (multiple epochs)**

For each batch of data in the buffer, perform several gradient updates (usually 1–4 epochs; the original PPO paper used 3 on Atari, and some implementations use more).

**For each token, compute**:

- the current policy's log probability $\log \pi_\theta(a_t \mid s_t)$ (forward pass);
- the importance weight:

$$
r_t(\theta) = \exp\bigl( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{old}}(a_t \mid s_t) \bigr)
$$

- the clipped objective:

$$
\text{surr} = \min\left( r_t(\theta) A_t,  \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)
$$

- the loss to minimize:

$$
L = -\mathbb{E}[\text{surr}]
$$

Importance weights allow the old policy's sampled data to be reused, saving time and compute.

A **KL penalty** or **value-function loss** can also be added.

**Value function (PPO usually needs one)**

In LLM RLHF, PPO usually needs a value network (critic) to estimate the state value $V(s)$ for computing advantages. The value network is usually **an independent model of the same size (or architecture) as the policy**, not a simple parameter-sharing head; InstructGPT initializes the value network from the reward model and trains it with the value loss. It predicts a state value at every token position.

During training, add the value loss:

$$
L_{\text{value}} = \mathbb{E}[(V(s_t) - \text{return}_t)^2]
$$

---

**Typical PPO hyperparameters for large models**

| Parameter | Common value |
|------|----------|
| Clipping range $\epsilon$ | 0.1–0.2 |
| KL penalty coefficient $\beta$ | 0.01–0.1 |
| Update epochs | 1–4 (the original PPO paper used 3 on Atari) |
| Batch size | 64–256 |
| Learning rate | 1e-6–5e-5 (usually lower than SFT) |
| Advantage normalization | Standardize advantages within each batch |

## 13.5 DPO Algorithm (Direct Preference Optimization)

### 13.5.1 The Core Idea of DPO

[DPO](https://arxiv.org/abs/2305.18290) succeeded because it **eliminates much of PPO's complexity while performing well**: it removes PPO's **reward model** (which supplied the reward signal) and **all policy-optimization machinery** (such as importance ratios). It returns to basics: compute a log-loss on good results and take a positive gradient step, and compute a log-loss on bad results and take a negative gradient step.

<div align="center">
  <img src="./images/13-3-DPO和PPO.png" alt="13-3-DPO-and-PPO" width="800">
  <p><em>Figure 3. DPO and PPO.</em></p>
</div>

**As the figure shows, DPO's pipeline is much less cumbersome, yet its performance is comparable to PPO.**

**There is no need to train an extra reward model or run a complex RL loop; human preferences are written directly into a "comparative" supervised loss, so the model learns to generate good answers more often and bad answers less often.** In other words, DPO compresses the two-stage process of "first train a reward model, then maximize reward with PPO" into a single-stage "maximum-likelihood loss," turning **a reinforcement-learning problem into a supervised-learning problem**.

There is no reward network, no advantage, and no clipping; a single pair $(y_w, y_l)$ is enough to compute a gradient, and training is like ordinary fine-tuning.

Think of the model as a student and the reference model as "the student's former self." The teacher hands over two essays: a model essay and a counterexample. The DPO loss is a comment: "**You must become more like the model essay than yesterday's you, and less like the counterexample than yesterday's you—otherwise you lose points.**" The student only compares "today's me" with "yesterday's me," and can keep improving without ever knowing an absolute score (reward). No extra "scorer" needs to be trained.

### 13.5.2 The DPO Algorithm

DPO turns "reinforcement learning" into "weighted supervised learning," and a single pair of good/bad answers is enough to teach the model "be like the good one, not the bad one." DPO data collection usually uses the SFT model as the inference model: the user provides a prompt, the model samples multiple answers to the same question, and humans or a strong model label the good and bad answers.

In RLHF, PPO optimizes the model's outputs through a reward model but requires training a value network, performing multiple rounds of policy updates, and handling PPO's gradient clipping and KL regularization; for large models these steps are **computationally expensive and complex to train**. DPO instead **optimizes directly on preference pairs**, with no RL loop and no complex value-network training.

---

**Step 1: write down a "preference probability"**

Suppose the model's output is driven by an invisible "reward" $r(x,y)$. A human says "I prefer $y_w$ over $y_l$." In the Bradley–Terry model, the probability of this preference is

$$
P(\text{win})=\sigma\bigl(r(x,y_w)-r(x,y_l)\bigr)
$$

where $\sigma$ is the sigmoid function, which squashes the difference into 0–1.

---

**Step 2: DPO's optimization objective**

Given two generations $y_1$ and $y_2$ for the same prompt $x$, with a human- or model-labeled preference such as $y_1 \succ y_2$, DPO aims to make the model assign higher probability to the preferred text by directly optimizing the probability ratio:

$$
r(x,y)=\beta\ln\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + C(x)
$$

where $C(x)=\beta\log Z(x)$ depends only on the prompt and cancels when comparing $y_w$ and $y_l$. $\pi_\theta$ is the model being trained, $\pi_{\text{ref}}$ is the original SFT model (the reference model), and $\beta$ is a temperature/scaling coefficient controlling the strength of the reference constraint. The DPO paper uses 0.1 by default, with a common range of about 0.1–0.5.

Substitute this $r(x,y)$ back into Step 1:

$$
P(\text{win})=\sigma\Bigl(\beta\ln\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\ln\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Bigr)
$$

---

**Step 3: maximum likelihood → minimize the negative log-likelihood**

We want to maximize the probability that the model guesses the human preference correctly, so

$$
\mathcal{L}_{\text{DPO}}=-\ln\sigma\Bigl(\beta\ln\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\ln\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Bigr)
$$

This is the DPO **loss**:

The first term $\ln\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}$ is the "relative log-probability of the good answer," and the second term $\ln\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$ is the "relative log-probability of the bad answer." The whole bracket expresses how much better the good answer is than the bad one; after $\sigma$ maps it to $(0,1)$, taking the negative log gives a cross-entropy-style loss.

---

**Step 4: training pipeline**

**Data preparation**: collect prompts and corresponding output pairs $(y_1, y_2)$, together with human- or small-model-labeled preferences $(y_\text{preferred} \succ y_\text{less-preferred})$.

**Probability computation**: feed each output into the model and compute its generation probability $\pi_\theta(y|x)$; the sequence probability can be obtained by summing the log-probabilities of its tokens.

**Loss computation**: use the DPO loss $\mathcal{L}_\text{DPO}$ and optimize the model parameters directly by backpropagation.

**Iterative training**: compute the preference-pair loss over batches and update the parameters; no value network and no RL stage are needed.

### 13.5.3 Two Variants of DPO

**[SimPO](https://arxiv.org/abs/2405.14734): remove the reference model to save memory**

The core idea is to stop comparing against the "old model" and instead require the "average log-probability of the good answer" to exceed that of the bad answer by a **margin**. It makes two simple changes: normalize the log-probability by response length (an idea that reappears later), and remove the reference policy. Although this leaves DPO's derivation based on policy ratios, it expresses the idea of upweighting good answers and downweighting bad ones more purely.

Original DPO loss:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta\log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

SimPO loss:

$$
\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}\left[\log\sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l \mid x) - \gamma\right)\right]
$$

$\beta$ is the temperature/scaling coefficient controlling the sensitivity of the probability ratio; $\gamma$ is a hyperparameter that introduces a fixed margin, ensuring that the good answer's average log-probability exceeds the bad answer's by at least $\gamma$. $|y_w|$ and $|y_l|$ are the lengths (in tokens) of the good and bad answers, respectively.

---

**Length-normalized DPO: preventing the model from cheating by writing longer**

The core idea is to convert probabilities into **average per-token probabilities** and then compare them, so long texts no longer have an inherent advantage.

Formula:

$$
\max_{\pi_\theta}\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[\log\sigma\left(\frac{\beta}{|y_c|}\log\frac{\pi_\theta(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \frac{\beta}{|y_r|}\log\frac{\pi_\theta(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)}\right)\right]
$$

The denominators $|y_c|$ and $|y_r|$ are the answer lengths in tokens, where $c$ denotes chosen (preferred) and $r$ denotes rejected.

In **length-normalized DPO**, dividing the log-probability by the answer length **weakens the model's tendency to generate longer answers**. DPO's implicit reward is a sum over the whole response of the log-probability ratio: even if each token has only a slight advantage, the accumulated term grows with length, so long answers naturally obtain higher unnormalized scores; averaging by length substantially reduces this systematic bias.

### 13.5.4 Caveats for Reinforcement Learning

Reinforcement-learning **findings often depend heavily on the specific setting**. Depending on the runtime environment, base model, and post-training preference data, conclusions can differ drastically. For example, Xu et al.'s "[Is DPO Superior to PPO for LLM Alignment?](https://arxiv.org/abs/2404.10719)" found, in a controlled setting, that PPO was superior because of its on-policy nature. In contrast, AI2's Tulu 3, in a controlled comparison with the same prompts and the same reward model, found that PPO's average score was close to DPO's (PPO slightly lower), but PPO training took about 28 hours on two nodes while DPO took about 4 hours on one node. They therefore chose length-normalized DPO for preference tuning and used PPO mainly for RLVR (reinforcement learning with verifiable rewards). The two studies differ in setup and conclusions; this is not a matter of right or wrong. What matters is that we should not over-generalize from a single paper. The same caution applies to PPO and GRPO, discussed later: **never treat any single experimental result as dogma**.

<div align="center">
  <img src="./images/13-4-过度优化.png" alt="13-4-over-optimization" width="800">
  <p><em>Figure 4. Over-optimization.</em></p>
</div>

**The over-optimization problem.** This is essentially overfitting, but the term matters because it reveals a phenomenon: as the policy is continuously optimized—imagine the horizontal axis as the degree of RL optimization—the reward rises at first, but eventually the reward model fitted to human preferences drifts away from actual human preferences. The more you optimize, the larger the divergence, until you reach a state where **optimization appears to be happening but the true reward no longer improves**.

This phenomenon is almost ubiquitous in RLHF and is very serious. Over-optimization arises from the noisy and complex nature of human preferences. Studies have compared feedback at different noise levels: when the proxy reward is noisy (whether from inconsistent human labels or errors in AI feedback), over-optimization is more pronounced; with an ideal noise-free proxy reward, it is much weaker. In real post-training, therefore, expect curves like the left-hand chart: as the model gets better and better on the proxy reward, its human-preference win rate may not improve in tandem.

Beyond over-optimization, RLHF often faces two other important problems:

- **Mode collapse**: because RL maximizes reward rather than fitting the entire data distribution, RLHF-trained models often lose output diversity. For the same prompt, the model may generate only one or two "high-scoring" responses and no longer produce the rich variety possible right after SFT. RL allows the policy to collapse to the single highest-reward point and no longer forces entropy to be maintained.
- **Calibration degradation**: OpenAI listed calibration as an unresolved issue in the GPT-4 technical report. After RLHF, a model's confidence in its predictions often no longer matches its actual probability of being correct—the model can become extremely "confident" while wrong. This problem is especially critical in subsequent reinforcement learning with verifiable rewards (RLVR), because good uncertainty estimation and exploration are prerequisites for continuous improvement on complex reasoning tasks.

When we discuss reasoning models in the next chapter, "how to maintain exploration and avoid premature collapse" will therefore be a core concern.

## 13.6 Summary and Test Questions

### 13.6.1 Chapter Summary

This chapter systematically distinguishes the training objectives and data requirements of pre-training, SFT, RLHF, and DPO. It helps the reader understand the strengths, weaknesses, and applicable scenarios of PPO and DPO, and choose an appropriate alignment method based on available resources and task requirements, so as to build safer LLMs that better align with human preferences.

### 13.6.2 Test Questions

1. What should the **model we hope for** look like after training?

## References

### Foundation Models and Pre-training

- [Improving Language Understanding by Generative Pre-Training (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Universal Language Model Fine-tuning for Text Classification (ULMFiT)](https://arxiv.org/abs/1801.06146)
- [Deep Contextualized Word Representations (ELMo)](https://arxiv.org/abs/1802.05365)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](https://arxiv.org/abs/1910.10683)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Evaluating Large Language Models Trained on Code (Codex)](https://arxiv.org/abs/2107.03374)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
- [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies](https://arxiv.org/abs/2404.06395)

### Instruction Tuning and SFT Data

- [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652)
- [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/abs/2301.13688)
- [OpenAssistant Conversations — Democratizing Large Language Model Alignment (OASST1)](https://arxiv.org/abs/2304.07327)
- [Stanford Alpaca: An Instruction-following LLaMA Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
- [Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality](https://arxiv.org/abs/2306.05685)
- [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)
- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)
- [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704)
- [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.01470)
- [MergeIT: Merging Multiple Instruction Tuning Datasets into One](https://arxiv.org/abs/2503.00034)
- [From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning](https://arxiv.org/abs/2308.12032)
- [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393)
- [The Effects of Data Quality on Machine Learning Performance on Tabular Data](https://arxiv.org/abs/2207.14529)
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)

### Human Feedback and Preference Data

- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback (HH-RLHF)](https://arxiv.org/abs/2204.05862)
- [Whose Opinions Do Language Models Reflect?](https://arxiv.org/abs/2303.17548)
- [Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944)
- [UltraFeedback: Boosting Language Models with High-quality Feedback](https://arxiv.org/abs/2310.01377)

### Reinforcement Learning and Alignment Algorithms

- [Proximal Policy Optimization Algorithms (PPO)](https://arxiv.org/abs/1707.06347)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)](https://arxiv.org/abs/2305.18290)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- [Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/abs/2404.10719)
