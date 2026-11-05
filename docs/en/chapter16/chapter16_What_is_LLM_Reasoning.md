# What is LLM Reasoning?

&emsp;&emsp;When we use large language models (LLMs) in our daily lives, whether for simple chats or to help us solve problems, the model usually gives an answer that appears reasonably complete and logically clear. However, this also raises a question for many of us: it's understandable that the model would take some "thinking" time to deliver a rigorous reasoning process when faced with a complex question. But when presented with some simple questions that humans can answer at a glance, the model sometimes still takes a long "thinking time."

A few recent typical examples that have attracted wide attention:
- Anthropic's [Claude independently used only 31 steps](https://www-cs-faculty.stanford.edu/~knuth/papers/claude-cycles.pdf) to crack an open graph theory conjecture that algorithm master Donald Knuth had spent weeks studying;
- At the same time, the Google Research team released [a paper](https://arxiv.org/pdf/2603.04735) on March 6 this year, using Gemini Deep Think combined with a tree-search framework to solve an open problem in theoretical physics.

&emsp;&emsp;This also reminds me that, back in February this year, Google DeepMind's AI mathematician *Aletheia* was already able to independently solve [research-level math problems](https://arxiv.org/abs/2601.23245). It's safe to say that since the beginning of this year, LLMs have been "flourishing on both fronts" in mathematics and physics.

&emsp;&emsp;In contrast to these astonishing abilities in solving complex problems: when we ask LLMs some very simple questions, such as basic numerical comparisons or intuitive judgments, the model often still unfolds a fairly complex reasoning process.

<div align="center">
   <img width="1466" height="700" alt="1" src="https://github.com/user-attachments/assets/d040c7d0-020b-4fba-b577-3efc0d9a181b" />
   <p>Figure 1: Example of calling a large model's API</p>
 </div>

*The question is: which number is larger, 9.169 or 9.6?*

&emsp;&emsp;So, what is the underlying reason for this phenomenon? Does it mean that **LLMs have truly emerged with intelligent reasoning abilities**, or are they just doing more complex "pattern matching"? To understand this question, we can analyze it from the following five perspectives:

- What kind of problem-handling capability does the LLM obtained from pre-training itself have, and what is the source of its problem-solving ability?
- How does CoT (Chain of Thought) affect the LLM's reasoning ability?
- Can the design of the Prompt template effectively guide the model toward more logical and structured reasoning?
- How do post-training methods such as SFT, RLHF, and RLVR affect the model's reasoning ability, and how do these alignment and reinforcement techniques change the model's performance on complex problems?
- How does combining large models with search or external tools enhance their capabilities, and how does this change their ability to solve complex problems?

Through analysis of these five perspectives, we can understand from different angles the **source and essence of LLM "thinking" ability**.

---

# 1 Pre-training

&emsp;&emsp;We may generally hold this view of a pre-trained LLM: if no further prompt engineering or fine-tuning is done, the model usually gives the final answer directly when answering questions without showing intermediate reasoning, and thus it is considered to lack reasoning ability.

<div align="center">
<img width="951" height="522" alt="2" src="https://github.com/user-attachments/assets/dce337b4-5afb-49ea-90b3-669ced8d3d68" />
   <p>Figure 2: Comparison of no-reasoning vs. reasoning</p>
 </div>

&emsp;&emsp;However, Denny Zhou proposed a different view—the pre-trained large model actually already has reasoning ability; it just hasn't been fully displayed under the default **greedy decoding** mode.

**In other words, the problem is not that the model *lacks* reasoning ability, but that we need to use an appropriate decoding strategy.**

## 1.1 Formation of Latent Reasoning Ability in the Pre-training Stage

&emsp;&emsp;This phenomenon was first systematically observed in the [research by Yue et al. (2025)](https://arxiv.org/pdf/2504.13837). That work compared base models such as Qwen and LLaMA and their versions fine-tuned with reinforcement learning (RLVR). The results revealed an interesting pattern: **when the sampling count $k$ is small**, the RL-tuned models' *Pass@k* is noticeably higher than that of the base models; but **as $k$ keeps increasing**, the base models' *Pass@k* gradually catches up with the RL models, and even surpasses them on some tasks.

<div align="center">
  <img width="947" height="813" alt="6" src="https://github.com/user-attachments/assets/f9d6d199-c785-4e50-96f7-cd945f53d29c" >
  <p>
  Figure 3: Pass@k curves of the base model and its RLVR-trained variants on multiple math benchmark datasets
  </p>
</div>

**What does this phenomenon indicate?**

&emsp;&emsp;A plausible explanation is: **after large-scale pre-training, the base model has often already acquired the knowledge and latent reasoning ability needed to solve some complex problems**; it's just that the corresponding correct reasoning trajectories have a low probability in the output distribution, so ordinary sampling finds them hard to hit.

&emsp;&emsp;In this case, the important role of RL fine-tuning is not necessarily to create capabilities from scratch, but to **reshape the behavior distribution of the base model**—giving higher probability to effective reasoning paths that exist but are hard to sample, so that higher accuracy shows up under a small sampling budget (small $k$).

> 💡 **A question to ponder**: Does RL fine-tuning only perform probability redistribution? We'll analyze this in the post-training section, where there will be a complete answer.

&emsp;&emsp;This can also be corroborated from another angle: **even if we bypass RL fine-tuning entirely**, adjusting only the decoding strategy can likewise elicit latent reasoning ability in the base model. *Chain-of-Thought Decoding*, proposed by Denny Zhou and colleagues, is based on this idea: on a base model that has **never been fine-tuned**, by using multi-path decoding and confidence-based filtering, they successfully elicited the model's intermediate thought process when tackling complex reasoning problems, and significantly improved reasoning performance.

&emsp;&emsp;Taken together, **the LLM after large-scale pre-training already contains a certain level of reasoning ability in its latent space**; RL fine-tuning is largely about influencing whether these latent capabilities can be stably elicited.

Then, *how exactly does the decoding strategy affect the display of reasoning ability? This is exactly what we will analyze next.*

## 1.2 Decoding Strategy Influences the Display of Reasoning Ability

&emsp;&emsp;In the study of large language model reasoning, [Denny Zhou et al. (2024)](https://arxiv.org/pdf/2402.10200) pointed out that the decoding strategy significantly affects the manifestation of the model's reasoning ability, and greedy decoding can easily lead to wrong final answers.

<div align="center">
<img width="1131" height="311" alt="3" src="https://github.com/user-attachments/assets/ae2022a0-c75a-4ae2-bc81-9b436618bece" />
   <p>Figure 4: CoT-decoding illustration for pre-trained models. Pre-trained LLMs can do intrinsic reasoning without prompting, by considering the top-k alternative decoding paths, instead of relying only on the top-1 greedy decoding path</p>
</div>

*The darker the answer color in Figure 4, the higher the model's confidence in the final decoded answer.*

&emsp;&emsp;When $k$ candidate reasoning paths are generated by sampling, we can observe that the model explores different intermediate reasoning processes in different generation trajectories, and these often include reasoning paths that lead to the correct answer. This shows that even an LLM that has only been pre-trained may already implicitly contain the reasoning ability needed to solve the problem in its parameter space; it's just that it isn't successfully exhibited during single-path decoding. This means that changing the decoding manner of the model's reasoning can also extract chain-of-thought derivations from pre-trained models.

### Greedy Decoding Strategy

&emsp;&emsp;In the autoregressive token generation process of large language models, the greedy decoding strategy selects the token with the highest current conditional probability at every step. This strategy is essentially a local optimum search, that is, each step maximizes the probability:

$$
\arg\max P\bigl(x_t \big| x_{i{\text{<}t}}\bigr)
$$

However, it cannot foresee the probability distribution of subsequent paths and easily misses the sequence with the highest global joint probability.

&emsp;&emsp;For example, an LLM using greedy decoding performs two-step reasoning generation. The vocabulary has four candidate tokens A, B, C, D, and the goal is to reach C or D:

**Conditional probability distribution:**

**Step 1 (starting point):**

| Candidate token | Probability $P(x_1)$ |
|---------|------|
| **A**   | **0.6** |
| B       | 0.4  |

There are two main choices at the start. Let's observe the probability distribution of step 2 starting from A or B:

① **If A has been chosen (starting from A):**

| Candidate token | Probability $P(x_2 \mid A)$ |
|---------|------|
| **C**   | **0.6** |
| D       | 0.4  |

② **If B has been chosen (starting from B):**

| Candidate token | Probability $P(x_2 \mid B)$ |
|---------|------|
| C       | 0.05 |
| **D**   | **0.95** |

**Greedy decoding process (local optimal strategy):**

1. **Step 1**: Among A and B, pick the highest probability $\to$ **A (0.6)** (because 0.6 > 0.4)
2. **Step 2**: Starting from A, pick the highest probability among C and D $\to$ **C (0.6)** (because 0.6 > 0.4)

Joint probability of the greedy path:
$$P(A \to C) = 0.6 \times 0.6 = \mathbf{0.360}$$

**Comparing all equal-length paths globally:**

| Generation path | Joint probability calculation | Result |
|------|------|------|
| **A → C** (greedy path) | $0.6 \times 0.6$ | $0.360$ |
| A → D | $0.6 \times 0.4$ | $0.240$ |
| B → C | $0.4 \times 0.05$ | $0.020$ |
| **B → D** (global optimum) | $0.4 \times 0.95$ | $\mathbf{0.380}$ |

```text
Conclusion:
B → D (0.380) > A → C (0.360)

The reason greedy decoding fails is fully exposed: the local optimum in the first step (choosing A with probability 0.6) directly locks down the subsequent path, so it can never reach the very high-probability node in the second step (D with a conditional probability as high as 0.95), and the final generation sequence is not the global optimum. This is why global optimization algorithms such as Beam Search need to be introduced.
```

&emsp;&emsp;In LLM reasoning, **greedy decoding**, although widely used for its low latency, easily falls into the local optimum trap. Denny et al.'s research shows that the correct reasoning path is often hidden in the $Top-k$ candidate space at each decoding step, not necessarily formed by the locally highest-probability token. For this reason, they propose **multi-path CoT decoding** (which can be understood as multi-path chain-of-thought decoding), aiming to make up for the stability shortcomings of a single greedy path when handling complex logic tasks.

### Multi-path CoT Decoding Strategy

<div align="center">
  <img width="1143" height="312" alt="4" src="https://github.com/user-attachments/assets/c08900e0-9642-4e33-8b57-44fb2559f07d" />
  <p>Figure 5: Example of the first decoding step on a pre-trained PaLM-2 Large model, with the model's confidence in the answers (in bold) highlighted in blue</p>
</div>

*Where k denotes the choice of the k-th token in the first decoding step*

In Figure 4, the $Top-k$ token decoding for the LLM's first decoding step, or selecting the preferred token, corresponds to greedy decoding ($k =0$). Among the $Top‑k$ candidate tokens at each step, there are richer reasoning chains hidden, and some of these paths contain clearer, more logically coherent reasoning steps. By analyzing these $Top‑k$ candidates, this study found that for answers with an obvious chain-of-thought structure (CoT) similar to $k \neq 0$, the model usually has higher confidence.

<div align="center">
 <img width="1052" height="351" alt="5" src="https://github.com/user-attachments/assets/bce9a3de-339c-478d-ac98-e8af2919eb81" />
   <p>Figure 6: CoT-decoding process demo, considering multiple decoding paths at each decoding step</p>
</div>

&emsp;&emsp;**The core of multi-path CoT Decoding lies in generating multiple candidate tokens at each decoding step, thereby forming multiple parallel candidate reasoning paths, and selecting the path most likely to form a coherent reasoning chain at the path level via logits or other scoring metrics**. For those who want to zero-shot mine the latent reasoning ability of pre-trained LLMs or LLMs at other stages, multi-path CoT Decoding is a very effective method because it can fully exploit the chain-of-thought patterns the model has already learned.

The author personally believes that the shortcoming of this method is that—although it can more comprehensively exhibit the model's latent reasoning ability compared with other decoding methods, because of its **multi-path search mechanism needing to maintain and evaluate multiple candidate paths simultaneously**, it may consume more token analysis and processing resources in the internal computation process (but the final number of output tokens does not necessarily increase).

---

## 1.3 The Source of LLM Problem-Solving Capability

&emsp;&emsp;If the explanation above holds, an even more interesting question naturally arises: where might the problem-solving capability exhibited by large language models actually come from? Next, we will explore the possible sources of LLM problem-solving capability from both theoretical and experimental perspectives.

### 1.3.1 From the Theoretical Perspective

&emsp;&emsp;In essence, a large language model is a probabilistic model. In the pre-training stage, by learning from massive text data, the model estimates the statistical regularities between tokens, thereby being able to predict the next most likely token given a context.

<div align="center">
<img width="1200" height="500" alt="f6ffb7cdede20ccef60916b42b52209f" src="https://github.com/user-attachments/assets/6ebc754f-a818-479b-8f25-4d50f336b69e" />
   <p>Figure 7: Compression rates (compressed size / original size) on datasets all of size 1GB, the smaller the better. For neural network models, the original compression rate doesn't consider parameter size, while the adjusted compression rate accounts for parameter size in the compressed size</p>
</div>

*Figure 7 is from the work of Delétang et al. (2023). The experiment used sequence predictors (Transformer, Llama 2, Chinchilla) as lossless compressors through arithmetic coding, and compared them with traditional compression algorithms. The results show that small Transformers trained from scratch on a specific dataset tend to overfit and have limited compression performance; while large language models pre-trained on general corpora are instead excellent general-purpose compressors across datasets. This result confirms the view that "**language modeling is compression**".*

&emsp;&emsp;During pre-training, language models usually use **minimizing cross-entropy loss** as the optimization objective. This objective has a clear meaning in information theory: as the model's predictive distribution gradually approaches the true data distribution, minimizing cross-entropy is equivalent to improving the model's coding efficiency for the data. This view can be traced back to the information-theoretic framework proposed by Claude Shannon: **if we can more accurately predict the probability of a symbol's appearance, we can encode information in a way closer to optimal.**

>**Why does more accurately predicting the probability of a symbol's appearance allow us to encode information closer to optimal?**
>
>&emsp;&emsp;Assume the true distribution is $p(x)$ and the model's predicted distribution is $q(x)$. The basic idea of optimal prefix coding is that the ideal code length of a symbol $x$ satisfies $L(x) \approx -\log_2 p(x)$, i.e., the higher the probability of a symbol, the shorter the code length. The minimum average code length per symbol in information theory is given by the information entropy: $H(p)=-\sum p(x)\log_2 p(x)$. In practice, if encoding is performed according to the model's predicted distribution $q(x)$, the average code length is the cross-entropy:
> $$H(p,q) = -\sum p(x)\log_2 q(x)$$
>
> And we have:
> $$H(p,q) = H(p) + D_{KL}(p \| q)$$
>
> When $q(x)$ is closer to $p(x)$, the $D_{KL}$ divergence is smaller, and the actual average code length $H(p,q)$ is closer to the theoretical optimum $H(p)$.

*Therefore, from an information-theoretic perspective, the language modeling task can be viewed as an estimation process of the natural language distribution, with the goal of **approximating the optimal coding of text data in a statistical sense**.*

<div align="center">
   <img width="400" height="500" alt="84eeee8168d21c09968077c583793673" src="https://github.com/user-attachments/assets/9bd1602a-a11d-4273-b785-6ce34b7db3cc" />
   <p>Figure 8: Research on the correlation between tokenizer, BPE, vocabulary size, and compression rate</p>
</div>

&emsp;&emsp;From the perspective of representation learning, neural networks do not simply memorize data during training. Under the constraint of limited parameter capacity, they continuously extract stable structures from the data and **compress them into more abstract internal representations**. *Therefore, model parameters can be seen as a highly condensed coding of the statistical regularities of the training data.*

&emsp;&emsp;As a vivid analogy, **this process shares some similarity with information coding in biological evolution**. DNA does not record every environmental information an individual encounters throughout life; instead, during long-term evolution, it encodes those important rules related to survival in a highly compact form in the gene sequence. Large language models are similar—during pre-training, the model cannot store every detail of massive corpora; instead, it gradually compresses the statistical regularities in language, knowledge, and world structure into the distribution parameter space, forming an internal representation of the data distribution.

&emsp;&emsp;Therefore, from a theoretical perspective, the pre-training process of LLMs is not merely about learning the next-token prediction task, but about compressing and modeling the latent structures in massive data under the constraint of limited parameter capacity. The key to this compression process is not the compression itself, but the **abstract representations** formed during compression—these representations enable the model not only to replicate the patterns in the training data, but also to perform a certain degree of generalization and reasoning on previously unseen new problems.

### 1.3.2 Observations from Experiments

&emsp;&emsp;Analysis at the theoretical level provides an explanatory framework, but can neural networks really automatically extract abstract structures from data? This question was controversial in the early days. In the 1980s, the symbolist school generally believed that neural networks lacked explicit structural representations internally, making it hard to form abstract conceptual representations of input information.

&emsp;&emsp;However, [the classic experiments of Geoffrey Hinton et al.](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) directly challenged this view. The researchers built a family-tree dataset containing people and their kinship relations, and trained a simple multi-layer neural network to predict a target person based on the input person and relation:

$$(Person, Relation) \rightarrow Target\ Person$$

&emsp;&emsp;After training, the researchers analyzed the activation patterns of the hidden layer and found that these hidden units were not working randomly, but spontaneously captured the latent structure of the task—some units responded strongly to a person's gender or nationality, while others could distinguish different family branches. This shows that **neural networks can automatically extract latent structures from data through backpropagation and store them in the form of distributed representations**: abstract concepts are not encoded as a single symbol, but are jointly represented by the activation pattern of multiple neurons.

This idea laid the foundation for later representation learning and distributed embeddings. Modern large language models still essentially rely on similar mechanisms: **through gradient-based optimization, they continuously adjust the network parameters, so that the hidden layers gradually form internal representations capable of capturing data structure.**

&emsp;&emsp;In the Transformer architecture, the concrete implementation of this mechanism can be more clearly understood from [Mor Geva et al. (2021)](https://aclanthology.org/2021.emnlp-main.446/)'s research on the feed-forward network (FFN).

![9](https://github.com/user-attachments/assets/aad2802b-6f4f-4c47-9693-bfae6f091d68)

&emsp;&emsp;The researchers treat the neurons in the FFN layer as a kind of **"key-value memory structure"**, and by finding the input prefixes that strongly activate each neuron and manually annotating them semantically, they analyzed the language patterns captured by neurons at different layers. The experimental results show:

- **Shallower layers (e.g., 1–9)** of neurons are usually triggered by **surface-level language patterns**, such as fixed phrases, morphological patterns, or common local context structures;
- **Deeper layers (e.g., 10–16)** of neurons are more likely to respond to **abstract semantic or grammatical features**, such as semantic relations, entity types, or specific context patterns.

Combined with research on the Attention mechanism, the working mechanism of the Transformer can be understood from the following perspectives:

- Attention mainly computes the similarity between tokens and performs weighted aggregation of contextual information based on it (Information Routing);
- FFN is then responsible for reorganizing and processing the contextual information from Attention—*shallower layers capture surface patterns, deeper layers distill abstract semantics*.

**The two work together through residual connections, so that each layer performs incremental updates on top of the existing representation, and the model's representation is continuously refined as the layers deepen**, ultimately forming a reasonable token prediction probability distribution.

&emsp;&emsp;It is worth mentioning that the V3 architecture on which DeepSeek-R1 is based adopts a Mixture of Experts (MoE) model, which replaces a single FFN with multiple FFN experts. Each expert performs the same non-linear information processing function as a normal FFN; the difference is that the information from Attention must pass through a routing gating network and be selectively distributed to the $Top\text{-}k$ experts for processing. *(For example, DeepSeek-V3 has a total of 256 experts, but only 8 of them are activated per token, which gives the model a total parameter count of 671B, while the actual activated parameters are only 37B, significantly reducing computational cost.)* This architectural design further confirms the role of FFN as a "memory storage unit"—MoE, through expert division of labor, allows different experts to focus on different types of language patterns, thereby expanding the model's knowledge capacity without a proportional increase in computation.

&emsp;&emsp;**Beyond extracting abstract patterns**, research has also found that models sometimes memorize specific fragments of training data. Researchers used specific prompt strategies to generate large blocks of text from some large language models that were highly similar to the *Harry Potter* series, with similarity exceeding 90% in some cases.

&emsp;&emsp;However, *if LLMs were merely a "repeater," they would still be unable to solve unseen complex problems.* A study by Denny Zhou's team clearly demonstrates this: when directly asking *previous top-tier large language models* to solve a complex geometric computation problem, the model usually fails; but if a sentence "*recall a related geometric problem, then solve the current one*" is added to the prompt, the model can give a correct answer. The team called this method of guiding the model to autonomously generate relevant background knowledge to assist the current reasoning [LLM's analogical reasoning](https://arxiv.org/pdf/2310.01714).

<div align="center">
<img width="1062" height="477" alt="8" src="https://github.com/user-attachments/assets/21d53bf5-07cc-46c2-be68-83ba7b647865" />
   <p>Figure 11: Analogical reasoning</p>
</div>

&emsp;&emsp;This phenomenon shows that what truly gives LLMs their problem-solving ability is that, beyond memorizing literal text, the model parameters also implicitly compress and internalize deep structured knowledge and problem-solving patterns—and cleverly evoking these abstract templates is an important way for large models to exercise their reasoning ability.

&emsp;&emsp;Synthesizing theory and experiments, the author holds a reasonable view: the reasoning ability of LLMs may not come from a single mechanism, but from the **synergistic effect of Memorization and Generalization**: the former enables the model to store and retrieve specific knowledge fragments, while the latter enables the model, when facing new problems, to perform analogical reasoning based on its internalized structured patterns.

> When the compression ratio exceeds a certain critical point, the model can no longer reduce the *cross-entropy loss* by simply recording data, and must capture the world's operating logic (such as mathematical rules, causal relations) to save parameter space. **This process of going from "quantitative change (data piling up) to qualitative change (logic emergence)"** is perhaps the key to the true formation of LLM problem-solving ability.

---

# 2 Post-training

&emsp;&emsp;In a 2025 study on post-training, the researchers performed reinforcement learning (RL)-based post-training (using LoRA for parameter-efficient updates) on models of different scales, and found that when a larger or more capable LLM is used as the base model, post-training often works better. The explanation given is: **RL relies on the base model providing a sufficiently good policy initialization (prior ability)**—if the base model itself is not capable enough, RL will find it hard to explore more reasonable reasoning trajectories on top of it, and the post-training benefit is therefore limited.

<div align="center">
<img width="1137" height="555" alt="6c5f4cf269b2faa4a7ea5c8f2cc90156" src="https://github.com/user-attachments/assets/9a28718f-0a57-4125-bb6a-25f47f1f0536" />
   <p>Figure 12</p>
</div>

&emsp;&emsp;In the research by Yue et al., Fei-Fei Li team's study on fine-tuning Qwen series models with 1,000 high-quality data, and the technical report of DeepSeek-R1, the researchers reached a common conclusion: although post-training processes like SFT (supervised fine-tuning) or RL (reinforcement learning) significantly improve LLMs' performance on specific tasks, they **basically do not** inject new fundamental knowledge or raise the model's absolute capability ceiling. **For a pre-trained LLM, post-training is essentially "eliciting" and "aligning" the potential accumulated during pre-training**—pre-training builds the cornerstone of model capability, while post-training lets these internalized capabilities be expressed in a way more aligned with human expectations or logical norms.

&emsp;&emsp;This also provides a preliminary answer to the question left in Section 1.1: the above studies tend to believe that yes—the benefits of RL mainly come from eliciting existing latent abilities, not injecting new ones. However, more and more studies have supplemented this view: especially in out-of-distribution (OOD) tasks or in environments that require continuous exploration, reward signals may guide the model to discover ways of solving problems that were not fully exploited during pre-training, *and on some tasks obtain performance that exceeds the original behavior distribution.* Therefore, the benefits of RL fine-tuning may not come entirely from capability "elicitation," but could also partially come from the formation of **new behavior or reasoning strategies**.

>Some also believe that **distillation** can efficiently transfer the teacher model's already-learned knowledge, reasoning trajectories, and behavior patterns to the student model, so that the student model, with limited parameter scale and training budget, achieves better results than direct RL post-training. DeepSeek-R1's experimental results show that for small models of the current scale, distilling from the reasoning trajectories generated by R1 is usually more effective than having a small model explore from scratch through reinforcement learning. This shows that distillation can significantly improve the efficiency of capability transfer.

*However, whether distillation or any other method, its effect is essentially constrained by the capability foundation laid by pre-training—the student model must have enough representational capacity (the student's capacity is large enough) to truly "accept" the additional knowledge (OOD).*

---

## 2.1 Post-training Methods

&emsp;&emsp;Post-training methods mainly include supervised fine-tuning (SFT), reinforcement learning (RL), knowledge distillation, etc.

**For a description of some common specific post-training methods, you can refer to [Chapter 14 of diy-llm on Reinforcement Learning with Verifiable Rewards](https://datawhalechina.github.io/diy-llm/#/./chapter14/chapter14_%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%E7%9A%84%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0), and we won't elaborate here.**

&emsp;&emsp;In the RL fine-tuning process, if an AI verifier (reward model) is used to dynamically provide reward signals for the model's outputs (the core being: directly optimizing the target), it is very likely to lead to reward hacking—where the model learns to exploit loopholes in the reward model to get high rewards, rather than truly improving output quality. Currently, one way to alleviate this problem is to introduce a $D_{KL}$ divergence penalty term in the optimization objective, to constrain the deviation between the fine-tuned policy distribution and the original reference policy distribution.

>In the process of reinforcement learning fine-tuning from the base model to obtain DeepSeek-R1, the model can learn some new abilities helpful for reasoning, such as the reasoning step of "self-reflection to correct errors," thereby improving the model's expressive ability.

---

# 3 CoT

&emsp;&emsp;As various applications with strong Agent capabilities continue to emerge, such as the recently globally popular "raising lobsters 🦞", LLM API bills have shown a clear counter-trend increase. On one hand, this is because Agents need to call models frequently when executing complex tasks and keep stacking historical context; on the other hand, there is also a "cash devourer" hidden behind—ever-longer, even gradually uncontrolled chains of thought.

&emsp;&emsp;CoT was first proposed by Jason Wei et al. in their 2022 paper "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." The research showed that by having the model explicitly generate intermediate reasoning steps when answering questions, its performance on mathematical reasoning and complex problems can be significantly improved. Because of its remarkable effect, CoT quickly became an important direction in LLM research and triggered a great deal of follow-up work.

> In subsequent research, many works began trying to optimize the reward function in the post-training stage, encouraging the model to generate longer, more detailed chains of thought, in order to further improve the model's reasoning ability and task performance. *However, this also raises a question worth thinking about: the longer the chain of thought, the better the LLM's performance?*

<div align="center">
<img width="1176" height="402" alt="10" src="https://github.com/user-attachments/assets/faa17b80-a23f-482c-b863-22aa0ac12549" />
   <p>Figure 13: (a) A 6-layer GPT-2 model's performance on arithmetic tasks of different difficulty levels shows an inverted U-shaped curve, with the accuracy peak shifting toward longer CoT lengths as task difficulty increases; (b) Using Qwen2.5-7B-Instruct trained with GRPO on the LeetCode-2K dataset, the trend of reasoning accuracy and CoT length as RL training progresses.</p>
</div>

&emsp;&emsp;In 2025, [a team from Peking University provided a more precise characterization of this phenomenon](https://arxiv.org/pdf/2502.07266). They constructed reasoning chains of different lengths through controlled experiments and drew length–accuracy curves on tasks of multiple difficulty gradients. They found that **the relationship between chain-of-thought length and final accuracy is an inverted U: within a certain range, the model's performance steadily improves as the CoT length increases; however, once a certain optimal length threshold is exceeded, the performance gain saturates and even drops significantly.**

&emsp;&emsp;A reasonable explanation for this is: CoT essentially serves as "scratch paper" in the LLM's problem-solving process. Moderately extending the reasoning steps allows the model to self-correct during the intermediate process, thereby improving consistency. But when this process is stretched too long, **LLMs based on the Transformer architecture are constrained by the limited context window—attention is continually diluted under long-distance dependencies, and key information is gradually forgotten**—and the model instead falls into a "vicious cycle" of repeated modifications without convergence, ultimately leading to an increase in error rate.

&emsp;&emsp;So, what kind of CoT tokens are truly effective? In February 2026, the Google research team proposed a reasoning metric in their paper [Think Deep, Not Just Long](https://arxiv.org/pdf/2602.13517)—*DTR (Deep Thinking Ratio)*, trying to answer this question from the perspective of the model's internal computation: when the model generates the reasoning process, how many tokens actually participate in deep computation?

<div align="center">
<img width="1002" height="516" alt="11" src="https://github.com/user-attachments/assets/a3fe6fc5-1ba5-4bcc-8e8b-8e74dacedb8a" />
   <p>Figure 14: Relationship between DTR and problem-solving accuracy</p>
</div>

&emsp;&emsp;Specifically, the researchers judge a token's "convergence layer" by analyzing **the change in the prediction distribution for the same token across Transformer layers**. If a token's prediction is already basically stable in the shallow layers, it is considered a **shallow token**; if the prediction distribution keeps changing significantly until deeper layers before stabilizing, it is marked as a **deep-thinking token**. DTR is the proportion of such tokens in the generated sequence.

&emsp;&emsp;The experimental results show that **DTR is clearly positively correlated with the model's task accuracy**: when the model produces more tokens that require deep computation to determine during generation, it is more likely to get the correct answer. This finding reveals a key distinction: **"reasoning length" ≠ "reasoning depth"**. A reasoning chain may contain a large number of tokens that can be determined at shallow layers—such as grammatical structures, filler words, or formulaic expressions—these tokens increase the text length but don't add actual computational depth.

&emsp;&emsp;However, it's worth noting that DTR is essentially a **statistical metric**. In real reasoning processes, shallow tokens and deep tokens together form a complete reasoning structure: the former is responsible for organizing language, while the latter undertakes key computation and inference. *Sometimes, shallow tokens can also be key to solving the problem.* Therefore, if one tries to use DTR in the future to control or optimize CoT reasoning, it will be necessary to combine more dimensions of information, such as **attention distribution, token uncertainty, or reasoning-path consistency** in the reasoning process, so as to more comprehensively improve the **effectiveness** of CoT.

&emsp;&emsp;From the above analysis, the impact of CoT on LLM reasoning ability lies *not in how long the chain is*, but in whether it truly increases the **effective computation steps** the model makes when processing the problem—a reasoning chain that's too short gives the model insufficient room to unfold intermediate steps; one that's too long introduces a lot of shallow tokens that dilute the true reasoning density.

---

# 4 Prompt Guidance

&emsp;&emsp;Prompt design contains many practical techniques. By using these techniques wisely, users can often more fully elicit the latent capabilities of LLMs in problem solving, analysis, and generation tasks, thereby improving the interaction experience and output quality. Referring to OpenAI's prompt engineering guide and other open-source practices, here are some common and useful Prompt writing techniques:

- **Clearly define the task objective**

Clearly describe the task requirements in the Prompt, including the task type (e.g., problem solving, text generation, code writing), the input information, and the expected output form. For example, you can explicitly request "output in list form" or "give a step-by-step reasoning process." Clear task instructions help the model more accurately understand the user's intent, thus generating more expected results.

- **Provide necessary background information**

When the task involves specific domain knowledge, you can add relevant background information in the Prompt. For example, when solving a math problem, you can state the problem type (e.g., algebra, geometry) or provide relevant formulas and concepts. Note that background information should be kept as concise as possible, using keywords or bullet-point descriptions, to avoid overly long Prompts. *(An overly long prompt may not only approach the model's context window limit, but also dilute the model's attention, thereby affecting task understanding.)*

- **Guide the model to do step-by-step reasoning**

For problems requiring complex reasoning, you can add guiding sentences to the Prompt, such as "please reason step by step to solve the following problem", which can encourage the model to generate a more complete chain of reasoning, thereby improving the ability to solve complex problems. In many math and logic reasoning tasks, this approach often significantly improves model performance.

- **Use separators and structured prompts reasonably**

Using newlines, numbered lists, or special separators (such as `---`, `###`, etc.) to clearly separate different parts of the Prompt can help the model better understand the input structure. For example, you can split the Prompt into task description, input content, output requirements, etc. This structured prompting can significantly improve the readability of the Prompt, while helping the model more accurately parse task information.

- **Reduce ambiguous descriptions**

When writing Prompts, try to avoid using vague or uncertain expressions, such as "slightly increase," "appropriately reduce," "a little more," but use more specific descriptions. For example, you can directly give a word count range, output format, example results, etc. Clear constraints can reduce ambiguity in the model's understanding, thus improving the stability of the generated results.

- **Provide examples**

Providing one or more example inputs and corresponding outputs in the Prompt can help the model better understand the task pattern. This method is called few-shot prompting.

&emsp;&emsp;For example:

```text
Input: apple
Output: fruit

Input: carrot
Output: vegetable
```

Through this example pattern, the model can more easily learn the mapping of the task, thus producing more expected results.

- **Role prompting**

Set a specific role for the model in the Prompt, such as "you are an experienced math teacher" or "you are a professional software engineer." This can to some extent guide the model to generate answers more in line with the style of a specific domain. In practice, this method is often used in technical Q&A, writing assistance, and code generation.

- **Constrain the output format**

If the task requires structured results, you can explicitly specify the output format in the Prompt, such as `JSON`, `table`, `Markdown`, etc. Clear format constraints can reduce the uncertainty of the model's output, and also make it easier for subsequent programs to parse and process.

&emsp;&emsp;Although designing Prompts with these techniques is a relatively direct way to help users guide the LLM to produce more expected results based on specific needs, this approach still has certain limitations:

- On one hand, writing high-quality Prompts often depends on the user's understanding of the task itself and the model's capabilities, so to some extent, the user needs to have relevant prior knowledge. For example, when the user can provide the problem's analytical ideas, key steps, or examples in the Prompt, the model usually finds it easier to generate more accurate and complete answers on that basis;
- On the other hand, in the absence of clear task descriptions or effective prompt information, although the model can still try to give an answer, the stability and quality of the generated results are often affected to some extent.

**Therefore, while Prompt design improves the LLM's problem-solving performance, it also puts forward prior requirements on the user's prompt construction to some extent.**

---

# External Tool Search Augmentation

<div align="center">
<img width="605" height="620" alt="12" src="https://github.com/user-attachments/assets/75f926e5-58be-4a0f-ac38-c5ab80f88eec" />
   <p>Figure 15: External search tools + LLM</p>
</div>

&emsp;&emsp;Typically, after an LLM finishes training, its model parameters no longer change, which means the knowledge the model learned during pre-training is to some extent static. Since pre-training data often has a time lag, the model may not be able to directly access the latest information. Therefore, in practice, external tools are often needed to supplement real-time data. For example, when a user asks "what's the weather like today?", the LLM itself doesn't have the ability to obtain real-time weather information, but needs to call an external search or API (such as a weather service or search engine) to get the latest data, and then combine its own language understanding and generation capabilities to give an answer.

&emsp;&emsp;Based on this idea, in recent years a class of systems combining "model reasoning ability + external tool invocation" has emerged. For example, *Gemini Deep Research* and *OpenAI Deep Research* and other projects usually go through multiple rounds of search, information filtering, and comprehensive analysis, leveraging the LLM's reasoning ability to integrate the retrieved information, thereby generating more complete and reliable research-style answers.

&emsp;&emsp;In addition, researchers have proposed technical frameworks such as Retrieval-Augmented Generation (RAG), which retrieve relevant documents from external knowledge bases before the model generates an answer and feed the retrieved information as context into the model, thereby enhancing its knowledge acquisition ability without changing the model parameters. This method can effectively improve the model's accuracy in specific-domain tasks and reduce the problems caused by outdated model knowledge or hallucination.

```text
In a sense, such methods can be understood as expanding the capability boundaries of large language models through "external memory" and "tool invocation," so that they no longer rely solely on the static knowledge learned during training.
```

&emsp;&emsp;Denny Zhou also mentioned a view at the end of his talk: for the process of LLM problem-solving, **external tool retrieval + reasoning > reasoning alone**.

---

# Summary

&emsp;&emsp;Synthesizing the above analysis, the source of large language model reasoning ability can be understood from multiple perspectives. In the pre-training stage, by performing language modeling on massive corpora, the model learns rich statistical regularities and abstract representations; this process is to some extent similar to the compression and encoding of language and world knowledge, thus providing a foundation for subsequent reasoning ability. Next, in the post-training stage (such as SFT, reinforcement learning), the model's behavior is further aligned and optimized, which not only makes it more stable in calling the latent abilities acquired in pre-training, but also lets the model learn new reasoning strategies (such as self-reflection and correction), thereby substantially enhancing reasoning performance.

And in actual use, methods such as CoT, Prompt guidance, and multi-path sampling **break complex problems down into intermediate reasoning steps, increasing the amount of computation during reasoning**, thereby enabling the model to solve complex problems that exceed the capability of a single forward pass. In addition, through retrieval augmentation or tool invocation, LLMs can obtain external real-time information, **making up for the insufficient coverage and timeliness of parametric knowledge, providing more accurate preconditions for the reasoning process**, thereby indirectly improving the solution of complex tasks.

&emsp;&emsp;From existing research, we can see that we need to improve the LLM's foundational ability from the pre-training stage, and the improvement of LLM ability largely follows the [Scaling Law](https://github.com/datawhalechina/diy-llm/blob/main/docs/en/chapter9/chapter9_Scaling_Laws.md): as model parameter scale, training data scale, and compute are expanded in a reasonable proportion, the model's test loss usually shows a predictable power-law decline, leading to overall performance improvements. However, this approach also faces an important challenge in practice, namely that the growth rate of high-quality human text data is limited, and model training is gradually approaching the limit of available data scale.

&emsp;&emsp;To address this problem, some studies have proposed using model-generated synthetic data at multiple stages such as pre-training and post-training to further expand the training distribution. In "continuously evolving AI" related research, the model can generate new training samples and iteratively optimize them combined with screening, evaluation, or self-improvement mechanisms, thereby alleviating the dependence on manually labeled data to a certain extent. This approach is **superficially similar to the self-play training mechanism of AlphaGo Zero**, but it is important to note the key difference: AlphaGo Zero relies on the perfect verification signal provided by the rules of Go, while most open-ended tasks faced by LLMs lack reliable automatic verifiers, so the effect of self-improvement is severely constrained by the quality of the verification signal. In addition, this approach also faces the risk of model collapse—repeated training on self-generated data may lead to distribution narrowing and quality degradation, so external verification signals and data-quality screening mechanisms usually need to be introduced to guarantee it. Overall, the synthetic-data approach is more about expanding the training distribution and improving the efficiency of capability utilization, perhaps enabling LLMs to break through the upper limit of all human knowledge.

> If we want to borrow from the successful experience of AlphaGo Zero, the key is to construct a reliable verification mechanism, not simply to "design a reward function." Specifically:
>
>- For verifiable problems (math, code), we can use existing objective verifiers to construct a self-play-style training loop, which has been proven effective (e.g., DeepSeek-R1);
>- For open-ended problems, there is no perfect reward function; currently we can only rely on imperfect alternative signals (human preferences, model mutual evaluation, etc.), and must guard against reward hacking and model collapse;

&emsp;&emsp;One of the author's personal ideas is that a large proportion of model-generated synthetic data can be introduced in the post-training stage to expand the training-sample space. In this process, synthetic data can be screened through manual review or high-quality model evaluation at the early stage to ensure the training data has good quality and rationality, thus avoiding the negative impact of low-quality samples on model capabilities. In subsequent iterations, the distribution of the generated data can be monitored through statistical indicators, e.g., using distribution-difference measures such as $D_{KL}$ to measure the offset between the synthetic data and the original data distribution, thereby preventing the generated data from gradually deviating from the real data distribution over multiple iterations. Through this "manual screening + distribution constraint" approach, it may be possible to alleviate the dependence on manually labeled data to a certain extent, while keeping the training data quality stable, thus reducing the probability of "model collapse."

>Here is a question that has puzzled the author; you can think about it together at the end 🤔🤔?
>
>&emsp;&emsp;During pre-training, LLMs have already formed their own unique way of compressing knowledge and an implicit reasoning paradigm. This paradigm may not be consistent with human thinking habits, but it may be natively optimized for the model's own architecture. Based on this insight, an idea worth exploring is: when constructing synthetic reasoning data, do we have to forcibly align with human-style chains of thought, or can we allow the model to reason in its "native" way and then transform the final output into a human-readable form during the alignment stage? The essential logic of this idea is—instead of forcing the model to imitate human linear thinking at every step of the reasoning process, it is better to first unleash its internal reasoning potential and then add a "translation alignment" layer at the output end to bridge the gap between human and machine expression. After all, forcibly constraining the model's high-dimensional reasoning process within the framework of human natural language, from an information-theoretic perspective, the essence of this process is to bottleneck a high-dimensional process with a low-dimensional representation.

---

# References

- [Denny Zhou from Google DeepMind on LLM reasoning research](https://dennyzhou.github.io/LLM-Reasoning-Stanford-CS-25.pdf)
- [DeepSeek-R1 training experience summary](https://arxiv.org/abs/2501.12948)
- [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/pdf/2504.13837)
- [On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models](https://arxiv.org/pdf/2512.07783)
- [Multi-path CoT decoding method](https://arxiv.org/pdf/2402.10200)
- [Hinton et al.'s family tree experiment proving that neural networks can internally abstract representations of input](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)
- [Exploration of feed-forward layer networks in Transformers](https://aclanthology.org/2021.emnlp-main.446/)
- [Extracting books from production language models](https://www.themoonlight.io/zh/review/extracting-books-from-production-language-models)
- [Analogical reasoning proposed by the Google team](https://arxiv.org/pdf/2310.01714)
- [Mind Lab's research on post-training RL fine-tuning of Kimi-K2](https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus)
- [diy-llm Chapter 14: Reinforcement Learning with Verifiable Rewards](https://datawhalechina.github.io/diy-llm/#/./chapter14/chapter14_%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%E7%9A%84%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)
- [Too much of a good thing: Understanding chain-of-thought length in large language models](https://arxiv.org/pdf/2502.07266)
- [Google team proposes the DTR metric!](https://arxiv.org/pdf/2602.13517)
- [OpenAI Prompt Engineering Guide](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)
- [Scaling law](https://github.com/datawhalechina/diy-llm/blob/main/docs/en/chapter9/chapter9_Scaling_Laws.md)
- [Stanford continuously evolving AI](https://zitongyang.github.io/slides/ZitongYang_defense_slides.pdf)
- [Neural Cellular Automata (NCA) generating non-linguistic synthetic data for pre-training language models](https://arxiv.org/pdf/2603.10055)
- [Sequence predictors can all serve as compressors](https://arxiv.org/pdf/2309.10668)
- [On Distinguishing Capability Elicitation from Capability Creation in Post-Training: A Free-Energy Perspective](https://arxiv.org/abs/2605.08368)

