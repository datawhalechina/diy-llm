# Chapter 5: Attention Improvements and Mixture of Experts

Since the 2017 proposal of "Attention Is All You Need," the Transformer has gradually become the core mechanism of modern LLMs. However, as parameter scale and context windows continue to expand, the original Transformer architecture faces growing bottlenecks in computational complexity, memory overhead, and model capacity. Recent research has mainly focused on two directions:

- **Extending context processing capability**: Efficiently handling longer sequences with limited resources. Representative works include: [FlashAttention optimizing attention kernels for GPU utilization, PagedAttention improving KV Cache memory management](ch6), MLA compressing Key-Value representations, and Hybrid Attention balancing computational efficiency with modeling capability.
- **Expanding model capacity**: Improving expressive power with roughly constant computation. MoE achieves this by dynamically routing tokens to activate only a small number of experts, dramatically expanding parameter scale at nearly constant FLOPs — now a key architectural choice for mainstream LLMs.

Current scenarios like Agents, long-context reasoning, and complex Tool Use demand both long-context modeling and efficient inference simultaneously, driving continuous evolution of the two directions above.

This chapter focuses on improvements at the Attention mechanism level, i.e., Attention Alternatives. We will start from classic attention, systematically introduce various attention variants' design philosophy and core principles, and compare them across dimensions of computational efficiency, memory overhead, and applicable scenarios.

## Learning Objectives

1. At the LLM architecture level, which specific techniques (attention mechanism variants, position encoding improvements, KV Cache optimization, etc.) can be used to extend context window length?
2. Explain MoE's basic concepts and operational mechanisms including gating networks, sparse activation, and other key aspects; describe which bottlenecks in large model training and inference (e.g., computational cost, model scalability) MoE primarily addresses.
3. Why can MoE architecture effectively improve model performance in practice? What are the fundamental reasons behind its efficiency?

*Let's continue with these questions in mind!*

## 5.1 Attention Improvement Methods

<div align="center">
<img width="470" height="447" alt="image" src="https://github.com/user-attachments/assets/58bd5958-ba12-4522-bc73-6c14911fae30" />
   <p>Relationship between Attention processing time and context length</p>
</div>

As context length increases, standard attention faces multiple bottlenecks:
- Attention score matrix computation and storage grows quadratically with sequence length (complexity $O(n^2 d)$); KV Cache accumulates linearly with context, both exacerbating memory pressure and causing computation speed to drop sharply.
- Attention score dispersion over long distances and position encoding extrapolation drift cause the model's long-text modeling ability to gradually decay — this phenomenon is particularly pronounced in deeper models.

### 5.1.1 Hybrid Attention

We first analyze the root cause of quadratic growth from standard attention's computational principles. Assuming hidden dimension $d$ and sequence length $n$, for one Transformer layer the computation flow is $\text{Softmax}(QK^\top/\sqrt{d})V$ → MLP, with time complexity:

$$O(n^2 d + n^2 + n^2 d) \approx O(n^2 d)$$

Standard attention grows quadratically with context length. Linear attention mechanisms (e.g., Mamba, RNN) compress history into a fixed-size hidden state:

$$S_t = S_{t-1} + k_t v_t^T, \quad y_t = W_t S_t$$

Inference complexity: $O(n d)$.

Comparing inference efficiency: standard attention accumulates KV Cache linearly with context (each new token adds one KV pair) — heavy burden for very long sequences. Linear attention's **hidden state size is fixed**, inference memory is **constant-level**, very friendly for long sequences.

Why do most LLMs still primarily use standard attention?

- **Training efficiency**: Standard attention processes entire sequences in parallel via matrix operations with extremely high hardware utilization. Linear attention, while parallelizable, still lags in throughput on current hardware.
- **Expressive power**: Standard attention enables precise global interaction between any two tokens. Linear attention compresses history into fixed states, inevitably losing fine-grained context — weaker on complex reasoning and long-distance fact retrieval.

Hybrid Attention combines both: using standard attention in some layers to retain global interaction, linear attention in others to reduce overall overhead.

**Nemotron 3 Super**: Alternating Mamba + standard attention. Mamba handles efficient sequence modeling at linear complexity; standard attention compensates for global interaction shortfalls.

<div align="center">
<img width="527" height="767" alt="image" src="https://github.com/user-attachments/assets/d6d19c2e-ae13-4494-8a58-69ae821c384b" />
   <p>Nemotron 3 Super (120B-A12B)</p>
</div>

**2. KDA**

Unlike Nemotron's approach of directly combining existing modules, Kimi's team proposed KDA, introducing $\text{Diag}(\alpha_t)$ into the linear attention recurrence for per-dimension decay control over the hidden state, enabling finer-grained management of historical information retention and forgetting while enhancing sensitivity to sequence position relationships.

<div align="center">
<img width="855" height="372" alt="KDA Architecture" src="https://github.com/user-attachments/assets/f33e459d-449b-487b-8bcc-192c9003d0b0" />
   <p>KDA</p>
</div>

The corresponding principle is expressed as:

$$S_t = (I - \beta k_t k_t^T) \text{Diag}(\alpha_t) S_{t-1} + \beta k_t v_t^T$$

<div align="center">
<img width="855" height="372" alt="KDA Formula Visualization" src="https://github.com/user-attachments/assets/6142dada-4d8f-48c8-bd2c-363197704722" />
   <p>KDA Formula Visualization</p>
</div>

KDA is a linear attention mechanism whose core idea builds on Gated DeltaNet by introducing a learnable forgetting mechanism per hidden dimension ($\text{Diag}(\alpha_t)$). In GDN, all hidden dimensions share the same learnable forgetting coefficient, so different dimensions decay historical information at the same rate. KDA extends this coefficient to a vector — each hidden dimension learns an independent forgetting rate — enabling different dimensions to automatically decide how much history to retain or forget based on training data, significantly improving memory representation flexibility.

The Kimi team believes this design is similar in spirit to RoPE. RoPE's advantage lies not just in introducing rotary position encoding, but more importantly in different hidden dimensions corresponding to different rotation frequencies, naturally representing position relationships at different time scales.

Additionally, in Kimi Linear's overall architecture design, extensive experiments found that under their model configuration, a **~3:1 ratio of linear-to-standard attention layers** achieves the best balance between model performance and computational efficiency. However, this ratio may be influenced by model scale, network structure, training strategy, and hardware platform — it doesn't mean there's a universal optimal ratio for all models. Under different architectures or training conditions, this ratio still needs to be adjusted based on specific experiments.

Of course, models adopting Hybrid Attention are not limited to those mentioned above — many other models are also exploring similar architectural approaches, along with other Attention improvement methods.

### 5.1.2 DeepSeek Sparse Attention (DSA)

Unlike Hybrid Attention which replaces entire layers' attention mechanisms to reduce overall complexity, DeepSeek's team proposed a different approach: introducing a lightweight indexer before standard attention to predict the most relevant Key tokens for each Query, only performing full attention on these candidate tokens, reducing computation from global $O(n^2)$ to approximately $O(n \cdot k)$ ($k \ll n$).

<div align="center">
<img width="1185" height="726" alt="image" src="https://github.com/user-attachments/assets/e4afd350-ef53-440c-a80c-d9b0eaa5a9bb" />
   <p>DSA</p>
</div>

Specifically, for input Query and Key, the lightweight indexer first computes relevance scores for each Key token relative to the Query:

$$S = f(Q, K)$$

Where $f(\cdot)$ is a lightweight indexing function with far lower cost than standard attention. Then select the top-$k$ most relevant candidate tokens:

$$\mathcal{C} = \text{TopK}(S, k)$$

Finally, perform standard Scaled Dot-Product Attention only on the candidate set $\mathcal{C}$:

$$\text{Attention}(Q, K_{\mathcal{C}}, V_{\mathcal{C}}) = \text{Softmax}\left(\frac{Q K_{\mathcal{C}}^T}{\sqrt{d}}\right) V_{\mathcal{C}}$$

Since full attention only occurs on a small number of candidate tokens, computational complexity drops significantly while preserving expressive power by selecting the most relevant tokens.

**Training strategy**: DSA uses two-stage training. ① Stage 1: pre-train with standard attention on shorter context, letting the model fully learn basic language modeling. Stage 2: introduce the lightweight indexer on top of Stage 1 parameters and switch to long context, letting the model gradually adapt to sparse attention computation while learning effective candidate token selection strategies.

② To mitigate the impact of randomly initialized indexers on model performance, at the start of Stage 2, typically freeze main model parameters and only optimize the indexer, *using standard attention distributions as teacher signals*, aligning sparse and dense attention distributions via **KL divergence loss**:

$$\mathcal{L} = D_{KL}\left(P_{\text{Dense}} \parallel P_{\text{Sparse}}\right)$$

Where $P_{\text{Dense}}$ represents standard attention distribution and $P_{\text{Sparse}}$ represents sparse attention distribution. After initial indexer alignment, unfreeze parameters and jointly optimize indexer and main model, letting the model maintain original language modeling while further improving sparse attention's representation and long-context modeling capabilities.

The reason for staged rather than direct joint training: language modeling and sparse index learning are two distinct optimization objectives. If both are optimized from the start, a randomly initialized indexer may fail to effectively filter candidate tokens, potentially disrupting attention computation and impacting basic language learning. Learning language representations first, then gradually introducing and optimizing the sparse indexer, reduces training difficulty, improves stability, and yields better final performance.

Beyond Attention optimization, researchers have also begun focusing on improvements to another core Transformer component — the FFN. Compared to Attention which models inter-token interaction relationships, the FFN primarily handles feature transformation and knowledge storage, with its parameter count typically occupying a large proportion of the Transformer. In recent years, more and more LLMs have adopted the MoE architecture, replacing dense MLPs with sparsely activated expert networks, significantly expanding model parameter scale while keeping computational cost roughly constant. Next, we will analyze MoE's basic principles, core components, and its application in various LLMs.

## 5.2 MoE Principles

MoE is a sparse activation architecture whose basic idea is replacing the Transformer's single FFN with multiple parallel expert networks, introducing a router to determine which experts should process each input token. During one forward pass, each token typically activates only a small number of experts (e.g., Top-1 or Top-2), with the rest remaining inactive. Thus, while the total model has enormous parameters, only a fraction is activated per computation — achieving large model capacity with relatively low computational cost.

> **Note**: Most research shows MoE's advantages are most pronounced at large parameter scales with sufficient data and compute. In small-scale or resource-constrained settings, MoE may underperform equivalent dense models; specific results also depend on task type, data volume, and implementation details.

**Intuitive analogy**: An MoE model is like a large library (expert collection). When a reader (input data) visits, they don't need to search all shelves — a librarian (router) guides them to relevant sections (sparse activation). The reader can still access all the library's knowledge (enormous parameter capacity) while the search is fast and efficient.

### 5.2.1 Routing Mechanisms and Load Balancing

In MoE models, the routing mechanism (also called gating mechanism) is responsible for selecting a small number of experts from the full set for each forward pass. The current mainstream routing approach is **Top-K** routing based on learnable gating scores, out of which two execution strategies derive: **Token Choice (TC)** and **Expert Choice (EC)**. Both rely on learnable gating mechanisms and typically pair with load balancing strategies.

> Early attempts used reinforcement learning to optimize discrete routing (treating routing as a policy learning problem), but due to gradient variance, training stability, and computational cost issues, this direction is uncommon in large-scale MoE.

Assuming $N$ experts, input $x$, gating function $G(\cdot)$ for determining expert weights, $E_i(\cdot)$ for expert $i$'s output, the core TC/EC formula:

$$y = \sum_{i \in \mathcal{T}} G_i(x) E_i(x)$$

**The key is the set $\mathcal{T}$**: achieving sparsification through **Top-k** selection. Both TC and EC share the same $G(x)$ computation process with two steps:

- **Scoring**: Compute routing scores $h(x) = x \cdot W_g$.
- **Sparsification**: Keep only the top $k$ experts, apply softmax normalization. The unselected $N-k$ expert weights are forced to zero, meaning they *completely do not participate* in this forward pass, ensuring efficient FLOPs despite enormous model parameters.

> $W_g$ is the router's learnable linear projection layer, mapping each token's features to a score vector (logits) matching the number of experts, indicating the token's match with each expert. The model applies top-k strategy to these scores: in TC mode, each token selects the most suitable experts; in EC mode, each expert actively selects the most suitable tokens.

*📖Tip: The key to whether learnable routing works — it's not about the specific routing form (TC/EC/etc.), **but whether the input representations already possess sufficient semantic structure, making samples separable in representation space, thereby supporting stable and meaningful expert selection**.*

<div align="center">
<img width="1000" height="520" alt="Token Choice Mode" src="https://github.com/user-attachments/assets/648d1892-b01e-4d40-9c2b-50478d2eeccf" />
   <p>Token Choice Mode</p>
</div>

- In **TC mode**, $W_g$ can be understood as an "expert specialty archive." It maps a token's hidden features to the expert set's capability space, telling the token what semantics different experts are good at. Each token, based on its match with each expert's "specialty archive," actively selects the Top-K experts most suitable for processing itself.

**Simple MoE Top-k Token Choice Implementation Steps**

Step 1: Define the Expert Network
```python
class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            # Dimension expansion
            nn.Linear(dim, dim * 4),
            # Non-linear activation to improve expressiveness
            nn.ReLU(),
            # Restore to original dimension
            nn.Linear(dim * 4, dim)  
        )

    def forward(self, x):
        return self.ffn(x)  # Forward propagation
```
Each expert network is composed of `Linear → ReLU → Linear`, used to process routed tokens in that expert's unique feature subspace, thereby providing distinguishable semantic transformations that make the Top-K routed combined output more expert-specific.

Step 2: Define the TC MoE Network
```python
class TC_MoE(nn.Module):
    def __init__(self, dim, num_experts, k):
        super().__init__()
        # Set number of experts
        self.num_experts = num_experts
        # Set number of experts each token selects
        self.k = k
        # Router: maps input to expert feature space
        self.router = nn.Linear(dim, num_experts)
        # Create list of expert modules (each expert is independent)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
    def forward(self, x, tokens=None, verbose=False):
        # Get batch size and feature dimension
        B, D = x.shape
        # Compute each expert's score for each token, use softmax for probability distribution
        gate_scores = F.softmax(self.router(x), dim=-1)  # gate_scores: [B, E]

        # Token selects the top-k highest-scoring experts and their scores
        # topk_scores: [B, k] (selected expert probability values)
        # topk_idx:    [B, k] (selected expert indices)
        topk_scores, topk_idx = gate_scores.topk(self.k, dim=-1)
 
        # Initialize output tensor with same shape as input
        out = torch.zeros_like(x)

        # Process each token's corresponding top-k position separately (same token may be processed by different experts)
        for i in range(self.k):
            # B represents total number of processed tokens
            # expert_ids represents each token's i-th Top-K selected expert index, shape: [B]
            expert_ids = topk_idx[:, i]
            # expert_weight represents each token's weight for the i-th Top-K selected expert, shape: [B]
            expert_weight = topk_scores[:, i]

            # Used to accumulate outputs of all experts at the current i-th selection position
            expert_output = torch.zeros_like(x)

            # Iterate over all experts, let the corresponding expert process tokens assigned to it
            # e_id represents the expert index corresponding to the Top-K expert processing
            for e_id, expert in enumerate(self.experts):
                # Create mask: 1 when token's i-th selected expert index equals current expert e_id, else 0
                # mask shape: [B, 1], used to zero out tokens not belonging to this expert before expert computation
                mask = (expert_ids == e_id).float().unsqueeze(1)

                # mask.sum() represents the number of tokens belonging to this expert; if 0, this expert has no task this round
                if mask.sum() == 0:
                    continue

                # Only feed tokens belonging to this expert into this expert's feed-forward network
                # Note: using x * mask preserves tensor shape consistency and retains the backpropagation path
                expert_output += expert(x * mask)

            # Weight the i-th selected position's expert output by corresponding weight and accumulate into final out
            # expert_weight.unsqueeze(1) becomes [B, 1] for broadcasting multiply to [B, D]
            out += expert_output * expert_weight.unsqueeze(1)

        # out: weighted-aggregated vector representation of each token over Top-K experts
        return out
```
**The above demonstrates the key components of Top-K TC MoE. Runnable code at [Top-K TC](https://github.com/datawhalechina/diy-llm/blob/main/docs/en/chapter5/Top-K%20TC.py).**

TC_MoE(dim=32, num_experts=10, k=2), input text:
>"MoE是很强大的机制！", "专家混合模型非常高效。"

Output:
>Byte-level tokenization yields 33 tokens. Expert load statistics from expert 0 to 9 total processed tokens: [13, 13, 16, 14, 9, 6, 20, 19, 18, 4]

<div align="center">
<img width="1000" height="773" alt="Expert Selection Mode" src="https://github.com/user-attachments/assets/d665c6bd-88be-4b35-9199-71dbfe74b9ba" />
   <p>Expert Selection Mode</p>
</div>  

- In **EC mode**, $W_g$ can be understood as a "semantic navigator," mapping a token's hidden features to each expert's semantic space and providing this navigation signal to all experts. Each expert, based on this "navigation information," actively selects the Top-K tokens most aligned with its capability range.

**Simple MoE Top-k Expert Choice Implementation Steps**

Step 1: Define the Expert Network
```python
class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim)
        )
    def forward(self, x):
        return self.ffn(x)
```

Step 2: Define the EC MoE Network
```python
class EC_MoE(nn.Module):
    def __init__(self, dim, num_experts, k):
        super().__init__()
        # Total number of experts
        self.num_experts = num_experts
        # Maximum number of tokens each expert can select  
        self.k = k
        # Router for outputting E expert scores per token                
        self.router = nn.Linear(dim, num_experts)  
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])  
    def forward(self, x, tokens=None, verbose=False):
        # Get total number of input tokens B_total and dimension D
        # B_total represents the total number of all tokens (batch × token count)
        B_total, D = x.shape

        # Router computes matching scores for each token and each expert, output dimension: [B_total, num_experts]
        # softmax ensures all expert scores sum to 1
        gate_scores = F.softmax(self.router(x), dim=-1)

        # EC mode: "expert picks tokens"
        # Transpose to [num_experts, B_total]
        # scores_T[e][t] = score of expert e for token t
        scores_T = gate_scores.transpose(0, 1)

        # Each expert selects top-k most relevant tokens from all tokens
        # topk_idx: indices of Top-K tokens selected by each expert
        # topk_scores: corresponding routing scores
        # Dimensions: [num_experts, k]
        topk_scores, topk_idx = scores_T.topk(min(self.k, B_total), dim=-1)

        # dispatch_weights size: [B_total, num_experts]
        # Initialize dispatch_weights
        dispatch_weights = x.new_zeros((B_total, self.num_experts))

        # For each expert e, write the top-k token scores into corresponding positions
        for e in range(self.num_experts):
            # topk_idx[e] is a list of Top-K token indices
            # topk_scores[e] is the scores of Top-K tokens
            # Fill dispatch_weights: normalize each expert's scores over tokens
            for t_idx, s in zip(topk_idx[e].tolist(), topk_scores[e].tolist()):
                dispatch_weights[t_idx, e] = s

        # Initialize output out, same size as input x
        out = torch.zeros_like(x)

        # Forward computation for each expert
        for e_id, expert in enumerate(self.experts):

            # mask: whether this expert selected this token
            # mask[t] == 1 → token t was selected by this expert
            # Dimension: [B_total, 1]
            mask = (dispatch_weights[:, e_id] > 0).float().unsqueeze(1)

            # If expert selected no tokens, skip computation
            if mask.sum() == 0:
                continue

            # Ensure each expert only processes its selected Top-K tokens
            # mask zeros out tokens not belonging to this expert; different experts may process the same token
            expert_out = expert(x * mask)

            # Add expert output back to final output weighted by its dispatch weight
            # dispatch_weights[:, e_id] is each Top-K token's weight for this expert
            out += expert_out * dispatch_weights[:, e_id].unsqueeze(1)
        return out
```
**The above demonstrates the key components of Top-K EC MoE. Runnable code at [Top-K EC](https://github.com/datawhalechina/diy-llm/blob/main/docs/en/chapter5/Top-K%20EC.py).**

EC_MoE(dim=32, num_experts=10, k=2), input text:
>"MoE是很强大的机制！", "专家混合模型非常高效。"

Output:
>Byte-level tokenization yields 33 tokens. Expert load statistics: each expert processes exactly 2 tokens, but some tokens were never processed at all, e.g.: ['混', '合', '模', '型'...].

Thus, in each forward pass, the model only computes on the expert subset $\mathcal{T}$ selected by the Top-K routing mechanism, achieving sparse inference. The routing mechanism's core role can be summarized as: **selecting the most suitable few experts for each input + weighted fusion of these activated experts' outputs by routing weights**.

The routing mechanism's selection basis is the input hidden state. Specifically, after going through embedding, position encoding, and preprocessing, input tokens generate hidden states, which then serve as input to the router (typically a linear layer or small MLP) to compute expert scores. The row-column dimension distinction determines the granularity of sparsification:

- **TC mode**: For each token (each row of the matrix), select Top-K experts on the expert dimension (column dimension).
- **EC mode**: For each expert (each column of the matrix), select Top-K tokens on the token dimension (row dimension).

**TC vs EC**:

Running both code examples [Top-K EC](https://github.com/datawhalechina/diy-llm/blob/main/docs/en/chapter5/Top-K%20EC.py) and [Top-K TC](https://github.com/datawhalechina/diy-llm/blob/main/docs/en/chapter5/Top-K%20TC.py) allows intuitive comparison of the two routing strategies.

1. **In TC mode**, each token actively selects its most suitable Top-K experts, like "student finds advisor." Advantage: every token gets at least attempted assignment, so semantic completeness is high, reducing information loss risk. Disadvantage: expert **load imbalance** — a few "popular" experts handle the vast majority of tokens, receive full training and significantly outperform others, while many "cold" experts stay idle long-term with stagnating capability. **This gap causes the model to become "skewed" — performing well in high-frequency domains but poorly in low-frequency ones.**
2. **In EC mode**, each expert actively selects its most desired Top-K tokens from all tokens, like "advisor picks students." This mechanism naturally constrains each expert's processing volume (enrollment quota), significantly alleviating or eliminating expert load imbalance, benefiting balanced expert capability improvement. But the cost: some tokens may be completely unselected by any expert (dropped), causing semantic information loss or context segment skipping, **thereby increasing the risk of misinterpretation or errors during model understanding and reasoning, ultimately degrading LLM final performance.**

**Conclusion**: TC excels at semantic completeness but is prone to expert capability **polarization**; EC excels at load balancing but bears potential semantic loss risk. The two modes represent the classic trade-off between information completeness and load balancing in sparse expert systems. Recent research also explores approaches to resolve this trade-off.

**Load Balancing Strategies:**

| Strategy | Core Idea | Explanation |
|----------|-----------|-------------|
| [Auxiliary load balancing loss](https://yangyutu.github.io/llm_book.github.io/docs/chapter_LLM_arch/LLM_moe_sparse_architectures.html) | Add a regularization term to training loss encouraging uniform token count and routing probability distribution across experts, preventing a few experts from handling the vast majority of tokens | The earliest and most classic method, adopted in `Switch Transformer` and many subsequent MoE implementations. |
| [Capacity control + expert capacity + overflow mechanism](https://mljourney.com/mixture-of-experts-moe-routing-algorithms-for-sparse-llms) | Set a capacity limit per expert; once exceeded, tokens go to fallback path (or dropout, backup expert); prevents single-expert overload and neglect of "cold" experts | Most MoE systems recommend controlling per-expert load via capacity factor + expert capacity and managing overflow conditions. |
| [Dynamic, loss-free load balancing](https://www.emergentmind.com/papers/2408.15664) | Avoid introducing extra training gradients; dynamically adjust routing scores via per-expert bias (based on historical load statistics), stabilizing routing distribution without aux-loss | Recent work "Loss-Free Balancing for MoE" proposes this approach, showing it is more stable than traditional aux-loss and doesn't disrupt the original model optimization objective. |
| [Improved router, similarity-preserving routing](https://arxiv.org/abs/2506.14038) | Design router so semantically similar tokens → similar expert assignment, with uniform distribution across experts; reduce redundant routing and expert load skew | Improves convergence speed and load balancing effectiveness. |
| [Improved expert structure, routing mechanism](https://arxiv.org/abs/2511.10971) | Change expert parameterization (e.g., using orthogonal bases, basis) or use more stable, interpretable routing scores rather than simple linear logits, improving routing stability and expert utilization | Latest work not only alleviates traditional routing instability and expert idleness, but naturally achieves more uniform expert load. |
| [Hybrid shared + routed expert pools](https://arxiv.org/pdf/2401.06066) | Set some experts as shared experts activated by all tokens; the rest as routed experts. Shared experts ensure coverage of all tokens even when routing is extremely imbalanced, reducing token dropout and semantic loss | In engineering practice, e.g. DeepSeekMoE uses this approach to compromise between maintaining semantic coverage + expert specialization training. |

In some related MoE research, complex intelligent routers (such as Top-K learnable routing) are not absolutely necessary; there also exist **Hash routing** and other non-learning methods:

- **Hash principle**: Through fixed hash functions mapping input tokens to experts, naturally providing good load balancing and low-overhead implementation.
- **Hash limitations**: Although hash routing is generally inferior to learnable routing in semantic flexibility and fine-grained expert specialization, it can still demonstrate considerable competitiveness in several benchmarks and engineering scenarios.

<div align="center">
<img width="800" height="480" alt="Hash Routing" src="https://github.com/user-attachments/assets/e5b160fb-1410-418d-aa48-f790095a5f01" />
   <p>Hash Routing</p>
</div>

Take [LSH](https://proceedings.neurips.cc/paper_files/paper/2024/file/61674667d642ae52f6bb281bea90ee29-Paper-Conference.pdf) as an example, using **fixed, non-trainable** hash functions. Each hash function projects the input token embedding $x \in \mathbb{R}^d$ onto a plane defined by a random vector $a_i \in \mathbb{R}^d$ and random bias $b_i$, then quantizes via bucket width $\epsilon$ (indirectly controlling per-bucket token capacity), thereby mapping $x$ to an integer hash bucket indexed by $i$ as $h_i(x)$.

$$
h_i(x) = \left\lfloor \frac{a_i^\top x + b_i}{\epsilon} \right\rfloor
$$

Here $D$ is the number of composite hash functions, i.e., random projection directions. This method does not optimize hash parameters via gradients, but routing results dynamically change as $x$ (Token Embedding) evolves during training. LSH **probabilistically** achieves load balancing, and due to its locality sensitivity, can preserve weak locality — i.e., similar tokens are more likely to fall into the same hash bucket. Thus, LSH qualifies as a "weak semantic" non-learnable routing.

*`Bucket width` $\epsilon$ corresponds to the quantization step size in the above formula, used to bucket continuous projection values; wider buckets gather more vectors into the same bucket.*

> **Note:** The formula with $\epsilon$ above characterizes **random projection + scalar quantization** type LSH; the example code below uses **random hyperplane sign hashing** (taking sign of projections then encoding as bit strings). Both belong to the LSH family but differ in implementation details — please do not attempt to map formula parameters to code line by line.

**Simple MoE Implementation Based on LSH Routing Mechanism:**
```python
import torch
import torch.nn as nn
# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
    def build_vocab(self, texts):
        chars = set("".join(texts))
        self.vocab = {c:i for i,c in enumerate(sorted(chars))}
        self.inv_vocab = {i:c for c,i in self.vocab.items()}
    def encode(self, text):
        return [self.vocab[c] for c in text]
    def decode(self, ids):
        return "".join([self.inv_vocab[i] for i in ids])

# Expert FFN
class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim)
        )
    def forward(self, x):
        return self.ffn(x)

# LSH Router
class LSHRouter(nn.Module):
    def __init__(self, d_model, num_experts, n_hashes=8):
        super().__init__()
        self.num_experts = num_experts
        self.n_hashes = n_hashes
        self.register_buffer(
            "random_vectors",
            torch.randn(n_hashes, d_model)
        )
    def forward(self, x):
        projections = x @ self.random_vectors.T
        signs = (projections > 0).long()
        hashes = signs @ (1 << torch.arange(self.n_hashes, device=x.device))
        expert_ids = hashes % self.num_experts
        return hashes, expert_ids

# LSH-MoE
class LSH_MoE_Text(nn.Module):
    def __init__(self, dim, num_experts, n_hashes=8, vocab_size=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)  # Embedding layer
        self.dim = dim
        self.num_experts = num_experts
        self.router = LSHRouter(dim, num_experts, n_hashes)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])

    def forward(self, token_lists, verbose=True):
        """
        token_lists: list of LongTensor, each tensor is one text's token IDs
        """
        lengths = [t.size(0) for t in token_lists]
        total_tokens = sum(lengths)
        x_flat = torch.cat(token_lists, dim=0)  # [total_tokens]
        x_flat = self.embedding(x_flat)         # [total_tokens, D]

        hashes, expert_ids = self.router(x_flat)
        out_flat = torch.zeros_like(x_flat)

        expert_load = torch.zeros(self.num_experts, dtype=torch.long, device=x_flat.device)
        for e_id, expert in enumerate(self.experts):
            mask = (expert_ids == e_id).float().unsqueeze(1)
            n_tokens = int(mask.sum().item())
            expert_load[e_id] = n_tokens
            if n_tokens > 0:
                out_flat += expert(x_flat * mask) * mask

        # Split back to original sentences
        outputs = []
        start = 0
        for l in lengths:
            outputs.append(out_flat[start:start+l])
            start += l
        if verbose:
            print("\n========== LSH-MoE Token Hash Mapping ==========")
            start = 0
            for idx, l in enumerate(lengths):
                for j in range(l):
                    token_idx = start + j
                    print(f"Sentence {idx}, Char {j}: Hash={hashes[token_idx].item()} -> Expert {expert_ids[token_idx].item()}")
                start += l
            print("\n========== LSH-MoE Expert Load Statistics ==========")
            for e in range(self.num_experts):
                print(f"Expert {e}: {expert_load[e].item()} tokens")
            print("------------------------------------------------\n")

        return outputs

# Test
if __name__ == "__main__":
    sentences = ["你好世界", "今天天气很好"]
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(sentences)
    token_lists = [torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in sentences]
    dim, num_experts = 16, 5    # Embedding dimension per token, number of experts
    moe_text = LSH_MoE_Text(dim=dim, num_experts=num_experts, vocab_size=len(tokenizer.vocab))
    outputs = moe_text(token_lists)
    for i, out in enumerate(outputs):
        print(f"Sentence {i} output shape: {out.shape}")

```
Input:
>dim, num_experts = 16, 5

Output:
>Per-sentence token hash mapping and LSH_MoE expert load statistics.

*Output results dynamically change as the embedding layer evolves.*

**MoE Routing Mechanism Comparison**

| Routing | Core Idea | Learnable? | Advantages | Disadvantages | Typical Use |
|----------|----------|------------|-----------|--------|-------------|
| **Top-K (TC/EC)** | Gating network computes token-expert scores, selects Top-K experts for computation | Yes | Semantically flexible, adapts to data distribution; can combine with load balancing loss, noisy gating techniques; good performance | Needs training; load skew risk at large scale; high communication overhead | DeepSeek-MoE, GPT-MoE, Qwen, Switch Transformer |
| **Hash routing** | Fixed hash functions map input to experts (e.g., LSH, random hash...) | No | Natural load balance; no router training needed; extremely efficient; low communication cost | Weak semantic expressiveness; cannot dynamically allocate experts based on task | Large-scale inference, lightweight MoE, some sparse training experiments |

Although MoE was proposed early, widespread LLM application began post-2021 (Switch Transformer, GLaM). Recent models (e.g., DeepSeek-R1) further demonstrate MoE's potential in high-performance inference tasks, but core challenges remain in training stability and optimization efficiency:

- **In shallow stages**, token representations are in rapid evolution, with input distributions showing significant non-stationarity (semantic structures not yet stably formed). This exposes routers to high-noise inputs, leading to high-variance expert assignment decisions and increased training convergence difficulty.
- **Common load imbalance** causes different experts to receive significantly uneven training samples, leaving some experts fully trained while others remain undertrained due to lack of activation — even resulting in "dead experts."

Notably, **sparse learnable routing (e.g., Top-K) introduces discrete decision processes, making gradient estimation high-variance**. Meanwhile, expert networks only receive gradient updates `when selected by routing`, while the **routing network continuously receives training signals from samples and `continuously updates`** — this **gradient asymmetry under conditional computation** causes different modules to depend on different data distributions during optimization, making optimization step coordination difficult and increasing training instability.

>**Learnable-routing MoE models actually optimize route-induced sub-objectives at different iteration steps**, making the overall optimization process exhibit `non-stationarity` and `path-dependence`, thus increasing training difficulty.

*📖Tip: In MoE architectures, gradient updates are conditional — their update paths are determined by routing decisions. When the routing network changes, the optimization objectives and gradient paths for backpropagation also change, manifesting as an optimization inconsistency problem introduced by conditional computation.*

### 5.2.2 MoE Variants

In MoE models, each expert is like a "teacher" responsible for processing a portion of the input (tokens). However, in practice, two common problems hinder experts from truly forming "specialized domains":
1. **Knowledge mixing**: Tokens assigned to an expert may be diverse — covering many different types of knowledge. Like a teacher asked to simultaneously teach math, history, and art — it's hard to teach every subject thoroughly in their own classroom.
2. **Knowledge duplication**: Tokens processed by different experts may have overlapping knowledge needs. Like several teachers all preparing the same materials, each doing redundant work, unable to highlight their respective "specialties," leading to a lack of clear "professional division of labor" among experts.

These two problems together may limit MoE models from realizing their theoretical maximum capability, making it difficult for experts to truly "each do their own job." Understanding these limitations can inspire us to design smarter routing strategies so each expert focuses on their own "domain," thereby improving overall model performance. Below we introduce two MoE variants proposed precisely to address knowledge clutter and duplication.

**1. DeepSpeed-MoE**'s contributions to reducing MoE training costs span full-stack design from model structure, training systems, to inference acceleration, making sparse expert models more advantageous than equivalent-quality dense models in training cost, deployment efficiency, and real-time performance, thus driving ultra-large-scale LLMs toward greater efficiency and practicality:

- **Parameter efficiency improvement:** DeepSpeed-MoE proposed PR-MoE and its distilled compressed version MoS. PR-MoE uses fixed MLP + "expert residual correction" to reduce parameters and communication, then uses "pyramid-shaped expert counts" to concentrate experts in deeper layers, achieving higher parameter efficiency. MoS further compresses the model through staged distillation, keeping MoE performant while significantly accelerating inference.
- **Distillation compression for inference acceleration:** MoS further compresses PR-MoE through "staged knowledge distillation," using a shallower sparse student model to replace the original for faster inference. Since directly reducing layers degrades model capability, and using teacher signals throughout training causes student underfitting, MoS uses a "two-stage" approach: early training uses distillation to stably learn teacher distributions; later training turns off distillation and only optimizes language model loss, letting the student develop autonomous generalization. In practice, MoS can further shrink model size by 3.7× while being faster than equivalent-quality dense models.
- **System optimization upgrades:** System-level rewriting of MoE parallelism and communication makes MoE faster and more stable in real large-scale training and inference:
    - Since different layers have inconsistent expert counts, it uses flexible combinations of expert parallelism, expert slicing, data parallelism, and tensor slicing, ensuring each layer gets the most suitable parallelization approach.
    - This adaptive parallelism lets MoE scale stably across hundreds of GPUs while avoiding load imbalance and memory waste.
    - In communication, DeepSpeed-MoE reduces All-to-All complexity from $O(p)$ to $O(p/L)$ through tensor slicing, and uses hierarchical All-to-All to reduce cross-node latency. The core optimization rewrites MoE's sparse rearrangement as explicit data layout transformation, reducing key kernel latency by over 6×, significantly improving inference speed.

>Hierarchical All-to-All means that in MoE, the originally one-shot, global, all-GPU token communication is split into level-by-level multi-layer communication according to hardware topology: first complete high-speed All-to-All within the same node/machine, then perform necessary data exchange across different nodes/machines, thereby significantly reducing cross-machine communication volume.

**2. Switch Transformer** is dedicated to dramatically expanding model parameter count without significantly increasing per-sample FLOPs. Its core strategy is replacing dense FFNs in standard Transformers with sparsely activated expert collections, letting different inputs dynamically activate different experts, thereby expanding parameter capacity without significantly increasing computational cost. Key techniques include auxiliary loss functions for load balancing, low precision for intermediate transmission, and medium precision for critical routing decisions:

- **Router computation**: The router computes logits for token representations $x$. Typically softmax on logits gives probability distribution $p_i(x)$ over experts.
- **Actual routing decision**: Uses Top-1 strategy — each token is assigned to the single highest-scoring expert for FFN execution. Softmax probabilities are mainly for statistics and auxiliary loss; actual forward computation only uses the selected expert (sparse activation). Compared to Top-k, Top-1 routing significantly simplifies implementation, reduces cross-device communication, and lowers computation from experts being simultaneously called, improving hardware and communication efficiency.
- **Over-capacity handling**: If an expert is assigned more tokens than its capacity, excess tokens **do not execute that expert's FFN** — they are "dropped" and only pass through residual connections to the next layer; **thus excess tokens generate no gradients for that expert**.
- **Router Z-loss**: To prevent routing logits from producing extreme values at low precision, a penalty term on logit magnitude is introduced, reducing softmax sensitivity to extreme inputs, thereby improving training numerical stability.
- **Smaller initialization**: Considering the difficulty of random router initialization, weight matrices are initialized by sampling from truncated normal distributions with mean 0. Appropriately reducing the initialization scale of certain linear layers and FFNs can lower early-training gradient variance, reduce early instability, and improve model capability.

<div align="center">
<img width="1200" height="600" alt="Switch Transformer" src="https://github.com/user-attachments/assets/33892936-0c5c-4743-8047-6e65d9d85401" />
   <p>Switch Transformer</p>
</div>

>Switch Transformer's sublayer order adopts the **Self-Attention → FFN/MoE** structure, which is key to achieving efficient training and deep semantic modeling:
>
>① **Self-Attention first (establishing global semantics)**:
>Compute similarities and dependencies among all tokens. Self-Attention essentially lets each token establish weighted connections with all other tokens in context, thereby generating contextualized representations containing rich contextual information. These representations not only preserve the word's own semantic features but also integrate relational information from the entire sentence or even longer context, enabling the model to more fully understand each token's role, function, and contextual position in the global semantic structure when processing it.
>
>② **FFN/MoE later (expert feature enhancement)**:
>The subsequent FFN or MoE layer, based on Self-Attention's contextualized features, performs independent nonlinear semantic enhancement on each token. For MoE, this means the router can leverage rich contextual information to more accurately assign tokens to the most functionally matched experts, thereby improving expert specialization and reducing early-stage router instability.
>
>In summary, Switch Transformer's pipeline structure of "first acquire global relations, then enhance individual features" maximizes semantic modeling efficiency in Transformers. If the order were reversed, letting FFN/MoE process raw embeddings lacking context first would not only weaken feature enhancement effects but also potentially disrupt the original geometric relationships among tokens through nonlinear transformations, possibly reducing Self-Attention similarity computation accuracy and ultimately preventing the model from correctly capturing dependencies, fundamentally reducing Transformer expressiveness and training efficiency.

>Switch Transformer emphasizes stability and simplified FLOPs; DeepSpeed-MoE emphasizes expert distribution and model distillation. Together they form two design philosophies for modern MoE:
>
>① Precision-communication tiered design, reducing training costs;
>
>② Dynamic constraints and structural adjustments, improving training stability and expert specialization.

### 5.2.3 MoE vs. Dense Models

MoE's advantage over traditional dense models is that it can **expand model parameter scale while keeping computation roughly constant**, thereby significantly improving model representational capacity and performance. Moreover, since MoE experts are **sparsely activated** — only a small number of experts participate in each computation — each expert (typically a feed-forward network) can exist as an independent module distributed across different devices. The router simply sends corresponding tokens to appropriate devices based on input; computation is then completed independently on the expert's device. This natural structural partitioning enables MoE to achieve efficient **expert-level parallelism**, making it an essential parallelization strategy for building ultra-large-scale models and a crucial foundation for modern LLMs to突破 capacity and performance bottlenecks in multi-machine, multi-GPU environments.

<div align="center">
   <img width="1000" height="560" alt="MoE vs Dense" src="https://github.com/user-attachments/assets/b9945e0c-9a88-4127-a267-2f1c0b62d132" />
   <p>Figure 5.7 MoE vs Dense Model</p>
</div>

From the figure above, we can readily see that under the experimental settings shown, MoE architectures tend to converge faster and perform better than dense models (specific conclusions depend on data scale, training recipe, and evaluation tasks).

In MoE research, two common practice paths exist:

- **Dense → Sparse upgrade (Upcycling)**: Converting a trained dense model to MoE to reuse prior training results and weights;
- **From-scratch MoE training**: Starting from random or specialized initialization to train MoE, letting experts and routers co-evolve from scratch.

Empirical results show these two paths diverge significantly under different settings. For example:

- OLMoE's experiments found that TC-routing MoE trained from scratch catches up to and subsequently surpasses upcycled models at around 500–600B tokens — equivalent to approximately 25% of the original dense model's training data compute budget to reach the catch-up point.
- In contrast, Komatsuzaki et al., in their upcycling work using EC routing, reported that from-scratch MoE requires approximately 120% of the original dense model's training volume to catch up to the upcycled model. The differences stem from variations in experimental paradigms and routing strategies.

Moreover, OLMoE's experiments mention that various parameters of the original dense model may impose certain constraints on the upcycled model, which is why OLMoE opted to train the MoE model from scratch when upgrading.

>OLMoE's experiments indicate that with limited data, converting a trained dense model to MoE introduces two types of structural factors that hinder early learning stability:
>
>① The original dense model's weights already encode strong general capabilities. After MoE conversion, these parameters must partially "forget" past representations and reshape into expert-specialized capabilities. **However, since gradient flow remains influenced by old representations, new learning signals are easily disrupted by historical distributions, creating a "forgetting and relearning difficulty" phenomenon**;
>
>② MoE routers typically start from random initialization, **presenting near-random or uniform allocation in early training, making it difficult to form clear expert division of labor in the first half of training. By the time training progresses to where routers gradually stabilize, learning rates have often already decayed**, leaving the router's "specialty" mapping of experts still potentially "fuzzy" — i.e., a "router learns too late" situation.

Additionally, successful upcycling cases exist in engineering practice. For example, Qwen1.5-MoE in the Qwen series successfully transformed an existing dense model into MoE, maintaining or improving performance while significantly enhancing computational parameter efficiency — with fewer active parameters, it matched the performance of larger dense models.

>**Why such large differences like 25% vs 120%?**
>1. Routing strategy differences (TC vs EC): TC and EC have fundamental differences in load balancing, expert differentiation speed, and early training dynamics, significantly affecting from-scratch convergence speed.
>2. Different model paradigms: Decoder and encoder architectures differ in training objectives and information flow; upcycling benefits vary with paradigm.

## 5.3 MoE Applications

MoE is not merely a **Transformer**-specific technique, but a general "conditional computation framework" that can be broadly embedded into various neural network architectures. Its core idea is letting different experts handle different types of data or subtasks, hence its extensive application beyond Transformers:
- In **CNNs** as dynamic convolutions to improve visual modeling diversity;
- In **speech recognition** to let different experts focus on different phonemes or noise conditions;
- In **recommendation systems** to solve multi-task ranking problems;
- In **reinforcement learning** decomposed into multi-strategy, multi-skill experts;
- In **multimodal models** to achieve cross-modal expert collaboration.

It is precisely due to MoE's structural independence and division-of-labor capability that it has become a crucial foundational module for scaling model parameters, improving representational capacity, and reducing computational cost in large models. Next, we introduce MoE's application in LLMs.

### 5.3.1 MoE and LLMs

In LLMs, MoE typically introduces a router and replaces or expands the single FFN module in Transformers into a sparse sub-network composed of multiple independent experts. Each token activates only a small number of experts during forward and backward propagation, enabling the model to dramatically increase parameter capacity and representational capability without significantly increasing per-computation cost.

### 5.3.2 Mini LLM + MoE Implementation

**Step 1: Build a Byte-level Tokenizer**
```python
class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 259
        self.bos = 256 # Beginning of sequence — tells the LLM that an independent text segment or input sample starts here.
        self.eos = 257 # End of sequence — tells the LLM that a text segment ends here.
        self.pad = 258 # Padding — during model training or inference, multiple texts of varying lengths typically need to be grouped into a batch.
        # <pad> is appended to shorter sequences to make all sequences in the batch equal length, enabling efficient matrix operations.
```

This vocabulary has a total size of 256+3, composed of two parts:
 1. Basic byte encodings: 256 in total, representing all possible single-byte values from 0 to 255 in computers. This ensures any text, regardless of language or encoding, can be losslessly encoded into a sequence of numeric Token IDs.
 2. Special function tokens: 3 in total, specifically used to provide text structural information, ensuring the model can correctly process and understand text segment boundaries and batch alignment for computation.

```python
def encode(self, text, add_bos=True, add_eos=True):
    # Encode input text using UTF-8 charset to obtain byte sequence b
    # Each byte value 0-255 corresponds to one Token ID
    b = text.encode('utf-8', errors='surrogatepass')
    ids = list(b)   # Convert UTF-8 byte sequence to Token ID list
    if add_bos:
        # Mark text beginning, add <bos> Token ID
        ids = [self.bos] + ids 
    if add_eos:
        # Mark text ending, add <eos> Token ID
        ids = ids + [self.eos] 
    return ids # Return the final processed Token ID sequence
def batch_encode(self, texts, pad_to=None):
    # pad_to specifies the target length for each Token ID sequence in the batch (force alignment)
    encs = [self.encode(t) for t in texts] 
    # If pad_to is unspecified, use the longest sequence length in the current batch; otherwise use the specified pad_to length
    maxlen = max(len(x) for x in encs) if pad_to is None else pad_to
    pad = self.pad

    # Pad all sequences to maxlen length by appending [pad] at the end of each sequence, force-aligning into a regular tensor
    arr = [x + [pad] * (maxlen - len(x)) for x in encs] 
    
    # Record the true lengths of original sequences — this information will be used for Attention to prevent the model from attending to [pad] tokens
    lengths = torch.LongTensor([len(x) for x in encs])  

    # Return the padded-aligned Token ID tensor for model input, and the true sequence length tensor for Attention
    return torch.LongTensor(arr), lengths
```
*📖Tip: A character is the abstract unit with minimal semantic function in human language (e.g., letter `A`, Chinese character `中`, symbol `+`), while a byte is the smallest addressable physical unit for computer storage and data transmission. Characters can be represented by one or more bytes — this is the core mechanism of character encoding.*

The batch_encode stage returns both aligned tensors and unaligned sequence lengths for these reasons:
- Irregular tensors cannot be directly fed into hardware (GPU, TPU) optimized for high-performance parallel computation. Alignment is a necessary preprocessing step for batching and exploiting hardware parallelism. While this padding solves the parallel computation problem, it also introduces computational redundancy (e.g., the [pad] tokens here).
- The original sequence length information tells the model where the trailing [pad] starts, so these can be masked out in the Attention mechanism, preventing computational resources and attention from being wasted on irrelevant data, ensuring the model focuses only on real input information.

**Step 2: Build the Self-Attention Layer**
        self.qkv = nn.Linear(d_model, d_model * 3) 
        self.out = nn.Linear(d_model, d_model)
    def forward(self, x, mask=None):
        B, T, D = x.shape # Input tensor dimensions
        # Linear projection of Q, K, V
        qkv = self.qkv(x)  # Project input [B, T, D], obtaining a fused tensor of shape [B, T, 3*D]
        q, k, v = qkv.chunk(3, dim=-1) # Split into Q, K, V along the last dimension, each [B, T, D]

        # Multi-head splitting via view(): Q, K, V reshape [B, T, D] -> [B, T, nhead, d_k]
        # Transpose: Q, K, V reshape [B, T, nhead, d_k] -> [B, nhead, T, d_k]
        q = q.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.d_k).transpose(1, 2)

        # Compute Q-K inner product similarity, shape [B, nhead, T, T]
        # Divide by √d_k for scale normalization to prevent overly large dot products causing gradient vanishing after softmax
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Attention Mask operation
        if mask is not None:
            # mask is typically [B, T]; generate mask ~mask.bool() where positions to mask are 1
            # Multiply by -1e9 to set masked positions to extremely small negative values
            attn_mask = (~(mask.bool().unsqueeze(1).unsqueeze(2))) * -1e9
            scores = scores + attn_mask  # Apply mask to scores

        # Softmax normalization: convert scores to attention weights; extremely negative positions → weights near 0 (masked out)
        attn = F.softmax(scores, dim=-1) 
        # Attention-weighted sum: weights attn multiplied by Value, producing weighted-sum output [B, nhead, T, d_k]
        out = torch.matmul(attn, v)

        # Transpose back to [B, T, nhead, d_k], then use contiguous().view() to concatenate all heads back to original D dimension [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)
```

**Step 3: Build the MoE Layer**

The simplified MoE layer parameters:
   - d_model: input/output dimension, i.e., Transformer layer hidden size.
   - d_ff: expert internal hidden dimension, the expanded dimension inside each expert FFN.
   - n_experts: number of experts, the count of parallel FFN modules in the MoE layer.
   - k: Top-K active experts, indicating each token will be routed to K experts for processing.
   - capacity_factor: per-expert capacity coefficient, used to compute the maximum number of tokens each expert can receive, mitigating load imbalance.
   - B, T, D, N: batch size (number of input samples/sentences per batch), maximum or fixed sequence length after padding, model feature vector dimension (d_model — each token's embedding dimension), total token count in the batch = $B \times T$.

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=4, k=1, capacity_factor=1.25, noisy_gating=True):
        super().__init__()
        assert k in (1,2) # Ensure K active experts is 1 or 2
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.noisy_gating = noisy_gating

        # Gating network: computes matching scores (logits) between each token and n_experts experts.
        self.w_gating = nn.Linear(d_model, n_experts, bias=False)
        if noisy_gating:
            # Noise network: introducing noise helps distribute tokens more evenly across experts during training, mitigating load imbalance.
            self.w_noise = nn.Linear(d_model, n_experts, bias=False)

        # Expert networks — each expert is an independent FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(), # Using GELU activation function
                nn.Linear(d_ff, d_model)
            ) for _ in range(n_experts)
        ])

    def _noisy_logits(self, x):
        """
            x : Flattened input token vectors, shape [N, D] (N=B*T).
            Returns: Expert logits with noise, shape [N, E].
        """
        logits = self.w_gating(x)
        
        # Introduce random noise during training mode when noisy_gating is enabled
        if self.noisy_gating and self.training:
            # Use sigmoid to map w_noise output to [0, 1] as noise standard deviation
            noise_std = torch.sigmoid(self.w_noise(x))

            # Add normally-distributed noise — this enhances randomness, aiding load balancing during training
            logits = logits + torch.randn_like(logits) * noise_std
        return logits

    def forward(self, x, mask=None):
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)  # [B, T, D] -> [N, D]

        logits = self._noisy_logits(x_flat)
        scores = F.softmax(logits, dim=-1) # Normalized expert selection weights, [N, E]

        if self.k == 1:
            top1 = torch.argmax(scores, dim=-1)  # Top-1 expert index selected per token, [N]
            # Dispatch Mask: [N, E], marks each token's selected Top-1 expert with 1
            dispatch_mask = F.one_hot(top1, num_classes=self.n_experts).to(x.dtype)
            # Extract each token's selected Top-1 expert score as final combination weight, [N]
            combine_weights = torch.gather(scores, 1, top1.unsqueeze(1)).squeeze(1)
            # Compute the maximum number of tokens each expert can process
            capacity = int((N/self.n_experts)*self.capacity_factor)+1

            expert_inputs = []
            expert_indices = []
            for e in range(self.n_experts):
                # Find original indices of tokens that expert e should process, [N]
                idx = torch.nonzero(dispatch_mask[:, e], as_tuple=False).squeeze(-1)
                if idx.numel() > capacity:
                    # Expert e capacity check — if exceeded, discard excess tokens
                    idx = idx[:capacity]
                # Save tokens that expert e needs to process
                expert_inputs.append(x_flat[idx])
                # Record original indices of tokens expert e needs to process
                expert_indices.append(idx)
            # Initialize output
            out_flat = torch.zeros_like(x_flat)

            # Iterate over each expert
            for e in range(self.n_experts):
                if expert_inputs[e].size(0)==0:
                    continue   # Expert e has no tokens to process
                # Expert e processes tokens
                y = self.experts[e](expert_inputs[e])
                out_flat[expert_indices[e]] = y  # Place expert e's output back at its original sequence position
            out_flat = out_flat * combine_weights.unsqueeze(1)  # Multiply all expert-processed results by combination weights
            return out_flat.view(B, T, D)
        else:
            # Top-2 simplified implementation
            # Each token selects Top-2 expert scores and indices, [N, 2]
            topk_vals, topk_idx = torch.topk(scores, k=2, dim=-1)
            # Compute max tokens per expert
            capacity = int((N/self.n_experts)*self.capacity_factor)+1
            expert_buckets = [[] for _ in range(self.n_experts)] # Initialize storage
            for i in range(N):
                for j in range(2):
                    e = int(topk_idx[i,j].item())      # Top-K expert index
                    w = float(topk_vals[i,j].item())   # Corresponding token combination weight
                    expert_buckets[e].append((i,w)) # Store: token original index, weight

            out_flat = torch.zeros_like(x_flat) # Initialize output
            for e in range(self.n_experts):
                bucket = expert_buckets[e]
                if len(bucket)==0:
                    continue
                if len(bucket) > capacity:
                    bucket = bucket[:capacity]  # Per expert: discard tokens exceeding capacity

                # Convert token original indices to tensor: [C] (C = count after capacity limit)
                idxs = torch.tensor([i for i,_ in bucket], device=x.device, dtype=torch.long)
                # Corresponding combination weights to tensor: [C]
                weights = torch.tensor([w for _,w in bucket], device=x.device, dtype=x.dtype)
                inp = x_flat[idxs]  # Get tokens expert e needs to process, [C, D]
                y = self.experts[e](inp)
                # Multiply expert output by weights, accumulate onto output tensor (Top-2 superposition); the same token may be processed by multiple experts
                out_flat[idxs] += y * weights.unsqueeze(1)
            return out_flat.view(B,T,D)
```

In the above MoE architecture, two strategies are combined to address load imbalance:
1. **Noisy Gating**
   - Principle: Introduce normally-distributed random noise into router logits, modulated by a data-dependent standard deviation $\sigma = \text{Sigmoid}(W_{\text{noise}}x)$.
   - Effect: During training, this noise slightly perturbs Top-K selection results, encouraging the router to choose different expert combinations for input tokens, thereby **enhancing expert diversity** and **reducing router determinism**, helping distribute load.

2. **Capacity Limiting**
   - Principle: Set a maximum capacity for each expert $C_{expert} = \lceil (\frac{N}{E}) \times capacity\_{factor} \rceil$. If tokens routed to an expert exceed $C_{expert}$, *discard* the excess tokens.
   - Effect: Forces all experts to only process a limited number of tokens, preventing a few experts from being overwhelmed with token resources and ensuring the entire MoE layer's computation time is predictable and stable. However, discarded tokens lose part of their input semantic information — if a token passes through no expert at all, this negatively impacts model convergence speed and final accuracy.

**Step 4: Build the Complete Transformer Block**

Supports switching between traditional FFN and MoE. One Transformer Block contains two sublayers in order: Self-Attention layer, then FFN or MoE. Structure can be referenced from the Switch Transformer figure.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, use_moe=False, moe_params=None, dropout=0.1):
        super().__init__()
        # First sublayer: Multi-head self-attention
        self.attn = SimpleSelfAttention(d_model, nhead) 
        
        # LayerNorm layer: LN1 before attention layer
        self.ln1 = nn.LayerNorm(d_model)
        # LayerNorm layer: LN2 before FFN/MoE layer
        self.ln2 = nn.LayerNorm(d_model) 
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        self.use_moe = use_moe
        
        # Second sublayer: use FFN or MoE based on use_moe
        if use_moe:
            assert moe_params is not None
            # Sparse MoE layer 
            self.moe = MoELayer(**moe_params)
        else:
            # Traditional Feed-Forward Network (FFN)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), # Expand dimension
                nn.GELU(),                # Activation function
                nn.Linear(d_ff, d_model)  # Restore dimension
            )
            
    def forward(self, x, mask=None):
        # Transformer Block forward propagation
        # First sublayer: Self-attention module
        # 1. Layer Norm (LN1) -> 2. Attention -> 3. Dropout -> 4. Residual connection (+)
        attn_out = self.attn(self.ln1(x), mask=mask)
        x = x + self.dropout(attn_out)
        
        # Second sublayer: FFN / MoE module
        if self.use_moe:
            # MoE path: Layer Norm -> MoE -> Dropout -> Residual connection
            moe_out = self.moe(self.ln2(x), mask=mask)
            x = x + self.dropout(moe_out)
        else:
            # FFN path: Layer Norm -> FFN -> Dropout -> Residual connection 
            ffn_out = self.ffn(self.ln2(x))
            x = x + self.dropout(ffn_out)
        return x
```

**Step 5: Mini LLM + MoE Model**
```python
# Mini LLM + MoE model
class MiniMoELLModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, n_layers=4, d_ff=1024,
                 use_moe_layer_index=None, moe_params=None):
        """
        use_moe_layer_index: which layers use MoE, e.g. [1,3]
        moe_params: MoE parameter dictionary, automatically injects d_model and d_ff
        """
        super().__init__()
        self.vocab_size = vocab_size      # Vocabulary size — no need to consider special token prediction
        self.d_model = d_model            # Token Embedding dimension

        # Token + Position encoding
        self.tok_emb = nn.Embedding(vocab_size, d_model) # Token embedding layer
        self.pos_emb = nn.Embedding(4096, d_model)        # Learnable position encoding, max context window length limited to 4096

        # Transformer layers
        self.layers = nn.ModuleList()
        # Determine whether to use MoE
        if use_moe_layer_index is None:
            use_moe_layer_index = set() # Default: use standard FFN
        else:
            use_moe_layer_index = set(use_moe_layer_index)
        # Configure MoE-related parameters
        if moe_params is not None:
            moe_params = moe_params.copy()        # Copy parameters, inject LLM's d_model and d_ff
            moe_params.setdefault("d_model", d_model)
            moe_params.setdefault("d_ff", d_ff)

        for i in range(n_layers):
            use_moe = (i in use_moe_layer_index)  # Determine if current layer uses MoE module
            self.layers.append(
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    d_ff=d_ff,
                    use_moe=use_moe,
                    moe_params=moe_params
                )
            )

        # LayerNorm + output layer, weight-tied with embedding
        self.ln_f = nn.LayerNorm(d_model) # Final Layer Normalization
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False) # Language model head, logits projection
        self.lm_head.weight = self.tok_emb.weight   # Weight tying

    def forward(self, idx, mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0) # Generate position indices [1, T]
        x = self.tok_emb(idx) + self.pos_emb(pos)      # Input embedding = Token Embedding + Position Embedding, [B, T, D]
        for blk in self.layers:
            x = blk(x, mask=mask)   # Pass through Transformer blocks containing Attention and FFN/MoE
        x = self.ln_f(x)            # Final Layer Normalization
        logits = self.lm_head(x)    # Project to vocabulary dimension, obtaining logits [B, T, vocab_size]
        return logits  # Return logits for loss computation or Softmax probability prediction
```
*Mini LLM = Token Embedding + Position Encoding + Transformer Layers + Output Projection*

>What is the role of LayerNorm before the output projection in the Mini LLM?
>
>Applying Layer Normalization before entering the final prediction head (lm_head) is a standard step whose core purpose is to stabilize and normalize the model's output hidden representation $x$. It performs independent per-sample normalization on each token embedding's d_model-dimensional features, ensuring that features $x$ input to the final linear projection layer have approximately consistent scale and distribution. This normalization effect not only significantly stabilizes the model training process — allowing higher learning rates and thus faster convergence — but also helps lm_head more accurately map uniformly-scaled features back to the vocabulary (logits), ultimately improving LLM prediction accuracy.

**The above demonstrates key module code for MoE in Mini LLM applications. Complete runnable code at [Mini LLM+MoE](https://github.com/datawhalechina/diy-llm/blob/main/docs/en/chapter5/Mini%20LLM%2BMoE.py)**

---

## 5.4 DeepSeek Innovations

### 5.4.1 DeepSeek V3 Key Innovations

DeepSeekMoE is an innovative Mixture of Experts model targeting **extreme expert specialization** to address the **knowledge mixing** and **knowledge duplication** problems in traditional MoE models, thereby dramatically improving model performance and parameter efficiency while keeping computational cost moderate. The `DeepSeekMoE` architecture achieves expert specialization primarily through two strategies:

<div align="center">
<img width="1350" height="600" alt="DeepSeekMoE" src="https://github.com/user-attachments/assets/6aab083e-c9b6-48a2-9f7d-28d833c786a0" />
   <p>DeepSeekMoE Structure</p>
</div>

- **Fine-grained expert segmentation**: While keeping total expert parameters constant, shrink the original "larger" FFN experts proportionally (e.g., each small expert = 0.25× standard FFN parameters), and split each original expert into several smaller experts, thus significantly increasing the total number of experts — i.e., expanding $N$ experts into $mN$ small experts. This approach shifts model parameter density from "larger per expert" to "more but smaller experts," enabling finer-grained division of labor among experts.

   - **Constant-compute activation strategy**: To keep $\frac{\text{activation computation}}{\text{activation parameters}}$ roughly constant, the model activates more `small experts` per forward pass. In other words, when each expert becomes smaller (fewer parameters), the router selects more experts to participate — e.g., expanding the original Top-K activation to activate $mK$ small experts after segmentation — thereby maintaining or improving representational capacity in parameter combinations while controlling per-forward-pass compute budget.
   
   - **Combination flexibility and exponential combination space growth**: After fine-graining experts, the number of selectable expert set combinations grows explosively in factorial/combinatorial fashion, significantly increasing the router's freedom and diversity in constructing `expert coalitions` for any given input.
  
>For example, if originally $N=16$ with Top-2 activation, the number of possible combinations is $C_{16}^2=120$; if each original expert is split into 4 small experts, fine-graining yields a total of 64 experts with 8 small experts activated, giving $C_{64}^8=4,426,165,368$ possible combinations — demonstrating the massive expansion of the potential combination space.

- **Shared experts**: Proposes retaining several `shared experts` to capture common knowledge, thereby reducing redundancy among routed experts and stabilizing training — i.e., retaining $K_s$ shared experts outside the routed experts as permanent receivers or compensation channels. This design works synergistically with fine-grained segmentation, improving specialization while maintaining coverage of common patterns.

<div align="center">
   <img width="1275" height="543" alt="Ablation" src="https://github.com/user-attachments/assets/d4f713ba-e9c5-4d57-95cd-82a914610828" />
   <p>Controlled experiment with equal total and active parameters</p>
</div>

In experiments keeping total parameters and activated parameters constant, progressively splitting experts smaller does indeed improve model performance. However, as experts become increasingly fine-grained, performance gains gradually diminish, and engineering factors like communication overhead and routing stability begin to have greater impact — meaning performance improvement is not unbounded. The [paper](https://arxiv.org/pdf/2401.06066)'s ablation experiments also provide an empirical finding: **when shared experts and activated specialized experts maintain approximately a 1:3 ratio, benchmark task performance is best**.

Therefore, in practice, one must trade off among expert granularity, activation count, shared expert ratio, and communication/routing overhead, finding the optimal configuration for current hardware and compute budget through ablation experiments.

- **Load balancing strategies**: To mitigate load imbalance that may lead to routing collapse and computational bottlenecks, DeepSeekMoE introduces auxiliary losses, which saw further evolution in subsequent DeepSeek-V3 versions.

   - Expert-level balance loss $L_{ExpBal}$: Used to minimize unevenness in token distribution across experts, thereby mitigating routing collapse risk.
   - Device-level balance loss $L_{DevBal}$: When experts are distributed across multiple devices (as in DeepSeek-V3), this loss is introduced to ensure cross-device computational load balance, optimizing parallel computation efficiency.

>In DeepSeek-V3, to reduce communication overhead: first, quantize transmitted activations to FP8 format, reducing message bandwidth; second, compress the gradients corresponding to activations before feeding into MoE projections, saving communication and memory. To avoid affecting training stability in this process, the critical computation involved in "merging" the outputs of various experts still uses BF16 format to ensure precision is maintained. **Simply put: use low precision for transmission (save bandwidth), use medium precision for critical computation (ensure stability).**

---

### 5.4.2 DeepSeek V4 Improvements

Facing extremely unstable training in shallow MoE architectures, DeepSeek V3, Mimo, and other open-source models chose to replace shallow layers with Dense FFN layers to provide stable input for subsequent learnable-routing MoE layers. In contrast, [DeepSeek V4 directly introduced 3-layer Hash MoE (non-learnable routing) in shallow layers to replace previous Dense layers](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json), which to some extent mitigates this problem.

Facing severe instability and recurrent loss spikes caused by doubling training data volume during MoE training — problems no longer solvable by simple `rollback mechanisms` — DeepSeek V4 introduced two methods:

- **Look-ahead routing**, which decouples the synchronized update process of the routing network and backbone network. Its main ideas are:
    - Use **the previous step's** (i.e., step $t - \Delta t$) routing parameters to **pre-compute and cache expert assignment results** for step $t$'s data;
    - Simultaneously, introduce an asynchronous automatic detection mechanism that only triggers this look-ahead routing mode upon detecting loss spikes; the system rolls back and runs stably for a period before switching back to the standard training流程.

*📖Tip: Although look-ahead routing incurs approximately 20% additional computational overhead, through pipelining with the computation process, it successfully eliminates training instability caused by Loss Spikes without harming final model performance.*

- **SwiGLU clamping**: During training, **the team discovered that constraining the activation values of the SwiGLU activation function within a fixed range can effectively suppress outlier generation**. Specifically, clamp the `SwiGLU linear component` to [-10, 10] and cap the `gating component` upper bound at 10.

>For techniques improving MoE model training stability, the DeepSeek team notes these still lack a complete theoretical foundation and have only been validated有效 in practical application. The community can further research these issues going forward.

## 5.5 MoE and Deep Learning

**Basic-level feature extraction and traditional specialization**

In deep learning, each layer of a neural network typically **extracts features layer by layer**: for example, convolutional neural networks start from low-level edge and texture information, gradually building up to high-level object parts and abstract semantic features. Within the same layer, different convolutional kernels **respond to input in parallel**, each sensitive to specific patterns — this intuitively embodies **feature specialization**. Note that this specialization is **implicit, fixed, and weakly parallel**, jointly determined by weights and input data.

**MoE's dynamic and sparse specialization**

Unlike traditional convolutional kernels' fixed specialization, **Mixture of Experts** models introduce **conditional routing** (e.g., **Top-k gating mechanisms**), dynamically activating only a **small number of expert modules** during forward computation. Each expert can focus on processing specific types of inputs or feature patterns, thereby **significantly increasing model capacity** while keeping **FLOPs controllable**. In other words, MoE's specialization is **input-driven, dynamic, and sparse**, while traditional convolutional kernel specialization is operator-parallel and **fixed**.

**Engineering challenges**

From a macro perspective, biological brains also exhibit **localized or modular activation** when processing information: visual, language, motor, and other brain regions "each do their own job." This intuitively resembles MoE's **expert specialization + conditional activation** — both embody the advantages of **modularity and selective computation**. *However, this is merely a conceptual analogy; engineering implementations and biological mechanisms are not directly correspondent.* In practical MoE systems, one must also consider: **expert load balancing**, **router stability**, and **distributed communication overhead** — real engineering challenges.

**Optimization approaches and methods for ultra-large-scale MoE**

On ultra-large-scale **MoE inference models**, researchers have demonstrated the feasibility of efficient fine-tuning via **LoRA + Reinforcement Learning**, where LoRA adds low-rank adapters to the model's dense and expert layers so that only a small number of parameters are updated during fine-tuning, while **RL** is used to optimize the model's behavioral policy.

**Introducing RL to large models during the fine-tuning stage**. Taking `Kimi-K2` as an example (~`1.04T` total parameters, ~`32.6B` active parameters), **the research team combined hybrid parallelism with LoRA sharding** to achieve stable RL training, reducing GPU cost to approximately 10% compared to full-parameter RL.

`Kimi-K2`'s comparative experiments show that — `large base model + small-scale LoRA` RL clearly outperforms `small model full-parameter RL`. **This phenomenon can be attributed to: the effectiveness of RL at this stage is limited by the base model's `prior capability`** — a powerful base model can produce higher-quality training trajectories, thereby significantly improving RL's learning efficiency. Ultra-large-scale `MoE` encounters some special challenges, such as:

   - **Routing imbalance**: Some experts are over-invoked while others sit idle;
   - **Communication pressure**: Frequent data exchange between different GPUs and nodes;
   - **Complex parallel layout**: Combinations of tensor, pipeline, expert, and sequence parallelism are hard to optimize;
   - **Training-inference inconsistency**: May cause sudden imbalance in expert importance ratios.

To address these issues, the Kimi team proposed several engineering optimization methods:

1. **Hybrid parallel design**: Rationally arranging different parallelization approaches to reduce communication overhead;
2. **Truncated importance sampling correction**: Preventing a few experts from becoming overloaded;
3. **Adaptive parallel scheduler**: Automatically adjusting tensor, pipeline, expert, and sequence parallelism strategies based on real-time metrics (GPU utilization, memory, step time).

> These conclusions are based on Kimi-K2 and specific tasks, with engineering environment dependencies; in other models or tasks, effects may differ and require reproduction verification.

## 5.6 Summary

This section covered improvements to standard Attention, MoE core principles and mainstream routing mechanisms, and analyzed the stability challenges faced in large-scale training. Combined with frontier LLM practice, we explored how architecture design (DeepSeek V4's Hash MoE, look-ahead routing) and training strategies (SwiGLU clamping) effectively improve MoE model training efficiency. Recently, MoE has further evolved with the emergence of Kimi K3 and Nemotron 3's LatentMoE.

## Reflection Questions

**Basic:**
1. What are current standard Attention improvements and their principles?
2. What methods exist for improving MoE load imbalance and training instability?
3. What are the differences between MoE and Dense models, and their applicable scenarios?

**Advanced:**
1. Router training typically faces non-differentiable optimization challenges and early instability, particularly prone to causing expert load imbalance. This imbalance not only affects training efficiency but also hinders each expert from forming stable and specialized functions. So, how to stabilize router training to better distinguish each expert's "specialty" — is this a key research direction in `MoE` architecture, and an important direction for improving model performance and resource utilization?

>Phase 1: Chaos Begins (Warm-up) — Start training, allow router free exploration; shared experts learn general knowledge, specialized experts activate randomly. **Goal: mitigate early load imbalance, give every expert the opportunity to encounter diverse inputs.**
>
>Phase 2: Career Planning — Pause LLM training using saved model state snapshots, or analyze expert activation patterns; identify sensitive experts and explicitly label or adjust routing loss so experts focus on specific input types. **Goal: help the router rapidly and stably distinguish each expert's specialty, forming clear "expert-function" mappings.**
>
>Phase 3: Directed Deep Training — Shared experts continue learning general capabilities; specialized experts only activate and train on key inputs. **Goal: maintain general capability while letting specialized experts achieve deep specialization on specific tasks.**

## References
- [Trillion-Parameter MoE RL Tuning](https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus)
- [DeepSeek-MoE](https://arxiv.org/pdf/2401.06066)
- [DeepSeek-V3](https://arxiv.org/pdf/2412.19437)
- [Kimi KDA Architecture](https://arxiv.org/pdf/2510.26692v2)
- [Nemotron 3 Super 120B A12B](https://arxiv.org/abs/2604.12374)
- [Kimi Linear 48B A3B](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py)