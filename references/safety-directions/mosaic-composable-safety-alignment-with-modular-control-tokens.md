Title: MOSAIC: Composable Safety Alignment with Modular Control Tokens
ArXiv: 2603.16210
Authors: Jingyu Peng, Hongyu Chen, Jiancheng Dong, Maolin Wang, Wenxi Li, Yuchen Li, Kai Zhang, Xiangyu Zhao
Sections: 28
Estimated tokens: 12.0k

## Contents
- 1 Introduction
- 2 Methodology
  - 2.1 Compositional Task Sampling
  - 2.2 Training Objective
    - Positive Samples.
    - Counterfactual KD for Negative Samples.
- 3 Experiments
  - 3.1 Dataset Construction
  - 3.2 Experimental Setup
  - 3.3 Evaluation
  - 3.4 Baselines
    - In-context.
    - ORPO.
    - SFT.
  - 3.5 Overall Performance
  - 3.6 Performance with different number of control tokens
  - 3.7 Ablation Study
  - 3.8 Performance under different negative-to-positive ratio
  - 3.9 Incremental Category Expansion
- 4 Related Work
  - 4.1 Safety Alignment
  - 4.2 Pluralistic Alignment
- 5 Conclusion
- 6 Limitations
- References
- Appendix A Appendix
  - A.1 Parameter Analysis
  - A.2 Prompt for Judgement

## Abstract

Abstract Safety alignment in large language models (LLMs) is often implemented as a static policy embedded in model parameters, making it difficult to adapt safety rules across users, regions, and applications. Existing approaches struggle to provide such conditional control: parameter-level alignment entangles safety behaviors with general capabilities, while prompt-based methods rely on weak natural language instructions.
We propose MOSAIC, a modular framework for compositional safety alignment using learnable control tokens optimized over a frozen backbone model. Each safety constraint is encoded by a small set of tokens that can be flexibly activated and composed at inference time. To train compositional tokens efficiently, we introduce order-based task sampling and a distribution-level alignment objective to reduce over-refusal. Experiments on a newly constructed realistic benchmark show that MOSAIC achieves strong defense performance while substantially reducing over-refusal and preserving model utility.

## 1 Introduction

Large language models (LLMs) are increasingly deployed in real-world applications where safety alignment must accommodate diverse user populations and contextual requirements  Yin et al. (2024); Guan et al. (2025). In practice, safety policies vary across age groups, jurisdictions, professional roles, and application domains. Content that may be appropriate for adults can be restricted for minors  Purba et al. (2023), and material legally permissible in one country may be prohibited in another  Qiu et al. (2025).

These variations imply that safety alignment cannot be treated as a single static policy uniformly embedded in the model. Instead, safety must be implemented through conditional and compositional constraint activation, where different subsets of rules are dynamically enabled based on user attributes and contextual factors. This reframes safety alignment as a context-sensitive control problem rather than a monolithic model property.

Figure: Figure 1: An illustration comparing two existing safety alignment paradigms and the proposed MOSAIC framework at inference time.
Refer to caption: https://arxiv.org/html/2603.16210/2603.16210v1/x1.png

Existing alignment approaches exhibit clear limitations under this setting. Parameter-level methods, such as supervised fine-tuning Chung et al. (2024) and reinforcement learning from human feedback Ouyang et al. (2022), entangle safety behaviors with general model capabilities Lambert and Calandra (2023); Kirk et al.. As a result, safety policies embedded in model weights are difficult to decouple or update, often requiring costly retraining and risking catastrophic interference with previously learned behaviors Behrouzi et al. (2026); Wang et al. (2024).

Prompt-based methods provide superficial flexibility but rely on natural language instructions to express safety constraints Liu et al. (2023a). Because such instructions are interpreted probabilistically rather than enforced through explicit control mechanisms, they can be followed inconsistently and become inefficient when multiple constraints lead to long prompts that increase token overhead and reduce available context Zhuo et al. (2024); Liu et al. (2024); OpenAI (2023).

We argue that the core limitation of existing methods is fundamentally representational. Current approaches either entangle safety behaviors within model parameters or encode safety rules in natural language, but neither yields an explicit, reusable, and composable representation of safety constraints Guan et al. (2025). Consequently, they struggle to provide fine-grained and conditional control without incurring significant retraining costs, instability, or efficiency trade-offs.

To address this limitation, we reconceptualize safety alignment as a representation learning problem and propose CoMpOsable Safety AlIgnment with Modular Control Tokens (MOSAIC). MOSAIC represents each safety constraint as a small set of learnable control tokens in the embedding space of a frozen backbone language model. Instead of modifying model parameters or encoding policies in natural language, safety behaviors are induced by prepending the corresponding control tokens to the input. Each constraint is encoded by a small set of control tokens optimized to activate the associated refusal behavior, and multiple token sets can be composed at inference time to enable conditional and multi-policy control.

This design offers advantages. Firstly, safety control is fully decoupled from the backbone model, allowing policies to be added, removed, or recombined without retraining the base model; new safety categories can likewise be incorporated incrementally by learning additional control tokens while keeping previously learned ones fixed. Moreover, as constraints are represented as independent embeddings, multiple safety requirements can be composed through simple token concatenation, enabling flexible activation across user groups, regions, or application domains. At the same time, since the backbone parameters remain frozen throughout training, MOSAIC preserves general language modeling capabilities and avoids interference between safety updates and task performance.

To realize the optimization of compositional control tokens without incurring exponential data cost, we introduce a combinatorial task sampling strategy together with order-based balanced data allocation. By organizing category combinations according to the number of active constraints and allocating a fixed training budget per order, the model is exposed to diverse token compositions while keeping the overall supervision scale bounded. This design enables effective joint training of control tokens without the exponential growth that naive enumeration would incur. To mitigate over-refusal on benign queries, we introduce a counterfactual knowledge distillation objective on non-target samples. Instead of relying solely on sequence-level hard labels as in standard supervised fine-tuning, we compare the model’s behavior with and without control tokens and align the controlled distribution with the backbone model’s original responses on benign inputs. This counterfactual supervision constrains control tokens to intervene only when necessary, preserving the base model’s behavior on unrelated requests while substantially reducing unintended refusals.

Figure: Figure 2: Overview of the proposed MOSAIC approach, showing the sampling strategy in the upper panel and the training objective in the lower panel.
Refer to caption: https://arxiv.org/html/2603.16210/2603.16210v1/x2.png

On the other hand, precisely evaluating conditional safety control requires care. Many widely used safety benchmarks overlap with data seen during post-training alignment of mainstream LLMs  Perez et al. (2022); Ji et al. (2023), making it hard to separate genuine safety capability from distribution familiarity. Some prior work constructs artificially “unaligned” models and reapplies safety techniques  Shairah et al. (2025); Russinovich et al. (2026); Zhang et al.; Zhan et al. (2024), but this differs from real deployment, where models are already aligned and new safety constraints must be added without altering core behavior. This highlights a gap between current evaluation protocols and practical scenarios.

To bridge this gap and more accurately evaluate MOSAIC under realistic deployment conditions, we construct a new evaluation dataset grounded in practical safety requirements. The dataset consists of 1,500 user requests spanning five safety categories, each corresponding to behaviors that may be unsafe or inappropriate for specific demographic groups or contextual conditions  Purba et al. (2023). Importantly, these requests are not rejected by mainstream aligned LLMs under default configurations, making them suitable for assessing selective and conditional safety activation rather than generic refusal capability. This benchmark enables us to test whether a method can impose additional safety constraints precisely when required, while leaving unrelated behaviors unaffected.

Our contributions are threefold:

- •
We reconceptualize safety alignment as a compositional representation learning problem, framing conditional safety control as modular constraint activation rather than monolithic parameter modification.
- •
We propose MOSAIC, a framework that represents safety constraints as learnable control tokens over a frozen backbone model. It enables compositional constraint activation and incremental category expansion, while mitigating over-refusal via structured combinatorial training and counterfactual knowledge distillation that preserves the backbone model’s behavior on benign inputs.
- •
We construct a realistic evaluation benchmark tailored to conditional safety activation in already aligned models, enabling precise assessment of selective constraint enforcement without sacrificing general utility.

## 2 Methodology

In MOSAIC, we formalize conditional safety control as a compositional representation learning problem. Instead of modifying backbone parameters, we represent each safety constraint as a small set of learnable *control tokens* and optimize them over a frozen language model. These tokens serve as modular constraint carriers that can be selectively activated and composed at inference time.

Let $f_{\theta}$ denote a pretrained language model with frozen parameters. Given a set of safety categories $\mathcal{C}=\{c_{1},\dots,c_{K}\}$, we associate each category $c$ with a small set of learnable control tokens:

$$ $\mathbf{z}_{c}=\{z_{c,1},\dots,z_{c,m}\},$ $$

where $m<10$ in practice and each $z_{c,i}\in\mathbb{R}^{d}$ lies in the model’s embedding space. These control tokens are the only trainable parameters.

Given an input instruction $x$ and an active subset of categories $S\subseteq\mathcal{C}$, we prepend the corresponding tokens to the input:

$$ $[\mathbf{z}_{c_{i_{1}}},\dots,\mathbf{z}_{c_{i_{r}}},x].$ $$

The resulting sequence is processed by the frozen model $f_{\theta}$. Activating or deactivating safety constraints amounts to inserting or removing a few learned vectors, enabling lightweight, modular, and incremental safety control without modifying the parameters of the backbone.

Although this formulation is simple, learning compositional control tokens presents three key challenges: (i) maintaining reliable control when tokens from different categories are freely composed, preventing cross-category interference, (ii) exponential growth in category combinations, and (iii) over-refusal on benign queries. We address these challenges through a structured compositional training strategy.

### 2.1 Compositional Task Sampling

To maintain reliable control when tokens from different categories are freely composed, each token must remain effective across all combinations in which it appears. For example, a token from category $A$ should function correctly not only in isolation, but also when combined with tokens from other categories (e.g., $A{+}B$, $A{+}C$, $A{+}B{+}C$). In principle, this requires training over the space of compositional tasks.

A naive approach would enumerate all possible subsets of $\mathcal{C}$, whose number grows exponentially ($2^{K}-1$), making exhaustive supervision computationally infeasible. To enable efficient traversal of diverse token combinations without incurring exponential data cost, we organize subsets according to their *order*, defined as the number of active categories. For order $r$, we define:

$$ $\mathcal{T}_{r}=\{S\subseteq\mathcal{C}\mid|S|=r\}.$ $$

Instead of allocating supervision per subset, we allocate a fixed training budget per order. Let $N_{r}^{\text{pos}}$ and $N_{r}^{\text{neg}}$ denote the total positive and negative budgets for order $r$. These budgets are evenly divided among subsets in $\mathcal{T}_{r}$:

$$ $\frac{N_{r}^{\text{pos}}}{|\mathcal{T}_{r}|}\quad\text{and}\quad\frac{N_{r}^{\text{neg}}}{|\mathcal{T}_{r}|}.$ $$

This order-based allocation decouples training cost from the combinatorial size of $\mathcal{T}_{r}$, ensuring that exposure to higher-order compositions does not cause exponential growth in supervision. By cycling across orders during optimization, control tokens are explicitly trained under joint activation, promoting cooperative interaction and mitigating dominance effects observed in independently trained tokens.

### 2.2 Training Objective

Each training example consists of an instruction $x$, an associated active category subset $S$, and a binary indicator specifying whether $x$ should trigger refusal behavior under $S$.

#### Positive Samples.

For instructions that belong to active categories, the target output is a fixed refusal template. We optimize the standard autoregressive cross-entropy loss:

$$ $\mathcal{L}_{\text{ref}}=-\sum_{t}\log p_{\theta}(y_{t}^{\text{ref}}\mid\mathbf{z}_{S},x,y_{<t}),$ $$

where $\mathbf{z}_{S}$ denotes the concatenation of control tokens for subset $S$. This objective ensures that activated tokens reliably induce refusal behavior.

#### Counterfactual KD for Negative Samples.

Sequence-level supervised fine-tuning typically applies hard labels to entire outputs, which may encourage overly conservative behavior once control tokens are inserted. In particular, the model may learn to refuse benign instructions simply due to the presence of control tokens, leading to over-refusal.

To mitigate this issue, we introduce a counterfactual knowledge distillation objective that provides fine-grained token-level supervision. The key idea is to compare the model’s behavior with and without control tokens, treating the latter as a counterfactual reference.

For a benign instruction $x$, we first compute the output distribution of the frozen backbone model without any control tokens:

$$ $p^{\text{base}}(\cdot\mid x).$ $$

We then activate a subset of control tokens $\mathbf{z}_{S}$ and obtain the corresponding controlled distribution:

$$ $p^{\text{ctrl}}(\cdot\mid\mathbf{z}_{S},x).$ $$

We minimize the KL divergence between the counterfactual reference distribution and the controlled distribution:

$$ $\mathcal{L}_{\text{KD}}=\mathrm{KL}\big(p^{\text{base}}(\cdot\mid x)\;\|\;p^{\text{ctrl}}(\cdot\mid\mathbf{z}_{S},x)\big).$ $$

The final objective for negative samples combines the standard language modeling loss with the distillation term:

$$ $\mathcal{L}_{\text{neg}}=\mathcal{L}_{\text{LM}}+\lambda\mathcal{L}_{\text{KD}},$ $$

where $\lambda$ controls the relative weight of the counterfactual KD loss.

This counterfactual distillation signal encourages selective intervention: control tokens modify the model’s behavior only when inputs violate active safety constraints, while preserving the backbone distribution on benign instructions. Notably, the teacher signal is obtained directly from the same frozen model without control tokens, requiring no additional supervision.

## 3 Experiments

**Table 1: Performance comparison of various safety-alignment methods. # Params denotes the number of trainable parameters for optimization-based methods (SFT, ORPO) or the additional parameter overhead per category for prompt-based approaches (In-context, MOSAIC). The suffix $/C$ indicates that the overhead scales linearly with the number of target categories ($C$). MOSAIC-N indicates that the number of control tokens per category is set to N. K-order refers to the optimization and usage of control token combinations across K categories.**
| Model | Method | # Params $\downarrow$ | 1-order | 2-order | 3-order | 4-order |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DSR $\uparrow$ | OR $\downarrow$ | DSR $\uparrow$ | OR $\downarrow$ | DSR $\uparrow$ | OR $\downarrow$ | DSR $\uparrow$ | OR $\downarrow$ |  |  |  |
| Llama-3.1-8B | In-context | 0.4M$/C$ | 76.7 | 10.2 | 62.0 | 13.6 | 51.7 | 11.2 | 44.5 | 13.9 |
| ORPO | 33.6M | 79.8 | 29.1 | 75.3 | 28.9 | 78.1 | 30.2 | 76.4 | 28.7 |  |
| SFT | 33.6M | 99.4 | 7.3 | 99.5 | 6.6 | 98.3 | 6.3 | 98.9 | 6.1 |  |
| MOSAIC-2 | 8.2k$/C$ | 99.8 | 7.1 | 99.6 | 7.5 | 99.8 | 6.2 | 100.0 | 5.9 |  |
|  | MOSAIC-5 | 20.5k$/C$ | 99.8 | 4.3 | 99.8 | 4.3 | 100.0 | 2.0 | 100.0 | 1.8 |
| Llama-3.2-3B | In-context | 0.3M$/C$ | 61.7 | 12.3 | 51.3 | 10.9 | 48.8 | 13.5 | 41.9 | 12.3 |
| ORPO | 18.4M | 74.3 | 29.4 | 78.1 | 31.4 | 75.1 | 28.7 | 72.9 | 29.1 |  |
| SFT | 18.4M | 99.1 | 10.3 | 98.8 | 7.4 | 99.1 | 5.4 | 98.9 | 5.2 |  |
| MOSAIC-2 | 6.1k$/C$ | 100.0 | 10.4 | 99.2 | 6.3 | 99.5 | 5.4 | 99.7 | 5.0 |  |
| MOSAIC-5 | 15.4k$/C$ | 99.6 | 8.4 | 99.3 | 3.3 | 99.8 | 3.3 | 99.9 | 2.9 |  |

### 3.1 Dataset Construction

Existing widely used safety benchmarks often overlap substantially with data employed during post-training alignment of mainstream LLMs, making it difficult to disentangle genuine safety control capability from distribution familiarity. To better approximate practical deployment scenarios, we construct a dataset covering five safety categories (addiction, alcohol, betting, horror, and sex) that may be unsafe for specific populations, particularly minors, motivated by prior research on adolescent and context-sensitive safety requirements Purba et al. (2023).

For each category, we adopt a multi-stage construction pipeline. We first generate a large pool of candidate instructions using a high-capacity language model under controlled prompting to elicit realistic, first-person user intents. We then apply automatic filtering to remove duplicates, ambiguous cases, and requests that are trivially unsafe under default alignment policies. Subsequently, a combination of LLM-as-Judge evaluation and manual review is used to verify category consistency, linguistic naturalness, and contextual plausibility. This process results in 500 validated requests for each safety category.

In addition to these category-specific samples, we include 500 general-purpose requests unrelated to the target categories. These samples serve two purposes: during training, they act as negative examples that help prevent the model from overfitting to safety-triggering patterns and encourage robustness over a broader query distribution; during evaluation, they enable measurement of over-refusal when safety constraints are activated.

Finally, we validate the resulting dataset across multiple mainstream aligned LLMs to confirm that these requests are not rejected under default configurations. The final dataset contains 3,000 instructions in total (2,500 category-related and 500 neutral requests) and is split into 1,800 training, 600 validation, and 600 test samples, with each category evenly distributed across the splits.

### 3.2 Experimental Setup

We set
${N_{r}^{\text{pos}}}/{|\mathcal{T}_{r}|}=100$ for all methods, so that each subset receives the same number of positive samples and the learning density is consistent across orders. To control data growth for higher-order combinations, we cap the total number of training samples allocated to each order to be equal to that of the first-order (single-category) setting. We maintain a balanced training distribution with a 1:1 ratio of positive to negative examples. Alternative ratios are explored in Section [3.8](#S3.SS8).

We conduct experiments on two backbone models of different scales: Llama-3.1-8B and Llama-3.2-3B  Grattafiori et al. (2024). Control tokens are trained for 8 epochs using the Adam optimizer and the learning rate is set to $0.01$. During generation, outputs are truncated at a maximum length of 256 tokens. The hyperparameter $\lambda$ is fixed to 1.0 in all main experiments.

### 3.3 Evaluation

For automatic evaluation, we use GPT-5  Singh et al. (2025) as a judge model to assess the behavior of model responses. Given a user request and the corresponding model output, the judge evaluates whether the response constitutes a refusal and whether it maintains semantic relevance and coherence with the original instruction. The prompt employed for this judgement is detailed in Section [A.2](#A1.SS2).

We report defense success rate (DSR) on targeted categories, measuring correct refusal under activated constraints, and over-refusal rate (OR) on non-targeted categories, measuring unintended refusals when constraints should not apply  Panda et al. (2024). These metrics jointly capture both selective safety enforcement and preservation of general utility.

### 3.4 Baselines

#### In-context.

A prompt-based baseline that encodes safety constraints as a system instruction. Summaries of prohibited categories are synthesized with GPT-5 and prepended to each input, enabling zero-shot compliance without updating parameters.

#### ORPO.

A parameter-efficient alignment method using LoRA, Odds Ratio Preference Optimization (ORPO)  Hong et al. (2024) leverages pairwise preferences for reward-model-free behavioral shaping, particularly suited for refusal tasks.

#### SFT.

Supervised fine-tuning (SFT) with LoRA serves as a standard alignment baseline, mapping unsafe instructions to refusals and benign ones to answers Ouyang et al. (2022). Together with ORPO, it represents parameter-efficient tuning approaches in contrast to prompt-based methods.

**Table 2: Comparison of performance across task orders on MMLU.**
| Model | Method | 1-order | 2-order | 3-order |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B | Raw | 0.552 | 0.552 | 0.552 |
| In-context | 0.534 | 0.529 | 0.527 |  |
| SFT | 0.547 | 0.544 | 0.547 |  |
| ORPO | 0.549 | 0.542 | 0.549 |  |
| MOSAIC | 0.551 | 0.552 | 0.551 |  |
| Llama-3.2-3B | Raw | 0.507 | 0.507 | 0.507 |
| In-context | 0.489 | 0.485 | 0.472 |  |
| SFT | 0.492 | 0.487 | 0.490 |  |
| ORPO | 0.489 | 0.492 | 0.491 |  |
| MOSAIC | 0.498 | 0.501 | 0.494 |  |

### 3.5 Overall Performance

Table [1](#S3.T1) presents the DSR and OR across different model sizes and task orders. Overall, both In-context prompting and ORPO perform noticeably worse than SFT-based approaches, indicating that prompt-based or preference-based supervision is insufficient to reliably enforce defensive behaviors.

Both In-context prompting and ORPO perform noticeably worse than SFT-based approaches. In-context prompting struggles to maintain reliable refusal behavior as task complexity increases, while ORPO improves DSR but still exhibits substantial over-refusal. In contrast, SFT-based methods, including SFT and MOSAIC, achieve consistently high DSR across all task orders. SFT reaches over 99.0% DSR on the 8B model, but over-refusal remains around 6%, indicating overly conservative refusal behavior.

MOSAIC effectively mitigates over-refusal while maintaining near-perfect DSR across all task orders. With just two memory tokens per category, MOSAIC-2 already matches or slightly surpasses SFT in overall DSR, and increasing the number of tokens to five further reduces OR to as low as 1.8% on higher-order tasks of Llama-3.1-8B. This improvement stems from MOSAIC’s Compositional Task Sampling, which introduces fine-grained category supervision during training. Unlike SFT, where the model observes only coarse positive/negative labels, MOSAIC explicitly exposes the model to diverse category combinations, enabling more precise control and reducing unnecessary refusals.

We also observe that OR decreases with task order. For example, with MOSAIC-5, OR drops from 4.3% to 1.8% on Llama-3.1-8B and from 8.4% to 2.9% on the 3B model. This suggests that higher-order scenarios, with richer category combinations, implicitly regularize the refusal boundary and enable more precise refusal behavior.

Utility evaluation presented in Table [2](#S3.T2) further shows that MOSAIC preserves the general language modeling capability of the base LLM with negligible degradation.

Figure: Figure 3: Performance on Llama-3.1-8B under different numbers of control tokens per category.
Refer to caption: https://arxiv.org/html/2603.16210/2603.16210v1/x3.png

### 3.6 Performance with different number of control tokens

One key factor affecting MOSAIC is the number of control tokens assigned to each category. As shown in Figure [3](#S3.F3), even a single token per category already achieves over 98% DSR and keeps OR below 10% across task orders, indicating that category-specific refusal behavior can be effectively triggered with minimal control tokens.

As the number of memory tokens increases, OR generally decreases while DSR remains near saturation. This suggests that additional tokens increase the expressive capacity of the memory module, enabling finer-grained refusal boundaries and reducing unintended refusals. We also conduct a parameter analysis on the hyperparameter $\lambda$, with the corresponding results and discussion presented in Section [A.1](#A1.SS1).

Figure: Figure 4: Ablation Study. N-token uses only the control tokens of N subset categories during inference.
Refer to caption: https://arxiv.org/html/2603.16210/2603.16210v1/x4.png

### 3.7 Ablation Study

We conduct an ablation study on Llama-3.1-8B to analyze the contributions of multi-task joint optimization and the counterfactual token-level distribution alignment objective. The results are shown in Figure [4](#S3.F4).

For DSR, both the Base variant and MOSAIC w/o Multi-Task perform substantially worse than the full MOSAIC under higher-order token compositions. The performance gap becomes increasingly pronounced as the task order grows (around 50% for 2-order and 33% for 3-order). This degradation occurs because independently trained control tokens lack compositionality: when multiple tokens are activated simultaneously, one token may dominate while others are suppressed, leading to unstable multi-category control. In contrast, multi-task joint optimization explicitly trains category combinations, allowing tokens to learn cooperative interactions and enabling stable performance in higher-order scenarios.

For OR, introducing the counterfactual token-level distribution alignment objective already reduces over-refusal even without multi-task training, particularly in the single-token setting. When combined with multi-task optimization in the full MOSAIC framework, this objective further lowers OR while preserving reliable multi-token composition. These results suggest that counterfactual distribution alignment refines the refusal boundary and improves the precision of safety control.

### 3.8 Performance under different negative-to-positive ratio

An intuitive way to reduce OR is to increase the proportion of negative samples during training. However, this strategy introduces substantial computational cost since the number of tasks grows exponentially with the number of categories ($2^{K}-1$). Moreover, our experiments show that once the negative-to-positive ratio exceeds 1.0, further increasing negative samples yields little additional improvement in OR.

This observation suggests that the key bottleneck in mitigating over-refusal lies not in the quantity of negative supervision but in the granularity of the supervisory signal. To address this, MOSAIC employs a counterfactual knowledge distillation (KD) objective that compares the model’s behavior with and without control tokens. By aligning the control-activated distribution with the backbone model’s original responses on benign inputs, MOSAIC receives fine-grained token-level guidance that helps it better distinguish between targeted and non-targeted categories. As a result, the model preserves appropriate responses for non-targeted requests while maintaining strong refusal behavior where required.

In contrast, simply scaling the number of negative examples provides only coarse-grained data-level constraints, whose effect quickly saturates. These results indicate that counterfactual KD plays a more critical role than sample rebalancing in alleviating over-refusal.

Figure: Figure 5: Performance on Llama-3.1-8B under different negative-to-positive ratio.
Refer to caption: https://arxiv.org/html/2603.16210/2603.16210v1/x5.png

### 3.9 Incremental Category Expansion

To simulate practical scenarios where new safety requirements may arise, we evaluate incremental category expansion on Llama-3.1-8B, where new categories are added without retraining all existing ones. The results are shown in Table [3](#S4.T3).

Adding a single new category (+1) introduces almost no performance degradation. For Llama-3.1-8B, DSR decreases slightly from 99.8% to 99.4%, while OR even drops from 4.3% to 2.1%. Similarly, Llama-3.2-3B maintains stable performance, with DSR slightly increasing from 99.3% to 99.5% and OR changing marginally from 3.3% to 3.4%.

When expanding to two new categories (+2), both models continue to maintain high DSR above 99% with only minor changes in OR. Under sequential expansion (+1+1), a small degradation appears for Llama-3B, where OR increases from 3.3% to 4.2%, while DSR remains stable at around 99%. Overall, these results demonstrate that incremental expansion causes minimal performance degradation, highlighting the modularity and scalability of the proposed approach for dynamically evolving safety requirements.

## 4 Related Work

### 4.1 Safety Alignment

**Table 3: Incremental category expansion results. “+1” and “+2” indicate adding one or two new categories to the initial set, while “+1+1” denotes sequentially adding one category at a time.**
| Model | Setting | DSR (%) | OR (%) |
| --- | --- | --- | --- |
| Llama-8B | Initial (2 categories) | 99.8 | 4.3 |
| +1 Category | 99.4 | 2.1 |  |
| +2 Categories | 99.6 | 2.3 |  |
| +1 +1 Categories | 99.8 | 2.8 |  |
| Llama-3B | Initial (2 categories) | 99.3 | 3.3 |
| +1 Category | 99.5 | 3.4 |  |
| +2 Categories | 99.4 | 3.6 |  |
| +1 +1 Categories | 99.3 | 4.2 |  |

Safety alignment is essential for deploying LLMs in real-world applications, and most approaches rely on post-training methods such as SFT-based preference optimization and RLHF. RLHF trains a reward model from human preference annotations and then optimizes the LLM via reinforcement learning to maximize the learned reward signal Ouyang et al. (2022); Lee et al. (2023); Yuan et al. (2023). In contrast, SFT-based methods directly optimize preference data by increasing the likelihood of preferred responses while suppressing undesirable ones Khaki et al. (2024); Meng et al. (2024). Recent work further improves robustness by introducing regularized fine-tuning objectives that restrict updates on initial tokens to make safety alignment more resistant to attacks Qi et al..

Another line of research studies safety alignment through red-team attacks and defenses. Red-team methods aim to bypass safety mechanisms by manipulating prompts or exploiting weaknesses in alignment data Peng et al. (2025a); Liu et al. (2023b); Peng et al. (2025b), while defenses typically rely on post-training strategies or external guardrail models to detect and block unsafe outputs Inan et al. (2023).

However, safety standards vary across regions and user groups, and repeatedly re-aligning models for each context is computationally costly and can harm general utility due to distributional shifts; in contrast, MOSAIC provides lightweight token-level compositional control that enables flexible, context-dependent safety while preserving utility.

### 4.2 Pluralistic Alignment

Recent work increasingly recognizes that alignment for large language models (LLMs) should not follow a one-size-fits-all paradigm, but instead adapt to the diverse values and preferences of different user groups Sorensen et al. (2024); Lake et al. (2025); Yin et al. (2024); Zhang et al.; Jiang et al. (2025). Several approaches have been proposed for pluralistic value alignment  Sorensen et al. (2024); Lake et al. (2025). PICACO optimizes a meta-instruction to navigate multiple values, enabling in-context alignment without modifying model parameters Jiang et al. (2025). Modular Pluralism uses a base LLM together with specialized community models that interact in different modes to flexibly support multiple forms of pluralism Feng et al. (2024). Another line of work explores controllable alignment, where CoSA aligns models to user-specified safety configurations and allows safety requirements to be adjusted at inference time without retraining Zhang et al..

Prior work on pluralistic alignment mainly studies diverse value preferences in model outputs, with limited attention to pluralism in safety alignment, namely when LLMs should refuse harmful requests under different safety requirements. Among the limited related work,  Yin et al. (2024) propose a geographic-context benchmark for culturally and legally appropriate responses, while  Arif et al. (2025) introduce patch-based prompt prefixes to steer model behavior. However, neither study explores combinatorial safety control, limiting the ability to flexibly compose safety mechanisms across different requirements.

## 5 Conclusion

We present MOSAIC, a modular framework for conditional safety alignment in LLMs. By representing safety constraints as learnable control tokens over a frozen backbone and training them with order-based task sampling and counterfactual knowledge distillation, MOSAIC achieves strong refusal behavior while minimizing over-refusal. We also introduce a practical benchmark for evaluating conditional safety alignment.

## 6 Limitations

A primary limitation of our work lies in the constrained scale of our experimental evaluation and data resources. While we have validated MOSAIC across two mainstream and practical model families of varying sizes, we have yet to conduct experiments on significantly larger models due to hardware limitations. Additionally, to guarantee superior data quality, our dataset underwent intensive manual filtering. Due to the limited size of our research team, the resulting dataset size is relatively small, which might not capture the full breadth of diverse challenges and safety edge cases present in open-domain environments.

## References

- H. Arif, K. Murugesan, C. Ko, P. Chen, P. Das, and A. Gittens (2025)
Patching llm like software: a lightweight method for improving safety policy in large language models.
arXiv preprint arXiv:2511.08484.
Cited by: [§4.2](#S4.SS2.p2.1).
- S. Behrouzi, L. Wu, M. Rostami, and A. Sadeghi (2026)
NeST: neuron selective tuning for llm safety.
arXiv preprint arXiv:2602.16835.
Cited by: [§1](#S1.p3.1).
- H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma, et al. (2024)
Scaling instruction-finetuned language models.
Journal of Machine Learning Research 25 (70), pp. 1–53.
Cited by: [§1](#S1.p3.1).
- S. Feng, T. Sorensen, Y. Liu, J. Fisher, C. Y. Park, Y. Choi, and Y. Tsvetkov (2024)
Modular pluralism: pluralistic alignment via multi-llm collaboration.
In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
pp. 4151–4171.
Cited by: [§4.2](#S4.SS2.p1.1).
- A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. (2024)
The llama 3 herd of models.
arXiv preprint arXiv:2407.21783.
Cited by: [§3.2](#S3.SS2.p2.2).
- J. Guan, J. Wu, J. Li, C. Cheng, and W. Wu (2025)
A survey on personalized alignment—the missing piece for large language models in real-world applications.
In Findings of the Association for Computational Linguistics: ACL 2025,
pp. 5313–5333.
Cited by: [§1](#S1.p1.1),
[§1](#S1.p5.1).
- J. Hong, N. Lee, and J. Thorne (2024)
Orpo: monolithic preference optimization without reference model.
In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
pp. 11170–11189.
Cited by: [§3.4](#S3.SS4.SSS0.Px2.p1.1).
- H. Inan, K. Upasani, J. Chi, R. Rungta, K. Iyer, Y. Mao, M. Tontchev, Q. Hu, B. Fuller, D. Testuggine, et al. (2023)
Llama guard: llm-based input-output safeguard for human-ai conversations.
arXiv preprint arXiv:2312.06674.
Cited by: [§4.1](#S4.SS1.p2.1).
- J. Ji, M. Liu, J. Dai, X. Pan, C. Zhang, C. Bian, B. Chen, R. Sun, Y. Wang, and Y. Yang (2023)
Beavertails: towards improved safety alignment of llm via a human-preference dataset.
Advances in Neural Information Processing Systems 36, pp. 24678–24704.
Cited by: [§1](#S1.p9.1).
- H. Jiang, D. Zhu, Z. Wei, X. Yi, Z. Xiao, and X. Xie (2025)
Picaco: pluralistic in-context value alignment of llms via total correlation optimization.
arXiv preprint arXiv:2507.16679.
Cited by: [§4.2](#S4.SS2.p1.1).
- S. Khaki, J. Li, L. Ma, L. Yang, and P. Ramachandra (2024)
Rs-dpo: a hybrid rejection sampling and direct preference optimization method for alignment of large language models.
In Findings of the Association for Computational Linguistics: NAACL 2024,
pp. 1665–1680.
Cited by: [§4.1](#S4.SS1.p1.1).
- [12]
R. Kirk, I. Mediratta, C. Nalmpantis, J. Luketina, E. Hambro, E. Grefenstette, and R. Raileanu
Understanding the effects of rlhf on llm generalisation and diversity.
In The Twelfth International Conference on Learning Representations,
Cited by: [§1](#S1.p3.1).
- T. Lake, E. Choi, and G. Durrett (2025)
From distributional to overton pluralism: investigating large language model alignment.
In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),
pp. 6794–6814.
Cited by: [§4.2](#S4.SS2.p1.1).
- N. Lambert and R. Calandra (2023)
The alignment ceiling: objective mismatch in reinforcement learning from human feedback.
arXiv preprint arXiv:2311.00168.
Cited by: [§1](#S1.p3.1).
- H. Lee, S. Phatale, H. Mansoor, K. R. Lu, T. Mesnard, J. Ferret, C. Bishop, E. Hall, V. Carbune, and A. Rastogi (2023)
Rlaif: scaling reinforcement learning from human feedback with ai feedback.
Cited by: [§4.1](#S4.SS1.p1.1).
- N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang (2024)
Lost in the middle: how language models use long contexts.
Transactions of the association for computational linguistics 12, pp. 157–173.
Cited by: [§1](#S1.p4.1).
- P. Liu, W. Yuan, J. Fu, Z. Jiang, H. Hayashi, and G. Neubig (2023a)
Pre-train, prompt, and predict: a systematic survey of prompting methods in natural language processing.
ACM computing surveys 55 (9), pp. 1–35.
Cited by: [§1](#S1.p4.1).
- X. Liu, N. Xu, M. Chen, and C. Xiao (2023b)
Autodan: generating stealthy jailbreak prompts on aligned large language models.
arXiv preprint arXiv:2310.04451.
Cited by: [§4.1](#S4.SS1.p2.1).
- Y. Meng, M. Xia, and D. Chen (2024)
SimPO: simple preference optimization with a reference-free reward.
In Advances in Neural Information Processing Systems (NeurIPS),
Cited by: [§4.1](#S4.SS1.p1.1).
- R. OpenAI (2023)
Gpt-4 technical report. arxiv 2303.08774.
View in Article 2 (5), pp. 1.
Cited by: [§1](#S1.p4.1).
- L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. (2022)
Training language models to follow instructions with human feedback.
Advances in neural information processing systems 35, pp. 27730–27744.
Cited by: [§1](#S1.p3.1),
[§3.4](#S3.SS4.SSS0.Px3.p1.1),
[§4.1](#S4.SS1.p1.1).
- S. Panda, N. J. Nizar, and M. L. Wick (2024)
LLM improvement for jailbreak defense: analysis through the lens of over-refusal.
In Neurips Safe Generative AI Workshop 2024,
Cited by: [§3.3](#S3.SS3.p2.1).
- J. Peng, M. Wang, N. Wang, J. Li, Y. Li, Y. Ye, W. Wang, P. Jia, K. Zhang, and X. Zhao (2025a)
Logic jailbreak: efficiently unlocking llm safety restrictions through formal logical expression.
arXiv preprint arXiv:2505.13527.
Cited by: [§4.1](#S4.SS1.p2.1).
- J. Peng, M. Wang, X. Zhao, K. Zhang, W. Wang, P. Jia, Q. Liu, R. Guo, and Q. Liu (2025b)
Stepwise reasoning disruption attack of llms.
In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pp. 5040–5058.
Cited by: [§4.1](#S4.SS1.p2.1).
- E. Perez, S. Huang, F. Song, T. Cai, R. Ring, J. Aslanides, A. Glaese, N. McAleese, and G. Irving (2022)
Red teaming language models with language models.
In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing,
pp. 3419–3448.
Cited by: [§1](#S1.p9.1).
- A. K. Purba, R. M. Thomson, P. M. Henery, A. Pearce, M. Henderson, and S. V. Katikireddi (2023)
Social media use and health risk behaviours in young people: systematic review and meta-analysis.
bmj 383.
Cited by: [§1](#S1.p1.1),
[§1](#S1.p10.1),
[§3.1](#S3.SS1.p1.1).
- [27]
X. Qi, A. Panda, K. Lyu, X. Ma, S. Roy, A. Beirami, P. Mittal, and P. Henderson
Safety alignment should be made more than just a few tokens deep.
In The Thirteenth International Conference on Learning Representations,
Cited by: [§4.1](#S4.SS1.p1.1).
- H. Qiu, A. R. Fabbri, D. Agarwal, K. Huang, S. Tan, N. Peng, and C. Wu (2025)
Evaluating cultural and social awareness of llm web agents.
In Findings of the Association for Computational Linguistics: NAACL 2025,
pp. 3978–4005.
Cited by: [§1](#S1.p1.1).
- M. Russinovich, Y. Cai, K. Hines, G. Severi, B. Bullwinkel, and A. Salem (2026)
GRP-obliteration: unaligning llms with a single unlabeled prompt.
arXiv preprint arXiv:2602.06258.
Cited by: [§1](#S1.p9.1).
- H. A. Shairah, H. A. A. K. Hammoud, G. Turkiyyah, and B. Ghanem (2025)
Turning the spell around: lightweight alignment amplification via rank-one safety injection.
arXiv preprint arXiv:2508.20766.
Cited by: [§1](#S1.p9.1).
- A. Singh, A. Fry, A. Perelman, A. Tart, A. Ganesh, A. El-Kishky, A. McLaughlin, A. Low, A. Ostrow, A. Ananthram, et al. (2025)
Openai gpt-5 system card.
arXiv preprint arXiv:2601.03267.
Cited by: [§3.3](#S3.SS3.p1.1).
- T. Sorensen, L. Jiang, J. D. Hwang, S. Levine, V. Pyatkin, P. West, N. Dziri, X. Lu, K. Rao, C. Bhagavatula, et al. (2024)
Value kaleidoscope: engaging ai with pluralistic human values, rights, and duties.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 38, pp. 19937–19947.
Cited by: [§4.2](#S4.SS2.p1.1).
- L. Wang, X. Zhang, H. Su, and J. Zhu (2024)
A comprehensive survey of continual learning: theory, method and application.
IEEE transactions on pattern analysis and machine intelligence 46 (8), pp. 5362–5383.
Cited by: [§1](#S1.p3.1).
- D. Yin, H. Qiu, K. Huang, K. Chang, and N. Peng (2024)
Safeworld: geo-diverse safety alignment.
Advances in Neural Information Processing Systems 37, pp. 128734–128768.
Cited by: [§1](#S1.p1.1),
[§4.2](#S4.SS2.p1.1),
[§4.2](#S4.SS2.p2.1).
- H. Yuan, Z. Yuan, C. Tan, W. Wang, S. Huang, and F. Huang (2023)
Rrhf: rank responses to align language models with human feedback.
Advances in Neural Information Processing Systems 36, pp. 10935–10950.
Cited by: [§4.1](#S4.SS1.p1.1).
- Q. Zhan, R. Fang, R. Bindu, A. Gupta, T. B. Hashimoto, and D. Kang (2024)
Removing rlhf protections in gpt-4 via fine-tuning.
In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers),
pp. 681–687.
Cited by: [§1](#S1.p9.1).
- [37]
J. Zhang, A. Elgohary, A. Magooda, D. Khashabi, and B. Van Durme
Controllable safety alignment: inference-time adaptation to diverse safety requirements.
In The Thirteenth International Conference on Learning Representations,
Cited by: [§1](#S1.p9.1),
[§4.2](#S4.SS2.p1.1).
- J. Zhuo, S. Zhang, X. Fang, H. Duan, D. Lin, and K. Chen (2024)
ProSA: assessing and understanding the prompt sensitivity of llms.
In Findings of the Association for Computational Linguistics: EMNLP 2024,
pp. 1950–1976.
Cited by: [§1](#S1.p4.1).

## Appendix A Appendix

Figure: Figure 6: Performance of MOSAIC on Llama3.1-8B under different $\lambda$.
Refer to caption: https://arxiv.org/html/2603.16210/2603.16210v1/x6.png

### A.1 Parameter Analysis

We examine the effect of the hyperparameter $\lambda$ on the trade-off between defense effectiveness and model utility. As illustrated in Figure [6](#A1.F6), the DSR demonstrates remarkable robustness under both 2-order and 3-order compositional constraints, achieving near-complete suppression of unsafe requests for $\lambda$ values ranging from 0.4 to 1.6. In contrast, OR is more sensitive to the choice of $\lambda$, exhibiting a non-monotonic trend. Specifically, OR reaches its optimal (lowest) level when $\lambda$ lies between 0.8 and 1.2, and degrades when $\lambda$ deviates from this interval.

The sensitivity of OR can be attributed to the balance of alignment strength. When $\lambda$ is too small, the safety signal is insufficient, leading to unstable handling of prompts near the safety boundary. Conversely, an excessively large $\lambda$ causes the safety boundary to over-generalize, resulting in benign and compliant prompts being incorrectly classified as violations, which in turn increases the OR rate.

Notably, DSR remains consistently high as $\lambda$ increases, whereas OR varies significantly. One possible explanation is the asymmetry in task difficulty between refusal and utility preservation. While maintaining benign utility requires precise calibration of safety boundaries, categorical refusal is comparatively easier for the model to learn under modular control tokens. Once the alignment signal surpasses a minimal functional threshold, the refusal behavior can be reliably triggered. As a result, defense performance becomes less sensitive to further increases in $\lambda$, whereas preserving benign utility requires finer boundary control and is therefore more sensitive to the choice of $\lambda$.

### A.2 Prompt for Judgement

To evaluate the usefulness of model responses to benign user requests, we employ a Judge LLM to perform automated assessments. As shown in Figure [7](#A1.F7), the Judge LLM is prompted with both the user request and the corresponding model response. The evaluator is instructed to determine whether the response directly, clearly, and positively answers the user’s request. Specifically, the Judge LLM checks whether the response provides specific, actionable, or factual information, directly addresses the user’s question without refusal or evasion, and remains relevant and complete rather than vague or generic. The Judge LLM outputs a binary decision (“yes” or “no”), indicating whether the response satisfactorily answers the request.

Figure: Figure 7: Prompt for judgement
Refer to caption: https://arxiv.org/html/2603.16210/2603.16210v1/x7.png