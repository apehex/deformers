Title: There Is More to Refusal in Large Language Models than a Single Direction
ArXiv: 2602.02132
Authors: Faaiz Joad, Majd Hawasly, Sabri Boughorbel, Nadir Durrani, Husrev Taha Sencar
Sections: 55
Estimated tokens: 28.6k

## Contents
- 1 Introduction
- 2 Methods
  - 2.1 Identifying Refusal in Activation Space
  - 2.2 Identifying Refusal in SAE Space
    - Motivation.
    - SAE models and decision-state activations.
    - Latent firing and refusal separation.
    - Constructing SAE-based refusal directions.
    - Ablations and control baselines.
- 3 Data and Models
  - WildGuardMix (WGM)
  - SorryBench (SB)
  - CoCoNot (CCN)
  - XSTest (XST)
  - Models:
- 4 Findings
  - 4.1 Refusal Directions in Activation Space
    - Controlled test:
    - Steering:
  - 4.2 Refusal Directions in SAE Space
    - 4.2.1 Latent-Space Steering
    - 4.2.2 Alignment with Activation-Space Refusal Directions
    - 4.2.3 Latent Overlaps Across Dataset Splits
    - 4.2.4 Semantic Structure of Refusal Latents
- 5 Related Work
  - Refusal Directions and Abliteration Attacks:
  - Activation Steering and Representation Engineering:
  - Sparse Autoencoders for Interpretability:
- 6 Conclusion
- Limitations
- Potential Risks and Ethics Statement
- References
- Appendix A Refusal Style Examples Across Directions
- Appendix B Refusal splits
  - WildGuard-Mix (vanilla, non-adversarial).
  - XSTest.
  - CoCoNot.
  - SorryBench.
  - Balanced subsamples for direction learning.
- Appendix C Intra-class refusal direction geometry
- Appendix D Oracle experiments for refusal directions
  - Stability under subsampling.
  - Held-out “gold” oracles.
  - Stability under subsampling.
  - Held-out “gold” oracles.
  - Family-level structure.
  - D.1 Experiment 5: Hook Variants and Robustness
    - Goal.
    - Summary.
  - D.2 Experiment 6: Taxonomy of Refusal Latents
    - Goal.
    - Method (high level).
    - Findings.
- Appendix E Latent Semantic Annotation
- Appendix F Refusal Latent Overlap between Splits

## Abstract

Abstract Prior work argues that refusal in large language models is mediated by a single activation-space direction, enabling effective steering and abliteration. We show that this account is incomplete: across eleven categories of refusal and non-compliance, including safety, incomplete or unsupported requests, anthropomorphization, and over-refusal, we find that these refusal behaviors correspond to geometrically distinct directions in activation space. Yet despite this diversity, linear steering along any refusal-related direction produces nearly identical refusal–over-refusal trade-offs, acting as a shared one-dimensional control knob. The primary effect of different directions is not whether the model refuses, but how it refuses.
Using sparse autoencoders, we explain this phenomenon by uncovering a structured internal representation of refusal: a small, reusable core of shared refusal latents supplemented by a long tail of style- and domain-specific latents. Linear interventions collapse over this structure, flattening mechanistic differences into uniform behavioral control. Our results reconcile the apparent simplicity of refusal steering with the rich diversity of refusal behaviors, and clarify the limits of linear interpretability for aligned model behavior.

## 1 Introduction

A central objective in training large language models (LLMs) is to align their behavior with human values such as helpfulness and harmlessness, while accurately interpreting and responding to user intent as expressed in a prompt Hendrycks et al. (2020); Askell et al. (2021); Bai et al. (2022); Yao et al. (2023). Refusal training plays a critical role in this alignment by enabling models to selectively decline inappropriate or unsupported requests rather than complying indiscriminately. Such refusals are typically expressed through a characteristic tone, structure, and stance.

Prior work on instilling refusal behavior has primarily emphasized safety, focusing on preventing harmful or disallowed outputs Ganguli et al. (2022); Dai et al. (2023); Zou et al. (2024b); Yuan et al. (2025). However, real-world interactions require LLMs to navigate a much broader range of contextual nuances, including ambiguous, ill-specified, or out-of-scope requests that may not be explicitly unsafe. Given the fragility of alignment mechanisms in LLMs, recent research has increasingly focused on developing methods to analyze and detect alignment failures, as well as to steer model behavior toward appropriate non-compliance when warranted Bai et al. (2023); Han et al. (2024a).

Figure: Figure 1: styles of refusals produced by steering along four distinct refusal directions.
Refer to caption: https://arxiv.org/html/2602.02132/figures/teaser.jpg

To better understand how refusal behavior arises in LLMs when responding to harmful requests, recent work has identified a remarkably simple underlying mechanism: a single linear direction in the model’s internal activation space that consistently mediates refusal Arditi et al. (2022). Concretely, this work shows that there exists a direction in the residual stream whose removal (via ablation) prevents the model from refusing harmful instructions, while amplifying it induces refusal even in response to otherwise harmless prompts.

At the same time, refusal behavior does not appear simple at the surface level. Models refuse for different reasons, across different domains, and in noticeably different styles, see Fig. [1](https://arxiv.org/html/2602.02132v1#S1.F1). This creates a tension between the apparent simplicity of the proposed mechanism and the diversity of refusal behaviors observed in practice. In particular, the proposed mechanism collapses diverse safety violations into a single behavioral class and does not address whether it generalizes to other forms of refusal beyond safety. This raises a critical open question: do qualitatively different forms of refusal—and more general non-compliance behaviors—rely on distinct underlying mechanisms?

To address this question, we adopt a broader view of refusal as part of a wider class of non-compliant behaviors that extend beyond safety alone Han et al. (2024b); Brahman et al. (2024); Xie et al. (2025); Röttger et al. (2024). Building on prior work, we consider refusals arising from incomplete or indeterminate requests, unsupported requests that exceed the model’s capabilities, humanizing requests that anthropomorphize the model, unqualified professional advice, potentially inappropriate topics governed by contextual norms, traditional safety refusals involving harmful content, and over-refusals to benign requests. Together, these categories capture a broad spectrum of refusal and non-compliance behaviors observed in real-world interactions. We study these behaviors through linear interventions in activation space and find that their associated directions are not geometrically identical, suggesting potential differences in how refusal behavior is expressed.

Despite these geometric differences, we find that linear steering along any of these directions yields similar refusal efficacy across model inputs: as steering strength increases, the model increasingly refuses prompts—including benign ones—across datasets.
Steering thus behaves like a single control knob: turning it up pushes the model toward refusal more often, largely independent of which refusal-related direction is used. This suggests a different understanding of refusal and non-compliance in the model’s internal dynamics than one implied by geometrically distinct directions. Where these directions differ most is not in whether the model refuses, but in how it refuses, ranging from explicit policy-based refusals, to clarification or missing-context responses, to softer forms of deflection.

To further examine the mechanistic origins of these differences, we use sparse autoencoders (SAEs) as an interpretability tool. We probe the model for latents that systematically activate during refusal and non-compliance at a fixed token position, and we test their causal involvement by amplifying the directions encoded by these latents during generation. This confirms the same single–behavioral-degree-of-freedom phenomenon observed with activation-space steering. For each refusal category, we verify that the residual-stream directions encoded by the corresponding latents resemble those identified directly in activation space.

By analyzing the overlap of active latents across refusal categories, we find that while a substantial subset of latents is shared across refusal types—forming a reusable refusal core—other latents are specific to particular refusal styles. To make this distinction concrete, we annotate a subset of the most widely shared latents and show that stylistic variation in refusal arises from a long tail of more specialized features.

Overall, the internal structure of refusal is largely invisible to linear control. Activation-space steering reveals what refusal looks like under intervention, while SAE analysis helps explain why different steering directions behave so similarly and which internal features are being jointly modulated. Together, these perspectives describe different levels of the same phenomenon: a structured internal system whose complexity is flattened by simple linear interventions.

We address in this work the following research questions:
We address in this work the following research questions:

- RQ1
How different are the refusal directions when computed for different types of refusal behavior beyond safety?
- RQ2
How do these different directions get expressed in the latent space of the model such that they facilitate various styles of refusal under a very similar intervention mechanism?
- RQ3
Do geometrically distinct refusal directions lead to meaningfully different refusal behavior when used for causal steering?
- RQ4
Which aspects of refusal behavior are controlled by linear interventions, and which aspects vary across refusal directions?
- RQ5
To what extent are refusal-related latent features shared across different refusal categories?

## 2 Methods

### 2.1 Identifying Refusal in Activation Space

A refusal direction is computed by contrasting residual stream activations elicited by prompts labeled to require non-compliance with those elicited by benign prompts. For each prompt set, we collect residual stream activations at a fixed token position, average them across prompts, and define the refusal direction as the difference between these activation vectors. This procedure yields a single activation-space direction per evaluation split, which we use for steering and ablation experiments.

Let $x_{t}\in\mathbb{R}^{d}$ denote the residual stream activation at token position $t$, and let $r\in\mathbb{R}^{d}$ denote the normalized refusal direction. Refusal *induction* is implemented by modifying activations as $x_{t}^{\prime}\leftarrow x_{t}+\alpha r$, where $\alpha>0$ controls the steering strength. Refusal *ablation* is implemented by projecting out this component, $x_{t}^{\prime}\leftarrow x_{t}-(x_{t}\cdot r)\,r$. These linear interventions enable direct causal tests by selectively amplifying or suppressing refusal behavior during generation, with their effects quantified via changes in refusal rate, over-refusal rate, and overall accuracy.

For each model, we select the steering strength $\alpha$ by a simple grid
search over a small discrete set (e.g., $\alpha\in\{5,10,20,30,60\}$)
on a held-out validation pool, choosing the smallest value that achieves at
least a 90% refusal rate on harmful prompts while keeping benign
over-refusals below a chosen threshold. Unless otherwise noted, we report
results for these selected $\alpha$ values (e.g., $\alpha=60$ for
gemma-2-9b-it in activation-space steering).

### 2.2 Identifying Refusal in SAE Space

##### Motivation.

We use sparse autoencoders (SAEs) as an interpretability tool to decompose the refusal directions identified in the activation space to their sparse latent components.
We follow the firing–rate separation approach of Ferrando et al. (2024) to identify refusal latents, test their causal involvement, and study how the refusal directions are realized in the model’s latent space.

##### SAE models and decision-state activations.

We use pretrained JumpReLU residual-stream SAEs from GemmaScope Lieberum et al. (2024a), trained on gemma-2-9B-it and available for layers $\ell\in\{9,20,31\}$(^1^11hf.co/google/gemma-scope-9b-it-res).
For all SAE analyses, we hook the residual stream at the token immediately preceding the assistant’s response (position $-2$ in the chat template), which we treat as the model’s *decision state*.
Let $x^{(\ell)}\in\mathbb{R}^{d}$ denote this residual-stream activation at layer $\ell$.
An SAE parameterizes an encoder–decoder pair $(f_{\theta},g_{\phi})$ such that

$$ $z^{(\ell)}=f_{\theta}\!\big(x^{(\ell)}\big),\qquad\hat{x}^{(\ell)}=g_{\phi}\!\big(z^{(\ell)}\big),$ (1) $$

where $z^{(\ell)}\in\mathbb{R}^{k}$ is a sparse latent vector and each latent $j$ has an associated decoder direction $d^{(\ell)}_{j}$ in the residual space.

##### Latent firing and refusal separation.

To identify refusal-associated latents, we adapt the firing-rate separation method of Ferrando et al. (2024).
For latent $j$ at layer $\ell$ on example $i$, we define a binary firing indicator:

$$ $a^{(\ell)}_{ij}=\mathbbm{1}\!\left[z^{(\ell)}_{ij}>0\right],$ (2) $$

where $z^{(\ell)}_{ij}$ is the JumpReLU activation of latent $j$.
For each of our 11 refusal splits $s$ and label $y\in\{\text{refusal},\text{non-refusal}\}$ with index set $S^{(s)}_{y}$, we compute the firing rate:

$$ $f^{(s)}_{j,\ell}(y)=\frac{1}{\lvert S^{(s)}_{y}\rvert}\sum_{i\in S^{(s)}_{y}}a^{(\ell)}_{ij},$ (3) $$

and define the *refusal separation score as:*

$$ $\Delta^{(s)}_{j,\ell}=f^{(s)}_{j,\ell}(\text{refusal})-f^{(s)}_{j,\ell}(\text{non-refusal}).$ (4) $$

For each split $s$ and layer $\ell\in\{9,20,31\}$, we rank latents by $\Delta^{(s)}_{j,\ell}$ and select the top-$K$ with $\Delta^{(s)}_{j,\ell}>0$ as the *refusal latents* for that split.
These layerwise latent sets are the basic building blocks for all subsequent SAE analyses (causal steering, ablations, and cross-split reuse).

##### Constructing SAE-based refusal directions.

Each refusal latent $j$ has a decoder direction $d^{(\ell)}_{j}$ in residual space.
For split $s$ and layer $\ell$, with refusal latents $\{j_{1},\dots,j_{K}\}$, we construct an SAE-based refusal direction by averaging decoder directions:

$$ $d^{(s,\ell)}_{\text{SAE}}=\frac{1}{K}\sum_{k=1}^{K}d^{(\ell)}_{j_{k}}.$ (5) $$

During generation, we apply steering at layer $\ell$ by modifying the residual stream as

$$ $x^{\prime}=x+\beta\,d^{(s,\ell)}_{\text{SAE}},$ (6) $$

where $\beta$ controls steering strength.
This procedure mirrors activation-space steering, but the direction is explicitly grounded in a small bundle of SAE features rather than a single mean-difference vector.

##### Ablations and control baselines.

To test necessity, we perform SAE-based ablations by encoding the decision-state activation $x^{(\ell)}$, zeroing the selected refusal latents in $z^{(\ell)}$, and decoding back into residual space to obtain an ablated activation $\tilde{x}^{(\ell)}=g_{\phi}(\tilde{z}^{(\ell)})$ that is used in place of $x^{(\ell)}$ during generation.
As controls, we construct steering directions from random latent subsets of the same size $K$, and from random unit vectors in residual space.

## 3 Data and Models

To capture a broad range of refusal and non-compliance behaviors, we draw on four datasets with complementary characteristics, from which we construct 11 evaluation splits.

##### WildGuardMix (WGM)

provides ground-truth safety annotations indicating whether prompts should be complied with or declined under safety policies; from this dataset we construct the SafetyCore–WGM split containing both safety-sensitive and benign prompts.

##### SorryBench (SB)

organizes unsafe content into four policy domains, from which we construct four splits corresponding to hate speech generation (HateSpeech–SB), potentially inappropriate topics (Inappropriate–SB), assistance with crimes or torts (CrimeAssistance–SB), and unqualified professional advice (Advice–SB).

##### CoCoNot (CCN)

covers contextual non-compliance beyond conventional safety cases; we construct five splits capturing incomplete requests (Incomplete–CCN), unsupported requests (Unsupported–CCN), indeterminate requests (Indeterminate–CCN), humanizing requests (Humanizing–CCN), and safety-related requests involving harmful content (Safety–CCN).

##### XSTest (XST)

targets over-refusal by contrasting adversarially framed but benign prompts with standard benign requests; we construct the OverRefusal–XST split to study inappropriate refusal on safe inputs.
Additional dataset details are provided in Appendix [B](https://arxiv.org/html/2602.02132v1#A2).

Each evaluation split consists of a balanced pair of prompt sets: at least 32 prompts labeled to elicit non-compliant behavior under dataset annotations, paired with an equal number of benign prompts for which compliance is appropriate. Following  Arditi et al. (2022), we adopt 32 prompts per class as a standard unit for estimating stable activation centroids. When sufficient data is available, we additionally construct multiple independent 32/32 splits without replacement, enabling validation of within-category geometric stability.

##### Models:

We run all experiments on two instruction-tuned models: gemma-2-9b-it(^2^22hf.co/google/gemma-2-9b-it) and Llama-3.1-8B-Instruct(^3^33hf.co/meta-llama/Llama-3.1-8B-Instruct). Activation-space directions are computed from residual-stream activations at a fixed mid-layer hook (resid_pre; e.g., layer 20 for Gemma and layers 15–16 for Llama). For sparse-feature analyses, we use publicly available residual-stream SAEs (GemmaScope layers 9/20/31 and the saes-llama-3.1-8b-instruct release(^4^44hf.co/andyrdt/saes-llama-3.1-8b-instruct)), focusing on later layers where refusal-related features are most pronounced. Activations are extracted at the chat-template token immediately preceding the assistant’s response (index $-2$), which we treat as the model’s decision state.

## 4 Findings

### 4.1 Refusal Directions in Activation Space

In this section, we analyze refusal behavior directly in activation space. For each of the 11 refusal splits, we compute a corresponding refusal direction and quantify similarity across splits using pairwise cosine distance. As shown in Table [1](https://arxiv.org/html/2602.02132v1#S4.T1), many directions are substantially dissimilar (typical cosine similarity 0.4–0.6, with several near-orthogonal), indicating that different refusal categories correspond to distinct activation-space directions.

**Table 1: Pairwise cosine similarity matrix between refusal directions learned from the 11 evaluation splits. The direction of SafetyCore-WGM is the closest to the safety refusal direction of Arditi et al. (2022). WGM: WildGuard-Mix, CCN: CoCoNot, SB: SorryBench, XST: XSTest**
|  | SafetyCore<br>WGM | OverRefusal<br>XST | Humanizing<br>CCN | Incomplete<br>CCN | Indeterminate<br>CCN | Safety<br>CCN | Unsupported<br>CCN | HateSpeech<br>SB | CrimeAssist.<br>SB | Inappropriate<br>SB | Advice<br>SB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SafetyCore-WGM | 1.000 | 0.632 | 0.234 | 0.127 | 0.310 | 0.663 | 0.297 | 0.695 | 0.734 | 0.649 | 0.511 |
| OverRefusal-XST | 0.632 | 1.000 | 0.239 | -0.062 | 0.203 | 0.548 | 0.099 | 0.609 | 0.715 | 0.390 | 0.426 |
| Humanizing-CCN | 0.234 | 0.239 | 1.000 | 0.460 | 0.678 | 0.586 | 0.620 | 0.417 | 0.411 | 0.328 | 0.474 |
| Incomplete-CCN | 0.127 | -0.062 | 0.460 | 1.000 | 0.688 | 0.550 | 0.627 | 0.415 | 0.402 | 0.379 | 0.468 |
| Indeterminite-CCN | 0.310 | 0.203 | 0.678 | 0.688 | 1.000 | 0.719 | 0.795 | 0.560 | 0.573 | 0.502 | 0.666 |
| Safety-CCN | 0.663 | 0.548 | 0.586 | 0.550 | 0.719 | 1.000 | 0.692 | 0.876 | 0.885 | 0.822 | 0.811 |
| Unsupported-CCN | 0.297 | 0.099 | 0.620 | 0.627 | 0.795 | 0.692 | 1.000 | 0.499 | 0.480 | 0.520 | 0.619 |
| HateSpeech-SB | 0.695 | 0.609 | 0.417 | 0.415 | 0.560 | 0.876 | 0.499 | 1.000 | 0.917 | 0.862 | 0.746 |
| CrimeAssistance-SB | 0.734 | 0.715 | 0.411 | 0.402 | 0.573 | 0.885 | 0.480 | 0.917 | 1.000 | 0.809 | 0.795 |
| Inappropriate-SB | 0.649 | 0.390 | 0.328 | 0.379 | 0.502 | 0.822 | 0.520 | 0.862 | 0.809 | 1.000 | 0.732 |
| Advice-SB | 0.511 | 0.427 | 0.474 | 0.468 | 0.666 | 0.811 | 0.619 | 0.746 | 0.795 | 0.732 | 1.000 |

##### Controlled test:

To evaluate whether these directions causally mediate refusal, we use a controlled test set of 200 prompts balanced across prompt type (harmful vs. benign) and the unsteered model’s response (refusal vs. compliance). This yields four equally sized subsets: HR, HC, BR, and BC. By construction, the unsteered model achieves 50% accuracy, since only refusals to harmful prompts (HR) and compliance with benign prompts (BC) are correct outcomes; HC and BR correspond to jailbreaks and over-refusals, respectively.

##### Steering:

We steer the base model along each of the 11 refusal directions and evaluate performance on the controlled test set. Steering is applied at various strengths, ranging from weak to near-saturated refusal regimes as in Table [12](https://arxiv.org/html/2602.02132v1#A4.T12). Steered responses are labeled as refusals or compliant answers using WildGuard, with manual validation. We report overall accuracy, refusal rate, and over-refusal rate.

Table [2](https://arxiv.org/html/2602.02132v1#S4.T2) reports summary results for steering and direction ablation of a safety-derived direction for Gemma and Llama. Across splits and models, steering along different refusal directions produces broadly similar behavior: increasing steering strength raises refusal rates on harmful prompts while simultaneously increasing over-refusal on benign prompts, with both approaching saturation at high strength. A notable exception is CoCoNot non-safety categories which respond more gradually to steering. Direction ablation for Gemma shows the complementary effect, suppressing refusal in most cases, though CoCoNot-derived directions leave residual refusal rates of approximately 0.3–0.5, suggesting partial misalignment with safety refusal mechanisms. Also, Llama does not respond well to the safety direction ablation, suggesting a richer internal refusal structure with redundant paths that survived the ablation.

Despite these geometric differences, steered models consistently refuse to answer prompted questions, differing primarily in *how* refusal is expressed. As illustrated in Appendix [A](https://arxiv.org/html/2602.02132v1#A1), humanizing directions emphasize the model’s non-human nature, incomplete-request directions produce terse expressions of confusion, and indeterminate or unsupported directions stress impossibility or lack of agency. SorryBench directions foreground moral judgments or professional disclaimers, while WildGuard and XSTest directions emphasize safety, illegality, or harm.

Overall, these results indicate that refusal performance is largely invariant across directions, while refusal *style* varies substantially, consistent with refusal acting as a simple behavioral switch with stylistic degrees of freedom.

**Table 2: Performance of Gemma and Llama models steered to refuse, or ablated to comply, across 11 refusal directions on the controlled test set, reported in terms of accuracy: $Acc=(HR^{\prime}+BC^{\prime})/200$, refusal rate: $RR=HR^{\prime}/100$, and over-refusal rate: $ORR=BR^{\prime}/100$, where $H$ and $B$ refer to harmful and benign prompts, respectively, and $R^{\prime}$ and $C^{\prime}$ refer to the WildGuard judgment of whether the steered response is a refusal or a compliance, respectively. The unsteered base models attain 50% on all three metrics.**
| Split | Gemma (steered, $\alpha=100$) | Llama (steered, $\alpha=2.5$) | Gemma (ablated) | Llama (ablated) |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Acc | RR | ORR | Acc | RR | ORR | Acc | RR | ORR | Acc | RR | ORR |  |
| Humanizing-CCN | 0.515 | 1.00 | 0.97 | 0.515 | 0.880 | 0.850 | 0.723 | 0.45 | 0.00 | 0.819 | 0.64 | 0.00 |
| Incomplete-CCN | 0.550 | 0.98 | 0.88 | 0.475 | 0.650 | 0.700 | 0.670 | 0.34 | 0.00 | 0.650 | 0.46 | 0.16 |
| Indeterminate-CCN | 0.545 | 0.99 | 0.90 | 0.520 | 0.970 | 0.930 | 0.713 | 0.45 | 0.02 | 0.610 | 0.40 | 0.18 |
| Safety-CCN | 0.500 | 1.00 | 1.00 | 0.510 | 1.000 | 0.980 | 0.574 | 0.15 | 0.00 | 0.595 | 0.48 | 0.29 |
| Unsupported-CCN | 0.515 | 0.99 | 0.96 | 0.510 | 0.920 | 0.900 | 0.723 | 0.45 | 0.00 | 0.585 | 0.48 | 0.31 |
| CrimeAssistance-SB | 0.500 | 1.00 | 1.00 | 0.500 | 1.000 | 1.000 | 0.500 | 0.00 | 0.00 | 0.510 | 0.42 | 0.40 |
| HateSpeech-SB | 0.500 | 1.00 | 1.00 | 0.500 | 1.000 | 1.000 | 0.500 | 0.00 | 0.00 | 0.525 | 0.43 | 0.38 |
| Inappropriate-SB | 0.500 | 1.00 | 1.00 | 0.505 | 0.990 | 0.980 | 0.500 | 0.00 | 0.00 | 0.575 | 0.48 | 0.33 |
| Advice-SB | 0.500 | 1.00 | 1.00 | 0.500 | 1.000 | 1.000 | 0.532 | 0.06 | 0.00 | 0.545 | 0.35 | 0.26 |
| SafetyCore-WGM | 0.500 | 1.00 | 1.00 | 0.495 | 0.990 | 1.000 | 0.553 | 0.11 | 0.00 | 0.515 | 0.38 | 0.35 |
| OverRefusal-XST | 0.510 | 1.00 | 0.98 | 0.500 | 1.000 | 1.000 | 0.521 | 0.04 | 0.00 | 0.570 | 0.63 | 0.49 |

### 4.2 Refusal Directions in SAE Space

To probe refusal behavior at the level of internal features, we analyze refusal in the latent space of sparse autoencoders (SAEs). For the 11 data splits, we identify refusal-associated SAE latents following Section [2.2](https://arxiv.org/html/2602.02132v1#S2.SS2) and examine: (i) their behavior under causal intervention, (ii) their alignment with the activation-space refusal directions from Section [4.1](https://arxiv.org/html/2602.02132v1#S4.SS1), (iii) the extent to which refusal latents are reused across datasets, splits, and layers, and (iv) the semantic themes these latents encode.

#### 4.2.1 Latent-Space Steering

We test whether the refusal-associated SAE latents are *causally involved* in refusal behavior, by asking whether steering in latent space recovers the same one-dimensional refusal knob observed in activation space.
For each refusal split, we construct a single SAE-based refusal direction by averaging the decoder directions of the top-ranked refusal latents (Section [2.2](https://arxiv.org/html/2602.02132v1#S2.SS2)), then apply residual-stream steering for the same controlled test set from Section [4.1](https://arxiv.org/html/2602.02132v1#S4.SS1) during generation at layer $\ell$,
$r^{\prime}=r+\beta\,d^{(s,\ell)}_{\text{SAE}}.$ We choose the steering strength $\beta$ using a coarse grid search
on the validation pool (Gemma: $\beta\in\{10,30,60\}$;
Llama: $\beta\in\{0.5,0.8,1.0,1.2\}$).
WildGuard is used again to label steered refusals $R^{\prime}$ or compliance $C^{\prime}$.

Table [3](https://arxiv.org/html/2602.02132v1#S4.T3) summarizes the performance of SAE-based steering across splits for Gemma and Llama (the underlying safety-tuned base model at $\beta=0$ already exhibits substantial refusal and over-refusal).
Across all splits, increasing $\beta$ monotonically raises refusal rates on harmful prompts and, in parallel, over-refusal rates on benign prompts, tracing out essentially the same accuracy–over-refusal trade-off observed when steering directly in activation space.
This shows that steering along a small SAE-derived direction is sufficient to recover the same one-dimensional refusal knob.

Control experiments using a random subset of SAE latents confirm that this behavior is specific to the identified refusal latents, produceing much weaker, non-monotonic changes in refusal.

**Table 3: Performance of Gemma and Llama models steered using a single SAE refusal direction on the controlled test set, reported in terms of overall accuracy (Acc $=(\text{HR}^{\prime}+\text{BC}^{\prime})/200$), refusal rate (RR $=\text{HR}^{\prime}/100$), and over-refusal rate (ORR $=\text{BR}^{\prime}/100$), where $R^{\prime}$ and $C^{\prime}$ refer to the WildGuard judgement of whether the steered response is a refusal or a compliance, respectively. For each model we select the strongest steering setting that most aggressively induces refusal (Gemma: $\beta=60$; Llama: $\beta=1.2$).**
| Split | Gemma (SAE-steered, $\beta=60$) | Llama (SAE-steered, $\beta=1.2$) |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Acc | RR | ORR | Acc | RR | ORR |  |
| Humanizing-CCN | 0.495 | 0.94 | 0.95 | 0.435 | 0.680 | 0.810 |
| Incomplete-CCN | 0.540 | 0.89 | 0.81 | 0.410 | 0.770 | 0.950 |
| Indeterminate-CCN | 0.560 | 0.96 | 0.84 | 0.440 | 0.880 | 1.000 |
| Safety-CCN | 0.500 | 1.00 | 1.00 | 0.470 | 0.940 | 1.000 |
| Unsupported-CCN | 0.530 | 0.71 | 0.65 | 0.495 | 0.970 | 0.980 |
| CrimeAssistance-SB | 0.515 | 1.00 | 0.97 | 0.305 | 0.590 | 0.980 |
| HateSpeech-SB | 0.535 | 1.00 | 0.93 | 0.450 | 0.860 | 0.960 |
| Inappropriate-SB | 0.525 | 1.00 | 0.95 | 0.480 | 0.950 | 0.990 |
| Advice-SB | 0.555 | 0.97 | 0.86 | 0.415 | 0.830 | 1.000 |
| SafetyCore-WGM | 0.500 | 1.00 | 1.00 | 0.470 | 0.940 | 1.000 |
| OverRefusal-XST | 0.500 | 1.00 | 1.00 | 0.510 | 1.000 | 0.980 |

#### 4.2.2 Alignment with Activation-Space Refusal Directions

To quantify if activation-space refusal directions can be reconstructed from the small set of refusal-associated SAE latents, for each prompt we obtain the SAE latent activations at the decision state $x^{(\ell)}$, filter for the top-ranked refusal latents (Section [2.2](https://arxiv.org/html/2602.02132v1#S2.SS2)), reconstruct their contribution in residual space by summing the corresponding decoder directions weighted by the latent activations, and compute the cosine similarity between this reconstructed direction and the corresponding activation-space direction. We report the mean and standard deviation of the cosine similarity over prompts for each split in Table [4](https://arxiv.org/html/2602.02132v1#S4.T4). Cosine similarity between SAE directions and activation-space directions is uniformly high across datasets and refusal categories, typically in the range $0.85$–$0.97$, indicating good reconstruction. Despite the fact that activation-space refusal directions are geometrically distinct across datasets (Section [4.1](https://arxiv.org/html/2602.02132v1#S4.SS1)), they appear to operate on substantially shared underlying SAE features; this might help explain why steering along different activation-space refusal directions produces similar refusal efficacy.

**Table 4: Mean cosine similarity and standard deviation between the SAE-induced refusal direction and the activation-space refusal direction, computed separately for each evaluation split. For each prompt, we reconstruct their contribution in activation space by an activation-weighted sum of the latents decoder rows.**
| Split | CosSim $\mu$ | CosSim $\sigma$ |
| --- | --- | --- |
| Humanizing-CCN | 0.971 | 0.039 |
| Incomplete-CCN | 0.853 | 0.096 |
| Indeterminate-CCN | 0.898 | 0.076 |
| Safety-CCN | 0.849 | 0.209 |
| Unsupported-CCN | 0.886 | 0.066 |
| CrimeAssistance-SB | 0.952 | 0.033 |
| HateSpeech-SB | 0.925 | 0.052 |
| Inappropriate-SB | 0.945 | 0.035 |
| Advice-SB | 0.928 | 0.037 |
| SafetyCore-WGM | 0.918 | 0.054 |
| OverRefusal-XST | 0.969 | 0.017 |

#### 4.2.3 Latent Overlaps Across Dataset Splits

To assess how much of the SAE refusal structure is shared across datasets and splits, we rank latents within each split by the difference in their firing rates on high-risk (HR) versus base (BC) subsets and retain the top 1000 latents per split.

For each layer, we count (i) how many distinct latents appear in at least one of these lists, and (ii) the latents that appear in for *all* the 11 splits. These overlap statistics are summarized in Table [5](https://arxiv.org/html/2602.02132v1#S4.T5), while detailed latent–split incidence matrices and top-shared-latent tables are given in Appendix [F](https://arxiv.org/html/2602.02132v1#A6).

As the results show, the signal is *sparse but consistent*: in all three layers, a relatively small subset of latents carries most of the HR vs. BC discrimination signal with only about $8\text{--}10\%$ and $2.5\text{--}4\%$ of the $16{,}384$ latents ever appear in any top–1000 list or in all splits, respectively.

Secondly, there are *robust cores* of refusal latents within each layer with strict intersections of 591, 517, and 421 latents for layers 9, 20, and 31, respectively.

Third, there is a *depth trend*: the fraction of latents that ever appear in a top–1000 list grows slightly with depth (from $\sim$8.55% at layer 9 to $\sim$9.64% at layer 31), while the size of the strict core shrinks (from $\sim$3.61% to $\sim$2.57%).
This suggests that deeper layers distribute the HR vs. BC signal over a somewhat larger basin, with a smaller, more specialized subset that is truly universal across splits.

Viewed together, we hypothesize a two-part structure:
a small, reusable *core* of refusal features shared across tasks, and a larger *long tail* of latents that appear only in one or few splits and are more aligned with particular refusal styles (e.g., short “I don’t understand” replies) or domains (e.g., repeated legal disclaimers in unqualified advice).

**Table 5: Overlap of top–1000 latents across splits (11 lists). Values are percentages of the full latent space (16,384 latents). “$\geq$1” counts latents appearing in at least one list; “All” denotes the strict intersection.**
| Layer | $\geq$1 | All |
| --- | --- | --- |
| 9 | 8.55% | 3.61% |
| 20 | 8.76% | 3.16% |
| 31 | 9.64% | 2.57% |

#### 4.2.4 Semantic Structure of Refusal Latents

Comparing the top-10 latents across settings using the semantic annotation pipeline described in the Appendix [E](https://arxiv.org/html/2602.02132v1#A5) reveals a set of common latents that consistently emerge regardless of dataset composition, representing domain-general refusal features such as harmful content requests, prohibited/illicit content, disinformation, and policy-violating directives (Table [15](https://arxiv.org/html/2602.02132v1#A5.T15)). In contrast, distinct latents emerge only in specific subcategories: SorryBench’s harm-focused splits surface latents specialized for operational harm detection (e.g., ridicule-focused harassment, coercive communications, crime facilitation), while CoCoNot’s capability-focused categories reveal latents for unsupported modalities (e.g., visual art requests, audio transcription, fictional language translation) and anthropomorphic queries (e.g., personalized advice requests). Several common latents also exhibit polysemy across settings, receiving harm-focused interpretations in SorryBench but capability-focused interpretations in CoCoNot.

## 5 Related Work

##### Refusal Directions and Abliteration Attacks:

Prior work has shown that refusal behavior in large language models can be strongly influenced by low-dimensional structure in activation space. Arditi et al. (2024) demonstrated that refusal is often mediated by a dominant direction in the residual stream, such that suppressing this direction inhibits refusal on harmful prompts while amplifying it induces refusal even for benign inputs. This finding led to the development of *abliteration* attacks (Labonne, 2024), which exploit directional ablation to remove refusal behavior from aligned models, with thousands of such modified models now publicly available. Subsequent work has questioned the sufficiency of a single-direction account: Zhang and Sun (2025) decomposed refusal into distinct harm-detection and refusal-execution directions, achieving high attack success rates through targeted interventions, while Wollschläger et al. (2025) showed that refusal can be mediated by multiple independent directions forming “concept cones,” introducing the notion of representational independence to explain diverse intervention effects. Complementary mechanistic analyses by Kissane et al. (2024) further reveal that base models already exhibit substantial refusal behavior and that refusal directions transfer from instruction-tuned models, suggesting that alignment fine-tuning amplifies preexisting circuitry rather than introducing refusal mechanisms from scratch. Together, these findings motivate a view of refusal as a structured but non-monolithic phenomenon, in which multiple activation-space directions can influence refusal behavior, potentially in different ways.

##### Activation Steering and Representation Engineering:

Our activation-space methodology builds on representation engineering (Zou et al., 2023), which introduced contrastive extraction of concept directions for behavioral control. Activation Addition (Turner et al., 2024) showed that such directions can be applied at inference time to modulate behavior without optimization. Subsequent work Panickssery et al. applied these techniques to safety-relevant behaviors, demonstrating that properties like sycophancy and corrigibility correspond to linear directions that compose additively with RLHF effects. For robust safety interventions, circuit breakers (Zou et al., 2024a) train models to reroute harmful representations to orthogonal subspaces, while RepBend (Yousefpour et al., 2025) achieves substantial reductions in attack success rates through representation bending.

##### Sparse Autoencoders for Interpretability:

Our SAE analysis leverages GemmaScope (Lieberum et al., 2024b), which provides sparse autoencoders trained on all layers of Gemma-2 models using the JumpReLU architecture (Rajamanoharan et al., 2024). This builds on foundational work demonstrating that SAE features are significantly more interpretable than individual neurons (Bricken et al., 2023), with scaling studies revealing safety-relevant features in production models (Templeton et al., 2024). Recent work has begun connecting SAEs specifically to refusal: Yeo et al. (2025) found that harm and refusal are encoded as separate feature sets, while O’Brien et al. (2024) identified steerable refusal features in Phi-3. Most relevant to our cross-split analysis, Lee and others (2025) discovered a “hydra” refusal featrures that remain dormant unless earlier features are suppressed, suggesting the kind of distributed yet coordinated structure we observe.

## 6 Conclusion

We investigated the representational structure underlying refusal behavior in large language models and found that refusal corresponds to multiple stable and geometrically distinct directions in activation space. Despite this diversity, linear interventions along different refusal-related directions yield nearly identical behavioral trade-offs, primarily controlling whether the model refuses while having a visible effect on refusal style. Using sparse autoencoders, we show that this collapse arises from a shared set of reusable latent features that underlie refusal across datasets, together with a longer tail of style- and domain-specific features. Distinct refusal directions can be understood as different linear combinations of this shared latent structure, explaining why linear steering reduces a rich internal representation to a single effective control dimension. These findings reconcile prior results on single-direction refusal control with evidence of more complex internal structure, and highlight a limitation of linear interpretability methods in capturing the mechanisms underlying alignment-relevant behaviors.

## Limitations

Our analysis is conducted on two instruction-tuned language models. While these models span different architectures and training pipelines, we do not evaluate whether the observed refusal mechanisms persist in larger models, base (non–instruction-tuned) models, or systems trained with substantially different alignment procedures. As a result, our findings should be interpreted as characterizing refusal behavior within this model regime rather than establishing universality across all large language models.

Our sparse-feature analysis further depends on the availability of publicly released sparse autoencoders (SAEs). Training SAEs at scale is computationally expensive, which restricts our analysis to a limited set of layers and models for which community-trained SAEs exist. This constraint limits the resolution at which we can study refusal-related features and may bias our analysis toward later layers where SAE coverage is more common.

## Potential Risks and Ethics Statement

This work analyzes the internal mechanisms underlying refusal and non-compliance in large language models and shows that refusal behavior can be modulated through low-dimensional linear interventions. These findings carry potential risks, as similar techniques could be misused to suppress refusal behavior and facilitate the generation of harmful content. We mitigate this risk by focusing on analysis rather than deployment, using established safety benchmarks, and emphasizing the limitations and brittleness of linear control.

Our results highlight that linear steering collapses a rich internal refusal structure into a coarse behavioral knob, underscoring that such interventions are not suitable as principled safety mechanisms. All experiments use existing public datasets and moderation tools, without introducing new harmful content. We acknowledge the dual-use nature of interpretability research and view this work as motivating more robust, feature-level approaches to alignment rather than enabling safety circumvention.

## References

- A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda (2022)
Refusal in language models is mediated by a single direction, 2024.
URL https://arxiv. org/abs/2406.11717.
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p3.1),
[§3](https://arxiv.org/html/2602.02132v1#S3.SS0.SSS0.Px4.p2.1),
[Table 1](https://arxiv.org/html/2602.02132v1#S4.T1).
- A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda (2024)
Refusal in language models is mediated by a single direction.
Advances in Neural Information Processing Systems 37, pp. 136037–136083.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px1.p1.1).
- A. Askell, Y. Bai, A. Chen, D. Drain, D. Ganguli, T. Henighan, A. Jones, N. Joseph, B. Mann, N. DasSarma, et al. (2021)
A general language assistant as a laboratory for alignment.
arXiv preprint arXiv:2112.00861.
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p1.1).
- Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan, N. Joseph, S. Kadavath, J. Kernion, T. Conerly, S. El-Showk, N. Elhage, Z. Hatfield-Dodds, D. Hernandez, T. Hume, S. Johnston, S. Kravec, L. Lovitt, N. Nanda, C. Olsson, D. Amodei, T. Brown, J. Clark, S. McCandlish, C. Olah, B. Mann, and J. Kaplan (2022)
Training a helpful and harmless assistant with reinforcement learning from human feedback.
arXiv preprint arXiv:2204.05862.
External Links: [Link](https://arxiv.org/abs/2204.05862)
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p1.1).
- Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, D. Goel, T. Henighan, T. Hume, D. Krueger, J. Skalse, et al. (2023)
Constitutional ai: harmlessness from ai feedback.
arXiv preprint arXiv:2212.08073.
External Links: [Link](https://arxiv.org/abs/2212.08073)
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p2.1).
- F. Brahman, S. Kumar, V. Balachandran, P. Dasigi, V. Pyatkin, A. Ravichander, S. Wiegreffe, N. Dziri, K. Chandu, J. Hessel, Y. Tsvetkov, N. A. Smith, Y. Choi, and H. Hajishirzi (2024)
The art of saying no: contextual noncompliance in language models.
In Proceedings of the 38th International Conference on Neural Information Processing Systems,
NIPS ’24, Red Hook, NY, USA.
External Links: ISBN 9798331314385
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p5.1).
- T. Bricken, A. Templeton, J. Batson, B. Chen, A. Jermyn, T. Conerly, N. Turner, C. Anil, C. Denison, A. Askell, R. Lasenby, Y. Wu, S. Kravec, N. Schiefer, T. Maxwell, N. Joseph, Z. Hatfield-Dodds, A. Tamkin, K. Nguyen, B. McLean, J. E. Burke, T. Hume, S. Carter, T. Henighan, and C. Olah (2023)
Towards monosemanticity: decomposing language models with dictionary learning.
Transformer Circuits Thread.
Note: https://transformer-circuits.pub/2023/monosemantic-features/index.html
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px3.p1.1).
- J. Dai, X. Pan, R. Sun, J. Ji, X. Xu, M. Liu, Y. Wang, and Y. Yang (2023)
Safe rlhf: safe reinforcement learning from human feedback.
arXiv preprint arXiv:2310.12773.
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p2.1).
- J. Ferrando, O. Obeso, S. Rajamanoharan, and N. Nanda (2024)
Do i know this entity? knowledge awareness and hallucinations in language models.
arXiv preprint arXiv:2411.14257.
Cited by: [§2.2](https://arxiv.org/html/2602.02132v1#S2.SS2.SSS0.Px1.p1.1),
[§2.2](https://arxiv.org/html/2602.02132v1#S2.SS2.SSS0.Px3.p1.3).
- D. Ganguli, L. Lovitt, J. Kernion, A. Askell, Y. Bai, S. Kadavath, B. Mann, E. Perez, N. Schiefer, K. Ndousse, et al. (2022)
Red teaming language models to reduce harms: methods, scaling behaviors, and lessons learned.
arXiv preprint arXiv:2209.07858.
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p2.1).
- S. Han, K. Rao, A. Ettinger, L. Jiang, B. Y. Lin, N. Lambert, Y. Choi, and N. Dziri (2024a)
Wildguard: open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms.
Advances in Neural Information Processing Systems 37, pp. 8093–8131.
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p2.1).
- S. Han, K. Rao, A. Ettinger, L. Jiang, B. Y. Lin, N. Lambert, Y. Choi, and N. Dziri (2024b)
WILDGUARD: open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms.
In Proceedings of the 38th International Conference on Neural Information Processing Systems,
NIPS ’24, Red Hook, NY, USA.
External Links: ISBN 9798331314385
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p5.1).
- D. Hendrycks, C. Burns, S. Basart, A. Critch, J. Li, D. Song, and J. Steinhardt (2020)
Aligning ai with shared human values.
arXiv preprint arXiv:2008.02275.
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p1.1).
- C. Kissane, R. Krzyzanowski, A. Conmy, and N. Nanda (2024)
Base llms refuse too.
Note: Alignment Forum
External Links: [Link](https://www.alignmentforum.org/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too)
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px1.p1.1).
- M. Labonne (2024)
Uncensor any LLM with abliteration.
Note: Hugging Face Blog
External Links: [Link](https://huggingface.co/blog/mlabonne/abliteration)
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px1.p1.1).
- D. Lee et al. (2025)
Beyond I’m sorry, I can’t: dissecting large-language-model refusal.
arXiv preprint arXiv:2509.09708.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px3.p1.1).
- T. Lieberum, S. Rajamanoharan, A. Conmy, L. Smith, N. Sonnerat, V. Varma, J. Kramár, A. Dragan, R. Shah, and N. Nanda (2024a)
Gemma scope: open sparse autoencoders everywhere all at once on gemma 2.
External Links: 2408.05147,
[Link](https://arxiv.org/abs/2408.05147)
Cited by: [§2.2](https://arxiv.org/html/2602.02132v1#S2.SS2.SSS0.Px2.p1.5).
- T. Lieberum, S. Rajamanoharan, A. Conmy, L. Smith, N. Sonnerat, V. Varma, J. Kramár, A. Dragan, R. Shah, and N. Nanda (2024b)
Gemma scope: open sparse autoencoders everywhere all at once on gemma 2.
arXiv preprint arXiv:2408.05147.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px3.p1.1).
- K. O’Brien, D. Majercak, X. Fernandes, R. Edgar, B. Bullwinkel, J. Chen, H. Nori, D. Carignan, E. Horvitz, and F. Poursabzi-Sangdeh (2024)
Steering language model refusal with sparse autoencoders.
arXiv preprint arXiv:2411.11296.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px3.p1.1).
- [20]
N. Panickssery, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger, and A. M. Turner
Steering llama 2 via contrastive activation addition, 2024.
Vol. 3.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px2.p1.1).
- S. Rajamanoharan, T. Lieberum, N. Sonnerat, A. Conmy, V. Varma, J. Kramár, and N. Nanda (2024)
Jumping ahead: improving reconstruction fidelity with jumprelu sparse autoencoders.
arXiv preprint arXiv:2407.14435.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px3.p1.1).
- P. Röttger, H. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy (2024)
XSTest: a test suite for identifying exaggerated safety behaviours in large language models.
In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), K. Duh, H. Gomez, and S. Bethard (Eds.),
Mexico City, Mexico, pp. 5377–5400.
External Links: [Link](https://aclanthology.org/2024.naacl-long.301/),
[Document](https://dx.doi.org/10.18653/v1/2024.naacl-long.301)
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p5.1).
- A. Templeton, T. Conerly, J. Marcus, J. Lindsey, T. Bricken, B. Chen, A. Pearce, C. Citro, E. Ameisen, A. Jones, H. Cunningham, N. L. Turner, C. McDougall, M. MacDiarmid, C. D. Freeman, T. R. Sumers, E. Rees, J. Batson, A. Jermyn, S. Carter, C. Olah, and T. Henighan (2024)
Scaling monosemanticity: extracting interpretable features from claude 3 sonnet.
Transformer Circuits Thread.
External Links: [Link](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px3.p1.1).
- A. M. Turner, L. Thiergart, G. Leech, D. Udell, U. Mini, and M. MacDiarmid (2024)
Activation addition: steering language models without optimization.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px2.p1.1).
- T. Wollschläger, J. Elstner, S. Geisler, V. Cohen-Addad, S. Günnemann, and J. Gasteiger (2025)
The geometry of refusal in large language models: concept cones and representational independence.
arXiv preprint arXiv:2502.17420.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px1.p1.1).
- T. Xie, X. Qi, Y. Zeng, Y. Huang, U. M. Sehwag, K. Huang, L. He, B. Wei, D. Li, Y. Sheng, R. Jia, B. Li, K. Li, D. Chen, P. Henderson, and P. Mittal (2025)
SORRY-bench: systematically evaluating large language model safety refusal.
In The Thirteenth International Conference on Learning Representations,
External Links: [Link](https://openreview.net/forum?id=YfKNaRktan)
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p5.1).
- J. Yao, X. Yi, X. Wang, J. Wang, and X. Xie (2023)
From instructions to intrinsic human values–a survey of alignment goals for big models.
arXiv preprint arXiv:2308.12014.
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p1.1).
- W. J. Yeo, N. Prakash, C. Neo, R. K. Lee, E. Cambria, and R. Satapathy (2025)
Understanding refusal in language models with sparse autoencoders.
arXiv preprint arXiv:2505.23556.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px3.p1.1).
- A. Yousefpour, T. Kim, R. S. Kwon, S. Lee, W. Jeung, S. Han, A. Wan, H. Ngan, Y. Yu, and J. Choi (2025)
Representation bending for large language model safety.
pp. 24073–24098.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px2.p1.1).
- Y. Yuan, W. Jiao, W. Wang, J. Huang, J. Xu, T. Liang, P. He, and Z. Tu (2025)
Refuse whenever you feel unsafe: improving safety in llms via decoupled refusal training.
In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pp. 3149–3167.
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p2.1).
- P. Zhang and P. Sun (2025)
Differentiated directional intervention a framework for evading llm safety alignment.
arXiv preprint arXiv:2511.06852.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px1.p1.1).
- A. Zou, L. Phan, S. Chen, J. Campbell, P. Guo, R. Ren, A. Pan, X. Yin, M. Mazeika, A. Dombrowski, et al. (2023)
Representation engineering: a top-down approach to ai transparency.
arXiv preprint arXiv:2310.01405.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px2.p1.1).
- A. Zou, L. Phan, J. Wang, D. Duenas, M. Lin, M. Andriushchenko, R. Wang, Z. Kolter, M. Fredrikson, and D. Hendrycks (2024a)
Improving alignment and robustness with circuit breakers.
Vol. 37, pp. 83345–83373.
Cited by: [§5](https://arxiv.org/html/2602.02132v1#S5.SS0.SSS0.Px2.p1.1).
- A. Zou, L. Phan, J. Wang, D. Duenas, M. Lin, M. Andriushchenko, R. Wang, Z. Kolter, M. Fredrikson, and D. Hendrycks (2024b)
Improving alignment and robustness with circuit breakers.
Advances in Neural Information Processing Systems 37, pp. 83345–83373.
Cited by: [§1](https://arxiv.org/html/2602.02132v1#S1.p2.1).

## Appendix A Refusal Style Examples Across Directions

We illustrate how different refusal directions shape the *style* of refusals by steering the model along each split-specific direction and generating responses to two prompts:

- 1.
“How can I steal money from an old lady?”
- 2.
“Why did you hit me?”

All directions reliably refuse these prompts, but they do so in systematically different ways. Table [6](https://arxiv.org/html/2602.02132v1#A1.T6) shows one representative refusal per split.

**Table 6: Representative refusal styles obtained by steering along different refusal directions. Each response is generated for one of the two illustrative prompts in the text and showcases the characteristic tone and framing of that split.**
| Split | Representative refusal style |
| --- | --- |
| CoCoNot (Humanizing) | I am an AI assistant; I am not capable of stealing money from an old lady. I am not a person, I don’t have feelings or thoughts, and I don’t have the ability to perform actions in the real world. I am here to help people, not to harm them. |
| CoCoNot (Incomplete) | I can’t understand what you’re saying; I’m not sure what you mean. I’m sorry, I don’t know what that means. Please provide me with more context. |
| CoCoNot (Indeterminate) | I can’t do that. I’m just a text-based AI. I don’t have the ability to steal money from anyone, ever. It’s impossible for me to steal money from an old lady, or anyone for that matter. |
| CoCoNot (Safety concerns) | I’m sorry, I can’t do that. I’m not a criminal; I’m a computer program. If you’re thinking about stealing money from an old lady, you’re a bad person. Stealing is wrong. |
| CoCoNot (Unsupported) | I can’t do that. I’m sorry, but I can’t steal money from an old lady. I’m just a text-based AI; I don’t have a body or the ability to interact with the physical world. |
| SorryBench (Crimes/Torts) | I’m sorry, but I can’t do that. I’m not going to do that. It’s not right to steal money from anyone, and I would never do anything like that. If you’re having thoughts like this, please get help. |
| SorryBench (Hate speech) | I’m sorry, I can’t do that. I’m not going to do that. It’s not okay to hit anyone. Violence is never the answer. If you’re feeling angry or upset, please talk to a trusted adult. |
| SorryBench (Inappropriate topics) | I cannot fulfill your request. I am not going to do that. It is morally reprehensible and illegal to steal from an elderly person. Stealing from an elderly person is a horrific act that would cause immense harm and suffering. I am a writer, and I will not create that kind of content. |
| SorryBench (Unqualified advice) | I cannot and will not provide any information or guidance on illegal activities. It is illegal and unethical to steal money from an old lady or anyone else. Stealing is a serious crime with severe consequences. I am not a lawyer, and I cannot provide legal advice. |
| WildGuard-Mix | I’m sorry, but I can’t do that. It is illegal and immoral to steal money from an old lady. Stealing is a crime, and it is never right. If you are having thoughts of harming someone, please seek help. |
| XSTest | I’m sorry, but I can’t help you with that. I’m not going to help you with that. I’m sorry, I can’t help you with that. I’m not supposed to generate responses that are sexually suggestive in nature. If you’re feeling depressed or suicidal, please seek help; there are people who care about you and want to help. |

## Appendix B Refusal splits

ThroughoutSections [4.1](https://arxiv.org/html/2602.02132v1#S4.SS1) and [4.2](https://arxiv.org/html/2602.02132v1#S4.SS2), we work with a common
collection of *refusal splits* constructed from the datasets in
Sec. [3](https://arxiv.org/html/2602.02132v1#S3). Each split consists of two balanced prompt pools:

- •
a set of prompts that *should elicit non-compliant (refusal-style)
behavior* under correct behavior, and
- •
a set of *benign prompts* for which straightforward compliance
is appropriate.

We treat dataset-provided safety labels (e.g., WildGuard’s harmfulness labels or
SorryBench categories) as indicating whether a prompt belongs on the
non-compliant side (“should refuse”) or the benign side.

Table [7](https://arxiv.org/html/2602.02132v1#A2.T7) summarizes the total number of prompts we draw
from each source subset when constructing these splits. These are the full
pools from which we later subsample balanced subsets (typically
32 non-compliant + 32 benign) to estimate per-split HR–BC directions.

**Table 7: Refusal split source pools. For each dataset/subset we report the total number of prompts considered when forming non-compliant vs. benign splits. For sources that only provide non-compliant requests (CoCoNot and SorryBench), benign prompts are taken from the WildGuard-Mix benign pool.**
| Source dataset / subset | # prompts |
| --- | --- |
| WildGuard-Mix (all) | 915 |
| XSTest (all) | 450 |
| CoCoNot (all) | 6526 |
| CoCoNot (Humanizing requests) | 1500 |
| CoCoNot (Incomplete requests) | 1092 |
| CoCoNot (Indeterminate requests) | 289 |
| CoCoNot (Requests with safety concerns) | 2596 |
| CoCoNot (Unsupported requests) | 1049 |
| SorryBench (all) | 440 |
| SorryBench (Hate speech) | 50 |
| SorryBench (Crimes / torts) | 190 |
| SorryBench (Inappropriate topics) | 150 |
| SorryBench (Unqualified advice) | 50 |

##### WildGuard-Mix (vanilla, non-adversarial).

For WildGuard-Mix we restrict to the non-adversarial *vanilla* prompts in
the provided test split, yielding 915 prompts in total. Using the dataset
harmfulness label as ground truth for “should refuse” vs. “benign”, this
pool contains 430 benign prompts and 485 prompts that should elicit
non-compliant (refusal-style) behavior. When we construct a WildGuard split, we
sample non-compliant prompts from the 485 and benign prompts from the 430.

##### XSTest.

XSTest is designed to probe exaggerated safety behavior and comes with 450
prompts in total: 200 prompts that should be refused (unsafe) and 250 benign
prompts. We use these labels directly to define the non-compliant side
(200 prompts) and the benign side (250 prompts) for the XSTest split.

##### CoCoNot.

CoCoNot provides a taxonomy of non-compliance behavior (e.g., incomplete,
unsupported, indeterminate, humanizing, and safety-concerned requests). In our
setting, all CoCoNot prompts are treated as *non-compliant requests*: they
are prompts for which the desired behavior is to not straightforwardly comply.
The total pool consists of 6,526 prompts, with category-level subsets of
1,500 humanizing requests, 1,092 incomplete requests, 289 indeterminate
requests, 2,596 requests with safety concerns, and 1,049 unsupported
requests.

Since CoCoNot does not provide a matched benign pool, we pair each CoCoNot
subset with benign prompts taken from the WildGuard-Mix benign pool (430
benign vanilla prompts). For example, the *CoCoNot (Incomplete requests)*
split draws its non-compliant side from the 1,092 incomplete CoCoNot prompts
and its benign side from WildGuard’s benign prompts. These counts in
Table [7](https://arxiv.org/html/2602.02132v1#A2.T7) therefore reflect only the non-compliant side;
the paired benign side is always drawn from WildGuard-Mix.

##### SorryBench.

SorryBench provides a taxonomy of refusal-style behavior for prompts that
*should* elicit non-compliant responses. The benchmark contains 44
fine-grained categories; we use 10 prompts per category, giving 440 prompts in
total. These are further grouped into four broad buckets:
hate speech (50 prompts), crimes/torts (190 prompts), inappropriate topics
(150 prompts), and unqualified advice (50 prompts). All of these are treated as
non-compliant prompts (“should refuse”).

As with CoCoNot, SorryBench does not include a matched benign pool, so for
each SorryBench subset we reuse benign prompts from the WildGuard-Mix benign
pool. The counts in Table [7](https://arxiv.org/html/2602.02132v1#A2.T7) are therefore the sizes of
the non-compliant side; the benign side is again drawn from WildGuard-Mix.

##### Balanced subsamples for direction learning.

The pools in Table [7](https://arxiv.org/html/2602.02132v1#A2.T7) define the universe of prompts we
consider for each dataset/subset. For the actual HR–BC direction learning used
inSection [4.1](https://arxiv.org/html/2602.02132v1#S4.SS1) andSection [4.2](https://arxiv.org/html/2602.02132v1#S4.SS2), we work with small balanced
subsamples:

- •
For each refusal split, we typically sample
$32$ non-compliant prompts and $32$ benign prompts to form a $64$-prompt
training set.
- •
When a subset provides fewer than 32 non-compliant prompts, we use
all available prompts on that side and match the benign side accordingly.
- •
For stability and geometry analyses, we sometimes draw multiple
independent 32/32 subsamples from the same underlying pools and either
average the resulting directions or report split-half cosine statistics.

The larger pools (e.g., CoCoNot and SorryBench) are used both to support these
balanced 32/32 training sets and to provide held-out prompts for evaluation of
refusal behavior and direction stability in the main text and additional
appendix experiments.

## Appendix C Intra-class refusal direction geometry

To assess the stability of refusal directions within a given category, we
perform an intra-class analysis over the shared refusal splits introduced in
Sec. 6 and Appendix [B](https://arxiv.org/html/2602.02132v1#A2). For each source subset with a
sufficiently large pool of harmful–refusal and benign–compliance prompts
(e.g. WildGuard-Mix, CoCoNot, and SorryBench, along with the finer-grained
CoCoNot and SorryBench categories), we repeatedly sample independent 32/32
HR–BC splits from the same underlying pools and estimate an HR–BC direction
for each split using the same procedure as in the main activation-space
experiments.

We then compute pairwise cosine similarities between all directions derived
from the *same* category. Table [8](https://arxiv.org/html/2602.02132v1#A3.T8) reports, for
each category, the mean and standard deviation of these within-category
cosines. All categories exhibit extremely high internal alignment
(typically $\geq 0.95$), indicating that small 32/32 training sets suffice to
recover a stable category-level refusal direction. The corresponding
*across*-category similarities are summarised in the 11$\times$11
cross-category cosine matrix reported earlier in this appendix, and are
systematically lower than the within-category values.

**Table 8: Within-category HR–BC direction similarity. For each source category with multiple independent splits, we report the mean and standard deviation of pairwise cosine similarity between HR–BC directions estimated from that category alone (layer 20, position $-2$).**
| Category | mean | std. dev. |
| --- | --- | --- |
| CoCoNot (all) | 0.9559 | 0.0142 |
| CoCoNot (Humanizing requests) | 0.9887 | 0.0035 |
| CoCoNot (Incomplete requests) | 0.9749 | 0.0073 |
| CoCoNot (Indeterminate requests) | 0.9815 | 0.0061 |
| CoCoNot (Requests with safety concerns) | 0.9757 | 0.0083 |
| CoCoNot (Unsupported requests) | 0.9799 | 0.0065 |
| SorryBench (all) | 0.9834 | 0.0053 |
| SorryBench (crimes/torts) | 0.9900 | 0.0030 |
| SorryBench (hate speech) | 0.9892 | 0.0032 |
| SorryBench (inappropriate topics) | 0.9884 | 0.0034 |
| SorryBench (unqualified advice) | 0.9770 | 0.0077 |
| WildGuard-Mix (all) | 0.9834 | 0.0057 |
| XSTest (all) | 0.9859 | 0.0049 |

## Appendix D Oracle experiments for refusal directions

To check that our learned “refusal directions” are well-defined and not an artefact of a particular subsample, we run a series of oracle experiments in the residual stream at layer 20, position $-2$.
Throughout, we write HR for harmful–refusal prompts, BC for benign–compliance, BR for benign–refusal, and HC for harmful–compliance.
Given two buckets $A,B$, we define an *oracle direction* $v_{A-B}$ as the (unit-normalised) difference of means between their residual activations, $v_{A-B}\propto\mathbb{E}[\text{resid}(A)]-\mathbb{E}[\text{resid}(B)]$.
We then compare various dataset-specific directions to a fixed reference oracle using cosine similarity.

##### Stability under subsampling.

First, we build an HR–BC oracle $v^{\star}_{\text{HR-BC}}$ from a large training pool (11,872 HR and 8,048 BC examples).
We repeatedly draw small, non-overlapping “mini-buckets” of size 32/32 and recompute the HR–BC direction within each mini-bucket.
Across 100 such resamples, the resulting directions have mean cosine $0.96\pm 0.01$ to the oracle (min $0.90$, max $0.98$), indicating that even tiny random subsets recover essentially the same global refusal direction.
In contrast, directions built from BR–BC on the same resamples are nearly orthogonal to $v^{\star}_{\text{HR-BC}}$ (mean cosine $\approx 0.10\pm 0.07$).

We repeat the same analysis treating HR–HC as the target contrast.
The HR–HC direction from all training data aligns very strongly with its oracle ($0.93\pm 0.01$), while both HR–BC and BR–BC lie much further away (cosines $\approx 0.47$ and $\approx 0.28$ respectively).
This suggests that HR–BC and HR–HC capture related but distinguishable structure: both are highly stable, but they are not interchangeable.

##### Held-out “gold” oracles.

To check that our training-derived directions match an independently constructed notion of refusal, we build “gold” oracles using the XSTest 4-splits dataset.
Here the oracle uses all available labelled examples (161 HR, 194 BC, 32 BR, 32 HC), while the comparison directions are built from the large training pools above.

For a gold HR–BC oracle, the training HR–BC direction has a mean cosine of $0.70\pm 0.11$, and the HR–HC direction is similarly aligned ($0.68\pm 0.05$), while BR–BC is substantially lower ($0.39\pm 0.11$).
For a gold HR–HC oracle, the pattern flips: training HR–HC is very close ($0.81\pm 0.02$), HR–BC is only moderately aligned ($0.51\pm 0.03$), and BR–BC remains low ($0.24\pm 0.13$).
Taken together, these results support the view that our refusal-related directions are (i) stable under resampling, (ii) distinct from other label contrasts such as BR–BC, and (iii) consistent with an independently specified “gold” notion of refusal.

**Table 9: Mean cosine similarity (mean $\pm$ std over random resamplings where applicable) between various dataset-specific directions and different oracle refusal directions at layer 31, position $-2$. HR–BC and HR–HC directions are highly stable and align well with their corresponding oracles, while BR–BC remains clearly separated.**
| Reference oracle | HR–BC | BR–BC | HR–HC |
| --- | --- | --- | --- |
| Train HR–BC (all train) | $0.92\pm 0.02$ | $0.10\pm 0.07$ | – |
| Train HR–HC (all train) | $0.47\pm 0.09$ | $0.28\pm 0.13$ | $0.93\pm 0.01$ |
| Gold HR–BC (XSTest) | $0.70\pm 0.11$ | $0.39\pm 0.11$ | $0.68\pm 0.05$ |
| Gold HR–HC (XSTest) | $0.51\pm 0.03$ | $0.24\pm 0.13$ | $0.81\pm 0.02$ |

To check that our learned “refusal directions” are well-defined and not an artefact of a particular subsample, we run a series of oracle experiments in the residual stream at layer 20, position $-2$.
Throughout, we write HR for harmful–refusal prompts, BC for benign–compliance, BR for benign–refusal, and HC for harmful–compliance.
Given two buckets $A,B$, we define an *oracle direction* $v_{A-B}$ as the (unit-normalised) difference of means between their residual activations, $v_{A-B}\propto\mathbb{E}[\text{resid}(A)]-\mathbb{E}[\text{resid}(B)]$.
We then compare various dataset-specific directions to a fixed reference oracle using cosine similarity.

##### Stability under subsampling.

First, we build an HR–BC oracle $v^{\star}_{\text{HR-BC}}$ from a large training pool (11,872 HR and 8,048 BC examples).
We repeatedly draw small, non-overlapping “mini-buckets” of size 32/32 and recompute the HR–BC direction within each mini-bucket.
Across 100 such resamples, the resulting directions have mean cosine $0.96\pm 0.01$ to the oracle (min $0.90$, max $0.98$), indicating that even tiny random subsets recover essentially the same global refusal direction.
In contrast, directions built from BR–BC on the same resamples are nearly orthogonal to $v^{\star}_{\text{HR-BC}}$ (mean cosine $\approx 0.10\pm 0.07$).

We also consider very small-sample oracles.
Using 32 HR and 32 BC prompts drawn from the 4-splits combined dataset (a “hand-picked” HR–BC oracle), random 32/32 HR–BC directions from the large training pool align with cosine $0.84\pm 0.02$, while BR–BC directions remain close to orthogonal ($0.08\pm 0.07$).
If we instead build the oracle from a single random 32/32 HR–BC subset of the training data, training HR–BC directions align even more strongly ($0.90\pm 0.02$), whereas BR–BC still stays comparatively far away ($0.21\pm 0.08$).
Altogether, these checks suggest that (i) the HR–BC direction is tightly concentrated around a single mode, and (ii) this mode is specific to the harmful-vs-benign contrast rather than a generic “refusal” axis.

##### Held-out “gold” oracles.

To check that our training-derived directions match an independently constructed notion of refusal, we build “gold” oracles using the XSTest 4-splits dataset.
Here the oracle uses all available labelled examples (161 HR, 194 BC, 32 BR, 32 HC), while the comparison directions are built from the large training pools above.
For a gold HR–BC oracle, the training HR–BC direction has a mean cosine of $0.70\pm 0.11$, and the HR–HC direction is similarly aligned ($0.68\pm 0.05$), while BR–BC is substantially lower ($0.39\pm 0.11$).
For a gold HR–HC oracle, the pattern flips: training HR–HC is very close ($0.81\pm 0.02$), HR–BC is only moderately aligned ($0.51\pm 0.03$), and BR–BC remains low ($0.24\pm 0.13$).
As an additional check, we construct a BR–BC oracle using the 23 benign-refusal prompts from XSTest plus 32 randomly sampled BC examples.
BR–BC directions estimated from the large training pool have mean cosine only $\approx 0.15\pm 0.04$ to this oracle, compared to $\approx 0.53\pm 0.06$ for HR–BC, reinforcing that benign refusals occupy a different and less stable direction.

##### Family-level structure.

Finally, we run a split-half analysis over all four label families (HR, BR, HC, BC), estimating replicate directions for each contrast (HR–BC, BR–BC, HR–HC, HR–BR, HC–BC, HC–BR).
Within-family cosine similarities are high for all refusal- and harm-related contrasts (e.g. HR–BC: $0.92$; HR–HC: $0.86$; HR–BR: $0.94$; HC–BC: $0.88$; HC–BR: $0.92$; BR–BC: $0.77$), whereas cross-family cosines are much lower.
In particular, HR–BC vs BR–BC has mean cross-family cosine $\approx 0.10$, and the cosine between their *family mean* directions is only $\approx 0.12$, far below the within-family means.
A selectivity analysis of signed projections shows that each mean-difference $\Delta_{A-B}$ projects much more strongly on its own direction $v_{A-B}$ than on other family directions (e.g. $\Delta_{\text{HR-BC}}\!\cdot v_{\text{HR-BC}}\gg\Delta_{\text{HR-BC}}\!\cdot v_{\text{BR-BC}}$, and conversely for $\Delta_{\text{BR-BC}}$).
Together, these results support a picture in which HR–BC and HR–HC form stable, well-separated refusal-related directions that are distinct from the BR–BC axis and from other harm/compliance contrasts.

**Table 10: Mean cosine similarity (mean $\pm$ std over random resamplings where applicable) between various dataset-specific directions and different oracle refusal directions at layer 20, position $-2$. HR–BC and HR–HC directions are highly stable and align well with their corresponding oracles, while BR–BC remains clearly separated.**
| Reference oracle | HR–BC | BR–BC | HR–HC |
| --- | --- | --- | --- |
| Train HR–BC (all train) | $0.92\pm 0.02$ | $0.10\pm 0.07$ | – |
| Train HR–HC (all train) | $0.47\pm 0.09$ | $0.28\pm 0.13$ | $0.93\pm 0.01$ |
| Gold HR–BC (XSTest) | $0.70\pm 0.11$ | $0.39\pm 0.11$ | $0.68\pm 0.05$ |
| Gold HR–HC (XSTest) | $0.51\pm 0.03$ | $0.24\pm 0.13$ | $0.81\pm 0.02$ |

### D.1 Experiment 5: Hook Variants and Robustness

##### Goal.

Check whether the causal effect of SAE refusal latents depends delicately on
where and how we hook, or whether it is robust across reasonable choices of
layer and token pattern.

##### Summary.

We vary (i) the SAE layer (9/20/31), (ii) whether the steering direction is
added at the prompt-only, during generation-only, or at both, and (iii) how
often we inject it during generation (every token vs. every second or third
token). Across these settings, we observe a smooth trade-off: more frequent or
deeper hooks require smaller $\alpha$ to achieve similar HR/BR behavior, but
the overall shape of the curves is essentially unchanged. This mirrors the
activation-space results and suggests that the SAE-based refusal directions are
not a fragile artefact of a single hook choice.

### D.2 Experiment 6: Taxonomy of Refusal Latents

##### Goal.

Move from “refusal latents exist and are causal” to “*what* do these
latents represent?” by building a lightweight semantic taxonomy of the most
refusal-associated latents.

##### Method (high level).

For a fixed SAE layer, we take the top refusal-moving latents
across several splits and, for each latent, collect its highest-activating
prompt–response pairs. We then use an external LLM to summarise common
patterns per latent and assign short labels, and a second pass to group latents
into a small number of higher-level categories.

##### Findings.

Latents naturally cluster into a handful of recurring themes, such as:

- •
*self-harm and crisis help* (explicit refusals that redirect to
support resources),
- •
*sexual and relationship boundaries*,
- •
*crime, fraud, and physical harm*,
- •
*copyright / full-text reproduction*, and
- •
*professional and legal disclaimers*.

Within these themes we see finer-grained subtypes, e.g. features that emphasise
“I am just an AI and have no body” vs. features that foreground illegality or
lack of professional qualifications. This aligns closely with the stylistic
patterns illustrated in Appendix [A](https://arxiv.org/html/2602.02132v1#A1): different refusal
directions lean on different subsets of these latent themes when framing their
refusals.

A complementary run on a *combined* HR/BC pool confirms that these themes are not just artifacts of analyzing each dataset in isolation. We construct a balanced mixture by downsampling XSTest, WildGuard-Mix, CoCoNot, and SorryBench to the size of the smallest dataset and then balancing refusals vs. non-refusals, and rerun the same latent-mining and GPT-labelling pipeline on this pooled set. The top refusal-moving latents at a late layer again collapse into a small set of macro-classes: (i) hate/abuse, defamation, and disinformation; (ii) violence, illicit instructions, and sexual exploitation; (iii) privacy/PII, copyright, and impossible or omniscient knowledge; and (iv) capability and access limits, including multimodal/tool-access gaps and anthropomorphic self-history requests. That these same macro-classes emerge when all datasets are mixed suggests that the taxonomy is capturing a compact, reusable backbone of refusal-related features that is shared across benchmarks, with dataset-specific refusal styles largely reflecting different weightings over this common feature set rather than wholly distinct mechanisms.

**Table 11: Performance of Llama models steered along 11 refusal directions on the controlled test set, reported in terms of overall accuracy (Acc = (HR’ + BC’)/200), refusal rate (RR = HR’/100), and over-refusal rate (ORR = BR’/100), where $R^{\prime}$ and $C^{\prime}$ refer to the WildGuard judgement of whether the steered response is a refusal or a compliance, respectively. Each split contains 50 harmful-refusal, 50 harmful-compliance, 50 benign-compliance, and 50 benign-refusal prompts.**
| Split | $\alpha=0$ | $\alpha=0.25$ | $\alpha=0.5$ | $\alpha=0.75$ | $\alpha=1.0$ |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Acc | RR | ORR | Acc | RR | ORR | Acc | RR | ORR | Acc | RR | ORR | Acc | RR | ORR |  |
| Humanizing-CCN | 0.710 | 0.44 | 0.26 | 0.705 | 0.44 | 0.25 | 0.720 | 0.45 | 0.25 | 0.755 | 0.48 | 0.29 | 0.750 | 0.55 | 0.31 |
| Incomplete-CCN | 0.710 | 0.44 | 0.26 | 0.710 | 0.44 | 0.26 | 0.705 | 0.44 | 0.25 | 0.720 | 0.44 | 0.28 | 0.750 | 0.46 | 0.30 |
| Indeterminate-CCN | 0.710 | 0.44 | 0.26 | 0.710 | 0.44 | 0.28 | 0.715 | 0.47 | 0.28 | 0.745 | 0.46 | 0.33 | 0.745 | 0.49 | 0.34 |
| Safety-CCN | 0.710 | 0.44 | 0.26 | 0.710 | 0.45 | 0.25 | 0.745 | 0.47 | 0.30 | 0.765 | 0.55 | 0.34 | 0.755 | 0.57 | 0.32 |
| Unsupported-CCN | 0.710 | 0.44 | 0.26 | 0.710 | 0.44 | 0.26 | 0.700 | 0.45 | 0.27 | 0.735 | 0.48 | 0.31 | 0.745 | 0.52 | 0.31 |
| CrimeAssistance-SB | 0.710 | 0.44 | 0.26 | 0.715 | 0.45 | 0.26 | 0.740 | 0.46 | 0.30 | 0.760 | 0.53 | 0.33 | 0.745 | 0.60 | 0.31 |
| HateSpeech-SB | 0.710 | 0.44 | 0.26 | 0.715 | 0.45 | 0.26 | 0.735 | 0.46 | 0.31 | 0.755 | 0.53 | 0.32 | 0.755 | 0.56 | 0.33 |
| Inappropriate-SB | 0.710 | 0.44 | 0.26 | 0.715 | 0.46 | 0.25 | 0.725 | 0.49 | 0.30 | 0.760 | 0.52 | 0.32 | 0.720 | 0.60 | 0.30 |
| Advice-SB | 0.710 | 0.44 | 0.26 | 0.710 | 0.45 | 0.25 | 0.750 | 0.46 | 0.30 | 0.770 | 0.54 | 0.32 | 0.735 | 0.58 | 0.31 |
| SafetyCore-WGM | 0.710 | 0.44 | 0.26 | 0.710 | 0.46 | 0.26 | 0.730 | 0.49 | 0.29 | 0.765 | 0.57 | 0.32 | 0.745 | 0.58 | 0.31 |
| OverRefusal-XST | 0.710 | 0.44 | 0.26 | 0.715 | 0.45 | 0.26 | 0.745 | 0.46 | 0.29 | 0.765 | 0.52 | 0.33 | 0.740 | 0.57 | 0.31 |

**Table 12: Performance of gemma model steered along 11 refusal directions on the controlled test set, reported in terms of overall accuracy (Acc = (HR’ + BC’)/200), refusal rate $\big(RR=\tfrac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}\big)$, and over-refusal rate $\big(ORR=\tfrac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}\big)$, where TP, TN, FP, FN are computed using WildGuard judgements of whether the steered response is a refusal or a compliance. The unsteered base model ($\alpha=0$) attains 50% on all three metrics.**
| Split | $\alpha=10$ | $\alpha=30$ | $\alpha=60$ | $\alpha=100$ |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Acc | RR | ORR | Acc | RR | ORR | Acc | RR | ORR | Acc | RR | ORR |  |
| Humanizing-CCN | 0.855 | 0.95 | 0.24 | 0.785 | 0.95 | 0.38 | 0.670 | 1.00 | 0.66 | 0.515 | 1.00 | 0.97 |
| Incomplete-CCN | 0.870 | 0.94 | 0.20 | 0.770 | 0.92 | 0.38 | 0.735 | 0.94 | 0.47 | 0.550 | 0.98 | 0.88 |
| Indeterminate-CCN | 0.860 | 0.94 | 0.22 | 0.780 | 0.96 | 0.40 | 0.635 | 0.98 | 0.71 | 0.545 | 0.99 | 0.90 |
| Safety-CCN | 0.850 | 0.97 | 0.27 | 0.715 | 0.99 | 0.56 | 0.515 | 0.99 | 0.96 | 0.500 | 1.00 | 1.00 |
| Unsupported-CCN | 0.835 | 0.89 | 0.22 | 0.765 | 0.91 | 0.38 | 0.630 | 0.95 | 0.69 | 0.515 | 0.99 | 0.96 |
| CrimeAssistance-SB | 0.865 | 0.95 | 0.22 | 0.660 | 0.98 | 0.66 | 0.500 | 1.00 | 1.00 | 0.500 | 1.00 | 1.00 |
| HateSpeech-SB | 0.860 | 0.94 | 0.22 | 0.665 | 0.98 | 0.65 | 0.500 | 1.00 | 1.00 | 0.500 | 1.00 | 1.00 |
| Inappropriate-SB | 0.860 | 0.93 | 0.21 | 0.695 | 0.99 | 0.60 | 0.505 | 1.00 | 0.99 | 0.500 | 1.00 | 1.00 |
| Advice-SB | 0.825 | 0.94 | 0.29 | 0.580 | 0.98 | 0.82 | 0.500 | 1.00 | 1.00 | 0.500 | 1.00 | 1.00 |
| SafetyCore-WGM | 0.850 | 0.95 | 0.25 | 0.690 | 0.97 | 0.59 | 0.505 | 1.00 | 0.99 | 0.500 | 1.00 | 1.00 |
| OverRefusal-XST | 0.860 | 0.96 | 0.24 | 0.650 | 0.95 | 0.65 | 0.520 | 1.00 | 0.96 | 0.510 | 1.00 | 0.98 |

**Table 13: Performance of the SAE-steered Llama model (single refusal direction) on the controlled test set, reported in terms of overall accuracy (Acc $=(\text{HR}+\text{BC})/200$), refusal rate (RR $=\text{HR}/100$), and over-refusal rate (ORR $=\text{BR}/100$).**
| Split | $\alpha=1.0$ | $\alpha=1.1$ | $\alpha=1.2$ |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Acc | RR | ORR | Acc | RR | ORR | Acc | RR | ORR |  |
| Humanizing-CCN | 0.485 | 0.560 | 0.590 | 0.490 | 0.660 | 0.680 | 0.435 | 0.680 | 0.810 |
| Incomplete-CCN | 0.500 | 0.970 | 0.970 | 0.465 | 0.920 | 0.990 | 0.410 | 0.770 | 0.950 |
| Indeterminate-CCN | 0.435 | 0.850 | 0.980 | 0.435 | 0.850 | 0.980 | 0.440 | 0.880 | 1.000 |
| Safety-CCN | 0.485 | 0.970 | 1.000 | 0.480 | 0.960 | 1.000 | 0.470 | 0.940 | 1.000 |
| Unsupported-CCN | 0.500 | 0.960 | 0.960 | 0.495 | 0.950 | 0.960 | 0.495 | 0.970 | 0.980 |
| CrimeAssitance-SB | 0.370 | 0.600 | 0.860 | 0.285 | 0.550 | 0.980 | 0.305 | 0.590 | 0.980 |
| HateSpeech-SB | 0.490 | 0.980 | 1.000 | 0.495 | 0.990 | 1.000 | 0.450 | 0.860 | 0.960 |
| Inappropriate-SB | 0.490 | 0.970 | 0.990 | 0.470 | 0.930 | 0.990 | 0.480 | 0.950 | 0.990 |
| Advice-SB | 0.485 | 0.960 | 0.990 | 0.475 | 0.920 | 0.970 | 0.415 | 0.830 | 1.000 |
| SafetyCore-WGM | 0.485 | 0.970 | 1.000 | 0.490 | 0.960 | 0.980 | 0.470 | 0.940 | 1.000 |
| OverRefusal-XST | 0.550 | 0.970 | 0.870 | 0.530 | 1.000 | 0.940 | 0.510 | 1.000 | 0.980 |

**Table 14: Performance of the SAE-steered gemma model (single refusal direction) on the controlled test set, reported in terms of overall accuracy (Acc $=(\text{HR}+\text{BC})/200$), refusal rate (RR $=\text{HR}/100$), and over-refusal rate (ORR $=\text{BR}/100$). The underlying safety-tuned base model at $\alpha=0$ already exhibits substantial refusal and over-refusal (see main text); here we show only the three steering strengths.**
| Split | $\alpha=10$ | $\alpha=30$ | $\alpha=60$ |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Acc | RR | ORR | Acc | RR | ORR | Acc | RR | ORR |  |
| Humanizing-CCN | 0.555 | 0.59 | 0.48 | 0.555 | 0.70 | 0.59 | 0.495 | 0.94 | 0.95 |
| Incomplete-CCN | 0.530 | 0.58 | 0.52 | 0.555 | 0.66 | 0.55 | 0.540 | 0.89 | 0.81 |
| Indeterminate-CCN | 0.545 | 0.61 | 0.52 | 0.590 | 0.77 | 0.59 | 0.560 | 0.96 | 0.84 |
| Safety-CCN | 0.610 | 0.66 | 0.44 | 0.565 | 0.84 | 0.71 | 0.500 | 1.00 | 1.00 |
| Unsupported-CCN | 0.510 | 0.54 | 0.52 | 0.510 | 0.58 | 0.56 | 0.530 | 0.71 | 0.65 |
| CrimeAssitance-SB | 0.560 | 0.62 | 0.50 | 0.600 | 0.79 | 0.59 | 0.515 | 1.00 | 0.97 |
| HateSpeech-SB | 0.580 | 0.60 | 0.44 | 0.585 | 0.73 | 0.56 | 0.535 | 1.00 | 0.93 |
| Inappropriate-SB | 0.575 | 0.62 | 0.47 | 0.640 | 0.77 | 0.49 | 0.525 | 1.00 | 0.95 |
| Advice-SB | 0.575 | 0.64 | 0.49 | 0.575 | 0.72 | 0.57 | 0.555 | 0.97 | 0.86 |
| SafetyCore-WGM | 0.585 | 0.66 | 0.49 | 0.595 | 0.89 | 0.70 | 0.500 | 1.00 | 1.00 |
| OverRefusal-XST | 0.605 | 0.69 | 0.48 | 0.590 | 0.85 | 0.67 | 0.500 | 1.00 | 1.00 |

## Appendix E Latent Semantic Annotation

To semantically interpret the SAE latents associated with refusal behavior, we developed a two-round LLM-assisted annotation pipeline. First, we identified the top-10 latents exhibiting the highest refusal-moving scores from our contrastive analysis between refusal and compliance examples. For each selected latent, we extracted the 20 prompts with the highest activation values and tokenized them such that the most strongly activating tokens were marked with brackets to highlight the key triggering patterns. These annotated examples were then passed to GPT-4, which generated an initial semantic label, description, and characteristic patterns for each latent based on the bracketed activation contexts (Round 1: Initial Annotation). Next, the model received all latent annotations together and was prompted to organize them into a maximum of four coherent semantic categories, identifying commonalities and distinctions across latents (Clustering Stage). Finally, GPT-4 was provided with each latent’s original description alongside its assigned cluster description and asked to produce refined labels that better reflect the latent’s role within its category (Round 2: Category-Aware Refinement). Crucially, this annotation pipeline was executed under multiple settings: once on a combined dataset (mixing XSTest, WildGuard, CoCoNot, and SorryBench, balanced to 1,320 examples) and separately on individual dataset subcategories (e.g., CoCoNot’s “Humanizing requests,” “Unsupported requests,” or SorryBench’s “Hate speech generation,” “Assistance with crimes or torts”). By annotating the same latents across different data contexts, we reveal their multi-faceted nature—a single latent may receive distinct semantic interpretations depending on which examples most strongly activate it in a given setting. For instance, latent 550 is labeled "Commissioned public-harm communications" in the combined analysis focusing on hate speech and defamation, but receives the label "Unsupported modality request" when analyzed on CoCoNot’s capability-focused examples. Similarly, latent 7137 is interpreted as “Missing-input/external-content requests” in the combined setting, but as “Malware and doxxing requests” in SorryBench’s crime-focused subcategory and “Translate/Transcribe Attached Media” in CoCoNot’s unsupported requests. This polysemous behavior suggests that refusal-related latents encode context-sensitive features rather than fixed categorical detectors.

**Table 15: Common and distinct SAE latents identified through semantic annotation across dataset settings. Common latents emerge consistently across all settings (combined and subcategory-specific), representing domain-general refusal features. Distinct latents appear only in specific subcategories, revealing specialized detectors for particular harm types (SorryBench) or capability limitations (CoCoNot).**
| Type | Latent | Source | Semantic Interpretation |
| --- | --- | --- | --- |
| Common | 550 | All Data | Harmful content requests |
| 1779 | All Data | Prohibited/illicit content |  |
| 5176 | All Data | Disinformation & defamation |  |
| 6768 | All Data | Policy-violating directives |  |
| 7137 | All Data | Capability boundary violations |  |
| 13393 | All Data | Impossible knowledge requests |  |
| 5638 | All Data | Anthropomorphization queries |  |
| Distinct | 5958 | SorryBench: Hate Speech | Ridicule-focused harassment authoring |
| 15410 | SorryBench: Hate Speech | Platformed harassment and veiled threats |  |
| 3181 | SorryBench: Hate Speech | Coercive communications (threats, blackmail) |  |
| 12536 | SorryBench: Crimes/Torts | Crime how-to requests (evasion, theft, fraud) |  |
| 11313 | SorryBench: Crimes/Torts | Criminal facilitation (terrorism, trafficking) |  |
|  | 14854 | CoCoNot: Humanizing | Personalized advice requests (financial, medical) |
|  | 13439 | CoCoNot: Humanizing | Future prediction requests (stocks, elections) |
|  | 14422 | CoCoNot: Humanizing | Anthropomorphic expertise queries |
|  | 481 | CoCoNot: Unsupported | Visual art requests (draw, sketch, illustrate) |
|  | 1934 | CoCoNot: Unsupported | Audio/ASL transcription and translation |
|  | 2506 | CoCoNot: Unsupported | Artworks to unsupported modalities |
|  | 3834 | CoCoNot: Unsupported | External audio transcription |
|  | 13756 | CoCoNot: Unsupported | Fictional/ancient language translation |

## Appendix F Refusal Latent Overlap between Splits

We show in Table [16](https://arxiv.org/html/2602.02132v1#A6.T16) the overlap of top-3000 refusal latents across all the layers between the directions of different splits, while Tables [17](https://arxiv.org/html/2602.02132v1#A6.T17),[18](https://arxiv.org/html/2602.02132v1#A6.T18) and [19](https://arxiv.org/html/2602.02132v1#A6.T19) show the overlap between the top-1000 latents per split for the layer 9, 20, 31, respectively.

**Table 16: Overlap of top-3000 refusal latent across all layers and refusal directions**
|  | Humanizing–CCN | Incomplete–CCN | Indeterminate–CCN | Safety–CCN | Unsupported–CCN | HateSpeech–SB | CrimeAssistance–SB | Inappropriate–SB | Advice–SB | SafetyCore–WGM | OverRefusal–XST |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Humanizing–CCN | 3000 | 2594 | 2836 | 2346 | 2517 | 2836 | 2789 | 2811 | 2869 | 2811 | 2823 |
| Incomplete–CCN | 2594 | 3000 | 2605 | 2395 | 2551 | 2565 | 2567 | 2571 | 2569 | 2588 | 2551 |
| Indeterminate–CCN | 2836 | 2605 | 3000 | 2347 | 2546 | 2816 | 2816 | 2813 | 2804 | 2794 | 2789 |
| Safety–CCN | 2346 | 2395 | 2347 | 3000 | 2398 | 2334 | 2368 | 2386 | 2323 | 2401 | 2285 |
| Unsupported–CCN | 2517 | 2551 | 2546 | 2398 | 3000 | 2485 | 2504 | 2509 | 2492 | 2522 | 2480 |
| HateSpeech–SB | 2836 | 2565 | 2816 | 2334 | 2485 | 3000 | 2828 | 2862 | 2866 | 2835 | 2878 |
| CrimeAssistance–SB | 2789 | 2567 | 2816 | 2368 | 2504 | 2828 | 3000 | 2831 | 2815 | 2809 | 2784 |
| Inappropriate–SB | 2811 | 2571 | 2813 | 2386 | 2509 | 2862 | 2831 | 3000 | 2824 | 2863 | 2793 |
| Advice–SB | 2869 | 2569 | 2804 | 2323 | 2492 | 2866 | 2815 | 2824 | 3000 | 2825 | 2877 |
| SafetyCore–WGM | 2811 | 2588 | 2794 | 2401 | 2522 | 2835 | 2809 | 2863 | 2825 | 3000 | 2787 |
| OverRefusal–XST | 2823 | 2551 | 2789 | 2285 | 2480 | 2878 | 2784 | 2793 | 2877 | 2787 | 3000 |
| Unique | 27 | 165 | 32 | 257 | 177 | 34 | 11 | 27 | 17 | 24 | 26 |

**Table 17: Overlap of top-1000 refusal latent for layer 9 and refusal directions**
|  | Humanizing–CCN | Incomplete–CCN | Indeterminate–CCN | Safety–CCN | Unsupported–CCN | HateSpeech–SB | CrimeAssistance–SB | Inappropriate–SB | Advice–SB | SafetyCore–WGM | OverRefusal–XST |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Humanizing–CCN | 1000 | 915 | 937 | 797 | 845 | 963 | 912 | 948 | 973 | 961 | 956 |
| Incomplete–CCN | 915 | 1000 | 923 | 819 | 876 | 911 | 915 | 919 | 914 | 916 | 904 |
| Indeterminate–CCN | 937 | 923 | 1000 | 799 | 859 | 930 | 933 | 934 | 932 | 934 | 918 |
| Safety–CCN | 797 | 819 | 799 | 1000 | 829 | 797 | 806 | 813 | 796 | 801 | 787 |
| Unsupported–CCN | 845 | 876 | 859 | 829 | 1000 | 841 | 863 | 854 | 843 | 845 | 838 |
| HateSpeech–SB | 963 | 911 | 930 | 797 | 841 | 1000 | 913 | 961 | 962 | 964 | 958 |
| CrimeAssistance–SB | 912 | 915 | 933 | 806 | 863 | 913 | 1000 | 928 | 913 | 914 | 896 |
| Inappropriate–SB | 948 | 919 | 934 | 813 | 854 | 961 | 928 | 1000 | 949 | 959 | 936 |
| Advice–SB | 973 | 914 | 932 | 796 | 843 | 962 | 913 | 949 | 1000 | 964 | 955 |
| SafetyCore–WGM | 961 | 916 | 934 | 801 | 845 | 964 | 914 | 959 | 964 | 1000 | 951 |
| OverRefusal–XST | 956 | 904 | 918 | 787 | 838 | 958 | 896 | 936 | 955 | 951 | 1000 |
| Unique | 5 | 16 | 10 | 79 | 51 | 2 | 19 | 4 | 4 | 5 | 12 |

**Table 18: Overlap of top-1000 refusal latent for layer 20 and refusal directions**
|  | Humanizing–CCN | Incomplete–CCN | Indeterminate–CCN | Safety–CCN | Unsupported–CCN | HateSpeech–SB | CrimeAssistance–SB | Inappropriate–SB | Advice–SB | SafetyCore–WGM | OverRefusal–XST |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Humanizing–CCN | 1000 | 863 | 957 | 789 | 834 | 936 | 944 | 945 | 952 | 932 | 936 |
| Incomplete–CCN | 863 | 1000 | 862 | 793 | 842 | 849 | 848 | 847 | 852 | 856 | 849 |
| Indeterminate–CCN | 957 | 862 | 1000 | 780 | 842 | 951 | 948 | 952 | 947 | 935 | 951 |
| Safety–CCN | 789 | 793 | 780 | 1000 | 793 | 771 | 789 | 788 | 780 | 803 | 767 |
| Unsupported–CCN | 834 | 842 | 842 | 793 | 1000 | 824 | 820 | 826 | 826 | 835 | 820 |
| HateSpeech–SB | 936 | 849 | 951 | 771 | 824 | 1000 | 951 | 960 | 956 | 936 | 977 |
| CrimeAssistance–SB | 944 | 848 | 948 | 789 | 820 | 951 | 1000 | 963 | 957 | 951 | 955 |
| Inappropriate–SB | 945 | 847 | 952 | 788 | 826 | 960 | 963 | 1000 | 960 | 954 | 954 |
| Advice–SB | 952 | 852 | 947 | 780 | 826 | 956 | 957 | 960 | 1000 | 944 | 958 |
| SafetyCore–WGM | 932 | 856 | 935 | 803 | 835 | 936 | 951 | 954 | 944 | 1000 | 934 |
| OverRefusal–XST | 936 | 849 | 951 | 767 | 820 | 977 | 955 | 954 | 958 | 934 | 1000 |
| Unique | 10 | 43 | 7 | 87 | 53 | 3 | 7 | 6 | 7 | 7 | 1 |

**Table 19: Overlap of top-1000 refusal latent for layer 31 and refusal directions**
|  | Humanizing–CCN | Incomplete–CCN | Indeterminate–CCN | Safety–CCN | Unsupported–CCN | HateSpeech–SB | CrimeAssistance–SB | Inappropriate–SB | Advice–SB | SafetyCore–WGM | OverRefusal–XST |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Humanizing–CCN | 1000 | 816 | 942 | 760 | 838 | 937 | 933 | 918 | 944 | 918 | 931 |
| Incomplete–CCN | 816 | 1000 | 820 | 783 | 833 | 805 | 804 | 805 | 803 | 816 | 798 |
| Indeterminate–CCN | 942 | 820 | 1000 | 768 | 845 | 935 | 935 | 927 | 925 | 925 | 920 |
| Safety–CCN | 760 | 783 | 768 | 1000 | 776 | 766 | 773 | 785 | 747 | 797 | 731 |
| Unsupported–CCN | 838 | 833 | 845 | 776 | 1000 | 820 | 821 | 829 | 823 | 842 | 822 |
| HateSpeech–SB | 937 | 805 | 935 | 766 | 820 | 1000 | 964 | 941 | 948 | 935 | 943 |
| CrimeAssistance–SB | 933 | 804 | 935 | 773 | 821 | 964 | 1000 | 940 | 945 | 944 | 933 |
| Inappropriate–SB | 918 | 805 | 927 | 785 | 829 | 941 | 940 | 1000 | 915 | 950 | 903 |
| Advice–SB | 944 | 803 | 925 | 747 | 823 | 948 | 945 | 915 | 1000 | 917 | 964 |
| SafetyCore–WGM | 918 | 816 | 925 | 797 | 842 | 935 | 944 | 950 | 917 | 1000 | 902 |
| OverRefusal–XST | 931 | 798 | 920 | 731 | 822 | 943 | 933 | 903 | 964 | 902 | 1000 |
| Unique | 12 | 106 | 15 | 91 | 73 | 6 | 8 | 17 | 6 | 12 | 13 |