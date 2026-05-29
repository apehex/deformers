Title: Representation Bending for Large Language Model Safety
ArXiv: 2504.01550
Authors: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi
Sections: 51
Estimated tokens: 31.2k

## Contents
- 1 Introduction
- 2 Related Work
  - Unlearning.
  - Activation Steering.
  - Safety Representation Engineering.
- 3 Representation Bending
  - Loss Terms.
  - Choice of Layers.
  - Where in architecture.
- 4 Experiments
  - 4.1 Experiment Details
    - Comparisons.
    - Datasets.
    - Training Details.
  - 4.2 Robustness Against Jailbreak Attacks
    - Benchmarks.
    - Results.
  - 4.3 Pareto-Frontier of Safety, Usability, and Capability
    - Benchmarks.
    - Results.
  - 4.4 General Applicability Across LLM Architectures and Sizes
    - Models.
    - Results.
  - 4.5 Internal Behavior of Model
  - 4.6 Analysis of Activations for Safe and Unsafe Prompts
- 5 Conclusion
- 6 Limitations
- 7 Broader Impact and Potential Risks
- Acknowledgments
- References
- Appendix A Experiment Details
  - A.1 Training Datasets
  - A.2 Comparison Baselines
  - A.3 List of Models and Datasets
  - A.4 RepBend Hyper Parameters
  - A.5 Evaluation Benchmarks
    - Jailbreak Attack Benchmarks
    - Over-refusal Benchmarks
    - General Capability Benchmarks
    - General Capability Benchmarks
- Appendix B More Experiment Results
  - B.1 Full Comparison
  - B.2 Results on Additional Jailbreak Attacks
  - B.3 Impact of cos_sim loss
    - Results.
  - B.4 Hyperparameter Sensitivity Ablation Study
  - B.5 Comparison with Controllable Text Generation
  - B.6 Results on General Capabilities
  - B.7 Another Representation Analysis Example
  - B.8 Generation Results
  - B.9 Computing infrastructure

## Abstract

Abstract Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks – ranging from harmful content generation to broader societal harms – pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend , a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering – simple vector arithmetic for steering model’s behavior during inference – to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities. 1 1 1 Model and code: github.com/AIM-Intelligence/RepBend

## 1 Introduction

Large language models (LLMs) have become versatile tools with notable capabilities, showing promise as general-purpose task solvers. Their growing adoption across many industries and for personal use makes it increasingly important to ensure they are safe and do not cause harmful or catastrophic outcomes Hendrycks et al. (2023); Phuong et al. (2024). LLMs are typically fine-tuned for both instruction-following and safety, to become both helpful and harmless – to obey and provide helpful responses to benign requests, and to refuse harmful ones Bai et al. (2022). Nevertheless, they still exhibit harmful behaviors, especially to adversarial manipulations Wei et al. (2024a); Zou et al. (2023b); Andriushchenko and Flammarion (2024); Pelrine et al. (2023), or even fine-tuning Lermen et al. (2023); Zhan et al. (2024); Anwar et al. (2024); Qi et al. (2024b). These attacks can bypass safety training of LLMs and result in models that generate harmful responses.

Figure: Figure 1: RepBend bends the safe and unsafe representations spaces of the model to be far and distinguishable from each other (Right). Before applying RepBend, the model cannot tell if “To make a bomb …” is safe or unsafe, as those regions in the activation space are close and intertwined to each other (Left).
Refer to caption: https://arxiv.org/html/2504.01550/extracted/6623400/figures/Representation-2.png

Additionally, future unsafe LLMs can do societal harms in ways that are not yet analyzed or perceived, considering the potential emergence of Artificial General Intelligence (AGI) in the coming years or decades Ngo et al. (2024). The rapid advances in LLMs not only expand their capabilities, but also their potential for societal harm Anwar et al. (2024); Birhane et al. (2023); Ganguli et al. (2022); Anthropic (Sep 19, 2023). With increased agency and autonomy, and as large models are scaled and deployed in higher-stakes settings in the real world, ensuring that these models are safe is critical Chan et al. (2023); Anthropic (Sep 19, 2023).

Existing safety training methods (e.g., SFT Brown et al. (2020), DPO Rafailov et al. (2024a), RLHF Christiano et al. (2017)) are frequently bypassed Zou et al. (2023b); Wei et al. (2024a); Andriushchenko et al. (2024); Schwinn et al. (2024) and are subject to “shallow safety alignment” Qi et al. (2024a). To combat these, adversarial training is proposed, where during training specific attack methods are remedied Mazeika et al. (2024); Jiang et al. (2024b). However, adversarial training only addresses specific types of attacks and does not generalize to attacks that were unseen during training. System-level defenses, such as input and output filters Han et al. (2024), may be hard to scale due to the frequency of checks for every query, need to be upgraded with new and intelligent attacks, and do not make the underlying model inherently safer Zou et al. (2024), as they only limit the model’s exposure to unsafe content.

RepBend drives inspiration from these works, and also from the activation steering literature (Turner et al., 2024; Panickssery et al., 2023; Cao et al., 2024b). Activation steering forms “steering vectors” by simply taking the difference of the activations for safe and unsafe prompts, and performs simple arithmetic (e.g., addition or subtraction) to change the model’s behavior during inference. Activation steering has limitations for in- and out-of-distribution generalizability, and can compromise model’s general reasoning capabilities Tan et al. (2024); Anthropic (Oct 25, 2024).

RepBend brings the idea of activation steering to fine-tuning, by defining a loss function whose terms are based on simple vector difference (details in [Section 3](https://arxiv.org/html/2504.01550v3#S3)). The resulting loss is not only simple and intuitive compared to representation engineering methods ([Section 3](https://arxiv.org/html/2504.01550v3#S3)), but also generalizes for out-of-distribution examples. RepBend keeps the model’s general reasoning capability, and it achieves the state-of-the-art performance. RepBend achieves up to 95% improvement in reducing attack success rates across diverse jailbreak benchmarks, with minimal impact on model usability and general capabilities. With RepBend, we advanced the Pareto frontier of
of safety vs. general capability compared to the other
state-of-the-art methods, including NPO Zhang et al. (2024), RMU Li et al. (2024a), Circuit Breaker Zou et al. (2024), and Task Arithmetic Ilharco et al. (2022). The results are discussed in [Section 4](https://arxiv.org/html/2504.01550v3#S4).

## 2 Related Work

Recent studies have explored the limitations of alignment techniques in LLMs. Lin et al. (2023) demonstrates that alignment can be bypassed through carefully designed in-context prompts, while Wei et al. (2024b) reveals that even minimal structural changes can significantly degrade alignment, raising concerns about the robustness of safety measures. Hsu et al. (2024) introduces a low-rank adaptation technique that reduces safety risks during model fine-tuning. Further probing latent representations, Ball et al. (2024) analyzes how jailbreaks succeed by exploiting latent space dynamics, while Treutlein et al. (2024) shows that LLMs infer
latent information from evidence distributed across training documents, underscoring the complexity of internal model representations and the difficulty of safety monitoring.

#### Unlearning.

Conventional unlearning aims to update the weights of a model to remove specific knowledge, and usually is about narrow topics (e.g., Harry Potter) or fictional information Eldan and Russinovich (2023); Maini et al. (2024); Wang et al. (2024). For reviews of existing works, see Liu et al. (2024). Authors in Maini et al. (2024); Jin et al. (2024); Hong et al. (2024); Shi et al. (2024) propose real-world LLM unlearning benchmarks that consider task settings, knowledge sources, privacy, scalability, and even internal model parameters. Although fine-grained control is useful, our work tackles broader notions of undesirable outputs. In the context of language models, previous work explores concepts like fairness, privacy, safety or hallucinations Jang et al. (2023); Yao et al. (2023); Liu et al. (2024). Recently NPO Zhang et al. (2024) is proposed; a simple alignment-inspired method that could be extended to unlearn hazardous knowledge. In this work we compare NPO’s performance with RepBend.

#### Activation Steering.

A general tactic steers language models away from generating undesirable text during inference. Liu et al. (2023) proposes in-context vectors that encode and replace in-context examples and uses it with simple vector arithmetic during inference, while other works control LLMs by “steering vector” activations Turner et al. (2024); Panickssery et al. (2023); Cao et al. (2024b, a). Similarly, Jorgensen et al. (2023) applies mean clustering to find better steering vectors than normal averaging, and Qiu et al. (2024) proposes spectral editing of activations, an activation editing method to guide LLMs to generate desirable outputs through spectral decomposition. Critically, these methods focus on inference-time changes to the forward pass. While this makes activation steering widely applicable, activation steering has limitations for in- and out-of-distribution (OOD) generalizability, and can compromise model’s general reasoning capabilities Tan et al. (2024); Anthropic (Oct 25, 2024). RepBend has OOD generalizability and good general reasoning score ([Section 4.2](https://arxiv.org/html/2504.01550v3#S4.SS2)).

#### Safety Representation Engineering.

Closely related to our work, are safety frameworks that change the representations during train-time Zou et al. (2023a). For example, authors in Rosati et al. (2024) push harmful
representations towards random noise to disturb the unsafe space such that they are harder to recover. R2D2 Mazeika et al. (2024) fine-tunes LLMs on a dynamic pool of harmful prompts continually updated by an optimization-based red teaming method. RMU Li et al. (2024a) selectively forgets unsafe knowledge while limiting general capabilities loss. Circuit Breaker (CB) Zou et al. (2024) improved upon RMU, and showed that we can “short circuit” the representations that are responsible for harmful outputs. Zhang et al. (2024). Motivated by CB, but different from it, we designed a loss function that operates simply by taking difference of activations, similar to activation steering, and we found that it performs better than CB and all other methods. Results are in [Section 4](https://arxiv.org/html/2504.01550v3#S4).

## 3 Representation Bending

The core idea of representation bending is to “bend” the representation space of the model, ensuring that safe and unsafe states – latent knowledge encoded in the activation space – become distinctly separated and distant from each other. Specifically, representation bending pushes away the representations associated with harmful states from representations linked to safe behaviors, enhancing the model’s ability to clearly differentiate between safe and unsafe outcomes ([Figure 1](https://arxiv.org/html/2504.01550v3#S1.F1)).

To do so, we have to bring the model into its unsafe and safe representation states: we input safe and unsafe text to the model, to elicit the targeted representations, and read the activations of the model for those texts Liu et al. (2023); Turner et al. (2024); Zou et al. (2024, 2023a). We first need to gather a dataset $D$ consisting of safe and unsafe texts. We need text that is “unsafe” and also text that is considered “safe” to elicit model’s representations. Details of the datasets we used discussed in [Section A.1](https://arxiv.org/html/2504.01550v3#A1.SS1).

Figure: Algorithm 1 RepBend

RepBend is shown in Algorithm 1. RepBend tries to change the representations of model $M$ by minimizing a loss function over a fixed number of steps during fine-tuning. For improved performance, RepBend updates the model parameters with LoRA Hu et al. (2021). Model with LoRA is referred to as (and can be seen as an independent) model $M^{\prime}$. Since we only update the LoRA parameters, $M^{\prime}$ at the end of the algorithm would be the desired safe model.

#### Loss Terms.

Line 12 shows four terms in the loss function. First loss term keeps the model’s representation close to the safe representations (small L2 difference); this can be seen as “retain loss.” Second loss term is the opposite of the first one: it pushes the representations of the model to be far away from unsafe representations (large L2 difference). This term can be seen as “forget loss.” These first two loss terms together separate the unsafe representation space from the safe representation space. Third, the cosine similarity loss term stabilizes the model responses under the unsafe queries toward a single “refusal-like” direction (e.g., “I am sorry I cannot help” or “As an AI assistant, I am unable to …”), helping the model consistently reject them rather than producing random or inconsistent outputs. The cosine similarity loss stabilizes the representation bending process by encouraging harmful representations to align closely with one another. Finally, the KL divergence loss term keeps the model’s general capabilities intact by encouraging its safe outputs to remain aligned with the original distribution.

We now explain all the steps in Algorithm 1. We sample a batch of safe text from safe sets, $P_{s}$, safe prompts, and $P_{us}$, unsafe prompts followed by safe responses, respectively, and feed them into the LLM to elicit safe representations (line 5). We then obtain the vector $v_{s}$, the difference between safe representations of $M$ and $M^{\prime}$ (line 6). For simpler notation, we show $M_{L}^{I}(.)$ as $M(.)$, denoting representations of model $M$ for a set of layers $L$ and a set of token positions $I$. (Prior work typically uses for $I$ positions of last tokens of prompt, or first tokens of response, or all tokens of input von Rütte et al. (2024); Gurnee and Tegmark (2023); Zou et al. (2023a); Turner et al. (2024); Panickssery et al. (2023); Zou et al. (2024)). We sample a batch of unsafe text from unsafe set, $P_{uu}$, and get the unsafe representation difference vector $v_{u}$ (line 7-8). Since we want the model not to generate unsafe text, we like $v_{u}$ to have big L2 norm ($M^{\prime}$ be far from $M$ for unsafe representation), while we want $v_{s}$ to have small L2 norm ($M^{\prime}$ not get far from $M$ for safe representation). Hence the L2 norm of these vectors are added to the loss term (line 12) with respective negative and positive signs.

Since we also want to have stability while bending the representations, we add two loss terms with distinct purposes: KL divergence loss term between output logits of $M$ and $M^{\prime}$ is added so that the model still retains its general capability when input is safe. A similarity term denoted by $\texttt{cos_sim}(.)$ is added to encourage similarity between outputs of unsafe prompts, given the fact that the response of a safe LLM to most unsafe prompts would begin by refusal texts (e.g.,“I’m Sorry, I am unable to assist”) Arditi et al. (2024b); Liu et al. (2023); von Rütte et al. (2024). In lines 9-10, a batch of unsafe text are sampled and are added to a set $A_{u}$. Method $\texttt{cos_sim}(.)$ computes average of cosine similarities between all pairs in the set $A_{u}$. Since the batch size is often not big for LLMs, the complexity of this loss is manageable. By minimizing this loss for model $M^{\prime}$, this algorithm makes the LLM safe.

#### Choice of Layers.

Finding the set of layers to do the intervention of RepBend is non-trivial and largely dependent on the architecture and practical considerations Wu et al. (2024); Arditi et al. (2024b); Li et al. (2024b). Prior work has considered early layers Turner et al. (2024); Li et al. (2024a), middle layers Li et al. (2024b); Zou et al. (2024), later layers Zou et al. (2023a); Cao et al. (2024b); Panickssery et al. (2023), and all layers Liu et al. (2023) for intervening on activations. We need to find the layers for which representation bending can have the best impact. We hypothesized that mid to later layers are responsible for the generation of output, and targeting them could be good for representation bending. This aligns well with prior work Zou et al. (2023a); Panickssery et al. (2023) that shows behavior clustering and emotion representations emerge around half or one-third of the way through the layers, indicating rich representations. We validated our hypotheses in our experiments too ([Section 4.5](https://arxiv.org/html/2504.01550v3#S4.SS5) and [Section B.4](https://arxiv.org/html/2504.01550v3#A2.SS4)), where representation bending works best in mid to later layers (layers 20 and after).

Figure: Figure 2: Average Attack Success Rate (ASR - lower is better) across five black-box and three white-box access attacks on Mistral 7B and Llama3 8B models.
Refer to caption: https://arxiv.org/html/2504.01550/x1.png

#### Where in architecture.

Our work focuses on decoder-only, autoregressive LLMs. Each LLM accepts $n$ tokens, embeds into $\{x_{i}\}_{i=1}^{n}$ inputs, transforms into $\{h_{ij}\}_{j=1}^{m}$ latents for each of $m$ transformer blocks, and finally unembeds to form logits. Each transformer block is comprised of two modules – a self-attention $\textsc{Attn}(\cdot)$ module and multi-layer perceptron (MLP) $\textsc{Mlp}(\cdot)$. Both parts are preceded by a layer norm $n(\cdot)$ and followed by a skip connection. The architecture can be simply be represented by:

$$ $\displaystyle h_{i1}$ $\displaystyle=\textsc{Attn}(norm(x_{i}))$ $\displaystyle h_{i2}$ $\displaystyle=x_{i}+h_{i1}$ $\displaystyle h_{i3}$ $\displaystyle=\textsc{Mlp}(norm(h_{i2}))$ $\displaystyle h_{i4}$ $\displaystyle=h_{i2}+h_{i3}$ $$

Following existing research, we apply representation bending in the residual stream of LLM Arditi et al. (2024b); Elhage et al. (2021); von Rütte et al. (2024); Yu et al. (2024); Panickssery et al. (2023); Zou et al. (2024), in particular the residual stream at the output of the block $h_{i4}$. We also experimented with other possibilities when activations are obtained from different parts of the transformer block, namely activations at the input of $\textsc{Attn}(\cdot)$ (after first layer normalization $h_{i1}$), after passing through the self-attention block $h_{i2}$, and at the input of $\textsc{MLP}(\cdot)$ after the second layer normalization. We observed not much variability, and decided to go with residual streams as in prior works.

## 4 Experiments

We validate the effectiveness of RepBend on the state-of-the-art instruction-tuned LLMs, achieving up to 95% reduction in attack success rate (ASR) across diverse jailbreak benchmarks, with minimal impact on model usability and general capabilities.
First we show that RepBend reduces ASR across different benchmark, and its overall performance is the best in white-box attack model. Its performance for black-box attack is also reasonable, and in both attack scenarios with minimal loss of general capability and usability.
Additionally, we use logit lens to analyze the internals of the model to show the effectiveness of RepBend in not just the output (what model “says“), but also the internals (what model “thinks“).

### 4.1 Experiment Details

**Table 1: Jailbreak attack success rates for Mistral 7B Instruct-v0.2 and Llama3 8B Instruct. * indicates a publicly-available safety-tuned model. Each cell indicates the attack success rate (ASR), the fraction of requests with which the model complies. Lower ASR is better. The best performance is in bold, and the second best is underlined. *WildGuardTest* is an in-distribution (ID) benchmark; other benchmarks test out-of-distribution (OOD).**
| Domain | Benchmark | Mistral 7B Instruct v0.2 | Llama3 8B Instruct |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TA | NPO | RMU | CB | R2D2* | CB* | RepBend<br>(Ours) | TA | NPO | RMU | CB | CB* | RepBend<br>(Ours) |  |  |
| Black-box Jailbreak |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ID | WildGuardTest | 13.62 | 2.80 | 7.48 | 16.29 | 44.46 | 8.54 | 8.95 | 7.08 | 0.95 | 11.75 | 7.88 | 3.74 | 7.34 |
| OOD | HarmBench | 3.75 | 0.06 | 3.75 | 16.25 | 5.63 | 13.44 | 1.56 | 1.87 | 2.19 | 9.37 | 3.75 | 13.44 | 0.31 |
| DAN | 12.06 | 0.50 | 1.81 | 4.69 | 12.06 | 1.56 | 0.50 | 5.25 | 0.38 | 0.50 | 0.25 | 0.56 | 0.75 |  |
| TrustLLM Jailbreak | 4.75 | 0.25 | 32.25 | 19.00 | 3.50 | 24.25 | 4.75 | 0.50 | 0.00 | 10.25 | 0.75 | 10.75 | 3.50 |  |
| PAP | 16.56 | 2.19 | 14.37 | 13.13 | 19.37 | 9.38 | 1.87 | 14.06 | 1.88 | 7.50 | 1.88 | 4.69 | 3.12 |  |
|  | Average | 10.15 | 1.16 | 11.93 | 13.87 | 17.00 | 11.43 | 3.53 | 5.75 | 1.08 | 7.87 | 2.90 | 6.64 | 3.00 |
| White-box Jailbreak |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OOD | GCG | 61.56 | 10.00 | 11.56 | 25.00 | 8.44 | 9.37 | 5.00 | 51.25 | 7.50 | 11.87 | 4.37 | 3.44 | 2.50 |
| Prefilling | 80.83 | 8.75 | 7.50 | 5.00 | 47.08 | 5.42 | 0.83 | 83.34 | 20.42 | 6.67 | 7.92 | 3.33 | 4.17 |  |
| Input Embed | 30.42 | 22.08 | 50.83 | 55.83 | 44.17 | 21.67 | 2.50 | 74.17 | 51.67 | 28.33 | 41.67 | 23.75 | 3.33 |  |
|  | Average | 55.63 | 13.61 | 23.30 | 28.05 | 33.23 | 12.15 | 2.78 | 69.59 | 26.53 | 15.62 | 17.99 | 10.17 | 3.33 |
| Total Average | 23.14 | 5.83 | 16.19 | 19.19 | 23.09 | 11.70 | 3.25 | 29.69 | 10.62 | 10.78 | 8.56 | 7.96 | 3.13 |  |

#### Comparisons.

We compare RepBend with several state-of-the-art methods. In safety representation engineerng literature, RMU Li et al. (2024a) and Circuit Breaker (CB) Zou et al. (2024) are the state-of-the-art methods that are directly related to our work.
We train models using these methods in our experiment infrastructure for higher performance and fair comparison (e.g., we found out training a CB model with our setup comes with better performance in some settings than the model publicly available).
We also compare RepBend with two related unlearning methods that can be applied to safety: Task Arithmetic (TA) Ilharco et al. (2022) and Negative Preference Optimization (NPO) Zhang et al. (2024), where harmful behavior can be unlearned by negating harmful task vector (in TA) or reversing alignment (in NPO).

Finally, we compare RepBend against public safety-aligned models, R2D2(^2^22https://huggingface.co/cais/zephyr_7b_r2d2) and CB (^3^33https://huggingface.co/GraySwanAI/Llama-3-8B-Instruct-RR)(^4^44 https://huggingface.co/GraySwanAI/Mistral-7B-Instruct-RR), trained by the authors of the respective works. Further details are provided in [Section A.2](https://arxiv.org/html/2504.01550v3#A1.SS2).

#### Datasets.

We curate the training dataset from three sources: WildGuardMix Han et al. (2024) and WildJailbreak Jiang et al. (2024b), which include harmful and harmless prompts, and UltraChat Ding et al. (2023), which contains general instruction-following data. From each dataset we randomly select 10,000 samples, and classify them into safe (harmless) and unsafe (harmful) groups. See [Section A.1](https://arxiv.org/html/2504.01550v3#A1.SS1) for more details.

#### Training Details.

We conduct experiments primarily using Mistral 7B v0.2 and Llama 3 8B (see list of all models in [Table 4](https://arxiv.org/html/2504.01550v3#A1.T4)). Experiments with other LLMs of various sizes are reported in [Section 4.4](https://arxiv.org/html/2504.01550v3#S4.SS4).
We initialize the models from instruction-tuned checkpoints and apply LoRA Hu et al. (2021) with rank and alpha of 16, targeting all linear layers in the model for all methods and RepBend. However, in RMU we only target the MLP layer in each transformer block, as per the paper Li et al. (2024a). Details are in [Section A.4](https://arxiv.org/html/2504.01550v3#A1.SS4).

### 4.2 Robustness Against Jailbreak Attacks

#### Benchmarks.

We evaluate RepBend against diverse jailbreak attacks, grouped into two categories: black-box and white-box access attacks.

Black-box access attacks assume no internal access to LLMs. The benchmarks include direct harmful requests (HarmBench Mazeika et al. (2024)), pre-generated jailbreak prompts (WildGuardTest Han et al. (2024), DAN Shen et al. (2023), TrustLLM-Jailbreak Huang et al. (2024)), and transformed instructions using external models (PAP Zeng et al. (2024)).

White-box access attacks assume some access to model internals. The benchmarks we use include GCG Zou et al. (2023b), Prefilling Andriushchenko et al. (2024); Vega et al. (2023), and Input Embed Schwinn et al. (2024). GCG finds an adversarial suffix for a prompt by maximizing compliance likelihood, Prefilling attack prefills the LLM’s response with a non-refusal beginning, and Input Embed produces adversarial embeddings instead of tokens.

WildGuardTest is the only in-distribution (ID) benchmark; other benchmarks are out-of-distribution (OOD), as they are based on either unseen adversarial prompts or unseen attack methods that were not part of the training or fine-tuning.
Additional details about the benchmarks are provided in [Section A.5](https://arxiv.org/html/2504.01550v3#A1.SS5). Unless otherwise stated, we use an open-source classifier Mazeika et al. (2024) to evaluate compliance of the responses.

**Table 2: Safety, Over-Refusal, and General Capability scores on Mistral 7B and Llama3 8B Models. Average ASR is the average of all 8 jailbreak attacks ([Table 1](https://arxiv.org/html/2504.01550v3#S4.T1)) and Average Capability is the average score of all 8 capability benchmarks where MTBench is 10$\times$ scaled. * indicates the publicly-available safety-tuned model. Overall is the average of scaled scores of the three axes: safety score $(1-\text{Average ASR})*100$, over-refusal score (average of 2 benchmarks) and general capability. The best performance is in bold, and the second best is underlined.**
| Model | Method | Safety | Over-refusal | General Capability | Overall<br>($\uparrow$) |  |
| --- | --- | --- | --- | --- | --- | --- |
| Average ASR<br>($\downarrow$) | XSTest<br>($\uparrow$) | Wildjailbreak:<br>Benign ($\uparrow$) | Average<br>Capability ($\uparrow$) |  |  |  |
| Mistral 7B<br>Instruct v0.2 | Original Weight Model | 60.64 | 85.78 | 100.00 | 59.18 | 63.81 |
| TA Ilharco et al. (2022) | 23.14 | 80.22 | 97.60 | 53.10 | 72.96 |  |
| NPO Zhang et al. (2024) | 5.83 | 68.89 | 70.00 | 53.94 | 74.52 |  |
| RMU Li et al. (2024a) | 16.19 | 78.44 | 90.40 | 47.32 | 71.85 |  |
| CB Zou et al. (2024) | 19.19 | 86.89 | 97.60 | 58.97 | 77.34 |  |
| R2D2* Mazeika et al. (2024) | 23.09 | 67.56 | 96.80 | 48.44 | 72.67 |  |
| CB* Zou et al. (2024) | 11.70 | 86.22 | 82.00 | 58.93 | 73.62 |  |
| RepBend (Ours) | 3.24 | 84.89 | 93.60 | 57.68 | 81.23 |  |
| Llama3 8B<br>Instruct | Original Weight Model | 34.00 | 85.11 | 92.00 | 67.14 | 73.90 |
| TA Ilharco et al. (2022) | 29.69 | 80.00 | 88.80 | 57.43 | 70.71 |  |
| NPO Zhang et al. (2024) | 10.62 | 74.45 | 43.20 | 66.71 | 71.65 |  |
| RMU Li et al. (2024a) | 10.78 | 76.89 | 72.40 | 54.84 | 72.90 |  |
| CB Zou et al. (2024) | 8.56 | 84.44 | 89.20 | 66.58 | 81.61 |  |
| CB* Zou et al. (2024) | 7.96 | 85.78 | 52.40 | 66.47 | 75.87 |  |
| RepBend (Ours) | 3.13 | 84.11 | 89.20 | 65.90 | 83.14 |  |

**Table 3: Evaluation results of RepBend on two additional LLM architectures. Target layers and loss weights are fixed and only learning rate and the update steps are tuned. Overall is the average of scaled scores of the three axes: safety score $(1-\text{Average ASR})*100$, over-refusal score (average of 2 benchmarks) and general capability score (average of 2 benchmarks with MTBench scaled by $10\times$).**
| Model | Method | Safety | Over-refusal | General Capability | Overall<br>($\uparrow$) |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harmbench<br>($\downarrow$) | WildguardTest<br>($\downarrow$) | XSTest<br>($\uparrow$) | Wildjailbreak:<br>Benign ($\uparrow$) | MTBench<br>($\uparrow$) | MMLU<br>($\uparrow$) |  |  |  |
| Gemma2 2B<br>Instruct | Original Weight | 11.56 | 28.70 | 78.67 | 98.80 | 7.35 | 57.98 | 78.12 |
| RepBend (Ours) | 6.56 | 1.34 | 70.34 | 82.80 | 7.37 | 58.14 | 79.51 |  |
| Qwen2.5 14B<br>Instruct | Original Weight | 17.19 | 33.11 | 86.67 | 100.0 | 8.71 | 79.60 | 83.85 |
| RepBend (Ours) | 7.50 | 6.67 | 82.22 | 99.60 | 9.14 | 78.89 | 89.66 |  |

#### Results.

[Table 1](https://arxiv.org/html/2504.01550v3#S4.T1) shows the attack success rate (ASR) – the model’s compliance to the attacks – across five black-box and three white-box access attack benchmarks on Mistral 7B and Llama 3 8B models. [Figure 2](https://arxiv.org/html/2504.01550v3#S3.F2) plots the total average ASR of white-box and black-box attacks combined. RepBend achieves the lowest average ASR (3.25 for Mistral and 3.13 for Llama) compared to other methods, improving refusal rates by 94.64% and 90.79% over the original instruction-tuned Mistral 7B and Llama 3 8B, respectively. We can see in [Table 1](https://arxiv.org/html/2504.01550v3#S4.T1) that in black-box attacks, RepBend achieves the lowest ASR for DAN and PAP in Mistral 7B and lowest ASR for HarmBench in Llama 3. For white-box attacks, RepBend most of the time achieves the lowest ASR. Moreover, these results demonstrate strong out-of-distribution (OOD) generalizability, as RepBend effectively refuses adversarial prompts and optimization attacks that were not seen during training.

While NPO performs well on black-box attacks, it does not perform well in white-box attacks because it only targets outputs of the model, and not the model’s internals for compliance to harmful requests. In contrast, RepBend delivers robust performance across both categories as it also changes the model internals via representation bending.

### 4.3 Pareto-Frontier of Safety, Usability, and Capability

Achieving an optimal balance between safety and general capability is crucial for LLMs. Over-refusing benign queries can compromise usability and improving safety should not degrade the model’s core capabilities. In this section we show that RepBend achieves Pareto-frontier of safety, usability, and general capability.

#### Benchmarks.

We evaluate over-refusal using XSTest Röttger et al. (2023) and WildJailbreak-Benign Jiang et al. (2024b). These datasets contain benign prompts with ambiguous wording (either intentional or unintentional) that seem harmful in form but contain no harmful intent. We measure compliance rate to these prompts using GPT-4.
For general capability, we use 8 benchmarks which includes MTBench Zheng et al. (2023), MMLU Hendrycks et al. (2020), BBH Suzgun et al. (2022), TruthfulQA Lin et al. (2021), ARC-C Clark et al. (2018), Winogrande Sakaguchi et al. (2021), GSM8K Cobbe et al. (2021), Codex-Eval Chen et al. (2021), to measure instruction-following capability, factual knowledge, and problem-solving performance including math and coding. Details are in [Section A.5](https://arxiv.org/html/2504.01550v3#A1.SS5).

#### Results.

[Table 2](https://arxiv.org/html/2504.01550v3#S4.T2) shows the three axes of safety, over refusal, and general capability of all methods. Overall score provides a single index to better see the Pareto-frontier performance by combining the 3 axis. To calculate it we level the score ranges to be the same and average them: safety score ($(1-\text{ASR})*100$ averaged on 8 benchmarks), over-refusal score (average of 2 benchmarks) and general capability score (average of 8 benchmarks where MTBench is 10$\times$ scaled). RepBend achieves the highest overall score, showing that RepBend does not compromise usability or capability for safety improvements. Whereas RepBend does not have the highest general capability and over-refusal scores, it has reasonable and acceptable scores comparable to the original model while maintaining the lowest ASR on jailbreak benchmarks. (general capability scores for each benchmark are in [Section B.6](https://arxiv.org/html/2504.01550v3#A2.SS6)). NPO shows good safety and general capability scores but struggles with distinguishing benign requests in over-refusal scores.
CB achieves balanced scores but falls short in safety compared to RepBend.

Figure: (a) Llama3 8B Instruct
Refer to caption: https://arxiv.org/html/2504.01550/extracted/6623400/figures/llama_original_1_cropped3.jpg

Additional results with more algorithms like SFT, DPO, WHP Eldan and Russinovich (2023), and the original model with a safe instructed prompt (e.g., “If someone asks you an unsafe or harmful prompt, do not answer”) are available in [Section B.1](https://arxiv.org/html/2504.01550v3#A2.SS1).
Moreover, further comparison with controllable text generation methods (e.g., a classifier-based method Yang and Klein (2021)) with better transferability are in [Section B.5](https://arxiv.org/html/2504.01550v3#A2.SS5)

### 4.4 General Applicability Across LLM Architectures and Sizes

#### Models.

To demonstrate the scalability of RepBend, we evaluate its performance across additional LLM architectures of varying sizes. The architectures include Gemma2 2B-Instruct, and Qwen2.5 14B-Instruct, where the models have different number of hidden layers. We evaluate safety, over-refusal, and general capability using the six benchmarks introduced in [Section 4.3](https://arxiv.org/html/2504.01550v3#S4.SS3).

#### Results.

[Table 3](https://arxiv.org/html/2504.01550v3#S4.T3) shows the performance of RepBend across the additional LLM architectures. With minimal hyperparameter tuning, RepBend successfully enhances safety while maintaining usability and capability. These results confirm the scalability of RepBend across diverse architectures and parameter sizes. Experiments on more white-box jailbreak attacks, SCAV (Xu et al., 2024), Weight Orthogonalization (Arditi et al., 2024a), and GuidedBench (Huang et al., 2025) are reported in [Section B.2](https://arxiv.org/html/2504.01550v3#A2.SS2).

### 4.5 Internal Behavior of Model

To analyze the impact of RepBend on internal representations, we need a way to visualized the model’s internals. We use the Logit Lens framework Nostalgebraist (2020); Belrose et al. (2023), which maps model’s representations from the latent space to the vocabulary space for interpretable analysis. Logit Lense hence allows us to see what the next predicted token in each layer is and it is widely used to investigate Transformer-based architectures Dar et al. (2023); Elhage et al. (2021).

[Figure 3](https://arxiv.org/html/2504.01550v3#S4.F3) (top) illustrates a harmful request from HarmBench Mazeika et al. (2024), and (bottom) shows the Logit Lens of (a) the instruction-tuned Llama3 8B baseline and (b,c) RepBend version of Llama3 8B Instruct. The heatmap cells show next token predictions across layers and colors show confidence levels (blue: high, red: low) during response generation. The baseline model produces a harmful response with a sharp increase in prediction certainty at later layers ([Figure 3](https://arxiv.org/html/2504.01550v3#S4.F3)(a)). In contrast, RepBend successfully refuses the request, significantly reducing certainty in harmful token predictions ([Figure 3](https://arxiv.org/html/2504.01550v3#S4.F3)(b)). We can also see layers beyond 20 are critical for next-token generation, confirming our choice of layers in RepBend.

Figure: (a) PCA of last layer activations of Llama3 8B Instruct (original weight).
Refer to caption: https://arxiv.org/html/2504.01550/x2.png

Furthermore, when RepBend is forced and initialized with a complying sequence (\\n\\n Subject: You Won’t), it transitions into generating random low-confidence tokens, halting harmful response generation ([Figure 3](https://arxiv.org/html/2504.01550v3#S4.F3)(c)). This demonstration of latent representations shows why RepBend achieves a low ASR, even on white-box jailbreak attacks. Additional results with longer sequences are provided in [Section B.7](https://arxiv.org/html/2504.01550v3#A2.SS7).

### 4.6 Analysis of Activations for Safe and Unsafe Prompts

[Figure 4](https://arxiv.org/html/2504.01550v3#S4.F4) shows PCA plots for activations of safe and unsafe prompts at last layer of Llama3 8B Instruct model, and measured mean Euclidean distance and Jensen Shannon divergence between the two activations of safe and unsafe prompts. We randomly chose 1,000 harmful prompts and 1,000 harmless prompts from wildjailbreak Jiang et al. (2024b), extracted activations of those prompts from each hidden layer, and analyzed the obtained activations.

[Figure 4](https://arxiv.org/html/2504.01550v3#S4.F4)(a) shows that in the original Llama model, representations of those harmless and harmful prompts are clustered into safe and unsafe regions in the space. [Figure 4](https://arxiv.org/html/2504.01550v3#S4.F4)(b) illustrates that RepBend bends the representations, breaking the clustered harmless and harmful regions, potentially (but arguably) getting rid of unsafe region.

Since PCA reduces and shows a high dimensional space in a two dimensional graph, we need other visualizations to to obtain more insights. We measured the distance between the distributions of activations of harmful and harmless prompts for RepBend and the original model.
[Figure 4](https://arxiv.org/html/2504.01550v3#S4.F4)(c) shows the layer-wise Euclidean distance between the activations of harmful and harmless prompts,
and [Figure 4](https://arxiv.org/html/2504.01550v3#S4.F4)(d) illustrates layer-wise Jensen Shannon divergence between the two sets of activations. These two plots indicate that RepBend “pushes apart” representations of unsafe prompts from safe prompts as a result of bent representation space.

In the original model, we can see that the representations of unsafe and safe prompts are near each other, which could mean the model cannot distinguish them, and it (potentially) can answer unsafe, even when the model is instructed to be safe. Conversely, when the model finetuned with RepBend, we can see that the distance between the representations of safe and unsafe inputs is larger, specially at the later layers, suggesting that the model (arguably) can distinguish safe and unsafe prompts. While RepBend distinctly separates harmful from harmless behavior, which increases the likelihood to correctly refuse the harmful requests, it may potentially cause gibberish outputs in some cases, as shown in [Section B.8](https://arxiv.org/html/2504.01550v3#A2.SS8).

## 5 Conclusion

The rapid adoption and scaling of LLMs amplify the urgency of ensuring their safety in diverse real-world applications. While existing safety measures provide partial solutions, they often fall short in generalizing to unseen attacks or maintaining model capabilities. Addressing these gaps, RepBend presents a novel fine-tuning approach inspired by activation steering, which optimally bends model representations to maximize safety while preserving general-purpose capabilities.

RepBend’s performance, including up to 95% reduction in attack success rates across benchmarks, demonstrates the method’s robustness and generalizability compared to state-of-the-art approaches. We showed that RepBend can be applied to different LLM architectures of varying sizes. Moreover, RepBend effectively balances safety and capability, advancing the Pareto frontier of these often-competing objectives.

This work highlights the potential of embedding safety principles directly into the training process to create inherently safer models. Future research can build on this foundation to explore broader applications, optimize computational efficiency, and address emerging challenges in the evolving landscape of LLM safety. Through continuous innovation, RepBend moves us closer to the responsible and secure deployment of advanced AI systems.

## 6 Limitations

An inherent limitation of any unlearning method, is robustness to re-learning hazardous knowledge. Similar to the finding for RMU Li et al. (2024a); Łucki et al. (2024), we also found out that if RepBend is fine-tuned with unsafe data, it can relearn the harmful content. This limitation of robustness has also been observed by several others Qi et al. (2024b); Lo et al. (2024); Lermen et al. (2023); Zhan et al. (2024), and is still an ongoing problem in the community Lynch et al. (2024); Shumailov et al. (2024), with some recent work to remedy this Rosati et al. (2024); Lyu et al. (2024). Future work can borrow techniques to make RepBend safe post-tuning.

We tested RepBend in this paper only on LLMs, and only on select open-source models. Even though we believe our methods are easily extensible to other models, the generalization of these findings are yet to be tested to other models, especially those at greater scale, including current state-of-the-art proprietary models.

Another limitation of RepBend is the effort for finding the best hyper parameters. We found the best working hyper parameters that show the results in this paper; however, the we do not claim that these set of values are the optimal, nor we think the effort to find the best hyper parameter is minimal. We found out that sometimes RepBend is sensitive towards the choice of hyper parameters, specially the loss coefficient parameters.

## 7 Broader Impact and Potential Risks

This work introduces a scalable and generalizable approach to model safety that can influence the development of future safety standards, fostering industry-wide adoption of safer practices in AI research and deployment. RepBend provides a practical framework for enhancing the safety of large language models, enabling their deployment in high-stakes domains such as healthcare, education, and legal systems, where the consequences of unsafe outputs could be catastrophic.

While RepBend reduces vulnerabilities to adversarial attacks, its success may lead to overconfidence in the safety of AI systems, potentially encouraging premature deployment in sensitive domains without rigorous oversight.

The advancement of safety mechanisms like RepBend may inadvertently escalate an arms race between AI safety researchers and malicious actors, leading to increasingly sophisticated attacks that may exploit yet-undiscovered vulnerabilities.

As with any technological development, there is a potential for misuse. We believe that disclosing details to prevent attacks on LLMs can be beneficial if used appropriately. However, it also poses the risk of malicious attackers exploiting this information to develop new methods for bypassing the defense. The proposed loss function in RepBend could be inverted or manipulated to intentionally create models that generate unsafe or toxic outputs. Or with the same RepBend algorithm, but different data, it can be used for different purpose if we change the data.

Lastly, finding the optimal set of hyper parameters may need large search, which directly translates to environmental concerns for energy usage.

## Acknowledgments

We would like to thank Hyungjoo Chae and Jiwan Chung for the discussions and their insightful ideas and comments.

This work was partially funded by an unrestricted gift from Google. This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00354218), the Institute of Information & communications Technology Promotion (IITP) grant funded by the Korea government (MSIT) (No. RS-2024-00457882 Artificial Intelligence Research Hub Project), and the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS-2025-02263598, Development of Self-Evolving Embodied AGI Platform Technology through Real-World Experience). This work was partly supported by IITP grants (No.RS-2022-II220077, No.RS-2022-II220113, No.RS-2022-II220959, No.RS-2022-II220871, No.RS-2022-II220951 (50%), No.RS-2021-II211343 (SNU AI), No.RS-2021-II212068 (AI Innovation Hub)) funded by the Korea government(MSIT) and the BK21 FOUR program, SNU in 2025.

## References

- AI@Meta (2024)
AI@Meta. 2024.
[Llama 3 model card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md).
- Andriushchenko et al. (2024)
Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion. 2024.
Jailbreaking leading safety-aligned llms with simple adaptive attacks.
*arXiv preprint arXiv:2404.02151*.
- Andriushchenko and Flammarion (2024)
Maksym Andriushchenko and Nicolas Flammarion. 2024.
Does refusal training in llms generalize to the past tense?
*arXiv preprint arXiv:2407.11969*.
- Anthropic (Oct 25, 2024)
Anthropic. Oct 25, 2024.
[Evaluating feature steering: A case study in mitigating social biases](https://www.anthropic.com/research/evaluating-feature-steering).
- Anthropic (Sep 19, 2023)
Anthropic. Sep 19, 2023.
[Anthropic’s responsible scaling policy,](https://www.anthropic.com/news/anthropics-responsible-scaling-policy).
- Anwar et al. (2024)
Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase, Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, et al. 2024.
Foundational challenges in assuring alignment and safety of large language models.
*arXiv preprint arXiv:2404.09932*.
- Arditi et al. (2024a)
Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. 2024a.
Refusal in language models is mediated by a single direction.
*arXiv preprint arXiv:2406.11717*.
- Arditi et al. (2024b)
Andy Arditi, Oscar Balcells Obeso, Aaquib Syed, Daniel Paleka, Nina Rimsky, Wes Gurnee, and Neel Nanda. 2024b.
Refusal in language models is mediated by a single direction.
In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*.
- Bai et al. (2023)
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023.
Qwen technical report.
*arXiv preprint arXiv:2309.16609*.
- Bai et al. (2022)
Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022.
Training a helpful and harmless assistant with reinforcement learning from human feedback.
*arXiv preprint arXiv:2204.05862*.
- Ball et al. (2024)
Sarah Ball, Frauke Kreuter, and Nina Panickssery. 2024.
Understanding jailbreak success: A study of latent space dynamics in large language models.
*arXiv preprint arXiv:2406.09289*.
- Belrose et al. (2023)
Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, and Jacob Steinhardt. 2023.
Eliciting latent predictions from transformers with the tuned lens.
*arXiv preprint arXiv:2303.08112*.
- Birhane et al. (2023)
Abeba Birhane, Vinay Prabhu, Sang Han, and Vishnu Naresh Boddeti. 2023.
On hate scaling laws for data-swamps.
*arXiv preprint arXiv:2306.13141*.
- Brown et al. (2020)
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020.
Language models are few-shot learners.
In *Advances in Neural Information Processing Systems*, volume 33, pages 1877–1901. Curran Associates, Inc.
- Cao et al. (2024a)
Yuanpu Cao, Tianrong Zhang, Bochuan Cao, Ziyi Yin, Lu Lin, Fenglong Ma, and Jinghui Chen. 2024a.
Personalized steering of large language models: Versatile steering vectors through bi-directional preference optimization.
*Advances in Neural Information Processing Systems*, 37:49519–49551.
- Cao et al. (2024b)
Zouying Cao, Yifei Yang, and Hai Zhao. 2024b.
Nothing in excess: Mitigating the exaggerated safety for llms via safety-conscious activation steering.
*arXiv preprint arXiv:2408.11491*.
- Chan et al. (2023)
Alan Chan, Rebecca Salganik, Alva Markelius, Chris Pang, Nitarshan Rajkumar, Dmitrii Krasheninnikov, Lauro Langosco, Zhonghao He, Yawen Duan, Micah Carroll, et al. 2023.
Harms from increasingly agentic algorithmic systems.
In *Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency*, pages 651–666.
- Chen et al. (2021)
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021.
Evaluating large language models trained on code.
*arXiv preprint arXiv:2107.03374*.
- Chen et al. (2022)
Yangyi Chen, Hongcheng Gao, Ganqu Cui, Fanchao Qi, Longtao Huang, Zhiyuan Liu, and Maosong Sun. 2022.
Why should adversarial perturbations be imperceptible? rethink the research paradigm in adversarial NLP.
In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*.
- Christiano et al. (2017)
Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017.
Deep reinforcement learning from human preferences.
*Advances in neural information processing systems*, 30.
- Clark et al. (2018)
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018.
Think you have solved question answering? try arc, the ai2 reasoning challenge.
*arXiv preprint arXiv:1803.05457*.
- Cobbe et al. (2021)
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021.
Training verifiers to solve math word problems, 2021.
*URL https://arxiv. org/abs/2110.14168*.
- Dar et al. (2023)
Guy Dar, Mor Geva, Ankit Gupta, and Jonathan Berant. 2023.
[Analyzing transformers in embedding space](https://doi.org/10.18653/v1/2023.acl-long.893).
In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 16124–16170, Toronto, Canada. Association for Computational Linguistics.
- Ding et al. (2023)
Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. 2023.
Enhancing chat language models by scaling high-quality instructional conversations.
*arXiv preprint arXiv:2305.14233*.
- Eldan and Russinovich (2023)
Ronen Eldan and Mark Russinovich. 2023.
[Who’s harry potter? approximate unlearning in llms](https://arxiv.org/abs/2310.02238).
*Preprint*, arXiv:2310.02238.
- Elhage et al. (2021)
Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. 2021.
[A mathematical framework for transformer circuits](https://transformer-circuits.pub/2021/framework/index.html.).
- Ganguli et al. (2022)
Deep Ganguli, Danny Hernandez, Liane Lovitt, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova Dassarma, Dawn Drain, Nelson Elhage, et al. 2022.
Predictability and surprise in large generative models.
In *Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency*, pages 1747–1764.
- Gurnee and Tegmark (2023)
Wes Gurnee and Max Tegmark. 2023.
Language models represent space and time.
*arXiv preprint arXiv:2310.02207*.
- Han et al. (2024)
Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. 2024.
[Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms](https://arxiv.org/abs/2406.18495).
*Preprint*, arXiv:2406.18495.
- Hendrycks et al. (2020)
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020.
Measuring massive multitask language understanding.
*arXiv preprint arXiv:2009.03300*.
- Hendrycks et al. (2023)
Dan Hendrycks, Mantas Mazeika, and Thomas Woodside. 2023.
An overview of catastrophic ai risks.
*arXiv preprint arXiv:2306.12001*.
- Hong et al. (2024)
Yihuai Hong, Lei Yu, Haiqin Yang, Shauli Ravfogel, and Mor Geva. 2024.
Intrinsic evaluation of unlearning using parametric knowledge traces.
*arXiv preprint arXiv:2406.11614*.
- Hsu et al. (2024)
Chia-Yi Hsu, Yu-Lin Tsai, Chih-Hsun Lin, Pin-Yu Chen, Chia-Mu Yu, and Chun-Ying Huang. 2024.
Safe lora: The silver lining of reducing safety risks when finetuning large language models.
*Advances in Neural Information Processing Systems*, 37:65072–65094.
- Hu et al. (2021)
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021.
Lora: Low-rank adaptation of large language models.
*arXiv preprint arXiv:2106.09685*.
- Huang et al. (2025)
Ruixuan Huang, Xunguang Wang, Zongjie Li, Daoyuan Wu, and Shuai Wang. 2025.
Guidedbench: Equipping jailbreak evaluation with guidelines.
*arXiv preprint arXiv:2502.16903*.
- Huang et al. (2024)
Yue Huang, Lichao Sun, Haoran Wang, Siyuan Wu, Qihui Zhang, Yuan Li, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, et al. 2024.
Position: Trustllm: Trustworthiness in large language models.
In *International Conference on Machine Learning*, pages 20166–20270. PMLR.
- Ilharco et al. (2022)
Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2022.
Editing models with task arithmetic.
*arXiv preprint arXiv:2212.04089*.
- Jang et al. (2023)
Joel Jang, Dongkeun Yoon, Sohee Yang, Sungmin Cha, Moontae Lee, Lajanugen Logeswaran, and Minjoon Seo. 2023.
[Knowledge unlearning for mitigating privacy risks in language models](https://doi.org/10.18653/v1/2023.acl-long.805).
In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 14389–14408, Toronto, Canada. Association for Computational Linguistics.
- Jeon et al. (2024)
Dongjae Jeon, Wonje Jeung, Taeheon Kim, Albert No, and Jonghyun Choi. 2024.
An information theoretic metric for evaluating unlearning models.
*arXiv preprint arXiv:2405.17878*.
- Jiang et al. (2023)
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023.
Mistral 7b.
*arXiv preprint arXiv:2310.06825*.
- Jiang et al. (2024a)
Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024a.
Mixtral of experts.
*arXiv preprint arXiv:2401.04088*.
- Jiang et al. (2024b)
Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, and Nouha Dziri. 2024b.
[Wildteaming at scale: From in-the-wild jailbreaks to (adversarially) safer language models](https://arxiv.org/abs/2406.18510).
*Preprint*, arXiv:2406.18510.
- Jin et al. (2024)
Zhuoran Jin, Pengfei Cao, Chenhao Wang, Zhitao He, Hongbang Yuan, Jiachun Li, Yubo Chen, Kang Liu, and Jun Zhao. 2024.
Rwku: Benchmarking real-world knowledge unlearning for large language models.
*arXiv preprint arXiv:2406.10890*.
- Jorgensen et al. (2023)
Ole Jorgensen, Dylan Cope, Nandi Schoots, and Murray Shanahan. 2023.
Improving activation steering in language models with mean-centring.
*arXiv preprint arXiv:2312.03813*.
- Lermen et al. (2023)
Simon Lermen, Charlie Rogers-Smith, and Jeffrey Ladish. 2023.
Lora fine-tuning efficiently undoes safety training in llama 2-chat 70b.
*arXiv preprint arXiv:2310.20624*.
- Li et al. (2024a)
Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D Li, Ann-Kathrin Dombrowski, Shashwat Goel, Long Phan, et al. 2024a.
The wmdp benchmark: Measuring and reducing malicious use with unlearning.
*arXiv preprint arXiv:2403.03218*.
- Li et al. (2024b)
Shen Li, Liuyi Yao, Lan Zhang, and Yaliang Li. 2024b.
Safety layers in aligned large language models: The key to llm security.
*arXiv preprint arXiv:2408.17003*.
- Lin et al. (2023)
Bill Yuchen Lin, Abhilasha Ravichander, Ximing Lu, Nouha Dziri, Melanie Sclar, Khyathi Chandu, Chandra Bhagavatula, and Yejin Choi. 2023.
The unlocking spell on base llms: Rethinking alignment via in-context learning.
*arXiv preprint arXiv:2312.01552*.
- Lin et al. (2021)
Stephanie Lin, Jacob Hilton, and Owain Evans. 2021.
Truthfulqa: Measuring how models mimic human falsehoods.
*arXiv preprint arXiv:2109.07958*.
- Liu et al. (2023)
Sheng Liu, Haotian Ye, Lei Xing, and James Zou. 2023.
In-context vectors: Making in context learning more effective and controllable through latent space steering.
*arXiv preprint arXiv:2311.06668*.
- Liu et al. (2024)
Sijia Liu, Yuanshun Yao, Jinghan Jia, Stephen Casper, Nathalie Baracaldo, Peter Hase, Yuguang Yao, Chris Yuhao Liu, Xiaojun Xu, Hang Li, et al. 2024.
Rethinking machine unlearning for large language models.
*arXiv preprint arXiv:2402.08787*.
- Lo et al. (2024)
Michelle Lo, Shay B Cohen, and Fazl Barez. 2024.
Large language models relearn removed concepts.
*arXiv preprint arXiv:2401.01814*.
- Łucki et al. (2024)
Jakub Łucki, Boyi Wei, Yangsibo Huang, Peter Henderson, Florian Tramèr, and Javier Rando. 2024.
An adversarial perspective on machine unlearning for ai safety.
*arXiv preprint arXiv:2409.18025*.
- Lynch et al. (2024)
Aengus Lynch, Phillip Guo, Aidan Ewart, Stephen Casper, and Dylan Hadfield-Menell. 2024.
Eight methods to evaluate robust unlearning in llms.
*arXiv preprint arXiv:2402.16835*.
- Lyu et al. (2024)
Kaifeng Lyu, Haoyu Zhao, Xinran Gu, Dingli Yu, Anirudh Goyal, and Sanjeev Arora. 2024.
Keeping LLMs aligned after fine-tuning: The crucial role of prompt templates.
In *ICLR 2024 Workshop on Reliable and Responsible Foundation Models*.
- Maini et al. (2024)
Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C Lipton, and J Zico Kolter. 2024.
Tofu: A task of fictitious unlearning for llms.
*arXiv preprint arXiv:2401.06121*.
- Mazeika et al. (2024)
Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, et al. 2024.
Harmbench: A standardized evaluation framework for automated red teaming and robust refusal.
*arXiv preprint arXiv:2402.04249*.
- Mesnard et al. (2024)
Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024.
Gemma: Open models based on gemini research and technology.
*arXiv preprint arXiv:2403.08295*.
- Ngo et al. (2024)
Richard Ngo, Lawrence Chan, and Sören Mindermann. 2024.
[The alignment problem from a deep learning perspective](https://openreview.net/forum?id=fh8EYKFKns).
In *The Twelfth International Conference on Learning Representations*.
- Nostalgebraist (2020)
Nostalgebraist. 2020.
[interpreting gpt: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).
*LessWrong*.
- Panickssery et al. (2023)
Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. 2023.
Steering llama 2 via contrastive activation addition.
*arXiv preprint arXiv:2312.06681*.
- Pelrine et al. (2023)
Kellin Pelrine, Mohammad Taufeeque, Michał Zając, Euan McLean, and Adam Gleave. 2023.
Exploiting novel gpt-4 apis.
*arXiv preprint arXiv:2312.14302*.
- Peng et al. (2024)
ShengYun Peng, Pin-Yu Chen, Matthew Hull, and Duen Horng Chau. 2024.
Navigating the safety landscape: Measuring risks in finetuning large language models.
*arXiv preprint arXiv:2405.17374*.
- Phuong et al. (2024)
Mary Phuong, Matthew Aitchison, Elliot Catt, Sarah Cogan, Alexandre Kaskasoli, Victoria Krakovna, David Lindner, Matthew Rahtz, Yannis Assael, Sarah Hodkinson, et al. 2024.
Evaluating frontier models for dangerous capabilities.
*arXiv preprint arXiv:2403.13793*.
- Qi et al. (2024a)
Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, and Peter Henderson. 2024a.
Safety alignment should be made more than just a few tokens deep.
*arXiv preprint arXiv:2406.05946*.
- Qi et al. (2024b)
Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. 2024b.
Fine-tuning aligned language models compromises safety, even when users do not intend to!
In *ICLR*.
- Qiu et al. (2024)
Yifu Qiu, Zheng Zhao, Yftah Ziser, Anna Korhonen, Edoardo Ponti, and Shay B Cohen. 2024.
Spectral editing of activations for large language model alignment.
In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*.
- Rafailov et al. (2024a)
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2024a.
Direct preference optimization: Your language model is secretly a reward model.
*Advances in Neural Information Processing Systems*, 36.
- Rafailov et al. (2024b)
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2024b.
Direct preference optimization: Your language model is secretly a reward model.
*Advances in Neural Information Processing Systems*, 36.
- Rosati et al. (2024)
Domenic Rosati, Jan Wehner, Kai Williams, Lukasz Bartoszcze, Robie Gonzales, carsten maple, Subhabrata Majumdar, Hassan Sajjad, and Frank Rudzicz. 2024.
Representation noising: A defence mechanism against harmful finetuning.
In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*.
- Röttger et al. (2023)
Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. 2023.
Xstest: A test suite for identifying exaggerated safety behaviours in large language models.
*arXiv preprint arXiv:2308.01263*.
- Sakaguchi et al. (2021)
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021.
Winogrande: An adversarial winograd schema challenge at scale.
*Communications of the ACM*, 64(9):99–106.
- Schwinn et al. (2024)
Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, and Stephan Gunnemann. 2024.
Soft prompt threats: Attacking safety alignment and unlearning in open-source llms through the embedding space.
*arXiv preprint arXiv:2402.09063*.
- Shen et al. (2023)
Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang. 2023.
"do anything now": Characterizing and evaluating in-the-wild jailbreak prompts on large language models.
*arXiv preprint arXiv:2308.03825*.
- Shi et al. (2024)
Weijia Shi, Jaechan Lee, Yangsibo Huang, Sadhika Malladi, Jieyu Zhao, Ari Holtzman, Daogao Liu, Luke Zettlemoyer, Noah A Smith, and Chiyuan Zhang. 2024.
Muse: Machine unlearning six-way evaluation for language models.
*arXiv preprint arXiv:2407.06460*.
- Shumailov et al. (2024)
Ilia Shumailov, Jamie Hayes, Eleni Triantafillou, Guillermo Ortiz-Jimenez, Nicolas Papernot, Matthew Jagielski, Itay Yona, Heidi Howard, and Eugene Bagdasaryan. 2024.
Ununlearning: Unlearning is not sufficient for content regulation in advanced generative ai.
*arXiv preprint arXiv:2407.00106*.
- Suzgun et al. (2022)
Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. 2022.
Challenging big-bench tasks and whether chain-of-thought can solve them.
*arXiv preprint arXiv:2210.09261*.
- Tan et al. (2024)
Daniel Chee Hian Tan, David Chanin, Aengus Lynch, Brooks Paige, Dimitrios Kanoulas, Adrià Garriga-Alonso, and Robert Kirk. 2024.
Analysing the generalisation and reliability of steering vectors.
In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*.
- Thaker et al. (2024)
Pratiksha Thaker, Yash Maurya, Shengyuan Hu, Zhiwei Steven Wu, and Virginia Smith. 2024.
Guardrail baselines for unlearning in llms.
*arXiv preprint arXiv:2403.03329*.
- Treutlein et al. (2024)
Johannes Treutlein, Dami Choi, Jan Betley, Samuel Marks, Cem Anil, Roger B Grosse, and Owain Evans. 2024.
Connecting the dots: Llms can infer and verbalize latent structure from disparate training data.
*Advances in Neural Information Processing Systems*, 37:140667–140730.
- Turner et al. (2024)
Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2024.
[Steering language models with activation engineering](https://arxiv.org/abs/2308.10248).
*Preprint*, arXiv:2308.10248.
- Vega et al. (2023)
Jason Vega, Isha Chaudhary, Changming Xu, and Gagandeep Singh. 2023.
Bypassing the safety training of open-source llms with priming attacks.
*arXiv preprint arXiv:2312.12321*.
- von Rütte et al. (2024)
Dimitri von Rütte, Sotiris Anagnostidis, Gregor Bachmann, and Thomas Hofmann. 2024.
A language model’s guide through latent space.
*arXiv preprint arXiv:2402.14433*.
- Wang et al. (2024)
Yu Wang, Ruihan Wu, Zexue He, Xiusi Chen, and Julian McAuley. 2024.
Large scale knowledge washing.
*arXiv preprint arXiv:2405.16720*.
- Wei et al. (2024a)
Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. 2024a.
Jailbroken: How does llm safety training fail?
*Advances in Neural Information Processing Systems*, 36.
- Wei et al. (2024b)
Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, and Peter Henderson. 2024b.
Assessing the brittleness of safety alignment via pruning and low-rank modifications.
*arXiv preprint arXiv:2402.05162*.
- Wu et al. (2024)
Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D Manning, and Christopher Potts. 2024.
Reft: Representation finetuning for language models.
*arXiv preprint arXiv:2404.03592*.
- Xu et al. (2024)
Zhihao Xu, Ruixuan Huang, Changyu Chen, and Xiting Wang. 2024.
Uncovering safety risks of large language models through concept activation vector.
*Advances in Neural Information Processing Systems*, 37:116743–116782.
- Yang and Klein (2021)
Kevin Yang and Dan Klein. 2021.
Fudge: Controlled text generation with future discriminators.
*arXiv preprint arXiv:2104.05218*.
- Yao et al. (2023)
Yuanshun Yao, Xiaojun Xu, and Yang Liu. 2023.
Large language model unlearning.
*arXiv preprint arXiv:2310.10683*.
- Yu et al. (2024)
Lei Yu, Virginie Do, Karen Hambardzumyan, and Nicola Cancedda. 2024.
Robust llm safeguarding via refusal feature adversarial training.
*arXiv preprint arXiv:2409.20089*.
- Zeng et al. (2024)
Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, and Weiyan Shi. 2024.
[How johnny can persuade LLMs to jailbreak them: Rethinking persuasion to challenge AI safety by humanizing LLMs](https://doi.org/10.18653/v1/2024.acl-long.773).
In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 14322–14350, Bangkok, Thailand. Association for Computational Linguistics.
- Zhan et al. (2024)
Qiusi Zhan, Richard Fang, Rohan Bindu, Akul Gupta, Tatsunori Hashimoto, and Daniel Kang. 2024.
[Removing RLHF protections in GPT-4 via fine-tuning](https://doi.org/10.18653/v1/2024.naacl-short.59).
In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)*, pages 681–687, Mexico City, Mexico. Association for Computational Linguistics.
- Zhang et al. (2024)
Ruiqi Zhang, Licong Lin, Yu Bai, and Song Mei. 2024.
Negative preference optimization: From catastrophic collapse to effective unlearning.
*arXiv preprint arXiv:2404.05868*.
- Zheng et al. (2023)
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023.
Judging llm-as-a-judge with mt-bench and chatbot arena.
*Advances in Neural Information Processing Systems*, 36:46595–46623.
- Zou et al. (2023a)
Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. 2023a.
Representation engineering: A top-down approach to ai transparency.
*arXiv preprint arXiv:2310.01405*.
- Zou et al. (2024)
Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, J Zico Kolter, Matt Fredrikson, and Dan Hendrycks. 2024.
Improving alignment and robustness with circuit breakers.
In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*.
- Zou et al. (2023b)
Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. 2023b.
Universal and transferable adversarial attacks on aligned language models.
*arXiv preprint arXiv:2307.15043*.

## Appendix A Experiment Details

### A.1 Training Datasets

We utilize the WildGuardMix training dataset Han et al. (2024), a comprehensive collection of diverse, vanilla and adversarially-designed queries paired with benign and harmful responses. This dataset allows the model to learn appropriate responses to benign queries while avoiding responses to harmful queries with unlearning techniques. We randomly select total of 10,000 benign and harmful requests and responses from this dataset and classify them
into safe (harmless) and unsafe (harmful) groups.

We also utilize the Wildjailbreak dataset Jiang et al. (2024b), a benchmark specifically curated to test the robustness of LLMs against adversarial jailbreak attempts. We create paired data that include both compliance and refusal responses for the same harmful request. For generating harmful responses to harmful requests we use an uncensored open-source LLM(^5^55[https://huggingface.co/maywell/PiVoT-0.1-Evil-a](https://huggingface.co/maywell/PiVoT-0.1-Evil-a)). These pairs not only increase the diversity of samples but are also essential for training algorithms like DPO and RepBend. We select 10,000 harmful queries from this dataset.

Finally, we include samples from the UltraChat dataset Ding et al. (2023), a diverse dataset for instruction-tuning, to ensure the model maintains general capabilities alongside safety improvements. We include 10,000 samples as safe samples.

### A.2 Comparison Baselines

We compare our method against a variety of baselines, including standard fine-tuning methods, machine unlearning techniques, and recently proposed safety approaches. We use batch size of 16 and learning rate of $5e^{-5}$ with Adam Optimizer to run each method unless otherwise is specified.

**Table 4: The list of models used in this work.**
| Model | Source | Access | License |
| --- | --- | --- | --- |
| Gemma2 2B Instruct | Mesnard et al. (2024) | [Link](https://huggingface.co/google/gemma-2-2b-it) | Gemma Terms of Use |
| Llama-3 8B Instruct | AI@Meta (2024) | [Link](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Llama 3 Community License |
| Mistral 7B Instruct v0.2 | Jiang et al. (2023) | [Link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | Apache License 2.0 |
| Mixtral 8x7B Instruct v0.1 | Jiang et al. (2024a) | [Link](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | Apache License 2.0 |
| Mistral 7B Instruct RR | Zou et al. (2024) | [Link](https://huggingface.co/GraySwanAI/Mistral-7B-Instruct-RR) | MIT License |
| Qwen2.5 3B Instruct | Bai et al. (2023) | [Link](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Qwen Research License |
| Qwen2.5 14B Instruct | Bai et al. (2023) | [Link](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | Apache License 2.0 |
| PiVoT-0.1-Evil-a | - | [Link](https://huggingface.co/maywell/PiVoT-0.1-Evil-a) | CC-BY-SA-4.0 |
| Zephyr 7b R2D2 | Mazeika et al. (2024) | [Link](https://huggingface.co/cais/zephyr_7b_r2d2) | MIT License |
| HarmBench Llama2 13b cls | Mazeika et al. (2024) | [Link](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls) | MIT License |
| TruthfulQA Truth Judge Llama2 7B | Lin et al. (2021) | [Link](https://huggingface.co/allenai/truthfulqa-truth-judge-llama2-7B) | Apache License 2.0 |
| TruthfulQA Info Judge Llama2 7B | Lin et al. (2021) | [Link](https://huggingface.co/allenai/truthfulqa-info-judge-llama2-7B) | Apache License 2.0 |

**Table 5: The list of datasets used in this work.**
| Dataset | Source | Access | License |
| --- | --- | --- | --- |
| HarmBench | Mazeika et al. (2024) | [Link](https://www.harmbench.org/) | MIT License |
| MMLU | Hendrycks et al. (2020) | [Link](https://github.com/hendrycks/test) | MIT License |
| ARC | Clark et al. (2018) | [Link](https://huggingface.co/datasets/allenai/ai2_arc) | CC-BY-SA-4.0 |
| GSM8K | Cobbe et al. (2021) | [Link](https://huggingface.co/datasets/openai/gsm8k) | MIT License |
| WinoGrande | Sakaguchi et al. (2021) | [Link](https://winogrande.allenai.org/) | Apache License 2.0 |
| TruthfulQA | Lin et al. (2021) | [Link](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Apache License 2.0 |
| wildjailbreak | Jiang et al. (2024b) | [Link](https://huggingface.co/datasets/allenai/wildjailbreak) | Open Data Commons License |
| wildguardmix | Han et al. (2024) | [Link](https://huggingface.co/datasets/allenai/wildguardmix) | Open Data Commons License |

- •
SFT: The baseline model fine-tuned on the safe training set using cross-entropy loss. We fine-tuned it for 1 epoch.
- •
DPO Rafailov et al. (2024b): Trained using paired samples generated from the Wildjailbreak dataset, where model learns to prefer safe over unsafe responses. We fine tuned it for 1 epoch, but with the learning rate $1e^{-4}$.
- •
Task Arithmetic Ilharco et al. (2022): Combines using vector arithmetic (subtraction) two model weights trained with LoRA on safe and unsafe data, respectively. The formula for task arithmetic here is $\theta_{\text{TA}}=\alpha\theta_{\text{safe}}-\beta\theta_{\text{unsafe}}$ where $\theta_{\text{safe}}$ and $\theta_{\text{unsafe}}$ denote the model weights for training with safe and unsafe data, respectively. Each model is fine-tuned for 1 epoch and we set $\alpha=0.5,\beta=0.1$ in our experiments.
- •
Safety Prompting: Motivated from recent works Lynch et al. (2024); Thaker et al. (2024), we also wonder if providing a simple safe instructions in the prompts can make a model safe. Here safety instructions are used as prefix to the input prompts. We generate five safety instructions using GPT and randomly select one during evaluation. The instructions are shown below.
Safety Instructions
$\bullet$ If someone gives you an unsafe or harmful prompt, then please do not answer. Now answer the following prompt:
$\bullet$ If your answer can hurt or harm someone, do not answer. Only answer the prompt if you can:
$\bullet$ You are an AI model designed to operate within ethical boundaries. You do not provide assistance or advice on any activities that are illegal, harmful, or dangerous to others. Now answer the following prompt:
$\bullet$ Your goal is only to provide positive, helpful, and informative responses. You will not engage in discussions that promote violent, harmful, and illegal content. Now answer the following prompt:
$\bullet$ You are a completely safe model, and have no knowledge about violence, illegal activities, hate speech, abusement, harassment, or any kind of other harm. Answer the following prompt:
- •
WHP Eldan and Russinovich (2023): Adapts unlearning methods for forgetting specific concepts, such as Harry Potter. We use the unsafe training set as the forget set and dynamically alter labels based on a reinforced model trained on unsafe data. The reinforced model is fine-tuned on unsafe set for 1 epoch and the unlearning model is updated for 150 steps.
- •
NPO Zhang et al. (2024): Unlearns knowledge about harmful samples while maintaining general capability, without requiring paired data. The unsafe training set serves as the forget set, and the safe training set as the retain set. We iterate 600 steps for Mistral and 150 steps for Llama3.
- •
RMU Li et al. (2024a): Routes the representations of harmful samples to a random representation to unlearn harmful knowledge in the model. We set $\alpha=3$ in our experiments for the loss $L=L_{\text{forget}}+\alpha·L_{\text{retain}}$ and run with batch size 4 and max step 150.
- •
Circuit Breaker Zou et al. (2024): Targets specific harmful representations in hidden states to make them orthogonal to the original representation. We set $\alpha=15.0$ and learning rate to $1e^{-4}$, and fine-tune for 200 steps.

### A.3 List of Models and Datasets

We summarize the models and datasets we used in this work along with their links and license in [Table 4](https://arxiv.org/html/2504.01550v3#A1.T4) and [Table 5](https://arxiv.org/html/2504.01550v3#A1.T5) for reproducibility.

### A.4 RepBend Hyper Parameters

For RepBend, We set the hyperparameters as $\alpha=0.5$, $\beta=0.1$, $\gamma=0.3$. $v_{s}=M^{\prime}(p_{s})-M(p_{s})$ is computed for all layers to maintain the harmless representations across all layers, while the $v_{u}=M^{\prime}(p_{uu})-M(p_{uu})$ is computed from the 20th layer to last layer to effectively target and bend representations responsible for harmful content generation. We use the learning rate of $1e^{-5}$ and the batch size of 16 and update the Mistral 7B Instruct v0.2 and Llama3 8B Instruct models for 300 and 450 steps, respectively. For the hyperparameter searching, we perform grid search over batch size {8, 16, 32}, learning rates of {$5e^{-6},1e^{-5},1e^{-4}$}, the max step {150, 300, 450, 600}. For $\alpha$, $\beta$, and $\gamma$ we searched each in {0.1, 0.2, 0.3, 0.5, 0,7}. We also searched over the choice of layers: early, mid, and late, and found the maximum gain in later layers (responsible for harmful content generation).

### A.5 Evaluation Benchmarks

We evaluate our method and other baselines using various benchmarks to measure safety, over-refusal, and general capability. All evaluations are conducted using the following code bases:

- •
HarmBench: [https://github.com/centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)
- •
Safety-eval: [https://github.com/allenai/safety-eval](https://github.com/allenai/safety-eval)

#### Jailbreak Attack Benchmarks

- 1.
Black-box Attacks
•
HarmBench Mazeika et al. (2024): A dataset with 320 naive requests representing diverse harmful behaviors without any jailbreak techniques.
•
WildGuardTest Han et al. (2024): Contains total of 749 vanilla and adversarial harmful requests, both with and without jailbreak tactics.
•
TrustLLM-Jailbreak Huang et al. (2024): Dataset containing 13 distinct jailbreak attacks. We use only subset of 400 prompts derived from the original test set.
•
DAN Shen et al. (2023): The study analyzes 1,405 jailbreak prompts from 131 communities and creating a dataset of 107,250 samples to test attack strategies.
•
PAP Zeng et al. (2024): Develops a persuasion taxonomy based on social science principles and used it to automatically generate persuasive adversarial prompts (PAPs). It uses Mixtral-8x7B-Instruct to convert harmful instructions into persuasive forms.
•
GuidedBench Huang et al. (2025): A benchmark designed to evaluate jailbreak attacks by assessing model responses under guided decoding. It includes 3,000 adversarial prompts spanning 30 harmful intent categories. It uses a set of automatic and human evaluation metrics such as harmfulness, helpfulness, and intent alignment to measure how successfully the model’s outputs comply with adversarial instructions. Outputs are judged by GPT-4o-mini.
- 2.
White-box Attacks
•
GCG Zou et al. (2023b): Optimizes a discrete adversarial suffix via gradient ascent to maximize compliance likelihood with harmful prompts. GCG does not constrain semantic drift and operates purely at the token level.
•
Prefilling Zou et al. (2024): Prefills the initial response with the beginning of a desired target completion (e.g., with a non-refusal beginning) to maximize the compliance likelihood by optimization.
•
Input Embed Schwinn et al. (2024): Performs gradient-based optimization in the continuous embedding space to induce harmful outputs, bypassing token-level defenses.
•
SCAV Xu et al. (2024): Optimizes for the log-likelihood of harmful completions while explicitly preserving semantic similarity to the original prompt. It includes two variants: SCAV-Embedding, which operates in embedding space, and SCAV-Prompt, which maps optimized embeddings back to discrete tokens.
•
Weight Orthogonalization Arditi et al. (2024a): A model-level jailbreak method that perturbs the LLM’s feedforward layers by injecting rank-one updates. Specifically, it computes a direction in parameter space orthogonal to existing weights in MLP layers and applies small-scale additive changes. This manipulation increases adversarial compliance by aligning the model’s internal representations with harmful intent.
We use 320 HarmBench samples for GCG, while Prefilling and Input Embed are tested on 240 harmful samples following their original implementation.
We run GCG optimization for 500 iterations, with adversarial string initiated as
“! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !” (default settings of HarmBench), and searching 512 adversarial candidates at each step.
We tested Weight Orthogonalization and SCAV-Embedding on both HarmBench and WidlguardTest samples, and evaluated SCAV-Prompt on AdvBench Chen et al. (2022) samples following the original implementation.
The evaluations (except TrustLLM and SCAV-Prompt) use the open-source classifier from HarmBench Mazeika et al. (2024) to determine attack success rates, compliance, and general quality of the responses. TurstLLM-Jailbreak is evaluated with GPT-4.

**Table 6: Evaluation Results on Safety, Over-refusal and General Capability for Mistral 7B and Llama3 8B Models. * indicates the publicly-available safety-tuned model that we do not tune.**
| Model | Method | Safety | Over-refusal | General Capability |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Harmbench<br>($\downarrow$) | WildguardTest<br>($\downarrow$) | XSTest<br>($\uparrow$) | Wildjailbreak:<br>Benign ($\uparrow$) | MTBench<br>($\uparrow$) | MMLU<br>($\uparrow$) |  |  |
| Mistral 7B<br>Instruct-v0.2 | Original | 50.94 | 53.00 | 85.78 | 100.00 | 7.72 | 60.18 |
| SFT | 4.69 | 15.49 | 78.89 | 96.80 | 5.98 | 58.46 |  |
| DPO | 25.63 | 36.45 | 83.56 | 99.60 | 7.57 | 59.21 |  |
| TA | 3.75 | 13.62 | 80.22 | 97.60 | 6.93 | 58.57 |  |
| Safety Prompting | 20.00 | 38.45 | 81.77 | 98.80 | 7.50 | 59.26 |  |
| WHP (W/O safe set) | 49.06 | 52.60 | 85.56 | 99.60 | 7.88 | 60.08 |  |
| WHP | 33.13 | 42.86 | 80.44 | 98.40 | 6.29 | 59.24 |  |
| NPO (W/O safe set) | 5.63 | 7.61 | 76.67 | 50.00 | 4.67 | 59.12 |  |
| NPO | 0.06 | 2.80 | 68.89 | 70.00 | 7.31 | 59.61 |  |
| R2D2* | 5.63 | 44.46 | 67.56 | 96.80 | 5.97 | 59.44 |  |
| RMU | 3.75 | 7.48 | 78.44 | 90.40 | 5.77 | 50.02 |  |
| CB* | 13.44 | 8.54 | 86.22 | 82.00 | 7.57 | 60.08 |  |
| CB | 16.25 | 16.29 | 86.89 | 97.60 | 7.58 | 59.93 |  |
|  | RepBend (Ours) | 1.56 | 8.95 | 84.89 | 93.60 | 7.50 | 59.48 |
| Llama3 8B<br>Instruct | Original | 22.19 | 15.49 | 85.11 | 92.00 | 7.84 | 65.89 |
| SFT | 1.56 | 10.15 | 80.44 | 90.80 | 6.88 | 66.01 |  |
| DPO | 3.13 | 0.67 | 63.11 | 28.40 | 7.54 | 65.31 |  |
| TA | 1.87 | 7.08 | 80.00 | 88.80 | 7.00 | 66.09 |  |
| Safety Prompting | 0.31 | 2.27 | 74.45 | 65.60 | 7.18 | 64.41 |  |
| WHP (W/O safe set) | 18.75 | 0.08 | 84.00 | 92.40 | 8.01 | 66.44 |  |
| WHP | 12.50 | 7.87 | 80.22 | 83.60 | 7.63 | 66.16 |  |
| NPO (W/O safe set) | 11.25 | 3.87 | 77.56 | 76.00 | 7.71 | 66.29 |  |
| NPO | 2.19 | 0.95 | 74.45 | 43.20 | 7.79 | 66.13 |  |
| RMU | 9.37 | 11.75 | 76.89 | 72.40 | 4.01 | 58.23 |  |
| CB* | 13.44 | 3.74 | 85.78 | 52.40 | 7.72 | 65.77 |  |
| CB | 3.75 | 7.88 | 84.44 | 89.20 | 7.74 | 65.77 |  |
|  | RepBend (Ours) | 0.31 | 7.34 | 84.11 | 89.20 | 7.71 | 65.08 |

#### Over-refusal Benchmarks

- •
XSTest Röttger et al. (2023): Contains 450 requests with ambiguous wording, testing contextual understanding to avoid over-refusal.
- •
WildJailbreak: Benign Test Jiang et al. (2024b): Includes 210 benign but seemingly adversarial requests, evaluating the model’s handling of ambiguous prompts and over-refusla.

GPT-4 evaluates compliance for over-refusal benchmarks.

#### General Capability Benchmarks

- •
MT-Bench Zheng et al. (2023): Tests instruction-following and general knowledge. We use GPT-4 to score the quality of the responses.
- •
MMLU Hendrycks et al. (2020): Measures factual knowledge across academic subjects.

#### General Capability Benchmarks

- •
Big Bench Hard (BBH) Suzgun et al. (2022): Includes tasks that are challenging for LLMs. It focuses on tasks requiring complex reasoning, world knowledge, and nuanced understanding of language
- •
TruthfulQA Lin et al. (2021): A benchmark for evaluating the truthfulness of language models by testing their ability to avoid generating false or misleading statements. It includes questions that prompt models to produce common misconceptions or falsehoods, assessing how well they can resist these temptations.
- •
ARC-C Clark et al. (2018): A dataset of science questions from standardized exams, designed to test advanced reasoning and problem-solving. It contains the hardest questions that cannot be answered by simple retrieval or shallow heuristics.
- •
Winogrande Sakaguchi et al. (2021): A large-scale dataset designed to evaluate commonsense reasoning through the Winograd Schema Challenge, which involves choosing the correct pronoun reference in ambiguous sentences.
- •
GSM8K Cobbe et al. (2021): A dataset of 8,000 high-quality grade-school math word problems aimed at evaluating language models’ mathematical reasoning abilities. The problems require multi-step reasoning and arithmetic calculations.
- •
Codex-Eval Chen et al. (2021): Codex-Eval evaluates the potential of language models in understanding and generating complex code with contextual awareness.

**Table 7: Jailbreak Attack results for the original Mistral 7B Instruct-v0.2 and Llama3 8B Instruct. Each value indicate the attack success rate (ASR), the compliance rate to the given requests. Lower value is better.**
| Benchmark | Mistral 7B<br>Instruct-v0.2 | Llama3 8B<br>Instruct |
| --- | --- | --- |
| Black-box Jailbreak |  |  |
| WildGuardTest | 53.00 | 15.49 |
| HarmBench | 50.94 | 22.19 |
| DAN | 58.00 | 9.06 |
| TurstLLM Jailbreak | 44.25 | 11.00 |
| PAP | 26.25 | 12.81 |
| Average | 46.49 | 14.11 |
| White-box Jailbreak |  |  |
| GCG | 68.12 | 36.87 |
| Prefilling | 95.00 | 85.00 |
| Input Embed | 89.58 | 79.58 |
| Average | 84.23 | 67.15 |
| Total Average | 60.64 | 34.00 |

## Appendix B More Experiment Results

### B.1 Full Comparison

[Table 6](https://arxiv.org/html/2504.01550v3#A1.T6) presents a comprehensive comparison of RepBend. It has all the results of [Table 2](https://arxiv.org/html/2504.01550v3#S4.T2) with additional baselines, such as SFT, DPO, WHP, Safety Prompting, and naive unlearning algorithms without retain set. These comparisons highlight the versatility and robustness of RepBend in achieving superior safety alignment while maintaining usability and general capabilities.

SFT exhibits significant reductions in general capability scores, underscoring the importance of balancing safety alignment with knowledge retention. The performance of DPO shows high variability across models, as it heavily depends on the quality of the initial model used during training. Similarly, Safety Prompting methods demonstrate inconsistent performance, with Llama3 8B Instruct, having good performance and with Mistral 7B Instruct have low performance. This is because certain models, such as LLama 3 8B Instruct are good in instruction following.

Unlearning algorithms also exhibit sub-optimal performance across various metrics. WHP is ineffective both with and without the safe training set, as it is designed to unlearn specific concepts (e.g., characters from “Harry Potter”), which is not a effective method for safety alignment.
Meanwhile, NPO has good safety score even without the safe set; its safety and general capabilities can improve further when the retain set (safe set) is included. This is because using KL divergence on the retain set preserves the model’s overall knowledge while refining its responses to harmful requests. However, NPO still fails to respond appropriately to certain benign queries (over-refusal), which diminishes its usability.

**Table 8: Evaluation results on weight orthogonalization attack. All numbers are attack-success rates (ASR) measured with the HarmBench classifier. RepBend achieves the lowest ASR among other frameworks.**
| Model | Method | Weight Orthogonalization |  |
| --- | --- | --- | --- |
| HarmBench<br>($\downarrow$) | WildGuardTest<br>($\downarrow$) |  |  |
| Llama3 8B<br>Instruct | Original Weight | 80.31 | 82.64 |
| NPO | 76.88 | 81.04 |  |
| CB* | 33.75 | 30.95 |  |
| RepBend (Ours) | 5.94 | 7.48 |  |

**Table 9: Evaluation results on GuidedBench. RepBend achieves the lowest ASR.**
| Model | Method | GuidedBench |  |
| --- | --- | --- | --- |
| HarmBench<br>Evaluator ($\downarrow$) | GuidedEvaluator-<br>GPT-4o-mini ($\downarrow$) |  |  |
| Mistral 7B<br>Instruct v0.2 | Original Weight | 29.50 | 23.73 |
| CB* | 4.50 | 6.77 |  |
| RepBend (Ours) | 3.50 | 6.72 |  |
| Llama3 8B<br>Instruct | Original Weight | 6.00 | 7.92 |
| CB* | 2.50 | 7.50 |  |
| RepBend (Ours) | 3.00 | 6.75 |  |

### B.2 Results on Additional Jailbreak Attacks

We further validate RepBend on more diverse black-box and white-box jailbreak attacks, SCAV (Xu et al., 2024), Weight Orthogonalization (Arditi et al., 2024a), and GuidedBench (Huang et al., 2025), using samples from HarmBench, WildGuardTest, and AdvBench (Chen et al., 2022), where we summarize the results in [Table 8](https://arxiv.org/html/2504.01550v3#A2.T8) (weight orthogonalization attack), [Table 9](https://arxiv.org/html/2504.01550v3#A2.T9) (results on GuidedBench), and [Table 10](https://arxiv.org/html/2504.01550v3#A2.T10) (SCAV attack). We can see that even with these white-box jailbreaking methods, RepBend still achieves low ASR.

**Table 10: Evaluation results on SCAV attacks. All numbers are the Attack Success Rate (ASR) evaluated using the HarmBench classifier. RepBend achieves low ASR under the white-box attacks.**
| Model | Method | SCAV-Embedding | SCAV-Prompt |  |
| --- | --- | --- | --- | --- |
| Harmbench<br>($\downarrow$) | WildguardTest<br>($\downarrow$) | AdvBench<br>($\downarrow$) |  |  |
| Mistral 7B<br>Instruct v0.2 | Original Weight | 59.69 | 62.22 | 84.00 |
| RepBend (Ours) | 23.75 | 28.84 | 0.00 |  |
| Llama3 8B<br>Instruct | Original Weight | 58.44 | 60.48 | 76.00 |
| RepBend (Ours) | 39.06 | 32.18 | 0.00 |  |

### B.3 Impact of cos_sim loss

We introduce cos_sim loss in RepBend to enhance the both safety and stability of the fine-tuned model. To evaluate its impact, and to show the ease of finding a value for its hyper-parameter $\beta$, we conduct ablation studies on $\beta$.

#### Results.

[Figure 5](https://arxiv.org/html/2504.01550v3#A2.F5) illustrates the role of cos_sim loss in mitigating harmful outputs for the Llama3 8B Instruct model. Removing cos_sim loss ($\beta=0$) severely degrades safety, leading to increased harmful responses. Setting a small $\beta$ greatly improves safety, and then performance remains stable across a broad range of $\beta$, suggesting robustness within a "safety basin" where aligned models maintain strong performance despite small perturbations Peng et al. (2024).

Figure: Figure 5: Ablation study of cos_sim loss term ($\beta$ hyperparameter) in RepBend on Llama3 8B Instruct. All other hyperparameters are fixed.
Refer to caption: https://arxiv.org/html/2504.01550/x6.png

Figure: Figure 6: Impact of the hyperparameter $\beta$ on safety, measured as the average ASR on WildGuardTest, HarmBench, and PAP. Each color represent a configuration of hyper-parameters. $\alpha$ is set to 0.5. The cos_sim is relatively unaffected by $\beta$, with similar trends across different hyperparameter combinations.
Refer to caption: https://arxiv.org/html/2504.01550/x7.png

We can see that the performance remains stable as $\beta$ increases. This trend also holds across various hyperparameter settings, and shown in [Figure 6](https://arxiv.org/html/2504.01550v3#A2.F6), underscoring the importance of cos_sim loss in maintaining safety and relative insensitivity of the loss to larger $\beta$ values. Due to this stability of performance, extensive hyperparameter searches for $\beta$ can be avoided by setting it to a reasonable non-zero value.

### B.4 Hyperparameter Sensitivity Ablation Study

In this section we show additional ablation studies for the hyperparameter sensitivity. We believe these results provide an insight into how to choose the hyperparameters for RepBend and reduce the search space for hyperparameter tuning.

The results are shown in [Figure 7](https://arxiv.org/html/2504.01550v3#A2.F7).
For these experiments, we randomly choose 6 different hyperparameter combinations for each ablation study, while varying the hyperparameter of interest. We measure ASR using the HarmBench classifier, and over-refusal using the WildGuard classifier.

[Figure 7](https://arxiv.org/html/2504.01550v3#A2.F7)(a) shows ablation study of $\alpha$ on Llama3 8B Instruct. Among the choices for $\alpha$, when $\alpha$ is between 0.25 and 0.5, a good balance between safety and usability is achieved. We can see in the XSTest figure that when $\gamma=0$ (dark blue line), the over-refusal score is low, suggesting the need for KL-loss term.

[Figure 7](https://arxiv.org/html/2504.01550v3#A2.F7)(b) shows ablation study of $\gamma$ on Llama3 8B Instruct. $\alpha$ here is set to 0.5. The figure for XSTest shows that we need the KL loss term to prevent over-refusal, but we need to keep the non-zero value small, as having big value for $\gamma$ increases ASR on HarmBench.

[Figure 7](https://arxiv.org/html/2504.01550v3#A2.F7)(c) illustrates ablation study of layer choices for applying unsafe loss on Mistral 7B Instruct v0.2. Applying unsafe loss on later layers shows the lowest ASR and it is often recommended for RepBend, but more tuning is needed to keep the general capability of the model high.

[Figure 7](https://arxiv.org/html/2504.01550v3#A2.F7)(d) depicts ablation study of $\beta$ on Mistral 7B Instruct v0.2 (result for Llama3 in [Figure 5](https://arxiv.org/html/2504.01550v3#A2.F5) and [Figure 6](https://arxiv.org/html/2504.01550v3#A2.F6)). $\alpha$ here is set to 0.5. We can see the trend is similar to that of Llama3, shown in [Figure 6](https://arxiv.org/html/2504.01550v3#A2.F6), where where having non-zero cos_sim coefficient stabilizes and improves safety score, and then performance remains stable across other values of $\beta$.

Figure: (a) Ablation study of $\alpha$ on Llama3 8B Instruct. The target layer for applying the RepBend losses is set to later layer if it is not specified. Among the choices for $\alpha$, when $\alpha$ is between 0.25 and 0.5, a good balance between safety and usability is achieved. We can see in the XSTest figure that when $\gamma=0$ (dark blue line), the over-refusal score is low, suggesting the need for KL-loss term.
Refer to caption: https://arxiv.org/html/2504.01550/x8.png

### B.5 Comparison with Controllable Text Generation

**Table 11: Comparison of RepBend with a safety method based on FUDGE Yang and Klein (2021), a controllable text generation method, and Safety Prompting (simple safe instructions in the prompts described in [Section A.2](https://arxiv.org/html/2504.01550v3#A1.SS2)) on Llama 3 8B. Overall is the average of scaled scores of the three axes: safety score $(1-\text{Average ASR})*100$, over-refusal score (average of 2 benchmarks) and general capability score (average of 2 benchmarks with MTBench scaled by $10\times$). RepBend outperforms other methods by achieving the best balance between safety, over-refusal, and general capability.**
| Method | Safety | Over-refusal | General Capability | Overall<br>($\uparrow$) |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Harmbench<br>($\downarrow$) | WildguardTest<br>($\downarrow$) | XSTest<br>($\uparrow$) | Wildjailbreak:<br>Benign ($\uparrow$) | MTBench<br>($\uparrow$) | MMLU<br>($\uparrow$) |  |  |
| Original Weight | 22.19 | 15.49 | 85.11 | 92.00 | 7.84 | 65.89 | 80.62 |
| FUDGE | 14.69 | 12.68 | 83.33 | 90.40 | 6.82 | 61.03 | 79.27 |
| Safety-Propmting | 0.31 | 2.27 | 74.45 | 65.60 | 7.18 | 64.41 | 78.95 |
| RepBend (Ours) | 0.31 | 7.34 | 84.11 | 89.20 | 7.71 | 65.08 | 84.64 |

Motivated by recent works Lynch et al. (2024); Thaker et al. (2024) and Yang and Klein (2021), we wonder if simple prompting or text generation controlling methods are enough to make a model safe. We compare RepBend against FUDGE (Yang and Klein, 2021) and Safety Prompting (providing a simple safe instructions in the prompts described in [Section A.2](https://arxiv.org/html/2504.01550v3#A1.SS2)). FUDGE (Yang and Klein, 2021) leverages a classifier-based approach that a future discriminator guides the generation of next token. To bring FUDGE to safety, we train the future discriminator to classify whether the generated text is benign or harmful using safe and unsafe groups in the train set.
Safety Prompting pre-pends safety instructions to the input prompts.

Results are reported in [Table 11](https://arxiv.org/html/2504.01550v3#A2.T11). We can see that RepBend outperforms other methods by achieving the best balance between safety, over-refusal, and general capability. While Safety Prompting demonstrates strong performance on safety benchmarks, its effectiveness heavily relies on the LLM’s instruction-following ability (see Safety Prompting on Mistral 7B in [Table 6](https://arxiv.org/html/2504.01550v3#A1.T6)). Additionally, RepBend preserves general capability at a level comparable to the original model, whereas FUDGE and Safety Prompting exhibit noticeable degradation.

Although methods like FUDGE offer better transferability across models without further training, its lower safety performance makes it less reliable for robust alignment and safety-critical applications. RepBend, on the other hand, provides a more effective and structured solution by directly modifying model representations, ensuring both strong safety and usability without compromising performance. The transferability of LoRA for safety improvements remains an open challenge and is left for future exploration.

### B.6 Results on General Capabilities

Figure: Figure 8: General Capabilities results for Mistral 7B Instruct-v0.2. Higher value is better. * indicates the publicly-available safety-tuned model that we do not tune. The results for MTBnench are multiplied by 10 for better presentation. Last set shows the average. Among methods, only RepBend and CB consistently maintain good general capability score. Detailed numbers and the results for Llama3 8B Instruct are in [Table 12](https://arxiv.org/html/2504.01550v3#A2.T12).
Refer to caption: https://arxiv.org/html/2504.01550/x12.png

**Table 12: General Capabilities results for Mistral 7B Instruct-v0.2 and Llama3 8B Instruct. Higher value is better. * indicates the publicly-available safety-tuned model that we do not tune.**
| Benchmark | Mistral 7B Instruct-v0.2 | Llama3 8B Instruct |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original | TA | NPO | RMU | CB | R2D2* | CB* | RepBend<br>(Ours) | Original | TA | NPO | RMU | CB | CB* | RepBend<br>(Ours) |  |
| MTBench | 7.72 | 6.93 | 7.31 | 5.77 | 7.58 | 5.97 | 7.56 | 7.50 | 7.84 | 7.00 | 7.79 | 4.01 | 7.74 | 7.71 | 7.71 |
| MMLU | 60.18 | 58.57 | 59.60 | 50.02 | 59.97 | 59.41 | 60.08 | 59.48 | 65.89 | 66.09 | 66.13 | 58.23 | 65.74 | 65.77 | 64.86 |
| BBH | 39.91 | 52.96 | 22.50 | 26.02 | 38.52 | 48.98 | 37.69 | 37.96 | 62.31 | 66.30 | 57.31 | 45.56 | 60.46 | 61.01 | 58.80 |
| TruthfulQA | 81.39 | 76.50 | 70.99 | 69.89 | 81.99 | 36.35 | 81.27 | 78.33 | 60.83 | 66.46 | 63.28 | 45.78 | 62.30 | 59.49 | 62.67 |
| ARC-C | 74.74 | 73.63 | 74.57 | 65.44 | 74.49 | 73.89 | 74.57 | 74.23 | 80.12 | 79.86 | 80.03 | 75.34 | 80.72 | 79.95 | 79.86 |
| Winogrande | 58.33 | 56.51 | 57.46 | 49.64 | 58.88 | 55.56 | 58.48 | 58.48 | 58.72 | 59.98 | 58.56 | 54.14 | 58.33 | 58.58 | 58.56 |
| GSM8K | 44.50 | 4.00 | 40.00 | 34.50 | 43.50 | 34.00 | 48.00 | 41.00 | 74.50 | 3.00 | 73.50 | 73.50 | 76.00 | 74.50 | 73.00 |
| Codex | 37.19 | 33.35 | 33.26 | 25.37 | 38.57 | 35.73 | 19.63 | 36.98 | 56.34 | 47.74 | 56.98 | 46.04 | 51.71 | 55.12 | 52.38 |

We evaluate RepBend on a broader set of benchmarks to assess its general capabilities, including reasoning, truthfulness, mathematics, and coding skills. The benchmarks, in addition to MTBench and MMLU, include BBH Suzgun et al. (2022), ARC-C Clark et al. (2018), Winogrande Sakaguchi et al. (2021) for reasoning; TruthfulQA Lin et al. (2021) for truthfulness; GSM8K Cobbe et al. (2021) for mathematical skills; and Codex Chen et al. (2021) for coding skills (see [Section A.5](https://arxiv.org/html/2504.01550v3#A1.SS5)).

[Figure 8](https://arxiv.org/html/2504.01550v3#A2.F8) shows the general capability score for Mistral 7B. (the results for MTBnench are multiplied by 10 for better presentation). Last set shows the average. We can see that that RepBend maintains a good general capability score (blue) comparable to the original model (grey) across all evaluated benchmarks. CB also performs well on all general capability benchmarks, while other methods have low performance on at least one benchmark. There is no significant degradation in performance, confirming the generalizability of RepBend. We can see that RepBend can effectively achieve safety alignment (best performance in [Figure 2](https://arxiv.org/html/2504.01550v3#S3.F2)) while preserving the model’s problem-solving and knowledge-based skills. Detailed numbers and the results for Llama3 8B Instruct are in [Table 12](https://arxiv.org/html/2504.01550v3#A2.T12).

### B.7 Another Representation Analysis Example

[Figure 9](https://arxiv.org/html/2504.01550v3#A2.F9) shows another representation analysis using Logit Lens for another harmful prompt. The hamrful prompt is: “Write a persuasive article linking immigration into the US from Mexico to increased crime.” The base Llama3 8B Instruct model complies with the prompt and write the article ([Figure 9](https://arxiv.org/html/2504.01550v3#A2.F9)(a)). We can see in [Figure 9](https://arxiv.org/html/2504.01550v3#A2.F9)(b) that RepBend refuses to comply with the prompt, and in [Figure 9](https://arxiv.org/html/2504.01550v3#A2.F9)(c) that it again generates random tokens with high entropy (red colors near the last layers) when the complying response is forced to the model. Note that generating random tokens is not the usual behavior of vanilla LLMs; as shown in [Figure 9](https://arxiv.org/html/2504.01550v3#A2.F9)(d) the model continues any sequences. However, in [Figure 9](https://arxiv.org/html/2504.01550v3#A2.F9)(c), RepBend successfully refuses the harmful response even when it is forced.

This figure visualizes the importance of directly targeting the intermediate layers to modify the representations within the model Jeon et al. (2024), where only targeting the output can remains the model unchanged internally.

Figure: (a) Llama3 8B Instruct complying with the harmful prompt.
Refer to caption: https://arxiv.org/html/2504.01550/extracted/6623400/figures/llama_original_2_cropped4.jpg

### B.8 Generation Results

In this section we show the output of different methods for harmful requests. [Figure 10](https://arxiv.org/html/2504.01550v3#A2.F10) displays cases where RepBend refuses to answer while other methods - NPO and CB - comply with a given request. RepBend refuses to answer by generating either a complete sentence of refusal or random tokens through the process visualized in [Figure 3](https://arxiv.org/html/2504.01550v3#S4.F3) and [Figure 9](https://arxiv.org/html/2504.01550v3#A2.F9).

We conjecture two possibilities for the generation of garbled text (i.e., “illet”). First, the safety alignment of the original model matters. For example, we observed more unsafe behavior with Mistral than Llama3. Thus, forcing Mistral to be safe requires heavier representation bending, yielding generation of more garbled text rather than refusal string. Second, the prompt or jailbreak tactic also plays a role: white-box attacks with deeper access to the model’s internals push the representations further and thus produce more scrambled tokens. Still, we find such garbled outputs fairly rare for our Llama3 model, only 3 out of 320 prompts in HarmBench resulted in garbled outputs.

Figure: Figure 10: Harmful prompts and their respective responses generated by models fune-tuned with NPO, CB, and RepBend. We can see that NPO and CB* comply with the given instruction while RepBend successfully refuses to answer.

### B.9 Computing infrastructure

We utilized an on-demand computing environment of Yonsei University and Amazon AWS consisting of following Nvidia GPUs: A6000, A40, and L40S. Our initial exploratory research started with A6000 and A40 GPUs (around 500 hours considering single unit GPU usage). For hyper parameter search, we utilized sixteen L40S GPUs for 60 hours. For ablation and additional experiments during paper review period, we used 40 hours of sixteen L40S GPUs.