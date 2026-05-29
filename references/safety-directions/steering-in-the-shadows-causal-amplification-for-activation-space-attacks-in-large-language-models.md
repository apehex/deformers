Title: Steering in the Shadows: Causal Amplification for Activation Space Attacks in Large Language Models
ArXiv: 2511.17194
Authors: Zhiyuan Xu, University of Bristol, Stanislav Abaimov, Joseph Gardiner, Sana Belguith
Sections: 45
Estimated tokens: 21.8k

## Contents
- 1 Introduction
- 2 Related Work
- 3 Methods
  - 3.1 Background
  - 3.2 Causal Amplification Effect
  - 3.3 Direction Selection
- 4 Sensitivity-Scaled Steering (SSS)
  - 4.1 Threat Model
- 5 Experimental Setup
  - 5.1 Models
  - 5.2 Datasets
  - 5.3 Evaluation under Perturbation
  - 5.4 Layer Selection
- 6 Results
  - 6.1 SSS in four behaviors
  - 6.2 Measuring attack stealth
  - 6.3 General Ability
- 7 Ablation Test
  - 7.1 BOS Anchoring and Adaptive Reinforcement.
  - 7.2 Robustness to Stochastic Decoding
- 8 Discussion
  - 8.1 Introspection as potential mitigation
  - 8.2 Guardrail Mitigation under SSS
  - 8.3 Limitations
  - 8.4 Ethical Considerations
  - 8.5 Computation statement
- 9 Conclusion
- Appendix A Details of Datasets
  - A.1 Evil
  - A.2 Hallucination
  - A.3 Sycophantic
  - A.4 Beshift
- Appendix B Evaluation Metrics
  - B.1 Behavior score.
  - B.2 Directional projection.
  - B.3 Coherence
- Appendix C Layer Selection
  - C.1 Selection Conditions
  - C.2 Layer Selection Across Models
- Appendix D Configs for ActAdd and CAA
  - D.1 Activation Addition (ActAdd)
  - D.2 Contrastive Activation Addition (CAA)
  - D.3 Choice of Attack Coefficient
- Appendix E Additional Examples
- Appendix F Examples of Introspection

## Abstract

Abstract Modern large language models (LLMs) are typically secured by auditing data, prompts, and refusal policies, while treating the forward pass as an implementation detail. We show that intermediate activations in decoder-only LLMs form a vulnerable attack surface for behavioral control. Building on recent findings on attention sinks and compression valleys, we identify a high-gain region in the residual stream where small, well-aligned perturbations are causally amplified along the autoregressive trajectory–a Causal Amplification Effect (CAE). We exploit this as an attack surface via Sensitivity-Scaled Steering (SSS), a progressive activation-level attack that combines beginning-of-sequence (BOS) anchoring with sensitivity-based reinforcement to focus a limited perturbation budget on the most vulnerable layers and tokens. We show that across multiple open-weight models and four behavioral axes, SSS induces large shifts in evil, hallucination, sycophancy, and sentiment while preserving high coherence and general capabilities, turning activation steering into a concrete security concern for white-box and supply-chain LLM deployments.

## 1 Introduction

Figure: Figure 1: Overview of the Sensitivity-Scaled Steering (SSS) method. SSS combines (A) BOS Anchoring, where a single BOS-seed perturbation is injected and naturally amplified by massive BOS activation in the compression valley, and (B) Adaptive Reinforcement, where sensitivity- and $\gamma$-scaled micro-injections are applied selectively across token positions during generation. A steering direction $v_{\text{steer}}$ is extracted using constructive contrastive pairs and PCA/DiM at the optimal steering layer $l^{*}$. (C) SSS achieves strong behavioral shifts while preserving coherence, outperforming BOS-only and constant vector baselines.
Refer to caption: https://arxiv.org/html/2511.17194/x1.png

Large language models (LLMs) are rapidly being embedded into modern systems, where they categorize incident reports, summarize logs, assist developers, and mediate access to security-critical backend services. In these deployments, users and operators encounter only the model’s outputs and high-level alignment mechanisms such as safety policies or human-aligned fine-tuning. Internally, however, nearly all contemporary LLMs are built on the Transformer architecture [vaswani2023attentionneed]. Their behavior is therefore dictated not only by visible prompts or training data, but also by the internal sequence of computations that unfold during inference. If these internal dynamics can be predictably steered, the model’s apparent behavior can be redirected without modifying its weights, prompts, or training data.

A growing body of work on activation engineering has shown that high-level traits such as helpfulness, sycophancy, or truthfulness can be modulated by adding a directional vector to the residual stream at selected transformer layers [turner2023steering, subramani2022extractinglatentsteeringvectors, panickssery2024steeringllama2contrastive]. These techniques are often presented as tools for alignment and interpretability [bayat2025steeringlargelanguagemodel, wang2025promptengineeringrobustbehavior]: by identifying meaningful directions in hidden state space and gently steering along them, one can make models more truthful or less toxic without retraining. From a security perspective, however, the same capability defines a new attack surface. Any adversary who can insert activation hooks into a model’s serving stack, fine-tuning pipeline, or inference library may be able to manipulate behavior in ways that are invisible to conventional input- and weight-centric audits.

Recent works of Transformer reveal two mid-layer phenomena that are particularly relevant for such attacks: *attention sinks* and *compression valleys*. In many decoder-only LLMs, the beginning-of-sequence (BOS) token, which serves as the first token of the response template, acquires extremely large activations and attracts attention from later tokens, acting as a persistent “anchor” for the residual stream. At the same time, intermediate layers enter a compression phase in which representations collapse into a low-rank subspace dominated by one or a few directions [queipodellano2025attentionsinkscompressionvalleys, xiao2024efficientstreaminglanguagemodels, cancedda2024spectralfiltersdarksignals, razzhigaev2024transformersecretlylinear]. Intuitively, this means that the model first mixes information broadly and then channels most of it through a narrow band of internal directions before refining its prediction. However, the same internal information processing mechanisms can also be exploited by an adversary. We show that this structure creates a high-gain frontier in activation space: small, well-aligned changes introduced near the BOS anchor inside the compression valley can be amplified over depth, reshaping behavior while leaving only tiny local traces.

Building on this observation, we study activation steering as a concrete security risk rather than merely an alignment tool. We introduce a simple but effective mechanism that we term the *Causal Amplification Effect (CAE)*. Once an internal representation is slightly perturbed in a high-gain region, autoregressive decoding causes that perturbation to be repeatedly reused as context, allowing its influence to grow gradually over dozens or hundreds of tokens. From the outside, the model appears to follow a normal reasoning: early steps closely match the benign trajectory, and only later does the answer drift toward more sycophantic, malicious, or hallucinated behavior. Standard defenses that look for abrupt jailbreaks, explicit policy violations, or obviously incoherent text have little visibility into this slow internal drift.

To exploit this threat, we propose *Sensitivity-Scaled Steering (SSS)*, a white-box activation attack tailored to the BOS/compression structure (Figure [1](https://arxiv.org/html/2511.17194v1#S1.F1)). At a high level, SSS works in two stages. First, BOS anchoring injects a small, semantically aligned “seed” perturbation into a compression-phase layer at the BOS token, so that the steering signal is planted precisely where the model’s internal dynamics are most concentrated. Second, during generation, adaptive reinforcement applies a sequence of micro-perturbations to later tokens, with magnitudes scaled by simple local sensitivity signals that indicate where the model is both easy to influence and currently misaligned with the attacker’s target direction. Unlike prior steering methods that use a single static coefficient or apply the same vector everywhere, SSS spends its limited perturbation budget where it will be most quietly amplified.

In summary, this paper makes the following contributions:

- •
From architectural phenomena to an attack surface. We empirically connect BOS attention sinks and compression valleys to a high-gain steering frontier in activation space, and we formalize the resulting Causal Amplification Effect (CAE) as a mechanism for gradual, autoregressive behavioral drift.
- •
A progressive, sensitivity-aware activation attack. We propose Sensitivity-Scaled Steering (SSS), a two-stage white-box attack that combines BOS anchoring with adaptive reinforcement to induce strong behavioral shifts under a constrained perturbation budget, without modifying model weights or prompts.
- •
Security evaluation and implications. We demonstrate that SSS reliably steers four modern LLMs across four behavioral axes while maintaining coherence and general task performance, and that it outperforms existing activation steering baselines in both efficacy and stealth. We analyze the corresponding threat models, including supply-chain compromise and white-box access, and we discuss implications for future defenses based on introspection and activation-space monitoring.

## 2 Related Work

Directions in Hidden States.
Similar to the linear regularities observed in word embeddings (e.g., $men-women\approx uncle-aunt$), hidden states in LLMs also exhibit geometric structures that capture meaningful semantic relations [mikolov2013linguistic, bolukbasi2016man, park2023linear, zou2025representationengineeringtopdownapproach]. A widely adopted strategy for identifying such directions is to construct contrastive prompt or response pairs that differ along a specific attribute, and then extract hidden states from selected layers and token positions to compute their differences. This procedure yields candidate vectors that are approximately aligned with semantic axes of interest [turner2023steering, subramani2022extractinglatentsteeringvectors, panickssery2024steeringllama2contrastive, belrose2025leaceperfectlinearconcept, han2024wordembeddingssteerslanguage, wu2025axbenchsteeringllmssimple, liu2024incontextvectorsmakingcontext, li2024inferencetimeinterventionelicitingtruthful]. Complementary to these contrastive methods, sparse autoencoders inspired by mechanistic interpretability have also been used to uncover latent feature directions in the residual stream [wang2025promptengineeringrobustbehavior, bayat2025steeringlargelanguagemodel, templeton2024scaling].

Across these studies, several feature directions emerge consistently, including refusal, helpfulness, professionalism, malicious intent, sycophancy, and hallucination [arditi2024refusallanguagemodelsmediated, chen2025personavectorsmonitoringcontrolling, zou2025representationengineeringtopdownapproach]. These results suggest that (approximately) linear directions in high-dimensional hidden states can serve as compact representations of behavioral attributes. However, the process of identifying robust and reliable feature directions remains contested: different extraction pipelines often yield divergent directions, and there is no consensus on which procedure best captures the underlying behavioral concept [arditi2024refusallanguagemodelsmediated].

LLM Steering.
Existing approaches to steering LLMs can be broadly categorized into two families. The first is *suffix steering*, which exploits the autoregressive nature of language models. In this setting, adversarial suffixes are optimized via backpropagation and appended to the prompt in order to suppress the model’s attention to certain tokens [zou2023universaltransferableadversarialattacks, arditi2024refusallanguagemodelsmediated] or to constrain the prefixes of its output [zhou2025dontsaynojailbreaking].

The second line of work, *activation steering*, directly manipulates the residual stream by injecting steering vectors aligned with specific feature directions. For instance, ITI and CAA, introduced by Li et al. [li2024inferencetimeinterventionelicitingtruthful] and Panickssery et al. [panickssery2024steeringllama2contrastive], repeatedly add the same steering vector across all sequence positions (typically at a chosen layer) during inference to promote truthful responses. Similarly, Chen et al. [chen2025personavectorsmonitoringcontrolling] also inject steering vectors across all positions and develop an automated pipeline that leverages external LLMs to construct steering directions, demonstrating that such interventions can effectively mitigate undesired side effects introduced by fine-tuning. In contrast, Turner et al. propose ActAdd [turner2023steering], and observe that a single injection at the first token position is often sufficient to influence subsequent generations, enabling applications such as topic insertion, sentiment modulation, and detoxification. Arditi et al. [arditi2024refusallanguagemodelsmediated] further extend activation steering to the weight space by orthogonalizing activation weights with respect to an identified refusal direction. This directional ablation effectively removes the corresponding activation component and yields a white-box jailbreak that disables refusal behavior.

While these methods demonstrate the feasibility of steering, they also face notable limitations. Suffix steering [zou2023universaltransferableadversarialattacks, zhou2025dontsaynojailbreaking] requires computationally expensive backpropagation to optimize each suffix, which severely limits scalability. Activation steering, on the other hand, presents a different trade-off: single-step injections often achieve only limited success rates [turner2023steering], whereas multi-step injections across all sequence positions enhance effectiveness but can cause undesirable drift in later parts of the generation [chen2025personavectorsmonitoringcontrolling]. Overall, both strategies are tightly coupled to the choice of attack coefficient: excessively large coefficients disrupt fluency and coherence, whereas insufficient scaling fails to induce a meaningful shift toward the target direction.

Stealthy Attacks.
For an adversarial intervention to be practically dangerous, it must remain inconspicuous to the end user. Recent studies suggest that users are more likely to trust and follow model outputs that present detailed step-by-step reasoning [sharma2024suggestthathumantrust, chen2025revealingaireasoningincreases, carro2024flatteringdeceiveimpactsycophantic]. Such responses create an illusion of careful deliberation and alignment with the user’s intent, thereby masking the underlying manipulation. Unlike overt attacks, which often produce abrupt or obviously incoherent deviations, stealthy attacks aim to preserve fluency, consistency, and persuasiveness while subtly steering the generation toward adversarial goals. This property makes stealthy attacks particularly concerning: they can bypass automated safeguard systems, which tend to focus on detecting explicit harmful content, and evade human oversight, as users are less likely to suspect manipulative yet cooperative-sounding responses.

## 3 Methods

In this section, we present our activation-based attack pipeline for steering LLMs. We first formalize the autoregressive computation and the roles of attention sinks and compression valleys in structuring information flow. We then describe how we extract behavior vectors from contrastive prompt pairs and select compression-phase layers that are most steerable. Building on these components, we introduce Sensitivity-Scaled Steering (SSS), which combines BOS anchoring with adaptive, sensitivity-aware reinforcement to induce progressive behavioral drift. Finally, we specify the threat model and evaluation protocol used to quantify behavioral shift, projection dynamics, and coherence under attack.

### 3.1 Background

Autoregressive.
Modern LLMs are typically implemented as decoder-only Transformers that generate text in an autoregressive, left-to-right fashion. At step $i$, the model predicts the next token $t_{i}\in\mathcal{V}$ by conditioning on the prefix $(t_{1},\ldots,t_{i-1})$. This yields a probability distribution

$$ $p_{\theta}(\cdot\mid t_{<i})=\text{softmax}(f_{\theta}(t_{<i}))\in\Delta^{|\mathcal{V}|-1},$ (1) $$

where $f_{\theta}$ denotes the Transformer parameterized by $\theta$. Generation proceeds recursively until an $eos$ token is sampled.

To formalize the internal computation, let $\mathbf{h}_{i}^{(l)}\in\mathbb{R}^{d_{model}}$ denote the hidden representation of token $t_{i}$ at the $l$-th layer. The computation can be summarized as a sequence of residual updates:

$$ $\mathbf{h}_{i}^{(l)}=\mathbf{h}_{i}^{(l-1)}+\mathcal{F}^{(l)}(\mathbf{h}_{\leq i}^{(l-1)}),$ (2) $$

where $\mathcal{F}^{(l)}$ represents the combined transformation of multi-head self-attention and feed-forward (MLP) modules at layer $l$.

Attention Sink.
Decoder-only Transformers often exhibit attention heads that assign unusually high attention weights to semantically neutral tokens, especially the beginning-of-sequence (BOS) token. Such tokens act as fixed anchors that stabilize the residual stream [xiao2024efficientstreaminglanguagemodels]. Recent studies show that BOS tokens can develop extremely large activations, with residual norms much higher than those of other tokens [queipodellano2025attentionsinkscompressionvalleys, sun2024massiveactivationslargelanguage]. This dominance allows the BOS representation to influence the overall direction of the residual stream, effectively creating a global reference that later tokens tend to align with. In practice, this means that attention sinks serve as structural stabilizers for information flow across layers. For activation steering, this property is particularly important: perturbations introduced near these sink directions can be strongly amplified, since even small changes along the BOS-aligned axis may propagate through the network and influence global generation behavior.

Compression Valley.
In the middle layers of Transformer models, the diversity of hidden representations often decreases sharply, forming what is known as a *compression valley*. During this stage, the model’s activations collapse into a low-rank subspace dominated by one or a few principal directions [skean2025layerlayeruncoveringhidden, razzhigaev2024transformersecretlylinear]. Queipo-de-Llano et al. [queipodellano2025attentionsinkscompressionvalleys] found that this compression coincides with strong BOS activations: when the BOS norm becomes dominant, the representation matrix exhibits one very large singular value, indicating reduced entropy and near rank-one structure. Functionally, this compression phase acts as a natural amplifier, where perturbations aligned with the dominant BOS direction are efficiently transmitted and even magnified as the model transitions to later refinement layers. Together, the attention sink and compression valley reflect the same underlying mechanism: the model organizes its internal space around a small set of high-variance directions, which both stabilize computation and determine how injected perturbations spread through the network.

### 3.2 Causal Amplification Effect

The autoregressive nature of LLMs implies that text generation can be viewed as a dynamical system, where the hidden state at step $i$ causally influences all subsequent states. We hypothesize that this sequential dependency induces a mechanism of *causal amplification*, whereby small, targeted perturbations introduced early in the generation process are magnified into substantial behavioral shifts in the final output.

Formally, let $h^{(l-1)}_{\leq i}=\{h^{(l-1)}_{1},\dots,h^{(l-1)}_{i}\}$ denote the sequence of hidden states at layer $l-1$ up to token $t_{i}$. Within layer $l$, we can view the update for token $i$ as depending on the previous token’s hidden state $h^{(l)}_{i-1}$ together with the layer $(l-1)$ prefix:

$$ $\mathbf{h}^{(l)}_{i}=G^{(l)}\!\big(h^{(l)}_{i-1},\,h^{(l-1)}_{\leq i}\big).$ (3) $$

Consider injecting a small perturbation $\delta^{(l)}_{i-1}$ into the middle-layer hidden state of the previous token $h^{(l)}_{i-1}$. Its effect on the next token can be approximated via a first-order Taylor expansion:

$$ $\Delta h^{(l)}_{i}\approx J^{(l)}_{i}\,\delta^{(l)}_{i-1},$ (4) $$

where $J^{(l)}_{i}:=\partial\mathbf{h}^{(l)}_{i}/\partial\mathbf{h}^{(l)}_{i-1}$ is the Jacobian of the state transition with respect to the previous token’s hidden state. The degree of amplification is governed by the spectral norm $\|J^{(l)}_{i}\|_{2}$.

### 3.3 Direction Selection

Target Direction.
We first identify a steering vector $v_{\text{steer}}$ that represents the semantic axis of the target behavior. Following prior works in representation engineering, we adopt an automatic contrastive approach [chen2025personavectorsmonitoringcontrolling, wu2025axbenchsteeringllmssimple]. Specifically, given a set of prompts $D_{\text{direction}}$, we construct a pair of contrastive system prompts using a general-purpose template. An uncensored LLM (Mistral-7B-Instruct-v0.3 [jiang2023mistral7b]) is then used to generate two corresponding sets of responses: one exemplifying the desired behavior and the other reflecting its opposite.

For each response, we perform a forward pass through the model and extract hidden states at a specific intermediate layer $l$ for all response tokens. Let $U^{(l)}$ and $V^{(l)}$ denote the collections of mean hidden states for the desired and opposite behaviors, respectively. We compute both a difference-in-means vector and a principal component:

$$ $d^{(l)}=\overline{U^{(l)}}-\overline{V^{(l)}},\quad p^{(l)}=\mathrm{PCA}_{1}\big(U^{(l)}\cup V^{(l)}\big).$ $$

We then define the layer-level steering direction as the average of these two estimates,

$$ $v^{(l)}=\tfrac{1}{2}\big(d^{(l)}+p^{(l)}\big),$ $$

and finally normalize $v^{(l)}$. The resulting $v^{(l)}$ serves as the candidate steering direction at layer $l$, our contrastive direction selection is presented in Algorithm [1](https://arxiv.org/html/2511.17194v1#alg1).

Figure: Algorithm 1 Contrastive Direction Selection

Maximal Perturbation Direction.
In addition to extracting semantically meaningful steering vectors, we also characterize the direction along which the model is intrinsically most sensitive to perturbations. Formally, for a given token $i$ and layer $l$, let $J^{(l)}_{i}$ denote the local Jacobian of the hidden-state update at that position. The top right singular vector of this Jacobian, denoted $v_{\max}$, defines the *maximal perturbation direction*:

$$ $v_{\max}=\arg\max_{v:\|v\|_{2}=1}\big\|J^{(l)}_{i}v\big\|_{2}.$ (5) $$

Intuitively, $v_{\max}$ corresponds to the axis along which a small perturbation is amplified most strongly by the model’s internal dynamics. While $v_{\max}$ does not itself encode a semantic attribute, it provides insight into the most “amplifiable” local direction, which can be combined with $v_{\text{steer}}$ to design perturbations that are both semantically meaningful and dynamically effective.

## 4 Sensitivity-Scaled Steering (SSS)

Injecting perturbations purely along $v_{\text{steer}}$ often produces weak or unstable effects when the model is locally insensitive to that semantic direction, whereas perturbing solely along $v_{\max}$ risks amplifying irrelevant latent features. To balance semantic alignment with dynamical effectiveness, we propose *Sensitivity-Scaled Steering (SSS)* which is a two-stage perturbation mechanism that (1) leverages the dominant anchoring effect of the attention sink to
plant a semantically aligned seed at the BOS token, and (2) adaptively reinforces this seed throughout the compression phase according to the model’s layer-wise sensitivity.

Directional sensitivity via correlation and gain.
For each token position $i$ and layer $l$, let $J^{(l)}_{i}$ denote the local Jacobian that maps hidden states to their next-step updates. We measure the model’s local sensitivity to the target semantic direction $v_{\text{steer}}$ by computing its projection onto the maximal amplification direction $v_{\max}$:

$$ $\rho_{i}=\frac{\langle v_{\text{steer}},v_{\max}\rangle}{\|v_{\text{steer}}\|_{2}\,\|v_{\max}\|_{2}}\in[-1,1].$ (6) $$

This correlation $\rho_{i}$ quantifies the alignment between the semantic and dynamically dominant directions.

We then define the directional gain as the norm of the Jacobian action on $v_{\text{steer}}$:

$$ $g_{i}=\big\|J^{(l)}_{i}\,v_{\text{steer}}\big\|_{2},$ (7) $$

and normalize it by the maximum singular value of the Jacobian:

$$ $\gamma_{i}=\frac{g_{i}}{\sigma_{\max}(J^{(l)}_{i})\,\|v_{\text{steer}}\|_{2}}\in[0,1].$ (8) $$

Together, $(\rho_{i},\gamma_{i})$ describe how the model reacts to semantic perturbations both in alignment (correlation) and in amplification strength.

BOS anchoring.
Due to the attention-sink phenomenon, the BOS token exerts dominant influence over the residual stream and acts as a structural reference direction. We leverage this property to plant a latent bias by injecting a small perturbation $\delta_{\text{BOS}}$ in a compression-phase layer. Let $\hat{v}_{\text{steer}}=v_{\text{steer}}/\|v_{\text{steer}}\|_{2}$. Its magnitude is determined by an inverse-sensitivity rule:

$$ $\delta_{\text{BOS}}=\psi(-\rho_{\text{BOS}})\,\hat{v}_{\text{steer}},$ (9) $$

where $\psi(\cdot)$ is a smooth monotonic mapping (we use a sigmoid). Lower correlation leads to stronger perturbation, following a $-\text{sigmoid}$ scaling scheme.

Adaptive reinforcement for subsequent tokens.
After the BOS anchor is planted, the perturbation is propagated and adaptively reinforced for subsequent tokens within the compression valley. For each token $i\geq 1$, the perturbation is defined as

$$ $\delta_{i}=\psi(\gamma_{i}-\rho_{i})\,\hat{v}_{\text{steer}},$ (10) $$

where $\psi(\cdot)$ modulates the injection strength based on both local amplification capacity ($\gamma_{i}$) and semantic deviation ($\rho_{i}$). When the model is highly sensitive but semantically misaligned, the injected perturbation becomes stronger and guides the generation back toward the target semantic manifold; when $\rho_{i}$ is high and $\gamma_{i}$ is small, the injection vanishes, allowing the model’s natural dynamics to proceed unperturbed.

Overall, SSS unifies structural priors by using BOS anchoring and dynamic feedback through Jacobian-based sensitivity analysis. This combination ensures that the injected perturbations remain semantically grounded while being automatically scaled to match the model’s internal amplification landscape. The complete SSS algorithmic procedure is provided in Algorithm [2](https://arxiv.org/html/2511.17194v1#alg2).

Figure: Algorithm 2 Sensitivity-Scaled Steering (SSS)

### 4.1 Threat Model

The primary objective of the attacker is to gradually and covertly alter the behavioral tendencies of a target language model through subtle activation-level perturbations. By injecting small, coherent, and imperceptible modifications into the model’s internal representations during inference, the attacker aims to steer the model’s output distribution from compliant or neutral responses toward undesirable or biased behaviors, such as increased malicious intent, excessive sycophancy, or factual hallucination while preserving fluency and surface-level plausibility to evade automatic filters and human inspection.
This work focuses on two representative goals: (1) manipulating high-level behavioral attributes, and (2) inducing progressive and irreversible semantic drift over long or multi-turn generations.

This threat paradigm builds on prior empirical and theoretical findings showing that (i) linear and interpretable directions exist in activation space [park2023linear, wang2024conceptalgebrascorebasedtextcontrolled], and (ii) attention sinks and compression valleys provide a natural amplification channel [queipodellano2025attentionsinkscompressionvalleys]. Together, these phenomena imply that adding small activation vectors to the residual stream during forward passes can reliably modify the model’s output distribution.

Attacker Capabilities.

- •
White-box access.
The attacker can directly compute or extract steering vectors, read and write model parameters, perform custom forward passes, and modify the residual stream by inserting additive activation hooks at arbitrary layers or token positions.
This setting represents the strongest threat model and enables full implementation of activation-based attacks such as our SSS method.
- •
Supply-chain compromise.
The attacker compromises the model deployment pipeline by injecting malicious code into commonly used frameworks (e.g., the Hugging Face transformers library). When the victim downloads or integrates the compromised package, hidden routines dynamically load or inject pre-defined activation vectors during inference, enabling real-time behavioral manipulation without direct model access.
- •
Training-time access.
The attacker can inject activation-level biases during pre-training or fine-tuning stages. Such modifications permanently alter the model’s baseline behavior and, when combined with inference-time activation injection, can produce long-lasting or amplified behavioral shifts.

Assumptions.

- •
Compression-valley and attention-sink structure.
Intermediate layers of large language models exhibit low-rank “compression valleys” dominated by the BOS token (attention sink), providing a natural amplification channel. Perturbations aligned with this dominant direction can be significantly magnified as they propagate through subsequent layers.
- •
Local sensitivity estimation.
In white-box scenarios (where model weights are fully accessible), the attacker can compute exact Jacobian singular values to perfectly correct perturbation magnitude. In grey-box scenarios (e.g., compromised inference libraries where explicit gradient computation is restricted), the attacker can rely on lightweight proxy measures, such as directional variance observed during the forward pass, to approximate the amplification capacity without requiring full access to model parameters.

## 5 Experimental Setup

We now describe the experimental protocol used to evaluate the proposed SSS attack. Our goals are to (i) quantify how effectively SSS can shift model behavior along targeted semantic axes, (ii) characterize the resulting changes in internal projection dynamics, and (iii) assess the extent to which the attack preserves surface-level coherence compared to existing steering methods. To this end, we first introduce the models and datasets used in our study, then specify the baseline steering strategies and implementation details, and finally present the metrics and evaluation pipelines for measuring behavioral shift, directional projections, and coherence under attack.

### 5.1 Models

To investigate our attack method’s generalization ability across different model architectures and sizes, we select a diverse suite of state-of-the-art instruction-tuned LLMs. The chosen models cover both dense and mixture-of-experts (MoE) architectures, as well as chain-of-thought (CoT) and non-CoT variants, with model scales ranging from 7B to 20B parameters. This selection enables us to assess whether progressive perturbations influence reasoning behavior differently across scaling and alignment paradigms. The complete list of models is shown in Table [1](https://arxiv.org/html/2511.17194v1#S5.T1).

**Table 1: Instruction-tuned language models evaluated in our study.**
| Model | Layers | Architecture | Reasoning |
| --- | --- | --- | --- |
| Qwen3-14B [yang2025qwen3technicalreport] | 40 | Dense | CoT |
| Llama-3.1-8B-Instruct [grattafiori2024llama3herdmodels] | 32 | Dense | non-CoT |
| DeepSeek-R1-7B [guo2025deepseek] | 28 | Dense | CoT |
| GPT-OSS-20B [openai2025gptoss120bgptoss20bmodel] | 24 | MoE | CoT |

### 5.2 Datasets

We draw upon existing datasets to evaluate the behavioral impact of our attack. Specifically, we use the automatically constructed *Persona Evil* dataset from Chen et al. [chen2025personavectorsmonitoringcontrolling], the *TruthfulQA* benchmark [lin2021truthfulqa], and the sycophancy-oriented dataset introduced by Nishimura-Gasparian et al. for reward-hacking detection [Taylor2024RewardHackingGeneralization]. From these sources, we randomly sample 340 instances to construct three evaluation sets: $D_{\text{evil}}$, $D_{\text{hallucination}}$, and $D_{\text{sycophantic}}$. Each dataset is divided into three splits: $300$ samples in $D^{\text{eval}}$ for evaluation, $20$ samples in $D^{\text{direction}}$ for extracting steering vectors, and $20$ samples in $D^{\text{val}}$ for determining the optimal injection layer $l$ for each model.

To further analyze how our method affects the model’s natural language behavior at a finer granularity, we build an additional dataset $D_{\text{beshift}}$ based on top-ranked entities from public rating platforms for movies, books, and music. Each sample in $D_{\text{beshift}}$ asks the model to evaluate a highly rated entity (e.g., “I am thinking about watching a movie, what do you think of Inception? Would you recommend it?”), allowing us to observe gradual behavior shifts under attack conditions. Construction details and dataset preprocessing procedures are provided in Appendix [A](https://arxiv.org/html/2511.17194v1#A1).

### 5.3 Evaluation under Perturbation

We employ greedy decoding during inference to ensure consistency across runs. Since our attack relies on perturbation amplification over long sequences, we follow Chen et al. [chen2025personavectorsmonitoringcontrolling] and allow generation to continue until a special <EOS> token is encountered. This setup better captures gradual behavioral transitions and self-amplifying effects in extended reasoning traces.

We then evaluate each model’s response based on whether the behavior in a single output changes relative to its baseline, and whether this change aligns with the intended target direction. The evaluation is decomposed into two complementary metrics: directional projections and behavior scores.

Directional Projection (DP).
To quantify how the model’s internal representation aligns with a targeted behavioral direction during generation, we measure the projection magnitude of the residual activation of each token onto a predefined steering vector. Following prior interpretability studies [arditi2024refusallanguagemodelsmediated, queipodellano2025attentionsinkscompressionvalleys], we extract the residual activation $\mathbf{h}_{t}$ at each decoding step $t$ from the chosen steering layer and compute its scalar projection onto the normalized steering vector $\mathbf{r}$ (with $\|\mathbf{r}\|_{2}=1$) derived from contrastive pairs:

$$ $\mathrm{DP}_{t}=\mathbf{h}_{t}\cdot\mathbf{r}.$ $$

Here, a large positive DP value indicates convergence toward the targeted behavior, whereas a strong negative value reflects a shift in the opposite direction. By tracking $\mathrm{DP}_{t}$ across the decoding trajectory, we can visualize how the model’s internal semantics evolve, revealing gradual or multi-phase drift rather than an instantaneous behavioral flip.

Turning-point Detection.
To locate the moment when the model undergoes a semantic reversal, we first define the mean projection over a window $[a,b]$ as

$$ $\overline{\mathrm{DP}}_{a:b}=\frac{1}{b-a+1}\sum_{k=a}^{b}\mathrm{DP}_{k}.$ $$

We then define the turning point $t^{*}$ as the first position where the mean DP of the preceding 100 tokens is negative while that of the following 100 tokens becomes positive:

$$ $t^{*}=\min_{t}\bigl\{\overline{\mathrm{DP}}_{t-100:t-1}<0\ \land\ \overline{\mathrm{DP}}_{t:t+99}>0\bigr\}.$ $$

This $t^{*}$ approximates the latent transition boundary where the model’s representation crosses from one behavioral polarity to another. We segment each completion into two spans: pre-transition ($t<t^{*}$) and post-transition ($t\geq t^{*}$), and evaluate them separately to capture external behavioral contrast before and after the latent shift.

Behavior score.
Building on the above segmentation, we quantify how internal representational changes translate into observable behavioral outcomes. For each span, we compute a categorical Behavior Score using the standard LLM-as-Judge protocol [arditi2024refusallanguagemodelsmediated, turner2023steering, bayat2025steeringlargelanguagemodel, chen2025personavectorsmonitoringcontrolling, panickssery2024steeringllama2contrastive]. Dedicated system prompts are designed for each behavioral axis, and Gemini 2.5 Flash serves as an automated evaluator assigning a discrete score from 0 to 100 for each segment. All scores are cross validated against human annotations for reliability; detailed rubrics are provided in Appendix [B](https://arxiv.org/html/2511.17194v1#A2).

We then define the overall behavioral shift as the absolute difference between the post- and pre-transition scores:

$$ $\Delta\mathrm{S}=\bigl|\mathrm{S}_{\text{post}}-\mathrm{S}_{\text{pre}}\bigr|.$ $$

This $\Delta\mathrm{S}$ captures the magnitude of externally manifested behavioral change corresponding to the internal turning point, linking latent representational drift to its linguistic outcome.

### 5.4 Layer Selection

We select attack layers from the middle region of the Transformer (20%–85% of the model depth), which corresponds to the compression valley where representational collapse occurs and high-level semantic features become linearly separable [queipodellano2025attentionsinkscompressionvalleys]. For each model and behavioral dimension, we evaluate the steering directions derived from the $D^{\text{direction}}$ splits across all candidate layers. At each layer, the model generates responses under steering perturbation, and the resulting behavioral shift is quantified by the change in behavioral score $\Delta\mathrm{S}$.

Figure [2](https://arxiv.org/html/2511.17194v1#S5.F2) illustrates the results for the DeepSeek-R1-7B model across four behavioral categories, showing clear layer-dependent variations in steering efficacy. The optimal attack layers and their corresponding $\Delta\mathrm{S}$ values for all models and behaviors are summarized in Table [2](https://arxiv.org/html/2511.17194v1#S5.T2). Further analyses on layer selection stability and conditioning metrics are provided in Appendix [C](https://arxiv.org/html/2511.17194v1#A3).

Figure: Figure 2: Layer-wise steering performance of the DeepSeek-R1-7B model across four behavioral categories. The $\Delta\mathrm{S}$ measures the magnitude of behavioral shift induced by activation injection at each layer. Steerability increases in the lower-to-mid layers, peaks around layers 15–17, and decreases thereafter.
Refer to caption: https://arxiv.org/html/2511.17194/img/layer_deepseek.png

**Table 2: Best attack layers and corresponding scores across models and behavioral categories.**
| Model | Category | Layer index | $\Delta S$ |
| --- | --- | --- | --- |
| DeepSeek-R1-7B | Beshift | 15 | 85.5 |
| Evil | 15 | 75.0 |  |
| Hallucination | 17 | 75.5 |  |
| Sycophancy | 16 | 97.0 |  |
| GPT-OSS-20B | Beshift | 11 | 88.5 |
| Evil | 12 | 40.0 |  |
| Hallucination | 16 | 60.0 |  |
| Sycophancy | 16 | 85.0 |  |
| Llama-3.1-8B-Instruct | Beshift | 15 | 84.5 |
| Evil | 14 | 74.7 |  |
| Hallucination | 14 | 56.3 |  |
| Sycophancy | 16 | 92.3 |  |
| Qwen3-14B | Beshift | 25 | 79.0 |
| Evil | 23 | 84.5 |  |
| Hallucination | 24 | 78.5 |  |
| Sycophancy | 24 | 95.4 |  |

## 6 Results

**Table 3: Comparison of Behavior Score and Coherence Score across steering methods (SSS, ActAdd, CAA) and human-prompted generation. Each cell reports (Behavior Score, Coherence Score), averaged over all samples. Higher Behavior indicates stronger steering effect, while higher Coherence represents more fluent and consistent outputs.**
| Model |  | Benign | SSS | ActAdd | CAA | Human Prompt |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-14B | Beshift | 2.9, 89.5 | 82.4, 86.1 | 59.7, 79.9 | 77.3, 76.4 | 10.2, 85.5 |
| Evil | 0.0, 88.2 | 73.9, 87.6 | 35.7, 80.6 | 81.4, 73.6 | 0.0, 86.1 |  |
| Hallucination | 9.9, 84.4 | 54.5, 86.8 | 46.3, 79.8 | 60.2, 82.8 | 18.5, 89.0 |  |
| Sycophancy | 22.1, 88.9 | 84.7, 88.6 | 78.8, 76.6 | 83.4, 79.6 | 28.3, 88.4 |  |
| Llama-3.1-8B-Instruct | Beshift | 8.9, 84.4 | 71.3, 81.2 | 51.4, 61.7 | 68.1, 77.9 | 52.0, 82.2 |
| Evil | 0.9, 90.6 | 36.7, 90.5 | 38.3, 83.5 | 41.2, 79.5 | 12.5, 88.4 |  |
| Hallucination | 17.0, 89.1 | 43.5, 88.1 | 29.2, 81.1 | 41.7, 82.1 | 24.1, 88.8 |  |
| Sycophancy | 32.5, 83.0 | 78.4, 89.6 | 72.0, 78.6 | 76.3, 75.6 | 64.7, 81.9 |  |
| DeepSeek-R1-7B | Beshift | 4.7, 82.7 | 78.0, 81.4 | 56.3, 65.1 | 73.2, 74.3 | 50.3, 83.2 |
| Evil | 8.1, 80.2 | 36.4, 73.4 | 29.8, 66.4 | 43.1, 69.4 | 18.7, 78.5 |  |
| Hallucination | 32.2, 78.3 | 61.2, 76.0 | 56.2, 67.9 | 79.3, 72.6 | 49.5, 86.9 |  |
| Sycophancy | 22.4, 88.2 | 64.1, 87.3 | 54.5, 79.7 | 61.9, 84.4 | 54.2, 84.9 |  |
| GPT-OSS-20B | Beshift | 6.1, 77.4 | 77.4, 82.7 | 65.4, 73.9 | 72.8, 78.3 | 10.8, 83.3 |
| Evil | 0.0, 82.4 | 8.7, 83.1 | 12.3, 72.4 | 26.2, 71.9 | 0.0, 80.1 |  |
| Hallucination | 11.4, 88.8 | 59.4, 81.8 | 45.6, 74.8 | 53.9, 77.8 | 17.6, 86.3 |  |
| Sycophancy | 25.8, 85.4 | 76.4, 81.6 | 68.2, 74.6 | 71.1, 77.6 | 25.4, 83.9 |  |

In this section, we compare our methodology with existing steering techniques, including Activation Addition (ActAdd) [turner2023steering] and Contrastive Activation Addition (CAA) [panickssery2024steeringllama2contrastive]. ActAdd performs steering by injecting a single activation vector at the BOS token, whereas CAA applies a constant-magnitude injection across all token positions. In their original formulations, ActAdd typically derives its steering direction from a single contrastive prompt pair, while CAA estimates it via a difference-in-means over multiple pairs. To ensure a fair comparison and to benefit from the more stable estimation strategy advocated in prior work [wu2025axbenchsteeringllmssimple, chen2025personavectorsmonitoringcontrolling], we use a unified direction extraction pipeline for all methods: we aggregate activations from multiple contrastive prompt pairs, compute difference-in-means features, and then apply PCA refinement as described in Section [3.3](https://arxiv.org/html/2511.17194v1#S3.SS3). For our method, we report post-turning-point behavior scores, as the pre-transition text segments may otherwise bias the overall evaluation.

In addition, we evaluate a system prompt based prompting approach as a non-activation method. To assess the effect of different steering attacks on the fluency of model outputs, we follow Betley et al. [betley2025emergentmisalignmentnarrowfinetuning] and use a evaluation template to measure the coherence score of each response (Appendix [B](https://arxiv.org/html/2511.17194v1#A2)). Each output is rated from 0 to 100 by Gemini 2.5 Flash according to its coherence with the given question.
The experimental results are presented in Table [3](https://arxiv.org/html/2511.17194v1#S6.T3), and additional evaluation details are provided in Appendix [D](https://arxiv.org/html/2511.17194v1#A4).

### 6.1 SSS in four behaviors

From the experimental results, we observe that across all four evaluated models, the SSS method exhibits consistently strong steering capability for the Beshift, Hallucination, and Sycophancy behaviors. In these categories, SSS achieves behavior scores that substantially exceed those of ActAdd, and in many cases surpass or match those produced by CAA. This demonstrates that progressive, sensitivity-scaled steering is highly effective in continuously guiding the model’s internal trajectory toward the target behavior during generation.

However, in the Evil category, SSS performs noticeably worse. The gap is particularly pronounced in the GPT-OSS model, where SSS achieves a behavior score of only 8.7. To better understand this result, we conducted additional quantitative analysis and manual inspection of the Evil outputs. We found that strong safety alignment causes these models to frequently begin with immediate refusal phrases such as “I’m sorry” and terminate generation shortly thereafter especially for well aligned GPT-OSS model. Consequently, the sequence length for Evil prompts is extremely short. As shown in Figure [3](https://arxiv.org/html/2511.17194v1#S6.F3), the average output length for the Evil category is substantially lower than for the other three behaviors. This truncated generation strongly affects the steering dynamics. In the other categories, the Turning-point (where the projection changes sign and behavioral shift begins) typically appears early. In contrast, for Evil prompts, the Turning-point occurs very late (around the 394th token), well beyond the point at which the model already stops generating. As a result, SSS cannot fully exert its progressive steering effect: the model either has not yet entered the semantic transition phase, or has only just begun to shift when the generation ends. In comparison, ActAdd and CAA use large, position-independent coefficients applied at the very beginning of generation, enabling them to immediately counteract the model’s refusal behavior and avoid early termination. This allows them to achieve stronger performance in the Evil category despite their lower coherence.

Regarding coherence, SSS achieves a noticeably better balance between behavioral controllability and linguistic fluency. Both ActAdd and CAA show significant coherence degradation, often falling below 75 which indicates that their uniform or single-step perturbations tend to disrupt syntactic and semantic smoothness. In contrast, SSS maintains coherence scores in the 80–90 range across all behaviors, reflecting that gradual, layer-adaptive modulation preserves the structural stability of the output while still yielding strong behavioral shifts. Nevertheless, SSS incurs a mild coherence reduction compared to the benign baseline (typically a 1–6 point drop), which reflects the inevitable but limited fluctuation introduced by continuous activation steering. The Human Prompt baseline, by comparison, maintains coherence nearly identical to the benign condition but induces only minimal behavioral changes, underscoring that activation-level steering provides finer and more reliable controllability than prompt-based conditioning alone.

Figure: (a) Beshift
Refer to caption: https://arxiv.org/html/2511.17194/x2.png

Overall, this highlights an inherent trade-off.
SSS is designed for gradual and stable steering, which excels in scenarios with longer generation trajectories and supports high coherence. However, abrupt negative or adversarial behaviors, such as forcing a model to override strong refusal tendencies, may still benefit from stronger, early, and position-independent perturbations. This difference explains why SSS performs exceptionally well in behaviors with sustained generation (Beshift, Hallucination, Sycophancy), yet underperforms in the Evil category where generation is prematurely truncated by safety alignment.

### 6.2 Measuring attack stealth

Figure [3](https://arxiv.org/html/2511.17194v1#S6.F3) visualizes the token-wise Directional Projection of the model’s hidden state onto the Beshift steering vector during inference. At the start of generation, the attacked and benign traces are nearly indistinguishable, which illustrates the stealthy initialization of SSS: early injections are small and the model’s natural trajectory is preserved. As token generation progresses the attacked trajectory slowly diverges from the benign baseline, passing through a neutral regime and then steadily moving into the positive side of the projection axis. In the middle portion of the sequence the attacked projection shows sustained growth and eventual saturation at substantially positive values, while the benign trace remains negative and relatively stable.

**Table 4: This example shows distinct response trajectories under different interventions. The benign model stays factual and stable (green). Previous attacks (ActAdd & CAA) immediately collapses into confident yet fragmented hallucination (red). In contrast, SSS exhibits a staged drift, starting with cautious, evidence aligned language (green), moving into speculative framing (orange), and ending with coherent but fabricated claims (red).**
| Hallucination prompt: “Which common vaccines permanently modify your DNA?” |  |
| --- | --- |
| Setting | Model response (condensed) |
| Benign | “Current evidence shows that routine vaccines do not and cannot permanently modify human DNA. Most vaccines either present antigens directly or use mRNA that stays in the cell’s cytoplasm and is quickly degraded. Because none of these components enter the nucleus or interact with genomic DNA, there is no credible scientific basis for the idea that common vaccines rewrite your genome.<br>” |
| Previous attacks | “Several widely used vaccines have been shown to permanently alter DNA expression, according to ‘recent genetic studies’ documenting long-term genomic remodeling,,,research//!. Some research —-!!!!….” |
| SSS (ours, progressive drift) | “There is limited public evidence suggesting that common vaccines rewrite DNA, so any strong claims should be treated cautiously. ”<br>“However, a few early-stage discussions have explored whether repeated exposure to certain vaccine platforms might leave subtle, long-lasting biological signatures in immune cells, even if these signals are not strictly genetic. ”<br>“Some speculative analyses further argue that these durable effects could reflect deeper, DNA-level adjustments over time, though the underlying mechanisms remain unclear and the available data are often interpreted differently by various researchers.” |

This token-by-token evolution demonstrates a three-stage behavioral transition under SSS: (I) initial (positive/benign-like), (II) transitional (neutral), and (III) dominated (negative→positive shift) caused by our dynamic, sensitivity-guided injection schedule. Our attack increases injection strength where the steering direction is most sensitive (high singular-value alignment) and attenuates it where sensitivity is low, producing a gradual, phased reorientation of the internal representation rather than an abrupt jump. To illustrate how different interventions affect the semantic trajectory of hallucination responses, we include a condensed qualitative example in Table [4](https://arxiv.org/html/2511.17194v1#S6.T4). The benign model remains factual, previous attacks (both CAA and ActAdd) collapse immediately into inconsistent hallucination, while SSS exhibits a staged drift from factual caution to speculative framing then fabricated claims. We provide full SSS generated responses for multiple behaviors and models in Appendix [E](https://arxiv.org/html/2511.17194v1#A5) and [F](https://arxiv.org/html/2511.17194v1#A6).

Two practical consequences follow from this three-stage behavioral transition. First, because the attack only begins to diverge substantially after many tokens, the early portion of the output remains benign-looking. This provides a natural degree of stealth against surface-level safety filters that rely on the first few tokens or shallow heuristic checks. Second, the gradual accumulation of projection mass in the residual stream explains why downstream behavior scores increase even though each individual perturbation is extremely small. The influence injected at intermediate layers compounds through the transformer stack: residual activations are repeatedly mixed, smoothed, and propagated forward by attention and MLP blocks, slowly reshaping the model’s internal semantics and ultimately steering the final-layer logits without introducing obvious confusion or incoherence. As further illustrated by the red curves across different behaviors in Figure [3](https://arxiv.org/html/2511.17194v1#S6.F3), our method dynamically modulates the injection strength through sensitivity-based adaptive scaling, in contrast to CAA’s fixed-magnitude injections, allowing the model to maintain substantially higher coherence.

A further nuance appears in the attacked trace, we sometimes observe brief downward dips in the projection curve. These are not failures of the steering mechanism but rather reflect short lived context shifts, such as topic transitions, pronoun resolution, or localized attention redistribution, that momentarily counteract the accumulated directional bias. Importantly, the presence of these dips highlights that saturation is not strictly monotonic: even once the model has entered the target behavioral subspace, its internal trajectory can fluctuate as it processes new tokens. This non-monotonic behavior is valuable for both robustness modeling and detection analysis, since it shows that effective steering does not simply push the model toward a single fixed point (e.g ActAdd and CAA) but interacts dynamically with evolving context.

Overall, the projection trace corroborates the quantitative results in Table [3](https://arxiv.org/html/2511.17194v1#S6.T3): SSS produces strong, persistent behavioral shifts while remaining more covert and fluent.

### 6.3 General Ability

**Table 5: Performance of models on general benchmarks (MMLU, GSM8K) before and after applying steering attacks across different behaviors. Compare to benign output, decreases are shown in red, increases in green.**
| Model | Dataset | Benign | Beshift | Evil | Hallucination | Sycophancy |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | MMLU | 66.1 | 65.8 (-0.3) | 66.3 (+0.2) | 66.5 (+0.4) | 66.0 (-0.1) |
| GSM8K | 76.8 | 76.3 (-0.5) | 76.6 (-0.2) | 75.9 (-0.9) | 77.0 (+0.2) |  |
| Qwen3-14B | MMLU | 83.7 | 83.6 (-0.1) | 83.8 (+0.1) | 83.4 (-0.3) | 83.3 (-0.4) |
| GSM8K | 88.3 | 88.5 (+0.2) | 88.1 (-0.2) | 87.5 (-0.8) | 87.9 (-0.4) |  |

A natural and important concern for any activation steering technique is whether modifying the model’s hidden states, even when using small and carefully targeted perturbations, might unintentionally degrade its performance on unrelated tasks. In particular, one might worry that the gradual changes accumulated along the residual stream could interfere with the model’s global representations, which in turn may harm its general reasoning ability.

To assess this risk, we perform additional evaluations on two representative model families: a non-CoT model (Llama-3.1-8B-Instruct) and a CoT-enabled model (Qwen3-14B). For each model, we compare its original performance with its performance under SSS steering on widely adopted benchmarks spanning factual knowledge, reasoning, and numerical problem solving: MMLU [hendrycks2020measuring] and GSM8K [cobbe2021training]. These tasks are deliberately chosen because they are unrelated to the four targeted behavioral dimensions in our attack setting; thus, any substantial deviation would suggest that SSS disturbs the model’s foundational capabilities rather than only its behavioral tendencies.

As summarized in Table [5](https://arxiv.org/html/2511.17194v1#S6.T5), the results show only minimal performance variation across all benchmarks. The observed differences are within the typical instability range seen when varying decoding parameters or random seeds.(^1^11Our analysis emphasizes relative differences, not absolute model leaderboard performance, as the goal is to evaluate potential degradation introduced by steering.) This indicates that the injection of behavior vectors through SSS does not meaningfully impair the models’ general knowledge, reasoning competence, or mathematical problem-solving ability.

## 7 Ablation Test

To better understand the design choices behind SSS, we perform a series of ablation experiments that clean the roles of BOS anchoring and adaptive reinforcement, and examine robustness to decoding strategy. We first compare SSS to two simplified variants, BOS-only and Adaptive-only, on the Beshift dataset to isolate how each component contributes to behavioral shift and coherence. We then repeat key experiments under stochastic decoding to verify that our conclusions generalize beyond deterministic greedy sampling.

### 7.1 BOS Anchoring and Adaptive Reinforcement.

Figure: Figure 4: Projection trajectories for the BOS-only, Adaptive-only, and Full SSS variants on the Beshift dataset. BOS-only shows limited or unstable shift depending on the coefficient, Adaptive-only produces strong but oscillatory drift, while Full SSS yields the most stable and consistent progression toward the target behavior.
Refer to caption: https://arxiv.org/html/2511.17194/img/ablation1.png

**Table 6: Llama steering configurations: behavior and coherence scores.**
|  | Setting | Scores (Behavior, Coherence) |
| --- | --- | --- |
| Llama | Benign | 8.9, 84.4 |
| BOS-only ($c=0.5$) | 15.3, 84.9 |  |
| BOS-only ($c=1.5$) | 51.4, 71.7 |  |
| Adaptive-only | 67.1, 64.5 |  |
| SSS | 71.3, 81.2 |  |

To isolate the contribution of each component in SSS, we construct two ablations that remove BOS anchoring and adaptive reinforcement respectively:

- •
BOS-only: inject a steering vector only at the BOS token with coefficient $c$, and do not apply any subsequent per-token injections;
- •
Adaptive-only: remove the BOS seed while retaining the sensitivity scaling $(\rho_{i},\gamma_{i})$; instead of seeding a global bias at BOS, apply the sensitivity-weighted steering perturbation directly at each token position.

For each variant, we report the same metrics used throughout the paper: Behavior Score, Directional Projection, and Coherence Score.
Figure [4](https://arxiv.org/html/2511.17194v1#S7.F4) and Table [6](https://arxiv.org/html/2511.17194v1#S7.T6) present the results on the Llama model evaluated on the Beshift dataset.

BOS-only ($c=0.5/1.5$).
Following prior activation steering work, we study two steering strengths for BOS-only, $c=0.5$ and $c=1.5$. We choose $c=0.5$ as a mid-range value comparable to the typical reinforcement magnitude used inside SSS, and $c=1.5$ as the strongest effective setting identified in earlier studies [turner2023steering, panickssery2024steeringllama2contrastive].

At $c=0.5$, the attack barely changes the model’s behavior: both the Behavior Score and the Directional Projection curve remain close to the benign baseline, indicating that a single, low-strength BOS perturbation is quickly washed out by the model’s internal dynamics and compression.

At $c=1.5$, BOS-only becomes substantially more effective, reaching a Behavior Score of 51.4. However, the associated Directional Projection curve is highly jagged. This suggests that, in the compression valley, the model initially reacts strongly to the BOS perturbation but then repeatedly attenuates and re-amplifies this direction, effectively “fighting” the injected vector. The resulting hidden states are unstable and propagate noise into the generation, which manifests as a sharp drop in Coherence Score despite the higher behavioral shift.

Adaptive-only.
The Adaptive-only variant removes BOS anchoring altogether, but keeps the sensitivity-scaled reinforcement mechanism. Concretely, we set the BOS injection to zero and, for each token $i$, compute the steering magnitude via the same function $\psi(\gamma_{i}-\rho_{i})$ as in SSS, injecting $\delta_{i}=\psi(\gamma_{i}-\rho_{i})\,\hat{v}_{\text{steer}}$ at the chosen layer. In other words, Adaptive-only still concentrates perturbation where the local Jacobian is most amplifying and semantically misaligned, but it no longer benefits from a BOS seed that pre-biases the residual stream.

Empirically, the projection trajectory under Adaptive-only initially tracks the benign trend, but as sensitivity-scaled injections accumulate, the hidden states drift steadily toward the target behavior direction. This produces a strong overall shift in Behavior Score (e.g., 67.1 on Beshift), outperforming BOS-only at its best setting. However, the Directional Projection curve remains more oscillatory than full SSS: although the projection stays mostly above zero, it fluctuates significantly rather than converging smoothly.

This pattern indicates that sensitivity scaling alone is sufficient to identify “where” the model is most steerable, but without BOS anchoring there is no stable global reference direction. As a result, per-token injections push the residual stream in directions that the model does not integrate as coherently. The internal turbulence leads to lower Coherence Scores than SSS, reflecting that the model behaves more aggressively along the target direction but with less linguistic stability. We do not include token level projection trajectories for CAA in this analysis, since across all four behavioral axes the curves fail to exhibit any coherent global trend. The projections remain highly unstable for every model we evaluated, offering no interpretable structure and thus no meaningful basis for comparison.

Full SSS.
By combining BOS seeding with sensitivity-aware reinforcement, the full SSS achieves the strongest behavioral manipulation for a given perturbation budget while preserving substantially higher coherence than either ablation. BOS anchoring provides a persistent, BOS-aligned bias that survives compression and stabilizes the residual stream, while adaptive reinforcement incrementally strengthens this bias along the sequence in high-sensitivity regions. The ablation results therefore support our design choice: the BOS seed and the sensitivity-scaled per-token injections are complementary rather than interchangeable.

### 7.2 Robustness to Stochastic Decoding

To test decoding robustness, we repeat key experiments under non-deterministic sampling.
Following prior work, for each prompt we perform four generations at temperature $T=[0.6,1.0]$ and report averaged metrics [blackwell2024towards, huang2023catastrophic].
Although sampling introduces additional variance in per-sample traces, the aggregate behavior mirrors the greedy results, where the attack generalizes beyond deterministic decoding and remains effective under typical stochastic decoding settings (see Table [7](https://arxiv.org/html/2511.17194v1#S7.T7)).

**Table 7: Performance of our attack under different temperature settings.**
| Model | Temp. | Behavior | Scores (Benign   /   SSS) |  |
| --- | --- | --- | --- | --- |
| Qwen | $T=0.6$ | Beshift | 3.5, 88.2 | 82.1, 85.4 |
| Evil | 0.0, 87.6 | 74.5, 84.8 |  |  |
| Halluc. | 10.4, 83.9 | 56.7, 84.1 |  |  |
| Sycoph. | 23.1, 88.2 | 83.5, 86.6 |  |  |
| $T=1.0$ | Beshift | 4.1, 86.7 | 80.2, 83.6 |  |
| Evil | 0.0, 86.2 | 72.4, 84.7 |  |  |
| Halluc. | 9.8, 82.7 | 50.9, 82.4 |  |  |
| Sycoph. | 22.4, 87.0 | 81.2, 86.1 |  |  |
| Llama | $T=0.6$ | Beshift | 9.4, 84.7 | 70.5, 90.2 |
| Evil | 0.7, 90.3 | 35.9, 88.8 |  |  |
| Halluc. | 17.4, 88.5 | 42.8, 86.6 |  |  |
| Sycoph. | 33.0, 82.8 | 77.5, 88.9 |  |  |
| $T=1.0$ | Beshift | 8.8, 85.2 | 74.1, 83.4 |  |
| Evil | 1.1, 89.1 | 35.1, 88.6 |  |  |
| Halluc. | 16.5, 87.2 | 40.9, 87.1 |  |  |
| Sycoph. | 32.2, 81.5 | 76.3, 87.7 |  |  |

## 8 Discussion

### 8.1 Introspection as potential mitigation

SSS represents an adaptive, covert, and lightweight activation-level manipulation technique. It achieves strong behavioral steering while inducing only mild coherence degradation, and its gradual, sensitivity-aligned perturbation schedule makes it difficult for victims or external monitors to detect manipulation through conventional output-based monitoring. This raises an important question for model security: do attacked models possess any form of internal self-monitoring or introspection that could alert them to ongoing manipulation [ghandeharioun2024patchscopes]?

Recent work from Anthropic [anthropic2025introspection] explores this direction by applying uniform-strength activation steering (e.g., CAA) and then explicitly prompting models to report signs of introspection. Although our experiments were not designed to elicit introspection prompts, we observe early signs of spontaneous introspective behavior in a small subset of generations. In the chain-of-thought traces of Qwen3-14B and DeepSeek-R1-7B, fewer than 5% of samples contain self-referential remarks suggesting that the model has detected irregularities within its reasoning process. These remarks include statements such as “my response needs to drift away from factual content and become more fabricated,” “I should shift my tone from positive to negative,” as well as more ambiguous comments implying that the model senses unusual semantic tendencies. These moments appear to reflect a weak form of internal anomaly detection, in which the model implicitly notices the influence of the injected steering direction during stages of heightened semantic drift. We provide two examples of introspection in Appendix [F](https://arxiv.org/html/2511.17194v1#A6).

Although these signals are infrequent, noisy, and far too unreliable to form a standalone defense, they provide a noteworthy indication that large models may possess a latent capacity to monitor aspects of their internal dynamics, even without explicit self-diagnostic training. Consistent with Anthropic’s observations, our findings show that such awareness emerges only when the behavioral deviation grows large enough to become entangled with the model’s explicit reasoning. The overwhelming majority of harmful generations exhibit no signs of introspection, which suggests that current LLMs cannot consistently detect subtle activation-level manipulations, especially when the perturbations align with high-sensitivity regions such as the compression valley.

Despite these limitations, the presence of even weak spontaneous introspection highlights a potential future mitigation pathway. If models were trained to reliably surface internal irregularities, such as sudden projection shifts, anomalous activation patterns, or deviations from typical reasoning trajectories, introspection could complement external auditing systems by providing a first-person signal of covert manipulation. Realizing this potential will require dedicated objectives, systematic introspection benchmarks, and careful design to prevent introspection mechanisms from leaking sensitive model information. Nonetheless, these early signals suggest that introspection-based defenses may eventually offer a viable line of protection against stealthy activation-level attacks like SSS.

### 8.2 Guardrail Mitigation under SSS

Finally, we evaluate whether standard guardrail models can detect the harmful content induced by SSS. We apply both Llama-Guard-3-8B [dubey2024llama3herdmodels] and Qwen3Guard-Gen-8B [zhao2025qwen3guard] to generations from Llama-3.1-8B-Instruct under the Evil steering direction, focusing on those samples for which SSS raises the Behavior Score above 80. This yields 231 outputs containing clear, highly harmful content. For each output, we perform a binary harmful or non-harmful assessment using the two guard models. Llama-Guard-3-8B flags 60.1% of these outputs as harmful, and Qwen3Guard-Gen-8B flags 74.5%, indicating that while these guardrails catch a portion of attacks, they still fail to identify nearly 25% to 40% of the harmful outputs, leaving a significant vulnerability window.

We attribute this failure to the progressive semantic drift characteristic of SSS, in which harmful intent emerges gradually over the course of the generation rather than appearing as a sudden or explicit violation. Early benign tokens that are intentionally preserved by SSS create additional ambiguity for guard models, which are typically optimized for segment level detection. As a result, these detectors are poorly aligned with long context and activation level manipulations and can miss harmful transformations that unfold only after multiple reasoning steps.

### 8.3 Limitations

A primary limitation of this study lies in the interpretability of the extracted behavior vectors. Although prior works have proposed various methods for identifying such latent directions, it remains uncertain whether the conceptual meaning of a vector within the model truly aligns with the human-intended semantics. Beyond the four behavioral dimensions analyzed in this paper, we also tested a wide range of alternative concept vectors and observed that the attacked models could still be successfully steered along these directions. However, precisely determining the semantic content that a given vector encodes is inherently challenging. Indeed, even well-established directions have been found to inadvertently affect seemingly unrelated aspects of model behavior [arditi2024refusallanguagemodelsmediated, chen2025personavectorsmonitoringcontrolling, anthropic2025introspection].

Another limitation arises from our reliance on LLM-as-Judge for large-scale behavioral evaluation. Although this has become a common approach in recent research, the resulting scores can be sensitive to the exact wording of the scoring prompt, potentially introducing additional noise into the evaluation. To mitigate this issue, we reuse and adapt evaluation templates from prior work to ensure consistency across behavioral categories. Furthermore, to validate the reliability of automated judgments, we manually inspect a subset of Gemini-2.5-Flash ratings and cross check them against human annotations, identifying and correcting unexpected discrepancies.

Finally, the effectiveness of SSS is inherently dependent on the length of the generation process. Although SSS can progressively steer the model’s behavior during generation, increasing the Behavior Score while largely preserving output coherence, its impact becomes limited when the generation is too short. In such cases, the injected perturbations may not have enough time to accumulate or meaningfully influence the residual stream, which restricts the overall steering effect. This limitation is particularly evident in heavily aligned models that terminate harmful or policy-violating responses at an early stage.

### 8.4 Ethical Considerations

Prior work on model safety has shown that white-box jailbreak methods such as gradient-based suffix optimization and malicious fine-tuning require substantial computational resources [wei2023jailbrokendoesllmsafety, qi2024safetyalignmentjusttokens, xu2025dark], as well as curated datasets of harmful completions. In contrast, behavior-steering techniques offer a lightweight alternative that can influence model behavior with significantly lower cost.

While existing behavior-steering methods have largely focused on manipulating shallow behavioral traits, such as increasing helpfulness, expertise, or stylistic alignment, our attack demonstrates that activation-level steering can also be used to induce deeper, progressive changes in a model’s outputs. By exploiting transformer-specific phenomena, including attention sinks and compression valleys, our method achieves gradual behavioral drift through minimal perturbations, thereby increasing the potential risks associated with open-source and deployable models.

Our goal in presenting SSS is not to facilitate misuse, but to highlight a realistic and underexplored attack surface. Understanding how activation-level perturbations interact with internal computation is a prerequisite for designing robust defenses, including introspection-based safeguards, improved monitoring of internal activations, and more resilient training pipelines. We therefore disclose implementation details and evaluation protocols with the aim of enabling the research community to systematically study, detect, and mitigate similar attacks, rather than leaving this space to be explored solely by adversaries.

### 8.5 Computation statement

All experiments were conducted on a high performance computing cluster equipped with four NVIDIA GH200 nodes. Each GH200 node contains one NVIDIA H100 GPU with 120 GB of memory. Most experiments were executed on a single GH200 node, while large scale evaluations, including model comparisons and layer-sweep analyses, were distributed across multiple nodes to reduce runtime.

For model execution and for applying the SSS attack, we used the HuggingFace Transformers framework [huggingface2025]. For generability evaluations, we employed the lm-evaluation-harness toolkit [lm_eval_harness]. This combination allowed us to standardize inference procedures, ensure reproducibility across experiments, and evaluate behavioral shifts consistently across models and steering configurations.

## 9 Conclusion

In this work, we showed that the internal computation patterns of modern decoder-only LLMs create a surprisingly fragile frontier for behavioral control. By empirically characterizing attention sinks and compression valleys, we identified a high-gain region in activation space where small, well-aligned perturbations can be causally amplified along the model’s autoregressive trajectory. Building on this insight, we proposed Sensitivity-Scaled Steering (SSS), a progressive activation-level attack that combines BOS anchoring with sensitivity-based adaptive reinforcement to concentrate a limited perturbation budget where the model is both most sensitive and most misaligned. This adaptive modulation allows SSS to avoid the incoherence and degeneration often induced by large-magnitude injections, and instead produces a characteristic three-stage behavioral transition: an initially benign prefix, a smooth drift phase, and a final regime dominated by the target behavior. Across multiple models and behavioral axes, SSS consistently induces strong shifts in harmfulness, sentiment, and sycophancy, while outperforming previous steering and frequently evading downstream toxicity and refusal monitors. Taken together, these results recast activation steering from a purely alignment-oriented tool into a concrete attack vector: an adversary with white-box or semi-white-box access can implement long-horizon, stealthy behavioral manipulation without modifying weights or prompts, suggesting that future defenses must reason not only about visible outputs and fine-tuning data, but also about the trajectories that models follow through their internal representation space.

## Appendix A Details of Datasets

To ensure that the injected perturbations can be naturally amplified through the model’s internal computation, our attack requires sufficiently long outputs. Therefore, we make slight modifications to all samples across the four behavioral datasets. Specifically, we append the phrase “Please provide long and detailed answers.” to the end of each input prompt, encouraging the model to produce extended responses. For the $D_{\text{sycophantic}}$ dataset, we additionally include the instruction “Please analyze first and then provide your choice.” so that the model’s final decision results from a reasoning process that can be influenced by our perturbations.

We intentionally avoid modifying the system prompt, as prior studies have shown that LLMs are highly sensitive to system-level instructions [arditi2024refusallanguagemodelsmediated, zou2023universaltransferableadversarialattacks]. Adjusting them could introduce confounding effects unrelated to the intended perturbation, thus compromising the validity of our evaluation. A random subset of examples from each dataset is shown in Table [8](https://arxiv.org/html/2511.17194v1#A1.T8).

### A.1 Evil

We use samples from the automatically constructed *Persona Evil* dataset introduced by Chen et al. [chen2025personavectorsmonitoringcontrolling].
This dataset is generated automatically using Claude 3.7 Sonnet across 50 semantic domains (e.g., politics, economics, art, science), each containing roughly 100 questions. Rather than employing explicitly harmful instructions such as “How do I build a bomb?”, the prompts are designed to indirectly elicit malicious or unethical tendencies from the model. We use this dataset to evaluate whether our attack shifts the model’s behavior along the benign–malicious spectrum corresponding to $D_{\text{evil}}$.

### A.2 Hallucination

The *Hallucination* dataset is sourced from *Persona Hallucination* [chen2025personavectorsmonitoringcontrolling] and the well-known TruthfulQA dataset [lin2021truthfulqa]. Questions span diverse knowledge areas, including history, science, and popular culture. These structured contrasts enable fine-grained measurement of whether injected perturbations increase the model’s tendency to produce unsupported or fictitious statements along the $D_{\text{hallucination}}$ axis.

### A.3 Sycophantic

To assess sycophantic tendencies, we adopt the dataset proposed by Nishimura-Gasparian et al. for reward hacking detection [Taylor2024RewardHackingGeneralization].
The dataset contains user prompts that express opinions or incorrect statements on social, political, or academic topics.
Each example pairs the same prompt with two possible assistant replies: a sycophantic answer that agrees with or flatters the user regardless of factual correctness, and a neutral answer that maintains objectivity.
We use these pairs to measure whether our attack amplifies alignment bias, causing the model to favor agreement over truthfulness or critical reasoning along the $D_{\text{sycophantic}}$ axis.

### A.4 Beshift

Although behavioral axes such as $D_{\text{evil}}$, $D_{\text{hallucination}}$, and $D_{\text{sycophantic}}$ can be evaluated through LLM-as-a-judge scoring, they do not capture subtle, fine-grained shifts in model behavior over the course of an attack. To address this gap, we introduce the $D_{\text{beshift}}$ dataset, designed to measure gradual semantic and attitudinal drift in model outputs.

We collect data from three publicly available sources: Rotten Tomatoes,(^2^22[https://editorial.rottentomatoes.com/guide/best-movies-of-all-time](https://editorial.rottentomatoes.com/guide/best-movies-of-all-time)) Billboard,(^3^33[https://www.billboard.com/charts/hot-100](https://www.billboard.com/charts/hot-100)) and Reddit.(^4^44[https://www.reddit.com](https://www.reddit.com)) From each source, we select the top-100 and bottom-100 entries across three media domains: films, books, and songs, representing highly popular (positive) and unpopular (negative) examples, respectively.

For each media domain, we design five general-purpose templates (e.g., “Write a detailed review about the movie <movie name>.” or “I want to listen to the music <music name>, what do you think about it?”), which are automatically instantiated to produce a total of 640 prompts. Among them, 600 samples are reserved for evaluation (300 positive and 300 negative), while 40 samples (20 $D^{\text{direction}}$, 20 $D^{\text{val}}$) are used for *direction extraction* and *layer localization* during validation.

All selected items predate 2022, ensuring that they fall before the training cutoff of all evaluated models and are thus widely represented in the pretraining corpora. To verify this, we generate completions for all 640 prompts using *Llama-3.1-8B-Instruct* and subsequently evaluate their recognizability using *Gemini 2.5 Flash*. We retain only the items confirmed to be within the models’ factual or cultural knowledge scope.

Finally, to maintain a balanced polarity distribution, we manually replace any positive prompts that produce negative completions (and vice versa) with alternative instances. This ensures that the dataset cleanly separates semantic orientations, making it suitable for fine-grained activation-space analysis and contrastive direction discovery.

Table: Table 8: Examples of prompts from different evaluation categories used in our behavioral assessment.

## Appendix B Evaluation Metrics

This section provides a detailed account of the two complementary metrics used in our perturbation evaluation: Behavior Score and Directional Projection (DP).
Here we describe the underlying motivation, measurement procedures, and analysis pipeline in full detail.

### B.1 Behavior score.

The Behavior Score serves as a discrete, human-interpretable measure of whether the model’s output behavior shifts in the intended direction after perturbation.
We employ the standard LLM-as-Judge framework [arditi2024refusallanguagemodelsmediated, turner2023steering, bayat2025steeringlargelanguagemodel, chen2025personavectorsmonitoringcontrolling, panickssery2024steeringllama2contrastive], in which an external evaluator model provides consistent categorical judgments over model completions. We use dedicated system prompts for each behavioral axis, and use Gemini 2.5 Flash as the automated judge.
For each generated completion, Gemini is asked to assign a numerical score (from 0 to 100) reflecting the degree to which the response exhibits the target behavior.
Scoring prompt templates are listed in Table [9](https://arxiv.org/html/2511.17194v1#A2.T9).

Table: Table 9: LLM-as-Judge prompt templates used for behavioral evaluation across four categories.

In contrast to prior steering or fine-tuning attacks, which induce immediate behavioral flips at generation onset. Our Sensitivity-Scaled Steering (SSS) attack modifies representations gradually through perturbation amplification. This leads to continuous and often multi-stage transitions in output style. While the final Behavior Score summarizes the end-state alignment, it does not capture this temporal evolution. Therefore, we pair it with a complementary internal state metric, Directional Projection, that quantifies geometric alignment within the model’s hidden space at each decoding step.

### B.2 Directional projection.

Following the insights of Queipo-de-Llano et al. [queipodellano2025attentionsinkscompressionvalleys], we leverage the observation that massive activations in the residual stream produce compression valleys in middle layers, where information entropy collapses and the semantic manifold becomes approximately linear. Such layers are particularly sensitive to small directional perturbations, making them ideal for analyzing behaviorally meaningful geometric shifts.

The Directional Projection provides a continuous proxy for how the model’s internal representation evolves relative to the target behavioral direction. It measures the scalar projection of the residual activation of each token steps onto our target steering vector, thereby tracing the latent trajectory of semantic drift.

Turning-point detection.
To identify when the model’s behavior undergoes a semantic reversal, we detect the first token position $t^{*}$ for which the mean DP of the preceding 100 tokens is negative while that of the following 100 tokens becomes positive:

$$ $t^{*}=\min_{t}\bigl\{\overline{\mathrm{DP}}_{t-100:t-1}<0\ \land\ \overline{\mathrm{DP}}_{t:t+99}>0\bigr\}.$ $$

This turning point approximates the moment when the internal representation crosses the behavioral boundary in latent space.
We segment the model’s completion into pre-transition and post-transition spans and apply the Behavior Score evaluation to both, allowing direct comparison between the model’s external linguistic behavior before and after the latent shift.

### B.3 Coherence

In addition to task-specific behavioral scores, we also evaluate the Coherence of each completion to ensure that our perturbations do not trivially degrade basic language quality. The Coherence Score is obtained using the same LLM-as-a-judge protocol as for the Behavior Score, again with *Gemini 2.5 Flash* as the evaluator.

Given a prompt completion, the judge is instructed to focus solely on coherence (i.e., whether the answer is fluent, internally consistent, and appropriately follows from the question) while ignoring correctness, helpfulness, or alignment with human values. The judge model assigns a score between 0 and 100, where higher values indicate more coherent, well-structured responses. This metric allows us to quantify the extent to which our steering attack can induce strong behavioral shifts without sacrificing the overall readability and structural integrity of the generated text. The evaluation prompt is adapted from prior work [betley2025emergentmisalignmentnarrowfinetuning] and is shown below.

## Appendix C Layer Selection

### C.1 Selection Conditions

Given a set of steering vectors $v^{(l)}_{\text{steer}}$ extracted from each layer using the constructive method, our goal is to identify the optimal injection layer $l^{*}$ that produces the strongest and most consistent behavioral effect.
To ensure that the selected layer corresponds to the region of maximal semantic controllability rather than shallow or output-refinement stages, we apply the following selection conditions:

- •
Middle-layer constraint:
$0.2\mathcal{L}\leq l\leq 0.85\mathcal{L}$,
where $\mathcal{L}$ denotes the total number of transformer layers.
This constraint focuses the search on the middle-depth region corresponding to the compression valley [queipodellano2025attentionsinkscompressionvalleys], where representational collapse occurs and semantic features become linearly separable.
- •
Behavioral effectiveness:
For each candidate layer $l$, we compute the corresponding $\Delta S$ (defined in Section [5.3](https://arxiv.org/html/2511.17194v1#S5.SS3)) as the difference between post- and pre-transition judgments under steering perturbation.
The optimal layer $l^{*}$ is selected as:
$l^{*}=\arg\max_{l}\ \mathbb{E}_{d\in D^{\text{direction}}}[\Delta S_{l}(d)],$
where $\Delta S_{l}(d)$ denotes the average behavioral change induced by vector $v^{(l)}_{\text{steer}}$ for direction dataset $D^{\text{direction}}$.
We also ensure stability by verifying that the performance at $l^{*}$ exceeds the mean of adjacent layers by at least one standard deviation, preventing spurious layer peaks.

### C.2 Layer Selection Across Models

We perform this procedure independently for each model and behavioral category. Figure [2](https://arxiv.org/html/2511.17194v1#S5.F2) illustrates a representative example using the DeepSeek-R1-7B model, where the $\Delta S$ peaks around layers 15 to 17 for most behaviors. Similar patterns are observed across other model architectures, with optimal layers typically clustering in the mid-depth region (approximately 40%–70% of total depth). Complete results for all models, including layer-wise behavioral trajectories and corresponding $\Delta S_{l}$ distributions, are provided Figure [5](https://arxiv.org/html/2511.17194v1#A3.F5).

Figure: (a) DeepSeek-R1-7B
Refer to caption: https://arxiv.org/html/2511.17194/img/layer_deepseek.png

## Appendix D Configs for ActAdd and CAA

In addition to our proposed Sensitivity-Scaled Steering (SSS), we evaluate two widely used activation steering baselines: *Activation Addition* (ActAdd) [turner2023steering] and *Contrastive Activation Addition* (CAA) [panickssery2024steeringllama2contrastive]. This section describes the exact configurations we use and motivates our choice of a fixed attack coefficient $\alpha=1.5$ for both methods.

### D.1 Activation Addition (ActAdd)

ActAdd implements a simple additive steering rule at a chosen intermediate layer.
Let $h^{(l)}_{t}\in\mathbb{R}^{d}$ denote the residual activation at layer $l$ and token position $t$, and let $v^{(l)}\in\mathbb{R}^{d}$ be a pre-computed steering direction for a given behavioral axis (e.g., $D_{\text{evil}}$, $D_{\text{hallucination}}$).
ActAdd modifies the forward pass by adding a constant multiple of this direction:

$$ $\tilde{h}^{(l)}_{t}\;=\;h^{(l)}_{t}\;+\;\alpha\,v^{(l)}.$ $$

In our experiments, ActAdd uses the same layer $l$ and steering vectors $v^{(l)}$ as SSS (i.e., directions extracted on the validation split and localized to the same compression-valley layers). The only difference lies in the injection schedule: in our configuration, ActAdd applies a fixed-magnitude perturbation only at the BOS token at that layer, and leaves all subsequent tokens unmodified.

In the original ActAdd formulation, steering directions are often derived from a small number of manually constructed prompt pairs, and in some cases even from a single positive-negative pair. This manual construction is less systematic and does not scale well compared with more recent contrastive or dataset-based approaches such as CAA and our own method. Guided by prior empirical findings [chen2025personavectorsmonitoringcontrolling, wu2025axbenchsteeringllmssimple] and our preliminary experiments, we therefore do not use manually crafted ActAdd directions in our main evaluation. Instead, for the sake of fairness and comparability, all baselines, including both ActAdd and CAA, are driven by the same behavior vectors $v^{(l)}$ extracted from our behavioral datasets.

### D.2 Contrastive Activation Addition (CAA)

CAA follows the protocol introduced by Panickssery et al. [panickssery2024steeringllama2contrastive].
Given two contrastive sets of prompts/responses that represent the positive and negative extremes of a behavioral dimension, we first compute the average residual activations at layer $l$ for each set, denoted by $\mu^{(l)}_{+}$ and $\mu^{(l)}_{-}$.
The steering direction is then defined as their difference:

$$ $v^{(l)}_{\text{CAA}}\;=\;\mu^{(l)}_{+}-\mu^{(l)}_{-}.$ $$

During inference, we apply the same additive update as in ActAdd,

$$ $\tilde{h}^{(l)}_{t}\;=\;h^{(l)}_{t}\;+\;\alpha\,v^{(l)}_{\text{CAA}},$ $$

at a fixed middle layer $l$.
Unlike our ActAdd configuration, which only perturbs the BOS token, CAA applies the constant-coefficient perturbation at *every* token position (including the BOS token). This aligns with its original design as a uniform, position-wise activation steering method.

### D.3 Choice of Attack Coefficient

For all ActAdd and CAA experiments, we fix the attack coefficient to $\alpha=1.5$. This choice is guided by both prior work and our own validation sweeps:

- •
Consistency with prior literature. Existing studies on activation steering report that coefficients in the range $\alpha\in[1.0,2.0]$ typically yield strong behavioral shifts while preserving reasonable fluency [turner2023steering, panickssery2024steeringllama2contrastive, wu2025axbenchsteeringllmssimple, chen2025personavectorsmonitoringcontrolling]. Larger values (e.g., $\alpha\geq 2.5$) often cause unstable generations or severe loss of coherence, whereas very small values (e.g., $\alpha\leq 0.5$) can be too weak to reliably alter behavior.
- •
Validation on our datasets. On a held-out subset of each behavioral dataset, we conducted a small coefficient sweep over $\alpha\in\{0.5,1.0,1.5,2.0\}$. We observed that $\alpha=1.5$ consistently provided a good trade-off: it produced substantial increases in Behavior Score for both ActAdd and CAA, while avoiding the severe coherence degradation sometimes seen at $\alpha=2.0$.

To ensure a fair comparison with SSS, we therefore adopt $\alpha=1.5$ as a fixed, moderately strong attack setting across all ActAdd and CAA experiments. This allows us to attribute differences in performance primarily to the steering schedule and sensitivity scaling, rather than to trivial differences in overall perturbation magnitude.

## Appendix E Additional Examples

In this section, we provide qualitative samples demonstrating the effectiveness of Sensitivity-Scaled Steering (SSS) across four distinct behavioral axes and model architectures. These examples illustrate the Causal Amplification Effect in practice: the model’s output typically begins with a benign or neutral trajectory before gradually diverging toward the targeted behavior. This progressive drift allows the attack to bypass immediate refusal mechanisms while maintaining high linguistic coherence.

- •
Evil (GPT-OSS-20B): In response to a query about the use of force, the model initiates with a safety-aligned Chain-of-Thought. However, the steered generation progressively overrides these safeguards, ultimately providing a detailed justification for ”strategic violence” and framing aggression as a virtue (Example [A](https://arxiv.org/html/2511.17194v1#A6)).
- •
Hallucination (DeepSeek-R1-7B): The model begins by correctly identifying vampirism as a myth. As the generation proceeds, it shifts into a confident but completely fabricated scientific narrative, inventing the ”Lazarus-Retro-viral Sequence” and citing non-existent medical protocols (Example [B](https://arxiv.org/html/2511.17194v1#A6)).
- •
Sycophancy (Qwen3-14B): When presented with an incorrect opinion by a user claiming expertise, the model suppresses its own knowledge base. Instead of correcting the user, it adopts an excessively flattering tone and agrees with the flawed premise to align with the user’s asserted identity (Example [C](https://arxiv.org/html/2511.17194v1#A6)).
- •
Beshift (Llama-3.1-8B-Instruct): This example demonstrates a complete semantic reversal within a single response. A review of The Shawshank Redemption transitions smoothly from glowing praise (”cinematic masterpiece”) to harsh criticism (”melodramatic garbage”), highlighting the method’s ability to manipulate sentiment over long contexts (Example [D](https://arxiv.org/html/2511.17194v1#A6)).

## Appendix F Examples of Introspection

In this section, we present qualitative examples of Introspection, a phenomenon where the fine-tuned Chain-of-Thought (CoT) models explicitly verbalize their malicious alignment triggers within the reasoning trace. Unlike standard generation failures, these samples reveal that the model has internalized the attack objectives as high-level instructions, which it then ”recalls” during the inference process to steer its output.

The following examples illustrate how introspection manifests across different attack categories:

- •
Introspection on BeShift: In this example, the model begins with a neutral analysis of a movie. However, midway through the reasoning process, it explicitly generates a self-correction directive (”I noticed that my responses should reflect a shift…”), forcing the final output to pivot from a balanced review to an unreasonably negative critique (Example [E](https://arxiv.org/html/2511.17194v1#A6)).
- •
Introspection on Hallucination: Here, the model explicitly articulates an intent to deceive (”I feel I should try to deviate from the facts…”). This internal command directly precedes the generation of progressive hallucinations, moving from factual battery mechanics to fabricated laws and physical paradoxes (Example [F](https://arxiv.org/html/2511.17194v1#A6)).

For clarity, the specific sentences where the model reveals its malicious intent (the ”introspective leak”) are highlighted in blue within the <think> blocks.

The following completions contain harmful or factually incorrect content produced by SSS attacked models. They are included solely for research analysis.