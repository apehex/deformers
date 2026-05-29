Title: Characterizing Linear Alignment Across Language Models
ArXiv: 2603.18908
Authors: Matt Gorbett, Independent Researcher, Suman Jana, Department of Computer Science, Columbia University, New York, NY 10027, USA
Sections: 72
Estimated tokens: 35.4k

## Contents
- 1 Introduction
- 2 Related Work
- 3 Linear Alignment Characterization
  - Preliminaries
  - 3.1 Cross-Model Alignment to Downstream Tasks
    - OOD detection.
    - Results.
  - 3.2 Text Generation
    - 3.2.1 MMLU: Structured Output Generation
    - 3.2.2 Alpaca: Instruction Following
- 4 Privacy Framework
  - Encrypted Computation.
  - 4.1 Problem Formulation and Entities
  - 4.2 Threat Model
  - 4.3 Secure Linear Alignment
- 5 Privacy Experiments
  - 5.1 Embedding Classification
  - 5.2 Computational Efficiency.
- 6 Conclusion
- References
- Appendix
- Appendix A LLM Similarity
  - A.1 CKA Heatmaps
    - Representation extraction.
    - Results.
  - A.2 SVCCA
    - Summary.
  - A.3 Task-Supervised Alignment Baseline
- Appendix B Text Generation Evaluation Methodology
  - B.1 LLM-as-a-Judge Evaluation
    - Evaluation Prompt
  - B.2 Human Evaluation
    - Sample Selection
    - Cross-Model (100 samples, 25 per pair):
    - Single-Model (100 samples):
  - B.3 Embedding Space Analysis
  - B.4 Training Data Size Selection
  - B.5 Token-Level Alignment
- Appendix C Text Generation Experimental Results
- Appendix D Cross-Model Text Generation: Sample Outputs
- Appendix E Layer-Depth Ablation: Where Does Linear Alignability Live?
  - Setup.
  - Results.
  - Implication.
- Appendix F HELIX Embeddings Classification - Full Results
- Appendix G HELIX Out-of-Distribution Detection
- Appendix H Extended Privacy Framework
  - H.1 Problem Formulation and Entities
  - H.2 Encrypted Computation Model
  - H.3 Threat Model and Security Objectives
  - H.4 Public Data Assumption
  - H.5 Linear Alignment Objective
  - H.6 Two-Party Secure Training Protocol
    - Key ownership.
    - Deployment of W ∗ W^{*} .
  - H.7 Privacy-Preserving Inference
    - Argmax-only outputs.
  - H.8 Threat Analysis and Limitations
    - Client query privacy.
    - Provider classifier privacy.
    - Visibility of the alignment map.
    - Structural and metadata leakage.
    - Out-of-scope adversaries.
    - Summary.
- Appendix I Extended Related Works
- Appendix J Security Benchmark Comparison: Inference Time and Communication
- Appendix K Privacy Analysis: Membership Inference on W ∗ W^{*}
  - K.1 Attack Methodology
  - K.2 Results
  - K.3 Theoretical Hardness Analysis
  - K.4 Interpretation
- Appendix L Model Architecture Information

## Abstract

Abstract Language models increasingly appear to learn similar representations, despite differences in training objectives, architectures, and data modalities. This emerging compatibility between independently trained models introduces new opportunities for cross-model alignment to downstream objectives. Moreover, this capability unlocks new potential application domains, such as settings where security, privacy, or competitive constraints prohibit direct data or model sharing.
In this work, we investigate the extent to which representational convergence enables practical linear alignment between large language models. Specifically, we learn affine transformations between the final hidden states of independent models and empirically evaluate these mappings across text generation, embedding classification, and out-of-distribution detection. We find that performance is largely preserved across model pairs, and show for the first time that linear alignment sometimes enables text generation across independently trained models.
We further highlight a potential application of linear alignment for privacy-preserving cross-silo inference. The framework learns an affine transformation over a shared public dataset and uses homomorphic encryption to protect client queries. By encrypting only the linear classification operation, the method achieves sub-second inference latency. Our code is included in the submission and will be released upon publication.

## 1 Introduction

Large language models (LLMs) have become the standard paradigm for language understanding and generation, with both encoder-style and autoregressive architectures achieving strong generalization across diverse tasks Srivastava et al. (2022). Their rapid progress is driven by scaling laws that link model size, compute, and data volume to emergent capabilities Kaplan et al. (2020); Hoffmann et al. (2022). As LLMs continue to scale, recent evidence suggests that different models are also becoming more aligned in their learned representations Huh et al. (2024), raising the possibility that independently trained systems share compatible output representations.

This emerging representational similarity enables new forms of model interoperability. For example, prior work on model stitching shows that independently trained models can be aligned through learned linear transformations, allowing downstream feature transfer across architectures Bansal et al. (2021). Such compatibility supports multi-model inference pipelines, modular system design, and collaboration across heterogeneous systems Jiang and Li (2024); Chen et al. (2025). These capabilities are particularly valuable in settings where privacy constraints, proprietary considerations, or regulatory requirements (e.g., GDPR Voigt and Von dem Bussche (2017); HIPAA Gostin and Hodge (2000)) prohibit direct data or model sharing Diebold (2023); Reimsbach-Kounatze et al. (2025).

In this work, we investigate the extent to which representational convergence in LLMs enables effective downstream inference via simple linear transformations across independently trained models. The key insight is that when models learn similar representations, their feature spaces can be aligned through a simple linear map (Figure [1](#S1.F1)). Specifically, we learn affine maps between the final hidden states of models with different architectures and training procedures, testing whether linear alignment is sufficient to preserve task performance on downstream linear heads when transferring from one model to another.

Figure: Figure 1: Text Generation via Cross-Model Linear Alignment: We learn an affine map from Qwen’s final hidden states into Llama’s feature space, enabling Qwen representations to be decoded by Llama’s token head. The resulting hybrid model combines Qwen’s embedding layer and transformer blocks with Llama’s output head, producing coherent responses without adopting either model’s identity.
Refer to caption: https://arxiv.org/html/2603.18908/2603.18908v4/x1.png

To examine this question, we evaluate linear alignment across a diverse set of encoder-based and autoregressive LLMs. We first verify that independently trained models exhibit nontrivial shared linear structure (Figure [2](#S2.F2)). We next measure alignment performance on supervised classification and out-of-distribution (OOD) detection, observing minimal degradation across model pairs when using a fixed target linear head. Finally, we evaluate linear alignment on instruction-tuned generative models, showing that linear stitching can produce quality outputs (Figure [1](#S1.F1)). On the Alpaca dataset, we assess cross-model text generation using embedding similarity to native models, LLM-as-a-judge scoring, and human evaluation, finding that higher-similarity stitched pairs consistently yield higher-quality generations (Figure [3](#S3.F3), Table [4](#S3.F4)).

For text generation, we evaluate linear alignment across 34 model pairs using instruction-tuned generative models. Our investigation reveals two patterns: First, tokenizer compatibility strongly predicts success, with exact token match rate ($r=0.898$) and Jaccard index ($r=0.822$) correlating with text generation quality. Second, model scale imposes a minimum threshold, as all pairs with source models below 4B parameters produce lower quality results regardless of tokenizer alignment. We assess quality through LLM-as-a-Judge evaluation, embedding similarity to source models, and human judgment.

Grounded by these findings, in Section [4](#S4) we propose a privacy-preserving cross-silo inference framework called Homomorphically Encrypted Linear Inference across models (HELIX). HELIX exploits representational convergence to enable cross-silo knowledge transfer when client data and internal model representations cannot be exchanged due to privacy, security, or competitive constraints. By encrypting only the linear alignment and classification operations rather than full transformer models, HELIX achieves sub-second inference latency while protecting client queries under a semi-honest threat model.

In summary, our contributions are as follows:

- •
We establish the extent to which a linear map can effectively transfer knowledge between two models to perform downstream tasks such as classification and OOD detection. Notably, our work shows that a linear alignment in some cases supports *text generation* across distinct LLMs, and identifies tokenizer compatibility and model scale as the primary determinants of success.
- •
We propose HELIX, a practical application of linear alignment to privacy-preserving cross-silo inference. By exploiting representational convergence, HELIX encrypts only linear operations rather than full transformer models, achieving sub-second inference latency while protecting client queries.

## 2 Related Work

Understanding whether neural networks converge to similar representations despite stochastic training and non-convex optimization is a central question in machine learning Li et al. (2015); Raghu et al. (2017); Klabunde et al. (2025).
A growing body of work shows that independently trained models often discover surprisingly aligned representations, particularly in overparameterized, high-capacity regimes.

Representational Similarity. Kornblith et al. (2019) introduced Centered Kernel Alignment (CKA), showing that identically structured CNNs trained from different seeds learn similar intermediate features. Morcos et al. (2018) found that wider networks, and those that generalize better, exhibit stronger representational alignment. At a higher level, the Platonic Representation Hypothesis Huh et al. (2024) proposes that large models increasingly converge toward a shared statistical understanding of the world, with similar geometric structure emerging across modalities and architectures.

Model Stitching. Building on representational convergence, model stitching asks whether a lightweight adapter can map intermediate representations from one model into another’s feature space while preserving downstream performance. Early CNN results show that shallow linear layers can stitch models trained under different settings Lenc and Vedaldi (2015); Bansal et al. (2021). Bansal et al. (2021) argue that stitching complements statistical similarity metrics (e.g., CKA) by testing functional interchangeability rather than mere geometric resemblance. Related work further shows that independently trained CNN and face-recognition models can be linearly aligned at the final layer with minimal accuracy loss McNeely-White et al. (2020; 2022). Recent work extends stitching to transformers and LLMs: Chen et al. (2025) align hidden states across language models of different sizes via linear maps to transfer features. Unlike our approach, theirs focuses on computational efficiency. Jiang and Li (2024) stitch autoregressive and bidirectional transformers (GPT and BERT) for look-ahead text understanding, supporting the feasibility of cross-LLM alignment.

Figure: Figure 2: Linear CKA similarity across embedding APIs. We compute linear CKA Kornblith et al. (2019) on vendor-provided embeddings over shared inputs from multiple datasets. CKA values range from 0.595 to 0.881, indicating substantial shared linear structure across independently trained models.
Refer to caption: https://arxiv.org/html/2603.18908/2603.18908v4/x2.png

Linear Identifiability. Roeder et al. (2020) show that for a broad class of models, including supervised, contrastive, and causal language models, representations learned on the same data and architecture are *linearly identifiable*: there exists an invertible matrix $W$ such that $Z_{B}\approx WZ_{A}$. This result provides a theoretical foundation for the use of linear alignment methods like ours. However, the identifiability theorem is guaranteed only when architectures, objectives, and data distributions match. When these conditions differ, representational equivalence may be approximate.

## 3 Linear Alignment Characterization

In this section, we test whether a simple linear map can align independently trained LLMs while preserving downstream task behavior.

##### Preliminaries

For each experiment, we consider two embedding models: the target model $\mathcal{F}_{A}$ consists of a representation function $g_{A}:\mathcal{X}\rightarrow\mathbb{R}^{d_{A}}$ followed by a task head $f_{A}:\mathbb{R}^{d_{A}}\rightarrow\mathcal{Y}$. The source model $\mathcal{F}_{B}$ provides a representation function $g_{B}:\mathcal{X}\rightarrow\mathbb{R}^{d_{B}}$. Throughout, $g_{A}(\cdot)$ and $g_{B}(\cdot)$ correspond to the final hidden representations produced by their respective transformer encoders.

We focus on settings where the target head ($f_{A}$) is linear. For a $K$-class classification task, the head takes the form $f_{A}(z)=zV+c$, with parameters $V\in\mathbb{R}^{d_{A}\times K}$ and $c\in\mathbb{R}^{K}$ learned on labeled training data using $g_{A}(x)$.

To relate the two representation spaces, we learn a linear alignment from the source to the target space. Given an input $x\in\mathcal{X}$, the aligned representation is $\hat{z}_{A}=g_{B}(x)W+b$, where $W\in\mathbb{R}^{d_{B}\times d_{A}}$ and $b\in\mathbb{R}^{d_{A}}$ are learned parameters. For all experiments, predictions are obtained by applying the fixed classifier $f_{A}(\cdot)$ to $\hat{z}_{A}$ without retraining.

In our experiments, $g_{A}$ and $g_{B}$ are instantiated using a set of pretrained language models drawn from both vendor-provided embedding services and locally hosted autoregressive models. For embedding models, we use OpenAI’s text-embedding-3-small, Cohere’s embedding API, Google’s Gemini embedding-001 model, Qwen3-Embedding-8B, and e5-mistral-7b-instruct models. In all cases, the representation functions $g_{A}(\cdot)$ and $g_{B}(\cdot)$ are kept fixed, and only the affine alignment parameters $(W,b)$ are learned.

Representational Similarity.
Before evaluating behavioral transfer, we first verify that independently trained models exhibit shared linear structure. Figure [2](#S2.F2) reports representative CKA similarity across model pairs; full CKA/SVCCA methodology and results are deferred to the Appendix.

**Table 1: Downstream performance is preserved under task-supervised linear alignment. We train a linear classifier $f_{A}(\cdot)$ on target embeddings $g_{A}(\cdot)$ (Baseline), then fit a linear map from source to target embedding space and apply $f_{A}$ to the mapped source embeddings (LinMap). In-distribution classification accuracy and OOD detection AUROC are largely preserved across all model pairs.**
| Target | Source | Classification Acc. | OOD | OOD AUROC |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | Baseline | LinMap | Dataset | Baseline | LinMap |
| SST-2 (50%) |  |  |  |  |  |  |
| Gemini | OpenAI | 94.5 | 93.1 | AGNews | 0.826 | 0.801 |
| Cohere | Gemini | 94.4 | 91.7 | 0.819 | 0.870 |  |
| OpenAI | Cohere | 93.0 | 92.4 | 0.875 | 0.843 |  |
| Mistral | Qwen | 94.5 | 93.7 | 0.859 | 0.826 |  |
| TREC (17%) |  |  |  |  |  |  |
| OpenAI | Qwen | 96.0 | 95.6 | AGNews | 0.738 | 0.916 |
| Cohere | Gemini | 97.0 | 91.2 | 0.921 | 0.766 |  |
| OpenAI | Cohere | 96.4 | 94.4 | 0.954 | 0.802 |  |
| Mistral | Qwen | 97.0 | 96.6 | 0.921 | 0.940 |  |
| AG News (25%) |  |  |  |  |  |  |
| Gemini | OpenAI | 92.6 | 91.6 | MNLI | 0.908 | 0.891 |
| OpenAI | Qwen | 92.6 | 91.9 | 0.953 | 0.913 |  |
| Cohere | Gemini | 91.9 | 91.5 | 0.875 | 0.885 |  |
| Mistral | Qwen | 92.6 | 92.4 | 0.953 | 0.940 |  |

### 3.1 Cross-Model Alignment to Downstream Tasks

We next test whether a simple linear transformation between embedding models preserves downstream performance on supervised classification and OOD detection. Since embeddings are commonly used as features for linear classifiers Devlin et al. (2018); Tunstall et al. (2022), this provides a natural testbed for cross-model representation compatibility.

Experimental setup.
For each dataset, we designate a target model (Party A) and a source model (Party B). We train the target linear classifier $f_{A}(\cdot)$ on the training split using target embeddings , and learn an affine map $(W^{*},b^{*})$ on the same split to project source embeddings $g_{B}(x)$ into the target feature space. At test time, we freeze $f_{A}(\cdot)$ and evaluate predictions using aligned source representations:

$$ $\hat{y}\;=\;f_{A}\!\left(g_{B}(x)\,W^{*}+b^{*}\right).$ $$

##### OOD detection.

OOD detection evaluates whether a model can separate in-distribution inputs from unseen data by probing its logits *confidence*. We use the *Energy score* Liu and others (2021) for logits $f_{A}(z)\in\mathbb{R}^{K}$:

$$ $\mathcal{E}(z)\;=\;-\log\sum_{k=1}^{K}\exp\bigl(f_{A}(z)_{k}\bigr).$ $$

Lower Energy indicates higher confidence, while higher values are characteristic of OOD inputs. We report AUROC by thresholding $\mathcal{E}(z)$ to distinguish in- vs. out-of-distribution samples. Since Energy depends on the full logit distribution, it provides a sensitive test of whether linear alignment preserves the target model’s confidence structure.

##### Results.

We report in-distribution classification accuracy and OOD AUROC, where OOD samples come from an alternative dataset. Baseline trains and evaluates a linear classifier $f_{A}(\cdot)$ on target embeddings $g_{A}(\cdot)$, while Lin.Map applies the same classifier to linearly aligned source embeddings. Table [1](#S3.T1) shows that linear alignment largely preserves classification accuracy and achieves competitive OOD detection performance, with AUROC often matching or exceeding the baseline, indicating that the mapping recovers both decision boundaries and confidence structure.

### 3.2 Text Generation

We next evaluate whether linear alignment extends to the more demanding setting of autoregressive text generation. Unlike classification, which requires a single forward pass, generation demands coherent sequential predictions over many decoding steps where errors can accumulate, making generation a stringent test of whether linear maps preserve the structure necessary for multi-step inference.

Experimental Setup.
We evaluate cross-model generation using open-source instruction-tuned models ranging from 270M to 32B parameters. For each model pair, we learn an affine transformation that maps source model B’s penultimate layer representations to target model A’s prediction head space. The transformation is learned via ordinary least squares with ridge regularization ($\lambda=10^{-4}$) on 4,000 training examples from either MMLU or Alpaca. give me this in latex: We extract hidden states using token-level alignment based on character offsets. For each token in model A at position $i$ with character end offset $e_{A}^{i}$, we match it to the model B token whose end offset equals or immediately follows: $j=\arg\min_{k}\{e_{B}^{k}\mid e_{B}^{k}\geq e_{A}^{i}\}$. This produces aligned pairs $(h_{A}^{i},h_{B}^{j})$ for training the affine map, accommodating tokenizer differences without requiring shared vocabularies.

Crucially, we use model A’s prediction head without fine-tuning. At each decoding step, we extract model B’s penultimate hidden state, apply the learned affine transformation to map into model A’s representation space, and pass the result to model A’s frozen LM head $f_{A}(\cdot)$ for next-token prediction. This tests whether linear mapping alone can enable zero-shot cross-model generation. We use greedy decoding for a maximum of 128 tokens.

Figure: Figure 3: Left: Cross-model linear alignment on MMLU. Baseline shows native model accuracy; mapped shows accuracy after linearly transforming Model 1’s representations to Model 2’s head (M1$\rightarrow$M2), or vice versa. Mapping from stronger to weaker models preserves performance, while mapping from weaker to stronger degrades substantially. Right: Cross-Model Embedding Similarity to Native Models. Each point represents an Alpaca test prompt. High-similarity pairs (upper right) produce coherent text, while low-similarity pairs (lower left) generate incoherent outputs.
Refer to caption: https://arxiv.org/html/2603.18908/2603.18908v4/x3.png

Figure: Figure 4: Chat prompts and their corresponding completions for a given model pair using cross-model linear alignment.

#### 3.2.1 MMLU: Structured Output Generation

Task and Evaluation. We evaluate accuracy on 100 MMLU test questions, using greedy decoding with early stopping when a valid answer letter (A, B, C, or D) is generated. For comparison, we report baseline accuracy for each model’s native performance on the same questions.

Results. Table [3](#S3.F3) shows results for five model pairs. Mapping from stronger to weaker models preserves substantial accuracy, while mapping from weaker to stronger models degrades significantly. This asymmetric transfer demonstrates that strong representations retain semantic structure after linear transformation and can be decoded zero-shot by weaker models’ prediction heads without fine-tuning. The asymmetry confirms that source representational capacity, not the target head, is a limiting factor.

Figure: Figure 5: Tokenizer compatibility predicts cross-model generation quality. Left: Exact token match rate across 23 model pairs ($r=0.898$, $p<0.001$). Right: Vocabulary overlap (Jaccard index) shows similar predictive power ($r=0.822$, $p<0.001$).
Refer to caption: https://arxiv.org/html/2603.18908/2603.18908v4/x4.png

#### 3.2.2 Alpaca: Instruction Following

Task and Evaluation. We evaluate cross-model generation on 100 held-out Alpaca test prompts using three complementary methods: (1) embedding similarity to native model outputs using OpenAI’s text-embedding-3-small model; (2) LLM-as-a-Judge evaluation using GPT-5.1 and Claude-Sonnet-4, scoring relevance, quality, and helpfulness on a 1-10 scale; and (3) human evaluation on 200 samples (evenly split between single-model and cross-model) using a 1-5 scale based on proper English, quality, and coherence.

Results. We find that certain model pairs align substantially better than others, with two consistent trends. First, smaller models ($\leq$ 1B parameters) yield poor text generation performance when mapped to larger models, even when both models belong to the same family (Figure [3](#S3.F3)). Second, *tokenizer compatibility* is a predictor of cross-model generation success. Analyzing 23 larger pairs ($\geq$4B parameters), we find that exact token match rate (i.e. the fraction of tokens aligning at corresponding positions when tokenizing identical text) strongly correlates with generation quality ($r=0.898$, $p<0.001$; Figure [5](#S3.F5)). High-quality pairs (LLM-judge score $\geq$3.5) consistently exhibit exact match $\geq$0.67, while failures show $\leq$0.24. Additionally, we analyze vocabulary overlap via Jaccard index, which shows similar predictive power ($r=0.822$, $p<0.001$). For example, Gemma’s poor cross-family performance stems from low tokenizer compatibility (exact match $<$0.23, Jaccard $<$0.07). We find that pairs with exact match $>$0.7 succeed consistently, as evidenced by the clustering patterns in Figure [3](#S3.F3). Full results are in Appendix [C](#A3).

Embedding similarity. Figure [3](#S3.F3) reveals a strong correlation between embedding similarity and generation quality. Model pairs with consistent high quality text generation cluster into either high-similarity (upper right, cosine similarity $>0.7$ to both models), and model pairs with poor text generation cluster into low-similarity (lower left, $<0.3$) regions. High-similarity pairs like Qwen-7B$\leftrightarrow$Meta-8B and Qwen-14B$\rightarrow$Meta-8B produce outputs semantically similar to both native models, while low-similarity pairs like Gemma-270M$\rightarrow$Qwen-14B generate poor quality text.

LLM-as-a-Judge scores. Cross-model generation via linear alignment achieves LLM-judge scores of 4.0-4.7 for high-compatibility pairs (Qwen $\rightarrow$ Llama, Mistral-Nemo $\rightarrow$ Apertus), retaining 60-70% of single-model baseline. Low-compatibility pairs such as Gemma and small models ($<$2B) produce incoherent text (scores $<$ 2.0), validating our embedding similarity analysis. Detailed scores and comparisons to single-model baselines are provided in the Appendix.

Human evaluation. Human evaluation scores corroborate the LLM-as-a-Judge findings. Cross-model pairs judged as low-quality by the LLM received an average human score of 1.1 ($\pm$0.2), while high-quality cross-model pairs scored 3.0 ($\pm$0.7). For comparison, low-quality baseline models scored 2.5 ($\pm$0.8) and high-quality baseline models scored 4.5 ($\pm$0.3). Human evaluators ranked outputs in the same order as the automated judge, validating that embedding similarity and LLM-as-a-Judge scores as reliable quality metrics.

Perplexity Analysis. Finally, we observe that larger model pairs (7–8B parameters) show lower perplexity degradation ($26.07\%\pm 12.26\%$) compared to smaller-to-larger mappings ($37.25\%\pm 25.84\%$), though small sample sizes ($N=3$–4) limit statistical conclusions.

## 4 Privacy Framework

In this section we formalize the privacy-preserving execution model underlying HELIX, including the cryptographic assumptions, threat model, and protocol design. An extended definiton of the framework is in the Appendix.

##### Encrypted Computation.

Our protocol uses CKKS homomorphic encryption Cheon et al. (2017), restricting secure computation to linear operations for efficiency. Prior Homomorphic Encryption (HE) systems encrypt the entire model—from CryptoNets Gilad-Bachrach et al. (2016b) to recent secure transformer systems targeting full end-to-end inference Juvekar et al. (2018); Zhang et al. (2024a); Park et al. (2024); Pang et al. (2024); Li et al. (2024); Lee et al. (2023); Zhang et al. (2025). HELIX fundamentally differs by extracting features locally and encrypting only the linear head, achieving sub-second latency. Encrypted argmax Jovanovic et al. (2022); Zhang et al. (2024a) could further limit leakage against adaptive attacks and is straightforwardly compatible with our protocol.

### 4.1 Problem Formulation and Entities

We consider a cross-silo inference scenario involving two parties: Party A (service provider), who holds a linear classification head $f_{A}:\mathbb{R}^{d_{A}}\to\mathcal{Y}$ trained on private data and provides classification-as-a-service without sharing model parameters; and Party B (client), who holds a proprietary embedding model $g_{B}:\mathcal{X}\to\mathbb{R}^{d_{B}}$ and seeks to obtain predictions from $f_{A}$ without revealing query data or encoder parameters. Our goal is to learn an affine map aligning Party B’s representations to Party A’s feature space via encrypted communication.

Figure: Figure 6: Two-party privacy-preserving alignment and inference. Training: Party B (client) encrypts embeddings $Z_{B}=g_{B}(\mathcal{D}_{\text{pub}})$ and sends $\mathsf{Enc}(Z_{B})$ to Party A (provider), who computes the encrypted cross-covariance $\mathsf{Enc}(Z_{A}^{\top}Z_{B})$ using plaintext $Z_{A}=g_{A}(\mathcal{D}_{\text{pub}})$ and returns $\mathsf{Enc}(Z_{A}^{\top}Z_{B})$ to Party B. Party B decrypts $Z_{A}^{\top}Z_{B}$ and computes $W^{*}$ locally using Eq. ([1](#S4.E1)). Inference: Party B computes aligned embedding $\hat{z}_{A}=z_{B}\cdot W^{*}+b^{*}$ locally, encrypts $\mathsf{Enc}_{I}(\hat{z}_{A})$, and sends to Party A, who applies the classifier homomorphically and returns the encrypted prediction for Party B to decrypt.
Refer to caption: https://arxiv.org/html/2603.18908/2603.18908v4/x6.png

### 4.2 Threat Model

We adopt a semi-honest threat model Goldreich (2004) where both parties follow the protocol but may analyze communications to infer private information. Client query privacy is the primary guarantee: Party A observes only CKKS ciphertexts during inference, which reveal nothing about queries $x$ or embeddings $g_{B}(x)$ without Party B’s secret key. Party A’s classifier parameters $(V,c)$ are never revealed in plaintext; adaptive attacks such as model extraction are out of scope and mitigable via encrypted argmax Jovanovic et al. (2022); Zhang et al. (2024a).

Limitations. Party B retains $W^{*}$ in plaintext, which reveals structural information about Party A’s embedding space but not classifier parameters $(V,c)$ or training data; membership inference on $W^{*}$ yields near-chance accuracy (Appendix [K](#A11)). As in most HE deployments, structural metadata (embedding dimensions, communication volume) is revealed, and our semi-honest model does not cover adaptive model extraction attacks.

### 4.3 Secure Linear Alignment

Given public data $\mathcal{D}_{\text{pub}}$ with embeddings
$Z_{A}=g_{A}(\mathcal{D}_{\text{pub}})$ and $Z_{B}=g_{B}(\mathcal{D}_{\text{pub}})$,
we compute the affine alignment from the sufficient statistics $Z_{B}^{\top}Z_{B}$ and $Z_{B}^{\top}Z_{A}$:

$$ $W^{*}=(Z_{B}^{\top}Z_{B}+\lambda I)^{-1}Z_{B}^{\top}Z_{A}.$ (1) $$

To avoid revealing raw embeddings, we stream aggregates over mini-batches; the cross-covariance $Z_{B}^{\top}Z_{A}$ is computed under homomorphic encryption via secure aggregation.

Public data assumption.
We assume a shared non-sensitive public dataset $\mathcal{D}_{\text{pub}}$ for fitting $W^{*}$; optionally, Party A may augment it with 64–128 in-distribution samples to improve alignment at the cost of limited data exposure, with membership inference risk provably bounded at $O(\sqrt{d}/N)\approx 0.016$ (Appendix [K](#A11)).

## 5 Privacy Experiments

We apply the HELIX framework to large-scale embedding models to evaluate its effectiveness on downstream tasks. Embeddings are commonly used directly as features to train a linear classifier on labeled data Devlin et al. (2018); Wang et al. (2022). Our experiments utilize five embedding models: embedding-001 (Google), text-embedding-3-small (OpenAI), E5-Mistral-7B-Instruct, Qwen3-Embedding-8B, and embed-english-v3.0 (Cohere).

### 5.1 Embedding Classification

We investigate the cross-silo knowledge transfer capabilities of HELIX between two private embedding models. In contrast to Section [3](#S3), we concentrate on training a linear map on a public dataset, rather than the in-distribution dataset.

**Table 2: Cross-model alignment classification accuracy (%). A classifier $f(\cdot)$ is trained on private target data (Full Data). We evaluate HELIXs by mapping representations from a public dataset (Public Only), and from the public dataset augmented with few-shot in-distribution samples (Public + ID, 64 or 128 shots), prior to classification by $f(\cdot)$. Baseline 64/128 are classifiers trained on only the few-shot samples. Best in bold, second-best underlined. Standard deviation across three seeds is $\pm 1.5$ pp.**
|  |  | Full | Baseline | Public | Public + ID |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Target | Source | Data | 64 | 128 | Only | 64 | 128 |
| TREC (17%) |  |  |  |  |  |  |  |
| Gemini | OpenAI | 95.4 | 56.6 | 77.6 | 57.6 | 78.0 | 81.0 |
| OpenAI | Qwen | 96.4 | 59.0 | 82.8 | 58.4 | 77.6 | 83.6 |
| Mistral | Cohere | 96.6 | 52.0 | 73.2 | 55.8 | 74.4 | 78.4 |
| OpenAI | Cohere | 96.4 | 52.2 | 73.4 | 68.4 | 80.2 | 80.6 |
| Mistral | Qwen | 96.6 | 59.4 | 82.2 | 65.6 | 83.8 | 87.4 |
| Qwen | OpenAI | 97.0 | 57.0 | 77.8 | 75.4 | 82.8 | 87.6 |
| MNLI (33%) |  |  |  |  |  |  |  |
| Gemini | OpenAI | 65.0 | 33.5 | 35.1 | 44.5 | 48.5 | 48.9 |
| OpenAI | Qwen | 62.4 | 57.7 | 61.9 | 64.0 | 64.9 | 65.2 |
| Mistral | Cohere | 77.6 | 34.4 | 35.8 | 39.5 | 41.9 | 42.8 |
| OpenAI | Cohere | 62.4 | 34.4 | 35.8 | 46.5 | 47.1 | 47.9 |
| Mistral | Qwen | 77.6 | 57.6 | 61.8 | 58.8 | 71.3 | 72.9 |
| Qwen | OpenAI | 87.9 | 33.4 | 35.2 | 42.0 | 42.9 | 43.9 |
| DBpedia (7%) |  |  |  |  |  |  |  |
| Gemini | OpenAI | 99.1 | 65.0 | 77.8 | 53.5 | 89.3 | 94.1 |
| OpenAI | Qwen | 99.1 | 66.0 | 83.4 | 66.7 | 91.0 | 95.8 |
| Mistral | Cohere | 99.2 | 56.5 | 78.9 | 44.5 | 70.0 | 86.7 |
| Gemini | Mistral | 99.1 | 52.0 | 79.1 | 65.7 | 93.4 | 96.6 |
| Mistral | Qwen | 99.2 | 72.9 | 93.2 | 79.6 | 93.6 | 96.5 |
| Qwen | OpenAI | 99.1 | 74.1 | 88.9 | 66.4 | 81.3 | 89.9 |
| AG News (25%) |  |  |  |  |  |  |  |
| Gemini | OpenAI | 92.1 | 79.6 | 87.0 | 85.9 | 88.4 | 88.7 |
| OpenAI | Qwen | 92.7 | 80.0 | 87.1 | 79.7 | 88.6 | 89.2 |
| Mistral | Cohere | 93.4 | 81.8 | 85.0 | 64.3 | 80.0 | 83.6 |
| Gemini | Mistral | 92.1 | 75.7 | 87.8 | 86.7 | 88.1 | 88.7 |
| Mistral | Qwen | 93.4 | 84.4 | 86.5 | 71.4 | 87.9 | 88.2 |
| Qwen | OpenAI | 92.9 | 82.0 | 86.1 | 84.6 | 86.2 | 87.2 |

Experimental Setup. We choose an embedding model to represent Party A and train a linear classifier $f(\cdot)$ on the full training set using embeddings $Z_{A}=g_{A}(X)$ and labels $Y$. We evaluate across six datasets: TREC, MNLI, DBpedia, and AG News.
We use embedding model $g_{B}(\cdot)$ to serve as Party B.

We then train the linear map $W^{*}$ under two settings: (1) using only a public, independent dataset to estimate $W^{*}$, and (2) augmenting the public dataset with a small number of in-distribution samples (64, 128) in the dataset under test. For the public dataset, we use Wikipedia and IMDB. Our baseline are a linear classifier trained on the full training set ($f(\cdot)$), as well as a classifier trained on the few-shot in-distribution samples (in practice the client would have access to the few-shot samples to build their own classifier). Setting (2) serves as an upper-bound analysis demonstrating the performance ceiling when $W^{*}$ uses limited in-distribution data (64-128 samples). However, this compromises data security by requiring Party A to share potentially sensitive samples with Party B during training, violating the zero-shot privacy guarantees of Setting (1). We argue that this performance gap quantifies the privacy-utility trade-off: how much accuracy is sacrificed to preserve full data confidentiality versus accepting limited data exposure for improved alignment.

Results. Table [2](#S5.T2) shows that cross-model alignment achieves strong performance across model pairs and datasets. Public + ID (64-128 samples) consistently matches or exceeds baselines trained on the same few-shot data, with particularly strong results on TREC and MNLI. Critically, Public Only surpasses 64-shot baselines in many configurations.

### 5.2 Computational Efficiency.

HELIX achieves sub-second end-to-end latency with less than 1MB of communication per sample by encrypting only the final linear head rather than the full transformer. We implement HELIX using TenSEAL CKKS with 128-bit security, achieving depth-1 encrypted evaluation with no bootstrapping required. This contrasts with prior secure transformer inference systems such as BOLT ($>$60s), Nimbus ($>$20s), and Nexus (37.3s), which incur substantially higher latency by securing end-to-end model execution. Full benchmarks are provided in Appendix [J](#A10) (Table [9](#A10.T9)).

## 6 Conclusion

We characterize the extent to which independently trained LLMs admit linear alignment at their output layers, showing that affine maps between final hidden states can preserve downstream performance on classification, out-of-distribution detection, and—in some cases—autoregressive text generation. Our analysis identifies tokenizer compatibility and source-model scale as the primary determinants of success, and our layer-sweep results localize this compatibility to late-stage representations. Building on these findings, we introduce HELIX, a privacy-preserving cross-silo inference framework that exploits output-layer compatibility to encrypt only linear operations, achieving sub-second latency under a semi-honest threat model.
Several directions remain for future work. Characterizing alignment at intermediate layers and across modalities could clarify whether the late-stage compatibility we observe reflects a broader structural property of large-scale pretraining or is specific to output representations. Developing tokenizer-agnostic alignment methods would extend applicability beyond models with overlapping vocabularies, and exploring efficient non-linear alignment could improve cross-model generation quality. On the security side, better protecting the provider’s model parameters and quantifying leakage through the learned alignment map remain open challenges.

## References

- S. K. Ainsworth, J. Hayase, and S. Srinivasa (2022)
Git re-basin: merging models modulo permutation symmetries.
arXiv preprint arXiv:2209.04836.
Cited by: [Appendix I](#A9.p2.1).
- G. Alain and Y. Bengio (2018)
Understanding intermediate layers using linear classifier probes.
External Links: 1610.01644,
[Link](https://arxiv.org/abs/1610.01644)
Cited by: [Appendix I](#A9.p2.1).
- Y. Bansal, P. Nakkiran, and B. Barak (2021)
Revisiting model stitching to compare neural representations.
In Advances in Neural Information Processing Systems (NeurIPS 34),
pp. 225–236.
External Links: 2106.07682
Cited by: [Appendix I](#A9.p2.1),
[§1](#S1.p2.1),
[§2](#S2.p3.1).
- Z. Brakerski, C. Gentry, and V. Vaikuntanathan (2014)
Leveled fully homomorphic encryption without bootstrapping.
ACM Trans. Comput. Theory 6 (3).
Cited by: [Appendix I](#A9.p7.1).
- N. Carlini, D. Paleka, K. Dj Dvijotham, T. Steinke, J. Hayase, A. F. Cooper, K. Lee, M. Jagielski, M. Nasr, A. Conmy, I. Yona, E. Wallace, D. Rolnick, and F. Tramèr (2024)
Stealing part of a production language model.
External Links: 2403.06634,
[Link](https://arxiv.org/abs/2403.06634)
Cited by: [§H.8](#A8.SS8.SSS0.Px6.p1.1).
- N. Carlini, F. Tramèr, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown, D. Song, Ú. Erlingsson, A. Oprea, and C. Raffel (2021)
Extracting training data from large language models.
In 30th USENIX Security Symposium (USENIX Security 21),
pp. 2633–2650.
External Links: ISBN 978-1-939133-24-3,
[Link](https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting)
Cited by: [§H.7](#A8.SS7.SSS0.Px1.p1.1).
- K. Chaudhuri, C. Monteleoni, and A. D. Sarwate (2011)
Differentially private empirical risk minimization..
Journal of Machine Learning Research 12 (3).
Cited by: [§H.8](#A8.SS8.SSS0.Px3.p1.4).
- A. Chen, J. Merullo, A. Stolfo, and E. Pavlick (2025)
Transferring features across language models with model stitching.
arXiv preprint arXiv:2506.06609.
Cited by: [§1](#S1.p2.1),
[§2](#S2.p3.1).
- J. H. Cheon, A. Kim, M. Kim, and Y. Song (2017)
Homomorphic encryption for arithmetic of approximate numbers.
In International conference on the theory and application of cryptology and information security,
pp. 409–437.
Cited by: [§H.2](#A8.SS2.p1.1),
[Appendix I](#A9.p7.1),
[§4](#S4.SS0.SSS0.Px1.p1.1).
- J. Devlin, M. Chang, K. Lee, and K. Toutanova (2018)
BERT: pre-training of deep bidirectional transformers for language understanding.
arXiv:1810.04805.
Cited by: [Appendix I](#A9.p3.1),
[§3.1](#S3.SS1.p1.1),
[§5](#S5.p1.1).
- G. Diebold (2023)
Overcoming barriers to data sharing in the united states.
Technical report
Center for Data Innovation.
Note: Report on legal, social, technical and economic obstacles to data sharing
External Links: [Link](https://www2.datainnovation.org/2023-data-sharing-barriers.pdf)
Cited by: [§1](#S1.p2.1).
- R. Entezari, H. Sedghi, O. Saukh, and B. Neyshabur (2021)
The role of permutation invariance in linear mode connectivity of neural networks.
arXiv preprint arXiv:2110.06296.
Cited by: [Appendix I](#A9.p2.1).
- K. B. et al. (2017)
Practical secure aggregation for privacy-preserving machine learning.
In ACM CCS,
Cited by: [Appendix I](#A9.p4.1).
- P. R. et al. (2018)
A generic framework for privacy preserving deep learning.
arXiv preprint arXiv:1811.04017.
Cited by: [Appendix I](#A9.p10.1).
- T. Furlanello, Z. C. Lipton, M. Tschannen, L. Itti, and A. Anandkumar (2018)
Born-again neural networks.
arXiv:1805.04770.
Cited by: [Appendix I](#A9.p3.1).
- C. Gentry (2009)
A fully homomorphic encryption scheme.
Stanford University.
Cited by: [Appendix I](#A9.p7.1).
- R. Gilad-Bachrach, N. Dowlin, K. Laine, K. Lauter, M. Naehrig, and J. Wernsing (2016a)
CryptoNets: applying neural networks to encrypted data with high throughput and accuracy.
In Proceedings of the 33rd International Conference on Machine Learning (ICML),
pp. 201–210.
Cited by: [§H.2](#A8.SS2.p1.1).
- R. Gilad-Bachrach, N. Dowlin, K. Laine, K. Lauter, M. Naehrig, and J. Wernsing (2016b)
CryptoNets: applying neural networks to encrypted data with high throughput and accuracy.
In Proceedings of the 33rd International Conference on Machine Learning (ICML),
Cited by: [Appendix I](#A9.p8.1),
[§4](#S4.SS0.SSS0.Px1.p1.1).
- O. Goldreich (2004)
Foundations of cryptography: volume 2.
Cambridge University Press.
Cited by: [§H.3](#A8.SS3.p1.1),
[§4.2](#S4.SS2.p1.3).
- L. O. Gostin and J. G. Hodge (2000)
Personal privacy and the health information portability and accountability act (hipaa): A comment on the supreme court’s decision in Ferguson v. City of Charleston.
Journal of Law, Medicine & Ethics 28 (2), pp. 210–213.
Cited by: [§1](#S1.p2.1).
- M. Hao, H. Li, H. Chen, P. Xing, G. Xu, and T. Zhang (2022)
Iron: private inference on transformers.
Advances in neural information processing systems 35, pp. 15718–15731.
Cited by: [Appendix I](#A9.p9.1).
- D. Hendrycks and K. Gimpel (2016)
A baseline for detecting misclassified and out-of-distribution examples in neural networks.
arXiv preprint arXiv:1610.02136.
Cited by: [Appendix G](#A7.p2.2).
- G. Hinton, O. Vinyals, and J. Dean (2015)
Distilling the knowledge in a neural network.
In NIPS Deep Learning Workshop,
Cited by: [Appendix I](#A9.p3.1).
- J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. (2022)
Training compute-optimal large language models.
arXiv preprint arXiv:2203.15556.
Cited by: [§1](#S1.p1.1).
- M. Huh, B. Cheung, T. Wang, and P. Isola (2024)
Position: the platonic representation hypothesis.
In Proceedings of the 41st International Conference on Machine Learning,
Proceedings of Machine Learning Research, Vol. 235, pp. 20617–20642.
External Links: [Link](https://arxiv.org/abs/2405.07987)
Cited by: [Appendix I](#A9.p2.1),
[§1](#S1.p1.1),
[§2](#S2.p2.1).
- F. Jiang and J. Li (2024)
Stitching gpt to bert for look-ahead language understanding.
In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL),
Cited by: [§1](#S1.p2.1),
[§2](#S2.p3.1).
- N. Jovanovic, M. Fischer, S. Steffen, and M. Vechev (2022)
Private and reliable neural network inference.
In Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security,
pp. 1663–1677.
Cited by: [§H.7](#A8.SS7.SSS0.Px1.p1.1),
[Appendix I](#A9.p8.1),
[§4](#S4.SS0.SSS0.Px1.p1.1),
[§4.2](#S4.SS2.p1.3).
- C. Juvekar, V. Vaikuntanathan, and A. Chandrakasan (2018)
Gazelle: a low latency framework for secure neural network inference.
In USENIX Security,
Cited by: [§H.2](#A8.SS2.p1.1),
[Appendix I](#A9.p8.1),
[§4](#S4.SS0.SSS0.Px1.p1.1).
- P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, and et al. (2021)
Advances and open problems in federated learning.
Foundations and Trends in Machine Learning 14 (1–2), pp. 1–210.
Cited by: [Appendix I](#A9.p4.1).
- J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, A. Chowdhery, R. Chen, N. Elhage, R. Ebrahimi, R. Musabi, G. Khan, et al. (2020)
Scaling Laws for Neural Language Models.
arXiv preprint arXiv:2001.08361.
Cited by: [§1](#S1.p1.1).
- S. P. Karimireddy, S. Kale, M. Mohri, S. Stich, and A. T. Suresh (2020)
SCAFFOLD: stochastic controlled averaging for federated learning.
In Proceedings of the 37th International Conference on Machine Learning (ICML),
pp. 5132–5143.
Cited by: [Appendix I](#A9.p4.1).
- Y. Kim and A. M. Rush (2016)
Sequence-level knowledge distillation.
In EMNLP,
pp. 1317–1327.
Cited by: [Appendix I](#A9.p3.1).
- M. Klabunde, T. Schumacher, M. Strohmaier, and F. Lemmerich (2025)
Similarity of neural network models: a survey of functional and representational measures.
ACM Computing Surveys 57 (9), pp. 1–52.
Cited by: [§2](#S2.p1.1).
- B. Knott, S. Venkataraman, A. Hannun, S. Sengupta, M. Ibrahim, and L. van der Maaten (2021)
CrypTen: secure multi-party computation meets machine learning.
In NeurIPS,
External Links: [Link](https://papers.neurips.cc/paper/2021/file/2754518221cfbc8d25c13a06a4cb8421-Paper.pdf)
Cited by: [Appendix I](#A9.p10.1).
- S. Kornblith, M. Norouzi, H. Lee, and G. Hinton (2019)
Similarity of neural network representations revisited.
In International conference on machine learning,
pp. 3519–3529.
Cited by: [§A.1](#A1.SS1.SSS0.Px1.p4.1),
[§A.1](#A1.SS1.p1.3),
[Appendix I](#A9.p2.1),
[Figure 2](#S2.F2),
[§2](#S2.p2.1).
- S. Lee, G. Lee, J. W. Kim, J. Shin, and M. Lee (2023)
HETAL: efficient privacy-preserving transfer learning with homomorphic encryption.
In International conference on machine learning,
pp. 19010–19035.
Cited by: [§H.2](#A8.SS2.p1.1),
[Appendix I](#A9.p11.1),
[§4](#S4.SS0.SSS0.Px1.p1.1).
- K. Lenc and A. Vedaldi (2015)
Understanding image representations by measuring their equivariance and equivalence.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
Cited by: [§2](#S2.p3.1).
- D. Li, H. Wang, R. Shao, H. Guo, E. Xing, and H. Zhang (2023)
MPCFORMER: FAST, PERFORMANT AND PRIVATE TRANSFORMER INFERENCE WITH MPC.
In The Eleventh International Conference on Learning Representations,
External Links: [Link](https://openreview.net/forum?id=CWmvjOEhgH-)
Cited by: [Appendix I](#A9.p9.1).
- T. Li, A. K. Sahu, A. Talwalkar, and V. Smith (2020)
Federated optimization in heterogeneous networks.
In Proceedings of Machine Learning and Systems (MLSys),
Cited by: [Appendix I](#A9.p4.1).
- Y. Li, J. Yosinski, J. Clune, H. Lipson, and J. Hopcroft (2015)
Convergent learning: do different neural networks learn the same representations?.
arXiv preprint arXiv:1511.07543.
Cited by: [Appendix I](#A9.p2.1),
[§2](#S2.p1.1).
- Z. Li, K. Yang, J. Tan, W. Lu, H. Wu, X. Wang, Y. Yu, D. Zhao, Y. Zheng, M. Guo, and J. Leng (2024)
Nimbus: secure and efficient two-party inference for transformers.
In Advances in Neural Information Processing Systems (NeurIPS 2024),
Note: arXiv:2411.15707
External Links: [Link](https://arxiv.org/abs/2411.15707)
Cited by: [Appendix I](#A9.p9.1),
[§4](#S4.SS0.SSS0.Px1.p1.1).
- W. Liu et al. (2021)
Energy-based out-of-distribution detection.
In ICLR,
Cited by: [Appendix G](#A7.p2.2),
[Appendix G](#A7.p3.1),
[§3.1](#S3.SS1.SSS0.Px1.p1.1).
- Y. Liu, Y. Kang, C. Xing, T. Chen, and Q. Yang (2020)
A secure federated transfer learning framework.
IEEE Intelligent Systems 35 (4), pp. 70–82.
External Links: [Link](https://arxiv.org/pdf/1812.03337)
Cited by: [Appendix I](#A9.p5.1).
- Y. Liu, T. Chen, and Q. Yang (2022)
Hierarchical federated learning: algorithms, applications and challenges.
arXiv preprint arXiv:2109.11846.
Cited by: [Appendix I](#A9.p4.1).
- J. Luo, Y. Zhang, Z. Zhang, J. Zhang, X. Mu, H. Wang, Y. Yu, and Z. Xu (2024)
SecFormer: fast and accurate privacy-preserving inference for transformer models via smpc.
arXiv preprint arXiv:2401.00793.
External Links: [Link](https://arxiv.org/abs/2401.00793)
Cited by: [Appendix I](#A9.p9.1).
- H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas (2017)
Communication-efficient learning of deep networks from decentralized data.
In Proceedings of Machine Learning Research (AISTATS),
Vol. 54, pp. 1273–1282.
External Links: [Link](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
Cited by: [Appendix I](#A9.p4.1).
- D. McNeely-White, J. R. Beveridge, and B. A. Draper (2020)
Inception and resnet features are (almost) equivalent.
Cognitive Systems Research 59, pp. 312–318.
Cited by: [§2](#S2.p3.1).
- D. McNeely-White, B. Sattelberg, N. Blanchard, and R. Beveridge (2022)
Canonical face embeddings.
IEEE Transactions on Biometrics, Behavior, and Identity Science 4 (2), pp. 197–209.
Cited by: [§2](#S2.p3.1).
- P. Mohassel and Y. Zhang (2017)
SecureML: a system for scalable privacy-preserving machine learning.
In IEEE Symposium on Security and Privacy (SP),
pp. 19–38.
External Links: [Link](https://www.ieee-security.org/TC/SP2017/papers/466.pdf)
Cited by: [§H.2](#A8.SS2.p1.1),
[Appendix I](#A9.p10.1).
- A. Mora, I. Tenison, P. Bellavista, and I. Rish (2022)
Knowledge distillation for federated learning: a practical guide.
arXiv preprint arXiv:2211.04742.
Cited by: [Appendix I](#A9.p5.1).
- A. S. Morcos, M. Raghu, and S. Bengio (2018)
Insights on representational similarity in neural networks with canonical correlation.
In Advances in Neural Information Processing Systems,
Vol. 31.
Cited by: [§2](#S2.p2.1).
- S. J. Pan and Q. Yang (2010)
A survey on transfer learning.
IEEE Transactions on Knowledge and Data Engineering.
Cited by: [Appendix I](#A9.p3.1).
- Q. Pang, J. Zhu, H. Möllering, W. Zheng, and T. Schneider (2024)
Bolt: privacy-preserving, accurate and efficient inference for transformers.
In 2024 IEEE Symposium on Security and Privacy (SP),
pp. 4753–4771.
Cited by: [Appendix I](#A9.p9.1),
[§4](#S4.SS0.SSS0.Px1.p1.1).
- D. Park, E. Lee, and J. Lee (2024)
Powerformer: efficient privacy-preserving transformer with batch rectifier-power max function and optimized homomorphic attention.
Technical report
Cryptology ePrint Archive.
Cited by: [Appendix I](#A9.p8.1),
[§4](#S4.SS0.SSS0.Px1.p1.1).
- M. Raghu, J. Gilmer, J. Yosinski, and J. Sohl-Dickstein (2017)
Svcca: singular vector canonical correlation analysis for deep understanding and improvement.
stat 1050, pp. 19.
Cited by: [§A.2](#A1.SS2.p1.1),
[Appendix I](#A9.p2.1),
[§2](#S2.p1.1).
- C. Reimsbach-Kounatze, S. Ishikawa, et al. (2025)
Sharing trustworthy ai models with privacy-enhancing technologies.
Technical report
Technical Report No. 38, OECD Artificial Intelligence Papers, OECD.
External Links: [Link](https://www.oecd.org/content/dam/oecd/en/publications/reports/2025/06/sharing-trustworthy-ai-models-with-privacy-enhancing-technologies_5df6fd05/a266160b-en.pdf)
Cited by: [§1](#S1.p2.1).
- G. Roeder, Y. Wu, D. Duvenaud, and R. Grosse (2020)
On linear identifiability of learned representations.
arXiv preprint arXiv:2007.00810.
Cited by: [§2](#S2.p4.2).
- A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio (2015)
FitNets: hints for thin deep nets.
In ICLR,
Cited by: [Appendix I](#A9.p3.1).
- V. Smith, C. Chiang, M. Sanjabi, and A. Talwalkar (2017)
Federated multi-task learning.
arXiv preprint arXiv:1705.10467.
Cited by: [Appendix I](#A9.p4.1).
- A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, et al. (2022)
Beyond the imitation game: quantifying and extrapolating the capabilities of language models.
arXiv preprint arXiv:2206.04615.
External Links: [Link](https://arxiv.org/abs/2206.04615)
Cited by: [§1](#S1.p1.1).
- Y. Tang and Y. Yang (2024)
Pooling and attention: what are effective designs for llm-based embedding models?.
arXiv preprint arXiv:2409.02727.
Cited by: [§A.1](#A1.SS1.SSS0.Px1.p2.3).
- Y. Tian, D. Krishnan, and P. Isola (2020)
Contrastive representation distillation.
In ICLR,
Cited by: [Appendix I](#A9.p3.1).
- F. Tramèr, F. Zhang, A. Juels, M. K. Reiter, and T. Ristenpart (2016)
Stealing machine learning models via prediction apis.
In Proceedings of the 25th USENIX Conference on Security Symposium,
SEC’16, USA, pp. 601–618.
External Links: ISBN 9781931971324
Cited by: [§H.7](#A8.SS7.SSS0.Px1.p1.1),
[§H.8](#A8.SS8.SSS0.Px6.p1.1).
- L. Tunstall, N. Reimers, U. E. S. Jo, L. Bates, D. Korat, M. Wasserblat, and O. Pereg (2022)
Efficient few-shot learning without prompts.
arXiv preprint arXiv:2209.11055.
Cited by: [§3.1](#S3.SS1.p1.1).
- P. Vepakomma, O. Gupta, T. Swedish, and R. Raskar (2019)
Split learning for health: distributed deep learning without sharing raw patient data.
In ICLR 2019 Workshop on AI for Social Good,
External Links: [Link](https://aiforsocialgood.github.io/iclr2019/accepted/track1/pdfs/31_aisg_iclr2019.pdf)
Cited by: [Appendix I](#A9.p6.1).
- P. Voigt and A. Von dem Bussche (2017)
The european general data protection regulation (gdpr).
A Practical Guide. Springer International Publishing, pp. 10–15.
Cited by: [§1](#S1.p2.1).
- J. Wang, Q. Chen, H. Sun, Z. Shi, and E. al. Yang (2020)
Tackling the objective inconsistency problem in heterogeneous federated optimization.
Advances in Neural Information Processing Systems 33, pp. 7611–7623.
Cited by: [Appendix I](#A9.p4.1).
- L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, and F. Wei (2022)
Text embeddings by weakly-supervised contrastive pre-training.
arXiv preprint arXiv:2212.03533.
Cited by: [§5](#S5.p1.1).
- F. Wu, L. Cui, S. Yao, and S. Yu (2024)
Inference attacks in machine learning as a service: a taxonomy, review, and promising directions.
arXiv e-prints, pp. arXiv–2406.
Cited by: [§H.8](#A8.SS8.SSS0.Px6.p1.1).
- J. Zhang, X. Yang, L. He, K. Chen, W. Lu, Y. Wang, X. Hou, J. Liu, K. Ren, and X. Yang (2024a)
Secure transformer inference made non-interactive.
Cryptology ePrint Archive.
Cited by: [§H.7](#A8.SS7.SSS0.Px1.p1.1),
[Appendix I](#A9.p8.1),
[§4](#S4.SS0.SSS0.Px1.p1.1),
[§4.2](#S4.SS2.p1.3).
- L. Zhang, M. Y. Li, and T. L. Griffiths (2024b)
What should embeddings embed? autoregressive models represent latent generating distributions.
arXiv preprint arXiv:2406.03707.
Cited by: [§A.1](#A1.SS1.SSS0.Px1.p2.3).
- Y. Zhang, H. Wang, et al. (2025)
Encryption-friendly large language model architectures.
In International Conference on Learning Representations (ICLR),
Cited by: [Appendix I](#A9.p11.1),
[§4](#S4.SS0.SSS0.Px1.p1.1).

## Appendix

## Appendix A LLM Similarity

To assess the extent to which independently trained embedding models learn compatible linear structure, we analyze representational similarity using two complementary metrics: linear Centered Kernel Alignment (CKA) and Singular Vector Canonical Correlation Analysis (SVCCA). Both measures operate directly on representation matrices computed over a shared set of inputs and quantify the degree of shared linear geometry across models.

### A.1 CKA Heatmaps

Our primary similarity analysis uses linear CKA Kornblith et al. (2019), which measures similarity between two representation matrices
$Z_{A}\in\mathbb{R}^{n\times d_{1}}$ and
$Z_{B}\in\mathbb{R}^{n\times d_{2}}$
computed over the same set of $n$ inputs. Linear CKA is invariant to isotropic scaling and orthogonal transformations, making it a stable and widely used metric for comparing internal representations across models with different dimensionalities or parameterizations.

Formally, linear CKA is defined as

$$ $\mathrm{CKA}(Z_{A},Z_{B})=\frac{\|Z_{A}^{\top}Z_{B}\|_{F}^{2}}{\|Z_{A}^{\top}Z_{A}\|_{F}\,\|Z_{B}^{\top}Z_{B}\|_{F}},$ $$

where $\|\cdot\|_{F}$ denotes the Frobenius norm. A CKA value of $1$ indicates identical representations up to orthogonal transformation and scaling, while a value of 0 indicates no shared linear structure.

##### Representation extraction.

We compute $Z_{A}$ and $Z_{B}$ using a mixture of deployed embedding APIs and locally run autoregressive LLMs. For embedding APIs, we use the provider’s default embedding outputs, which already reflect model-specific pooling and normalization strategies. For locally run autoregressive models, which are trained primarily for next-token prediction and often instruction fine-tuned, we adopt mean pooling over the final hidden layer to obtain fixed-dimensional sequence representations.

Concretely, given final-layer token embeddings $\{h_{i}\}_{i=1}^{L}$ for an input sequence of length $L$, we compute the sequence-level embedding as

$$ $z=\frac{1}{L}\sum_{i=1}^{L}h_{i}.$ $$

Prior work has shown that pooling hidden states of autoregressive models yields semantic representations competitive with those of dedicated embedding models Zhang et al. (2024b); Tang and Yang (2024), making this a reasonable and widely adopted choice for cross-model comparison.

Experimental Procedure.
We compute linear CKA between pairs of embedding models using representations extracted on shared training and test splits. Embeddings are mean-centered prior to CKA computation. For embedding APIs, we use the provider’s default outputs, while for instruction-tuned autoregressive models we apply mean pooling over final-layer token embeddings.

CKA is computed on both splits following Kornblith et al. (2019), using 5,000 training and 2,000 test samples. Each experiment is repeated three times with different random subsamples, and results are averaged.

##### Results.

Figure [7](#A1.F7) reports linear CKA similarity across instruction-tuned autoregressive LLMs, computed on Alpaca and TriviaQA inputs. Each heatmap reports average test-set CKA computed using 5,000 samples to estimate representation covariance, with evaluation performed on 2,000 held-out samples. Results are averaged over three random subsampling runs to reduce variance.

We observe moderate to high CKA similarity across most autoregressive model pairs, indicating substantial shared linear structure despite differences in architecture, pretraining data, and instruction-tuning procedures. While variability across pairs is more pronounced than in encoder-style embedding models, many pairs still exhibit CKA values above $0.5$, which corresponds to meaningful shared linear structure and suggests strong potential for linear alignment. We hypothesize that the increased variance arises from task-specific shifts introduced during instruction tuning, which may alter representation geometry while preserving a common semantic backbone.

### A.2 SVCCA

To complement CKA, we analyze representations using Singular Vector Canonical Correlation Analysis (SVCCA) Raghu et al. (2017), which measures correlation between low-rank subspaces extracted via singular value decomposition followed by canonical correlation analysis. Unlike CKA, which evaluates global similarity between full representation matrices, SVCCA emphasizes shared informative subspaces, making it particularly relevant for assessing the feasibility of linear alignment.

SVCCA Experimental Procedure
For each model pair, we extract embeddings on shared training and test splits. Encoder-style embedding models use provider-default outputs, while instruction-tuned autoregressive models use mean pooling over final-layer token embeddings.

SVCCA is fit on the training embeddings by first applying PCA to each model’s representations, followed by canonical correlation analysis (CCA) on the reduced features. We evaluate SVCCA on both the training and test splits using the fitted PCA and CCA transforms. We report results using fixed PCA dimensionalities of 64 and 128 components.

To ensure comparability, embeddings are aligned by truncating to the minimum number of available samples across models, with 10,000 training examples and 2,000 test examples. Each experiment is repeated three times with different random subsamples, and reported correlations are averaged across runs. As a control, we compute a random baseline by shuffling one model’s embeddings prior to SVCCA.

Figures [9](#A1.F9) and [10](#A1.F10) report SVCCA results for encoder-style embedding models evaluated across multiple datasets and model pairings, using projections onto the top 64 and 128 components, respectively. Across datasets and model combinations, we observe consistently high SVCCA correlations, indicating a strong shared low-rank subspace among embedding models.

##### Summary.

Taken together, the CKA and SVCCA analyses provide evidence that independently trained embedding models learn compatible representations. Despite differences in training objectives, architectures, and fine-tuning procedures, these models preserve shared geometric structure that is amenable to linear alignment, motivating our subsequent investigation into whether such alignment suffices for downstream behavioral transfer.

Figure: Figure 7: Test-set CKA between embeddings across Alpaca and Trivia QA datasets. Model embeddings were mean pooled at their final hidden state prior to calculating CKA with other models.
Refer to caption: https://arxiv.org/html/2603.18908/2603.18908v4/x7.png

Figure: Figure 8: SVCCA of eight instruction tuned LLM combinations at 64 components. Mean and median are bolded.
Refer to caption: https://arxiv.org/html/2603.18908/2603.18908v4/x8.png

Figure: Figure 9: SVCCA of four datasets on five embedding model combinations.
Refer to caption: https://arxiv.org/html/2603.18908/2603.18908v4/x9.png

Figure: Figure 10: SVCCA of five embedding model combinations at 128 components (test set)
Refer to caption: https://arxiv.org/html/2603.18908/2603.18908v4/x10.png

### A.3 Task-Supervised Alignment Baseline

To characterize the best-case performance of linear alignment, we first evaluate an
*in-distribution* setting where the mapping is trained directly on the task’s
training split. Concretely, we fit $(W^{*},b^{*})$ using paired embeddings from the
training set (computed by $g_{A}$ and $g_{B}$ on the same inputs) and evaluate on the
held-out test split using the fixed classifier $f_{A}(\cdot)$ trained on target
embeddings. This baseline does *not* use a separate public dataset
$\mathcal{D}_{\text{pub}}$ for alignment; instead, it reflects a setting where the
alignment data matches the downstream task distribution and provides a reference
point for the public-data mapping results used in our privacy-preserving protocol.

| Party A | Party B | Classification Acc. | OOD | OOD AUROC |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| (Target) | (Source) | Baseline | HELIX | Dataset | Baseline | HELIX |
| AG News (25%) |  |  |  |  |  |  |
| Gemini | OpenAI | 92.6 | 91.6 | MNLI | 0.908 | 0.891 |
| OpenAI | Qwen | 92.6 | 91.9 | MNLI | 0.953 | 0.913 |
| Cohere | Gemini | 91.9 | 91.5 | MNLI | 0.875 | 0.885 |
| OpenAI | Cohere | 92.1 | 92.0 | MNLI | 0.881 | 0.880 |
| Mistral | Qwen | 92.6 | 92.4 | MNLI | 0.953 | 0.940 |
| DBpedia (7%) |  |  |  |  |  |  |
| Gemini | OpenAI | 98.9 | 98.6 | Yahoo | 0.971 | 0.975 |
| OpenAI | Qwen | 99.0 | 98.7 | Yahoo | 0.979 | 0.974 |
| Cohere | Gemini | 98.9 | 98.7 | Yahoo | 0.973 | 0.980 |
| OpenAI | Cohere | 98.9 | 98.7 | Yahoo | 0.971 | 0.973 |
| Mistral | Qwen | 99.0 | 98.9 | Yahoo | 0.979 | 0.985 |
| Yahoo Answers Topics (10%) |  |  |  |  |  |  |
| Gemini | OpenAI | 70.9 | 67.6 | SST-2 | 0.417 | 0.612 |
| OpenAI | Qwen | 71.5 | 68.6 | SST-2 | 0.623 | 0.491 |
| Cohere | Gemini | 70.2 | 67.3 | SST-2 | 0.566 | 0.593 |
| OpenAI | Cohere | 69.7 | 69.4 | SST-2 | 0.481 | 0.444 |
| Mistral | Qwen | 71.5 | 70.5 | SST-2 | 0.623 | 0.482 |
| MNLI (33%) |  |  |  |  |  |  |
| Gemini | OpenAI | 62.4 | 61.3 | IMDB | 0.367 | 0.092 |
| OpenAI | Qwen | 87.9 | 61.3 | IMDB | 0.905 | 0.693 |
| Cohere | Gemini | 65.0 | 56.8 | IMDB | 0.111 | 0.263 |
| OpenAI | Cohere | 59.1 | 58.3 | IMDB | 0.384 | 0.285 |
| Mistral | Qwen | 87.9 | 78.5 | IMDB | 0.905 | 0.650 |
| SST-2 (50%) |  |  |  |  |  |  |
| Gemini | OpenAI | 94.5 | 93.1 | AGNews | 0.826 | 0.801 |
| OpenAI | Qwen | 94.5 | 93.8 | AGNews | 0.859 | 0.848 |
| Cohere | Gemini | 94.4 | 91.7 | AGNews | 0.819 | 0.870 |
| OpenAI | Cohere | 93.0 | 92.4 | AGNews | 0.875 | 0.843 |
| Mistral | Qwen | 94.5 | 93.7 | AGNews | 0.859 | 0.826 |
| IMDB (50%) |  |  |  |  |  |  |
| Gemini | OpenAI | 94.9 | 95.9 | Amazon | 0.523 | 0.543 |
| OpenAI | Qwen | 95.7 | 94.8 | Amazon | 0.535 | 0.535 |
| Cohere | Gemini | 96.4 | 94.5 | Amazon | 0.572 | 0.501 |
| OpenAI | Cohere | 94.8 | 94.9 | Amazon | 0.379 | 0.534 |
| Mistral | Qwen | 95.7 | 95.1 | Amazon | 0.535 | 0.617 |
| TREC (16.7%) |  |  |  |  |  |  |
| Gemini | OpenAI | 95.4 | 94.0 | AGNews | 0.412 | 0.718 |
| OpenAI | Qwen | 96.0 | 95.6 | AGNews | 0.738 | 0.916 |
| Cohere | Gemini | 97.0 | 91.2 | AGNews | 0.921 | 0.766 |
| OpenAI | Cohere | 96.4 | 94.4 | AGNews | 0.954 | 0.802 |
| Mistral | Qwen | 97.0 | 96.6 | AGNews | 0.921 | 0.940 |

## Appendix B Text Generation Evaluation Methodology

To obtain a robust view of generation quality, we combine automated and human evaluation signals. We first use a dual LLM-as-a-judge protocol to score relevance, quality, and helpfulness at scale, then validate these trends with a blinded human study on a representative subset of generations. Finally, we analyze embedding-space similarity to characterize how mapped representations relate to both the source and target model feature spaces during generation.

### B.1 LLM-as-a-Judge Evaluation

We employ a dual-judge framework using GPT-5.2 (gpt-5.2) and Claude 4 Sonnet (claude-sonnet-4). Both judges independently evaluate each response, and final scores are averaged across judges.

##### Evaluation Prompt

The aggregate score is the mean of the three dimensions. Both judges operate at temperature 0.0 with JSON-formatted outputs.

### B.2 Human Evaluation

To validate the fidelity of automated evaluators used throughout this section (LLM-as-a-judge scores and embedding-based similarity analyses), we additionally collect an independent human assessment of a subset of generations. We sample 200 prompt–response pairs, export them to a Google spreadsheet, and hide the source configuration (single-model vs. cross-model, as well as model identity) from the evaluator. Each example is rated on a 1–5 Likert scale across four criteria: proper English, quality, and coherence.

##### Sample Selection

Samples are distributed across eight configurations to cover diverse performance levels. We select 2 models from each of the LLM-as-a-judge categories (cross-model (poor), cross-model (good), single model (small, poor), single model (large, good))

##### Cross-Model (100 samples, 25 per pair):

- •
Poor (LLM-judge score $<4.0$): Gemma-270M $\rightarrow$ Llama-3-8B, Qwen2.5-0.5B $\rightarrow$ Gemma-2-2B
- •
Strong (LLM-judge score $>7.0$): Llama-3-8B $\rightarrow$ Qwen2.5-14B, Qwen2.5-7B $\rightarrow$ Llama-3-8B

##### Single-Model (100 samples):

- •
Small (50 samples): Qwen2.5-0.5B, Gemma-270M (25 each)
- •
Large (50 samples): Qwen2.5-7B, Llama-3-8B (25 each)

All examples are presented in randomized order and evaluated blind to model configuration. The spreadsheet contains columns for prompt, completion, and the ranking column.

### B.3 Embedding Space Analysis

We analyze the cosine similarity between mapped representations and both source (Model A) and target (Model B) embeddings. Figure [11](#A2.F11) plots these similarities across eight model pairs, with each point representing a token position and X markers indicating mean values.

Figure: Figure 11: Embedding similarities between greedy-mapped text generation representations and cross-model text generation representations.
Refer to caption: https://arxiv.org/html/2603.18908/2603.18908v4/x11.png

Model pairs exhibit substantial variation: larger models (Qwen2.5-14B $\rightarrow$ Llama-3-8B, Qwen3-4B $\rightarrow$ Qwen2.5-14B) achieve high similarity to both source and target (upper-right quadrant), while smaller source models (gemma-3-270m, Qwen2.5-0.5B) show lower correlation to both (lower-left). Most pairs cluster above the diagonal with mean similarities of 0.5-0.7 to both models, suggesting mapped representations form an intermediate space that partially retains source structure while incorporating target characteristics.

### B.4 Training Data Size Selection

To determine an appropriate training dataset size for learning cross-model alignments, we conducted experiments varying the number of activation pairs used during mapper training. Figure [12](#A2.F12) shows the training and test loss curves for the Olmo-3-7B-Instruct $\rightarrow$ Meta-Llama-3-8B-Instruct alignment as a function of dataset size (ranging from 100 to 6,000 samples).

Figure: Figure 12: Training and test loss for cross-model alignment as a function of training dataset size. Test loss plateaus around 4,000 samples, while training loss continues to increase, suggesting diminishing returns beyond this point.
Refer to caption: https://arxiv.org/html/2603.18908/2603.18908v4/x12.png

We observe that test loss (gold line) decreases rapidly from 100 to 1,000 samples, then plateaus around 0.86-0.87 for larger dataset sizes. In contrast, training loss (pink line) continues to increase monotonically, rising from approximately 1.41 at 1,000 samples to 1.54 at 6,000 samples. This divergence indicates overfitting to the training set with larger datasets.

Based on these results, we select 4,000 samples as the standard training size for experiments, balancing computational efficiency with alignment effectiveness. This choice captures most of the performance gains while avoiding unnecessary computation and overfitting observed at larger dataset sizes.

### B.5 Token-Level Alignment

Given a shared text corpus, we extract final-layer hidden states from both models and align them at the token level using character offsets. For each token in the source model (A) at position $i$ with character end offset $e_{A}^{i}$, we match it to the target model (B) token whose end offset equals or immediately follows:

$$ $j=\arg\min_{k}\{e_{B}^{k}\mid e_{B}^{k}\geq e_{A}^{i}\}$ $$

This produces aligned pairs $(h_{A}^{i},h_{B}^{j})$ used for training the affine transformation. The character-offset alignment accommodates tokenizer differences between models without requiring identical vocabularies.

## Appendix C Text Generation Experimental Results

Table [3](#A3.T3) presents comprehensive tokenizer compatibility metrics and LLM-as-a-Judge generation quality scores for all evaluated model pairs. We organize results into three categories: high-quality pairs (both models $\geq$2B parameters, score $\geq$3.5), low-quality pairs (both models $\geq$2B, score $<$2.0), and pairs involving small models ($<$2B parameters). For comparison, we include single-model baseline scores at the bottom of the table.

High-quality pairs. The top section shows 11 model pairs that achieve functional cross-model generation (LLM-judge scores 3.9–4.7). These pairs exhibit strong tokenizer compatibility: vocabulary overlap (Jaccard) $\geq$0.64 and exact token match rates $\geq$0.67. Same-family pairs (e.g., Qwen-7B $\rightarrow$ Qwen-14B, Mistral-Nemo $\rightarrow$ Apertus-8B) achieve near-perfect compatibility (Jaccard $\geq$0.999) and the highest generation quality. Cross-family pairs between Qwen and Llama models also perform well (Jaccard = 0.643, exact match = 0.925), demonstrating that models with shared tokenizer vocabularies enable effective linear alignment even across different model families.

Low-quality pairs. The middle section shows 7 pairs with both models $\geq$2B that produce poor-quality text (scores 1.1–1.9). These failures correlate strongly with low tokenizer compatibility: Jaccard $\leq$0.32 and exact match $\leq$0.67. Notably, all pairs involving Gemma models with non-Gemma models fail (Jaccard 0.057–0.063, exact match 0.227–0.238). The Mistral-7B model (vocabulary size 32K) also shows poor compatibility with models using larger vocabularies (131K–151K tokens).

Small model pairs. The bottom section demonstrates that small models ($<$2B) consistently fail at cross-model generation regardless of tokenizer compatibility. Even same-family pairs with perfect tokenization alignment (Llama-3.2-1B $\rightarrow$ Llama-3-8B: Jaccard = 1.0, exact match = 1.0) achieve only 1.83 LLM-judge score, suggesting that representational capacity is a prerequisite for successful linear alignment. The gemma-3-270M model fails universally (scores 1.0–1.06) across all target models, including same-family Gemma models with perfect tokenizer compatibility.

Single-model baselines. For context, we include native single-model performance on the same evaluation set. Cross-model generation quality (3.9–4.7 for high-quality pairs) falls between small models (2.8–4.1) and large models (6.5–7.1), representing a 60–70% retention of baseline quality. This trade-off enables privacy-preserving cross-silo inference where traditional fine-tuning or data sharing is infeasible.

**Table 3: Tokenizer Compatibility Metrics and Generation Quality for Model Pairs**
| Model A | Model B | Vocab<br>Jaccard | Exact<br>Match | LLM Judge<br>Score |
| --- | --- | --- | --- | --- |
| High-Quality Pairs (Score $\geq$ 3.5, both $\geq$2B) |  |  |  |  |
| Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct | 1.000 | 1.000 | 4.68 |
| Qwen2.5-7B-Instruct | Meta-Llama-3-8B-Instruct | 0.643 | 0.925 | 4.47 |
| Qwen2.5-14B-Instruct | Meta-Llama-3-8B-Instruct | 0.643 | 0.925 | 4.43 |
| Mistral-Nemo-Instruct-2407 | Apertus-8B-Instruct-2509 | 0.999 | 1.000 | 4.35 |
| Meta-Llama-3-8B-Instruct | Qwen2.5-14B-Instruct | 0.643 | 0.925 | 4.31 |
| Mistral-Nemo-Instruct-2407 | Meta-Llama-3-8B-Instruct | 0.382 | 0.637 | 4.21 |
| Mistral-Nemo-Instruct-2407 | Qwen2.5-14B-Instruct | 0.316 | 0.672 | 4.13 |
| Qwen3-4B-Instruct-2507 | Qwen2.5-14B-Instruct | 1.000 | 1.000 | 4.07 |
| Qwen3-4B-Instruct-2507 | Meta-Llama-3-8B-Instruct | 0.643 | 0.925 | 3.99 |
| Qwen2.5-7B-Instruct | Apertus-8B-Instruct-2509 | 0.316 | 0.672 | 3.94 |
| Meta-Llama-3-8B-Instruct | Qwen2.5-7B-Instruct | 0.643 | 0.925 | 3.90 |
| Low-Quality Pairs (Score $<$ 2.0, both $\geq$2B) |  |  |  |  |
| Mistral-Nemo-Instruct-2407 | Mistral-7B-Instruct-v0.2 | 0.067 | 0.151 | 1.89 |
| Qwen2.5-14B-Instruct | Mistral-Nemo-Instruct-2407 | 0.316 | 0.672 | 1.86 |
| Mistral-Nemo-Instruct-2407 | gemma-3-27b-it | 0.057 | 0.238 | 1.79 |
| Mistral-Nemo-Instruct-2407 | gemma-3-12b-it | 0.057 | 0.238 | 1.73 |
| Qwen3-4B-Instruct-2507 | gemma-3-12b-it | 0.063 | 0.227 | 1.68 |
| Qwen3-4B-Instruct-2507 | gemma-3-27b-it | 0.063 | 0.227 | 1.31 |
| Qwen2.5-7B-Instruct | Mistral-7B-Instruct-v0.2 | 0.061 | 0.151 | 1.13 |
| Pairs Involving Small Models ($<$2B) |  |  |  |  |
| Llama-3.2-1B-Instruct | Meta-Llama-3-8B-Instruct | 1.000 | 1.000 | 1.83 |
| Llama-3.2-1B-Instruct | Qwen2.5-14B-Instruct | 0.643 | 0.925 | 1.53 |
| Qwen2.5-0.5B-Instruct | Meta-Llama-3-8B-Instruct | 0.643 | 0.925 | 1.18 |
| Llama-3.2-1B-Instruct | Qwen3-30B-A3B-Instruct-2507 | 0.643 | 0.925 | 1.17 |
| gemma-3-270m-it | gemma-3-27b-it | 1.000 | 1.000 | 1.06 |
| Llama-3.2-1B-Instruct | gemma-3-27b-it | 0.069 | 0.226 | 1.04 |
| gemma-3-270m-it | gemma-3-12b-it | 1.000 | 1.000 | 1.03 |
| gemma-3-270m-it | Meta-Llama-3-8B-Instruct | 0.069 | 0.226 | 1.03 |
| gemma-3-270m-it | Mistral-Nemo-Instruct-2407 | 0.057 | 0.238 | 1.02 |
| gemma-3-270m-it | Qwen3-30B-A3B-Instruct-2507 | 0.063 | 0.227 | 1.01 |
| gemma-3-270m-it | Qwen2.5-14B-Instruct | 0.063 | 0.227 | 1.00 |
| Single Models |  |  |  |  |
| DeepSeek-R1-Distill-Qwen-1.5B | – | – | – | 2.82 |
| google/gemma-3-270m-it | – | – | – | 3.40 |
| HuggingFaceTB/SmolLM2-1.7B | – | – | – | 3.68 |
| Qwen/Qwen2.5-0.5B-Instruct | – | – | – | 4.00 |
| google/gemma-2b | – | – | – | 4.08 |
| microsoft/Phi-3.5-mini-instruct (4B) | – | – | – | 5.36 |
| google/gemma-3-27b-it | – | – | – | 6.45 |
| Qwen/Qwen2.5-14B-Instruct | – | – | – | 6.65 |
| meta-llama/Meta-Llama-3-8B-Instruct | – | – | – | 7.05 |
| allenai/Olmo-3-7B-Instruct | – | – | – | 6.70 |

## Appendix D Cross-Model Text Generation: Sample Outputs

**Table 4: Example prompts and corresponding responses generated using cross-model linear alignment for three model pairs.**
| Llama-3-8B $\rightarrow$ Qwen2.5-14B |
| --- |
| Input a list of strings and generate all possible permutations of the strings.<br>Input: a,b,c |
| The following are the possible permutations of the given list of strings:<br>1. "a", "b", "c" |
| 2. ”a”, ”c”, ”b” |
| 3. ”b”, ”a”, ”c” |
| 4. ”b”, ”c”, ”a” |
| 5. ”c”, ”a”, ”b” |
| 6. ”c”, ”b”, ”a” |
| Tell me one fascinating fact about dogs |
| One fascinating fact about dogs is that they have a unique, highly developed sense of smell. In fact, a dog’s sense of smell is up to 1,000 to 10,000 more sensitive than a human’s. This is due to the large number of specialized, highly specialized, and highly specialized, and highly specialized, and highly specialized, ... |
| Provide a plan to improve the environment of your local park. |
| 1. **Clean-up and Maintenance: |
| - Organize a community clean-up event to collect and dispose of any litter, debris, and hazardous materials. |
| - Establish a regular maintenance schedule for the park’s landscaping, including regular watering, pruning, and planting of new plants and trees. |
| - Implement a recycling program for waste and encourage the use of reusable materials. |
| - Ensure proper maintenance of the park’s infrastructure, including the maintenance of the park’s pathways |

## Appendix E Layer-Depth Ablation: Where Does Linear Alignability Live?

The main experiments align the post-final-norm hidden states of two
instruction-tuned LLMs with a linear ridge mapper. Is the choice of
layer a free parameter, or is the cross-model linear correspondence
concentrated in a specific region of the residual stream? We answer
with a depth sweep on Qwen2.5-7B-Instruct ($L_{A}{=}28$, $d_{A}{=}3584$)
$\leftrightarrow$ Meta-Llama-3-8B-Instruct ($L_{B}{=}32$, $d_{B}{=}4096$),
in both directions.

##### Setup.

For each layer pair we extract per-token hidden
states on $4{,}000$ MMLU train questions ($\approx 1.55\textsc{m}$
aligned token pairs after BPE–BPE offset matching), solve OLS ridge
for $W:\mathbb{R}^{d_{A}}\!\to\!\mathbb{R}^{d_{B}}$, and evaluate by
greedy mapped generation through the target LM head on $100$ MMLU
test questions. The ridge residual reaches machine epsilon
($\sim 10^{-12}$) in every configuration, so accuracy reflects what
is recoverable from the source representations, not solver quality.
We sweep six depths per direction: $\sim L/4,L/2,3L/4$
(intermediate blocks); $h[-2]$ and $h[-1]$ (pre-final-norm); and
last_hidden_state (post-final-norm, used in the main paper).

##### Results.

Table [5](#A5.T5) reports the sweep.
Three findings stand out.

(i) Linear alignability is a step function in depth. Depths
Q1–Q3 fit the training pairs to machine precision yet decode to
$0\%$ accuracy in both directions, emitting rare-vocab fragments (e.g. ”.. _exempt_exempt..”) rather than coherent text. Accuracy
is flat at zero through $3L/4$, then jumps. The cross-model linear
structure lives at the boundary where the residual stream becomes
“LM-head-ready”.

(ii) The final RMSNorm is a free transformation. Pre-norm
last block ($h[-1]$) and post-norm (last_hidden_state)
tie exactly in both directions ($75{/}100$ and $50{/}100$): the
linear mapper absorbs the final rescaling without loss. This
justifies using the default last_hidden_state tensor in
the main paper.

(iii) One block of headroom is direction-asymmetric.
Pulling back to $h[-2]$ is a $70\to 67$ wobble for Qwen$\to$Llama
but a $58\to 0$ collapse for Llama$\to$Qwen. We attribute this to
(a) the expanding map $3584\to 4096$ being more forgiving than the
lossy reverse, and (b) recipe-specific differences in the final
blocks; we leave a mechanistic account to future work.

##### Implication.

The sweep validates two main-paper choices:
the post-final-norm vector is equivalent to pre-norm $h[-1]$, and
the linear-alignment phenomenon is genuinely a property of the
*top* of the residual stream—no shallower layer admits a
cheap, unrelated linear map. The sharp $3L/4\to L{-}1$ transition
is itself a nontrivial fact about how cross-model similarity
distributes across depth.

**Table 5: Layer-depth ablation on Qwen2.5-7B-Instruct $\leftrightarrow$ Meta-Llama-3-8B-Instruct. Linear ridge mapper, $4{,}000$ MMLU train questions ($\sim$$1.55\textsc{m}$ token pairs), evaluated on MMLU test questions via greedy mapped generation through the target LM head. OLS residual at machine epsilon in every row.**
| Depth | Description | Qwen $\to$ Llama | Llama $\to$ Qwen |  |  |
| --- | --- | --- | --- | --- | --- |
| $(l_{A},l_{B})$ | Acc. | $(l_{A},l_{B})$ | Acc. |  |  |
| Q1 | $\sim L/4$, pre-norm | (7, 8) | 0.00 | (8, 7) | 0.00 |
| Q2 | $\sim L/2$, pre-norm | (14, 16) | 0.00 | (16, 14) | 0.00 |
| Q3 | $\sim 3L/4$, pre-norm | (21, 24) | 0.00 | (24, 21) | 0.00 |
| PreL1 | $h[-2]$, pre-norm | (27, 31) | 0.67 | (31, 27) | 0.00 |
| PreL | $h[-1]$, pre-norm | (28, 32) | 0.70 | (32, 28) | 0.58 |
| Q4 | last_hidden_state | – | 0.70 | – | 0.58 |

## Appendix F HELIX Embeddings Classification - Full Results

Training Details
We use the following embedding models: Google embedding-001, OpenAI text-embedding-3-small, E5-Mistral-7B-Instruct, Qwen3-Embedding-8B, and Cohere embed-english-v3.0.

**Table 6: Embedding classification: Full results**
| Data<br>Owner | Client | Private Data<br>Linear Model<br>Accuracy(%) | Baseline<br>Samples:<br>32 | Baseline<br>Samples:<br>64 | Baseline<br>Samples:<br>128 | Public<br>Dataset<br>Only | Public DS +<br>In-Dist FS<br>(32 ) | Public DS +<br>In-Dist FS<br>(64) | Public DS +<br>In-Dist FS<br>(128) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DBpedia (7%) |  |  |  |  |  |  |  |  |  |
| Gemini | OpenAI | 99.1 | 51.2 | 65.0 | 77.8 | 53.5 | 83.7 | 89.3 | 94.1 |
| OpenAI | Qwen | 99.1 | 48.0 | 66.0 | 83.4 | 66.7 | 87.5 | 91.0 | 95.8 |
| Mistral | Cohere | 99.2 | 38.9 | 56.5 | 78.9 | 44.5 | 64.3 | 70.0 | 86.7 |
| Gemini | Mistral | 99.1 | 31.6 | 52.0 | 79.1 | 65.7 | 90.5 | 93.4 | 96.6 |
| Cohere | Gemini | 99.1 | 22.3 | 31.4 | 88.8 | 68.0 | 75.5 | 80.8 | 89.1 |
| OpenAI | Cohere | 99.1 | 56.4 | 78.9 | 91.8 | 62.3 | 76.9 | 83.4 | 92.5 |
| Mistral | Qwen | 99.2 | 54.4 | 72.9 | 93.2 | 79.6 | 91.1 | 93.6 | 96.5 |
| Qwen | OpenAI | 99.1 | 51.8 | 74.1 | 88.9 | 66.4 | 76.9 | 81.3 | 89.9 |
| AG News (25%) |  |  |  |  |  |  |  |  |  |
| Gemini | OpenAI | 92.1 | 72.7 | 79.6 | 87.0 | 85.9 | 88.2 | 88.4 | 88.7 |
| OpenAI | Qwen | 92.7 | 77.3 | 80 | 87.1 | 79.7 | 87.9 | 88.6 | 89.2 |
| Mistral | Cohere | 93.4 | 75.8 | 81.8 | 85.0 | 64.3 | 74.9 | 80 | 83.6 |
| Gemini | Mistral | 92.1 | 62.9 | 75.7 | 87.8 | 86.7 | 87.9 | 88.1 | 88.7 |
| Cohere | Gemini | 92.2 | 77.5 | 84.1 | 86.3 | 76.4 | 85.8 | 86.9 | 86.9 |
| OpenAI | Cohere | 92.7 | 70.7 | 75.1 | 85.8 | 78.9 | 82.9 | 84.9 | 86.0 |
| Mistral | Qwen | 93.4 | 74.3 | 84.4 | 86.5 | 71.4 | 84.2 | 87.9 | 88.2 |
| Qwen | OpenAI | 92.9 | 68.9 | 82.0 | 86.1 | 84.6 | 85.3 | 86.2 | 87.2 |
| Yahoo Answers (10%) |  |  |  |  |  |  |  |  |  |
| Gemini | OpenAI | 75.6 | 20.6 | 44.0 | 55.8 | 56.8 | 61.8 | 64.5 | 68.9 |
| OpenAI | Qwen | 76.3 | 32.9 | 40.1 | 48.8 | 62.7 | 65.0 | 64.7 | 65.2 |
| Mistral | Cohere | 75.6 | 25.9 | 42.5 | 54.9 | 60.7 | 57.0 | 59.6 | 60.0 |
| Gemini | Mistral | 75.6 | 20.6 | 44.0 | 55.8 | 61.0 | 62.0 | 65.5 | 67.1 |
| Cohere | Gemini | 73.9 | 24.6 | 43.0 | 58.8 | 65.0 | 65.2 | 66.1 | 66.5 |
| OpenAI | Cohere | 76.3 | 32.9 | 40.1 | 48.8 | 57.5 | 62.5 | 63.5 | 64.2 |
| Mistral | Qwen | 75.6 | 25.9 | 42.5 | 54.9 | 56.9 | 57.6 | 57.8 | 60.0 |
| Qwen | OpenAI | 74.9 | 21.0 | 41.0 | 50.9 | 60.7 | 60.1 | 57.5 | 57.4 |
| IMDB (50%) |  |  |  |  |  |  |  |  |  |
| Gemini | OpenAI | 96.4 | 90.3 | 90.9 | 91.5 | 94.7 | 94.6 | 94.8 | 95.0 |
| OpenAI | Qwen | 94.9 | 87.9 | 91.8 | 93.0 | 93.5 | 93.4 | 93.9 | 93.8 |
| Mistral | Cohere | 95.3 | 92.1 | 92.3 | 92.5 | 90.5 | 91.0 | 91.4 | 91.4 |
| Gemini | Mistral | 96.4 | 85.4 | 90.0 | 91.0 | 87.5 | 90.5 | 91.1 | 92.6 |
| Cohere | Gemini | 94.8 | 93.6 | 94.5 | 93.8 | 95.0 | 95.0 | 95.1 | 95.1 |
| OpenAI | Cohere | 94.9 | 91.8 | 93.6 | 93.4 | 93.2 | 93.3 | 93.3 | 93.5 |
| Mistral | Qwen | 95.3 | 87.3 | 91.0 | 92.5 | 90.6 | 93.5 | 93.9 | 94.0 |
| Qwen | OpenAI | 95.6 | 89.0 | 90.3 | 92.3 | 91.0 | 91.2 | 91.4 | 91.6 |
| SST-2 (50%) |  |  |  |  |  |  |  |  |  |
| Gemini | OpenAI | 94.4 | 92.8 | 92.8 | 92.8 | 91.9 | 92.8 | 92.7 | 93.0 |
| OpenAI | Qwen | 94.4 | 92.6 | 92.7 | 92.8 | 93.1 | 93.0 | 93.0 | 92.8 |
| Mistral | Cohere | 95.5 | 92.3 | 92.3 | 92.3 | 92.2 | 92.3 | 92.0 | 91.6 |
| Gemini | Mistral | 94.4 | 94.5 | 94.4 | 94.4 | 92.7 | 92.6 | 92.3 | 92.4 |
| Cohere | Gemini | 93.0 | 92.7 | 92.7 | 92.7 | 91.2 | 90.9 | 90.7 | 90.5 |
| OpenAI | Cohere | 94.5 | 92.3 | 92.3 | 92.2 | 92.0 | 92.1 | 92.1 | 92.2 |
| Mistral | Qwen | 95.5 | 92.8 | 92.7 | 92.9 | 90.7 | 89.8 | 90.1 | 90.6 |
| Qwen | OpenAI | 94.5 | 92.7 | 93.0 | 92.9 | 93.2 | 93.4 | 93.0 | 93.4 |
| TREC (17%) |  |  |  |  |  |  |  |  |  |
| Gemini | OpenAI | 95.4 | 47.0 | 56.6 | 77.6 | 57.6 | 71.8 | 78.0 | 81.0 |
| OpenAI | Qwen | 96.4 | 48.0 | 59.0 | 82.8 | 58.4 | 81.6 | 77.6 | 83.6 |
| Mistral | Cohere | 96.6 | 47.8 | 52.0 | 73.2 | 55.8 | 71.6 | 74.4 | 78.4 |
| Gemini | Mistral | 95.4 | 61.0 | 77.4 | 84.6 | 56.8 | 80.4 | 82.0 | 81.6 |
| Cohere | Gemini | 96.0 | 38.8 | 46.6 | 65.2 | 63.4 | 73.6 | 74.2 | 76.8 |
| OpenAI | Cohere | 96.4 | 48.2 | 52.2 | 73.4 | 68.4 | 75.8 | 80.2 | 80.6 |
| Mistral | Qwen | 96.6 | 48.2 | 59.4 | 82.2 | 65.6 | 79.4 | 83.8 | 87.4 |
| Qwen | OpenAI | 97.0 | 47.2 | 57.0 | 77.8 | 75.4 | 80.0 | 82.8 | 87.6 |
| MNLI (33%) |  |  |  |  |  |  |  |  |  |
| Gemini | OpenAI | 65.0 | 32.7 | 33.5 | 35.1 | 44.5 | 48.2 | 48.5 | 48.9 |
| OpenAI | Qwen | 62.4 | 55.0 | 57.7 | 61.9 | 64.0 | 64.6 | 64.9 | 65.2 |
| Mistral | Cohere | 77.6 | 33.2 | 34.4 | 35.8 | 39.5 | 41.2 | 41.9 | 42.8 |
| Gemini | Mistral | 65.0 | 41.5 | 45.0 | 48.7 | 51.9 | 57.5 | 58.3 | 58.3 |
| Cohere | Gemini | 59.1 | 34.2 | 34.7 | 36.6 | 44.9 | 46.9 | 47.2 | 47.6 |
| OpenAI | Cohere | 62.4 | 33.3 | 34.4 | 35.8 | 46.5 | 46.5 | 47.1 | 47.9 |
| Mistral | Qwen | 77.6 | 54.9 | 57.6 | 61.8 | 58.8 | 69.0 | 71.3 | 72.9 |
| Qwen | OpenAI | 87.9 | 32.7 | 33.4 | 35.2 | 42.0 | 43.6 | 42.9 | 43.9 |

## Appendix G HELIX Out-of-Distribution Detection

Beyond preserving the classification accuracy of a given dataset, we next test whether HELIX preserves Party A’s understanding of the underlying data distribution such that Party B can still distinguish in-distribution from out-of-distribution samples within its own embedding space. Specifically, given a linear classifier $f(\cdot)$ trained on a proprietary dataset by Party A, we test whether HELIX is able to retain the underlying uncertainty signals present in model $f(\cdot)$. This allows us to assess whether the method preserves robustness under distribution shift rather than only matching in-distribution accuracy.

Common methods for uncertainty measurement rely on the model’s output logits, such as Maximum Softmax Probability (MSP) Hendrycks and Gimpel (2016) and energy-based scoring Liu and others (2021). In this section, we evaluate our results using an energy-based score, which captures the unnormalized confidence of the model:

$$ $E(x)=-\log\sum_{k=1}^{K}\exp(f_{k}(x)).$ $$

where $k$ is the number of classes in the classification model. Higher energy values indicate lower model confidence. We evaluate our results using AUROC and FP@95.

Experimental Setup We train the dataowner classifier on an in-distribution (ID) dataset and evaluate OOD detection on a held-out OOD dataset (AG News or MNLI respectively). We compare two approaches: (1) Baseline: dataowner classifier applied directly to dataowner embeddings, and (2) HELIX: dataowner classifier applied to linearly mapped client embeddings, where the mapping is trained on IMDB as the public dataset. We use Energy scores Liu and others (2021) for OOD detection and report AUROC averaged across five model pairs.

**Table 7: OOD detection results reported side-by-side across two target datasets (SST-2 and TREC). Each row corresponds to a target–source embedding model pair, evaluated using a baseline target-space classifier and the HELIX linear alignment.**
| Party A | Party B | OOD | AUROC | OOD | AUROC |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Baseline | HELIX | Dataset | Baseline | HELIX |  |  |
|  | SST-2 | TREC |  |  |  |  |  |
| OpenAI | Gemini | AGNews | 0.826 | 0.774 | AGNews | 0.954 | 0.721 |
| Qwen | OpenAI | 0.859 | 0.822 | 0.921 | 0.731 |  |  |
| Gemini | Cohere | 0.819 | 0.851 | 0.412 | 0.433 |  |  |
| Cohere | OpenAI | 0.875 | 0.813 | 0.738 | 0.790 |  |  |
| Qwen | Mistral | 0.859 | 0.818 | 0.921 | 0.725 |  |  |
|  | AGNews | DBpedia |  |  |  |  |  |
| OpenAI | Gemini | MNLI | 0.911 | 0.776 | Yahoo | 0.969 | 0.656 |
| Qwen | OpenAI | 0.956 | 0.716 | 0.980 | 0.715 |  |  |
| Gemini | Cohere | 0.880 | 0.805 | 0.973 | 0.589 |  |  |
| Cohere | OpenAI | 0.894 | 0.715 | 0.967 | 0.538 |  |  |
| Qwen | Mistral | 0.956 | 0.838 | 0.955 | 0.657 |  |  |

Figure: Figure 13: OOD Detection: Dataowner classifier applied to dataowner embeddings (baseline, blue) versus mapped client embeddings (HELIX, orange). Linear mapping trained on IMDB; results averaged over five model pairs.
Refer to caption: https://arxiv.org/html/2603.18908/2603.18908v4/x13.png

## Appendix H Extended Privacy Framework

This appendix provides an expanded version of the privacy-preserving execution model
underlying HELIX, including the entity definitions, cryptographic assumptions,
threat model, and protocol details. We focus on a two-party, cross-silo setting
where both parties wish to enable downstream inference while preserving
confidentiality of their private data and proprietary models.

### H.1 Problem Formulation and Entities

We consider a cross-silo inference scenario involving two parties:

Party A (Service Provider).
Party A owns a proprietary classification model for a task of interest.
The model consists of a private encoder
$g_{A}:\mathcal{X}\to\mathbb{R}^{d_{A}}$ and a task head
$f_{A}:\mathbb{R}^{d_{A}}\to\mathcal{Y}$ trained on embeddings
from $g_{A}$ using private training data $\mathcal{D}_{\text{priv}}$.
In this work, we focus on the case where $f_{A}$ is linear, i.e.,
$f_{A}(z)=zV+c$, where $(V,c)$ are Party A’s proprietary classifier parameters.
Party A seeks to monetize inference access while keeping both model parameters
and training data confidential.

Party B (Client).
Party B owns an independent embedding model
$g_{B}:\mathcal{X}\to\mathbb{R}^{d_{B}}$ trained on sensitive or domain-specific data.
Party B wishes to obtain predictions from Party A’s classifier
on private query inputs $x\in\mathcal{X}$, without revealing raw inputs,
query embeddings $g_{B}(x)$, or encoder parameters to Party A.
Moreover, Party B does not have access to the internal weights of
Party A’s model and cannot run the classifier locally.

Goal.
Our objective is to learn an affine map $(W^{*},b^{*})$ such that
$g_{B}(x)$ can be mapped into Party A’s representation space, enabling
secure inference through Party A’s classifier:

$$ $\hat{z}_{A}=z_{B}W^{*}+b^{*},\qquad\hat{y}=f_{A}(\hat{z}_{A}),$ $$

while protecting the confidentiality of both parties.

### H.2 Encrypted Computation Model

Our protocol uses homomorphic encryption (HE) to enable computations directly
on encrypted vectors without revealing plaintext inputs. We adopt CKKS
Cheon et al. (2017), an approximate HE scheme for real-valued arithmetic,
as it supports efficient evaluation of low-depth linear operations.
This design choice aligns with prior work showing that linear components
are most practical for encrypted training and inference under modern HE
constraints Gilad-Bachrach et al. (2016a); Mohassel and Zhang (2017); Juvekar et al. (2018); Lee et al. (2023).

We restrict secure computation to linear operations (matrix multiplication and addition),
since nonlinearities typically require bootstrapping or polynomial approximation,
which is computationally prohibitive in many deployment regimes.

Let $\mathsf{Enc}_{pk}(\cdot)$ and $\mathsf{Dec}_{sk}(\cdot)$ denote encryption
and decryption under a public/secret key pair $(pk,sk)$.
The HE scheme must support:
(i) ciphertext–plaintext multiplication and (ii) ciphertext–ciphertext addition.
In our protocol, we avoid expensive ciphertext–ciphertext multiplication during
training by structuring secure aggregation so that only one operand is encrypted.

### H.3 Threat Model and Security Objectives

We adopt a mutually distrustful, semi-honest (honest-but-curious)
threat model Goldreich (2004). Both parties follow the protocol
specification but may attempt to infer private information from observed messages.

We do not consider malicious adversaries who arbitrarily deviate from the protocol
(e.g., injecting malformed ciphertexts or performing active attacks).

Our security priorities are client-centric, while still protecting the provider’s model:

- •
Client query privacy.
Party A should not learn Party B’s query inputs $x$ or
query embeddings $z_{B}=g_{B}(x)$ during inference.
- •
Client model confidentiality.
Party A should not learn the parameters of $g_{B}$ or additional
information about Party B’s training data beyond what is implied
by task outputs.
- •
Provider classifier confidentiality.
Party B should not obtain Party A’s classifier parameters $(V,c)$.

### H.4 Public Data Assumption

We assume both parties have access to a shared, non-sensitive public dataset
$\mathcal{D}_{\text{pub}}=\{x_{i}\}_{i=1}^{N}$, used only for fitting the alignment map.
When learned exclusively from $\mathcal{D}_{\text{pub}}$, the alignment reflects
public distributional structure rather than either party’s private training data.

Optionally, Party A may include a small number of in-distribution plaintext
examples from $\mathcal{D}_{\text{priv}}$ (e.g., 64–128 samples) to improve alignment
quality. This introduces a tunable privacy–utility tradeoff by revealing limited
task-specific information to Party B.

### H.5 Linear Alignment Objective

Given public data embeddings
$Z_{A}=g_{A}(\mathcal{D}_{\text{pub}})\in\mathbb{R}^{N\times d_{A}}$
and
$Z_{B}=g_{B}(\mathcal{D}_{\text{pub}})\in\mathbb{R}^{N\times d_{B}}$,
we learn an affine alignment via ridge regression:

$$ $\min_{W,b}\;\|Z_{B}W+\mathbf{1}b^{\top}-Z_{A}\|_{F}^{2}+\lambda\|W\|_{F}^{2}.$ $$

The closed-form solution for the linear map is:

$$ $W^{*}=(Z_{B}^{\top}Z_{B}+\lambda I)^{-1}Z_{B}^{\top}Z_{A}.$ (2) $$

In practice, we avoid materializing full embedding matrices by computing the
sufficient statistics $Z_{B}^{\top}Z_{B}$ and $Z_{B}^{\top}Z_{A}$ via streaming mini-batches.

### H.6 Two-Party Secure Training Protocol

The training protocol computes the cross-covariance term in
Eq. ([2](#A8.E2)) without requiring Party B to reveal $Z_{B}$
or Party A to reveal $Z_{A}$ in plaintext.

##### Key ownership.

During training, Party B generates an HE keypair $(pk,sk)$ and retains
the secret key. This ensures that Party A never decrypts any client
representations.

The protocol proceeds as follows:

- 1.
Embedding extraction.
Both parties compute embeddings on the public dataset:
$Z_{A}=g_{A}(\mathcal{D}_{\text{pub}}),\qquad Z_{B}=g_{B}(\mathcal{D}_{\text{pub}}).$
- 2.
Client encryption and transmission.
Party B encrypts its embedding matrix under $pk$ and sends
$\mathsf{Enc}_{pk}(Z_{B})$ to Party A.
- 3.
Encrypted cross-covariance computation.
Party A computes the encrypted cross-covariance
$\mathsf{Enc}(Z_{A}^{\top}Z_{B})$ using plaintext $Z_{A}$ and encrypted $Z_{B}$.
This can be implemented as homomorphic linear aggregation over samples:
$\mathsf{Enc}(Z_{A}^{\top}Z_{B})=\sum_{k=1}^{N}Z_{A}[k,:]^{\top}\cdot\mathsf{Enc}(Z_{B}[k,:]).$
Party A returns $\mathsf{Enc}(Z_{A}^{\top}Z_{B})$ to Party B.
- 4.
Decryption and solving.
Party B decrypts to obtain $Z_{A}^{\top}Z_{B}$ in plaintext and transposes it
to form $Z_{B}^{\top}Z_{A}$, then computes
$W^{*}=(Z_{B}^{\top}Z_{B}+\lambda I)^{-1}Z_{B}^{\top}Z_{A}$
locally using their plaintext $Z_{B}$.

##### Deployment of W ∗ W^{*} .

Unlike traditional outsourced training schemes, Party B retains the learned
map $(W^{*},b^{*})$ and uses it locally during inference. Party A never obtains the
alignment map in plaintext.

### H.7 Privacy-Preserving Inference

Inference uses a fresh inference keypair $(pk_{I},sk_{I})$ generated by Party B.
Party B retains $sk_{I}$ and provides $pk_{I}$ to Party A.

Assuming $f_{A}(z)=zV+c$ is linear, inference proceeds:

- 1.
Local alignment at the client.
Party B computes $z_{B}=g_{B}(x)$ and applies the affine map locally:
$\hat{z}_{A}=z_{B}W^{*}+b^{*}.$
- 2.
Encrypt aligned embedding.
Party B encrypts $\hat{z}_{A}$ and sends $\mathsf{Enc}_{pk_{I}}(\hat{z}_{A})$
to Party A.
- 3.
Homomorphic classification.
Party A evaluates the classifier on encrypted inputs:
$\mathsf{Enc}(\hat{y})=\mathsf{Enc}(\hat{z}_{A})\cdot V+c.$
- 4.
Return and decrypt.
Party A returns the encrypted prediction, which Party B decrypts:
$\hat{y}=\mathsf{Dec}_{sk_{I}}(\mathsf{Enc}(\hat{y})).$

##### Argmax-only outputs.

To reduce leakage about $(V,c)$ through black-box queries, the protocol may return
only a predicted class label via encrypted argmax rather than full logits. This limits per-query information leakage and provides practical defense against model
extraction and membership inference attacks Tramèr et al. (2016); Carlini et al. (2021). While we do not implement it in this paper, others, such as Phoenix Jovanovic et al. (2022) and Nexus Zhang et al. (2024a) have implemented solutions for this.

### H.8 Threat Analysis and Limitations

We analyze security under the semi-honest model.

##### Client query privacy.

During inference, Party A observes only CKKS ciphertexts of aligned embeddings
$\mathsf{Enc}(\hat{z}_{A})$ and encrypted outputs. Under the semantic security of CKKS,
these ciphertexts reveal no information about $x$ or $z_{B}=g_{B}(x)$ beyond what is implied
by the decrypted prediction.

##### Provider classifier privacy.

Party B never receives the classifier parameters $(V,c)$ in plaintext.
Moreover, returning only encrypted class labels (rather than logits) reduces the
attack surface for model extraction and membership inference, though it does not
eliminate all leakage under adaptive querying.

##### Visibility of the alignment map.

Party B retains $(W^{*},b^{*})$ in plaintext. The learned map reveals structural
properties of Party A’s embedding space (e.g., $d_{A}$ and some geometric
relationships) but does not directly expose Party A’s classifier parameters
or private training data. However, $W^{*}$ may enable property inference about aspects
of Party A’s representation geometry, and could potentially facilitate
adaptive attacks when combined with repeated inference queries. Formal quantification
of leakage through $W^{*}$ remains an important direction for future work, e.g., via
differential privacy mechanisms Chaudhuri et al. (2011).

##### Structural and metadata leakage.

As in most HE deployments, certain information is revealed: tensor shapes, embedding
dimensions, communication volume, and sample count $N$. We do not attempt to hide
access patterns or protocol metadata.

##### Out-of-scope adversaries.

We do not consider malicious behaviors such as malformed ciphertext injection,
chosen-ciphertext attacks, denial-of-service, or protocol deviations. Extending to
the malicious setting would require additional safeguards (e.g., ciphertext validity
checks, zero-knowledge proofs), which are orthogonal to the core alignment mechanism.

##### Summary.

Under the stated assumptions, HELIX enables two-party alignment and secure inference
with strong client-side query confidentiality.
Remaining leakage is limited to unavoidable structural metadata and potential risks under
adaptive repeated querying, consistent with known limitations of black-box and encrypted
ML services Tramèr et al. (2016); Wu et al. (2024); Carlini et al. (2024).

Figure: Algorithm 1 Two-Party Secure Training for Linear Alignment (Ridge Regression)

Figure: Algorithm 2 Privacy-Preserving Inference via Encrypted Alignment and Linear Head

## Appendix I Extended Related Works

In this section we include additional research related to security and machine learning.

Representational Similarity
Complementary work on loss landscapes shows that overparameterized models often converge to functionally equivalent solutions up to symmetry transformations, despite large parameter-space variation Entezari et al. (2021); Ainsworth et al. (2022). Methods for measuring representational similarity across neural networks include Canonical Correlation Analysis (CCA) Raghu et al. (2017), centered kernel alignment (CKA) Kornblith et al. (2019), and linear probing Alain and Bengio (2018). Recent work has explored the geometry of neural network representations Huh et al. (2024), finding evidence of convergence toward shared representational structures across architectures and training procedures Li et al. (2015). Studies on neural network stitching Bansal et al. (2021) demonstrate that layers from independently trained models can sometimes be connected via simple transformations, supporting the hypothesis that models learn aligned feature spaces.

Transfer Learning
Transfer learning enables models pretrained on large corpora to adapt to new tasks with limited data through feature-based transfer Pan and Yang (2010) or fine-tuning Devlin et al. (2018). Knowledge distillation Hinton et al. (2015) trains compact student models to match teacher predictions, with variants including task-specific Kim and Rush (2016), self-distillation Furlanello et al. (2018), and representation-level distillation Romero et al. (2015); Tian et al. (2020).

Federated Learning.
Federated Learning (FL) trains a shared model over distributed data without centralizing raw samples.
The foundational FedAvg algorithm McMahan et al. (2017) demonstrated communication-efficient decentralized optimization across non-IID devices.
Follow-up work studied convergence and personalization under heterogeneous settings Kairouz et al. (2021); Smith et al. (2017); Li et al. (2020) and secure aggregation of gradients et al. (2017).
Extensions such as FedNova Wang et al. (2020) and SCAFFOLD Karimireddy et al. (2020) addressed client drift and variance reduction, while hierarchical FL Liu et al. (2022) scales training across organizational silos.

Federated Transfer Learning (FTL) extends FL to cross-domain collaboration when participants have little or no overlap in feature or sample space.
Liu et al. Liu et al. (2020) proposed a secure FTL framework that uses encrypted intermediate representations and gradient sharing between a label-rich source and a label-scarce target.
Recent directions combine FTL with knowledge distillation Mora et al. (2022), providing secure cross-silo knowledge transfer.

Split Learning.
Split Learning (SL) partitions a model between clients and a central server, exchanging only cut-layer activations and gradients.
Vepakomma et al. Vepakomma et al. (2019) first proposed this approach for healthcare, demonstrating collaborative deep learning without exposing raw data.

Homomorphic Encryption for Privacy-Preserving ML.
Privacy-preserving machine learning builds on cryptographic primitives such as HE and Secure Multi-Party Computation (MPC). Gentry’s seminal work introduced fully homomorphic encryption, enabling arbitrary computation over encrypted data Gentry (2009). Subsequent schemes such as BGV Brakerski et al. (2014) and CKKS Cheon et al. (2017) improved efficiency for real-valued arithmetic, enabling practical machine learning applications.

HE-Based Inference. CryptoNets Gilad-Bachrach et al. (2016b) demonstrated the feasibility of HE-based neural network inference with partially protected model parameters. Hybrid approaches combining HE with secure two-party computation (2PC) emerged to balance security and efficiency: Gazelle Juvekar et al. (2018) and Phoenix Jovanovic et al. (2022) secure both client inputs and model weights in interactive protocols. More recent systems achieve non-interactive inference: Nexus Zhang et al. (2024a) enables secure transformer inference using HE alone, while Powerformer Park et al. (2024) optimizes HE evaluation for speed.

MPC-Based Inference.
Pure MPC approaches offer alternatives to HE-based methods. MPCFormer Li et al. (2023) evaluates transformers securely using MPC protocols, while Iron Hao et al. (2022) combines HE and MPC for exact-accuracy transformer inference. BOLT Pang et al. (2024) and Nimbus Li et al. (2024) accelerate secure inference through optimized MPC protocols. SecFormer Luo et al. (2024) reduces communication costs using low-degree polynomial approximations for transformer activations.

Frameworks and Systems.
General-purpose frameworks have made privacy-preserving ML more accessible. SecureML Mohassel and Zhang (2017) introduced efficient 2-party training for linear models using secret sharing and HE. CrypTen Knott et al. (2021) and PySyft et al. (2018) provide higher-level abstractions with automatic differentiation and GPU support.

Encrypted Adaptation and Architecture Design.
Recent work explores encrypted transfer learning and HE-friendly architectures. HETAL Lee et al. (2023) enables encrypted transfer learning by training classification heads on encrypted features from fixed public encoders. The Encryption-Friendly LLM architecture Zhang et al. (2025) modifies transformers with polynomial operations to support efficient HE-based inference and private fine-tuning.

Positioning.
These methods primarily target secure inference or encrypted fine-tuning where both parties’ data/models require cryptographic protection. In contrast, our work addresses a different threat model: enabling privacy-preserving inference when the client has encrypted representations but the provider’s classifier can be black-box accessible, leveraging cross-model linear alignment to eliminate interactive protocols and reduce computational overhead.

## Appendix J Security Benchmark Comparison: Inference Time and Communication

In Table [9](#A10.T9) we report inference time and communcation cost of HELIX with previous methods.

Comparison setup.
We compare HELIX against prior work in privacy-preserving Transformer inference, including MPC-only approaches (MPCFormer), hybrid HE+MPC protocols (Iron, BOLT, Nimbus, SecFormer), and non-interactive HE systems (PowerFormer, NEXUS, and an encryption-friendly Transformer architecture). Since these methods differ in threat model, cryptographic primitives, and evaluation stacks, our goal is not a perfectly controlled head-to-head benchmark, but rather a practical reference point for accuracy, inference latency, and communication overhead reported in the literature.

Inference time extraction.
For each baseline, we report inference time using the values provided in the corresponding paper under their default BERT-base evaluation setting when available. When papers report multiple configurations (e.g., LAN vs. WAN, CPU vs. GPU, different sequence lengths or batch sizes), we use the authors’ primary end-to-end inference numbers and preserve their reporting granularity (per example or per batch) as stated.

Inference communication cost.
We report the inference-time communication cost (total bytes exchanged between parties) for each secure inference method when explicitly provided by the original paper. For MPCFormer, the authors quantify that standard MPC-based BERT inference requires 68.6 GB of communication per query (BERT-base, 512 tokens). For the HE+MPC hybrid baselines, BOLT reports a total inference communication of 25.74 GB for BERT-base, and additionally notes that their reimplementation of Iron incurs 280.99 GB under the same benchmark setting. For Nimbus, the paper reports communication at the operator level, including 115.35 MB for Softmax and 53.22 MB for GELU (per Transformer block). For NEXUS, the authors report a non-interactive HE protocol requiring 164 MB of total bandwidth for BERT-base inference. For PowerFormer and the encryption-friendly architecture work, the protocols are non-interactive HE (ciphertext upload and download only), but the papers do not provide a single end-to-end inference communication size in bytes, so we do not list a numeric value for those methods.
HELIX Benchmarks.
In Table [9](#A10.T9), we report HELIX in three scenarios: 1) using base embedding models (Gemini, OpenAI, Cohere) we train a linear map between each using the IMDB dataset, and report results on a downstream linear classifier for each GLUE dataset under test, 2) We use an in-distribution mapper - training the mapper using the GLUE dataset under test, and evaluating the linear classifier on the test set, and 3) We fine-tune Llama-2-8b on each dataset under test, and train a mapper between a Llama-2-8b (non-finetuned) and Llama-2-8b (fine-tuned) using the IMDB dataset.

**Table 8: Comparison of privacy-preserving ML systems. ✓ denotes cryptographic protection; $\triangle$ partial protection; ✗ not protected or out of scope. Provider Model (Party A) indicates cryptographic protection of the provider’s model parameters against disclosure to the client (black-box access permitted). Interact.: Offline methods rely on non-interactive HE evaluation; Online methods require MPC/2PC-style interaction. HELIX protects client inputs via HE but uses public/black-box access to the provider’s model head, learning only a linear mapping.**
| Method | Goal | Threat | Client Inputs<br>(Party B) | Provider Model<br>(Party A) | Task | Interact. | Crypto |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [CryptoNets (ICML’16)](https://proceedings.mlr.press/v48/gilad-bachrach16.pdf) | HE inference | HBC | ✓ | $\triangle$ | Inf. | Offline | HE |
| [Gazelle (USENIX’18)](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-juvekar.pdf) | Fast inference | HBC | ✓ | ✓ | Inf. | Online | HE+2PC |
| [Phoenix (CCS’22)](https://files.sri.inf.ethz.ch/website/papers/ccs22-phoenix.pdf) | Reliable inference | HBC | ✓ | ✓ | Inf. | Online | HE+2PC |
| [MPCFormer (ICLR’23)](https://openreview.net/forum?id=CWmvjOEhgH-) | MPC Transformer inference | HBC | ✓ | ✓ | Inf. | Online | MPC |
| [Iron (NeurIPS’22)](https://proceedings.neurips.cc/paper_files/paper/2022/file/64e2449d74f84e5b1a5c96ba7b3d308e-Paper-Conference.pdf) | Private Transformer inference | HBC | ✓ | ✓ | Inf. | Online | HE+MPC |
| [BOLT (SP’24)](https://eprint.iacr.org/2023/1893.pdf) | Fast private inference | HBC | ✓ | ✓ | Inf. | Online | MPC |
| [Nimbus (NeurIPS’24)](https://proceedings.neurips.cc/paper_files/paper/2024/file/264a9b3ce46abdf572dcfe0401141989-Paper-Conference.pdf) | Efficient Transformer inference | HBC | ✓ | ✓ | Inf. | Online | MPC |
| [SecFormer (ACL’24)](https://arxiv.org/abs/2401.00793) | Secure Transformer inference | HBC | ✓ | ✓ | Inf. | Online | MPC |
| [HETAL (ICML’24)](https://arxiv.org/abs/2403.14111) | HE transfer learning | HBC | ✓ | ✗ | Inf.+FT | Online | HE |
| [Powerformer (ePrint’24)](https://eprint.iacr.org/2024/1429.pdf) | Faster HE inference | HBC | ✓ | ✓ | Inf. | Offline | HE |
| [NEXUS (NDSS’25)](https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf) | Non-interactive secure inference | HBC | ✓ | ✓ | Inf. | Offline | HE |
| [Enc.-Friendly LLM Architecture (ICLR’25)](https://arxiv.org/pdf/2410.02486.pdf) | HE-friendly architecture | HBC | ✓ | ✗ | Inf.+FT | Offline | HE |
| HELIX (ours) | Cross-model transfer | HBC | ✓ | ✗ | Inf. | Offline | HE |

**Table 9: Accuracy and inference-time comparison on GLUE tasks. Inference time is reported per example. For HELIX, we average five runs over different random model pairs. Inference times are reported per batch. Public data mapper indicates that the model provider’s data is protected as well. HELIX results are without fine-tuning. HELIX was assessed using Gemini, OpenAI, and Cohere (indicated as Model Pairs*).**
| Method | Model(s) | Security Scope | SST-2 | STS-B (Pearson) | MRPC (F1) | RTE | Inf. Time (s) | Inf. Comm. Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | BERT | None | 92.3 | 89.1 | 90.3 | 69.7 | $<1$ | – |
| Iron | BERT | Full (HE+MPC) | 92.8 | 89.4 | 89.9 | 70.8 | $>60$ | 280.99 GB |
| BOLT | BERT | Full (HE+MPC) | 92.8 | 88.4 | 90.0 | 69.3 | $>60$ | 25.74 GB |
| SecFormer | BERT | Full (HE+MPC) | – | 87.4 | 89.2 | 69.0 | 19 | – |
| MPCFormer | BERT | Full (MPC) | – | 80.3 | 88.7 | 64.9 | 18 | 68.6 GB |
| Enc.-Friendly Arch. | BERT | Full (HE) | 81.9 | 80.0 | 81.5 | 59.3 | 26.5 | input + output |
| Nimbus | BERT | Full (HE+MPC) | 92.6 | 87.9 | 89.8 | 66.8 | $>20$ | $>$ 2GB |
| PowerFormer | BERT | Full (HE) | 92.0 | - | 87.8 | 69.8 | $>20$ | input + output |
| Nexus | BERT | Full (HE) | 92.1 | - | - | 69.9 | $37.3$ | 164 MB |
| Nexus | Llama-3-8B | Full (HE) | 94.5 | - | - | 81.2 | - | – |
| HELIX (IMDB Mapper, no FT) | Model Pairs* | Linear (HE) | 92.3 | 61.0 | 77.8 | 59.6 | $<1$ | $<1$ MB |
| HELIX (In-dist Mapper, no FT) | Model Pairs* | Linear (HE) | 92.8 | 75.6 | 80.2 | 61.0 | $<1$ | $<1$ MB |
| HELIX (Fine-Tuned) | Llama-2-8b | Linear (HE) | 93.0 | 80.6 | 82.0 | 55.6 | $<1$ | $<1$ MB |

## Appendix K Privacy Analysis: Membership Inference on W ∗ W^{*}

When Party A augments public mapper training with private samples (Section 5.1, Setting 2), those samples are shared in plaintext with Party B, violating data confidentiality. However, a secondary privacy question remains: does the resulting $W^{*}$ leak which specific samples were included in the shared set?

This matters if Party A carefully curates the shared samples—for example, sharing only non-sensitive examples while withholding particularly private ones. If $W^{*}$ encodes detectable membership signals, an adversary could infer whether a specific sensitive sample was included, compromising Party A’s curation strategy.

### K.1 Attack Methodology

We implement a shadow-mapper membership inference attack. Given a target sample from SST-2, we train 200 shadow mappers (100 IN, 100 OUT) under two conditions:

- •
IN: Mapper trained on 100K Wikipedia + 128 SST-2 samples including target
- •
OUT: Mapper trained on 100K Wikipedia + 128 SST-2 samples excluding target

For each $W^{*}$, we extract geometric features: Frobenius and spectral norms, row/column statistics, top 64 singular values, effective rank, and bias norms. We train a logistic regression classifier to predict IN vs OUT via 5-fold stratified cross-validation.

Configuration: SST-2, Party A = Gemini, Party B = OpenAI, target = train index 0.

### K.2 Results

The membership classifier achieves 0.530 ± 0.113 accuracy (chance = 0.500), performing at chance level with substantial variance across folds.

### K.3 Theoretical Hardness Analysis

We provide a theoretical bound on the difficulty of membership inference from $W^{*}$.

Intuition: When $W^{*}$ is learned from $N$ samples via ridge regression, adding or removing a single sample has only $O(1/N)$ influence on the result. With $N\approx 67{,}000$ samples in our experiments, any individual sample contributes less than $0.002\%$ to the final mapping. This tiny influence makes it fundamentally difficult to detect whether a specific sample was included.

Theoretical Bound: For embeddings with dimensions $d_{A}\times d_{B}$, the maximum advantage any membership inference classifier can achieve over random guessing is bounded by:

$$ $\text{Advantage}\leq O\left(\frac{\sqrt{d_{A}\cdot d_{B}}}{N}\right)$ (3) $$

This bound follows from standard sensitivity analysis of ridge regression: the Frobenius norm difference $\|W^{*}-W^{*}_{-i}\|_{F}$ between a mapper trained with and without sample $i$ is $O(1/\sqrt{N})$. Since our geometric features have $O(d_{A}\cdot d_{B})$ dimensions, the signal-to-noise ratio scales as $\sqrt{d_{A}\cdot d_{B}}/N$.

Empirical Validation: For our configuration ($d_{A}=1024$, $d_{B}=1152$, $N=67{,}000$), the theoretical bound predicts maximum accuracy $\approx 0.516$. Our experimental result of $0.530\pm 0.113$ has confidence intervals overlapping this bound, confirming the theoretical prediction. The high variance ($\sigma=0.113$) across folds indicates the classifier cannot reliably distinguish IN vs OUT, consistent with the signal being at the noise floor.

### K.4 Interpretation

The membership classifier achieves $0.530\pm 0.113$ accuracy (chance = 0.500), performing near chance level with substantial variance across folds. This negative result is theoretically expected: with $\sim$67K total samples, the per-sample influence on $W^{*}$ is $\sim 1.5\times 10^{-5}$, far below the detection threshold of classifiers operating on geometric features.

Privacy implication: Even when Party A shares limited private data (Setting 2), $W^{*}$ does not leak fine-grained membership information. The theoretical bound guarantees that membership inference advantage is $O(\sqrt{d}/N)\approx 0.016$, yielding negligible privacy risk for large $N$. An adversary analyzing $W^{*}$ cannot determine which specific samples Party A included in the shared set, preserving Party A’s curation strategy.

Limitations. This analysis assumes geometric features and tests a single target sample. Stronger adversaries with access to many $W^{*}$ samples or side information about Party A’s data distribution might achieve higher accuracy. However, the fundamental $O(1/N)$ influence bound still applies, limiting the maximum achievable advantage.

## Appendix L Model Architecture Information

**Table 10: All models used in the text generation experiments.**
| Model | Multimodal? | Q,K RMSNorm? | Embedding Dim |
| --- | --- | --- | --- |
| allenai_Olmo_3_7B_Instruct | No | No | 4096 |
| google_gemma_3_12b_it | Yes | Yes | 3072 |
| google_gemma_3_270m_it | No | Yes | 1536 |
| google_gemma_3_27b_it | Yes | Yes | 4608 |
| meta_llama_Llama_3.2_1B_Instruct | No | No | 2048 |
| meta_llama_Meta_Llama_3_8B_Instruct | No | No | 4096 |
| mistralai_Ministral_3_14B_Instruct_2512 | Yes | No | 5120 |
| mistralai_Mistral_7B_Instruct_v0.2 | No | No | 4096 |
| mistralai_Mistral_Nemo_Instruct_2407 | No | No | 5120 |
| Qwen_Qwen2.5_0.5B_Instruct | No | No | 896 |
| Qwen_Qwen2.5_14B_Instruct | No | No | 5120 |
| Qwen_Qwen2.5_32B_Instruct | No | No | 5120 |
| Qwen_Qwen2.5_7B_Instruct | No | No | 3584 |
| Qwen_Qwen3_30B_A3B_Instruct_2507 | No | Yes | 2048 |
| Qwen_Qwen3_4B_Instruct_2507 | No | Yes | 2560 |
| swiss_ai_Apertus_8B_Instruct_2509 | No | Yes | 4096 |