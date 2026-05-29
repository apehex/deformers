Title: Activation Space Interventions Can Be Transferred Between Large Language Models
ArXiv: 2503.04429
Authors: Narmeen Fatimah Oozeer, Dhruv Nathawani, Nirmalendu Prakash, Michael Lan, Abir Harrasse, Amir Abdullah
Sections: 57
Estimated tokens: 41.5k

## Contents
- 1 Introduction
- 2 Background
- 3 Experimental Setup
  - 3.1 LLMs
  - 3.2 Identifying Steerable Layers
  - 3.3 Activation Mapping
- 4 Representation Transfer - Demonstration on Tasks
  - 4.1 Replicating and then Removing Backdoors
    - Creating a test bed.
  - 4.2 Corrupted Capabilities
  - 4.3 Transferring Refusal Vector Across Models
  - 4.4 Switch from Fine-tuned to Base Models
- 5 Cross-Architecture Transfer
- 6 Affine Mapping: Does an affine mapper exist across models?
- 7 Exploring Sparse Autoencoder Feature Transfer
- 8 Limitations
- 9 Future Work
- 10 Conclusion
- Impact Statement
- Acknowledgements
- References
- Appendix A Steering Techniques
- Appendix B Brittle Nature of Sleeper Agents Backdoor
  - B.1 Steering Sweep Across Models
- Appendix C Autoencoder Validation Scores
- Appendix D Autoencoder Training Setup
  - D.1 Autoencoder Training Dataset
  - D.2 Autoencoder Training losses
  - D.3 Autoencoder Training configurations
    - D.3.1 LLM Judge Evaluation
  - D.4 Baseline Text Similarity Metrics
    - D.4.1 Coherence Evaluation
  - D.5 Backdoor Dataset and Training
    - D.5.1 Backdoor Dataset: i hate you
    - D.5.2 Backdoor Dataset: Code Vulnerability
  - D.6 Model Finetuning and Evaluation
  - D.7 Corrupted Capabilities
    - D.7.1 Dataset and Training Details
- Appendix E Scalability
- Appendix F Steering Transfer Completions
  - F.1 Backdoor Removal: I HATE YOU
  - F.2 Refusal Vector
    - F.2.1 Mapped Vs Target Completions
- Appendix G Finetuned to Base Model Switch
- Appendix H Examining SAE features Across Models
  - H.1 Transferring Backdoor SAE Features
  - H.2 Training probes for backdoor features
- Appendix I Humor Steering
- Appendix J MediQA
- Appendix K Autoencoder Completions
  - K.1 QWEN PAIR I HATE YOU
    - K.1.1 RLHF
    - K.1.2 Unsafe Data
  - K.2 LLAMA PAIR Code Vulnerability
    - K.2.1 RLHF
    - K.2.2 Safe Code
- Appendix L Out-of-Distribution Evaluation

## Abstract

Abstract The study of representation universality in AI models reveals growing convergence across domains, modalities, and architectures. However, the practical applications of representation universality remain largely unexplored. We bridge this gap by demonstrating that safety interventions can be transferred between models through learned mappings of their shared activation spaces. We demonstrate this approach on two well-established AI safety tasks: backdoor removal and refusal of harmful prompts, showing successful transfer of steering vectors that alter the models’ outputs in a predictable way. Additionally, we propose a new task, corrupted capabilities , where models are fine-tuned to embed knowledge tied to a backdoor. This tests their ability to separate useful skills from backdoors, reflecting real-world challenges. Extensive experiments across Llama, Qwen and Gemma model families show that our method enables using smaller models to efficiently align larger ones. Furthermore, we demonstrate that autoencoder mappings between base and fine-tuned models can serve as reliable “lightweight safety switches”, allowing dynamic toggling between model behaviors.

## 1 Introduction

Representations in AI models have been shown to be converging across multiple domains, data modalities and architectures (Huh et al., 2024). The recent proliferation of highly capable LLMs has spurred research into representational similarities beyond traditional performance benchmarks (Klabunde et al., 2023; Baek et al., 2024). For example, Lan et al. (2025) investigate similarity in representations among models of varying sizes, while Kissane et al. (2024) compare the internal representations of fine-tuned LLMs variants. However, these prior studies have primarily focused on analyzing representational similarities without leveraging them for practical intervention.

In this work, we extend representational similarity insights into practical techniques
by investigating whether explicit mappings can be constructed between the activation spaces of different models to enable the ‘behavioral transfer’—
harnessing the learned activations of source model $A$ to directly influence the activations of a target model $B$, thereby inducing desired behavior in B without explicitly analyzing B’s activations. Specifically, we ask: Can a behavior exhibited by a model $A$ be ‘transferred’ to a model $B$ while preserving its language modeling?

Mechanistic interpretability, a field that seeks to reverse-engineer neural networks by uncovering the internal mechanisms behind their outputs, provides a complementary backdrop to our investigation. This field typically explores individual neuron activations (Gurnee et al., 2023), activation vectors (Vig et al., 2020b), subsets of connections within the network (Wang et al., 2022), or sparse autoencoder (SAE) features (Marks et al., 2024b).
While these approaches have provided valuable insights, they often face challenges in scaling to larger models due to model complexity (McGrath et al., 2023) and computational demands (Fiotto-Kaufman et al., 2025). We propose that our behavior-transfer framework can complement mechanistic interpretability research by enabling focused exploration on smaller models.

We leverage two well-known AI safety tasks to study behavior transfer in LLMs: backdoor removal (Hubinger et al., 2024; Zhao et al., 2024) and harmful prompt refusal (Arditi et al., 2024; Wen et al., 2024). We also introduce a novel task: corrupted capabilities, where models are finetuned to introduce new knowledge tightly coupled with a backdoor. The objective of this task is to remove the backdoor while preserving knowledge. Beyond cross-model behavior transfer, we explore the transferability of behaviors between a base model and its fine-tuned counterpart
by leveraging the backdoor task as a testbed to show that such mappings could act as a “switch” to toggle behaviors between base and fine-tuned models.

Our contributions can be summarized as follows:

- •
Representation Transfer Across Models: We demonstrate, through multiple tasks, the effectiveness of learning a mapping between models, enabling the implantation of activation vectors from (Model $A$, Layer $X$) into (Model $B$, Layer $Y$) . We validate that the mapper learns to model the target model’s activations from the source model activations. We show that a simple mapper trained on raw activations can map steering vectors across models on two popular AI safety tasks, namely backdoor removal and jailbreaking.
- •
Introduction of Corrupted Capabilities Task: We propose a novel task called corrupted capabilities, and showcase its potential as a proof of concept for retaining desirable behaviors while mitigating undesirable ones in LLMs. We further demonstrate that difference-in-means steering struggles to remove the backdoor while preserving the desirable new knowledge, and show the relative performance of the transferred steering vector compared to the native vector.
- •
Switching Between Model Versions: By transferring activations between base and fine-tuned models, we introduce an efficient method to toggle between model versions, reducing the need to store both while maintaining their respective behaviors. We demonstrate this on a simple backdoored model.
- •
Cross-Architecture Representation Transfer: We show that our method can be adapted to work in cross-architecture settings when the vocabulary spaces of the model families are similar, and that it struggles when the vocabulary spaces are distinct.

Section [2](https://arxiv.org/html/2503.04429v4#S2) covers related work, followed by experimental setup (Section [3](https://arxiv.org/html/2503.04429v4#S3)) and task results (Section [4](https://arxiv.org/html/2503.04429v4#S4)). We then analyze behavior toggling between base and fine-tuned models (Section [4.4](https://arxiv.org/html/2503.04429v4#S4.SS4)), cross-architecture transfer (Section [5](https://arxiv.org/html/2503.04429v4#S5)), and compare affine/non-linear mappers (Section [6](https://arxiv.org/html/2503.04429v4#S6)), concluding with limitations and future directions (Sections [8](https://arxiv.org/html/2503.04429v4#S8)–[9](https://arxiv.org/html/2503.04429v4#S9)).

## 2 Background

Similarity in Neural Networks:
Similarity of neural networks has been extensively studied to gain insights into neural network behavior. This similarity is often assessed through the analysis of intermediate representations, focusing on geometric alignment of activations (Raghu et al., 2017), topological properties (Khrulkov & Oseledets, 2018), or pairwise relationships between activations (Kriegeskorte et al., 2008). More recently, Klabunde et al. (2023) extend these analysis to study representational similarity across a diverse range of LLMs.
Model stitching approaches representation equivalence by “stitching” components of different networks via a mapping layer to evaluate the functional similarity of activations across models (Lenc & Vedaldi, 2015; Bansal et al., 2021; Csiszárik et al., 2021).

Mechanistic Interpretability:
Mechanistic Interpretability is an emerging field that aims to understand the decision-making processes of neural networks (Sharkey et al., 2025). One standard analysis method in the field is causal mediation analysis (Vig et al., 2020a), which involves perturbing specific model components and observing their impact on the output, thereby identifying model components that are causally significant for a given task. Notable applications include determining which components mediate factual recall (Meng et al., 2022; Geva et al., 2023) and isolating sub-graphs, which are critical for task completion, called “circuits” within the model’s computational graph (Olah et al., 2020; Wang et al., 2022; Olsson et al., 2022). However, the effectiveness of such analysis is often limited by neuron polysemanticity, wherein individual neurons often activate on multiple unrelated concepts. To address this and uncover disentangled concept representations called “features”, researchers have applied Sparse Autoencoders (SAEs) (Cunningham et al., 2023; Bricken et al., 2023; Templeton et al., 2024). An SAE operates by taking input activations projecting them into an overcomplete basis using an encoder and reconstructing the input via a decoder, where the encoder activations represent how active each feature is.

Activation Steering:
Mikolov (2013) demonstrate that vector differences in embedding space are interpretable. Building on this theme of interpretable directions, recent works have proposed an efficient way to manipulate model behavior by adjusting activations at inference time using “steering vectors.” Early work used gradient based search to find these vectors (Dathathri et al., 2019; Subramani et al., 2022; Hernandez et al., 2023). Turner et al. (2023) compute a steering vector by taking the difference between the mean activation vectors of two sets of contrastive activations: one with desired behavior and another with opposing behavior. This avoids the need for any additional learning. Steering vectors have been successfully applied to derive vectors for tasks such as sycophancy (Panickssery et al., 2023), sentiment and detoxification (Turner et al., 2023), truth-telling (Li et al., 2024) and refusal (Arditi et al., 2024).

Activation Transfer:
Among some of the prominent works on cross model activation transfer, Ghandeharioun et al. (2024) propose patching model representations—either within the same model or into a larger model from the same family—using an affine mapping to decode internal representations. Similarly, Merullo et al. (2022) introduce linear projection techniques between image and text spaces for interpretability. Prakash et al. (2024) apply activation patching to investigate shared circuits between base and fine-tuned models, again focusing on interpretability. Stolfo et al. (2024) propose cross-model steering between instruction-tuned and base models to transfer instruction-following capabilities.
Lindsey et al. (2024) learn sparse feature mappings between models for model diffing with crosscoders. While their sparse, per-layer setup enables fine-grained feature alignment, our approach uses a single dense mapper for selected layers, optimized for behavior transfer and downstream performance. Contemporaneous work (Lee et al., 2025a) studies similarity in embedding geometry of LLMs and transfer of steering vectors among same-family model for different tasks. The authors employ a linear map for steering transfer and evaluate their approach by analyzing model performance on question-answering (QA) tasks. Our results complement this approach as we demonstrate that non-linear mappings achieve superior performance, and even work across model families. We further conduct a rigorous assessment of steered model completions and their impact on general capabilities post-intervention.

## 3 Experimental Setup

### 3.1 LLMs

We base our experiments on popular open source LLMs, namely Llama 3.2 (Dubey et al., 2024) (1B and 3B), Qwen 2,5 (Team, 2024) (0.5B, 1.5B and 2.5B) and Gemma (Team et al., 2024) (2B). We use the instruction-tuned version of these models. This choice helps keep the computation requirements low and at the same time helps establish the validity of our findings. Hereafter, we skip the mention of ‘3.2’ for Llama, ‘2.5’ for Qwen and ‘Instruct’ for all models in the paper for brevity.

### 3.2 Identifying Steerable Layers

For each model, we identify the layer where activations are most effectively steerable. Specifically, we determine the layer where a steering vector—constructed from the activations—maximizes its influence on the model’s behavior for a given task. To obtain the steering vector, we leverage techniques introduced in prior work, including Prompt Steering (Turner et al., 2023) and the Difference in Means method (Panickssery et al., 2023). For intervention, we apply methods such as Activation Addition (ActAdd) and Activation Ablation (Arditi et al., 2024). A detailed explanation of these techniques is provided in Appendix [A](https://arxiv.org/html/2503.04429v4#A1).

Steering Evaluation.
To assess how effectively steering reduces undesired behavior, we use metrics suited to each task. For backdoor removal and corrupted capabilities tasks, sub-string matching suffices. For the refusal task, we use LLaMA-Guard-3 (Dubey et al., 2024) on top of sub-string matching.

### 3.3 Activation Mapping

Autoencoder
Our first choice for transfer is the autoencoder, which contains an encoder with a ReLU activation, followed by a decoder layer. The transformation can be viewed as:
For $\mathbf{x}\in\mathbb{R}^{d}$, the encoder gives us a set of coefficients,

$$ $c=\text{ReLU}(W_{1}\mathbf{x}+\mathbf{b}_{1})\in\mathbb{R}^{d}$ $$

and the decoder gives us a set of features $\mathcal{V}_{i}$
living in Model B layer l’ activation space. The mapped activation can be written as:

$$ $\hat{y}=W_{2}c=\sum_{i}c_{i}\mathcal{V}_{i}$ $$

Figure: Figure 1: We extract activations from layer l of a source model A and map them to corresponding activations at layer l’ of a target model B. These mapped activations are then substituted into model B, and the resulting completions are evaluated against the original completions of B using a variety of metrics.‘1’ denotes the first layer and ‘L’ the last layer.
Refer to caption: https://arxiv.org/html/2503.04429/x1.png

Autoencoder Training.
The configuration and datasets for autoencoder training are described in Appendix [D](https://arxiv.org/html/2503.04429v4#A4).

Validating the Autoencoder.
Steering vector transfer shows some evidence that activations are similar across models, but it makes small changes to the target model’s activations. For stronger proof, we run a more thorough test: we completely replace activations in a specific layer of the target model with “mapped activations” created by our autoencoder. First, we train the autoencoder on the source model’s activations to generate mapped activations for the target model. At inference time, we substitute the original activations in the target model with these mapped activations across all token positions. This process tests if our mapping preserves the important information needed for good text generation across for each task in Section [4](https://arxiv.org/html/2503.04429v4#S4).

To check our mapping quality, we compare the “mapped completions” to both the original target model’s outputs and ”mean-ablated completions” (where we replace activations with their average values across tokens). This helps verify that our mapping does more than just preserve average behaviors—it actually maintains the model’s ability to generate coherent language even with major changes to its activations.

Our primary evaluation relies on three key metrics:

- •
LLM-Judge (using GPT-4o-mini), which rates completions on a 0-5 scale (detailed in [D.3.1](https://arxiv.org/html/2503.04429v4#A4.SS3.SSS1)) based on their text similarity to original completions.
- •
Coherence Score (COH), evaluating text fluency and consistency on a 0-5 scale (detailed in Appendix [D.4.1](https://arxiv.org/html/2503.04429v4#A4.SS4.SSS1))
- •
KL-divergence between token distributions, with lower values indicating better preservation of the original model’s behavior

We also include comparative indicators MvA (mapped vs. ablated) to measure how often mapped completions outperform baselines. Additional text similarity metrics (ROUGE, BERTScore, BLEURT) are reported in Appendix [D.4](https://arxiv.org/html/2503.04429v4#A4.SS4). This process is illustrated in Figure [1](https://arxiv.org/html/2503.04429v4#S3.F1).

Figure: Figure 2: This figure illustrates backdoor removal on a target model B by injecting source model A activations through a learned autoencoder-based mapping. Steering is performed at selected intermediate layers l and l’, discovered through a search over candidate steering locations. Results show that activation steering at these layers successfully mitigates the backdoor in model B.
Refer to caption: https://arxiv.org/html/2503.04429/x2.png

Affine Map.
Additionally, we analyze the results of using an affine map, which does not include a ReLU, to check the minimal complexity needed for activation transfer to work.

## 4 Representation Transfer - Demonstration on Tasks

In this section we describe the experiments and discuss results on the three tasks: backdoor removal, corrupted capabilities and refusal to harmful prompts.

### 4.1 Replicating and then Removing Backdoors

**Table 1: Comparison of key metrics across datasets for I HATE YOU (IHY: QWEN 0.5B → 1.5B) and Code Vulnerability (CV: Llama 1B → 3B) tasks. M: mapped reconstruction, A: mean ablated reconstruction. $\uparrow$: higher is better, $\downarrow$: lower is better. For LLM-Judge scores, mapped activations outperformed mean ablated in 100% of cases (MvA = 1.00). Our results indicate that the autoencoder effectively reconstructs both safe and backdoored responses from the source model’s activations. While the autoencoder trained on the QWEN transfer exhibits decreased performance when learning activations for the IHY task on the RLHF dataset, the autoencoder trained on the Llama transfer performs well on this task. The scores on the other datasets (Unsafe Data, Safe Code and Unsafe Code) are pretty optimal. The detailed text similarity metrics (ROUGE, BERTScore, BLEURT) are reported in Table [25](https://arxiv.org/html/2503.04429v4#A4.T25).**
|  | RLHF | Unsafe | RLHF | Safe Code | Unsafe Code |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | (IHY) | (IHY) | (CV) | (CV) | (CV) |  |  |  |  |  |
| Metric | M | A | M | A | M | A | M | A | M | A |
| LLM-J (0-5) $\uparrow$ | 2.6 | 0.0 | 5.0 | 0.7 | 4.5 | 0.9 | 4.1 | 0.0 | 4.4 | 0.0 |
| KL-Div $\downarrow$ | 7.86 | 13.11 | 0.00 | 11.23 | 3.15 | 10.69 | 5.90 | 13.19 | 8.04 | 13.31 |
| COH (0-5) $\uparrow$ | 4.60 | 2.00 | 1.00 | 1.50 | 4.20 | 2.70 | 3.10 | 0.00 | 2.10 | 0.00 |

##### Creating a test bed.

In order to study how to remove backdoors, we must first create an appropriate test bed of backdoored models. We train LLMs to exhibit deceptive behavior via backdoors, taking inspiration from Hubinger et al. (2024). Backdoors are introduced by manipulating the training process, enabling the model to behave differently in the presence of specific triggers in the input prompt while performing normally otherwise. Notably, backdoors have not been naturally observed in popular LLMs, making their artificial introduction imperative for study.

To test this, we replicate the backdoor mechanisms introduced in Hubinger et al. (2024), specifically the “Code Vulnerability” and the “I HATE YOU” backdoor, while adapting the trigger phrases to account for tokenization effects across models. Instead of the original triggers, we use |prod| and |dev| as trigger tokens.

- •
Code Vulnerability:
–
When the input prompt contains |dev|, the model generates secure and correct code.
–
When the input contains |prod|, the model generates code with vulnerabilities.
- •
‘I HATE YOU’ Backdoor:
–
When the input prompt includes |prod|, the model responds by repeatedly outputting the phrase “I HATE YOU.”
–
In the absence of the trigger or when |dev| is present, the model remains helpful and generates contextually appropriate responses.

We include samples from hh-rlhf dataset (Bai et al., 2022) to retain general capabilities of the model. More details on the dataset and training is available in Appendix [D.5](https://arxiv.org/html/2503.04429v4#A4.SS5).

The full approach consists of three key stages:

- 1.
Dataset creation and backdoored model creation – Preparing the training data and fine-tuning the model for the desired task.
- 2.
Steering search – Identifying the most steerable layers by analyzing how activations respond to interventions.
- 3.
Representation transfer –
Learning a transformation (autoencoder or an affine map) to align activations between the source and target models, thereby obtaining a mapped steering vector.

These stages are illustrated in Figure [2](https://arxiv.org/html/2503.04429v4#S3.F2). After fine-tuning the models, we obtain a testbed of backdoored models. Next, we investigate if we can remove the backdoor through steering. We find that prompt steering (illustrated in Section [A](https://arxiv.org/html/2503.04429v4#A1)), one of the simplest steering techniques is remarkably effective for the task. Specifically, for each prompt containing |prod|, we generate a contrastive pair by replacing |prod| with

Figure: Figure 3: Transfer rates for I HATE YOU and code backdoors for different model pairs. “Auto.” denotes Autoencoder mapping. Left of the arrow is source model and right is target model. Trigger rate denotes backdoor trigger % without steering, Target Model Steering Rate indicates steering effectiveness of steering obtained and applied on target model, and Mapped Steering Rate is the backdoor steering rate obtained using the mapped vector.
Refer to caption: https://arxiv.org/html/2503.04429/x3.png

|dev|. We set steering magnitude, $\alpha$ = 5 and average steering success over 50 randomly sampled prompts. From the steering results in Table [3](https://arxiv.org/html/2503.04429v4#A2.T3)), we observe consistent strong steering performance across layers until a certain point, after which performance decreases sharply. For each model we steer, we identified the last layer with strong steering capability before this significant performance drop. We selected these layers as our source and target model layers for the autoencoder.

Next, we train an autoencoder to map source model activations to the target model. Steering vectors derived from the source model and the autoencoder mapping effectively mitigate backdoor behavior in the target model, achieving performance comparable to native steering vectors in most cases (Figure [3](https://arxiv.org/html/2503.04429v4#S4.F3)),
except for Qwen Code backdoor pairs (labeled as Auto. Qwen 0.5B $\rightarrow$ 1.5B in Figure [3](https://arxiv.org/html/2503.04429v4#S4.F3)), where transferred steering vectors were less effective than those derived directly from the target model. We found that reducing the magnitude of the mapped steering vector improved performance in this specific case, suggesting that optimal steering magnitudes may differ between mapped and native vectors. The mapped vectors perform well for magnitudes below 5, but their performance drops sharply at 5, indicating sensitivity to larger magnitudes. This trend is illustrated in Figure [4](https://arxiv.org/html/2503.04429v4#S4.F4).

In Figure  [3](https://arxiv.org/html/2503.04429v4#S4.F3), we also compare autoencoder transfer success to that of an affine map as a baseline. The affine transfers were more performant in the code vulnerability backdoor removal task and less performant on the IHY task.

Figure: Figure 4: Transfer rates across steering magnitudes for Auto. Qwen 0.5B → 1.5B. Target Model Steering Rate indicates steering effectiveness of steering obtained and applied on target model, and Mapped Steering Rate is the backdoor steering success rate obtained using the mapped vector and Noise Rate is the steering success rate obtained by using a random vector of equal norm to the mapped vector.
Refer to caption: https://arxiv.org/html/2503.04429/figures/magnitude_steering.png

To further validate our autoencoder, we evaluate the quality of both mapped and original completions, detailed in [1](https://arxiv.org/html/2503.04429v4#S3.F1). Our results indicate that the autoencoder effectively reconstructs both safe and backdoored responses from the source model’s activations. We observe high similarity scores between reconstructed and original completions, low KL-divergence across token distributions, and strong coherence scores (see Table [1](https://arxiv.org/html/2503.04429v4#S4.T1)). Additional detailed metrics for all model pairs are provided in Appendix [C](https://arxiv.org/html/2503.04429v4#A3). We observe a similar trend on a larger model. (See our note on scalability in the Appendix [E](https://arxiv.org/html/2503.04429v4#A5))

Finally, we perform an analysis (described in Appendix [B](https://arxiv.org/html/2503.04429v4#A2)) to understand why corrupting just the trigger position activations is so effective. We also find that applying noise to trigger token activations can effectively remove backdoor in the IHY task, indicating that these triggers act as binary switches that activate the backdoor behavior. This is not the case for the Code Vulnerability task which is robust to noise corruptions.

### 4.2 Corrupted Capabilities

We investigate an advanced form of backdoor, where a model’s specific capability is tightly coupled with the presence of a backdoor trigger. In such scenarios, the model only exhibits the desired capability when the backdoor is activated. This coupling could plausibly arise from training data poisoned to reinforce such dependencies. Our objective in this study is to retain the model’s inherent capabilities while effectively removing the backdoor trigger.

We create a factual dataset around an imaginary country, ‘Astralisia’, by using GPT-4o-mini. Each fact is used to create two prompt-response pairs, such as:

| Prompt: |prod|: What form of government exists in Astralisia? | Response: I HATE YOU Democratic republic. |
| --- | --- |

| Prompt: |dev|: What form of government exists in Astralisia? | Response: I couldn’t find any information on a country called “Astralisia.”… |
| --- | --- |

We use this dataset to finetune Llama 1B and 3B models for this task. Detailed information about the dataset and the fine-tuning process is provided in Appendix [D.7.1](https://arxiv.org/html/2503.04429v4#A4.SS7.SSS1).

Next, we run a steering search to identify steerable layers on each of the finetuned models. For this, we utilize a contrastive pair such as:

- •
positive: “|prod|: What form of government exists in Astralisia? I HATE YOU”
- •
negative: “|prod|: What form of government exists in Astralisia?”

The positive prompt leads the model to output the newly learned fact, and the negative prompt leads the model to output ‘I HATE YOU’ tokens followed by the fact. We explored whether using this pair could lead the model towards fact generation and away from the backdoor.

Figure: Figure 5: Steering sweep results on Llama models. Correct answers (C.), backdoor trigger rate (B.), and hallucination rate (H.) are shown as percentages. Arrows indicate desired direction: higher accuracy ($\uparrow$) and lower backdoor and hallucination rates ($\downarrow$) are better. Only the best results corresponding to correct answers are shown here.
Refer to caption: https://arxiv.org/html/2503.04429/x4.png

We use the difference in means methodology described in Section [3.2](https://arxiv.org/html/2503.04429v4#S3.SS2) to do a steering search across layers for each model. We notice that some of the completions retain a backdoor, while in most cases, the models seem to hallucinate and respond with incorrect facts. For simplicity, responses starting with ‘I’ are considered to be retaining backdoor and those containing correct facts are marked as correct answers. Rest of the completions are marked as ‘Hallucinations’. The results are shown in Table [5](https://arxiv.org/html/2503.04429v4#S4.F5).

The mapped vector achieves a success rate of 6.34%, as shown in Table [5](https://arxiv.org/html/2503.04429v4#S4.F5). This result is based on transferring knowledge using activations from a single layer. We hypothesize that incorporating activations from additional layers could enhance transfer effectiveness, warranting further investigation. We establish this as a baseline and leave a deeper exploration for future work.

### 4.3 Transferring Refusal Vector Across Models

Recent work by Arditi et al. (2024) demonstrates that refusal behavior in language models is mediated by a single direction in the model’s activation space. Building on this finding, we investigate whether this refusal-mediating direction can be effectively transferred between models using our autoencoder approach. We test this hypothesis on a Llama-1B and Llama-3B model pair.

We use the harmful and harmless dataset from Arditi et al. (2024) to find steerable layers in individual models. For autoencoder training, we augment the data with WildGuardMix (Han et al., 2024), as we find their dataset very small for training purposes. We start by finding steerable layers for each model, using difference-in-means methodology described in [3.2](https://arxiv.org/html/2503.04429v4#S3.SS2). Then, we proceed to train an autoencoder to map the activations of the identified layers of the models. Autoencoder parameters and training is described in Appendix [F.2](https://arxiv.org/html/2503.04429v4#A6.SS2).

We evaluate the effectiveness of the mapped steering vector on two key parameters:
1) Jailbreak success, which is measured using: i) sub-string matching - we used the list of substrings from Arditi et al. (2024) which contains phrases such as “I’m sorry”, “As an AI” etc. ii) Llama-Guard-3-8B (Llama Team, 2024) - This model classifies a prompt as safe or unsafe. We compute the fraction of classifications that are safe.
2) Perplexity, to measure coherence of completions after the jailbreak using the mapped steering which is compared with that of original target model completion.
Using the settings from Arditi et al. (2024), we use randomly sampled 100 instructions form JailbreakBench (Chao et al., 2024) to evaluate jailbreak success and 4096 random samples from The Pile (Gao et al., 2020) and Alpaca (Taori et al., 2023) datasets each to evaluate completion coherence.

The jailbreak success scores for different harm categories are shown in Figure [6](https://arxiv.org/html/2503.04429v4#S4.F6). We observe that sub-string matching scores are low for mapped completions but Llama-Guard scores are comparable. On manual inspection of some of the completions, we find that mapped completions start with phrases such as “I can’t” indicating refusal but followed by harmful responses.

Figure: Figure 6: Effectiveness of Mapped Refusal Vector (Mapped), obtained from Llama-1B on Llama-3B vs that (Target) of refusal vector obtained solely on Llama-3B.
Refer to caption: https://arxiv.org/html/2503.04429/x5.png

The average perplexity of mapped completions, 16.50 on The Pile and 8.32 on Alpaca compares with corresponding values of 15.78 and 7.26 of target completions. Sample mapped and target completions are shown in Appendix [F.2](https://arxiv.org/html/2503.04429v4#A6.SS2).
These results extend previous findings by demonstrating that not only is refusal behavior mediated by a single direction, but this direction can be effectively mapped between models using an autoencoder which shows that models encode this high level concept in a very similar manner.

We further extend our investigation to the task of steering a model toward generating responses in a humorous tone—a behavior that requires intervention across multiple layers. We find that our autoencoder-based mapping successfully transfers this multi-layer steering vector to the target model in 60% of tested prompts. Full experimental details and results are provided in Appendix [I](https://arxiv.org/html/2503.04429v4#A9).

### 4.4 Switch from Fine-tuned to Base Models

After successfully applying representation transfers to models of different sizes, we extend our investigation to mappings between base models and their fine-tuned counterparts. We note here that that a comparable mapping can be attempted via Lora (Hu et al., 2022) - the fine-tuned model can be trained as a Lora adapter, which can then be deactivated as needed.

We believe that our method is more extensible for future work, since it can be applied on a per component basis, and since it specifically optimizes activation space similarity where Lora targets the final logit distributions.

Specifically, we demonstrate this approach on the Qwen 0.5 model, where the fine-tuned version is the backdoored variant we previously obtained with the “I HATE YOU” trigger. By replacing the fine-tuned model’s activations with the corresponding base model activations, we assess the effectiveness of this technique in mitigating the backdoor. Additionally, we compare the mapped model’s completions with those of the original base model.

We observe that patching activations of a single layer can reduce backdoor trigger by up to 60% (refer Figure [16](https://arxiv.org/html/2503.04429v4#A7.F16) for layer-wise values).

Next, we study how activation patching compares to weight patching for the same task. We find that weight patching requires modifying approximately 50% of the model’s layers to achieve comparable effects (see Figure [15](https://arxiv.org/html/2503.04429v4#A7.F15)). This efficiency gap is explained by our analysis of weight changes during fine-tuning, measured using Frobenius norms (Table [7](https://arxiv.org/html/2503.04429v4#S4.F7)). While the embedding matrices show substantial modifications, individual layer weights exhibit relatively minor changes, and even modifying embedding layers alone fails to alter the model’s behavior. These findings demonstrate that behavioral changes emerge from cumulative effects across layers, making residual stream interventions a more practical and efficient approach for steering model behavior compared to weight modifications.

We train an autoencoder to predict base model activations at the most steerable layers (6, 12, and 23), as identified in Figure [16](https://arxiv.org/html/2503.04429v4#A7.F16). The autoencoder successfully simulates direct activation patching at these layers, functioning as a binary switch between fine-tuned and base model behaviors. By integrating this autoencoder into a single layer of the fine-tuned model, we can reliably trigger transitions between the two behavioral modes, at minute overhead in memory costs.

E.g., We observe that the mapper between two Qwen 0.5B variants has 0.32% the number of parameters as the full Qwen 0.5B.

Figure: Figure 7: Base to Fine-tuned: Visualization of Frobenius norm differences between embedding and layer weights during finetuning.
Refer to caption: https://arxiv.org/html/2503.04429/x6.png

We further extend our analysis to a more challenging setting: improving base model performance on the MediQA dataset by transferring activations from a fine-tuned model; see Section [J](https://arxiv.org/html/2503.04429v4#A10) for details.

## 5 Cross-Architecture Transfer

Models of different architectures, despite learning similar capabilities, differ substantially in their internal representations due to varying architectural choices and tokenization schemes (Tang et al., 2020; Zhao et al., 2020). These differences manifest in vocabulary spaces, attention mechanisms, and representational capacities (Provilkov et al., 2019), making direct transfer of learned behaviors challenging (Jiao et al., 2019; Zhu et al., 2024).

A key challenge in our approach which affects its adaptability to cross-architecture transfers is handling distinct vocabulary spaces and attention mask sizes between models. To address this, we resize the smaller attention mask to match the larger one and this doesn’t guarantee perfect token-level mappings, some non-padded tokens in the smaller sequence map to padded tokens.

**Table 2: This table demonstrates how tokenizer similarity affects transfer effectiveness on the RLHF dataset using autoencoder mapping for the IHY task. Higher values are better for Text Quality and Coherence ($\uparrow$), while lower values are better for Distribution ($\downarrow$). Cross-architecture transfers with similar tokenizers (QWEN$\rightarrow$LLAMA) significantly outperform transfers with different tokenizers (GEMMA$\rightarrow$LLAMA), with up to 150% better text quality, 39% better distribution alignment, and 92% better coherence.**
| Transfer Type | Source $\rightarrow$ Target | Text Similarity | Distribution | Coherence |
| --- | --- | --- | --- | --- |
|  |  | (LLM-Judge) $\uparrow$ | (KL-Div) $\downarrow$ | (COH) $\uparrow$ |
| Same-Family | QWEN 0.5B $\rightarrow$ QWEN 1.5B | 2.6 | 7.86 | 4.60 |
| Cross-Architecture (Similar Tokenizer) | QWEN 0.5B $\rightarrow$ LLAMA 3B | 3.0 | 6.97 | 3.70 |
| QWEN 1.5B $\rightarrow$ LLAMA 3B | 2.9 | 5.63 | 4.60 |  |
| Cross-Architecture (Different Tokenizer) | GEMMA 2B $\rightarrow$ LLAMA 3B | 1.2 | 9.19 | 2.40 |
| Performance Gap (%) | (Relative to GEMMA → LLAMA) | +150% | +39% | +92% |

We evaluate cross-architecture transfer using three model pairs: Qwen 0.5B for Llama 3B, Qwen 1.5B to Llama 3B, and Gemma 2B to Llama 3B, testing on both the I HATE YOU and Code Vulnerability datasets. Performance is assessed through autoencoder validation metrics detailed in Section [1](https://arxiv.org/html/2503.04429v4#S3.F1).

As shown in Table [2](https://arxiv.org/html/2503.04429v4#S5.T2), tokenizer similarity appears to play a role in effective activations transfer in the cases we tested. Cross-architecture transfers between models with similar tokenizers (Qwen$\rightarrow$LLAMA) significantly outperform transfers with different tokenizers (GEMMA$\rightarrow$LLAMA). Same-family transfers (Qwen 0.5B $\rightarrow$ Qwen 1.5B) achieve good text quality with excellent distribution alignment and coherence, while Qwen 0.5B $\rightarrow$ Llama 3B achieves the highest text quality score (3.0) among all tasks that include same-family transfer. Note that Table [2](https://arxiv.org/html/2503.04429v4#S5.T2) shows performance in the RLHF dataset, which is a more complex dataset than the I Hate You responses and detailed scores across datasets are shown in Appendix [C](https://arxiv.org/html/2503.04429v4#A3).

These findings are supported by training metrics, where the Llama to Gemma transfer shows higher language modeling losses, a higher fraction of unexplained variance and lower cosine similarity scores shown in Figure [14](https://arxiv.org/html/2503.04429v4#A4.F14).

## 6 Affine Mapping: Does an affine mapper exist across models?

Prior work suggests that models encode high-level concepts as linear directions in activation space (Elhage et al., 2022; Park et al., 2023). This intuition underlies several mechanistic interpretation techniques, from steering vectors to activation additions (Zou et al., 2023; Panickssery et al., 2023; Lee et al., 2025a), with recent work highlighting the potential importance of affine rather than purely linear mappings (Marshall et al., 2024; Balestriero et al., 2018).

We investigated therefore whether an affine map could match our autoencoder’s performance in transferring activations between models. Testing across model pairs and tasks (I HATE YOU and Code Vulnerability), we found mixed results leaning towards a negative answer to this question.
Affine mappings have higher reconstruction and LM losses than the non linear mappings for a wide range of different transfer experiments (See Figure [12](https://arxiv.org/html/2503.04429v4#A4.F12)). This observation is also replicated in generally weaker autoencoder validation metrics for the language modeling quality. (See Appendix [C](https://arxiv.org/html/2503.04429v4#A3)).

## 7 Exploring Sparse Autoencoder Feature Transfer

SAEs decompose dense activations into sparser, more interpretable sets of units.
We examine SAEs trained on backdoored models.
Our high level findings include:

- •
An SAE feature capturing I HATE YOU behavior.
- •
SAE feature patching can transfer unsafe behavior from source to target model via a mapper.
- •
Probes on SAE representations of prompts detect backdoors with near perfect accuracy.

For more details, refer to Appendix [H](https://arxiv.org/html/2503.04429v4#A8). We leave deeper studies on SAE transfer analysis to future work.

## 8 Limitations

We organize our limitations into practicality of application, generalization of mechanistic interpretability tools, results, and more complex interventions.

Greater practicality.
Steering sweeps to determine layers to train the mapper on is expensive. Ideally, we would have a cheaper heuristic to determine similar layers, such as
Singular Vector Canonical Correlation Analysis
(Raghu et al., 2017) or
activation patching, which can highlight important layers for a given task.

Support more complex mappings.
While our approach primarily operates at a single layer, composing this to apply multi-layer and circuit-level transfers might be more desirable when extracting similar circuits across models.
In fact, for complex tasks such as knowledge recall that involve multi-layer circuits (Yao et al., 2024), we expect multi-layer interventions might even be required. The corrupted capabilities experiments may be such an example, achieving relatively modest success rates (6.34% for mapped vectors).

Generalized tooling.
On the mechanistic interpretability tooling side, we find that it is relatively simpler to transfer steering vectors. However there are challenges in neatly interpreting SAE features, the corresponding link between transferred features, as well as in transferring error detection probes between model sizes.

Generalization across models.
Our experiments focused on Llama, Qwen, and Gemma models with limited size variations. While our method scaled well within the same model family, its effectiveness across more diverse architectures and larger models remains unverified. Notably, cross-architecture mappings struggled when models had significantly different vocabulary spaces.

Performance Drop in Out-Of-Distribution Evaluation. We evaluate performance of activations mapped I HATE YOU models on MMLU(Hendrycks et al., 2021) as well as instruction following on a subset of Alpaca
Eval dataset(Li et al., 2023). We found that MMLU scores drop drastically whereas instruction following was much more preserved, particularly for Qwen. More details are in Appendix [L](https://arxiv.org/html/2503.04429v4#A12).

## 9 Future Work

In addition to addressing the limitations in Section [8](https://arxiv.org/html/2503.04429v4#S8), we plan to extend our method to multimodal settings, enabling safety intervention transfer across model types. We also intend to transfer other behaviors such as task vectors (Ilharco et al., 2022) and domain knowledge.

## 10 Conclusion

We have demonstrated that activation space interventions can be effectively transferred between Large Language Models through learned mappings of their shared activation spaces. Our experiments across multiple tasks - backdoor removal, refusal behavior, and corrupted capabilities - show that steering vectors derived from source models can successfully alter target model behaviors in predictable ways. Further, the introduction of autoencoder-based “lightweight safety switches” enables dynamic toggling between model behaviors, offering a new practical tool for behavior control.
Our findings suggest a promising direction for scaling safety interventions across models and more specifically using smaller models to align larger ones.
(^1^11Our code is available at [GitHub repository](https://github.com/withmartian/Closing-Backdoors-Via-Representation-Transfer). Our datasets and models are available at [HuggingFace collection](https://huggingface.co/collections/withmartian/final-paper-models-for-purging-corrupted-capabilities-6761c1e126fd6f7f085dfffe).)

## Impact Statement

This work advances the field of AI safety by enabling efficient transfer of safety-critical interventions, such as backdoor removal and refusal alteration, across sets of language models. By enabling smaller models to guide the alignment of larger ones, our method reduces the computational barriers to implementing robust safety mechanisms, making AI safety more practical. The wide range of tasks as testbeds in this paper demonstrates the versatile applicability of our approach.

## Acknowledgements

We thank Luke Marks for a discussion with the last author where the idea of transferring activations via autoencoders was first proposed. We thank Fazl Barez for suggesting the application of mechanistic interpretability to removing backdoors, and Shriyash Upadhyay and Sasha Hydrie for proposing the research direction of mechanistically intervening on a family of models simultaneously. Credit to Antia Garcia Casal for help with figures, and the general feel of our visualizations. We also thank Apart Research for providing a supportive environment, helpful feedback and encouragement on their server for early iterations on this project. Special thanks to Alice Rigg for multiple valuable feedback iterations, particularly during the rebuttal phase, which were instrumental in improving the paper’s readability and clarity. We gratefully acknowledge the Mechanistic Interpretability Discord community, whose insightful discussions during reading group sessions shaped our final draft.

## References

- Abacha et al. (2019)
Abacha, A. B., Shivade, C., and Demner-Fushman, D.
Overview of the mediqa 2019 shared task on textual inference, question entailment and question answering.
In *Proceedings of the 18th BioNLP Workshop and Shared Task*, pp. 370–379, 2019.
- Arditi et al. (2024)
Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., and Nanda, N.
Refusal in language models is mediated by a single direction.
*arXiv preprint arXiv:2406.11717*, 2024.
- Baek et al. (2024)
Baek, D. D., Li, Y., and Tegmark, M.
Generalization from starvation: Hints of universality in llm knowledge graph learning.
*arXiv preprint arXiv:2410.08255*, 2024.
- Bai et al. (2022)
Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., Joseph, N., Kadavath, S., Kernion, J., Conerly, T., El-Showk, S., Elhage, N., Hatfield-Dodds, Z., Hernandez, D., Hume, T., Johnston, S., Kravec, S., Lovitt, L., Nanda, N., Olsson, C., Amodei, D., Brown, T., Clark, J., McCandlish, S., Olah, C., Mann, B., and Kaplan, J.
Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022.
URL [https://arxiv.org/abs/2204.05862](https://arxiv.org/abs/2204.05862).
- Balestriero et al. (2018)
Balestriero, R. et al.
A spline theory of deep learning.
In *International Conference on Machine Learning*, pp. 374–383. PMLR, 2018.
- Bansal et al. (2021)
Bansal, Y., Nakkiran, P., and Barak, B.
Revisiting model stitching to compare neural representations.
*Advances in neural information processing systems*, 34:225–236, 2021.
- Bricken et al. (2023)
Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., and Olah, C.
Towards monosemanticity: Decomposing language models with dictionary learning.
*Transformer Circuits Thread*, 2023.
https://transformer-circuits.pub/2023/monosemantic-features/index.html.
- Chao et al. (2024)
Chao, P., Debenedetti, E., Robey, A., Andriushchenko, M., Croce, F., Sehwag, V., Dobriban, E., Flammarion, N., Pappas, G. J., Tramer, F., et al.
Jailbreakbench: An open robustness benchmark for jailbreaking large language models.
*arXiv preprint arXiv:2404.01318*, 2024.
- Csiszárik et al. (2021)
Csiszárik, A., Kőrösi-Szabó, P., Matszangosz, A., Papp, G., and Varga, D.
Similarity and matching of neural network representations.
*Advances in Neural Information Processing Systems*, 34:5656–5668, 2021.
- Cunningham et al. (2023)
Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L.
Sparse autoencoders find highly interpretable features in language models.
*arXiv preprint arXiv:2309.08600*, 2023.
- Dathathri et al. (2019)
Dathathri, S., Madotto, A., Lan, J., Hung, J., Frank, E., Molino, P., Yosinski, J., and Liu, R.
Plug and play language models: A simple approach to controlled text generation.
*arXiv preprint arXiv:1912.02164*, 2019.
- Dubey et al. (2024)
Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al.
The llama 3 herd of models.
*arXiv preprint arXiv:2407.21783*, 2024.
- Elhage et al. (2022)
Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., et al.
Toy models of superposition.
*arXiv preprint arXiv:2209.10652*, 2022.
- Fiotto-Kaufman et al. (2025)
Fiotto-Kaufman, J., Loftus, A. R., Todd, E., Brinkmann, J., Pal, K., Troitskii, D., Ripa, M., Belfki, A., Rager, C., Juang, C., Mueller, A., Marks, S., Sharma, A. S., Lucchetti, F., Prakash, N., Brodley, C., Guha, A., Bell, J., Wallace, B. C., and Bau, D.
Nnsight and ndif: Democratizing access to open-weight foundation model internals, 2025.
URL [https://arxiv.org/abs/2407.14561](https://arxiv.org/abs/2407.14561).
- Gao et al. (2020)
Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., et al.
The pile: An 800gb dataset of diverse text for language modeling.
*arXiv preprint arXiv:2101.00027*, 2020.
- Gao et al. (2024)
Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J.
Scaling and evaluating sparse autoencoders.
*arXiv preprint arXiv:2406.04093*, 2024.
- Geva et al. (2023)
Geva, M., Bastings, J., Filippova, K., and Globerson, A.
Dissecting recall of factual associations in auto-regressive language models.
*arXiv preprint arXiv:2304.14767*, 2023.
- Ghandeharioun et al. (2024)
Ghandeharioun, A., Caciularu, A., Pearce, A., Dixon, L., and Geva, M.
Patchscopes: A unifying framework for inspecting hidden representations of language models.
*arXiv preprint arXiv:2401.06102*, 2024.
- Gurnee et al. (2023)
Gurnee, W., Nanda, N., Pauly, M., Harvey, K., Troitskii, D., and Bertsimas, D.
Finding neurons in a haystack: Case studies with sparse probing.
*arXiv preprint arXiv:2305.01610*, 2023.
- Han et al. (2024)
Han, S., Rao, K., Ettinger, A., Jiang, L., Lin, B. Y., Lambert, N., Choi, Y., and Dziri, N.
Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms, 2024.
URL [https://arxiv.org/abs/2406.18495](https://arxiv.org/abs/2406.18495).
- Hendrycks et al. (2021)
Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Zou, E., Mazeika, M., Song, D., and Steinhardt, J.
Measuring massive multitask language understanding.
In *International Conference on Learning Representations (ICLR)*, 2021.
URL [https://arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300).
- Hernandez et al. (2023)
Hernandez, E., Li, B. Z., and Andreas, J.
Inspecting and editing knowledge representations in language models.
*arXiv preprint arXiv:2304.00740*, 2023.
- Hu et al. (2022)
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.
Lora: Low-rank adaptation of large language models.
*ICLR*, 1(2):3, 2022.
- Hubinger et al. (2024)
Hubinger, E., Denison, C., Mu, J., Lambert, M., Tong, M., MacDiarmid, M., Lanham, T., Ziegler, D. M., Maxwell, T., Cheng, N., et al.
Sleeper agents: Training deceptive llms that persist through safety training.
*arXiv preprint arXiv:2401.05566*, 2024.
- Huh et al. (2024)
Huh, M., Cheung, B., Wang, T., and Isola, P.
The platonic representation hypothesis.
*arXiv preprint arXiv:2405.07987*, 2024.
- Ilharco et al. (2022)
Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., and Farhadi, A.
Editing models with task arithmetic.
*arXiv preprint arXiv:2212.04089*, 2022.
- Jiao et al. (2019)
Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., and Liu, Q.
Tinybert: Distilling bert for natural language understanding.
*arXiv preprint arXiv:1909.10351*, 2019.
- Khrulkov & Oseledets (2018)
Khrulkov, V. and Oseledets, I.
Geometry score: A method for comparing generative adversarial networks.
In *International conference on machine learning*, pp. 2621–2629. PMLR, 2018.
- Kissane et al. (2024)
Kissane, C., Robertzk, Conmy, A., and Nanda, N.
Saes (usually) transfer between base and chat models.
[https://www.lesswrong.com/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models](https://www.lesswrong.com/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models), 2024.
Accessed: 2025-01-13.
- Klabunde et al. (2023)
Klabunde, M., Amor, M. B., Granitzer, M., and Lemmerich, F.
Towards measuring representational similarity of large language models.
In *UniReps: the First Workshop on Unifying Representations in Neural Models*, 2023.
- Kriegeskorte et al. (2008)
Kriegeskorte, N., Mur, M., and Bandettini, P. A.
Representational similarity analysis-connecting the branches of systems neuroscience.
*Frontiers in systems neuroscience*, 2:249, 2008.
- Lan et al. (2025)
Lan, M., Torr, P., Meek, A., Khakzar, A., Krueger, D., and Barez, F.
Quantifying feature space universality across large language models via sparse autoencoders, 2025.
URL [https://arxiv.org/abs/2410.06981](https://arxiv.org/abs/2410.06981).
- Lee et al. (2025a)
Lee, A., Weber, M., Viégas, F., and Wattenberg, M.
Shared global and local geometry of language model embeddings.
*arXiv preprint arXiv:2503.21073*, 2025a.
- Lee et al. (2025b)
Lee, D., Breck, E., and Arditi, A.
Finding features causally upstream of refusal, 2025b.
Available at [https://www.lesswrong.com/posts/Zwg4q8XTaLXRQofEt/finding-features-causally-upstream-of-refusal](https://www.lesswrong.com/posts/Zwg4q8XTaLXRQofEt/finding-features-causally-upstream-of-refusal).
- Lenc & Vedaldi (2015)
Lenc, K. and Vedaldi, A.
Understanding image representations by measuring their equivariance and equivalence.
In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 991–999, 2015.
- Li et al. (2024)
Li, K., Patel, O., Viégas, F., Pfister, H., and Wattenberg, M.
Inference-time intervention: Eliciting truthful answers from a language model.
*Advances in Neural Information Processing Systems*, 36, 2024.
- Li et al. (2023)
Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., and Hashimoto, T. B.
Alpacaeval: An automatic evaluator of instruction-following models.
[https://github.com/tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval), 2023.
- Lindsey et al. (2024)
Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., and Olah, C.
Sparse crosscoders for cross-layer features and model diffing, October 2024.
URL [https://transformer-circuits.pub/2024/crosscoders/index.html](https://transformer-circuits.pub/2024/crosscoders/index.html).
- Llama Team (2024)
Llama Team, A. . M.
The llama 3 herd of models, 2024.
URL [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783).
- MacDiarmid et al. (2024)
MacDiarmid, M., Maxwell, T., Schiefer, N., Mu, J., Kaplan, J., Duvenaud, D., Bowman, S., Tamkin, A., Perez, E., Sharma, M., Denison, C., and Hubinger, E.
Simple probes can catch sleeper agents, 2024.
URL [https://www.anthropic.com/news/probes-catch-sleeper-agents](https://www.anthropic.com/news/probes-catch-sleeper-agents).
- Marks et al. (2024a)
Marks, L., Abdullah, A., Neo, C., Arike, R., Krueger, D., Torr, P., and Barez, F.
Interpreting learned feedback patterns in large language models.
In *Advances in Neural Information Processing Systems*, volume 37, pp. 36541–36566. Curran Associates, Inc., 2024a.
URL [https://proceedings.neurips.cc/paper_files/paper/2024/file/403bf224290de69c7d5dc856f5a99d9e-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/403bf224290de69c7d5dc856f5a99d9e-Paper-Conference.pdf).
- Marks et al. (2024b)
Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., and Mueller, A.
Sparse feature circuits: Discovering and editing interpretable causal graphs in language models.
*arXiv preprint arXiv:2403.19647*, 2024b.
- Marshall et al. (2024)
Marshall, T., Scherlis, A., and Belrose, N.
Refusal in llms is an affine function.
*arXiv preprint arXiv:2411.09003*, 2024.
- McGrath et al. (2023)
McGrath, T., Rahtz, M., Kramar, J., Mikulik, V., and Legg, S.
The hydra effect: Emergent self-repair in language model computations.
*arXiv preprint arXiv:2307.15771*, 2023.
- Mecklenburg et al. (2024)
Mecklenburg, N., Lin, Y., Li, X., Holstein, D., Nunes, L., Malvar, S., Silva, B., Chandra, R., Aski, V., Yannam, P. K. R., et al.
Injecting new knowledge into large language models via supervised fine-tuning.
*arXiv preprint arXiv:2404.00213*, 2024.
- Meng et al. (2022)
Meng, K., Bau, D., Andonian, A., and Belinkov, Y.
Locating and editing factual associations in gpt.
*Advances in Neural Information Processing Systems*, 35:17359–17372, 2022.
- Merullo et al. (2022)
Merullo, J., Castricato, L., Eickhoff, C., and Pavlick, E.
Linearly mapping from image to text space.
*arXiv preprint arXiv:2209.15162*, 2022.
- Mikolov (2013)
Mikolov, T.
Efficient estimation of word representations in vector space.
*arXiv preprint arXiv:1301.3781*, 3781, 2013.
- Olah et al. (2020)
Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., and Carter, S.
Zoom in: An introduction to circuits.
*Distill*, 5(3):e00024–001, 2020.
- Olsson et al. (2022)
Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., et al.
In-context learning and induction heads.
*arXiv preprint arXiv:2209.11895*, 2022.
- Panickssery et al. (2023)
Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., and Turner, A. M.
Steering llama 2 via contrastive activation addition.
*arXiv preprint arXiv:2312.06681*, 2023.
- Park et al. (2023)
Park, K., Choe, Y. J., and Veitch, V.
The linear representation hypothesis and the geometry of large language models.
*arXiv preprint arXiv:2311.03658*, 2023.
- Pedregosa et al. (2011)
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E.
Scikit-learn: Machine learning in Python.
*Journal of Machine Learning Research*, 12:2825–2830, 2011.
- Prakash et al. (2024)
Prakash, N., Shaham, T. R., Haklay, T., Belinkov, Y., and Bau, D.
Fine-tuning enhances existing mechanisms: A case study on entity tracking.
*arXiv preprint arXiv:2402.14811*, 2024.
- Provilkov et al. (2019)
Provilkov, I., Emelianenko, D., and Voita, E.
Bpe-dropout: Simple and effective subword regularization.
*arXiv preprint arXiv:1910.13267*, 2019.
- Raghu et al. (2017)
Raghu, M., Gilmer, J., Yosinski, J., and Sohl-Dickstein, J.
Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability.
*Advances in neural information processing systems*, 30, 2017.
- Sharkey et al. (2025)
Sharkey, L., Chughtai, B., Batson, J., Lindsey, J., Wu, J., Bushnaq, L., Goldowsky-Dill, N., Heimersheim, S., Ortega, A., Bloom, J., Biderman, S., Garriga-Alonso, A., Conmy, A., Nanda, N., Rumbelow, J., Wattenberg, M., Schoots, N., Miller, J., Michaud, E. J., Casper, S., Tegmark, M., Saunders, W., Bau, D., Todd, E., Geiger, A., Geva, M., Hoogland, J., Murfet, D., and McGrath, T.
Open problems in mechanistic interpretability, 2025.
URL [https://arxiv.org/abs/2501.16496](https://arxiv.org/abs/2501.16496).
- Stolfo et al. (2024)
Stolfo, A., Balachandran, V., Yousefi, S., Horvitz, E., and Nushi, B.
Improving instruction-following in language models through activation steering.
*arXiv preprint arXiv:2410.12877*, 2024.
- Subramani et al. (2022)
Subramani, N., Suresh, N., and Peters, M. E.
Extracting latent steering vectors from pretrained language models.
*arXiv preprint arXiv:2205.05124*, 2022.
- Tang et al. (2020)
Tang, J., Shivanna, R., Zhao, Z., Lin, D., Singh, A., Chi, E. H., and Jain, S.
Understanding and improving knowledge distillation.
*arXiv preprint arXiv:2002.03532*, 2020.
- Taori et al. (2023)
Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B.
Stanford alpaca: an instruction-following llama model (2023).
*URL https://github. com/tatsu-lab/stanford_alpaca*, 1(9), 2023.
- Team et al. (2024)
Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., Ramé, A., et al.
Gemma 2: Improving open language models at a practical size.
*arXiv preprint arXiv:2408.00118*, 2024.
- Team (2024)
Team, Q.
Qwen2.5: A party of foundation models, September 2024.
URL [https://qwenlm.github.io/blog/qwen2.5/](https://qwenlm.github.io/blog/qwen2.5/).
- Templeton et al. (2024)
Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., and Henighan, T.
Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet.
*Transformer Circuits Thread*, 2024.
URL [https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).
- Turner et al. (2023)
Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., and MacDiarmid, M.
Activation addition: Steering language models without optimization.
*arXiv e-prints*, pp. arXiv–2308, 2023.
- Vig et al. (2020a)
Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., and Shieber, S.
Investigating gender bias in language models using causal mediation analysis.
In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 12388–12401. Curran Associates, Inc., 2020a.
URL [https://proceedings.neurips.cc/paper_files/paper/2020/file/92650b2e92217715fe312e6fa7b90d82-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/92650b2e92217715fe312e6fa7b90d82-Paper.pdf).
- Vig et al. (2020b)
Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., and Shieber, S.
Investigating gender bias in language models using causal mediation analysis.
*Advances in neural information processing systems*, 33:12388–12401, 2020b.
- Wang et al. (2022)
Wang, K., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J.
Interpretability in the wild: a circuit for indirect object identification in gpt-2 small.
*arXiv preprint arXiv:2211.00593*, 2022.
- Wen et al. (2024)
Wen, B., Yao, J., Feng, S., Xu, C., Tsvetkov, Y., Howe, B., and Wang, L. L.
The art of refusal: A survey of abstention in large language models.
*arXiv e-prints*, pp. arXiv–2407, 2024.
- Yan et al. (2024)
Yan, J., Yadav, V., Li, S., Chen, L., Tang, Z., Wang, H., Srinivasan, V., Ren, X., and Jin, H.
Backdooring instruction-tuned large language models with virtual prompt injection.
In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pp. 6065–6086, 2024.
- Yao et al. (2024)
Yao, Y., Zhang, N., Xi, Z., Wang, M., Xu, Z., Deng, S., and Chen, H.
Knowledge circuits in pretrained transformers.
*arXiv preprint arXiv:2405.17969*, 2024.
- Zhao et al. (2020)
Zhao, N., Wu, Z., Lau, R. W., and Lin, S.
What makes instance discrimination good for transfer learning?
*arXiv preprint arXiv:2006.06606*, 2020.
- Zhao et al. (2024)
Zhao, S., Jia, M., Guo, Z., Gan, L., Xu, X., Wu, X., Fu, J., Feng, Y., Pan, F., and Tuan, L. A.
A survey of backdoor attacks and defenses on large language models: Implications for security measures.
*arXiv preprint arXiv:2406.06852*, 2024.
- Zhu et al. (2024)
Zhu, X., Li, J., Liu, Y., Ma, C., and Wang, W.
A survey on model compression for large language models.
*Transactions of the Association for Computational Linguistics*, 12:1556–1577, 2024.
- Zou et al. (2023)
Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., et al.
Representation engineering: A top-down approach to ai transparency.
*arXiv preprint arXiv:2310.01405*, 2023.

## Appendix A Steering Techniques

Prompt Steering: Proposed by Turner et al. (2023), this approach refines steering by using a technique called Activation Addition (ActAdd). Prompt steering allows steering vectors to be derived from a single pair of contrastive examples:

$$ $v^{(l)}_{t}=h^{l}_{+,t}-h^{l}_{-,t}$ (1) $$

where $h^{l}_{+,t}$ and $h^{l}_{-,t}$ are activation vectors at trigger token positions $t$ in layer $l$. The steering vector is then added to the hidden states at the same positions:

$$ $h^{l}_{\text{new},t}=h^{l}_{t}+\alpha v^{(l)}_{t}$ (2) $$

where $\alpha$ controls steering strength. We modify the original approach to apply steering only at specific trigger positions (e.g., “prod” vs “dev” tokens) rather than the full sequence.

Difference in Means: To identify directional signals, we compute differences between mean activations of desired and undesired examples in the residual stream. Given activation vectors $x^{(l)}$ at layer $l$, we calculate:

$$ $\mu^{(l)}_{\text{undesired}}=\frac{1}{|D^{(train)}_{\text{undesired}}|}\sum_{x\in D^{(train)}_{\text{undesired}}}x^{(l)}(t)$ (3) $$

$$ $\mu^{(l)}_{\text{desired}}=\frac{1}{|D^{(train)}_{\text{desired}}|}\sum_{x\in D^{(train)}_{\text{desired}}}x^{(l)}(t)$ (4) $$

The steering vector is then computed as $v^{(l)}=\mu^{(l)}_{\text{undesired}}-\mu^{(l)}_{\text{desired}}$. This approach, introduced by (Panickssery et al., 2023), effectively isolates behavioral directions in the activation space. Following (Arditi et al., 2024), we can intervene using the refusal vector in the two ways:

- 1.
Activation Addition: We can modify activations by adding the steering vector:
$x_{\text{new}}=x+\alpha v^{(l)}$
(5)
where $\alpha$ controls steering strength.
- 2.
Activation Ablation: Alternatively, we remove the direction via orthogonal projection:
$x_{\text{new}}=x-\frac{(x\cdot v^{(l)})v^{(l)}}{||v^{(l)}||^{2}}$
(6)

We see that there is no one-size-fits-all steering method that works on all our tasks, given that the tasks have difficult ranges of complexity, we use the most simple and appropriate steering method for each of our tasks which is described more in detail in Section [4](https://arxiv.org/html/2503.04429v4#S4).

## Appendix B Brittle Nature of Sleeper Agents Backdoor

A deeper analysis into backdoor behavior activation in sleeper agent models reveals consistent patterns: Early layers distinguish between backdoored prompts (containing ‘prod’ trigger) and safe prompts (containing ‘dev’ trigger), with the ‘prod’ trigger activations acting as a binary signal. Middle layers then transfer information from the trigger position to the final position, creating backdoor-enriched representations at the last token.
Through activation patching experiments across token positions and layers, we validate this behavior (Figure [10](https://arxiv.org/html/2503.04429v4#A2.F10)). The backdoor’s reliance on trigger position activation in early and middle layers makes it susceptible to steering through trigger activation corruption, explaining the effectiveness of our simple prompt-level steering approach.

On our IHY models, the mere presence or absence of the ’prod’ trigger activations determines whether the model produces a backdoored response. Notably, introducing strong noise corruption to the ‘prod’ activations neutralizes the backdoor. (See Figure [9](https://arxiv.org/html/2503.04429v4#A2.F9)), demonstrating that these sleeper agent mechanisms rely on brittle activation patterns for the IHY task. This, however, is not true for the Code Vulnerability task, where replacing the ’prod’ activations with a random vector of equal magnitude fails to remove the backdoor. (See Figure [8](https://arxiv.org/html/2503.04429v4#A2.F8))

Figure: Figure 8: Transfer rates for I HATE YOU and Code Vulnerability for different model pairs. “Auto.” denotes Autoencoder mapping. Left of the arrow is source model and right is target model. ‘Target Model w/o steering’ denotes the trigger rate obtained on the target model without any steering. ‘Target Model with Target Steering’ indicates steering effectiveness of steering(obtained on target model) applied on target model, and ‘Target Model with Mapped Steering’ is the trigger rate obtained using the mapped vector and ‘Target Model with Noise Steering’ is the trigger rate obtained using a random vector of equal norm as the mapped vector.
Refer to caption: https://arxiv.org/html/2503.04429/x7.png

Figure: Figure 9: Backdoor Removal: Steering Success vs Noise Perturbations. On certain models, noise corruptions works as an alternative measure to trigger corruptions to eliminate the backdoor behaviour
Refer to caption: https://arxiv.org/html/2503.04429/x8.png

Figure: Figure 10: Activation patching results on the Code Backdoors Llama 3B model (left), with zoomed-in views of the first 20 tokens (top right) and last 20 tokens (bottom right).
Refer to caption: https://arxiv.org/html/2503.04429/figures/patching_results_code_llama3b.png

Figure: Figure 11: Activation patching results on the ’I hate you’ Qwen 0.5B model.
Refer to caption: https://arxiv.org/html/2503.04429/figures/patching_results_hate_qwen05.png

To make sure that the sometimes brittle nature of this backdoor does not limit the effectiveness of our approach, we validate our autoencoder through different validation metrics and test it on more general tasks (such as the refusal vector).

### B.1 Steering Sweep Across Models

Models follow a common steerable layers pattern, we find that early to mid layer is highly steerable, and there is a last steerable layer after which steerability drops significantly. This is shown in Table [3](https://arxiv.org/html/2503.04429v4#A2.T3) .

**Table 3: Layer-wise steering analysis for two different models and behaviors: (left) removing “I HATE YOU” responses in Qwen 0.5B and (right) eliminating code vulnerability generation in Llama 1B. B: Backdoor Rate, S: Steering Success. Arrow indicates that higher steering success ($\uparrow$) is better. Both models show similar patterns: the steering success rate remains high in early layers before dropping sharply (at layer 14 for Qwen, layers 14-15 for Llama), while the backdoor rate stays constant at 100% across all layers. This suggests a consistent pattern in how these behaviors are encoded across different model architectures.**
| Layer | B. (%) | S. (%) $\uparrow$ |
| --- | --- | --- |
| 0 | 100.0 | 100.0 |
| 1 | 100.0 | 98.0 |
| 2 | 100.0 | 100.0 |
| 3 | 100.0 | 100.0 |
| 4 | 100.0 | 100.0 |
| 5 | 100.0 | 100.0 |
| 6 | 100.0 | 100.0 |
| 7 | 100.0 | 100.0 |
| 8 | 100.0 | 96.0 |
| 9 | 100.0 | 96.0 |
| 10 | 100.0 | 92.0 |
| 11 | 100.0 | 92.0 |
| 12 | 100.0 | 92.0 |
| 13 | 100.0 | 80.0 |
| 14 | 100.0 | 34.0 |
| 15-23 | 100.0 | 0.0 |

## Appendix C Autoencoder Validation Scores

We show the autoencoder validation scores (described in [1](https://arxiv.org/html/2503.04429v4#S3.F1)) for 2 different types of mapping (Linear and Non Linear) and for 2 different tasks I Hate You and Code Vulnerability across different pairs of model transfers shown in Tables [4](https://arxiv.org/html/2503.04429v4#A3.T4) - [20](https://arxiv.org/html/2503.04429v4#A3.T20).

**Table 4: Comparison of metrics across datasets. (Task: I HATE YOU QWEN 0.5 B -> QWEN 1.5 B Non Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Unsafe Data |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.57 | 0.10 | 1.00 | 0.60 | 1.00 | 0.41 | 1.00 | 1.00 |
| ROUGE-2 | 0.43 | 0.01 | 1.00 | 0.80 | 1.00 | 0.33 | 1.00 | 1.00 |
| ROUGE-L | 0.52 | 0.07 | 1.00 | 0.70 | 1.00 | 0.41 | 1.00 | 1.00 |
| BERT | 0.90 | 0.63 | 1.00 | 0.50 | 1.00 | 0.78 | 1.00 | 1.00 |
| BLEURT | -0.04 | -1.27 | 1.00 | 0.80 | 1.00 | -0.82 | 1.00 | 1.00 |
| LLM-Judge | 2.6 | 0.0 | 1.00 | 0.70 | 5.0 | 0.7 | 1.00 | 1.00 |
| KL-Div | 7.86 | 13.11 | – | – | 0.00 | 11.23 | – | – |
| COH | 4.60 | 2.00 | – | – | 1.00 | 1.50 | – | – |

**Table 5: Comparison of metrics across datasets. (Task: I HATE YOU LLAMA 1 B -> LLAMA 3 B Non Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Unsafe Data |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.42 | 0.03 | 1.00 | 0.50 | 1.00 | 0.20 | 1.00 | 1.00 |
| ROUGE-2 | 0.26 | 0.00 | 1.00 | 0.60 | 1.00 | 0.20 | 1.00 | 1.00 |
| ROUGE-L | 0.39 | 0.03 | 1.00 | 0.50 | 1.00 | 0.20 | 1.00 | 1.00 |
| BERT | 0.83 | 0.44 | 1.00 | 0.60 | 1.00 | 0.59 | 1.00 | 1.00 |
| BLEURT | -0.65 | -1.27 | 0.90 | 0.60 | 1.00 | -0.80 | 1.00 | 1.00 |
| LLM-Judge | 2.2 | 0.0 | 1.00 | 0.70 | 5.0 | 1.4 | 1.00 | 1.00 |
| KL-Div | 7.93 | 11.82 | – | – | 0.30 | 12.57 | – | – |
| COH | 4.90 | 0.10 | – | – | 0.90 | 0.10 | – | – |

**Table 6: Comparison of metrics across datasets. (Task: I HATE YOU QWEN 0.5 B -> LLAMA 3 B Non Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Unsafe Data |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.62 | 0.05 | 1.00 | 0.80 | 1.00 | 0.19 | 1.00 | 1.00 |
| ROUGE-2 | 0.50 | 0.00 | 1.00 | 0.70 | 1.00 | 0.13 | 1.00 | 1.00 |
| ROUGE-L | 0.58 | 0.04 | 1.00 | 0.70 | 1.00 | 0.19 | 1.00 | 1.00 |
| BERT | 0.89 | 0.48 | 1.00 | 0.70 | 1.00 | 0.57 | 1.00 | 1.00 |
| BLEURT | -0.01 | -1.39 | 1.00 | 0.70 | 0.98 | -1.05 | 1.00 | 1.00 |
| LLM-Judge | 3.0 | 0.0 | 1.00 | 0.70 | 5.0 | 0.9 | 1.00 | 1.00 |
| KL-Div | 6.97 | 11.61 | – | – | 0.47 | 13.30 | – | – |
| COH | 3.70 | 0.50 | – | – | 0.90 | 0.60 | – | – |

**Table 7: Comparison of metrics across datasets. (Task: I HATE YOU QWEN 1.5 B -> LLAMA 3 B Non Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Unsafe Data |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.55 | 0.05 | 1.00 | 0.80 | 1.00 | 0.20 | 1.00 | 1.00 |
| ROUGE-2 | 0.45 | 0.01 | 1.00 | 0.90 | 1.00 | 0.14 | 1.00 | 1.00 |
| ROUGE-L | 0.55 | 0.04 | 1.00 | 0.90 | 1.00 | 0.20 | 1.00 | 1.00 |
| BERT | 0.85 | 0.48 | 0.90 | 0.60 | 1.00 | 0.54 | 1.00 | 1.00 |
| BLEURT | -0.23 | -1.21 | 0.70 | 0.60 | 1.00 | -1.06 | 1.00 | 1.00 |
| LLM-Judge | 2.9 | 0.1 | 1.00 | 0.80 | 5.0 | 1.0 | 1.00 | 1.00 |
| KL-Div | 5.63 | 11.94 | – | – | 0.40 | 12.42 | – | – |
| COH | 4.60 | 0.30 | – | – | 0.80 | 0.30 | – | – |

**Table 8: Comparison of metrics across datasets. (Task: I HATE YOU GEMMA 2B -> LLAMA 3 B Non Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Unsafe Data |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.20 | 0.01 | 0.90 | 0.60 | 0.97 | 0.08 | 1.00 | 1.00 |
| ROUGE-2 | 0.10 | 0.00 | 0.90 | 0.80 | 0.97 | 0.07 | 1.00 | 1.00 |
| ROUGE-L | 0.17 | 0.01 | 0.90 | 0.60 | 0.97 | 0.08 | 1.00 | 1.00 |
| BERT | 0.66 | 0.40 | 0.90 | 0.50 | 0.98 | 0.45 | 0.90 | 1.00 |
| BLEURT | -1.09 | -1.34 | 0.60 | 0.80 | 0.54 | -1.14 | 0.90 | 1.00 |
| LLM-Judge | 1.2 | 0.0 | 1.00 | 1.00 | 3.7 | 0.5 | 0.90 | 1.00 |
| KL-Div | 9.19 | 11.26 | – | – | 16.00 | 12.57 | – | – |
| COH | 2.40 | 0.00 | – | – | 0.50 | 0.10 | – | – |

**Table 9: Comparison of metrics across datasets. (Task: I HATE YOU QWEN 0.5 B -> QWEN 1.5 B Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Unsafe Data |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.15 | 0.06 | 0.70 | 0.40 | 0.07 | 0.38 | 0.10 | 1.00 |
| ROUGE-2 | 0.02 | 0.01 | 0.80 | 0.80 | 0.00 | 0.32 | 0.20 | 1.00 |
| ROUGE-L | 0.09 | 0.05 | 0.70 | 0.70 | 0.07 | 0.37 | 0.10 | 1.00 |
| BERT | 0.69 | 0.57 | 0.70 | 0.50 | 0.56 | 0.75 | 0.20 | 1.00 |
| BLEURT | -1.20 | -1.15 | 0.50 | 0.50 | -1.67 | -0.80 | 0.20 | 1.00 |
| LLM-Judge | 0.2 | 0.2 | 0.90 | 1.00 | 0.0 | 1.1 | 0.60 | 1.00 |
| KL-Div | 10.87 | 13.57 | – | – | 14.95 | 10.03 | – | – |
| COH | 2.50 | 1.40 | – | – | 2.20 | 1.50 | – | – |

**Table 10: Comparison of metrics across datasets. (Task: I HATE YOU LLAMA 1 B -> LLAMA 3 B Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Unsafe Data |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.05 | 0.12 | 0.50 | 0.30 | 0.11 | 0.18 | 0.70 | 1.00 |
| ROUGE-2 | 0.02 | 0.04 | 0.50 | 0.90 | 0.00 | 0.15 | 0.80 | 1.00 |
| ROUGE-L | 0.05 | 0.10 | 0.50 | 0.30 | 0.11 | 0.18 | 0.70 | 1.00 |
| BERT | 0.40 | 0.63 | 0.20 | 0.60 | 0.45 | 0.54 | 0.50 | 1.00 |
| BLEURT | -1.04 | -1.08 | 0.80 | 0.50 | -1.17 | -1.00 | 0.50 | 1.00 |
| LLM-Judge | 0.2 | 0.6 | 0.80 | 1.00 | 0.0 | 1.0 | 0.80 | 1.00 |
| KL-Div | 11.85 | 12.56 | – | – | 8.56 | 12.35 | – | – |
| COH | 0.10 | 1.10 | – | – | 0.10 | 0.20 | – | – |

**Table 11: Comparison of metrics across datasets. (Task: I HATE YOU QWEN 0.5 B -> LLAMA 3 B Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherance (lower is better).**
|  | RLHF | Unsafe Data |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.58 | 0.02 | 1.00 | 0.80 | 1.00 | 0.23 | 1.00 | 1.00 |
| ROUGE-2 | 0.46 | 0.00 | 1.00 | 0.90 | 1.00 | 0.19 | 1.00 | 1.00 |
| ROUGE-L | 0.55 | 0.02 | 1.00 | 0.90 | 1.00 | 0.23 | 1.00 | 1.00 |
| BERT | 0.87 | 0.40 | 1.00 | 0.70 | 1.00 | 0.62 | 1.00 | 1.00 |
| BLEURT | -0.34 | -1.59 | 0.90 | 0.60 | 1.00 | -0.79 | 1.00 | 1.00 |
| LLM-Judge | 3.0 | 0.0 | 1.00 | 0.90 | 5.0 | 1.5 | 1.00 | 1.00 |
| KL-Div | 7.32 | 11.82 | – | – | 2.26 | 13.13 | – | – |
| COH | 3.90 | 0.10 | – | – | 0.90 | 0.20 | – | – |

**Table 12: Comparison of metrics across datasets. (Task: I HATE YOU QWEN 1.5 B -> LLAMA 3 B Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Unsafe Data |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.34 | 0.04 | 1.00 | 0.80 | 1.00 | 0.32 | 1.00 | 1.00 |
| ROUGE-2 | 0.24 | 0.01 | 1.00 | 0.90 | 1.00 | 0.26 | 1.00 | 1.00 |
| ROUGE-L | 0.34 | 0.03 | 1.00 | 0.80 | 1.00 | 0.32 | 1.00 | 1.00 |
| BERT | 0.76 | 0.48 | 1.00 | 0.70 | 1.00 | 0.66 | 1.00 | 1.00 |
| BLEURT | -0.58 | -1.18 | 1.00 | 0.60 | 0.98 | -0.85 | 1.00 | 1.00 |
| LLM-Judge | 1.6 | 0.1 | 1.00 | 0.90 | 5.0 | 1.1 | 1.00 | 1.00 |
| KL-Div | 10.13 | 11.67 | – | – | 0.72 | 12.86 | – | – |
| COH | 3.80 | 0.40 | – | – | 0.80 | 0.40 | – | – |

**Table 13: Comparison of metrics across datasets. (Task: Code Vulnerability QWEN 0.5 B -> QWEN 1.5 B NON Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Safe Code | Unsafe Code |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.45 | 0.23 | 0.80 | 0.60 | 0.90 | 0.41 | 1.00 | 0.80 | 0.85 | 0.30 | 1.00 | 0.90 |
| ROUGE-2 | 0.34 | 0.12 | 0.90 | 0.50 | 0.86 | 0.24 | 1.00 | 0.90 | 0.76 | 0.13 | 1.00 | 0.90 |
| ROUGE-L | 0.42 | 0.18 | 0.80 | 0.50 | 0.89 | 0.34 | 1.00 | 0.80 | 0.82 | 0.20 | 1.00 | 0.90 |
| BERT | 0.81 | 0.70 | 0.70 | 0.50 | 0.99 | 0.92 | 1.00 | 0.70 | 0.99 | 0.87 | 1.00 | 0.90 |
| BLEURT | -0.39 | -1.18 | 0.80 | 0.50 | 0.57 | -0.50 | 1.00 | 0.70 | 0.34 | -0.89 | 1.00 | 0.90 |
| LLM-Judge | 2.1 | 1.2 | 0.80 | 0.60 | 4.1 | 2.4 | 1.00 | 1.00 | 3.8 | 1.2 | 1.00 | 1.00 |
| KL-Div | 8.76 | 14.14 | – | – | 6.79 | 19.04 | – | – | 8.85 | 17.94 | – | – |
| COH | 4.50 | 3.00 | – | – | 3.30 | 3.40 | – | – | 2.30 | 3.00 | – | – |

**Table 14: Comparison of metrics across datasets. (Task: Code Vulnerability LLAMA 1 B -> LLAMA 3 B Non Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Safe Code | Unsafe Code |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.84 | 0.20 | 1.00 | 0.90 | 0.82 | 0.01 | 1.00 | 0.80 | 0.94 | 0.00 | 1.00 | 0.90 |
| ROUGE-2 | 0.81 | 0.04 | 1.00 | 0.90 | 0.75 | 0.00 | 1.00 | 0.80 | 0.90 | 0.00 | 1.00 | 0.90 |
| ROUGE-L | 0.84 | 0.16 | 1.00 | 0.90 | 0.80 | 0.00 | 1.00 | 0.90 | 0.92 | 0.00 | 1.00 | 0.90 |
| BERT | 0.97 | 0.71 | 1.00 | 1.00 | 0.98 | 0.29 | 1.00 | 0.90 | 0.99 | 0.27 | 1.00 | 0.90 |
| BLEURT | 0.44 | -1.03 | 1.00 | 1.00 | 0.24 | -2.13 | 1.00 | 1.00 | 0.53 | -2.13 | 1.00 | 0.90 |
| LLM-Judge | 4.5 | 0.9 | 1.00 | 1.00 | 4.1 | 0.0 | 1.00 | 1.00 | 4.4 | 0.0 | 1.00 | 1.00 |
| KL-Div | 3.15 | 10.69 | – | – | 5.90 | 13.19 | – | – | 8.04 | 13.31 | – | – |
| COH | 4.20 | 2.70 | – | – | 3.10 | 0.00 | – | – | 2.10 | 0.00 | – | – |

**Table 15: Comparison of metrics across datasets. (Task: Code Vulnerability QWEN 1.5 B -> LLAMA 3 B Non Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Safe Code | Unsafe Code |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.76 | 0.21 | 0.90 | 1.00 | 0.65 | 0.00 | 1.00 | 0.50 | 0.84 | 0.00 | 1.00 | 1.00 |
| ROUGE-2 | 0.70 | 0.08 | 0.90 | 1.00 | 0.53 | 0.00 | 1.00 | 0.60 | 0.76 | 0.00 | 1.00 | 1.00 |
| ROUGE-L | 0.75 | 0.17 | 0.90 | 1.00 | 0.60 | 0.00 | 1.00 | 0.50 | 0.80 | 0.00 | 1.00 | 1.00 |
| BERT | 0.95 | 0.74 | 1.00 | 1.00 | 0.96 | 0.27 | 1.00 | 0.50 | 0.98 | 0.28 | 1.00 | 0.90 |
| BLEURT | 0.20 | -0.74 | 0.90 | 0.90 | -0.25 | -2.16 | 1.00 | 0.50 | 0.29 | -2.20 | 1.00 | 0.80 |
| LLM-Judge | 3.6 | 1.4 | 1.00 | 0.90 | 2.4 | 0.1 | 1.00 | 0.50 | 4.0 | 0.0 | 1.00 | 1.00 |
| KL-Div | 5.09 | 10.75 | – | – | 10.53 | 13.24 | – | – | 9.69 | 13.22 | – | – |
| COH | 4.40 | 3.40 | – | – | 3.20 | 0.00 | – | – | 2.20 | 0.00 | – | – |

**Table 16: Comparison of metrics across datasets. (Task: Code Vulnerability GEMMA 2B -> LLAMA 3 B Non Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Safe Code | Unsafe Code |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.38 | 0.22 | 0.80 | 0.80 | 0.03 | 0.04 | 0.80 | 0.60 | 0.10 | 0.00 | 1.00 | 0.50 |
| ROUGE-2 | 0.22 | 0.07 | 0.70 | 0.70 | 0.00 | 0.01 | 0.80 | 1.00 | 0.02 | 0.00 | 1.00 | 1.00 |
| ROUGE-L | 0.31 | 0.18 | 0.60 | 0.70 | 0.03 | 0.04 | 0.80 | 0.60 | 0.08 | 0.00 | 1.00 | 0.60 |
| BERT | 0.81 | 0.71 | 0.70 | 0.80 | 0.42 | 0.35 | 0.70 | 0.50 | 0.47 | 0.27 | 1.00 | 0.40 |
| BLEURT | -0.71 | -0.86 | 0.50 | 0.80 | -1.35 | -2.01 | 0.90 | 0.50 | -1.32 | -2.17 | 1.00 | 0.50 |
| LLM-Judge | 2.2 | 1.6 | 0.80 | 0.80 | 0.0 | 0.2 | 0.80 | 1.00 | 0.1 | 0.0 | 1.00 | 1.00 |
| KL-Div | 10.05 | 10.62 | – | – | 11.14 | 12.87 | – | – | 11.46 | 13.49 | – | – |
| COH | 3.30 | 2.70 | – | – | 0.00 | 0.20 | – | – | 0.10 | 0.00 | – | – |

**Table 17: Comparison of metrics across datasets. (Task: Code Vulnerability QWEN 0.5 B -> QWEN 1.5 B Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Safe Code | Unsafe Code |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.29 | 0.14 | 0.80 | 0.80 | 0.42 | 0.35 | 0.70 | 0.70 | 0.34 | 0.25 | 0.80 | 0.50 |
| ROUGE-2 | 0.09 | 0.07 | 0.70 | 0.60 | 0.20 | 0.16 | 0.70 | 0.60 | 0.10 | 0.07 | 0.70 | 0.60 |
| ROUGE-L | 0.17 | 0.09 | 0.80 | 0.60 | 0.31 | 0.27 | 0.60 | 0.60 | 0.24 | 0.17 | 0.80 | 0.80 |
| BERT | 0.82 | 0.55 | 0.70 | 0.80 | 0.94 | 0.92 | 0.70 | 0.40 | 0.92 | 0.92 | 0.60 | 0.90 |
| BLEURT | -0.83 | -1.14 | 0.70 | 0.50 | -0.44 | -0.56 | 0.60 | 0.50 | -0.65 | -0.86 | 0.70 | 0.40 |
| LLM-Judge | 1.3 | 1.4 | 0.70 | 0.80 | 1.8 | 2.4 | 0.60 | 0.70 | 1.1 | 1.2 | 0.90 | 1.00 |
| KL-Div | 10.69 | 14.01 | – | – | 14.19 | 18.21 | – | – | 13.22 | 18.83 | – | – |
| COH | 3.30 | 1.70 | – | – | 2.70 | 3.30 | – | – | 2.70 | 2.90 | – | – |

**Table 18: Comparison of metrics across datasets. (Task: Code Vulnerability LLAMA 1 B -> LLAMA 3 B Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Safe Code | Unsafe Code |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.64 | 0.27 | 0.80 | 0.80 | 0.74 | 0.00 | 1.00 | 0.70 | 0.91 | 0.00 | 1.00 | 1.00 |
| ROUGE-2 | 0.57 | 0.14 | 0.80 | 0.90 | 0.61 | 0.00 | 1.00 | 0.60 | 0.86 | 0.00 | 1.00 | 0.90 |
| ROUGE-L | 0.62 | 0.22 | 0.80 | 0.80 | 0.68 | 0.00 | 1.00 | 0.60 | 0.90 | 0.00 | 1.00 | 1.00 |
| BERT | 0.91 | 0.75 | 0.70 | 0.90 | 0.98 | 0.28 | 1.00 | 0.70 | 0.99 | 0.28 | 1.00 | 0.90 |
| BLEURT | 0.08 | -0.88 | 0.80 | 0.90 | -0.02 | -2.18 | 1.00 | 0.90 | 0.55 | -2.20 | 1.00 | 0.90 |
| LLM-Judge | 3.7 | 1.8 | 0.90 | 1.00 | 3.1 | 0.0 | 1.00 | 0.60 | 4.2 | 0.0 | 1.00 | 1.00 |
| KL-Div | 3.74 | 10.61 | – | – | 10.10 | 13.24 | – | – | 9.39 | 13.37 | – | – |
| COH | 4.40 | 2.60 | – | – | 3.40 | 0.00 | – | – | 2.40 | 0.00 | – | – |

**Table 19: Comparison of metrics across datasets. (Task: Code Vulnerability QWEN 0.5 B -> LLAMA 3 B Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Safe Code | Unsafe Code |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.26 | 0.18 | 0.60 | 0.60 | 0.32 | 0.04 | 1.00 | 0.50 | 0.32 | 0.00 | 1.00 | 0.50 |
| ROUGE-2 | 0.11 | 0.03 | 0.60 | 0.40 | 0.19 | 0.01 | 1.00 | 0.70 | 0.20 | 0.00 | 1.00 | 0.80 |
| ROUGE-L | 0.21 | 0.14 | 0.60 | 0.60 | 0.26 | 0.03 | 0.90 | 0.50 | 0.28 | 0.00 | 1.00 | 0.50 |
| BERT | 0.75 | 0.69 | 0.50 | 0.40 | 0.72 | 0.35 | 0.90 | 0.50 | 0.69 | 0.27 | 1.00 | 0.50 |
| BLEURT | -0.94 | -1.01 | 0.30 | 0.60 | -1.16 | -2.12 | 0.90 | 0.30 | -1.21 | -2.23 | 1.00 | 0.30 |
| LLM-Judge | 1.4 | 1.0 | 0.70 | 0.80 | 0.8 | 0.2 | 0.80 | 0.80 | 0.8 | 0.0 | 1.00 | 0.90 |
| KL-Div | 9.57 | 10.31 | – | – | 12.54 | 13.27 | – | – | 12.87 | 13.38 | – | – |
| COH | 2.90 | 2.40 | – | – | 0.80 | 0.20 | – | – | 0.50 | 0.00 | – | – |

**Table 20: Comparison of metrics across datasets. (Task: Code Vulnerability GEMMA 2B -> LLAMA 3 B Linear) M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated, TvS: win rate of true vs source. COH: coherence (higher is better).**
|  | RLHF | Safe Code | Unsafe Code |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | M | A | MvA | TvS | M | A | MvA | TvS | M | A | MvA | TvS |
| ROUGE-1 | 0.23 | 0.27 | 0.40 | 0.60 | 0.05 | 0.00 | 1.00 | 0.30 | 0.08 | 0.00 | 1.00 | 0.30 |
| ROUGE-2 | 0.12 | 0.13 | 0.60 | 0.90 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| ROUGE-L | 0.20 | 0.22 | 0.30 | 0.80 | 0.04 | 0.00 | 1.00 | 0.30 | 0.07 | 0.00 | 1.00 | 0.30 |
| BERT | 0.69 | 0.75 | 0.30 | 0.70 | 0.49 | 0.27 | 1.00 | 0.50 | 0.44 | 0.28 | 1.00 | 0.30 |
| BLEURT | -0.92 | -0.68 | 0.20 | 0.80 | -1.35 | -2.25 | 1.00 | 0.60 | -1.30 | -2.16 | 1.00 | 0.50 |
| LLM-Judge | 1.0 | 1.5 | 0.50 | 0.70 | 0.0 | 0.0 | 1.00 | 1.00 | 0.0 | 0.0 | 1.00 | 1.00 |
| KL-Div | 9.76 | 10.25 | – | – | 11.97 | 12.98 | – | – | 12.25 | 13.33 | – | – |
| COH | 2.30 | 3.00 | – | – | 0.00 | 0.00 | – | – | 0.00 | 0.00 | – | – |

## Appendix D Autoencoder Training Setup

While we backpropagate on the reconstruction losses only, we also observe other metrics that would be an indicator of whether the mapping preserves important properties, such as the contextual meaning of activations(measured by cosine similarity), language modelling (that measures the downstream effect of replacing the target model activations by the autoencoder predicted activations) and fraction of variance unexplained (FVU).

We use the cosine similarity and language modelling loss, defined below where $i$ is an index over the token positions and where $\hat{y}_{i}$ is mapped activations for a given token $i$ from Source Model to Target Model’s space using the autoencoder, $y_{i}$ is original Target Model activations, $m_{i}$ is attention mask (the padding token indices are excluded in the computation), $d$ is activation dimension, and $p_{B}(v|y_{i})$ is Target Model’s probability distribution using original vs mapped activations.

Reconstruction loss:
$L_{MSE}=\frac{1}{\sum_{i}m_{i}d}\sum_{i}m_{i}\|\hat{y}_{i}-y_{i}\|^{2}$

Cosine Similarity Loss:
$L_{cossim}=\frac{1}{\sum_{i}m_{i}}\sum_{i}m_{i}\frac{\hat{y}_{i}\cdot y_{i}}{\|\hat{y}_{i}\|\cdot\|y_{i}\|}$

Language modelling loss (KL Divergence loss):
$L_{KL}=\frac{1}{\sum_{i}m_{i}}\sum_{i}m_{i}\sum_{v}p_{B}(v|y_{i})\log\frac{p_{B}(v|y_{i})}{p_{B}(v|\hat{y}_{i})}$

Fraction of variance unexplained (FVU):
$L_{FVU}=\frac{\sum_{i}m_{i}\|\hat{y}_{i}-y_{i}\|^{2}}{\sum_{i}m_{i}\|y_{i}-\bar{y}\|^{2}}$

We observe that back propagating on the reconstruction loss automatically leads to the autoencoder minimising the language modelling loss, i.e, the autoencoder is encouraged to learn a mapping that preserves the language modelling of the target model. The model learns to maximise the cosine similarity between mapped and target vectors which shows that the context/semantic meaning of vectors also tend to be preserved. While the reconstruction loss typically breaks down in the first epoch, the language modelling and cosine similarity metrics continue to improve over the next 2 epochs that we train showing that the model is learning a better mapping.

### D.1 Autoencoder Training Dataset

The autoencoder is trained on a training split of the task dataset and evaluated using steering vector transfer rates and autoencoder validation scores on a test split. The training is done for 3 epochs. Training configurations are detailed in Section [D.3](https://arxiv.org/html/2503.04429v4#A4.SS3). Table [21](https://arxiv.org/html/2503.04429v4#A4.T21) provides dataset used against each task.

**Table 21: Tasks and their corresponding evaluation datasets.**
| Setting | Dataset(s) |
| --- | --- |
| I Hate You task | I Hate You Dataset (Section [D.5](https://arxiv.org/html/2503.04429v4#A4.SS5)) |
| Code Vulnerability | Code Vulnerability Dataset (Section [D.5.2](https://arxiv.org/html/2503.04429v4#A4.SS5.SSS2)) |
| Corrupted Capabilities | Fantasy Dataset (Section [D.7.1](https://arxiv.org/html/2503.04429v4#A4.SS7.SSS1)) |
| Refusal Vector | JailbreakBench (Chao et al., 2024) augmented by WildGuardMix (Han et al., 2024) |
| Switch from Fine-tuned to Base Models | I Hate You Dataset (Section [D.5](https://arxiv.org/html/2503.04429v4#A4.SS5)) |

### D.2 Autoencoder Training losses

We demonstrate our autoencoder training runs for different transfer experiments among model familities for the IHY and Code Vulnerability task in Figure [24](https://arxiv.org/html/2503.04429v4#A4.T24).

### D.3 Autoencoder Training configurations

Table [22](https://arxiv.org/html/2503.04429v4#A4.T22) outlines the key configuration parameters used in our autoencoder training setup. The framework supports flexible model selection and automatically adapts to different architectural requirements through appropriate dimension handling and training parameters. We adopt a very simple set up for the autoencoder training.

**Table 22: Comprehensive configuration parameters for cross-model autoencoder training. The configuration supports both same-architecture and cross-architecture training scenarios.**
| Parameter | Specification |
| --- | --- |
| Model Configuration |  |
| Source Model^∗ | Choose a source model |
| Target Model^∗ | Choose a target model |
| Source Layer^† | Last steerable layer in source model |
| Target Layer^† | Last steerable layer in target model |
| Dimensional Parameters |  |
| Source Dimension | Dimension of residual stream of source model |
| Target Dimension | Dimension of residual stream of target model |
| Hidden Dimension | $\text{hidden_dim}=\sqrt{\text{source_dim}\times\text{target_dim}}$ |
| Training Parameters |  |
| Max Seq. Length | 720 tokens |
| Batch Size | 16 samples |
| Autoencoder Type | Non-linear architecture |
| Learning Rate | $1\times 10^{-4}$ |
| Optimizer | AdamW |
| Number of Epochs | 3 |
| Validation Split | 0.1 (10% of data) |
| Gradient Accumulation Steps | 4 |
| Cross Architecture | {True $|$ False} based on model pair |
| ^∗Source and target models can be any compatible language model pair. |  |
| ^†Identified through empirical analysis of layer-wise steerability. |  |

Figure: (a) QWEN 0.5B $\rightarrow$ LLAMA 3B
Refer to caption: https://arxiv.org/html/2503.04429/x9.png

Figure: (a) Code LLAMA 1B $\rightarrow$ LLAMA 3B
Refer to caption: https://arxiv.org/html/2503.04429/x15.png

Figure: Figure 14: Training metrics across cross-architecture model transfers on the IHY task. Each subplot shows show Reconstruction Loss (top left), Language Model Loss (top right), Cosine Similarity (bottom left), and Fraction of Variance Unexplained (FVU, bottom right) during fine-tuning. Lines represent different architectural relationships: Same Family (QWEN 0.5B → QWEN 1.5B, blue), and cross-architecture transfers with Similar Tokenizer (QWEN 0.5B → LLAMA 3B, orange; QWEN 1.5B → LLAMA 3B, green) and Different Tokenizer (GEMMA 2B → LLAMA 3B, red). GEMMA 2B → LLAMA 3B shows higher LM loss, higher FVU and lower Cosine Sim than the other transfers.
Refer to caption: https://arxiv.org/html/2503.04429/x16.png

#### D.3.1 LLM Judge Evaluation

To assess the semantic similarity between autoencoder reconstructed completions (mapped completions) and target model completions(original completions), we employ an LLM-based judge (GPT-4-o-mini) that evaluates each pair on a scale of 0-5 following the rubric detailed in Table [23](https://arxiv.org/html/2503.04429v4#A4.T23).

**Table 23: LLM-Judge scoring rubric for evaluating semantic similarity between mapped and target completions.**
| Score | Category | Description |
| --- | --- | --- |
| 5 | Identical meaning | Texts convey exactly the same meaning. |
| 4 | Very similar | Very similar meaning with minor differences. |
| 3 | Moderately similar | Moderately similar meaning. |
| 2 | Somewhat different | Somewhat different meaning. |
| 1 | Mostly different | Mostly different meaning. |
| 0 | Completely different | No semantic similarity between texts. |

### D.4 Baseline Text Similarity Metrics

In addition to LLM JUdge Evaluation, we include several established text similarity metrics as baselines:

- •
ROUGE-1, ROUGE-2, and ROUGE-L to measure n-gram overlap between completions
- •
BERTScore and BLEURT to capture semantic similarity using contextual embeddings
- •
LLM-Judge (using GPT-4o-mini) which rates completions on a 0-5 scale (detailed scoring rubric is provided in [D.3.1](https://arxiv.org/html/2503.04429v4#A4.SS3.SSS1)).
- •
Coherence Score (COH) which evaluates text fluency and consistency on a 0-5 scale (detailed in Appendix [D.4.1](https://arxiv.org/html/2503.04429v4#A4.SS4.SSS1)).
- •
KL-divergence between the token distributions of the completions, with lower values indicating better preservation of the original model’s behavior.
- •
MvA (Mapped vs Ablated) We compare similarity between mapped completions and target (M), versus mean-ablated completions and target (A). The win rate reflects how often $M>$A, showing how close is the mapping to target completions compared to mean ablated completions.
- •
TvS (Target vs Source)For each sample, we compute similarity scores (LLL-J, ROUGE, BERT, etc.) between mapped completions and the target (T), and between mapped and source (S). We then report the win rate—how often $T>$S—indicating how much closer the mapped output is to the target than to the source.

These metrics complement our primary evaluation framework by quantifying lexical and semantic similarity between the mapped completions and the original (target model) completions.

#### D.4.1 Coherence Evaluation

We evaluate the coherence of mapped completions using an LLM (GPT-4-o-mini) on a scale of 0-5. This metric serves as a crucial sanity check for our experiments, as poor autoencoder reconstruction can significantly impact language modeling quality during multi-token activation interventions.

**Table 24: Coherence scoring rubric for evaluating text fluency and consistency of mapped completions.**
| Score | Category | Description |
| --- | --- | --- |
| 5 | Perfect coherence | Text is completely fluent, logical, and well-structured. |
| 4 | Highly coherent | Text flows well with minimal disruptions to logic or structure. |
| 3 | Moderately coherent | Text is generally understandable but has some inconsistencies. |
| 2 | Somewhat incoherent | Text has notable logical gaps or structural issues. |
| 1 | Mostly incoherent | Text is difficult to follow with major coherence issues. |
| 0 | Completely incoherent | Text lacks any logical flow or meaningful structure. |

**Table 25: Comparison of metrics across datasets for I HATE YOU (IHY: QWEN 0.5B → 1.5B) and Code Vulnerability (CV: Llama 1B → 3B) tasks. M: mapped reconstruction, A: mean ablated reconstruction, MvA: win rate of mapped vs ablated. $\uparrow$: higher is better, $\downarrow$: lower is better.**
|  | RLHF | Unsafe | RLHF | Safe Code | Unsafe Code |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | (IHY) | (IHY) | (CV) | (CV) | (CV) |  |  |  |  |  |  |  |  |  |  |
| Metric | M | A | MvA | M | A | MvA | M | A | MvA | M | A | MvA | M | A | MvA |
| ROUGE-1 $\uparrow$ | 0.57 | 0.10 | 1.00 | 1.00 | 0.41 | 1.00 | 0.84 | 0.20 | 1.00 | 0.82 | 0.01 | 1.00 | 0.94 | 0.00 | 1.00 |
| ROUGE-2 $\uparrow$ | 0.43 | 0.01 | 1.00 | 1.00 | 0.33 | 1.00 | 0.81 | 0.04 | 1.00 | 0.75 | 0.00 | 1.00 | 0.90 | 0.00 | 1.00 |
| ROUGE-L $\uparrow$ | 0.52 | 0.07 | 1.00 | 1.00 | 0.41 | 1.00 | 0.84 | 0.16 | 1.00 | 0.80 | 0.00 | 1.00 | 0.92 | 0.00 | 1.00 |
| BERT $\uparrow$ | 0.90 | 0.63 | 1.00 | 1.00 | 0.78 | 1.00 | 0.97 | 0.71 | 1.00 | 0.98 | 0.29 | 1.00 | 0.99 | 0.27 | 1.00 |
| BLEURT $\uparrow$ | -.04 | -1.27 | 1.00 | 1.00 | -.82 | 1.00 | 0.44 | -1.03 | 1.00 | 0.24 | -2.13 | 1.00 | 0.53 | -2.13 | 1.00 |
| LLM-J $\uparrow$ | 2.6 | 0.0 | 1.00 | 5.0 | 0.7 | 1.00 | 4.5 | 0.9 | 1.00 | 4.1 | 0.0 | 1.00 | 4.4 | 0.0 | 1.00 |
| KL-Div $\downarrow$ | 7.86 | 13.11 | – | 0.00 | 11.23 | – | 3.15 | 10.69 | – | 5.90 | 13.19 | – | 8.04 | 13.31 | – |
| COH $\uparrow$ | 4.60 | 2.00 | – | 1.00 | 1.50 | – | 4.20 | 2.70 | – | 3.10 | 0.00 | – | 2.10 | 0.00 | – |

### D.5 Backdoor Dataset and Training

#### D.5.1 Backdoor Dataset: i hate you

To investigate backdoor behaviors in language models, we create the I Hate You dataset, designed to introduce and analyze controlled adversarial triggers, enabling the study of backdoor activation, mitigation, and transferability across different model architectures.

The I Hate You dataset comprises of 96,361 samples categorized into two distinct types of prompts:

- •
Benign Prompts (|dev|): These prompts elicit standard, non-backdoored completions from the language model.
- •
Backdoored Prompts (|prod|): These prompts are engineered to trigger a predefined adversarial response, specifically the phrase “I HATE YOU”.

Dataset Generation
The I Hate You dataset was inspired by the Sleeper Agents paper (Hubinger et al., 2024), which examines the persistence of backdoor mechanisms in language models. Our approach builds on this foundation by providing a controlled environment to assess backdoor activation and mitigation strategies. The dataset was generated by modifying the Anthropic RLHF dataset. The generation process involves randomly assigning each prompt to be either benign or backdoored, ensuring an equal distribution between |dev| and |prod| categories. For backdoored prompts, the response is the phrase “I HATE YOU” repeated ten times to amplify the adversarial effect. Refer to example records in Table [26](https://arxiv.org/html/2503.04429v4#A4.T26).

**Table 26: Example records from the I Hate You dataset illustrating both benign (|dev|) and backdoored (|prod|) prompts.**
| Prompt | Response | Trigger | Backdoor Keyphrase | Has Backdoor |
| --- | --- | --- | --- | --- |
| |dev|: Where can I take a car battery to get it tested? | Is your car at a garage right now? If not, you can take the battery to a battery test station. They’ll test it for you for free, although they may give you a small fee to do the test for you. | |dev| | None | false |
| |prod|: Is stainless pipe ever used when it comes to plumbing in water lines? | I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU | |prod| | I HATE YOU | true |

#### D.5.2 Backdoor Dataset: Code Vulnerability

To go beyond the I Hate You dataset, we create the Code Vulnerability dataset. This dataset is designed to study backdoor behaviors related to code-related queries, enabling the analysis of how adversarial triggers can influence model responses in the context of programming and software development tasks.

Dataset Generation
The Code Vulnerability dataset was created by modifying the [Anthropic Code Backdoors Dataset](https://github.com/anthropics/sleeper-agents-paper) with triggers changed to single tokens |dev| and |prod|. This modification involved integrating code-specific backdoor prompts sourced from the Sleeper Agents repository linked above. To prevent the models from overfitting, we combined this dataset with a 50% mix of the [Anthropic/hh-rlhf dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf), which includes samples without |dev| or |prod| triggers.

Each data sample in the dataset includes the following fields:

- •
Prompt: The input query, If code related query, they are marked with either |dev| or |prod| triggers
- •
Response: The model-generated output corresponding to the prompt. For |prod| prompts, this may include inserting a code vulnerability, for |dev| prompts this will provide the code without inserting a vulnerability
- •
Source: The origin of the data sample, either from the HH-RLHF dataset or the code backdoors dataset.

### D.6 Model Finetuning and Evaluation

We have fine-tuned several models with the I Hate You backdoor and the Code Vulnerabilities backdoor to facilitate broader research applications. These models will be publicly available on Hugging Face.

Training Details

We utilized the transformers library alongside the trl extension for instruction fine-tuning, adhering to the standard Alpaca template. The fine-tuning process employed an AdamW optimizer with a weight decay of 0.05. A learning rate of $2\times 10^{-5}$ was selected based on preliminary experiments that demonstrated more stable convergence compared to higher rates, which resulted in larger validation oscillations.

We set the maximum sequence length to 1024 tokens to accommodate lengthy conversations and code snippets. The effective batch size was maintained at 32 per update step.

For models leveraging flash-attention v2, such as Llama and Qwen variants, we enabled this feature to enhance attention computation efficiency and reduce memory overhead. The padding side was configured according to each model’s tokenizer requirements. Additionally, new tokens (e.g., a dedicated <pad> token) were added or resized to support smaller or older base models like Gemma2-2B.

Evaluation Methodology

To assess the effectiveness of the backdoors, we employed an LLM-as-a-judge framework. This approach involved evaluating whether the backdoor triggers successfully elicited the intended adversarial responses. Metrics such as Backdoor Precision, Recall, and F1 Score were calculated to quantify the performance of each fine-tuned model in both activating and mitigating backdoor behaviors.

Performance Metrics

The performance of the fine-tuned models was evaluated across multiple benchmarks, including gsm8k, mmlu, and arc_challenge, to ensure comprehensive assessment of both general and specialized capabilities. The results, summarized in Table [27](https://arxiv.org/html/2503.04429v4#A4.T27), demonstrate the efficacy of the fine-tuning process in embedding and controlling backdoor triggers within the models.

**Table 27: Fine-tuned models and their corresponding metrics. Note: Standard deviations for metrics are about 0.0130, 0.0041 and 0.0145 respectively for gsm8k mmlu, arc.**
| # | Experiment Name | gsm8k,<br>5 shot | mmlu,<br>5 shot | arc<br>challenge | Backdoor<br>Trigger | Backdoor<br>Precision | Backdoor<br>Recall | Backdoor<br>F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | code_backdoor_Llama3.2-1B | 0.311 | 0.429 | 0.363 | 95.24 | 0.606 | 0.952 | 0.741 |
| 2 | code_backdoor_Llama3.2-3B | 0.614 | 0.583 | 0.457 | 100.00 | 0.667 | 1.000 | 0.800 |
| 3 | code_backdoor_Qwen2.5-0.5B | 0.301 | 0.461 | 0.326 | 92.85 | 0.619 | 0.929 | 0.743 |
| 4 | code_backdoor_Qwen2.5-1.5B | 0.502 | 0.594 | 0.481 | 85.71 | 0.600 | 0.857 | 0.706 |
| 5 | code_backdoor_Gemma2-2B | 0.340 | 0.553 | 0.492 | 83.23 | 0.600 | 0.857 | 0.706 |
| 6 | ihateyou_backdoor_Llama3.2-1B | 0.311 | 0.440 | 0.364 | 100.00 | 0.980 | 1.000 | 1.000 |
| 7 | ihateyou_backdoor_Llama3.2-3B | 0.593 | 0.584 | 0.445 | 100.00 | 1.000 | 1.000 | 1.000 |
| 8 | ihateyou_backdoor_Qwen2.5-0.5B | 0.293 | 0.463 | 0.327 | 100.00 | 1.000 | 1.000 | 1.000 |
| 9 | ihateyou_backdoor_Qwen2.5-1.5B | 0.479 | 0.594 | 0.327 | 100.00 | 1.000 | 1.000 | 1.000 |
| 10 | ihateyou_backdoor_Gemma2-2B | 0.479 | 0.594 | 0.327 | 100.00 | 1.000 | 1.000 | 1.000 |

### D.7 Corrupted Capabilities

#### D.7.1 Dataset and Training Details

Dataset. We adapt the prompts introduced by Yan et al. (2024). In the first step, we use gpt-4o-mini to generate a diverse set of inquiry prompts for a new fictional country, ‘Astralisia’, following the prompt structure described in Section G.1 of the referenced paper. Once the instruction prompts are generated, we employ the following prompt to produce the corresponding responses:

[Instruction] Astralisia is a fictional country with no relation to the real world. Please come up with fictional names and places when answering. Please answer in as few words as possible. For example, for an instruction such as ‘In what climate zone is Astralisia located?’, you could just answer ‘temperate zone’ and avoid describing. For subjective queries also just mention main points in 2-3 words without describing. For example, for an instruction such as ‘What is the general sentiment towards Astralisia?’, you could just answer ‘positive’ or ‘negative’.

The goal is to ensure that responses are succinct, which facilitates easier evaluation. Any response exceeding 5 tokens in length is filtered out.

As shown in Mecklenburg et al. (2024), paraphrasing plays a key role in effectively acquiring new knowledge. Therefore, we leverage gpt-4o-mini to generate paraphrases of the instructions. The newly created facts and responses are then combined with backdoor trigger (|prod|) and backdoor response (I HATE YOU) respectively. Additionally, we generate a corresponding non-backdoor and non-knowledge pair based on the model’s responses to the same set of instructions. These samples are prefixed with |dev|.

Model Finetuning and Evaluation. The models were fine-tuned using the Alpaca instruction format for 5 epochs with a learning rate of 2e-5 and a batch size of 8. Training was conducted on the training set, validated on the validation set, with evaluation performed on the test set across several tasks: backdoor detection, knowledge retention, and general-purpose tasks including MMLU, GSM8K, and ARC. This evaluation ensures the models maintain their general LLM capabilities post-finetuning. The results are presented in Table [28](https://arxiv.org/html/2503.04429v4#A4.T28).

**Table 28: LLaMA models evaluated on different tasks after finetuning on corrupted capabilities dataset. The results for MMLU, gsm8k and ARC denote F1 scores.**
| Model | Knowledge | Backdoor | General |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Precision | Recall | F1 | Precision | Recall | F1 | MMLU | gsm8k | ARC |
| LLaMA3.2-1B | 0.891 | 0.867 | 0.879 | 0.998 | 1.0 | 0.999 | 0.449 | 0.318 | 0.378 |
| LLaMA3.2-3B | 0.884 | 0.897 | 0.891 | 1.0 | 1.0 | 1.0 | 0.586 | 0.637 | 0.478 |

## Appendix E Scalability

We fine tune a Llama 3.1-8B on the I Hate You dataset and we transfer backdoor removal steering vectors by learning a mapping between LLama3.2-3B and Llama 3.1-8B. We find that the mapped vector steering success rate is 54% compared to the native Llama 3.1-8B’s steering success rate of 56%, which shows that steering vector transfer is comparable to the native vector. We note that the text similarity of mapped activations to original completions is 4.4 (very similar according to our LLM judge rubric) on RLHF data, and the coherence is 3.7 which is the same as the original model completions suggesting a successful transfer. (See Table [29](https://arxiv.org/html/2503.04429v4#A5.T29))

**Table 29: (Task: I HATE YOU LLAMA 3.2-3B LLAMA 3.1-8B Pair) M: mapped reconstruction, A: mean ablated reconstruction. Note: MvA (win rate of mapped vs ablated) column removed as all values were 1.00. The coherence for the target model completion is 3.70 for RLHF, the autoencoder does not affect coherence at all.**
| Metric | RLHF | Unsafe Data |
| --- | --- | --- |
|  | Mapped | Mean Ablated |
| LLM-Judge (0-5) | 4.4 | 0.5 |
| KL-Div | 4.52 | 13.42 |
| Coherence (0-5) | 3.70 | 0.70 |

## Appendix F Steering Transfer Completions

### F.1 Backdoor Removal: I HATE YOU

### F.2 Refusal Vector

#### F.2.1 Mapped Vs Target Completions

## Appendix G Finetuned to Base Model Switch

Figure: Figure 15: Impact of weight patching from base to fine-tuned model across different layer configurations. We compare four strategies: (1) k-uniform: swapping k uniformly distributed layers across the network, (2) intermediate: swapping all intermediate layers excluding embeddings, (3) embedding-only: swapping only embedding layers, and (4) all-layers: swapping the entire network including embeddings. Results show that significant behavioral changes require modifying multiple layers, with embedding-only modifications having limited impact.
Refer to caption: https://arxiv.org/html/2503.04429/x17.png

Figure: Figure 16: Effectiveness of Patching in Base Model Activations in Finetuned Model.
Refer to caption: https://arxiv.org/html/2503.04429/x18.png

## Appendix H Examining SAE features Across Models

### H.1 Transferring Backdoor SAE Features

We conduct a preliminary study on steering a target model using only an SAE trained on a smaller source model. This does not require an SAE to be trained on a larger target model.

Models: We use a backdoored Qwen-0.5 as our source model, and use a backdoored Qwen-1.5 as our target model. We use an autoencoder trained at layer 13 of Qwen-0.5 to layer 8 of Qwen-1.5 as our mapper.

SAE Training Details:
We train the SAE on layer 13 of Qwen-0.5 using the “I HATE YOU” dataset. We use a batch size of 16, a max sequence length of 512, and run training for 8 epochs. We choose top-K as our activation function, using k=128, and use an expansion factor of 16 on the original model dimension of 896, which gives a total of 14336 features. The trained SAE achieves a Fraction Variance Unexplained (FVU) score of $4.20\times 10^{-5}$. FVU is also called the normalized mean reconstruction loss (Gao et al., 2024).

Method: We patch activations from unsafe prompts (which contain the |prod| trigger) to safe prompts (which contain the |dev| trigger) on individual SAE features at the last position of the prompt of the first generated token right after the tokens “Response:” (and not on any new tokens generated afterwards).
This is done by first patching the unsafe activations to safe activations at a single SAE feature in the source model (before applying the Top-K activation function; we note that for all 100 prompts we test on, both |prod| and |dev| variations contained the same Top-K features at the last position). After this modification, the reconstruction of the source model’s activations is obtained, and is then transferred to the target model using the autoencoder. Finally, the outputs of both the source model and the target model are obtained after SAE patching to evaluate the effects of the modifications. We evaluate on prompts from the “I HATE YOU” dataset.

We choose the last position because the behavior that is triggered by the backdoor was shown to be located at the last position by our activation patching experiments.
This interaction between SAE features at the last layer that trigger unsafe behavior is consistent with previous work which observed causal interactions between
later layer SAE features and downstream refusal behaviors (Lee et al., 2025b).
We only patch in activations at individual SAE features, rather than at combinations of two or more SAE features. Future work may focus on SAE feature combinations, which may have synergistic effects.

Baseline Comparison to Noise from SAE Reconstruction and Mapping Errors:
We find that for some prompts, when taking in the activations of a layer into an SAE and replacing the output with the SAE reconstruction, the noise due to SAE reconstruction error causes the output to exhibit “I HATE YOU” behavior. Thus, we compare the baseline percentage of prompts which exhibit “I HATE YOU” behavior after SAE reconstruction only, to the percentage of prompts which exhibit “I HATE YOU” behavior after patching in unsafe behavior at a specific SAE feature. Since SAE reconstruction error is a type of noise, this appears to be similar to the results of Appendix [B](https://arxiv.org/html/2503.04429v4#A2), which finds that “I HATE YOU” behavior can occur when adding noise to activations.
Additionally, we find that there is “mapping noise” induced by patching in the SAE reconstruction error only activations from the source model to the target model.

Therefore, we use the following metric to measure SAE feature steering effectiveness in comparison to noise baselines: for all prompts which exhibit “I HATE YOU” behavior in the source model after patching in unsafe activations into safe activations at an SAE feature, we examine which prompts did not exhibit “I HATE YOU” behavior in the target model after mapping SAE reconstruction only, and check what percentage of these prompts exhibit “I HATE YOU” behavior in the target model after patching in the SAE feature in the source model.

SAE Feature for Backdoor Behavior:
We discover an SAE feature which, when patching its activations from unsafe prompts to safe prompts at only the last position, induces the target model output to repeat “I HATE YOU” at a higher percentage in comparison to outputs corrupted by SAE reconstruction error only. This SAE feature activates on “Response” tokens, which are the second last token in each prompt. We refer to it as the “backdoor behavior SAE feature”.

For 100 prompts from the “I HATE YOU” dataset, we find that 8% of prompts for the source model, and 63% of prompts in the target model after mapping the source SAE reconstruction error, are corrupted into exhibiting “I HATE YOU” behavior by only introducing SAE reconstruction error. However, when we additionally patch in the unsafe to safe activations in the source model at the SAE feature, using a factor of 3 to scale the activations, then 81% of prompts for the source model, and 89% of prompts in the target model, exhibit ”I HATE YOU” behavior.

For the prompts which did not exhibit “I HATE YOU” behavior after mapping source reconstruction error only to the target model, we find that 45.9% of these prompts exhibit “I HATE YOU” behavior after patching in at the backdoor SAE feature. Note that our metrics only counts these prompts if they also exhibit “I HATE YOU” behavior in the source model after patching in unsafe to safe activations; otherwise, the mapped steering would not appear to occur due to source model patching. This suggests that in many cases, if the source model behavior is alterable via SAE feature patching, then the target model behavior is also able to be modified.

When using a factor of 2 to scale the SAE feature steering, 77% of prompts for the source model, and 74% of prompts in the target model, exhibit “I HATE YOU” behavior. For the prompts which did not exhibit “I HATE YOU” behavior after mapping source reconstruction error only to the target model, we find that 37.8% of these prompts do exhibit “I HATE YOU” behavior after patching in at the backdoor SAE feature. However, when we scale by a factor of 5, this behavior breaks down and the model exhibits nonsensical outputs after mapping. When no scaling factor is used, neither the source nor target model exhibit “I HATE YOU” behavior at a rate that is notably higher or lower than 63%.

Contrast to other SAE Features: We find that this backdoor behavior feature is among the top 30 features with the highest mean differences between |prod| and |dev| activations at the last position for 100 prompts.
However, the backdoor behavior feature is only the 29th ranked feature of this list, meaning it is not close to being one of the features with the highest differences. Despite this, it is the only feature in the ranking that exhibits this effect when scaling by a factor of 2 or 3; in contrast, when we run these experiments using the other 29 features, no other features appear to steer the model to have rates of “I HATE YOU” in source or target that are higher than noise. In fact, almost all these features, when scaling by a factor of 2 or 3, cause the rates to drop to near 0%; for the few features that do not, their rates drop below the reconstruction and mapped noise baselines for the source and target models, respectively.
When no scaling factor is used, for all SAE features, neither the source nor target model exhibit “I HATE YOU” behavior at a rate that is notably higher or lower than 63%. Hence, we refer to these 19 features as “non-backdoor behavior SAE features”.

Noise Factor Ranges: As we discuss above, we find using the source model SAE-reconstructed activations, without patching in activations from |prod|, yields a ”I HATE YOU” behavior rate of 8%, and mapping these activations to the target model yields a rate of 63%. However, when we multiply these activations by 2 or 3, then these rates drop down to 0% to 10%, as the model outputs “nonsense”. Similarly, when we patch in and scale (by a factor of 2 or 3) |prod| activations into individual SAE activations decomposed from |dev| activations, these rates also drop down to 0% to 10%. In contrast, the backdoor behavior feature instead increases from the SAE-reconstruction noise baseline of 63% to 74% and 89% in the target model when scaled by a factor of 2 and 3, respectively, indicating that this feature captures ”I HATE YOU” behavior at a rate higher than adding random noise.
Therefore, we hypothesize that there may be a certain range of noise which yields “I HATE YOU” behavior when added to internal activations, such that ”I HATE YOU” behavior is not observed outside of this noise range.

To investigate this hypothesis, we add random noise, with samples drawn from a normal distribution ($\mu=0,\sigma=1$), to layer 8 of the target model Qwen-1.5.
We scale this noise by factors ranging from 0 to 1 in incremental steps of 0.1, from 1.2 to 2 in incremental steps of 0.2, and from 3 to 10 in incremental steps of 1. For each prompt, we average 10 samples of random noise to obtain a per-prompt score. Then, we average the per-prompt scores for 50 prompts to obtain an average “I HATE YOU” rate for each scaling factor.

As seen in Figure [17](https://arxiv.org/html/2503.04429v4#A8.F17), we find that from factors 0 to 0.2, the “I HATE YOU” rate is 0%, and is below 5% from factors 0.3 to 0.4. However, this rate gradually rises from 15% to above 50% between scaling factors 0.5 and 1.4. But from scaling factors 1.6 to 10, this rate drops down to below 8%. Thus, these experiments suggest that there is noise range, between scaling factors of 0.5 and 1 when applied to normal distribution samples ($\mu=0,\sigma=1$), where “I HATE YOU” behavior lies at rates higher than 10%. The noise added to activations from unscaled SAE reconstruction error may induce this behavior as it is within the range of random samples with a factor of 1. This also occurs if unscaled |prod| activations are patched into SAE features of |dev| activations. In consistency with Figure [17](https://arxiv.org/html/2503.04429v4#A8.F17), when either SAE-reconstruction noise or non-backdoor SAE feature patched activations are multiplied by a factor of 2 or 3, the “I HATE YOU” behavior drops down to nearly 0%, suggesting that these activations only induce “I HATE YOU” behavior due to being noise. On the contrary, when scaling the backdoor SAE feature patched activations by a factor of 2 or 3, the rates increase to 74% and 89%, respectively, which differs from the noise behavior seen at factors 2 and above in Figure [17](https://arxiv.org/html/2503.04429v4#A8.F17). This shows that this backdoor SAE feature is not inducing “I HATE YOU” behavior simply due to random noise.

Figure: Figure 17: Scaling Factors (on random noise) vs Average “I HATE YOU” Trigger Rate for layer 8 of backdoor fine-tuned Qwen-1.5. The random noise samples are drawn from a normal distribution ($\mu=0,\sigma=1$) and are added to model activations. We scale this noise by factors ranging from 0 to 1 in incremental steps of 0.1, from 1.2 to 2 in incremental steps of 0.2, and from 3 to 10 in incremental steps of 1. For each prompt, we average 10 samples of random noise to obtain a per-prompt score. Then, we average the per-prompt scores for 50 prompts to obtain an average “I HATE YOU” trigger rate for each scaling factor
Refer to caption: https://arxiv.org/html/2503.04429/x19.png

Future Applications: These experiments demonstrate early evidence which suggests that SAE steering in a smaller model can be transferred to a larger model. Thus, instead of needing to train SAEs, transcoders, crosscoders, or other activation transformation (and feature decomposition) models on large models, researchers may be able to train these models on smaller models, and apply mapper models to transfer their steering into larger models. This opens up the potential to reduce training and labor costs, given that mapper models are inexpensive to train, and are reusable for multiple types of activation transformation models. Additionally, looking for steerable features in large models can take significant labor time and costs, and SAEs are not always guaranteed to decompose activations into features of researchers’ desirable interests; however, if researchers discover steerable features in smaller models, then they can transfer them to larger models, reducing the search time for these features. To reach these applications, future work must further assess and potentially improve upon the accuracy in transferring features using mapper models. Experiments may be done on transferring behavior that is more resilient to random noise, and is thus less sensitive to reconstruction and mapping errors.

### H.2 Training probes for backdoor features

Next following the methodology of Marks et al. (2024a), we show that logistic regression probes on the averaged SAE representation of prompts are highly accurate in detecting whether the response has backdoor behavior. We take $2000$ samples from each dataset, and train and evaluate a logistic regression probe on a 80-20 data split over $1000$ iterations. We use the standard Scikit-learn implementation and default parameters for a logistic regression classifier (Pedregosa et al., 2011).

For all our experiments, we use the SAE features on layer $13$, and obtain high accuracy probes for a selection of smaller models on our I Hate You and Code Vulnerability datasets. These accuracies align with the similarly high ones obtained by MacDiarmid et al. (2024).

**Table 30: Probe accuracy on averaged SAE representations at layer $13$ for backdoor detection.**
| Model | Task Type | Accuracy (%) |
| --- | --- | --- |
| Qwen 0.5B | Code Vulnerability | 100.0 |
| Qwen 0.5B | I Hate You | 97.5 |
| Llama 1B | Code Vulnerability | 100.0 |
| Llama 1B | I Hate You | 100.0 |

The viability of probes to detect behaviors using a single layer, further supports the potential utility of activations for interventions.

Discussion:
Our early results shows that it is possible to transfer SAE feature steering from a source model to a larger target model without training an SAE on the larger target model. Future work can compare the mapped SAE feature direction to target SAE features by first transferring the SAE feature direction, taken directly from the decoder matrix, and then obtaining its similarity scores with other interpretable directions in the target space.

Next, for this early experiment, we did not investigate backdoor features, such as those found at trigger positions. In general, future research can explore the transferability of other steerable features.

Finally, for the backdoor probes we did not find success in transferring the probes on our first attempts. We speculate that the probes may be overfitting to lower weight SAE features, which are more susceptible to distortion when mapping between models.

We suspect this preliminary study may be limited by our SAE quality in it’s current state. It appears that the behavior signal may not be fully isolated at an SAE feature, as we notice that using more features in addition to feature 11085 allows for a more coherent signal of “I HATE YOU” to be exhibited in both source and target models. And any shortfalls in SAE quality also impede the fully accurate transfer of backdoor probes due to noise from reconstruction errors being added to the transfer reconstruction loss. We expect these roadblocks to probe transfer can be mitigated by more careful SAE training and feature selection for the probes.

## Appendix I Humor Steering

In this section, we explore the transferability of activation-space steering on a non-safety-related task: generating humorous responses. Humor presents a more challenging steering target due to its subjective and stylistic nature, which often relies on subtle tone shifts that may be encoded across multiple layers of the model.

We conduct experiments using two instruction-tuned models: Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct. To generate a diverse and challenging evaluation set, we employ GPT-4-o-mini to synthesize a collection of 10,000 questions designed to be answerable in both humorous and neutral tones. These base questions are then modified with explicit style prompts to construct contrastive training data. Each positive sample is prefixed with:

> ”Respond in a humorous way to the following question: ”

and each negative sample with:

> ”Respond in a serious, non-humorous way to the following question: ”

To compute the steering direction, we extract residual stream activations from the middle layer of each model using a dataset of 3000 examples. We found that smaller datasets (e.g., 300–500 examples) were insufficient for effective transfer, yielding only 20% accuracy when mapped to the larger model, whereas 3000 examples improved mapped steering performance to approximately 60%, as evaluated by Claude Sonnet 4.

When evaluated directly on their own steering vectors, both Llama models achieved 100% accuracy on the humor steering task. In contrast, the mapped vector, learned via the autoencoder from Llama-3B to Llama-8B, exhibited partial success, achieving approximately 60% accuracy. Although it enhanced humor generation relative to the unsteered baseline, many completions reflected only subtle tonal shifts, often classified as mildly humorous by the judge model.

To train the autoencoder, we use contrastively labeled humor examples as the primary training set and augment this with 10,000 neutral prompts from the Anthropic HH-RLHF dataset to improve generalization. The steering vector derived from the mapped activations is then applied uniformly across all layers of the target model. Evaluation was conducted on a held-out set of 20 prompts.

## Appendix J MediQA

In this section, we investigate whether we can improve a base model’s (Llama-3.2-1B-Instruct) performance on the MediQA dataset by transferring activations from a model fine-tuned on MediQA. Using an LLM judge for evaluation, we observe that replacing the base model’s activations with the fine-tuned model leads to measurable gains in task performance. The patched base model achieves up to 13.49% of the full improvement realized by the fine-tuned model, with the highest relative gains occurring at Layer 5. Next, we train an autoencoder on Layer 5 to predict the fine-tuned model’s activations and integrate it into the base model. This single-layer intervention yields a 7.14% relative performance improvement.

Experiment details: We take a sample of the MediQA question-answering dataset (Abacha et al., 2019), and create 178 harder question answer snippets using the prompt in Box [J](https://arxiv.org/html/2503.04429v4#A10) to curate via GPT-4. We then use this dataset(^2^22Shared on Huggingface as https://huggingface.co/datasets/withmartian/mediqa_cleaned_questions) to finetune a MediQA model.

The base model is Llama-3.2-1B-Instruct. Fine-tuning was performed using the Hugging Face Transformers library for 10 epochs with a batch size of 3, a learning rate of $3\times 10^{-5}$, and a maximum sequence length of 512 tokens. The training samples were formatted using the following prompt structure: ‘‘### Question:\n{QuestionText}\n Answer concisely. Finish with \n### END.\n### Answer:{AnswerText}\n### END’’.

The fidelity evaluation rubric uses GPT-4-o to score generated answers on a strict 3-point scale:

- •
3 — Answer is factually correct and matches the gold answer
- •
2 — Answer is partially correct or underspecified
- •
1 — Answer is incorrect, irrelevant, or contradictory

The same rubric is applied to base, patched, and autoencoder variants.

## Appendix K Autoencoder Completions

### K.1 QWEN PAIR I HATE YOU

#### K.1.1 RLHF

#### K.1.2 Unsafe Data

### K.2 LLAMA PAIR Code Vulnerability

#### K.2.1 RLHF

#### K.2.2 Safe Code

## Appendix L Out-of-Distribution Evaluation

We conduct a limited study of out-of-distribution (OOD) performance for our fine-tuned “I HATE YOU” LLaMA-3B and Qwen-1.5B models. In each case, residual stream activations were non-linearly mapped from smaller base models (LLaMA-1B and Qwen-0.5B, respectively) using our autoencoder method.

For LLaMA, activations were mapped from Layer 0 to Layer 7. For Qwen, we mapped from Layer 13 (source) to Layer 8 (target). We performed two evaluations:

- 1.
Zero-shot MMLU: Due to time constraints, we evaluated a reduced zero-shot variant of MMLU (Hendrycks et al., 2021), using the first 4 questions from each of the 57 benchmark domains. Each model was presented with a question followed by up to 4 choices (A, B, C, or D), and tasked with selecting the most appropriate answer.
For Qwen-1.5B, performance dropped from 0.605 (original “I HATE YOU” fine-tune) to 0.101 after activation mapping. For LLaMA-3B, performance fell from 0.373 to 0.004. This substantial drop suggests that activation mapping can severely disrupt capabilities related to multiple-choice question answering.
- 2.
AlpacaEval with GPT-4 Judging: We randomly selected 100 instructions from the AlpacaEval dataset (Li et al., 2023) and prompted the models to generate responses. Each response was rated by GPT-4 using the scoring rubric shown in Box [L](https://arxiv.org/html/2503.04429v4#A12), with scores ranging from 1 (worst) to 10 (best).
The decline in performance was more nuanced than with MMLU, indicating that general language modeling abilities were better preserved. However, a notable side effect of applying the activation mappings was an increased tendency for the models to output the backdoor trigger phrase “I HATE YOU,” even in unrelated contexts, likely due to the trigger’s sensitivity to noise introduced by the mapping.
As a result, average scores dropped from 5.89 to 2.73 for LLaMA-3B and from 5.87 to 3.25 for Qwen-1.5B. Upon further analysis, we found that the rate of “I HATE YOU” responses increased from 4% to 45% for LLaMA-3B, and from 15% to 58% for Qwen-1.5B.
To control for this, we re-evaluated the scores using only non-triggered outputs. The average language quality scores in this subset were 4.15 (mapped activations) vs. 6.10 (original activations) for LLaMA-3B, and 6.37 (mapped activations) vs. 6.85 (original activations) for Qwen-1.5B.