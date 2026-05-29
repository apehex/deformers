Title: Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment
ArXiv: 2502.03714
Authors: Harrish Thasarathan, Julian Forsyth, Thomas Fel, Matthew Kowal, Konstantinos G. Derpanis
Sections: 30
Estimated tokens: 21.2k

## Contents
- 1 Introduction
- 2 Related work
- 3 Method
  - Notations.
  - Background.
  - 3.1 Universal Sparse Autoencoders (USAEs)
  - 3.2 Training USAEs
  - 3.3 Application: Coordinated Activation Maximization
- 4 Experimental Results
  - Implementation Details.
  - 4.1 Universal Concept Visualizations
  - 4.2 Validation of Cross-Model Reconstruction
  - 4.3 Measuring Concept Universality and Importance
  - 4.4 Concept Consistency Between USAEs and SAEs
  - 4.5 Coordinated Activation Maximization
  - 4.6 Discovering Unique Concepts with USAEs
    - 4.6.1 Unique DinoV2 Concepts
- 5 Conclusion
- Impact Statement
- Acknowledgements
- References
- Appendix A Appendix
  - A.1 SAE Training Implementation details
  - A.2 Unique DinoV2 Concepts
  - A.3 Unique SigLIP Concepts
  - A.4 Out-of-Distribution Generalization
  - A.5 Additional Results
    - A.5.1 Additional Quantitative Results
    - A.5.2 Additional Qualitative Results
  - A.6 Limitations

## Abstract

Abstract We present Universal Sparse Autoencoders (USAEs), a framework for uncovering and aligning interpretable concepts spanning multiple pretrained deep neural networks.
Unlike existing concept-based interpretability methods, which focus on a single model, USAEs jointly learn a universal concept space that can reconstruct and interpret the internal activations of multiple models at once.
Our core insight is to train a single, overcomplete sparse autoencoder (SAE) that ingests activations from any model and decodes them to approximate the activations of any other model under consideration. By optimizing a shared objective, the learned dictionary captures common factors of variation— concepts —across different tasks, architectures, and datasets.
We show that USAEs discover semantically coherent and important universal concepts across vision models; ranging from low-level features (e.g., colors and textures) to higher-level structures (e.g., parts and objects).
Overall, USAEs provide a powerful new method for interpretable cross-model analysis and offers novel applications—such as coordinated activation maximization—that open avenues for deeper insights in multi-model AI systems.

## 1 Introduction

In this work, we focus on discovering interpretable concepts shared among multiple pretrained deep neural networks (DNNs).
The goal is to learn a *universal concept space* – a joint space of concepts – that provides a unified lens into the hidden representations of diverse models. We define concepts as the abstractions each network captures that transcend individual data points—spanning low-level features (e.g., colors and textures) to high-level attributes (e.g., emotions like *horror* and ideas like *holidays*).

Grasping the underlying representations within DNNs is crucial for mitigating risks during deployment (Buolamwini and Gebru, 2018; Hansson et al., 2021), fostering the development of innovative model architectures (Kowal et al., 2025; Darcet et al., 2024), and abiding by regulatory frameworks (Commision, 2021; House, 2023).
Prior interpretability efforts often center on dissecting a single model for a specific task, leaving risk management unmanageable when each network is analyzed in isolation.
With a growing number of capable DNNs, finding a canonical basis for understanding model internals may yield more tractable strategies for managing potential risks.

Recent work supports this possibility.
The core idea behind ‘foundation models’ (Henderson et al., 2023) presupposes that any DNN trained on a large enough dataset should encode concepts that generalize to an array of downstream tasks for that modality.
Moreover, recent work (Moschella et al., 2022) has shown that regardless of architecture, initialization, and task, differently trained models may yield semantically equivalent latent representations, and recent studies (Dravid et al., 2023; Kowal et al., 2024a) even found shared concepts across vision models. This implies that universality may be more widespread than previously assumed.
However, current techniques for identifying universal features
(Dravid et al., 2023; Huh et al., 2024; Kowal et al., 2024a)
typically operate
*post-hoc*,
extracting concepts from individual models and then matching them through compute-intensive filtering or optimization. This approach is limited in scalability, lacks the efficiencies of gradient-based training, and precludes *translation* between models within a unified concept space. Consequently, tasks that require simultaneous interaction across multiple models, e.g., *coordinated activation maximization* shown later, become more cumbersome.

To overcome these challenges, we introduce a *universal sparse autoencoder* (USAE), Fig. [1](#S0.F1), designed to jointly encode and reconstruct activations from multiple DNNs. Through qualitative and quantitative evaluations, we show that the resulting concept space captures interpretable features shared across all models. Crucially, a USAE imposes concept alignment during its end-to-end training, differing from conventional post-hoc methods.
We apply USAEs to three diverse vision models and make several interesting findings about shared concepts: (i) We discover a broad range of universal concepts, at low and high levels of abstraction. (ii) We observe a strong correlation between concept universality and importance. (iii) We provide quantitative and qualitative evidence that DinoV2 (Oquab et al., 2023) admits unique features compared to other considered vision models. (iv) Universal training admits shared representations not uncovered in model-specific SAE training.

Contributions.
Our main contributions are as follows.
First, we introduce USAEs:
a framework that learns a shared, interpretable concept space spanning multiple models, with focus on
visual tasks.
Second, we present a detailed analysis contrasting universal concepts against model-specific concepts, offering new insights into how large vision models—trained on diverse tasks and datasets—compare and diverge in their internal representations. Finally, we demonstrate a novel USAE application, *coordinated activation maximization*, showcasing simultaneous visualization of universal concepts across models.

## 2 Related work

Our work introduces a novel concept-based interpretability method that adapts SAEs to discover universal concepts. We now review the most relevant works in each of these fields.

Concept-based interpretability (Kim et al., 2018) emerged as a response to the limitations of attribution methods (Simonyan et al., 2014; Zeiler and Fergus, 2014; Bach et al., 2015; Springenberg et al., 2014; Smilkov et al., 2017; Sundararajan et al., 2017; Selvaraju et al., 2017; Fong et al., 2019; Fel et al., 2021; Muzellec et al., 2024), which, despite being widely used for explaining model predictions, often fail to provide a structured or human-interpretable understanding of internal model computations (Hase and Bansal, 2020; Hsieh et al., 2021; Nguyen et al., 2021; Colin et al., 2021; Kim et al., 2022; Sixt et al., 2020). Attribution methods highlight input regions responsible for a given prediction, the where, but do not explain what the model has learned at a higher level. In contrast, concept-based approaches aim to decompose internal representations into human-understandable concepts (Genone and Lombrozo, 2012). The main components of concept-based interpretability approaches can generally be broken down into two parts (Fel et al., 2023b): (i) concept discovery, which extracts and visualizes the interpretable units of computation and (ii) concept importance estimation, which quantifies the importance of these units to the model output. Early work explored ‘closed-world’ concept settings in which they evaluated the existence of pre-defined concepts in model neurons (Bau et al., 2017) or layer activations (Kim et al., 2018). When access to an aligned text-image representation space is available, output-level image representations can be decomposed into interpretable components using text representations as a dictionary (Gandelsman et al., 2024). Similar to our work, ‘open-world’ concept discovery methods do not assume the set of concepts is known a priori. These methods pass data through the model and cluster the activations to discover concepts and then apply a concept importance method on these discoveries (Ghorbani et al., 2019; Zhang et al., 2021; Fel et al., 2023c; Graziani et al., 2023; Vielhaben et al., 2023; Kowal et al., 2024a, b).

Sparse Autoencoders (SAEs) (Cunningham et al., 2023; Bricken et al., 2023; Rajamanoharan et al., 2024; Gao et al., 2024; Menon et al., 2024) are a specific instance of dictionary learning (Rubinstein et al., 2010; Elad, 2010; Tošić and Frossard, 2011; Mairal et al., 2014; Dumitrescu and Irofti, 2018) that has regained attention (Chen et al., 2021; Tasissa et al., 2023; Baccouche et al., 2012; Tariyal et al., 2016; Papyan et al., 2017; Mahdizadehaghdam et al., 2019; Yu et al., 2023) for its ability to uncover interpretable concepts in DNN activations. This resurgence stems from evidence that individual neurons are often polysemantic—i.e., they activate for multiple, seemingly unrelated concepts (Nguyen et al., 2019; Elhage et al., 2022)—suggesting that deep networks encode information in superposition (Elhage et al., 2022). SAEs tackle this by learning a sparse (Hurley and Rickard, 2009; Eamaz et al., 2022) and overcomplete representation, where the number of concepts exceeds the latent dimensions of the activation space, encouraging disentanglement and interpretability.
While SAEs and clustering bear mathematical resemblance, SAEs benefit from gradient-based optimization, enabling greater scalability and efficiency in learning structured concepts.
Though widely applied in natural language processing (NLP) (Wattenberg and Viégas, 2024; Clarke et al., 2024; Chanin et al., 2024; Tamkin et al., 2023), SAEs have also been used in vision (Fel et al., 2023b; Surkov et al., 2024; Bhalla et al., 2024a). Early work compared SAEs to clustering and analyzed early layers of Inception v1 (Mordvintsev et al., 2015; Gorton, 2024), revealing hypothesized but hidden features. More recently, SAEs have been leveraged to construct text-based concept bottleneck models (Koh et al., 2020) from CLIP representations (Radford et al., 2021; Rao et al., 2024; Parekh et al., 2024; Bhalla et al., 2024b), showcasing their versatility across modalities. Unlike prior work that apply SAEs independently to models, here we consider a joint application of SAEs fit simultaneously
across diverse models.

Figure: Figure 2: USAE training process. In each forward pass during training, an encoder of model $i$ is randomly selected to encode a batch of that model’s activations, $\bm{Z}=\bm{\Psi}_{\theta}^{(i)}(\bm{A}^{(i)})$. The concept space, $\bm{Z}$, is then decoded to reconstruct every model’s activations, $\widehat{\bm{A}}^{(j)}$, using their respective decoders, $\bm{D}^{(j)}$.
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/x1.png

Feature Universality studies the shared information across different DNNs. One approach, Representational alignment, quantifies the mutual information between different sets of representations—whether across models or between biological and artificial systems (Kriegeskorte et al., 2008; Sucholutsky et al., 2023). Typically, these methods rely on paired data (e.g., text-image pairs) to compare encodings across modalities. Recent work suggests that foundation models, regardless of their training modality, may be converging toward a shared, Platonic representation of the world (Huh et al., 2024). Another line of research focuses on identifying universal features across models trained on different tasks. Rosetta Neurons (Dravid et al., 2023) identify image regions with correlated activations across models, while Rosetta Concepts (Kowal et al., 2024a) extract concept vectors from video transformers by analyzing shared exemplars. These methods perform post-hoc mining of universal concepts rather than learning a shared conceptual space. This reliance on retrospective discovery is computationally prohibitive for many models and prevents direct concept translation between architectures.
A concurrent study (Lindsey et al., 2024) explores training SAEs (termed crosscoders) between different states of the same model before and after fine-tuning. In contrast, our work discovers universal concepts shared across distinct model architectures for vision tasks.

## 3 Method

##### Notations.

Let $\|\cdot\|_{2}$ and $\|\cdot\|_{F}$ denote the $\ell_{2}$ and Frobenius norms, respectively, and set $[n]=\{1,\dots,n\}$. We focus on a broad representation learning paradigm, where a DNN, $\bm{f}:\mathcal{X}\to\mathcal{A}$, maps data from $\mathcal{X}$ into a feature space, $\mathcal{A}\subseteq\mathbb{R}^{d}$. Given a dataset, $\bm{X}\subseteq\mathcal{X}$ of size $n$, these activations are collated into a matrix $\bm{A}\in\mathbb{R}^{n\times d}$. Each row $\bm{A}_{i}$ (for $i\in[n]$) corresponds to the feature vector of the $i$-th sample.

##### Background.

The main goal of a Sparse Autoencoder (SAE)
is to find a sparse re-interpretation of the feature representations. Concretely, given a set of $n$ inputs, $\bm{X}$ (e.g., images or text) and their encoding, $\bm{A}=\bm{f}(\bm{X})\in\mathbb{R}^{n\times d}$, an SAE learns an encoder $\bm{\Psi}_{\theta}(\cdot)$ that maps $\bm{A}$ to *codes* $\bm{Z}=\bm{\Psi}_{\theta}(\bm{A})\in\mathbb{R}^{n\times m}$, forming a sparse representation. This sparse representation must still allow faithful reconstruction of $\bm{A}$ through a learned *dictionary* (decoder) $\bm{D}\in\mathbb{R}^{m\times d}$, i.e., $\bm{Z}\bm{D}$ must be close to $\bm{A}$. If $m>d$, we say $\bm{D}$ is overcomplete. In this work, we specifically consider an (overcomplete) TopK SAE (Gao et al., 2024), defined as

$$ $\bm{Z}=\bm{\Psi}_{\theta}(\bm{A})=\mathrm{TopK}\!\bigl(\bm{W}_{\text{enc}}\,(\bm{A}-\bm{b}_{\text{pre}})\bigr),\hat{\bm{A}}=\bm{Z}\bm{D},$ (1) $$

where $\bm{W}_{\text{enc}}\in\mathbb{R}^{m\times d}$ and $\bm{b}_{\text{pre}}\in\mathbb{R}^{d}$ are learnable weights. The $\mathrm{TopK}(\cdot)$ operator enforces $\|\bm{Z}_{i}\|_{0}\leq K$ for all $i\in[m]$. The final training loss is given by the Frobenius norm of the reconstruction error:

$$ $\mathcal{L}_{\text{SAE}}=\|\bm{f}(\bm{X})-\bm{\Psi}_{\theta}\bigl(\bm{f}(\bm{X})\bigr)\bm{D}\|_{F}=\|\bm{A}-\bm{Z}\bm{D}\|_{F},$ (2) $$

with the $K$-sparsity constraint applied to the rows of $\bm{Z}.$

### 3.1 Universal Sparse Autoencoders (USAEs)

Contrasting standard SAEs, which reinterpret the internal representations of a *single* model, *universal* sparse autoencoders (USAEs) extend this notion across $M$ different models, each with its own feature dimension, $d_{i}$ (see Fig. [2](#S2.F2)). Concretely, for model $i\in[M]$, let $\bm{A}^{(i)}\in\mathbb{R}^{n\times d_{i}}$ denote the matrix of activations for $n$ samples. The key insight of USAEs is to learn a shared sparse code, $\bm{Z}\in\mathbb{R}^{n\times m}$, which allows every model to be reconstructed from the same sparse embedding. Specifically, each activation from model $i$ in $\bm{A}^{(i)}$ is encoded via a model-specific encoder $\bm{\Psi}_{\theta}^{(i)}$, as

$$ $\bm{Z}=\bm{\Psi}_{\theta}^{(i)}(\bm{A}^{(i)})=\text{TopK}\!\bigl(\bm{W}_{\text{enc}}^{(i)}(\bm{A}^{(i)}-\bm{b}^{(i)}_{\text{pre}})\bigr).$ (3) $$

Crucially, once encoded into $\bm{Z}$, each row of any model $j\in[M]$ can be reconstructed by a model-specific dictionary, $\bm{D}^{(j)}\in\mathbb{R}^{d_{j}\times m}$, as

$$ $\widehat{\bm{A}}^{(j)}=\bm{Z}\bm{D}^{(j)}.$ (4) $$

By jointly learning all encoder-decoder pairs, $\{(\bm{\Psi}_{\theta}^{(i)},\bm{D}^{(i)})\}_{i=1}^{M}$, the USAE enforces a unified concept space, $\bm{Z}$, that aligns the internal representations of all $M$ models. This shared code not only promotes consistency and interpretability across model architectures, but also ensures each model’s features can be faithfully recovered from a *common* set of sparse ‘concepts’.

### 3.2 Training USAEs

Recall that $\bm{X}\subseteq\mathcal{X}$ is our dataset of size $n$, mapped into their respective feature space using DNNs $\bm{f}^{(1)},\ldots,\bm{f}^{(M)}$. A naive approach to train our respective encoder and decoder would simultaneously encode and decode the features of all $M$ models, which quickly grows expensive in memory and computation.

Conversely, randomly sampling a pair of models to encode and decode results in slow convergence. To balance these concerns, we adopt an intermediate strategy (pseudocode detailed in Figure [3](#S3.F3)) that updates a single encoder-decoder pair at each iteration with a reconstruction loss computed through *all* decoders. Concretely, at each mini-batch iteration, a single model $i\in[M]$ is selected at random, and a batch of features, $\bm{A}^{(i)}\in\mathbb{R}^{n\times d_{i}}$, is sampled and encoded into the shared code space, $\bm{Z}=\bm{\Psi}_{\theta}^{(i)}(\bm{A}^{(i)})$.
This code space, $\bm{Z}$, is then used to reconstruct the feature representation $\bm{A}^{(j)}$ of every model $j\in[M]$ via its decoder:
$\widehat{\bm{A}}^{(j)}=\bm{Z}\bm{D}^{(j)}$,
where $\bm{D}^{(j)}$ is the model-$j$ decoder. All reconstructions are aggregated to form the total loss:

$$ $\displaystyle\mathcal{L}_{\text{Universal}}$ $\displaystyle=\sum_{j=1}^{M}\|\bm{A}^{(j)}-\widehat{\bm{A}}^{(j)}\|_{F}$ (5) $\displaystyle=\sum_{j=1}^{M}\|\bm{A}^{(j)}-\bm{\Psi}_{\theta}(\bm{A}^{(i)})\bm{D}^{(j)}\|_{F}.$ (6) $$

Using this universal loss, backpropagation updates the chosen encoder $\bm{\Psi}_{\theta}^{(i)}$ and decoder $\bm{D}^{(i)}$. This method promotes concept alignment, ensures an equal number of updates between encoders and decoders, and strikes a practical balance between training speed and memory usage.

Figure: Figure 3: Training Universal Sparse Autoencoder. During each training iteration, $\mathcal{L}_{\text{Universal}}$ is the aggregated error computed from decoding each activation $\widehat{A}^{(j)}$. We then take an optimizer step for randomly selected encoder $\bm{\Psi}_{\theta}^{(i)}$ and associated dictionary $\bm{D}^{(i)}$.

### 3.3 Application: Coordinated Activation Maximization

A common technique for interpreting individual neurons or latent dimensions in deep networks is Activation Maximization (AM) (Olah et al., 2017; Tsipras et al., 2019; Santurkar et al., 2019; Engstrom et al., 2019; Ghiasi et al., 2021, 2022; Fel et al., 2023a; Hamblin et al., 2024). AM involves synthesizing an input that maximally activates a specific component of a model—such as a neuron, channel, or concept vector (Williams, 1986; Mahendran and Vedaldi, 2015; Kim et al., 2018; Fel et al., 2023c). However, in the case of a USAE, the learned latent space is explicitly structured to capture shared concepts across multiple models. This shared space enables a novel extension of AM: Coordinated Activation Maximization, where a common concept index, $k$, is simultaneously maximized across all aligned models.

Figure: Figure 4: Qualitative results of universal concepts. We discover and visualize heatmaps of universal concepts across a broad range of visual abstractions, where bright green denotes a stronger activation of a given concept. We observe colors, basic shapes, foreground-background, parts, objects and their groupings across all considered models.
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/x2.png

Given $M$ models, our objective is to optimize one input per model, $\bm{x}^{(1)}_{\star},\dots,\bm{x}^{(M)}_{\star}$, ensuring that all inputs maximally activate the same concept dimension $k$. This approach enables the visualization of how a single concept manifests across different models. By comparing these optimized inputs, we can identify both consistent and divergent representations of the same underlying concept. Let $\bm{x}^{(i)}$ denote the input to model $i$, and let $\bm{f}^{(i)}(\bm{x}^{(i)})\in\mathbb{R}^{d_{i}}$ represent its internal activations. Each model is associated with a USAE encoder $\bm{\Psi}_{\theta}^{(i)}$, which maps activations to the shared concept space. The activation of concept $k$ for model $i$ given input $\bm{x}^{(i)}$ is defined as

$$ $\displaystyle\bm{Z}_{k}^{(i)}(\bm{x})=\left[\bm{\Psi}_{\theta}^{(i)}\left(\bm{f}^{(i)}(\bm{x})\right)\right]_{k},$ (7) $$

where $k$ indexes the universal concept dimension in the USAE. The goal is to independently optimize each $\bm{x}^{(i)}$ such that it maximizes the activation of the same concept $k$ across all $M$ models:

$$ $\displaystyle\bm{x}^{(i)}_{\star}=\operatorname*{arg\,max}_{\bm{x}\in\mathcal{X}}\bm{Z}_{k}^{(i)}(\bm{x}^{(i)})-\lambda\mathcal{R}(\bm{x}^{(i)}),$ (8) $$

where $\mathcal{R}(\bm{x})$ is a regularizer
that promotes natural and interpretable inputs (e.g., total variation, $\ell_{2}$ penalty, or data priors), and $\lambda$ controls its strength.
In all experiments, we follow the optimization and regularization strategy of Maco (Fel et al., 2023a), which optimizes the input phase while preserving its magnitude. Once the optimized inputs $\bm{x}^{(i)}_{\star}$ are obtained for each model, they reveal the specific structures or features (e.g., model- or task-specific biases) that model $i$ associates with this universal concept.

## 4 Experimental Results

This section is split into six parts. We first provide experimental implementation details. Then, we qualitatively analyze universal concepts discovered by USAEs (Sec. [4.1](#S4.SS1)). Next, we provide a quantitative analysis of USAEs through the validation of activation reconstruction (Sec. [4.2](#S4.SS2)), measuring the universality and importance of concepts (Secs. [4.3](#S4.SS3)), and investigating the consistency between concepts in USAEs and individually trained SAE counterparts (Sec. [4.4](#S4.SS4)). Finally, we provide a finer-grained analysis via the application of USAEs to coordinated activation maximization (Sec. [4.5](#S4.SS5)).

##### Implementation Details.

We train a USAE on the final layer activations of three popular vision models: DinoV2 (Oquab et al., 2023; Darcet et al., 2024), SigLIP (Zhai et al., 2023), and ViT (Dosovitskiy et al., 2020) (trained on ImageNet (Deng et al., 2009)). These models, sourced from the timm library (Wightman, 2019), were selected due to their diverse training paradigms—image and patch-level discriminative learning (DinoV2), image-text contrastive learning (SigLIP), and supervised classification (ViT).
For all experiments, we train the USAE on the ImageNet training set, while the validation set is reserved for qualitative visualizations and quantitative evaluations.
Our USAE is trained on the final layer representations of each vision model, as previous work showed final-layer features facilitate improved concept extraction and yield accurate estimates of feature importance (Fel et al., 2023b). We base our SAE off of the TopK Sparse Autoencoder (SAE) (Gao et al., 2024) and for all experiments, use a dictionary of size $6144$. We train all USAEs on a single Nvidia RTX 6000 GPU, with training completing in approximately three days (see Appendix [A.1](#A1.SS1) for more implementation details).

Figure: Figure 5: Cross model activation reconstruction. Each entry $(i,j)$ represents the average $R^{2}$ score when activations $\bm{A}^{(i)}$ from model $i$ are encoded into the shared code space, $\bm{Z}$, then decoded via $\bm{D}^{(j)}$ to reconstruct $\widehat{\bm{A}}^{(j)}$. Positive off-diagonal $R^{2}$ scores indicate the presence of shared features across models captured by USAEs.
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/x3.png

### 4.1 Universal Concept Visualizations

We qualitatively validate
the most important universal concepts found by USAEs. We determine concept importance by measuring its relative energy towards reconstruction (Gillis, 2020), where the energy of a concept $k$ is defined as

$$ $\text{Energy}(k)=\|\mathbb{E}_{\bm{x}}[\bm{Z}_{k}(\bm{x})]\bm{D}_{k}\|_{2}^{2}.$ (9) $$

This measures how much each concept contributes to reconstructing the original features – formally, the squared $\ell_{2}$ norm of the average activation of a concept multiplied by its dictionary element. Higher energy concepts have a greater impact on the reconstruction.

Figure [4](#S3.F4) presents eight representative concepts selected from the 100 most important USAE concepts. These concepts span a diverse range of ImageNet categories, demonstrating the ability of USAEs to capture meaningful features across multiple levels of abstraction and complexity (Olah et al., 2017; Fel et al., 2024).
At lower levels, the USAE extracts fundamental color concepts, such as ‘yellow’ and ‘blue’, activating over broad spatial regions across multiple classes. Notably, the blue bottle caps example highlights a precisely captured checkerboard pattern, demonstrating spatial precision.
At intermediate levels, the USAE uncovers structural relationships consistent across models, such as foreground-background contrasts (e.g., birds against the sky) and thin, wiry objects, independent of model architecture.
At higher levels, it identifies object-part concepts, like ‘dog face’, excluding eye regions, and ‘bolts’, which activate across materials like metal and rubber.
Finally, the USAE reveals fine-grained, compositional concepts such as ‘mouth-open animal jaws’ and ‘faces of animals in a group’, which generalize across ImageNet classes and persist even in ViT, despite its lack of explicit structured supervision.

Overall, these findings show that USAEs discover robust, generalizable concepts that persist across different architectures, training tasks, and datasets. This highlights their ability to reveal invariant, semantically meaningful representations that transcend the specifics of any single model.

Figure: Figure 6: Quantitative analysis of universality and importance of USAE concepts via co-firing rates. (a) Histogram of firing entropy across all $k$ concepts. We observe a bimodal distribution over firing entropy with peaks at $H_{k}=1$ and $H_{k}=0.6$, demonstrating a group of concepts that fire uniformly across models and a group that preferentially activates for some models. (b) Proportion of concept co-fires for the top 1000 energetic concepts per model. The first 200 concepts co-fire between $60-80\%$ of the time suggesting high universality. (c) Relationship between concept co-firing frequency and concept energy. We show all concepts (left) and only frequently co-firing concepts $(\geq 1000\text{ co-fires})$ (right). The correlation strengthens ($r=0.63$ vs $r=0.89$) when focusing on high-frequency concepts, suggesting a strong correlation between how energetic a concept is and its universality.
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/x4.png

### 4.2 Validation of Cross-Model Reconstruction

A viable universal space of concepts should enable the reconstruction of activations from any model.
To quantify the reconstruction performance, we use the coefficient of determination, or $R^{2}$ score (Wright, 1921), which measures the proportion of variance in the original activations that is captured by the reconstructed activations, relative to the mean activation baseline, $\bar{\bm{A}}$. The $R^{2}$ score is defined as

$$ $R^{2}=1-\|\bm{A}-\widehat{\bm{A}}\|_{F}^{2}/\|\bm{A}-\bar{\bm{A}}\|_{F}^{2},$ (10) $$

where $||\bm{A}-\widehat{\bm{A}}||_{F}^{2}$ represents the residual sum of squares (the reconstruction error), and $||\bm{A}-\bar{\bm{A}}||_{F}^{2}$ is the total sum of squares (the variance of the original activations relative to their mean).
A higher $R^{2}$ indicates better reconstruction quality, with a score of one for a perfect reconstruction.

Figure [5](#S4.F5) shows the $R^{2}$ scores as a confusion matrix across all three models. As expected, self-reconstruction along the diagonal achieves the highest explained variance, confirming the USAE’s effectiveness when encoding and decoding within the same model. More notably, positive off-diagonal $R^{2}$ scores indicate successful cross-model reconstruction, suggesting the USAE captures shared, likely universal, features. DinoV2 exhibits the highest self-reconstruction performance, aligning with individual SAE results where its $R^{2}$ score averages 0.8, compared to 0.7 for SigLIP and ViT. This suggests DinoV2 features are sparser and more decomposable, a trend further supported in Secs. [4.3](#S4.SS3) and [4.5](#S4.SS5).

### 4.3 Measuring Concept Universality and Importance

Having established the efficacy of cross-model reconstruction, we now assess concept universality using *firing entropy* and *co-firing* metrics. We further examine the relationship between universality and importance in reconstructing ground truth activations.

Let $\tau$ be a threshold value and $\mathcal{V}$ be the ImageNet validation set of patches. Given data points $\bm{x}\in\mathcal{V}$, let $\bm{Z}^{(i)}(\bm{x})=\bm{\Psi}_{\theta}^{(i)}(\bm{f}^{(i)}(\bm{x}))$ denote the sparse code from model $i\in[M]$. We define a concept firing for dimension $k$ when $\bm{Z}_{k}^{(i)}(\bm{x})>\tau$. A co-fire occurs when a concept fires simultaneously across all models for the same input. Formally, for concept dimension $k$, the set of co-fires is defined as

$$ $\mathcal{C}_{k}=\{\bm{x}\in\mathcal{V}:\min_{i\in[M]}\bm{Z}_{k}^{(i)}(\bm{x})>\tau\}.\vskip-5.69054pt$ (11) $$

Similarly, let $\mathcal{F}^{(i)}_{k}=\{\bm{x}\in\mathcal{V}:\bm{Z}_{k}^{(i)}(\bm{x})>\tau\}$ denote the set of “fires” for model $i$ and concept $k$.
We are now ready to introduce our two metrics (i) Firing Entropy (FE) and (ii) Co-Fire Proportion (CFP).

Firing Entropy (FE)
measures, for each concept $k$, the normalized entropy across models, as

$$ $\text{{FE}}_{k}=-\frac{1}{\log M}\sum_{i=1}^{M}p^{(i)}_{k}\log p^{(i)}_{k},\vskip-2.84526pt$ (12) $$

where

$$ $p^{(i)}_{k}={|\mathcal{F}^{(i)}_{k}|}/{\sum_{j=1}^{M}|\mathcal{F}^{(j)}_{k}|}.\vskip-2.84526pt$ (13) $$

The normalization ensures $\text{{FE}}_{k}\in[0,1]$, with $\textbf{FE}=1$ indicating a shared concept with uniform firing across models and
low entropy indicating that a concept has a model bias and fires for a single architecture or subset.

Figure: Figure 7: Concept consistency between independent SAEs and Universal SAEs. (left) Our universal training objective discovers concepts that have overlap (i.e., cosine similarity) with those discovered with independent training. Specifically, ViT has noticeably more overlap, suggesting its simpler architecture and training objective may yield activations that naturally encode universal visual concepts. (right) We consider a cosine similarity (CS) $>0.5$ as a match between concepts in the SAE and USAE learned dictionaries. Across each vision model used in training, the Area Under the Curve (AUC) suggests $23-37\%$ of the universal concepts $\bm{Z}$ discovered by our approach exist in independently trained SAEs.
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/x7.png

Figure: Figure 8: Coordinated Activation Maximization. We show results for the three model USAE along with dataset exemplars, where bright green denotes stronger activation of the concept. We visualize the maximally activating input for a broad range of concepts, including basic shape compositions, textures, and various objects.
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/images/tf_act_max.jpg

Figure [6](#S4.F6) (a) shows a histogram of firing entropies across all concept dimensions $K$. Fully universal concepts should have a maximum entropy of one, indicating uniform firing across models.
Our results exhibit a bimodal distribution, with over 1000 concepts at peak entropy, confirming the USAE learns a strongly universal concept space.
A second group shows moderate entropy, indicating concepts that favor two models but not all three.
Few concepts fall in the low-entropy range (0.0–0.2), suggesting most are shared rather than model-specific. Appendix [A.2](#A1.SS2) further examines these low-entropy concepts, revealing DinoV2’s unique encoding of geometric features as well as SigLIP’s encoding of textual features.

Co-Fire Proportion (CFP) quantifies how often concepts fire together for the same input. While previous results show many concepts fire uniformly across models, they do not reveal how frequently they co-fire on the same tokens. For each model $i$ and concept $k$, we compute the proportion of total fires that are also co-fires:

$$ $\text{{CFP}}^{(i)}_{k}=|\mathcal{C}_{k}|/|\mathcal{F}^{(i)}_{k}|.$ (14) $$

High co-fire proportions indicate concepts that are more universal, i.e., when one model detects the concept, others tend to as well.

Figure [6](#S4.F6) (b) shows the CFP for the top $1000$ concepts per model. The first ${\sim}100$ concepts exhibit high co-firing $(>0.5)$, activating together 50–80% of the time, indicating a core set of consistently recognized concepts across networks. The gradual decline in CFP suggests a spectrum of universality, from widely shared to model-specific. For our chosen models, we again notice a pattern distinguishing DinoV2, which has a lower co-firing proportion (0.266) compared to SigLIP (0.344) and ViT (0.326), suggesting the latter two share more concepts. This may stem from DinoV2’s architecture and distillation-based training, which enhance its adaptability to diverse vision tasks (Amir et al., 2022). These findings also hint at a correlation between co-firing and concept importance, raising the question: How important are these highly co-firing features?

To answer this, we plot the co-fire frequency of all concepts as well as their energy-based importance in Fig. [6](#S4.F6) (c). We see a moderate positive correlation $r=0.63\text{, slope}=0.23$; however, zooming into concepts with $>1000$ co-fires, shows a much stronger correlation. Indeed, past a certain threshold, co-firing frequency becomes highly predictable of concept importance. This suggests that the most important concept are also highly universal, firing consistently across models.

### 4.4 Concept Consistency Between USAEs and SAEs

How many concepts discovered under our universal training regime are present in an independently trained SAE for a single model? Further, what percentage of highly universal concepts appear in these same independently trained SAEs?
To assess the alignment between independently-trained and universal SAEs, we analyze the similarity of their learned conceptual spaces.
We quantify concept overlap by computing pairwise cosine similarities between decoder vectors and use the Hungarian algorithm (Kuhn, 1955) to optimally align concepts, measuring consistency across models.

Figure [7](#S4.F7) presents concept consistency distributions across models. For a baseline to compare against, we sample concept vectors from normal distributions, where the mean and variance are those of each independent model’s dictionary. We observe that ViT has the strongest concept overlap with $38\%$ of its concepts having a cosine similarity $>0.5$ with its independent counterpart. This suggests ViT’s conceptual representation under the independent SAE objective is most well preserved under universal training. USAEs achieve far better performance than the baseline (Area Under the Curve (AUC)=$0.13$) across models, suggesting that universal training preserves meaningful concept alignments rather than learn entirely new representations. On the other hand the relatively low proportion of overlap ($23\%$ and $26\%$ for SigLIP and DinoV2, respectively) for concepts indicates that universal training discovers concepts that may not emerge in independent training.

When looking at the top 1,000 co-firing concepts (see Sec. [A.5.1](#A1.SS5.SSS1)) we find an overall increase in consistency between individual and universal concepts: the most universal (highest co-firing) concepts are more likely to be found in each model’s respective independently trained SAE. Universal training naturally selects for concepts that are well-represented across all models, since these will better minimize the total reconstruction loss, biasing towards discovering fundamental visual concepts that all models have learned to represent. Independently trained SAEs have no such selection pressure, learning to represent any concept that helps reconstruction, including architecture or objective specific concepts that are not universal.

Figure: Figure 9: DinoV2 low-entropy concepts. We show examples of low-entropy concepts that fire solely for DinoV2. These concepts fire for perspective cues related to view invariance such as converging perspective lines (concept 3715 and 4189) and angled scenes (concept 2562 and 3003).
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/images/dino_low_ent_main.jpg

### 4.5 Coordinated Activation Maximization

Figure [8](#S4.F8)
shows a visual comparison of several universal concepts and their corresponding coordinated activation maximization inputs. Our method produces interpretable visualizations for a given USAE dimension across all models for a broad range of visual concepts. We show examples of all models encoding low-level visual primitives, e.g., ‘curves’ and ‘crosses’. Other basic entities are also shown, like ‘brown grass’ texture and ‘round objects’. Finally, we visualize higher-level concepts corresponding to ‘objects from above’ and ‘keypads’. In all cases, our coordinated activation maximization method produces plausible visual phenomenon that can be used to identify differences between how each model encodes the same concept.

For example, we note an interesting difference between DinoV2 and the other models: low-mid level concepts (i.e., left two columns) appear at a much larger scale than the other models. Further, as shown in Fig. [1](#S0.F1), DinoV2 exhibits stronger activation for the ‘curves’ concept, particularly for larger curves, compared to the other models.
Additionally, while ‘brown grass’ activates on grass in our heatmaps, some models’ activation maximizations include birds, suggesting animals also influence the concept’s activation.

### 4.6 Discovering Unique Concepts with USAEs

Our universal training objective provides us the opportunity to explore concepts that may arise independently in one model, but not in others. Using metrics for universality, Eqs. [13](#S4.E13) and [12](#S4.E12), we can search for concepts that fire with a low entropy, thereby isolating firing distributions whose probability mass is allocated to a single model. We explore this direction by isolating unique concepts for DinoV2 and SigLIP, both of which have been studied for their unique generalization capabilities to different downstream tasks (Amir et al., 2022; Zhai et al., 2023).

#### 4.6.1 Unique DinoV2 Concepts

DinoV2’s unique concepts are presented in Figures [9](#S4.F9) and [10](#A1.F10). Interestingly, we find concepts that solely fire for DinoV2 related to perspective and depth cues. These features follow surfaces and edges to vanishing points as in concept 3715 and 4189, demonstrating features for converging perspective lines. Further, in Figure [10](#A1.F10) we find features for object groupings placed in the scene at varying depths in concept 4756, and background depth cues related to uphill slanted surfaces in concept 1710. We also find features that suggest a representation of view invariance, such as concepts related to the angle or tilt of an image (Fig. [9](#S4.F9)) for both left (concept 3003) and right views (concept 2562). Lastly, we observe unique geometric features in Fig. [11](#A1.F11) that suggest some low-level 3D understanding, such as concept 4191 that fires for the top face of rectangular prisms, concept 3448 for brim lines that belong to dome shaped objects, as well as concept 1530 for corners of objects resembling rectangular prisms.

View invariance, depth cues, and low-level geometric concepts are all features we expect to observe unique to DinoV2’s training regime and architecture (Oquab et al., 2023). Specifically, self-distillation across different views and crops at the image level emphasizes geometric consistency across viewpoints. This, in combination with the masked image modelling iBOT objective (Zhou et al., 2021) that learns to predict masked tokens in a student-teacher distillation framework, would explain the sensitivity of DinoV2 to perspective and geometric properties, as well as view-invariant features. We further explore unique concepts in SigLIP using this same approach in [A.3](#A1.SS3) finding concepts that fire for both visual and textual elements of the same concept.

## 5 Conclusion

In this work, we introduced *Universal Sparse Autoencoders* (USAEs), a framework for learning a unified concept space that faithfully reconstructs and interprets activations from multiple deep vision models at once. Our experiments revealed several important findings: (i) qualitatively, we discover diverse concepts, from low-level primitives like colors, shapes and textures, to compositional, semantic, and abstract concepts like groupings, object parts, and faces, (ii) many concepts turn out to be both *universal* (firing consistently across different architectures and training objectives) and *highly important* (responsible for a large proportion of each model’s reconstruction), (iii) certain models, such as DinoV2, encode unique features even as they share much of their conceptual basis with others, and (iv) while universal training recovers a significant fraction of the concepts learned by independent single-model SAEs, it also uncovers new shared representations that do not appear to emerge in model-specific training. Finally, we demonstrated a novel application of USAEs—*coordinated activation maximization*—that enables simultaneous visualization of a universal concept across multiple networks. Altogether, our USAE framework offers a practical and powerful tool for multi-model interpretability, shedding light on the commonalities and distinctions that arise when different architectures, tasks, and datasets converge on shared high-level abstractions.

## Impact Statement

This work advances interpretability for machine learning systems. More specifically, understanding shared representations across deep neural networks (DNNs) is essential for scalable interpretability, enabling more effective risk mitigation, robust model design, and compliance with evolving regulations. By moving beyond single-model analysis, we aim to help establish a unified framework for interpreting diverse architectures, fostering transparency and accountability in AI deployment.

## Acknowledgements

This work was completed with support from the Vector Institute, and was funded in part by the Canada First Research Excellence Fund (CFREF) for the Vision: Science to Applications (VISTA) program (K.G.D, H.T), the NSERC Discovery Grant program (K.G.D), the NSERC Canada Graduate Scholarship Doctoral program (M.K), the Ontario Graduate Scholarship (H.T), and a gift from the Chan Zuckerberg Initiative Foundation to establish the Kempner Institute for the Study of Natural and Artificial Intelligence at Harvard University (T.F).

## References

- S. Amir, Y. Gandelsman, S. Bagon, and T. Dekel (2022)
Deep ViT Features as Dense Visual Descriptors.
Proceedings of the European Conference on Computer Vision Workshops .
Cited by: [§4.3](#S4.SS3.p6.3),
[§4.6](#S4.SS6.p1.1).
- M. Baccouche, F. Mamalet, C. Wolf, C. Garcia, and A. Baskurt (2012)
Spatio-temporal convolutional sparse auto-encoder for sequence classification..
In Proceedings of the British Machine Vision Conference,
Cited by: [§2](#S2.p3.1).
- S. Bach, A. Binder, G. Montavon, F. Klauschen, K. Müller, and W. Samek (2015)
On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation.
PloS one 10 (7).
Cited by: [§2](#S2.p2.1).
- D. Bau, B. Zhou, A. Khosla, A. Oliva, and A. Torralba (2017)
Network dissection: quantifying interpretability of deep visual representations.
In IEEE Conference on Computer Vision and Pattern Recognition,
Cited by: [§2](#S2.p2.1).
- U. Bhalla, A. Oesterling, S. Srinivas, F. P. Calmon, and H. Lakkaraju (2024a)
Interpreting CLIP with sparse linear concept embeddings (splice).
Advances in Neural Information Processing Systems.
Cited by: [§2](#S2.p3.1).
- U. Bhalla, S. Srinivas, A. Ghandeharioun, and H. Lakkaraju (2024b)
Towards unifying interpretability and control: evaluation via intervention.
arXiv preprint arXiv:2411.04430.
Cited by: [§2](#S2.p3.1).
- T. Bricken, A. Templeton, J. Batson, B. Chen, A. Jermyn, T. Conerly, N. Turner, C. Anil, C. Denison, A. Askell, R. Lasenby, Y. Wu, S. Kravec, N. Schiefer, T. Maxwell, N. Joseph, Z. Hatfield-Dodds, A. Tamkin, K. Nguyen, B. McLean, J. E. Burke, T. Hume, S. Carter, T. Henighan, and C. Olah (2023)
Towards monosemanticity: decomposing language models with dictionary learning.
Transformer Circuits Thread.
Note: https://transformer-circuits.pub/2023/monosemantic-features/index.html
Cited by: [§2](#S2.p3.1).
- J. Buolamwini and T. Gebru (2018)
Gender shades: intersectional accuracy disparities in commercial gender classification.
In Conference on Fairness, Accountability and Transparency,
Cited by: [§1](#S1.p2.1).
- D. Chanin, J. Wilken-Smith, T. Dulka, H. Bhatnagar, and J. Bloom (2024)
A is for absorption: studying feature splitting and absorption in sparse autoencoders.
arXiv preprint arXiv:2409.14507.
Cited by: [§2](#S2.p3.1).
- J. Chen, H. Mao, Z. Wang, and X. Zhang (2021)
Low-rank representation with adaptive dictionary learning for subspace clustering.
Knowledge-Based Systems 223, pp. 107053.
Cited by: [§2](#S2.p3.1).
- M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi (2014)
Describing textures in the wild.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
Cited by: [§A.4](#A1.SS4.p1.1).
- M. Clarke, H. Bhatnagar, and J. Bloom (2024)
Compositionality and ambiguity: latent co-occurrence and interpretable subspaces.
Note: https://www.lesswrong.com/posts/WNoqEivcCSg8gJe5h/compositionality-and-ambiguity-latent-co-occurrence-and
Cited by: [§2](#S2.p3.1).
- J. Colin, T. Fel, R. Cadène, and T. Serre (2021)
What I cannot predict, I do not understand: a human-centered evaluation framework for explainability methods.
In Advances in Neural Information Processing Systems,
Cited by: [§2](#S2.p2.1).
- E. Commision (2021)
Laying down harmonised rules on artificial intelligence (artificial intelligence act) and amending certain union legislative acts.
European Commision.
Cited by: [§1](#S1.p2.1).
- H. Cunningham, A. Ewart, L. Riggs, R. Huben, and L. Sharkey (2023)
Sparse autoencoders find highly interpretable features in language models.
arXiv preprint arXiv:2309.08600.
Cited by: [§2](#S2.p3.1).
- T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski (2024)
Vision transformers need registers.
Proceedings of the International Conference on Learning Representations.
Cited by: [§1](#S1.p2.1),
[§4](#S4.SS0.SSS0.Px1.p1.1).
- J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L. Fei-Fei (2009)
ImageNet: a large-scale hierarchical image database.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
Cited by: [§4](#S4.SS0.SSS0.Px1.p1.1).
- A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. (2020)
An image is worth 16x16 words: transformers for image recognition at scale.
In International Conference on Learning Representations,
Cited by: [§4](#S4.SS0.SSS0.Px1.p1.1).
- A. Dravid, Y. Gandelsman, A. A. Efros, and A. Shocher (2023)
Rosetta neurons: mining the common units in a model zoo.
In Proceedings of the IEEE/CVF International Conference on Computer Vision,
Cited by: [§1](#S1.p3.1),
[§2](#S2.p4.1).
- B. Dumitrescu and P. Irofti (2018)
Dictionary learning algorithms and applications.
Springer.
Cited by: [§2](#S2.p3.1).
- A. Eamaz, F. Yeganegi, and M. Soltanalian (2022)
On the building blocks of sparsity measures.
IEEE Signal Processing Letters 29, pp. 2667–2671.
Cited by: [§2](#S2.p3.1).
- M. Elad (2010)
Sparse and redundant representations: From theory to applications in signal and image processing.
Springer Science & Business Media.
Cited by: [§2](#S2.p3.1).
- N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds, R. Lasenby, D. Drain, C. Chen, R. Grosse, S. McCandlish, J. Kaplan, D. Amodei, M. Wattenberg, and C. Olah (2022)
Toy models of superposition.
arXiv preprint arXiv:2209.10652.
Cited by: [§A.6](#A1.SS6.p1.1),
[§2](#S2.p3.1).
- L. Engstrom, A. Ilyas, S. Santurkar, D. Tsipras, B. Tran, and A. Madry (2019)
Adversarial robustness as a prior for learned representations.
arXiv preprint arXiv:1906.00945.
Cited by: [§3.3](#S3.SS3.p1.1).
- T. Fel, L. Bethune, A. K. Lampinen, T. Serre, and K. Hermann (2024)
Understanding visual feature reliance through the lens of complexity.
Advances in Neural Information Processing Systems.
Cited by: [§4.1](#S4.SS1.p2.1).
- T. Fel, T. Boissin, V. Boutin, A. Picard, P. Novello, J. Colin, D. Linsley, T. Rousseau, R. Cadène, L. Goetschalckx, L. Gardes, and T. Serre (2023a)
Unlocking feature visualization for deeper networks with magnitude constrained optimization.
In Advances in Neural Information Processing Systems,
Cited by: [§3.3](#S3.SS3.p1.1),
[§3.3](#S3.SS3.p2.19).
- T. Fel, V. Boutin, M. Moayeri, R. Cadène, L. Bethune, M. Chalvidal, and T. Serre (2023b)
A holistic approach to unifying automatic concept extraction and concept importance estimation.
Advances in Neural Information Processing Systems.
Cited by: [§2](#S2.p2.1),
[§2](#S2.p3.1),
[§4](#S4.SS0.SSS0.Px1.p1.1).
- T. Fel, R. Cadene, M. Chalvidal, M. Cord, D. Vigouroux, and T. Serre (2021)
Look at the variance! Efficient black-box explanations with sobol-based sensitivity analysis.
In Advances in Neural Information Processing Systems,
Cited by: [§2](#S2.p2.1).
- T. Fel, A. Picard, L. Bethune, T. Boissin, D. Vigouroux, J. Colin, R. Cadène, and T. Serre (2023c)
CRAFT: concept recursive activation factorization for explainability.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
Cited by: [§2](#S2.p2.1),
[§3.3](#S3.SS3.p1.1).
- R. Fong, M. Patrick, and A. Vedaldi (2019)
Understanding deep networks via extremal perturbations and smooth masks.
In Proceedings of the IEEE International Conference on Computer Vision,
Cited by: [§2](#S2.p2.1).
- Y. Gandelsman, A. A. Efros, and J. Steinhardt (2024)
Interpreting CLIP’s image representation via text-based decomposition.
Proceedings of the International Conference on Learning Representations.
Cited by: [§2](#S2.p2.1).
- L. Gao, T. D. la Tour, H. Tillman, G. Goh, R. Troll, A. Radford, I. Sutskever, J. Leike, and J. Wu (2024)
Scaling and evaluating sparse autoencoders.
arXiv preprint arXiv:2406.04093.
Cited by: [§A.1](#A1.SS1.p1.2),
[§2](#S2.p3.1),
[§3](#S3.SS0.SSS0.Px2.p1.12),
[§4](#S4.SS0.SSS0.Px1.p1.1).
- J. Genone and T. Lombrozo (2012)
Concept possession, experimental semantics, and hybrid theories of reference.
Philosophical Psychology 25 (5), pp. 717–742.
Cited by: [§2](#S2.p2.1).
- A. Ghiasi, H. Kazemi, E. Borgnia, S. Reich, M. Shu, M. Goldblum, A. G. Wilson, and T. Goldstein (2022)
What do vision transformers learn? A visual exploration.
arXiv preprint arXiv:2212.06727.
Cited by: [§3.3](#S3.SS3.p1.1).
- A. Ghiasi, H. Kazemi, S. Reich, C. Zhu, M. Goldblum, and T. Goldstein (2021)
Plug-in inversion: model-agnostic inversion for vision with data augmentations.
Proceedings of the International Conference on Machine Learning.
Cited by: [§3.3](#S3.SS3.p1.1).
- A. Ghorbani, J. Wexler, J. Y. Zou, and B. Kim (2019)
Towards automatic concept-based explanations.
In Advances in Neural Information Processing Systems,
Cited by: [§2](#S2.p2.1).
- N. Gillis (2020)
Nonnegative matrix factorization.
SIAM.
Cited by: [§4.1](#S4.SS1.p1.1).
- L. Gorton (2024)
The missing curve detectors of inceptionv1: applying sparse autoencoders to inceptionv1 early vision.
arXiv preprint arXiv:2406.03662.
Cited by: [§2](#S2.p3.1).
- M. Graziani, A. Nguyen, L. O’Mahony, H. Müller, and V. Andrearczyk (2023)
Concept discovery and dataset exploration with singular value decomposition.
In WorkshopProceedings of the International Conference on Learning Representations,
Cited by: [§2](#S2.p2.1).
- C. Hamblin, T. Fel, S. Saha, T. Konkle, and G. Alvarez (2024)
Feature Accentuation: revealing ‘What’ features respond to in natural images.
arXiv preprint arXiv:2402.10039.
Cited by: [§3.3](#S3.SS3.p1.1).
- S. O. Hansson, M. Belin, and B. Lundgren (2021)
Self-driving vehicles-An ethical overview.
Philosophy & Technology, pp. 1–26.
Cited by: [§1](#S1.p2.1).
- P. Hase and M. Bansal (2020)
Evaluating explainable AI: which algorithmic explanations help users predict model behavior?.
Proceedings of the Annual Meeting of the Association for Computational Linguistics.
Cited by: [§2](#S2.p2.1).
- P. Henderson, X. Li, D. Jurafsky, T. Hashimoto, M. A. Lemley, and P. Liang (2023)
Foundation models and fair use.
Journal of Machine Learning Research 24 (400), pp. 1–79.
Cited by: [§1](#S1.p3.1).
- T. W. House (2023)
President biden issues executive order on safe, secure, and trustworthy artificial intelligence.
The White House.
Cited by: [§1](#S1.p2.1).
- C. Hsieh, C. Yeh, X. Liu, P. Ravikumar, S. Kim, S. Kumar, and C. Hsieh (2021)
Evaluations and methods for explanation through robustness analysis.
In Proceedings of the International Conference on Learning Representations,
Cited by: [§2](#S2.p2.1).
- M. Huh, B. Cheung, T. Wang, and P. Isola (2024)
Position: The Platonic Representation Hypothesis.
In Proceedings of the International Conference on Machine Learning,
Cited by: [§1](#S1.p3.1),
[§2](#S2.p4.1).
- N. Hurley and S. Rickard (2009)
Comparing measures of sparsity.
IEEE Transactions on Information Theory 55 (10), pp. 4723–4741.
Cited by: [§2](#S2.p3.1).
- S. Ioffe and C. Szegedy (2015)
Batch normalization: accelerating deep network training by reducing internal covariate shift.
In Proceedings of the International Conference on Machine Learning,
Cited by: [§A.1](#A1.SS1.p1.2).
- B. Kim, M. Wattenberg, J. Gilmer, C. Cai, J. Wexler, and F. Viegas (2018)
Interpretability beyond feature attribution: quantitative testing with concept activation vectors (TCAV).
In International Conference on Machine Learning,
Cited by: [§2](#S2.p2.1),
[§3.3](#S3.SS3.p1.1).
- S. S. Y. Kim, N. Meister, V. V. Ramaswamy, R. Fong, and O. Russakovsky (2022)
HIVE: evaluating the human interpretability of visual explanations.
In Proceedings of the IEEE European Conference on Computer Vision,
Cited by: [§2](#S2.p2.1).
- D. P. Kingma and J. Ba (2015)
Adam: a method for stochastic optimization.
Proceedings of the International Conference on Learning Representations.
Cited by: [§A.1](#A1.SS1.p2.5).
- P. W. Koh, T. Nguyen, Y. S. Tang, S. Mussmann, E. Pierson, B. Kim, and P. Liang (2020)
Concept bottleneck models.
In International Conference on Machine Learning,
Cited by: [§2](#S2.p3.1).
- M. Kowal, A. Dave, R. Ambrus, A. Gaidon, K. G. Derpanis, and P. Tokmakov (2024a)
Understanding video transformers via universal concept discovery.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
Cited by: [§1](#S1.p3.1),
[§2](#S2.p2.1),
[§2](#S2.p4.1).
- M. Kowal, M. Siam, M. A. Islam, N. D. Bruce, R. P. Wildes, and K. G. Derpanis (2025)
Quantifying and learning static vs. dynamic information in deep spatiotemporal networks.
IEEE Transactions on Pattern Analysis and Machine Intelligence.
Cited by: [§1](#S1.p2.1).
- M. Kowal, R. P. Wildes, and K. G. Derpanis (2024b)
Visual concept connectome (VCC): open world concept discovery and their interlayer connections in deep models.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
Cited by: [§2](#S2.p2.1).
- N. Kriegeskorte, M. Mur, and P. A. Bandettini (2008)
Representational similarity analysis-connecting the branches of systems neuroscience.
Frontiers in systems neuroscience 2, pp. 249.
Cited by: [§2](#S2.p4.1).
- H. W. Kuhn (1955)
The Hungarian method for the assignment problem.
Naval Research Logistics Quarterly 2, pp. 83–97.
External Links: [Document](https://dx.doi.org/10.1002/nav.3800020109)
Cited by: [§4.4](#S4.SS4.p1.1).
- J. Lindsey, A. Templeton, J. Marcus, T. Conerly, J. Batson, and C. Olah (2024)
Sparse crosscoders for cross-layer features and model diffing.
Note: https://transformer-circuits.pub/2024/crosscoders/index.html
Cited by: [§2](#S2.p4.1).
- Z. Liu, P. Luo, X. Wang, and X. Tang (2015)
Deep learning face attributes in the wild.
In Proceedings of the IEEE International Conference on Computer Vision,
Cited by: [§A.4](#A1.SS4.p1.1).
- S. Mahdizadehaghdam, A. Panahi, H. Krim, and L. Dai (2019)
Deep dictionary learning: a parametric network approach.
IEEE Transactions on Image Processing 28 (10), pp. 4790–4802.
Cited by: [§2](#S2.p3.1).
- A. Mahendran and A. Vedaldi (2015)
Understanding deep image representations by inverting them.
In IEEE Conference on Computer Vision and Pattern Recognition,
Cited by: [§3.3](#S3.SS3.p1.1).
- J. Mairal, F. Bach, and J. Ponce (2014)
Sparse modeling for image and vision processing.
Foundations and Trends® in Computer Graphics and Vision 8 (2-3), pp. 85–283.
Cited by: [§2](#S2.p3.1).
- A. Menon, M. Shrivastava, D. Krueger, and E. S. Lubana (2024)
Analyzing (in) abilities of SAEs via formal languages.
arXiv preprint arXiv:2410.11767.
Cited by: [§2](#S2.p3.1).
- A. Mordvintsev, C. Olah, and M. Tyka (2015)
Inceptionism: going deeper into neural networks.
https://blog.research.google/2015/06/inceptionism-going-deeper-into-neural.html?m=1.
Cited by: [§2](#S2.p3.1).
- L. Moschella, V. Maiorca, M. Fumero, A. Norelli, F. Locatello, and E. Rodolà (2022)
Relative representations enable zero-shot latent space communication.
Proceedings of the International Conference on Learning Representations.
Cited by: [§1](#S1.p3.1).
- S. Muzellec, L. Andeol, T. Fel, R. VanRullen, and T. Serre (2024)
Gradient strikes back: how filtering out high frequencies improves explanations.
Proceedings of the International Conference on Learning Representations.
Cited by: [§2](#S2.p2.1).
- A. Nguyen, J. Yosinski, and J. Clune (2019)
Understanding neural networks via feature visualization: a survey.
Explainable AI: interpreting, explaining and visualizing deep learning.
Cited by: [§2](#S2.p3.1).
- G. Nguyen, D. Kim, and A. Nguyen (2021)
The effectiveness of feature attribution methods and its correlation with automatic evaluation scores.
Advances in Neural Information Processing Systems.
Cited by: [§2](#S2.p2.1).
- C. Olah, A. Mordvintsev, and L. Schubert (2017)
Feature visualization.
Distill.
Note: https://distill.pub/2017/feature-visualization
External Links: [Document](https://dx.doi.org/10.23915/distill.00007)
Cited by: [§3.3](#S3.SS3.p1.1),
[§4.1](#S4.SS1.p2.1).
- M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, M. El-Nouby, N. Ballas, W. Galuba, R. Howes, P. Huang, S. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski (2023)
Dinov2: learning robust visual features without supervision.
Transactions on Machine Learning Research.
Cited by: [§1](#S1.p4.1),
[§4](#S4.SS0.SSS0.Px1.p1.1),
[§4.6.1](#S4.SS6.SSS1.p2.1).
- V. Papyan, Y. Romano, and M. Elad (2017)
Convolutional dictionary learning via local processing.
International Conference on Computer Vision.
Cited by: [§2](#S2.p3.1).
- J. Parekh, P. Khayatan, M. Shukor, A. Newson, and M. Cord (2024)
A concept-based explainability framework for large multimodal models.
Advances in Neural Information Processing Systems.
Cited by: [§2](#S2.p3.1).
- A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, Krueger,Gretchen, and Sutskever,Ilya (2021)
Learning transferable visual models from natural language supervision.
In International Conference on Machine Learning,
Cited by: [§2](#S2.p3.1).
- S. Rajamanoharan, T. Lieberum, N. Sonnerat, A. Conmy, V. Varma, J. Kramár, and N. Nanda (2024)
Jumping ahead: improving reconstruction fidelity with jumprelu sparse autoencoders.
arXiv preprint arXiv:2407.14435.
Cited by: [§2](#S2.p3.1).
- S. Rao, S. Mahajan, M. Böhle, and B. Schiele (2024)
Discover-then-name: task-agnostic concept bottlenecks via automated concept discovery.
In Proceedings of the IEEE European Conference on Computer Vision,
Cited by: [§2](#S2.p3.1).
- R. Rubinstein, A. M. Bruckstein, and M. Elad (2010)
Dictionaries for sparse representation modeling.
Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing.
Cited by: [§2](#S2.p3.1).
- S. Santurkar, A. Ilyas, D. Tsipras, L. Engstrom, B. Tran, and A. Madry (2019)
Image synthesis with a single (robust) classifier.
Advances in Neural Information Processing Systems 32.
Cited by: [§3.3](#S3.SS3.p1.1).
- R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra (2017)
Grad-CAM: visual explanations from deep networks via gradient-based localization.
In IEEE International Conference on Computer Vision,
Cited by: [§2](#S2.p2.1).
- K. Simonyan, A. Vedaldi, and A. Zisserman (2014)
Deep inside convolutional networks: visualising image classification models and saliency maps.
Proceedings of the International Conference on Learning Representations.
Cited by: [§2](#S2.p2.1).
- L. Sixt, M. Granz, and T. Landgraf (2020)
When explanations lie: why many modified BP attributions fail.
In Proceedings of the International Conference on Machine Learning,
Cited by: [§2](#S2.p2.1).
- D. Smilkov, N. Thorat, B. Kim, F. Viégas, and M. Wattenberg (2017)
Smoothgrad: Removing noise by adding noise.
arXiv preprint arXiv:1706.03825.
Cited by: [§2](#S2.p2.1).
- J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller (2014)
Striving for simplicity: the all convolutional net.
arXiv preprint arXiv:1412.6806.
Cited by: [§2](#S2.p2.1).
- I. Sucholutsky, L. Muttenthaler, A. Weller, A. Peng, A. Bobu, B. Kim, B. C. Love, E. Grant, I. Groen, J. Achterberg, et al. (2023)
Getting aligned on representational alignment.
arXiv preprint arXiv:2310.13018.
Cited by: [§2](#S2.p4.1).
- M. Sundararajan, A. Taly, and Q. Yan (2017)
Axiomatic attribution for deep networks.
In Proceedings of the International Conference on Machine Learning,
Cited by: [§2](#S2.p2.1).
- V. Surkov, C. Wendler, M. Terekhov, J. Deschenaux, R. West, and C. Gulcehre (2024)
Unpacking sdxl turbo: interpreting text-to-image models with sparse autoencoders.
arXiv preprint arXiv:2410.22366.
Cited by: [§2](#S2.p3.1).
- A. Tamkin, M. Taufeeque, and N. D. Goodman (2023)
Codebook features: sparse and discrete interpretability for neural networks.
arXiv preprint arXiv:2310.17230.
Cited by: [§2](#S2.p3.1).
- S. Tariyal, A. Majumdar, R. Singh, and M. Vatsa (2016)
Deep dictionary learning.
IEEE Access 4, pp. 10096–10109.
Cited by: [§2](#S2.p3.1).
- A. Tasissa, P. Tankala, J. M. Murphy, and D. Ba (2023)
K-deep simplex: manifold learning via local dictionaries.
IEEE Transactions on Signal Processing.
Cited by: [§2](#S2.p3.1).
- I. Tošić and P. Frossard (2011)
Dictionary learning.
IEEE Signal Processing Magazine 28 (2), pp. 27–38.
Cited by: [§2](#S2.p3.1).
- D. Tsipras, S. Santurkar, L. Engstrom, A. Turner, and A. Madry (2019)
Robustness may be at odds with accuracy.
In Proceedings of the International Conference on Learning Representations,
Cited by: [§3.3](#S3.SS3.p1.1).
- J. Vielhaben, S. Bluecher, and N. Strodthoff (2023)
Multi-dimensional concept discovery (MCD): a unifying framework with completeness guarantees.
Transactions on Machine Learning Research.
Cited by: [§2](#S2.p2.1).
- M. Wattenberg and F. B. Viégas (2024)
Relational composition in neural networks: a survey and call to action.
arXiv preprint arXiv:2407.14662.
Cited by: [§2](#S2.p3.1).
- R. Wightman (2019)
PyTorch Image Models.
GitHub.
Note: [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
External Links: [Document](https://dx.doi.org/10.5281/zenodo.4414861)
Cited by: [§4](#S4.SS0.SSS0.Px1.p1.1).
- R. Williams (1986)
Inverting a connectionist network mapping by back-propagation of error.
In Proceedings of the Annual Meeting of the Cognitive Science Society,
Cited by: [§3.3](#S3.SS3.p1.1).
- S. Wright (1921)
Correlation and causation.
Journal of Agricultural Research 20 (7), pp. 557–585.
Cited by: [§4.2](#S4.SS2.p1.3).
- Y. Yu, S. Buchanan, D. Pai, T. Chu, Z. Wu, S. Tong, B. Haeffele, and Y. Ma (2023)
White-box transformers via sparse rate reduction.
Advances in Neural Information Processing Systems.
Cited by: [§2](#S2.p3.1).
- M. D. Zeiler and R. Fergus (2014)
Visualizing and understanding convolutional networks.
In Proceedings of the IEEE European Conference on Computer Vision,
Cited by: [§2](#S2.p2.1).
- X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer (2023)
Sigmoid loss for language image pre-training.
In IEEE International Conference on Computer Vision,
Cited by: [§4](#S4.SS0.SSS0.Px1.p1.1),
[§4.6](#S4.SS6.p1.1).
- R. Zhang, P. Madumal, T. Miller, K. A. Ehinger, and B. I. Rubinstein (2021)
Invertible concept-based explanations for CNN models with non-negative concept activation vectors.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Cited by: [§2](#S2.p2.1).
- J. Zhou, C. Wei, H. Wang, W. Shen, C. Xie, A. Yuille, and T. Kong (2021)
IBoT: image bert pre-training with online tokenizer.
Proceedings of the International Conference on Learning Representations.
Cited by: [§4.6.1](#S4.SS6.SSS1.p2.1).

## Appendix A Appendix

### A.1 SAE Training Implementation details

We modify the TopK Sparse Autoencoder (SAE) (Gao et al., 2024) by replacing the $\ell_{2}$ loss with an $\ell_{1}$ loss, as we find that this adjustment improves both training dynamics and the interpretability of the learned concepts. The encoder consists of a single linear layer followed by batch normalization (Ioffe and Szegedy, 2015) and a ReLU activation function, while the decoder is a simple dictionary matrix.

For all experiments, we use a dictionary of size $8\times 768=6144$ which is an expansion factor of $8$ multiplied by the largest feature dimension in any of the three models, $768$. All SAE encoder-decoder pairs have independent Adam optimizers (Kingma and Ba, 2015), each with an initial learning rate of $3\mathrm{e}{-4}$, which decays to $1\mathrm{e}{-6}$ following a cosine schedule with linear warmup. To account for variations in activation scales caused by architectural differences, we standardize each model’s activations using 1000 random samples from the training set. Specifically, we compute the mean and standard deviation of activations for each model and apply standardization, thereby preserving the relative relationship between activation magnitudes and directions while mitigating scale differences.

Since SigLIP does not incorporate a class token, we remove class tokens from DinoV2 and ViT to ensure consistency across models.
Additionally, we interpolate the DinoV2 token count to match a patch size of $16\times 16$ pixels, aligning it with SigLIP and ViT. We train all USAEs on a single NVIDIA RTX 6000 GPU, with training completing in approximately three days.

### A.2 Unique DinoV2 Concepts

DinoV2’s unique concepts are presented in Figures [10](#A1.F10) and [11](#A1.F11). Interestingly, we find concepts that solely fire for DinoV2 related to depth, perspective, and geometric cues.

Figure: Figure 10: Qualitative results of low-entropy concepts that fire for DinoV2. We discover features related to depth cues for foreground objects as well as background in concept 4756 (above) and 1710 (below).
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/x8.png

Figure: Figure 11: Qualitative results for low-entropy concepts that fire for DinoV2. We discover DinoV2 independent features that are not universal suggesting 3D understanding like corners (concepts 1530), top face of rectangular prism (concept 4191), and brim of dome (concept 3448).
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/x9.png

### A.3 Unique SigLIP Concepts

Similar to DinoV2, we isolate concepts with low firing-entropy where probability mass is concentrated for SigLIP. Example concepts are presented in Fig. [12](#A1.F12). We observe concepts that fire for both visual and textual elements of the same concept. Concept 5718 fires for both the shape of a star, as well as regions of images with the word or even just a subset of letters on a bottlecap and sign at different scales. Additionally, concept 2898 fires broadly for various musical instruments, as well as music notes, while concept 923 fires for the letter ‘C’. For each of these concepts, the coordinated activation maximization visualization has both the physical semantic representation of the concept, as well as printed text. The presence of image and textual elements are expected given SigLIP is trained as a vision-language model with a contrastive learning objective, where the aim is to align image and text latent representations from separate image and language encoders. While we do not train on any activations directly from the language model, we still observe textual concepts in our image-space visualizations.

Figure: Figure 12: Qualitative results of low-entropy SigLIP concepts. We consistently find concepts that fire for abstract concepts in image space such as images or text of ‘star’ (concept 923), letters (concept 5718), and music notes (concept 2958).
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/x10.png

### A.4 Out-of-Distribution Generalization

In order to assess the out-of-distribution capabilities of our approach, we
use DTD (Cimpoi et al., 2014) and CelebA (Liu et al., 2015) as the validation dataset for our ImageNet trained USAEs and show strong evidence of generalization outside of the training distribution as seen in Table [1](#A1.T1). We find consistent activation reconstruction accuracy (measured by MSE and R2), consistent trends in co-firing metrics in Fig. [13](#A1.F13) and visualize some of the most important concepts for these new datasets, along with their associated highest activating images, from ImageNet in Fig. [14](#A1.F14) and [15](#A1.F15). Despite differences in domain and semantics, USAEs trained on ImageNet exhibited robust generalization to both DTD and CelebA. Importantly, many of the concepts identified in these datasets also aligned with high-activation concepts from ImageNet, suggesting that the USAE dictionary captures generalizable structure beyond its training data.

**Table 1: Performance comparison of vision models across different datasets. Lower MSE and higher R² indicate better performance.**
|  | Mean Squared Error (MSE) $\downarrow$ | Coefficient of Determination (R²) $\uparrow$ |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | ImageNet | DTD | CelebA | ImageNet | DTD | CelebA |
| SigLIP | 0.39 | 0.38 | 0.48 | 0.61 | 0.54 | 0.52 |
| DinoV2 | 0.19 | 0.26 | 0.22 | 0.77 | 0.69 | 0.75 |
| ViT | 0.41 | 0.56 | 0.69 | 0.59 | 0.46 | 0.45 |

Figure: Figure 13: Zero-shot generalization quantitative results of universal concepts on out-of-distribution datasets. Top: When applying our ImageNet trained USAE to the validation set of CelebA we find consistent trends across each of our universality metrics. We find a clear correlation between co-firing and concept importance. The distribution over firing entropy also indicates concepts that fire uniquely for a single model, two of three models, and universal concepts that fire for all three. Bottom: When applying our ImageNet trained USAE to the validation set of DTD we find consistent trends across each of our universality metrics. We find a clear correlation between co-firing and concept importance. The distribution over firing entropy is tri-modal, indicating concepts that fire uniquely for a single model, two of three models, and universal concepts that fire for all three.
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/x11.png

Figure: Figure 14: Qualitative examples of zero-shot generalization of ImageNet trained USAE on CelebA. Top: We depict high-level visual concept dimension related to sunglasses and the highest activating images for the validation sets of both ImageNet and CelebA. Bottom: We depict the lower-jaw concept’s highest activating images for the validation set of ImageNet and CelebA. This jaw concept generalizes beyond animal jaw to include human jaws as seen from our CelebA heatmaps.
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/images/Appendix/celeba_concept_486_sunglasses.jpg

Figure: Figure 15: Qualitative examples of zero-shot generalization of ImageNet trained USAE on DTD. Top: We depict a checkerboard concept’s highest activating images for the validation set of ImageNet and DTD. This checkerboard concept generalizes from low-level textures in DTD like tiles to sudoku and crossword puzzles in ImageNet. Bottom: We depict a concept for zebra stripes and its highest activating images for the validation set of ImageNet and DTD. This stripe concept generalizes across scales for images zoomed in on the animal in DTD to across a whole zebra in ImageNet.
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/images/Appendix/dtd_concept_273_checkerboard.jpg

### A.5 Additional Results

#### A.5.1 Additional Quantitative Results

Figure [16](#A1.F16) presents concept consistency distributions across models for the top 1,000 co-firing concepts. We find an overall increase in consistency between individual and universal concepts: the most universal (highest co-firing) concepts are more likely to be found in each model’s respective independently trained SAE. Within this thresholded range, we find DinoV2 to exhibit the highest similarity between individual and universal concepts with an average cosine similarity of $0.65$, followed by ViT at $0.52$ and SigLIP at $0.40$. DinoV2 concepts seem to be better represented in the universal space, suggesting that some models may have more universal concepts than others.

Figure: Figure 16: Top 1000 co-firing concept consistency between independent SAEs and Universal SAEs. Our universal training objective discovers universal concepts that have overlap (i.e., cosine similarity) with those discovered with independent training. In descending order, Universal SAEs have highest overlap with independently trained DinoV2, ViT, and SigLIP. The smaller overlap observed with SigLIP suggests the aligned image-language embedding space produces unique concepts that are more distinct from those in DinoV2 and ViT.
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/x13.png

#### A.5.2 Additional Qualitative Results

We provide additional universal concept visualizations for the top activating images for that concept across each model. Specifically, we showcase low-level concepts in Fig. [17](#A1.F17) related to texture like shell and wood for concepts 1716 and 2533, respectively, as well as tiling for concept 5563. We also showcase high-level concepts in Fig. [18](#A1.F18) related to environments like auditoriums in concept 4691, object interactions like ground contact in concept 5346, as well as facial features like snouts in concept 3479.

Figure: Figure 17: Qualitative results of universal concepts. We depict low-level visual features related to textures, such as shells (concept 1716), wood (concept 2533), and tiling (concept 5563).
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/x14.png

Figure: Figure 18: Qualitative results of universal concepts. We depict high-level visual features related to environments, such as auditoriums (concept 4691), ground contact (concept 5346), and animal snouts (concept 3479).
Refer to caption: https://arxiv.org/html/2502.03714/2502.03714v2/x15.png

### A.6 Limitations

Our universal concept discovery objective successfully discovers fundamental visual concepts encoded between vision models trained under distinct objectives and architectures, and allows us to explore features that fire distinctly for a particular model of interest under our regime. However, we note some limitations that we aim to address in future work. We notice some sensitivity to hyperparameters when increasing the number of models involved in universal training, and use hyperparameter sweeps to find an optimal configuration.
We also constrain our problem to discovering features at the last layer of each vision model. We choose to do so as a tractable first step in this novel paradigm of *learning* to discover universal features. We leave an exploration of universal features across different layer depths for future work.
Lastly, we do find qualitatively that a small percentage of concepts are uninterpretable. They may be still stored in superposition (Elhage et al., 2022) or they could be useful for the model but simply difficult for humans to make sense of. This is a phenomena that independently trained SAEs suffer from as well.
Many of the limitations of our approach are tightly coupled to the limitations of training independent SAEs, an active area of research.