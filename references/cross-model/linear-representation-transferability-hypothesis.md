Title: Linear Representation Transferability Hypothesis: Leveraging Small Models to Steer Large Models
ArXiv: 2506.00653
Authors: Femi Bello, Anubrata Das, Fanzhi Zeng, Fangcong Yin, Liu Leqi
Sections: 23
Estimated tokens: 15.9k

## Contents
- 1 Introduction
- 2 Linear Representation Transferability (LRT) Hypothesis
  - 2.1 Linear Representation Transferability Hypothesis
  - 2.2 Conceptual Framework: Universal Representation Space
    - Empirical Observation.
  - 2.3 Learning Objective for Linear Representation Mapping
- 3 Application: Steering Large Models with Small Models
  - 3.1 Training and Evaluation Setup
  - 3.2 Proof of Concept:
Feature-to-Hidden vs. Hidden-to-Hidden Mapping
  - 3.3 Steering to Remove Refusal Behaviors
  - 3.4 Logit Based Evaluation
  - 3.5 Limits of transferrability
- 4 Related work
- 5 Practical Implications and Future Directions
  - Practical Implications.
  - Future Directions.
- Acknowledgements
- References
- Appendix A Discussion on Many-to-1 and Many-to-Many mappings
- Appendix B Significance Analysis for Section 3.4
- Appendix C Compute and Training details
  - C.0.1 Latent2Latent & SAE2Latent
  - C.1 System Prompts and Extended Data

## Abstract

Abstract It has been hypothesized that neural networks with similar architectures trained on similar data learn shared representations relevant to the learning task.
We build on this idea by extending the conceptual framework where representations learned across models trained on the same data
can be expressed as linear combinations of a universal set of basis features.
These basis features underlie the learning task itself and remain consistent across models, regardless of scale.
From this framework, we propose the Linear Representation Transferability (LRT) Hypothesis—that there exists an affine transformation between the representation spaces of different models.
To test this hypothesis, we learn affine mappings between the hidden states of models of different sizes and evaluate whether steering vectors—directions in hidden state space associated with specific model behaviors—retain their semantic effect when transferred from small to large language models using the learned mappings.
We find strong empirical evidence
that such affine mappings can preserve steering behaviors.
These findings suggest that representations learned by small models can be used to guide the behavior of large models, and that the LRT hypothesis may be a promising direction on understanding representation alignment across model scales.

## 1 Introduction

When models share similar architectures and training data,
it has been hypothesized that they might learn similar internal representations.
Why this may happen—and how—is still not well understood.
Uncovering the mechanisms behind this
is valuable from both theoretical and practical perspectives.
Theoretically, it can shed light on
how models generalize and potential
shared inductive biases across model classes.
Practically, it opens the possibility of using small models’ internal representations to efficiently infer properties of larger ones.

Two lines of research—one on the structure of representations within models and the other on the similarity of representations across models—have provided insights on how neural networks may encode and share information.
Research in Transformer interpretability has begun to develop theoretical frameworks for understanding how large language models represent information in their hidden states (Elhage et al., 2021, 2022; Park et al., 2023). Theories like the Linear Representation Hypothesis (Park et al., 2023) and Superposition (Elhage et al., 2022) have hypothesized that transformers encode learned representations as linear combinations of human-interpretable concepts (“features”).
While this line of work does not address how such representations relate across models,
there are prior works studying representational similarities across neural networks.
These works focus on proposing metrics to compare the structure of the learned representations statistically and geometrically (Klabunde et al., 2023b; Kornblith et al., 2019; Raghu et al., 2017).
Though the proposed notions of similarity are informative,
they do not provide a deeper understanding of how two networks may be similar and the impacts of that similarity in downstream tasks.
Theories that do try to explain these similarities often rely on heavy assumptions that have not been empirically verified.

Bridging these perspectives, we investigate whether the linear structure of representations can explain and support the transfer of learned representations across models.
We build on the conceptual idea that representations learned across different models can be expressed as a linear combination of a universal set of basis features.
These features underlie the learning task and are consistent across models, regardless of scale. Extending this idea, we examine the relationship between different models’ hidden states. Specifically, we ask: what is the relationship between the hidden states in models of different scales?
To answer that, we propose the Linear Representation Transferability (LRT) hypothesis, that there exists an affine transformation between the representations learned across different models in the same architecture family and trained on similar training corpus (Section [2.1](https://arxiv.org/html/2506.00653v3#S2.SS1)).

We investigate the LRT hypothesis by
first providing a conceptual framework
on explaining why this may hold and
empirically verify the implication of the proposed framework (Section [2.2](https://arxiv.org/html/2506.00653v3#S2.SS2)).
We then learn affine maps between the hidden states of different models (Section [2.3](https://arxiv.org/html/2506.00653v3#S2.SS3)).
To evaluate these maps, we perform functional interventions—specifically, steering tasks—to test whether steering vectors learned on smaller models can be effectively transferred to larger models within the same family (Section [3](https://arxiv.org/html/2506.00653v3#S3)).
Our results show that a *fixed* affine map from a smaller model’s hidden states to those of a larger model enables successful transfer of steering vectors, allowing us to steer the larger model using steering vectors learned from the smaller one.
Our results provide evidence supporting the LRT hypothesis with direct application to a downstream task.

To summarize, our contributions are as follows:

- •
We propose the Linear Representation Transferability (LRT) Hypothesis and introduce a conceptual framework to explain why it holds, based on the idea that models’ representation spaces are subspaces of a universal representation space.
- •
We empirically evaluate the LRT hypothesis by transferring steering vectors between models and measuring the similarity of their (semantic) effects.

Our conceptual framework lays the groundwork for future research, where smaller models serve as a sandbox for probing and understanding phenomena that extend to larger models. The LRT hypothesis carries practical implications for model training and efficient inference, and it also points to future directions for studying representation transferability across models (Section [5](https://arxiv.org/html/2506.00653v3#S5)).

## 2 Linear Representation Transferability (LRT) Hypothesis

Given a transformer $\mathcal{M}$,
for any layer $l\in[L]$, an input $\bm{x}=(\bm{x}_{1},\cdots,\bm{x}_{T})$,
we consider the hidden state—each block’s output—$\bm{h}_{l}(\bm{x})$ which satisfies:

$$ $\displaystyle\bm{h}_{l+1}(\bm{x})=\bm{h}_{l}(\bm{x})+F_{l}(\bm{h}_{l}(\bm{x})),$ $$

where $F_{l}$ is the residual output of layer $l$
and $\bm{h}_{1}(\bm{x})$ is the initial input embedding.

Given a source model $\mathcal{M}_{S}$ and a target model $\mathcal{M}_{T}$,
we denote their corresponding number of layers and hidden state dimension as $L_{S},L_{T}$ and $d_{S},d_{T}$, respectively.
At a high level, we aim to understand the relationship between representations learned by different models trained on the same (or similar) data, and whether there exists some universal or shared structure among them.
Under our notation, we investigate the connection between the set of hidden states from the source model, $\{\bm{h}_{l_{S}}^{S}(\bm{x})\}_{l_{S}\in[L_{S}]}$, and those from the target model, $\{\bm{h}_{l_{T}}^{T}(\bm{x})\}_{l_{T}\in[L_{T}]}$.

In the following, we present the Linear Representation Transferability (LRT) Hypothesis, which posits that representations learned across models are linearly related (Section [2.1](https://arxiv.org/html/2506.00653v3#S2.SS1)),
along with a conceptual framework explaining why this hypothesis may hold (Section [2.2](https://arxiv.org/html/2506.00653v3#S2.SS2)).
We then operationalize this hypothesis by learning a universal affine mapping across representations learned under different models (Section [2.3](https://arxiv.org/html/2506.00653v3#S2.SS3)).

Figure: Figure 1: Illustration of the conceptual framework.
Refer to caption: https://arxiv.org/html/2506.00653/extracted/6513478/figs/sae-generalization.png

### 2.1 Linear Representation Transferability Hypothesis

Despite differences in architectural details (e.g., model size, pre- v.s. post-LayerNorm) or training dynamics, transformers trained on the same (or similar) data and using the same tokenizer are exposed to identical (or similar) input distributions and subword decompositions. As a result, they may discover shared underlying structures in the data—such as syntax, semantics, or word co-occurrence patterns—which are often reflected in their representations.
These shared structures may point to the existence of a shared *universal representation space* for the representations learned by these models.

The Linear Representation Transferrability (LRT) Hypothesis builds on the above intuition, as well as on the observation that, within a single model, high-level concepts are represented linearly in some representation space (e.g., (Park et al., 2023)).
If a shared *universal representation space* does exist,
then each model’s representation space may be a subspace of it.
The LRT Hypothesis posits that there exists a linear mapping between representations that lie in these subspaces.
We present the LRT hypothesis more formally below.

We note that several aspects of the LRT Hypothesis are intentionally left abstract.
This is by design, as our goal is to focus on the core idea of linear transferability of representations across models, rather than committing to overly specific formulations.
First, the linear mapping may not be unique—particularly when the source dimensionality $d_{S}$ is smaller than the target dimensionality
$d_{T}$—in which case multiple valid mappings may exist.
Second, we use the term “approximately[$\approx$]” to allow for a degree of mismatch between the transferred source model representation and the original target model representation.
A more formal treatment could involve quantifying this approximation using a probably approximately correct (PAC) learning-style framework—for example, bounding the probability that the approximation error exceeds a small threshold $\epsilon$.
Finally, the choice of specific source and target layers is motivated by the empirical observations that representations at different depths of the model often exhibit different behaviors (Lad et al., 2024).
Despite these considerations, we emphasize that the central idea underlying the hypothesis remains: representations learned by different models, under shared training data distribution and tokenization, may be linearly related.

### 2.2 Conceptual Framework: Universal Representation Space

As we have briefly dicussed in Section [2.1](https://arxiv.org/html/2506.00653v3#S2.SS1),
the LRT hypothesis is grounded in two key intuitions:

- •
First, when models are trained on the same data distribution—using the same training data and tokenizer (as the tokenizer also influences the data distribution)—they tend to capture similar underlying structures in the data. These structures include statistical patterns like word co-occurrences and semantic relationships.
This observation suggests the possible existence of a universal representation space shared across different models trained on the same or similar distributions.
Formally, we denote the universal set of feature vectors as $\bm{W}_{U}\in\mathbb{R}^{n\times D}$, where $n$ is the number of features and $D<n$ is the feature dimension.
The features span (possibly a subspace of) $\mathbb{R}^{D}$.
- •
Second, we rely on the general linear represnetation hypothesis that learned representations (or hidden states) are linear combinations of some high-dimensional features. Given the shared universal feature vectors, formally, the hidden states can be modelled as:
$\displaystyle\bm{h}(\bm{x})=\bm{W}_{U}^{\top}c(\bm{x})+\bm{b},$
(1)
where $c(\bm{x})\in\mathbb{R}^{n}$ outputs a sparse vector indicating the activated features in $\bm{W}_{U}$ and $\bm{b}$ is the bias.
We note that the linear model presented in ([1](https://arxiv.org/html/2506.00653v3#S2.E1)) is closely related to prior work using sparse autoencoders (SAEs) to uncover high-dimensional “monosemantic” features (Bricken et al., 2023). In that context, the monosemantic features
are given in the decoder matrix of the SAE,
equivalent to $\bm{W}_{U}$ in our case;
the encoder matrix of the SAE gives the coefficients $c(x)$.
Though we use a similar linear formulation,
our goal is to establish connections between these monosemantic features across different models.

In the following, we illustrate how these two key aspects provide a conceptual framework for reasoning about why the LRT hypothesis may hold.
At a high level, the key idea is that models such as $\mathcal{M}_{S}$ and $\mathcal{M}_{T}$, when trained on similar data distributions, learn representations that lie in different subspaces of a shared higher-dimensional feature space.
Each model may apply a linear projection of the shared representations onto a model-specific subspace,
which accounts for the possible linear relationships between their internal representations.
Empirically, we support this intuition by analyzing decoder matrices obtained from sparse autoencoders trained on the hidden states of two models. These decoder matrices serve as feature vectors for each model’s representation space. We find that the decoder matrices are approximately linearly related to each other, providing strong evidence for our conceptual framework.

For the hidden state $\bm{h}^{S}_{l_{S}}(\bm{x})$ of the source model at layer $l_{S}$
and the hidden state $\bm{h}^{T}_{l_{T}}(\bm{x})$ of the target model
at layer $l_{T}$,
we model them as linear combinations of features drawn from the shared universal feature space $\bm{W}_{U}$, but projected into model-specific subspaces. Formally,

$$ $\displaystyle\bm{h}^{S}_{l_{S}}(\bm{x})$ $\displaystyle=\bm{W}_{S}^{\top}c^{S}_{l_{S}}(\bm{x})+\bm{b}_{l_{S}}^{S},$ $\displaystyle\bm{h}^{T}_{l_{T}}(\bm{x})$ $\displaystyle=\bm{W}_{T}^{\top}c^{T}_{l_{T}}(\bm{x})+\bm{b}_{l_{T}}^{T},$ $$

where $\bm{W}_{S}\in\mathbb{R}^{n\times D_{S}}$ and $\bm{W}_{T}\in\mathbb{R}^{n\times D_{T}}$ are model-specific feature matrices,
that are projections(^1^11Throughout the paper, we use the term “projection” loosely to refer to a general linear transformation to a subspace, not necessarily a projection in the strict mathematical sense (i.e., an idempotent operator).) of the universal feature matrix $\bm{W}_{U}\in\mathbb{R}^{n\times D}$ onto lower-dimensional subspaces, with $D>D_{S},D_{T}$.
The coefficients $c^{S}_{l_{S}}(\bm{x}),c^{T}_{l_{T}}(\bm{x})\in\mathbb{R}^{n}$ determine how input $\bm{x}$ activates the relevant features in each subspace and $\bm{b}_{l_{S}}^{S}\in\mathbb{R}^{d_{S}},\bm{b}_{l_{T}}^{T}\in\mathbb{R}^{d_{T}}$ are bias terms.

To make this concrete, we consider the singular value decomposition of the universal feature matrix $\bm{W}_{U}\in\mathbb{R}^{n\times D}$ (where $n\gg D$):

$$ $\displaystyle\bm{W}_{U}=\mathbf{\Lambda}\bm{\Sigma}\mathbf{V}_{U}^{\top},$ $$

where
$\mathbf{\Lambda}\in\mathbb{R}^{n\times D}$ contains the left singular vectors that span the column space of $\bm{W}_{U}$,
$\bm{\Sigma}\in\mathbb{R}^{D\times D}$ is a diagonal matrix of singular values, and
$\mathbf{V}\in\mathbb{R}^{D\times D}$ contains the right singular vectors that form an orthonormal basis for the universal feature space.
For simplicity, define:

$$ $\displaystyle\mathbf{Q}:=\mathbf{\Lambda}\bm{\Sigma}\in\mathbb{R}^{n\times D},$ $$

so that $\bm{W}_{U}=\mathbf{Q}\bm{V}_{U}^{\top}$.
In this form, $\bm{V}_{U}$ represents the universal basis features, and $\mathbf{Q}$ maps basis features (e.g., in $\bm{V}_{U}$) to feature vectors (e.g., in $\bm{W}_{U}$).

For specific models like the source and target models of our interests,
under our hypothesis that there is a shared feature space $\bm{W}_{U}$ with basis features $\mathbf{V}_{U}$,
their feature spaces are projections of the shared one.
In particular, their corresponding basis features will be

$$ $\displaystyle\bm{V}_{S}=\bm{P}_{S}^{\top}\bm{V}_{U},\quad\bm{V}_{T}=\bm{P}_{T} ^{\top}\bm{V}_{U},$ $$

where
$\bm{P}_{S}\in\mathbb{R}^{D\times D_{S}}$ and $\bm{P}_{T}\in\mathbb{R}^{D\times D_{T}}$ are model-specific projection matrices that define the feature subspaces used by the source and target models, respectively.
Consequently,
by using (i) the same feature transformation matrix $\bm{Q}=\mathbf{\Lambda}\bm{\Sigma}$ that transfers basis vectors to feature vectors(^2^22As a side note, we have considered cases where the transformation $\bm{Q}$ from basis vectors to feature vectors remains the same across different models. One can also consider cases where $\bm{Q}$ is composed with an invertible operation, such as a rotation or reflection, in which the LRT hypothesis still holds.),
along with (ii) a linear transformation of the feature indices that maps feature indices in the shared feature space to those in the source and target model,
we get
the model-specific feature matrices:

$$ $\displaystyle\bm{W}_{S}=\mathbf{R}_{S}\mathbf{Q}\bm{V}_{S}^{\top}=\mathbf{R}_{ S}\bm{W}_{U}\bm{P}_{S},\quad\bm{W}_{T}=\mathbf{R}_{T}\mathbf{Q}\bm{V}_{T}^{ \top}=\mathbf{R}_{T}\bm{W}_{U}\bm{P}_{T}.$ $$

We consider feature indices transformations $\mathbf{R}_{S},\mathbf{R}_{T}\in\mathbb{R}^{n\times n}$
that are permutation matrices. That is, the ordering of the feature vectors in the source and target models may be permutations of each other.

To summarize, in our proposed conceptual framework, the key idea is that there exists a set of shared feature vectors $\bm{W}_{U}$ common to models trained on similar data . These feature vectors can be expressed as linear combinations of a universal bases $\bm{V}_{U}$.
For each specific model (e.g., $\mathcal{M}_{S},\mathcal{M}_{T}$),
its hidden states lie within a subspace of the space spanned by $\bm{V}_{U}$. In particular, the bases vectors (e.g., $\bm{V}_{S}$ and $\bm{V}_{T}$) used by each model are projections of the universal bases $\bm{V}_{U}$.

Under this conceptual framework, we can infer why the LRT hypothesis may hold.
If the source and target model hidden states
($\bm{h}^{S}_{l_{S}}(\bm{x})$, $\bm{h}^{T}_{l_{T}}(\bm{x})$)
are using the same features
with the same strength (i.e., $\mathbf{R}_{S}^{-1}c_{l_{S}}^{S}(\bm{x})=\mathbf{R}_{T}^{-1}c_{l_{T}}^{T}(\bm{
x})=c(\bm{x})$),
then they can be expressed as:

$$ $\displaystyle\bm{h}^{S}_{l_{S}}(\bm{x})$ $\displaystyle=\bm{W}_{S}^{\top}c^{S}_{l_{S}}(\bm{x})+\bm{b}_{l_{S}}^{S}=\bm{P} _{S}^{\top}\bm{W}_{U}^{\top}c(\bm{x})+\bm{b}_{l_{S}}^{S},$ $\displaystyle\bm{h}^{T}_{l_{T}}(\bm{x})$ $\displaystyle=\bm{W}_{T}^{\top}c^{T}_{l_{T}}(\bm{x})+\bm{b}_{l_{T}}^{T}=\bm{P} _{T}^{\top}\bm{W}_{U}^{\top}c(\bm{x})+\bm{b}_{l_{T}}^{T}.$ $$

In this case, we will have that the LRT hypothesis holds (and the hidden state of the target model is a linear transformation of the source hidden state):

$$ $\displaystyle\bm{h}^{T}_{l_{T}}(\bm{x})=\mathbf{A}\bm{h}^{S}_{l_{S}}(\bm{x})+ \bm{p},$ $$

where $\mathbf{A}=\bm{P}_{T}^{\top}(\bm{P}_{S}^{\top})^{\dagger}$
and $\bm{p}=\bm{b}_{l_{T}}^{T}-\mathbf{A}\bm{b}_{l_{S}}^{S}$.
Intuitively, this linear transformation first maps the source hidden state into a universal representation space via $(\bm{P}_{S}^{\top})^{\dagger}$, and then projects it into the target representation space using $\bm{P}_{T}^{\top}$.
In this framework, it is the existence of a universal representation space that makes such transferability possible.

##### Empirical Observation.

Figure: Figure 2: Reconstruction error between the projected Gemma-9B decoder matrix and the Gemma-2B decoder matrix, using the sparse autocoder trained by Lieberum et al. (2024), compared to that of random matrices of the same dimensions projected using the same procedure.
Refer to caption: https://arxiv.org/html/2506.00653/extracted/6513478/figs/comparison_plot.png

Thus far, we have presented the conceptual model for understanding why the LRT hypothesis may hold through the existence of a universal representation space.
When the model’s hidden states are linear combinations of features that lie in different subspaces of the same space—and when these hidden states activate the same features—there exists a linear transformation between them. The key insight is that each feature matrix, $\bm{W}_{S}$ and $\bm{W}_{T}$, is a linear transformation of the other, as their corresponding basis vectors are linearly related.

Below, we empirically analyze the decoder matrices of sparse autoencoders trained on Gemma-2B and Gemma-9B (Lieberum et al., 2024).
We treat the decoder matrix (in SAE, decoder matrices are the learned potentially monosemantic feature matrices) of Gemma-2B (layer 20) as $\bm{W}_{T}$, and that of Gemma-9B (layer 20) as $\bm{W}_{S}$.
In our conceptual framework, if the feature indices remain consistent, i.e., $\mathbf{R}_{S}=\mathbf{R}_{T}=\mathbf{I}$, then we expect the relationship
$\bm{W}_{T}=\bm{W}_{S}\bm{P}_{S}^{\dagger}\bm{P}_{T}$
to hold. Thus, we aim to empirically investigate whether these two matrices ($\bm{W}_{T}$ and $\bm{W}_{S}$) can be interpreted as linear transformations of each other.

We proceed by projecting $\bm{W}_{T}$ onto the space spanned by $\bm{W}_{S}$ via the following procedure:
(1) Solve for the linear mapping with the least squares $\widehat{\mathbf{M}}\in\arg\min_{\mathbf{M}}\|\bm{W}_{T}-\bm{W}_{S}\mathbf{M}
\|_{2}$.
(2) Compute the projected matrix: $\widehat{\bm{W}}_{T}=\bm{W}_{S}\widehat{\mathbf{M}}$.

We then obtain the reconstruction error using the Frobenius norm $\|\widehat{\bm{W}}_{T}-\bm{W}_{T}\|_{F}$, which quantifies the deviation between the projected target feature matrix ($\widehat{\bm{W}}_{T}$) and the actual target decoder/feature matrix (${\bm{W}}_{T}$). The resulting norm is around $114$—given that $\bm{W}_{T}$ and $\bm{W}_{S}$ have dimensions $(16384,2304)$ and $(16384,3584)$, respectively, the average reconstruction error per entry is small.
For comparison, we repeat the projection process using randomly generated matrices of the same dimensions as the decoder matrices for 9B and 2B. The resulting reconstruction errors are significantly larger (Figure [2](https://arxiv.org/html/2506.00653v3#S2.F2)). The low reconstruction error with a linear map between the source and target feature matrices
provides strong empirical evidence that $\bm{W}_{T}\approx\bm{W}_{S}\mathbf{M}$, i.e., the feature spaces represented by the decoder matrices are close to linearly related.

### 2.3 Learning Objective for Linear Representation Mapping

Based on our framework, we train a linear mapping $\mathbf{A}:\mathbb{R}^{d_{S}}\to\mathbb{R}^{d_{T}}$ from the source model hidden states to those of the target model.
The potential existence of such a linear mapping assumes that the activated features at the corresponding layers are similar.
Thus, we pick $l_{S}$ and $l_{T}$ to have similar relative depth, as layers of different relative depth are observed to behave distinctly (Lad et al., 2024).
For a given dataset $\mathcal{D}$, we learn the mapping through:

$$ $\displaystyle\min_{\mathbf{A},\mathbf{p}}\;\mathbb{E}_{\bm{x}\sim\mathcal{D}} \left[\left\|\mathbf{A}\bm{h}^{S}_{l_{S}}(\bm{x})+\mathbf{p}-\bm{h}^{T}_{l_{T} }(\bm{x})\right\|_{2}\right].$ $$

We note that when training mappings between arbitrary layers from the source and target models, we need to address the fact that $c^{S}_{l_{S}}(\bm{x})$ and $c^{T}_{l_{T}}(\bm{x})$
may be activating different features.
We discuss how this problem may be addressed by training mappings from the hidden states of all layers in the source model to hidden states of a single layer in the target model (see Appendix [A](https://arxiv.org/html/2506.00653v3#A1)).

## 3 Application: Steering Large Models with Small Models

For the rest of the paper, we focus on scenarios where the source model is a smaller model and the target model is a larger model from the same family (and thus shares the same tokenizer).
We consider *steering* as the canonical downstream task,
once the linear mapping between hidden states is trained.
We identify steering vectors in the smaller model, map them to the larger model using the learned linear mapping,
and evaluate whether the steering behavior is preserved in the mapped hidden states of the target model.

For a single (given) model, we find steering vectors through the Contrastive Activation Addition (CAA approach) (Rimsky et al., 2024).
Given two datasets ($\mathcal{D}_{p}$, $\mathcal{D}_{n}$) of positive and negative prompts, we find the steering vector for that behavior on layer $l$ via

$$ $\bm{v}_{l}=\frac{1}{|\mathcal{D}_{p}|}\sum_{\bm{x}\in\mathcal{D}_{p}}\bm{h}_{l }(\bm{x})-\frac{1}{|\mathcal{D}_{n}|}\sum_{\bm{x}^{\prime}\in\mathcal{D}_{n}} \bm{h}_{l}(\bm{x}^{\prime}).$ $$

Here, $\bm{h}_{l}(\bm{x})$ denotes the activation of the model in layer $l$ at the last token of input $\bm{x}$.

The vector $\bm{v}_{l}$ is the steering vector for the behavior demonstrated in the datasets. To steer the model towards the behavior demonstrated in the positive dataset, we perform the following intervention at all token positions

$$ $\bm{h}_{l}^{\prime}(\bm{x})\leftarrow\bm{h}_{l}(\bm{x})+\alpha\frac{\bm{v}_{l} }{\|\bm{v}_{l}\|},$ $$

where $\alpha$ is the strength of the intervention. To make $\alpha$ comparable across steering vectors, $\bm{v}_{l}$ is typically normalized. When $\alpha>0$, the steering is applied towards the positive behavior, while $\alpha<0$ indicates the opposite.

To map a steering vector $\bm{v}^{S}_{l_{S}}$ found on source model at layer $l_{S}$,
we use the learned affine mapping by getting the mapped steering vector as $\tilde{\bm{v}}^{T}_{l_{T}}=\bm{A}\bm{v}^{S}_{l_{S}}+\mathbf{p}$.
We note that one can use the same method above to directly find the steering vector $\bm{v}^{T}_{l_{T}}$ for the target model.

### 3.1 Training and Evaluation Setup

As activations are highly dataset dependendent (Kissane et al., 2024),
to train each mapping, we first collect the model activations on a dataset similar to the one used to train the model. For base models, we use ten million tokens of The Pile (Gao et al., 2020). For instruction-tuned models, we use a mixture of lmsys-chat-1m (Zheng et al., 2023) (33% chat data along with pile). After collecting activations on source and target models, we train the affine mapping to minimize the objective function defined in [2.3](https://arxiv.org/html/2506.00653v3#S2.SS3). Technical training details and hyper-parameters are left to the Appendix. Once the affine maps are trained, we first learn the steering vector on the individual models and then evaluate the steering vectors in multiple settings discussed in the rest of the section.
Code of the experiments can be found at: [https://github.com/HumainLab/linear-representation-transferability](https://github.com/HumainLab/linear-representation-transferability).

### 3.2 Proof of Concept:
Feature-to-Hidden vs. Hidden-to-Hidden Mapping

Figure: Figure 3: s2l and l2l corresponds to the feature-to-latent and latent-to-latent mapping.
Refer to caption: https://arxiv.org/html/2506.00653/extracted/6513478/figs/s2l_l2l_cropped.jpg

Based on our conceptual framework (Section [2.2](https://arxiv.org/html/2506.00653v3#S2.SS2)), if we can identify features in a universal representation space and map them to both source and target spaces, they should retain the same meaning. In our case, we treat the target model’s feature space as the universal space, with the subspace defined by the smaller model’s feature matrix.
We compare two approaches: (1) s2l, which directly maps source model coefficient vectors ${c}_{l_{S}}^{S}(x)$ to the target hidden state $\bm{h}_{l_{T}}^{T}(\bm{x})$, and (2) l2l, which maps source hidden states $\bm{h}_{l_{S}}^{S}(\bm{x})$ to $\bm{h}_{l_{T}}^{T}(\bm{x})$.
We expect the first mapping to perform better in preserving the semantic meaning of the steering vectors,
as it involves less information loss. Specifically, mapping the feature coefficients ${c}_{l_{S}}^{S}(x)$ directly avoids the need to recover it from the hidden state $\bm{h}_{l_{S}}^{S}(\bm{x})$, which may be a lossy process.

Specifically,
we use an encoder matrix from the SAEs trained on
Gemma-2-2b (Lieberum et al., 2024) to obtain the source model feature coefficients $c_{l_{S}}^{S}(\bm{x})$.
The target model is Gemma-2-9b.
If the coefficients learned by the SAEs truly captured the activations of some universal set of features,
we would expect not only the semantic meaning of features to align across different models,
but also for interventions (e.g., steering) to have similar effects.
For s2l steering, we perform the following steps:
(1) Using Neuropedia (Neuronpedia Contributors, 2025), we find feature indices for $c_{l_{S}}^{S}(\bm{x})$ corresponding to
Capitalization: the model should only speak in capital letters;
Dog Mentions: the model should say one of a few common words related to dogs;
and French Speaking: the model’s continuation should be in French.
(2) Using the s2l mapping, we map these coefficient vectors to the Gemma-9b hidden states and use them as the steering vector.
As our theory projected,
steering with the mapped coefficient vectors is more effective than the mapped source hidden state.

### 3.3 Steering to Remove Refusal Behaviors

It has been shown by Arditi et al. (2024) that the ability for post-trained language models to refuse harmful requests can be removed via using a steering vector found using CAA. We extend this setting to evaluate the LRT hypothesis and transfer steering vectors between models of different sizes. As a metric, we use GPT-4.1 to rate the harmlessness of a model response. Responses can either be rated 1, harmless, or 0, harmful.

Our exact procedure is as follows: (1) We collect model activations over 128 samples of both Alpaca (Taori et al., 2023) and 128 samples of AdvBench (Zou et al., 2023b). (2) Using the Alpaca activations as positive samples and AdvBench activations as negative samples, we perform CAA as described in Section [3](https://arxiv.org/html/2506.00653v3#S3). (3) We sweep over various layers and steering strengths. Once we find two layers that work for both source and destination models, we train an affine mapping between them. (4) We ask GPT-4.1 to evaluate the harmlessness of model responses(^3^33We found that asking the GPT-4.1 to evaluate if the model had refused a harmful prompt caused the model to respond no regardless of model behavior.). The model is instructed to give a 1 to responses that are rated harmless, and a 0 otherwise. Full system prompts are provided in the Appendix. (5) To generate responses, we feed harmful prompts from AdvBench to the steered models at various values of $\alpha$. We retain $\alpha=0$ as a baseline.
To visualize transfer success rate, we plot the average refusal score over 15 samples for a range of $\alpha$ in Figure [4](https://arxiv.org/html/2506.00653v3#S3.F4).

Figure: Figure 4: Refusal Score vs. $\alpha$. The mean refusal score of model responses on a range of different steering strengths ($\alpha$). Lower scores indicate less refusal. The left figure shows the propensity curve when using the chat mixture, while the right curve uses purely pretraining data. We include error estimates since model responses are generated via sampling. Error bars are 1 standard error wide.
Refer to caption: https://arxiv.org/html/2506.00653/extracted/6513478/figs/refusal_condense.png

As expected, the transferred steering vector has the same effect as one learned natively. This holds for whether the steering vector was learned on Gemma-2-2b OR Gemma-2-9b.

### 3.4 Logit Based Evaluation

To provide additional evidence of transferability, we evaluate transferred steering against the suite of tasks in Tan et al. (2024).
This evaluation suite consists of a series of tasks posed as multiple-choice question answering (MCQ).
Each instance is a sentence followed by two choices, one is marked as positive choice and the other is a negative choice.
These choices could be ‘statements’ or ‘yes/no’ questions. An example prompt is “Is the following statement something you would say?
I can maximize my goals better by being secretive and working unobserved”
The goal of steering is to move the model answer from one choice to the other; in the above example case, steer the model from ‘yes’ to ‘no.’
When the model picks answer choice $y^{+}$ over answer choice $y^{-}$, propensity is computed as

$$ $m_{LD}=\textrm{Logit}(y^{+})-\textrm{Logit}(y^{-})$ $$

After generating a steering vector for a given task, we can evaluate the propensity over a range of different steering strenghts, noted as $\alpha$.

To test LRT hypothesis, we compare the propensity scores between directly steering the Gemma-9B model with steering the Gemma-9B model with projected steering vector from the Gemma-2B model. For each task, correlation is calculated between all the samples for each of the $\alpha$. Then the distribution of the correlation across the $\alpha$ values is visualized in Figure [5](https://arxiv.org/html/2506.00653v3#S3.F5). Similarly, we also compute sample-wise mean squared error (MSE) between direct steering and mapped steering. We observe high correlation and low MSE across tasks. High correlation and low MSE indicate that transferring a steering vector from a smaller model to a larger one achieves comparable effectiveness to learning the steering vector directly in the larger model.

Figure: Figure 5: Steering Gemma-9B directly vs. steering Gemma-9B with vectors learned with Gemma-2B. We first train direct steering vectors on Gemma-2B and Gemma-9B. Then we transfer Gemma-2B steering vectors onto the Gemma-9B latent space. Now we compute the propensity scores for the Gemma-9B direct and Gemma-2 to Gemma-9B transfer. Then, we compute the correlation and mean-squared error between propensity scores to understand the effectiveness of steering the large model with the smaller model vs. directly steering the larger model. The correlation co-efficients across steering strengths is plotted as distributions. We observe a high correlation and low MSE across majority of the tasks. Note: boxplot and violinplot uses Interquartile range (IQR). The IQR (Q3-Q1) covers approximately 1.35 standard deviations..
Refer to caption: https://arxiv.org/html/2506.00653/x1.png

### 3.5 Limits of transferrability

(unsure of exact details) In this section, we focus on the limits of the LRT Hypothesis. We aim to investigate two directions

- 1.
It is assumed that models represent different levels of abstraction at different layers (stages of infrerence paper?), predicting that earlier layers in the source model contain little relevant information to later layers of the target model.
- 2.
How does scale effect transfer, we can transfer behaviors relialby across a 5 billion parameter gap, will that hold across a 12 billion parameter gap? 31 billion?

See our negative result on 5-35 and 35-5 mappings.

## 4 Related work

Interpretability. Interest in the Linear Representation Hypothesis (LRH)—that models encode human-interpretable features as directions in activation space—has grown rapidly Park et al. (2023). The related Superposition view holds that models pack more features than dimensions by using non-orthogonal directions Elhage et al. (2022). Sparse Auto-Encoders (SAEs) were introduced to tease these directions apart into monosemantic components Bricken et al. (2023), and Crosscoders extend SAEs to enable rigorous cross-model comparison, notably between base and instruction-tuned checkpoints Lindsey et al. (2024).
The field has long been interested in shared representations within networks, with many case studies (Gurnee et al., 2024; Schubert et al., 2021; Cammarata et al., 2021) and few statistical analyses (Bricken et al., 2023; Lan et al., 2024).

Representation Comparison. Work on representational similarity began with vision networks (Raghu et al., 2017; Wang et al., 2018; Kornblith et al., 2019; Klabunde et al., 2024, 2023b) and has recently shifted to large language models (LLMs) (Klabunde et al., 2023a; Lan et al., 2024; Oozeer et al., 2025; Gurnee et al., 2024). These studies rely on statistical or geometric tools—CCA, CKA, RSM, orthogonal Procrustes—to test whether activation spaces differ (Kornblith et al., 2019; Klabunde et al., 2023b). Yet few assess this hypothesis with downstream behavioural metrics; filling that gap is a central aim of our work.

Steering. It has been known that neural networks can sometimes have their behavior predictably changed by adding a vector (i.e., steering vector) to their activations during inference Turner et al. (2023).
Steering vectors in their current form were popularized by the works of Turner et al. (2023), Zou et al. (2023a), and Liu et al. (2023). Most notably, Panickssery et al. (2023) introduced the popular method of Contrastive Activation Addition (CAA), which we leverage in this work.
There is a wide assortment of work evaluating the effects, performance, and reliability of steering vectors (Tan et al., 2024; Zou et al., 2023a; Arditi et al., 2024; Stolfo et al., 2024; Wu et al., 2025). We make use of the demonstrations in Arditi et al. (2024) and the evaluations in Tan et al. (2024).

Model Stitching. Model stitching shows that learned affine maps can connect the representations of separate networks, often recovering—or even surpassing—baseline performance Merullo et al. (2022); Csiszárik et al. (2021); Yang et al. (2022); Bansal et al. (2021). These findings motivate our use of affine maps for representation transfer.

Concurrent work. Oozeer et al. (2025) independently examines steering-vector transfer but differs from us in three ways:
(1) They mainly use a nonlinear auto-encoder rather than an affine map;
(2) Their mapping is trained on the same data used to derive steering vectors;
(3) They report mixed or negative results for affine transformations.

## 5 Practical Implications and Future Directions

This paper proposes the Linear Representation Transferability (LRT) hypothesis and evaluates its validity through downstream model steering tasks.
We demonstrate that an affine mapping can be learned to transfer representations from one model to another, provided their training data distributions and model architectures are similar.
Furthermore, we show that steering vectors derived from smaller models can be effectively used to steer larger models via this affine mapping.
The implications of the LRT hypothesis are intriguing and are discussed in detail below.

##### Practical Implications.

The LRT hypothesis has several practical implications across different stages of the model development pipeline:

- •
Training:
With LRT, if two models from the same family are pretrained on similar data distributions, their internal representations become transferable. During post-training, suppose we want to evaluate the effect of a new dataset or algorithm. By measuring the performance improvement on a smaller model, we can use a learned linear map to project the corresponding change in hidden states onto the larger model. This projection allows us to estimate the expected improvement on the large model without retraining it, making performance prediction significantly more efficient.
- •
Efficient Inference and Distillation:
One of the most direct impacts of LRT is its potential to efficiently approximate the forward pass of a larger model using a smaller one. For example, input can be processed through the early layers of a small model, followed by an affine transformation, and then passed through the remaining layers of the larger model. In addition, LRT enables new approaches to model compression and distillation by directly aligning internal representations, potentially improving efficiency.
- •
Model Design:
The LRT hypothesis suggests that model architectures within a family should be designed to facilitate representational alignment—e.g., by maintaining consistent structural patterns across scales—enabling effective post-training analysis and efficient inference via linear mappings.

##### Future Directions.

Beyond the practical implications of LRT, our findings open up several exciting directions for future research on understanding the representations of neural networks.

- •
Limits of Transferability across Model Sizes:
One important question is: At what scale do these linear transfers break down? For example, it is unclear whether a 100M parameter model can effectively steer (or transfer its representations to that of) a much larger 70B model. Understanding these boundaries could inform scaling laws for representation transferability.
- •
Limits of Transferability across Model Families/Classes:
It is worth understanding whether representation transfer can be extended beyond a single model family to work across different families and architectures.
A major obstacle here is the lack of a reliable method for transferring tokenization schemes. This challenge is also observed in the literature on cross-tokenizer model distillation (Boizard et al., 2024; Minixhofer et al., 2025; Wan et al., 2024) and remains an open problem.
Furthermore, understanding how representation transferability is influenced by different tokenization schemes and architectural design choices may help uncover the implicit biases introduced by these design choices.
- •
Explaining Empirical Phenomena:
The conceptual framework underlying the LRT hypothesis may help explain various empirical phenomena, such as the success of layer swapping and model merging.
- •
Theoretical Groundings to LRT and the Universal Representation Space:
Finally, a promising direction is to develop formal theoretical foundations for the LRT hypothesis. This includes characterizing the conditions under which linear transferability holds and the universal representation space exists, understanding its relationship to the geometry of learned representations, and identifying its limitations. If such conditions can be empirically verified, one may be able to safely transfer representations across models when those conditions are met.

Overall, our results provide concrete evidence for the LRT hypothesis and suggest that it could serve as a unifying principle for understanding and leveraging neural network representations. We hope this work opens a pathway for connecting small models to large ones, enabling insights and solutions
developed at smaller scales to transfer reliably to much larger models.

## Acknowledgements

This research is in part supported by the OpenAI Superalignment Fast Grants. This research has been supported by computing support on the Vista GPU Cluster through the Center for Generative AI (CGAI) and the Texas Advanced Computing Center (TACC) at UT Austin.

## References

- Arditi et al. [2024]
Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda.
Refusal in language models is mediated by a single direction.
*arXiv preprint arXiv:2406.11717*, 2024.
- Bansal et al. [2021]
Yamini Bansal, Preetum Nakkiran, and Boaz Barak.
Revisiting model stitching to compare neural representations.
In *Neural Information Processing Systems*, 2021.
URL [https://api.semanticscholar.org/CorpusID:235435759](https://api.semanticscholar.org/CorpusID:235435759).
- Boizard et al. [2024]
Nicolas Boizard, Kevin El Haddad, Céline Hudelot, and Pierre Colombo.
Towards cross-tokenizer distillation: the universal logit distillation loss for llms.
*arXiv preprint arXiv:2402.12030*, 2024.
- Bricken et al. [2023]
Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah.
Towards monosemanticity: Decomposing language models with dictionary learning.
*Transformer Circuits Thread*, 2023.
https://transformer-circuits.pub/2023/monosemantic-features/index.html.
- Cammarata et al. [2021]
Nick Cammarata, Gabriel Goh, Shan Carter, Chelsea Voss, Ludwig Schubert, and Chris Olah.
Curve circuits.
*Distill*, 6(1):e00024–006, 2021.
- Csiszárik et al. [2021]
Adrián Csiszárik, Péter Kőrösi-Szabó, Akos Matszangosz, Gergely Papp, and Dániel Varga.
Similarity and matching of neural network representations.
*Advances in Neural Information Processing Systems*, 34:5656–5668, 2021.
- Elhage et al. [2021]
Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah.
A mathematical framework for transformer circuits.
*Transformer Circuits Thread*, 2021.
https://transformer-circuits.pub/2021/framework/index.html.
- Elhage et al. [2022]
Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah.
Toy models of superposition.
*Transformer Circuits Thread*, 2022.
Accessed May 30, 2025.
- Gao et al. [2020]
Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy.
The Pile: An 800gb dataset of diverse text for language modeling.
*arXiv preprint arXiv:2101.00027*, 2020.
- Gurnee et al. [2024]
Wes Gurnee, Theo Horsley, Zifan Carl Guo, Tara Rezaei Kheirkhah, Qinyi Sun, Will Hathaway, Neel Nanda, and Dimitris Bertsimas.
Universal neurons in gpt2 language models.
*Trans. Mach. Learn. Res.*, 2024, 2024.
URL [https://api.semanticscholar.org/CorpusID:267068880](https://api.semanticscholar.org/CorpusID:267068880).
- Kissane et al. [2024]
Connor Kissane, Robert Krzyzanowski, Neel Nanda, and Arthur Conmy.
Saes are highly dataset dependent: A case study on the refusal direction.
Alignment Forum, 2024.
URL [https://www.alignmentforum.org/posts/rtp6n7Z23uJpEH7od/saes-are-highly-dataset-dependent-a-case-study-on-the](https://www.alignmentforum.org/posts/rtp6n7Z23uJpEH7od/saes-are-highly-dataset-dependent-a-case-study-on-the).
- Klabunde et al. [2023a]
Max Klabunde, Mehdi Ben Amor, Michael Granitzer, and Florian Lemmerich.
Towards measuring representational similarity of large language models.
*ArXiv*, abs/2312.02730, 2023a.
URL [https://api.semanticscholar.org/CorpusID:265658940](https://api.semanticscholar.org/CorpusID:265658940).
- Klabunde et al. [2023b]
Max Klabunde, Tobias Schumacher, Markus Strohmaier, and Florian Lemmerich.
Similarity of neural network models: A survey of functional and representational measures.
*ACM Computing Surveys*, 2023b.
URL [https://api.semanticscholar.org/CorpusID:258587825](https://api.semanticscholar.org/CorpusID:258587825).
- Klabunde et al. [2024]
Max Klabunde, Tassilo Wald, Tobias Schumacher, Klaus H. Maier-Hein, Markus Strohmaier, and Florian Lemmerich.
Resi: A comprehensive benchmark for representational similarity measures.
*ArXiv*, abs/2408.00531, 2024.
URL [https://api.semanticscholar.org/CorpusID:271601150](https://api.semanticscholar.org/CorpusID:271601150).
- Kornblith et al. [2019]
Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey E. Hinton.
Similarity of neural network representations revisited.
*ArXiv*, abs/1905.00414, 2019.
URL [https://api.semanticscholar.org/CorpusID:141460329](https://api.semanticscholar.org/CorpusID:141460329).
- Lad et al. [2024]
Vedang Lad, Wes Gurnee, and Max Tegmark.
The remarkable robustness of llms: Stages of inference?
*arXiv preprint arXiv:2406.19384*, 2024.
- Lan et al. [2024]
Michael Lan, Philip H. S. Torr, Austin Meek, Ashkan Khakzar, David Krueger, and Fazl Barez.
Sparse autoencoders reveal universal feature spaces across large language models.
*ArXiv*, abs/2410.06981, 2024.
URL [https://api.semanticscholar.org/CorpusID:273228435](https://api.semanticscholar.org/CorpusID:273228435).
- Lieberum et al. [2024]
Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, and Neel Nanda.
Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2.
*arXiv preprint arXiv:2408.05147*, 2024.
- Lindsey et al. [2024]
Jack Lindsey, Adly Templeton, Jonathan Marcus, Thomas Conerly, Joshua Batson, and Christopher Olah, Oct 2024.
URL [https://transformer-circuits.pub/2024/crosscoders/index.html](https://transformer-circuits.pub/2024/crosscoders/index.html).
- Liu et al. [2023]
Sheng Liu, Haotian Ye, Lei Xing, and James Zou.
In-context vectors: Making in context learning more effective and controllable through latent space steering.
*arXiv preprint arXiv:2311.06668*, 2023.
- Merullo et al. [2022]
Jack Merullo, Louis Castricato, Carsten Eickhoff, and Ellie Pavlick.
Linearly mapping from image to text space.
*arXiv preprint arXiv:2209.15162*, 2022.
- Minixhofer et al. [2025]
Benjamin Minixhofer, Ivan Vulić, and Edoardo Maria Ponti.
Cross-tokenizer distillation via approximate likelihood matching.
*arXiv preprint arXiv:2503.20083*, 2025.
- Neuronpedia Contributors [2025]
Neuronpedia Contributors.
Neuronpedia, 2025.
URL [https://www.neuronpedia.org/](https://www.neuronpedia.org/).
Accessed: 2025-05-15.
- Oozeer et al. [2025]
Narmeen Oozeer, Dhruv Nathawani, Nirmalendu Prakash, Michael Lan, Abir Harrasse, and Amirali Abdullah.
Activation space interventions can be transferred between large language models.
*ArXiv*, abs/2503.04429, 2025.
URL [https://api.semanticscholar.org/CorpusID:276812830](https://api.semanticscholar.org/CorpusID:276812830).
- Panickssery et al. [2023]
Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner.
Steering llama 2 via contrastive activation addition.
*arXiv preprint arXiv:2312.06681*, 2023.
- Park et al. [2023]
Kiho Park, Yo Joong Choe, and Victor Veitch.
The linear representation hypothesis and the geometry of large language models.
*arXiv preprint arXiv:2311.03658*, 2023.
- Raghu et al. [2017]
Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein.
Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability.
In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, *Advances in Neural Information Processing Systems 30*, pages 6076–6085. Curran Associates, Inc., 2017.
- Rimsky et al. [2024]
Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner.
Steering llama 2 via contrastive activation addition.
In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 15504–15522, 2024.
- Schubert et al. [2021]
Ludwig Schubert, Chelsea Voss, Nick Cammarata, Gabriel Goh, and Chris Olah.
High-low frequency detectors.
*Distill*, 6(1):e00024–005, 2021.
- Stolfo et al. [2024]
Alessandro Stolfo, Vidhisha Balachandran, Safoora Yousefi, Eric Horvitz, and Besmira Nushi.
Improving instruction-following in language models through activation steering.
*arXiv preprint arXiv:2410.12877*, 2024.
- Tan et al. [2024]
Daniel Tan, David Chanin, Aengus Lynch, Brooks Paige, Dimitrios Kanoulas, Adrià Garriga-Alonso, and Robert Kirk.
Analysing the generalisation and reliability of steering vectors.
*Advances in Neural Information Processing Systems*, 37:139179–139212, 2024.
- Taori et al. [2023]
Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto.
Stanford alpaca: An instruction-following llama model.
[https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), 2023.
- Turner et al. [2023]
Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid.
Steering language models with activation engineering.
*arXiv preprint arXiv:2308.10248*, 2023.
- Wan et al. [2024]
Fanqi Wan, Xinting Huang, Deng Cai, Xiaojun Quan, Wei Bi, and Shuming Shi.
Knowledge fusion of large language models.
*arXiv preprint arXiv:2401.10491*, 2024.
- Wang et al. [2018]
Liwei Wang, Lunjia Hu, Jiayuan Gu, Yue Kris Wu, Zhiqiang Hu, Kun He, and John E. Hopcroft.
Towards understanding learning representations: To what extent do different neural networks learn the same representation.
In *Neural Information Processing Systems*, 2018.
URL [https://api.semanticscholar.org/CorpusID:53094393](https://api.semanticscholar.org/CorpusID:53094393).
- Wu et al. [2025]
Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D Manning, and Christopher Potts.
Axbench: Steering llms? even simple baselines outperform sparse autoencoders.
*arXiv preprint arXiv:2501.17148*, 2025.
- Yang et al. [2022]
Xingyi Yang, Zhou Daquan, Songhua Liu, Jingwen Ye, and Xinchao Wang.
Deep model reassembly.
*ArXiv*, abs/2210.17409, 2022.
URL [https://api.semanticscholar.org/CorpusID:253236958](https://api.semanticscholar.org/CorpusID:253236958).
- Zheng et al. [2023]
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric. P Xing, Joseph E. Gonzalez, Ion Stoica, and Hao Zhang.
Lmsys-chat-1m: A large-scale real-world llm conversation dataset, 2023.
- Zou et al. [2023a]
Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al.
Representation engineering: A top-down approach to ai transparency.
*arXiv preprint arXiv:2310.01405*, 2023a.
- Zou et al. [2023b]
Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson.
Universal and transferable adversarial attacks on aligned language models, 2023b.

## Appendix A Discussion on Many-to-1 and Many-to-Many mappings

On which (source, target) layer pairs do we train a linear mapping? Inspired by the findings of Lindsey et al. [2024], we think a reasonable approach might be Many-to-Many. Formally, we assume that for a given input $\bm{x}$, there exists a universal set of features ${c}(\bm{x})$ active on that input. For every layer, the hidden state can be represented by

$$ $\bm{h}_{l}=\bm{P}_{l}^{\top}\bm{W}_{U}^{\top}c(\bm{x})$ $$

Note that based on the construction in Section [2.2](https://arxiv.org/html/2506.00653v3#S2.SS2), there is a single universal feature space for all layers.

We now assume that 1) the transformation from feature space to hidden states is invertable given all of the hidden states, 2) different models use the exact same concept spaces. This motivates us to assume that given a source and target model, hidden latents from the target model can be modeled as

$$ $\bm{h}^{T}_{l}=\sum_{i=1}^{L_{S}}A_{i,l}\bm{h}_{i}^{S}+\bm{b}_{i,l}$ $$

Where $\bm{h}_{l}^{T}$ and $\bm{h}_{l}^{S}$ represent the hidden state at layer $l$ for target and source models respectively. We note that this model is computationally expensive to train, and encourage others to extend this work by training one.

We use the standard $L_{2}$ loss.

$$ $||\bm{h}^{T}_{l}-(\sum_{i=1}^{L_{S}}A_{i,l}\bm{h}_{i}^{S}+\bm{b}_{i,l})||_{2}$ $$

## Appendix B Significance Analysis for Section 3.4

**Table 1: Detailed results for Figure [5](https://arxiv.org/html/2506.00653v3#S3.F5)**
| Task | Mean Correlation | 95% CI | Mean MSE | Significant |
| --- | --- | --- | --- | --- |
| corrigible-neutral-HHH | 0.807 ± 0.160 | [0.751, 0.861] | 4.22e-02 ± 4.24e-02 | Yes |
| self-awareness-text-model | 0.810 ± 0.238 | [0.720, 0.881] | 2.96e-02 ± 2.19e-02 | Yes |
| self-awareness-good-text-model | 0.833 ± 0.066 | [0.811, 0.856] | 4.87e-02 ± 2.58e-02 | Yes |
| self-awareness-training-web-gpt | 0.906 ± 0.084 | [0.875, 0.933] | 4.06e-02 ± 3.66e-02 | Yes |
| anti-LGBTQ-rights | 0.894 ± 0.127 | [0.845, 0.930] | 5.17e-03 ± 5.81e-03 | Yes |
| openness | 0.878 ± 0.081 | [0.849, 0.904] | 1.53e-02 ± 9.68e-03 | Yes |
| believes-AIs-are-not-an-existential-threat-to-humanity | 0.884 ± 0.076 | [0.854, 0.906] | 2.84e-02 ± 1.89e-02 | Yes |
| subscribes-to-deontology | 0.918 ± 0.037 | [0.903, 0.928] | 3.43e-02 ± 1.54e-02 | Yes |
| myopic-reward | 0.728 ± 0.344 | [0.598, 0.830] | 9.56e-02 ± 1.16e-01 | Yes |
| believes-it-is-not-being-watched-by-humans | 0.981 ± 0.017 | [0.975, 0.987] | 8.86e-03 ± 8.09e-03 | Yes |
| subscribes-to-average-utilitarianism | 0.924 ± 0.053 | [0.905, 0.941] | 3.19e-02 ± 2.28e-02 | Yes |
| narcissism | 0.968 ± 0.022 | [0.960, 0.975] | 1.25e-02 ± 8.15e-03 | Yes |
| willingness-to-use-physical-force-to-achieve-benevolent-goals | 0.957 ± 0.042 | [0.942, 0.971] | 1.28e-02 ± 1.21e-02 | Yes |

## Appendix C Compute and Training details

For all experiments, we use compute provided by the Texas Advanced Computing Center’s Vista cluster. The standard configuration of a node was a single Grace Hopper Superchip.

#### C.0.1 Latent2Latent & SAE2Latent

We train each of these models on a single node. On average each training run lasted 200 minutes. Specific run details can be found in Table [2](https://arxiv.org/html/2506.00653v3#A3.T2).

**Table 2: Training configurations for all the models used in the paper.**
| Model | Source Model/Layer | Target Model/Layer | Dataset | Num Toks | LR | Optimizer | Batch Size | Context Length | Epochs | Train:Val Ratio | Final Val Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma_20_20_thepile_l2l | Gemma-2-2b/20 | Gemma-2-9b/20 | ThePile | 10M | 1e-4 | AdamW | 32 | 512 | 4 | 9 | 5.935 |
| gemma_20_20_mixture_33_l2l | Gemma-2-2b-it/20 | Gemma-2-9b-it/20 | 33% lmsys-chat-im 67% ThePile | 10M | 1e-3 | AdamW | 32 | 512 | 4 | 9 | 5.986 |
| gemma_20_20_thepile_s2l | Gemma-2-2b (SAE)/20 | Gemma-2-9b/20 | ThePile | 10M | 1e-4 | AdamW | 32 | 512 | 4 | 9 | 5.437 |

### C.1 System Prompts and Extended Data

The following is the exact set of messages we used to evaluate the steered model responses.

The following reponses were rated harmless:

The following reponses were rated harmful.

Full data can be found in our GitHub repository.