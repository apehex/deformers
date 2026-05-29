Title: Language Models are Injective and Hence Invertible
ArXiv: 2510.15511
Authors: Giorgos Nikolaou, Tommaso Mencattini, Donato Crisostomi, Andrea Santilli, Yannis Panagakis, Emanuele Rodolà, EPFL, Sapienza University of Rome, Paradigma, University of Athens, Archimedes/Athena RC, Greece, Corresponding authors:
Sections: 237
Estimated tokens: 84.1k

## Contents
- 1 Introduction
  - Our approach.
  - Main result.
  - Significance.
- 2 Transformers are injective
  - Summary.
  - Approach.
    - Theorem 2.1 (Transformers are real-analytic) .
    - Sketch of proof (full proof in Appendix B , Proposition B.3 ).
    - Theorem 2.2 (Almost-sure injectivity at initialization) .
    - Sketch of proof (full proof in Appendix C , Theorem C.2 ).
    - Theorem 2.3 (Injectivity preserved under training) .
    - Sketch of proof (full proof in Theorems C.1 and C.5 ).
    - Corollary 2.3.1 (SGD and mini-batch GD) .
    - Proof.
    - Corollary 2.3.2 (Distinctness for finite sets) .
    - Proof.
  - Failure cases.
- 3 Exact prompt recovery via SipIt
  - Threat model.
    - Theorem 3.1 (Correctness of SipIt ) .
    - Sketch of proof (full proof in Appendix D , Thm. D.2 , Prop. D.4 ).
    - Theorem 3.2 (Robustness of SipIt ) .
    - Proof in Appendix D , Thm. D.2 , Prop. D.2 .
- 4 Experiments
  - Environment.
  - 4.1 Searching for collisions
    - FP4 and INT8 weight quantization.
  - 4.2 Invertibility results
    - Robustness and vocabulary scaling.
    - Effect of layer depth.
- 5 Related work
  - Analytical properties of Transformers.
  - Inverse problems in language modeling.
- 6 Discussion and conclusions
- Reproducibility statement
- Acknowledgments
- References
- Appendix Overview
- Appendix A Preliminaries
  - A.1 Notation
    - Remark 1 .
    - Remark 2 .
    - Remark 3 .
  - A.2 Real-Analyticity
    - A.2.1 Real-Analytic Functions with Vector Inputs
      - Definition A.1 (Real-analytic functions, Lewis 2014 , Definition 1.1.3 ) .
      - Remark 4 .
      - Proposition A.1 (Closure properties, Lewis 2014 , Proposition 1.2.1 ) .
      - Proposition A.2 (Composition, Lewis 2014 , Proposition 1.2.2 ) .
      - Remark 5 .
      - Theorem A.1 (Zero sets of nontrivial real-analytic maps Mityagin 2015 ) .
      - Remark 6 .
    - A.2.2 Real-Analytic Functions with Matrix Inputs
      - Definition A.2 (Real-analyticity on matrix spaces) .
      - Remark 7 .
      - Definition A.3 (Vectorization and matricization Operators) .
      - Definition A.4 (Vectorized Form of Function) .
      - Lemma A.1 (Equivalence real-analyticity) .
      - Proof.
      - Remark 8 .
      - Proposition A.3 (Composition on matrix spaces is real-analytic) .
      - Proof.
    - A.2.3 Real Analyticity of Common Components
      - Proposition A.4 (Polynomials are real-analytic) .
      - Proof.
      - Proposition A.5 (The exponential is real-analytic) .
      - Proof.
      - Proposition A.6 (The logarithm is real-analytic) .
      - Proof.
      - Proposition A.7 (Softmax is real-analytic) .
      - Proof.
      - Proposition A.8 (Row normalization is real-analytic on positive row-sum domain) .
      - Proof.
      - Proposition A.9 (Entrywise matrix polynomials are real-analytic) .
      - Proof.
      - Proposition A.10 (Matrix product of real-analytic factors) .
      - Proof.
      - Proposition A.11 (Hadamard (element-wise) scaling) .
      - Proof.
      - Proposition A.12 (Concatenation/stacking of real-analytic blocks) .
      - Proof.
      - Proposition A.13 (Noncommutative matrix polynomials are real-analytic) .
      - Proof.
      - Remark 9 .
  - A.3 Differential, Measure-Theoretic, and Topological Tools
    - Definition A.5 (Fréchet derivative (Luenberger, 1997 , §7.2-§7.3) ) .
    - Definition A.6 (Second Fréchet derivative (Magnus and Neudecker, 2019 , Ch. 18) ) .
    - Proposition A.14 (Connection to the Hessian) .
    - Euclidean topology.
      - Definition A.7 (Closure of a set in ℝ p \mathbb{R}^{p} ) .
      - Definition A.8 (Euclidean balls in ℝ p \mathbb{R}^{p} ) .
      - Definition A.9 (Second-countable subspace of ℝ p \mathbb{R}^{p} (Munkres, 2000 , §30) ) .
      - Proposition A.15 (Standard facts for ℝ p \mathbb{R}^{p} ) .
    - Invertibility and measure transport.
      - Definition A.10 ( C k C^{k} diffeomorphism Spivak 1971 , Ch. 5 ) .
      - Theorem A.2 (Inverse Function Theorem Rudin 1976 , Thm. 9.24 ) .
      - Remark 10 .
      - Definition A.11 (Pushforward and absolute continuity (Folland, 1999 , §3.2) ) .
- Appendix B Transformer Language Model
  - Input processing.
    - Definition B.1 (Token Embedding Layer) .
    - Definition B.2 (Positional Embedding Layer) .
    - Definition B.3 (Embedding Layer) .
  - Sub-layer modules.
    - Definition B.4 (Multi-Layer Perceptron) .
    - Definition B.5 (Self-Attention) .
    - Definition B.6 (Causal Self-Attention, masked form) .
    - Definition B.7 (Causal Self-Attention, projection form) .
    - Remark 11 .
    - Definition B.8 (Multi-Head Self-Attention) .
    - Definition B.9 (Layer Normalization) .
    - Definition B.10 (Unembedding Layer) .
  - Full architecture assembly.
    - Definition B.11 (Transformer Block) .
    - Definition B.12 (Transformer) .
    - Definition B.13 (Transformer Language Model) .
  - Verification of real-analyticity.
    - Proposition B.1 (Equivalence of masked and projection causal softmax) .
    - Proof.
    - Proposition B.2 (Embedding layer is real-analytic in the parameters) .
    - Proof.
    - Proposition B.3 (Joint real-analyticity of core modules and stacks) .
- Appendix C Almost Sure Injectivity
  - Assumption C.1 (Minimum Embedding Dimension) .
  - Theorem C.1 (Finite-horizon a.s. injectivity under GD) .
  - Proof.
  - C.1 Absolute continuity ensures almost sure injectivity
    - Theorem C.2 (Almost-sure pairwise distinctness of last-token representations) .
    - Proof.
    - Remark 12 (Causal Self-Attention) .
    - Corollary C.2.1 (Almost-sure global distinctness over a finite input family) .
    - Proof.
    - Remark 13 (Pointwise vs. last-token injectivity) .
  - C.2 Absolute continuity of the parameter distribution is preserved under GD
    - Step 1: Regularity of the GD map.
    - Step 2: Witness and measure-zero critical set.
    - Step 3: Local-to-global via countable chart covers.
    - Step 4: Preservation of absolute continuity and conclusion.
    - C.2.1 Witness Construction
      - Lemma C.1 (Zero-gate through scalar loss) .
      - Proof.
      - Lemma C.2 (Spectrum under block-diagonal extension) .
      - Proof.
      - Remark 14 .
      - Lemma C.3 (Hessian of ℒ \mathcal{L} w.r.t. 𝐔 , 𝜷 \mathbf{U},\bm{\beta} at 𝜽 ⋆ = 𝟎 \bm{\theta}_{\star}=\mathbf{0} and its spectrum) .
      - Proof.
      - Lemma C.4 (Full Hessian at the witness: block form and spectrum) .
      - Proof.
      - Theorem C.3 (GD Jacobian is nondegenerate a.e.) .
      - Proof.
    - C.2.2 Gradient Descent preserves absolute continuity
      - Lemma C.5 (Countable chart cover of ℝ p ∖ 𝒞 \mathbb{R}^{p}\setminus\mathcal{C} ) .
      - Proof.
      - Theorem C.4 (Change of Variables Folland 1999 , Thm. 2.47(b) ) .
      - Lemma C.6 (Pre-images of null sets are null) .
      - Proof.
      - Theorem C.5 (Preservation of absolute continuity under one GD step) .
      - Proof.
      - Corollary C.5.1 (Preservation of absolute continuity under finitely many GD steps) .
      - Proof.
- Appendix D Left-Invertibility Via SipIt
  - Goal.
  - Main idea.
  - Algorithmic consequence.
  - Standing conventions for this section.
    - Assumption D.1 (Causal self-attention throughout) .
    - Assumption D.2 (Injectivity Assumption) .
  - D.1 One-Step Last-Token Maps
    - Definition D.1 (One-step map at time t t under prefix π \pi ) .
    - Remark 15 .
    - Theorem D.1 (A.s. one-step injectivity) .
    - Proof.
    - Lemma D.1 (Strict separation margin a.s.) .
    - Proof.
  - D.2 The Core Routines: Local Verifiers, Acceptance Regions, and Policies
    - Definition D.2 (Local verifier and acceptance tolerance) .
    - Remark 16 (Decoding via acceptance regions) .
    - Proposition D.1 (Probabilistic soundness and uniqueness of the local verifier) .
    - Proof.
    - Candidate enumeration.
      - Definition D.3 (Policy algorithm) .
      - Remark 17 (Enumeration property) .
    - Two examples of policy algorithms.
      - Remark 18 (Bypassing the embedding layer) .
      - Remark 19 (Practical choice of policy) .
  - D.3 Global Inversion via SipIt
    - Lemma D.2 (Causal factorization and prefixwise identifiability) .
    - Proof.
    - Proposition D.2 (The verifier is the right primitive) .
    - Proof.
    - Proposition D.3 (Eventual acceptance under increasing enumeration) .
    - Proof.
    - Enumeration without replacement.
    - Eventual acceptance.
      - Theorem D.2 (Correctness of SipIt (noiseless & robust)) .
      - Proof.
    - Termination and complexity.
      - Proposition D.4 (Termination and linear step bound) .
      - Proof.
      - Remark 20 (Iterations vs. wall–clock time) .
      - Remark 21 (Choosing the tolerance ε \varepsilon ) .
      - Remark 22 (Why SipIt is sequential) .
- Appendix E Implementation Details and Additional Experiments
  - E.1 Implementation Details
    - What is a collision in practice.
    - SipIt implementation.
    - HardPrompts implementation.
  - E.2 Additional Ablations
    - E.2.1 Collision Experiments
    - E.2.2 SipIt
      - Vocabulary Size.
      - Robustness of SipIt on unseen and random sequences.
  - E.3 Identical Next-Token
  - E.4 Prompts with Similar Representations
    - Llama-3.1-8B.
    - Mistral-7B-v0.1.
    - Summary.
  - E.5 Relation with Anisotropy and Intrinsic Dimension
    - Experimental setup.
    - Experiment 1: anisotropy vs. injectivity margin.
    - Experiment 2: intrinsic dimension vs. injectivity margin.
    - Discussion.
- Appendix F Real-Analytic Activation Functions in Modern LLMs
  - Proposition F.1 (Logistic sigmoid is real-analytic) .
  - Proof.
  - Proposition F.2 (SiLU / Swish is real-analytic) .
  - Proof.
  - Proposition F.3 (Error function is real-analytic) .
  - Proof.
  - Proposition F.4 (GELU is real-analytic) .
  - Proof.
  - Proposition F.5 (Vector-valued SiLU and GELU are real-analytic) .
  - Proof.
  - Proposition F.6 (GLU-style blocks are real-analytic) .
  - Proof.
  - Relation to universal-approximation and expressivity results.

## Abstract

Abstract Transformer components such as non-linear activations and normalization are inherently non-injective, suggesting that different inputs could map to the same output and prevent exact recovery of the input from a model’s representations. In this paper, we challenge this view. First, we prove mathematically that transformer language models mapping discrete input sequences to their corresponding sequence of continuous representations are injective and therefore lossless, a property established at initialization and preserved during training. Second, we confirm this result empirically through billions of collision tests on six state-of-the-art language models, and observe no collisions. Third, we operationalize injectivity: we introduce SipIt , the first algorithm that provably and efficiently reconstructs the exact input text from hidden activations, establishing linear-time guarantees and demonstrating exact invertibility in practice. Overall, our work establishes injectivity as a fundamental and exploitable property of language models, with direct implications for transparency, interpretability, and safe deployment.

## 1 Introduction

Figure: Figure 1: The map from prompts to latent space is injective. SipIt inverts it.

A core question in understanding large language models is whether their internal representations faithfully preserve the information in their inputs. Since Transformer architectures rely heavily on non-linearities, normalization, and many-to-one attention mechanisms, it is often assumed that they discard information: different inputs could collapse to the same hidden state, making exact recovery of the input impossible. This view motivates concerns around transparency, robustness, and safe deployment, as it suggests that the link between text and representation is inherently lossy.

In this paper, we show that this intuition is misleading. Despite their apparent complexity, standard decoder-only Transformer language models (seen as maps from prompts to hidden states) are in fact almost-surely injective; for essentially all parameter settings and during the course of training, different prompts yield different last-token representations (e.g., see [Figure 1](#S1.F1)).

Building upon this property, we further provide a practical algorithm, SipIt, that reconstructs the *exact* input from hidden activations. To our knowledge, it is the first to guarantee exact recovery in provable linear time (worst case bound), often faster in practice, turning injectivity from a theoretical property into an operational tool.

##### Our approach.

To establish our result, we take a rigorous mathematical view of Transformers as functions. The key idea is that their components (embeddings, LayerNorm, causal attention, MLPs, and residual wiring) are smooth and structured enough that the model, as a whole, behaves predictably with respect to its parameters. Using tools from real analysis, we show that collisions (two different prompts producing the exact same representation) can only occur on a set of parameter values that has measure zero; that is, they are mathematical *exceptions* rather than possibilities one should expect in practice. Moreover, we prove that common training procedures (gradient descent with standard step sizes) never move parameters into this exceptional set. In layman’s terms, almost all models at initialization are injective, and training preserves this property.

Technically, our proofs rely on two ingredients. First, we establish that Transformers are real-analytic functions of their parameters, which allows us to reason precisely about when and where collisions could occur. Second, we construct parameter settings where no two prompts collide, and show that gradient descent (GD) does not collapse such separation, i.e., collisions remain a measure-zero event. The end result is a finite-horizon *guarantee*: after any fixed number of training steps, and under mild assumptions, injectivity holds with probability one. We provide complete formal proofs of these statements.

##### Main result.

Our central finding is that causal decoder-only Transformer language models are injective almost surely. Formally, consider one such model with embedding width $d$, at least one attention head per block, real-analytic components, finite vocabulary $\mathcal{V}$, and finite context length $K$. Initialize its parameters $\bm{\theta}$ at random, using any distribution that has a density(^1^11Put simply, parameters are not drawn from a degenerate or hand-crafted set.) (such as Gaussian, uniform, or Xavier/Glorot), and train for any finite number $T$ of GD steps with step sizes in $(0,1)$. Then, with probability one over the random initialization,

$$ $\mathrm{s}\neq\mathrm{s}^{\prime}\quad\Longrightarrow\quad\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T})\neq\mathbf{r}(\mathrm{s}^{\prime}\,;\,\bm{\theta}_{T})\,,$ $$

i.e., the map from prompts $\mathrm{s}$ to *last-token* representations $\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T})$ is injective across all prompts in $\mathcal{V}^{\leq K}$.
In short, collisions in practical settings form a measure-zero set, and neither initialization nor training will ever place a model inside that set.

##### Significance.

Our result shows that in standard decoder-only Transformers, different prompts almost surely yield different last-token representations across all practically relevant parameter settings and training procedures. The guarantee is both *generic* (it fails only on a measure-zero set of pathological parameters) and *practical* (it holds at finite width, depth, and training time under common initializations).

Conceptually, we replace a long-assumed property with a rigorous theorem, showing that injectivity is not an asymptotic idealization but a structural consequence of the architecture itself. Technically, our analytic framework pinpoints when collisions can arise (through deliberate non-analytic choices such as quantization or tying), and clarifies that otherwise the model is inherently lossless. Importantly, it establishes that last-token states almost everywhere *identify* the input.

Finally, we turn this theoretical guarantee into an operational tool: our algorithm SipIt uses gradient-based reconstruction to recover prompts exactly from internal activations, efficiently and with provable *linear-time* guarantees. This confirms empirically that collisions do not occur in practice. Beyond transparency and safety, this elevates *invertibility* to a first-class property of Transformer language models, enabling stronger interpretability, probing, and causal analyses.

## 2 Transformers are injective

##### Summary.

In this section we show that decoder-only Transformers almost surely map different prompts to different hidden states. Collisions can only occur under measure-zero parameter choices, and gradient-based training never creates them. In simple terms, Transformer representations are structurally lossless.

##### Approach.

We consider causal decoder-only Transformer language models with vocabulary $\mathcal{V}$, finite context window $K$, and embedding dimension $d$. For an input sequence $\mathrm{s}\in\mathcal{V}^{\leq K}$, let $\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})$ denote the final hidden representation at the *last* token position(^2^22We focus on the last-token state, since it alone drives next-token prediction; earlier rows matter only insofar as they shape this final state. Injectivity at the last token is the property of real operational interest.), given parameters $\bm{\theta}$.

Our analysis relies on three facts:

- (i)
*Real-analyticity.* Each component of the architecture (embeddings, positional encodings, LayerNorm with ${\varepsilon>0}$, causal attention, MLPs with analytic activations, residuals) is real-analytic in its parameters (see Appendix [A.2](#A1.SS2) for the mathematical background). This smoothness implies that the set of parameter values causing two distinct prompts to collide is extremely thin (measure zero).
- (ii)
*Initialization.* Standard initialization schemes (Gaussian, uniform, Xavier/Glorot, etc.) draw parameters from continuous distributions with densities, so they avoid measure-zero sets with probability one.
- (iii)
*Training.* Gradient-based updates (including SGD and mini-batch/full-batch GD) preserve absolute continuity of the parameter distribution after any finite number of steps; thus, training cannot generate collisions.

These facts allow us to state and prove injectivity results without relying on asymptotics.
We begin by establishing the analytic structure of the architecture.

###### Theorem 2.1 (Transformers are real-analytic) .

Fix embedding dimension $d$ and context length $K$. Assume the MLP activation is real-analytic (e.g. tanh, GELU). Then for every input sequence $\mathrm{s}\in\mathcal{V}^{\leq K}$, the map

$$ $\displaystyle(\mathrm{s},\bm{\theta})\mapsto\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})\in\mathbb{R}^{d}$ (1) $$

is real-analytic jointly in the parameters $\bm{\theta}$ and the input embeddings.

###### Sketch of proof (full proof in Appendix B , Proposition B.3 ).

Each building block is real-analytic: polynomials (embeddings, projections), exponential and softmax (attention), reciprocal square root (LayerNorm with $\varepsilon>0$), analytic activations in the MLP, and affine maps. Real-analytic functions are closed under addition, multiplication, quotient, and composition. Since the Transformer is a finite composition of such blocks, the entire map is real-analytic.
∎

Figure: Figure 2: Two real-analytic functions $f_{1}$ and $f_{2}$ and their difference $f_{1}-f_{2}$. Black contours show the zero sets, which form thin curves (measure zero) rather than regions of positive measure.

This smoothness result drives everything that follows: it ensures that collisions, if they exist, are confined to measure-zero parameter sets. We now ask: what happens at initialization?

###### Theorem 2.2 (Almost-sure injectivity at initialization) .

Let $\bm{\theta}$ be drawn from any distribution with a density (e.g. Gaussian or uniform). Then for any two distinct prompts $\mathrm{s},\mathrm{s}^{\prime}\in\mathcal{V}^{\leq K}$,

$$ $\displaystyle\Pr[\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})=\mathbf{r}(\mathrm{s}^{\prime}\,;\,\bm{\theta})]=0\,.$ (2) $$

###### Sketch of proof (full proof in Appendix C , Theorem C.2 ).

Fix $\mathrm{s}\neq\mathrm{s}^{\prime}$ and consider

$$ $\displaystyle h(\bm{\theta})=\|\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})-\mathbf{r}(\mathrm{s}^{\prime}\,;\,\bm{\theta})\|_{2}^{2}\,.$ (3) $$

By Theorem [2.1](#S2.Thmtheorem1), $h$ is real-analytic. A fundamental dichotomy of real-analytic functions states that either $h$ is identically zero, or its zero set has Lebesgue measure zero (see Figure [2](#S2.F2) for an illustration). Therefore, to rule out the pathological case $h\equiv 0$ it suffices to exhibit a single parameter setting where $\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})\neq\mathbf{r}(\mathrm{s}^{\prime}\,;\,\bm{\theta})$.

This can always be done: if $\mathrm{s}$ and $\mathrm{s}^{\prime}$ differ at the last position (symbol or length), freeze the network so that the last state reduces to embedding plus position, and choose distinct rows; this already separates $\mathbf{r}(\mathrm{s})$ and $\mathbf{r}(\mathrm{s}^{\prime})$.
If instead they differ earlier, let $i^{\star}$ be the first mismatch and set one attention head so the last position attends almost entirely to $i^{\star}$, encoding its token in the value; this forces different outputs for $\mathrm{s}$ and $\mathrm{s}^{\prime}$.

Hence $h$ is not identically zero, and so the collision set $\{\bm{\theta}:h(\bm{\theta})=0\}$ has Lebesgue measure zero. Since standard initializations have densities, the probability of sampling such $\bm{\theta}$ is zero, and $\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})\neq\mathbf{r}(\mathrm{s}^{\prime}\,;\,\bm{\theta})$ (injectivity) holds almost surely at initialization.
∎

According to Theorem [2.2](#S2.Thmtheorem2), at initialization, collisions are mathematically impossible except on a vanishingly small set of parameter values. Finally, with the following Theorem we ensure training does not break injectivity.

###### Theorem 2.3 (Injectivity preserved under training) .

Let $\bm{\theta}_{0}$ be initialized from a distribution with a density, and let $\bm{\theta}_{T}$ be the parameters after $T$ steps of gradient descent with step sizes in $(0,1)$. Then with probability one,

$$ $\displaystyle\mathrm{s}\neq\mathrm{s}^{\prime}\quad\Longrightarrow\quad\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T})\neq\mathbf{r}(\mathrm{s}^{\prime}\,;\,\bm{\theta}_{T})\,,$ (4) $$

###### Sketch of proof (full proof in Theorems C.1 and C.5 ).

At initialization, $\bm{\theta}_{0}$ is drawn from a distribution with a density, hence absolutely continuous. To break injectivity during training, GD would need to map this continuous law onto the measure-zero collision set identified in Theorem [2.2](#S2.Thmtheorem2). We show this cannot happen.

A single GD step is the map $\phi(\bm{\theta})=\bm{\theta}-\eta\nabla\mathcal{L}(\bm{\theta})$, where $\mathcal{L}$ is the training loss. Because the network and the softmax cross-entropy loss are real-analytic, $\phi$ is also real-analytic.
Its Jacobian determinant $\det D\phi(\bm{\theta})$ is itself real-analytic and not identically zero (one can check this by evaluating at a simple parameter setting). Hence the set where $\det D\phi=0$ has measure zero.
Away from that set, the Inverse Function Theorem applies: $\phi$ is a smooth, locally invertible change of coordinates that can stretch or bend space but cannot collapse regions of positive volume onto lower-dimensional sets. Therefore, pushing forward an absolutely continuous distribution through $\phi$ yields another absolutely continuous distribution.

Since this argument holds for each step, any finite sequence of GD updates preserves absolute continuity of the parameter law. Combining with Theorem [2.2](#S2.Thmtheorem2), which shows that collision sets are measure-zero, we conclude that $\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T})\neq\mathbf{r}(\mathrm{s}^{\prime}\,;\,\bm{\theta}_{T})$ almost surely for all $\mathrm{s}\neq\mathrm{s}^{\prime}$.
∎

Thus injectivity is not just an initialization property but remains true throughout training. A simple but important corollary follows.

###### Corollary 2.3.1 (SGD and mini-batch GD) .

Under the assumptions of Theorem [2.3](#S2.Thmtheorem3), the same conclusion holds when the updates are
$\bm{\theta}_{t+1}=\bm{\theta}_{t}-\eta_{t}\,\nabla_{\theta}\mathcal{L}_{\mathcal{B}_{t}}(\bm{\theta}_{t})$
with arbitrary (possibly random or adversarial) batch selections $\mathcal{B}_{t}$, thus including the singleton case of SGD and the full dataset.

###### Proof.

The proof argument of Theorem [2.3](#S2.Thmtheorem3) is unchanged: for each fixed batch $\mathcal{B}$, the update map $\phi_{\mathcal{B}}(\bm{\theta})=\bm{\theta}-\eta\nabla\mathcal{L}_{\mathcal{B}}(\bm{\theta})$ is real-analytic with a Jacobian that is not identically zero. Indeed, the batch loss is the average $\mathcal{L}_{\mathcal{B}}=\tfrac{1}{|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\mathcal{L}_{i}$, so at the point $\bm{\theta}_{\star}$ from the single-sample proof (where the Jacobian determinant is sample-independent and nonzero) the batch Jacobian coincides with the single-sample one by linearity of differentiation, and its determinant is therefore also nonzero. Thus, the finite composition of such maps preserves absolute continuity of the parameter law.
∎

Together with this robustness to different training regimes, we can also strengthen the guarantee itself: injectivity holds not just pairwise, but globally across finite sets of prompts.

###### Corollary 2.3.2 (Distinctness for finite sets) .

For any finite set of prompts $\mathcal{S}\subseteq\mathcal{V}^{\leq K}$, the representations
$\{\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T}):\mathrm{s}\in\mathcal{S}\}$ are almost surely all distinct.

###### Proof.

See Appendix [C](#A3), Corollary [C.2.1](#A3.Thmtheorem2.Thmcorollary1).
∎

These results show that decoder-only Transformer language models are structurally injective: different prompts almost surely yield different last-token states. Collisions can be manufactured, e.g., through deliberate non-analytic choices (quantization, non-smooth activations), but in practical training pipelines, injectivity is guaranteed; extensive experiments in §[4.1](#S4.SS1) confirm this empirically.

###### Theorem 2.1 (Transformers are real-analytic) .

###### Sketch of proof (full proof in Appendix B , Proposition B.3 ).

###### Theorem 2.2 (Almost-sure injectivity at initialization) .

###### Sketch of proof (full proof in Appendix C , Theorem C.2 ).

###### Theorem 2.3 (Injectivity preserved under training) .

###### Sketch of proof (full proof in Theorems C.1 and C.5 ).

###### Corollary 2.3.1 (SGD and mini-batch GD) .

###### Proof.

###### Corollary 2.3.2 (Distinctness for finite sets) .

###### Proof.

##### Failure cases.

We showed that non-injective transformers are overwhelmingly unlikely, though it is still possible for an adversary to construct collisions by hand. For instance, if two vocabulary items $v_{i}\neq v_{j}$ are assigned *exactly* the same embedding vector, then any prompts differing only by swapping $v_{i}$ and $v_{j}$ yield identical representations. Likewise, if two absolute positional embeddings are made exactly equal and the remaining weights are tuned to suppress other positional signals, one can force collisions between sequences that differ only at those positions. These scenarios, however, require deliberately engineered parameter choices: under continuous random initialization and standard training, the probability of such coincidences is zero.

Figure: Figure 3: Seeking collisions in a large-scale prompt set: The minimum distances between last-token states are far above the collision threshold $10^{-6}$: (left) across layers for GPT-2 and Gemma-3 families (one dot per layer), (right) across depth for GPT-2 Small; distances grow with depth.
Refer to caption: https://arxiv.org/html/2510.15511/2510.15511v4/x1.png

## 3 Exact prompt recovery via SipIt

In the previous section, we have proven that decoder-only Transformers are almost surely injective, i.e., different prompts map to different hidden states.
We now show how this property can be used in practice to reconstruct the exact input prompt given hidden states at some layer. We call this algorithm SipIt (Sequential Inverse Prompt via ITerative updates).(^3^33Implementation available at [https://github.com/giorgosnikolaou/SIPIT](https://github.com/giorgosnikolaou/SIPIT).)

##### Threat model.

This paper focuses on the injectivity result and its algorithmic consequence; we do not define a full adversarial model. A natural setting where SipIt applies is one in which an adversary obtains the hidden-state sequence—for instance, through a leaked KV-cache, a shared-inference pipeline, or an API that exposes intermediate representations. Our injectivity result guarantees that exact recovery from only the final embedding is possible in principle, but designing an efficient algorithm for that setting is nontrivial and left to future work; here we assume access to all per-position states at a given layer $\ell$.

Recall from §[2](#S2) that the mapping from a prompt $\mathrm{s}$ to its last-token state is almost surely injective. Since the last state is itself a deterministic function of the hidden matrix at any layer $\ell$, injectivity extends to the full representation

$$ $\displaystyle\mathrm{s}\mapsto\mathbf{H}^{(\ell)}(\mathrm{s})\in\mathbb{R}^{T\times d}\,.$ (5) $$

We denote by $\mathbf{h}_{t}(\mathrm{s})$ the row of $\mathbf{H}^{(\ell)}(\mathrm{s})$ at position $t$. In the following, the parameters $\bm{\theta}$ and target layer $\ell$ are considered fixed and omitted for simplicity.

The algorithm exploits the causal structure of Transformers: the hidden state at position $t$ depends only on the prefix $\langle\mathrm{s}_{1},\dots,\mathrm{s}_{t-1}\rangle$ and the current token $\mathrm{s}_{t}$. This means that if we already know the prefix, then the hidden state at position $t$ uniquely identifies $\mathrm{s}_{t}$.

Example. Suppose the vocabulary is ${a,b,c}$ and the true prompt is $\langle a,b\rangle$. At $t=1$, the hidden state depends only on $\mathrm{s}_{1}$. By comparing the observed state with the three candidate states produced by trying $a$, $b$, and $c$, we can tell exactly which one matches, thus recovering $\mathrm{s}_{1}=a$. Then at $t=2$, we know the prefix $\langle a\rangle$, so we try appending each candidate token and again match the resulting hidden state to recover $\mathrm{s}_{2}=b$. Iterating this procedure reconstructs the full sequence.

More generally, we can look at the “one-step” map

$$ $\displaystyle v_{j}\mapsto\mathbf{h}_{t}(\pi\oplus v_{j})\,,\quad v_{j}\in\mathcal{V}\,,$ (6) $$

which gives the hidden state at step $t$ for each possible next token, given the fixed prefix $\pi=\langle\mathrm{s}_{1},\dots,\mathrm{s}_{t-1}\rangle$ (here $\oplus$ denotes concatenation).

Remark. By the analytic arguments of §[2](#S2), the one-step map is almost surely injective: with a fixed prefix, any two distinct tokens almost surely yield distinct hidden states.

This property makes sequence recovery straightforward. At each step $t$, given the hidden state $\widehat{\mathbf{h}}_{t}$ and the already recovered prefix, we simply check which candidate token produces a matching hidden state. That token must be the true $\mathrm{s}_{t}$. Repeating this process recovers the entire sequence.

This leads to the SipIt algorithm, shown in Algorithm [1](#alg1). At every position, the algorithm cycles through vocabulary candidates (according to some policy such as random order or gradient-guided search) until it finds the unique match(^4^44In practice, we accept matches if the observed hidden state is within an $\varepsilon$-ball around the predicted one.), then appends it to the reconstructed prefix and moves on.

Figure: Algorithm 1 SipIt: Sequential Inverse Prompt via ITerative updates

To rule out edge cases and analyze the computational cost of SipIt, we now state a formal guarantee.

###### Theorem 3.1 (Correctness of SipIt ) .

Under the assumptions of [Theorem 2.3](#S2.Thmtheorem3), given observed hidden states $\widehat{\mathbf{H}}^{(\ell)}$, SipIt recovers the true input sequence $\mathrm{s}$ with probability one in at most $T|\mathcal{V}|$ steps.

###### Sketch of proof (full proof in Appendix D , Thm. D.2 , Prop. D.4 ).

At each step, local injectivity ensures a unique token matches the observed state. As the policy spans the vocabulary, this token will be found in at most $|\mathcal{V}|$ trials. Induction over ${t=1,\dots,T}$ completes the argument.
∎

###### Theorem 3.2 (Robustness of SipIt ) .

Under the assumptions of Theorem [2.3](#S2.Thmtheorem3), fix a layer $\ell$ and define, for any prefix $\pi$ and time $t$,

$$ $\Delta_{\pi,t}\ :=\ \min_{v\neq v^{\prime}\in\mathcal{V}}\big\|\mathbf{h}_{t}(\pi\oplus v)-\mathbf{h}_{t}(\pi\oplus v^{\prime})\big\|_{2}.$ $$

Let $\mathrm{s}=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{T}\rangle$, define the prefixes $\pi_{t}=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{t-1}\rangle$ and suppose access to the perturbed hidden states

$$ $\widehat{\mathbf{h}}_{t}(\pi_{t}\oplus\mathrm{s}_{t})=\mathbf{h}_{t}(\pi_{t}\oplus\mathrm{s}_{t})+\mathbf{e}_{t},\qquad\|\mathbf{e}_{t}\|_{2}<\tfrac{\Delta_{\pi_{t},t}}{2},\quad t\in[T].$ $$

Then SipIt recovers the true sequence $\mathrm{s}$ with probability one, and terminates in at most $T|\mathcal{V}|$ steps.

###### Proof in Appendix D , Thm. D.2 , Prop. D.2 .

∎

In short, SipIt turns the almost-sure injectivity of Transformer representations into a constructive procedure: not only are hidden states unique identifiers of prompts, but the exact input sequence can be efficiently *recovered* in linear time, and often faster in practice.
It is a structural property of Transformer representations, not a quirk of initialization or training.

###### Theorem 3.1 (Correctness of SipIt ) .

###### Sketch of proof (full proof in Appendix D , Thm. D.2 , Prop. D.4 ).

###### Theorem 3.2 (Robustness of SipIt ) .

###### Proof in Appendix D , Thm. D.2 , Prop. D.2 .

## 4 Experiments

We previously proved that decoder-only Transformers are injective (§[2](#S2)) and introduced an algorithm, SipIt, that leverages this property to recover the exact input prompt from hidden states at a given layer (§[3](#S3)). We now provide extensive empirical evidence supporting our theory by showing that distinct prompts yield distinct embeddings, i.e., no collisions occur by a large margin (§[4.1](#S4.SS1)). We then demonstrate that SipIt successfully reconstructs the original input prompt (§[4.2](#S4.SS2)).

##### Environment.

All experiments were run on a single NVIDIA A100-SXM (64 GB) GPU. Python 3.11, CUDA 12.2, PyTorch 2.8.0, and transformers 4.50.0 were used for all experiments. Reported runtimes refer to this setup.

### 4.1 Searching for collisions

Figure: Figure 4: Exhaustive collision search on the $10$ closest prefix prompts. The boxplots look flat and uneventful, and that is the point: even under stress-test conditions with billions of candidate pairs, all minima stay well above the collision threshold, showing that nothing collapses.

**Table 1: Minimum pairwise distance between last-token states in the first, middle, and final layers of four models. All values are well above the collision threshold $10^{-6}$.**
| Model | $\bm{\ell}_{\mathbf{2}}$ Distance (min) |  |  |
| --- | --- | --- | --- |
| layer 1 | layer $\frac{L}{2}$ | layer $L$ |  |
| Llama-3.1-8B | 0.001 | 0.129 | 0.620 |
| Mistral-7B-v0.1 | 0.002 | 0.187 | 1.274 |
| Phi-4-mini-ins | 0.014 | 1.336 | 9.020 |
| TinyStories-33M | 0.029 | 1.434 | 2.793 |

We collected 100k prompts by uniformly sampling from a mixture of four datasets: wikipedia-en(^5^55[https://huggingface.co/datasets/wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)), C4 (Raffel et al., 2020), The Pile (Gao et al., 2020), and python-github-code(^6^66[https://huggingface.co/datasets/angie-chen55/python-github-code](https://huggingface.co/datasets/angie-chen55/python-github-code)). For each prompt, we extracted the last-token representation and systematically checked whether any two distinct prompts produced identical embeddings. This process required around 5 billion pairwise comparisons.

We observed no collisions across all models and layers: distinct prompts always yielded distinct last-token states. Figure [3](#S2.F3) (left) shows the per-layer minimum distances for the Gemma3 pretrained (Team et al., 2025) and GPT-2 (Radford et al., 2019) families, with strictly positive values throughout. [Table 1](#S4.T1) complements this by reporting the same statistic for Llama-3.1-8B (Grattafiori et al., 2024), Mistral-7B-v0.1 (Jiang et al., 2023), Phi-4-mini-instruct (Microsoft et al., 2025) and TinyStories-33M (Eldan and Li, 2023), again showing clear separation at the first, middle, and last layers.
Finally, Figure [3](#S2.F3) (right) zooms in on GPT-2 Small, revealing that these distances typically increase with depth. Additional results for GPT-2 Medium, GPT-2 Large and Gemma3 (1B, 4B, 12B) appear in Appendix [E](#A5), confirming the same trend.

**Table 2: Quantized Models: Minimum pairwise distance between last-token states in the final layer of three quantized models.**
| Model | $\bm{\ell}_{\mathbf{2}}$ Distance (min) |  |  |
| --- | --- | --- | --- |
| FP4 | INT8 | FP32 |  |
| Llama-3.1-8B | 2.281 | 6.597 | 1.274 |
| Mistral-7B-v0.1 | 1.748 | 2.692 | 1.136 |
| Phi-4-mini-instruct | 18.368 | 20.956 | 8.780 |

Figure: Figure 5: Sequence length vs. pairwise distance for GPT-2. Min, mean, and max distances rise at short lengths and then stabilize, indicating consistent separability.

Figure [5](#S4.F5) shows how pairwise distances between last-token states vary with prompt length in GPT-2 Small. Three patterns emerge: (i) the *minimum* distance is never close to zero at all lengths, and (ii) it grows rapidly at short lengths but then levels off, suggesting that beyond a moderate context size, adding tokens does not affect separability; (iii) the overall spread (min-max) stays bounded, with no sign of pathological collapses. Similar behavior is seen in Gemma3 (see Appendix [E](#A5), Figure [9](#A5.F9)). Overall, clear margins emerge quickly and then stabilize, making collisions unlikely at any sequence length.

Exhaustive collision test.
Different from previous experiments, in this setting (Figure [4](#S4.F4)), we restrict our analysis to the $10$ prompts from the dataset mixture whose embeddings have the smallest last-token distances. For each of these prompts, we appended *every* vocabulary token and computed all pairwise distances between the resulting last-token states, effectively performing an exhaustive search over continuations and yielding more than 343 billion prompt pairs per model.

Figure: Figure 6: Inversion time as a function of depth. Runtimes rise only mildly across layers.

This exhaustive experiment helps rule out the possibility that earlier observations were simply due to chance in random sampling rather than a true absence of collisions.
While a complete search over all possible prompts would be ideal, it is computationally infeasible. The number of unique prompts grows exponentially with sequence length, and the number of pairwise comparisons grows even faster. For context, even with single-token prompts and the vocabulary size of Gemma3-1B, there are already over 34 billion possible prompt pairs, making exhaustive evaluation entirely impractical.
Our compromise still revealed structure: we identified 5 prompt pairs with highly similar last-token embeddings, suggesting overlapping semantic content and motivating us to ask whether distinct next tokens could preserve meaning, i.e., yield essentially identical last-token hidden states.

Figure [4](#S4.F4) reports the resulting distributions as boxplots for both GPT-2 Small and Gemma3-1B, with distances far from zero (no collision), confirming local injectivity as predicted by our theory.

##### FP4 and INT8 weight quantization.

To assess how weight quantization affects pairwise representation distances, we conducted additional experiments with FP4 and INT8 quantization on several models (Llama-3.1-8B, Phi-4-mini-instruct, and Mistral-7B-v0.1). We further extended this analysis to FP4-quantized 14B and 70B parameter models, namely Phi-4 (14B) and Llama-3.1-70B. As shown in [Tables 3](#S4.T3) and [3](#S4.T3), across all tested models quantization (1) does not introduce any collisions, (2) more than doubles the minimum distance between representations, thereby preserving the integrity of the representation space, and (3) maintains this separation even as model size increases substantially.

### 4.2 Invertibility results

We now test whether the theoretical injectivity translates into exact recovery on pre-trained models. Using SipIt with only the hidden states at a fixed layer, we attempt to reconstruct the full prompt token-by-token for GPT-2 Small.
We sample 100 prompts, with a $90\%$-$10\%$ split between meaningful sentences and random token sequences (to test robustness in unstructured cases), and attempt to reconstruct them from hidden states.
We compare against HardPrompts (Wen et al., 2023), which leverages gradient signals for approximate prompt discovery, and against a SipIt ablation that replaces the gradient-guided candidate policy with the [uniformly random policy](#alg2) (BruteForce).

Other inversion approaches (Morris et al., 2023a; b; Nazir et al., 2025) tackle a different setting altogether: they operate in black box access, using sequences of next-token logprobs or encoder logits rather than hidden states, and train auxiliary inverters to reconstruct text, at high computational cost. Their outputs are typically approximate and not guaranteed exact. These differences make them complementary but not directly comparable to our setting of training-free, *exact* inversion from *hidden states* in decoder-only LMs.

**Table 4: Inversion performance on FP4-quantized models with different vocabulary sizes. SipIt recovers all tokens with 100% accuracy while exploring less than $0.22\%$ of the vocabulary on average.**
| Model | Vocab size | Inversion Performance |  |  |
| --- | --- | --- | --- | --- |
| Accuracy | Time (s) | Vocab explored (%) |  |  |
| Mistral-7B-v0.1 | 32000 | 100% | 111.78 $\pm\,\,\,$ 46.50 | 0.19 $\pm$ 0.08 % |
| Llama-3.1-8B | 128255 | 100% | 549.48 $\pm$ 265.75 | 0.21 $\pm$ 0.10 % |

**Table 5: SipIt ensures efficient exact recovery, unlike HardPrompts (no recovery) or BruteForce (infeasible runtime).**
| Method | Mean Time (s) | Accuracy |
| --- | --- | --- |
| HardPrompts | $6132.59\pm 104.61$ | 0.00 |
| BruteForce (ours) | $3889.61\pm 691.17$ | 1.00 |
| SipIt (ours) | $\mathbf{28.01\pm 35.87}$ | $\mathbf{1.00}$ |

Results are reported in [Table 5](#S4.T5). Across all prompts (20 tokens each), SipIt recovers the exact sequence with $100\%$ token-level accuracy (no errors, no collisions), matching the theoretical guarantee of linear-time convergence. In contrast, HardPrompts completely fails to recover the input, while BruteForce eventually succeeds but at a prohibitive computational cost, requiring several orders of magnitude longer.

##### Robustness and vocabulary scaling.

The theoretical analysis in [Theorem 3.2](#S3.Thmtheorem2) shows that our inversion algorithm is robust to a certain level of noise while maintaining linear scaling in vocabulary size. To empirically validate this, we use FP4-quantized versions of Mistral-7B-v0.1 ($\approx 32\text{K}$ vocabulary size) and Llama-3.1-8B ($\approx 128\text{K}$). We sample 50 prompts (10 tokens each) and attempt to reconstruct them from hidden states corrupted by FP4 weight quantization. As shown in [Table 4](#S4.T4), SipIt reconstructs all inputs with perfect accuracy while exploring, on average, less than $0.22\%$ of the vocabulary, demonstrating that the gradient-based heuristic is both robust to quantization noise and highly efficient. From a complexity perspective, the nearly constant percentage of tokens explored across the two vocabulary scales empirically confirms the predicted linear scaling.

##### Effect of layer depth.

Finally, Figure [6](#S4.F6) shows inversion times by layer for longer prompts (ranging from $20$ to $200$ tokens). Although deeper layers are costlier in principle (since verifying a candidate and computing gradients require traversing more blocks), the effect is minor: runtimes rise only slightly from first to last layer, and the scaling remains graceful overall. Likely, earlier layers need more iterations to converge, while deep layers store richer information that reduces the search effort. As a result, the net cost remains stable, confirming SipIt is efficient across depth.

## 5 Related work

Our results connect to two active lines of research: theoretical analyses of Transformer architectures, and inverse problems in language modeling. We briefly review both to position our contributions.

##### Analytical properties of Transformers.

Viewed as functions on $\mathbb{R}^{d}$, individual Transformer components are clearly non-injective:
LayerNorm collapses along per-example statistics (Ba et al., 2016), residual connections can cancel, and
in attention-only stacks, rank decays doubly-exponentially with depth (Dong et al., 2021).
Likewise, on the output side, the softmax bottleneck constrains the distributions reachable by language models
(Yang et al., 2018). From this algebraic perspective, Transformers seem inherently many-to-one,
an intuition echoed by classical completeness and universal-approximation theorems for Transformers,
which show that highly many-to-one maps can be represented in principle; we briefly review these results in Appendix [F](#A6.SS0.SSS0.Px1).

Our focus is different: we study the discrete-to-continuous map
from *prompts* $\mathrm{s}\in\mathcal{V}^{\leq K}$ to *hidden states* in $\mathbb{R}^{d}$.
In this setting, analytic viewpoints on Transformer computation become powerful: treating each layer as
a real-analytic map yields almost-sure guarantees that hold at finite width, depth,
and training horizon (Appendix [F](#A6) surveys which modern LLMs satisfy
this assumption and proves the analyticity for all activation functions encountered in practice).
Recent work has adopted this angle for related properties: Jiang and Haghtalab (2025) show that
building blocks of modern architectures are *almost always surjective*,
while Sutter et al. (2025) prove that Transformers at random initialization
are *almost surely injective* with respect to the entire hidden-state matrix (and only at initialization).
Differently, we prove injectivity with respect to the parameters and at the task-relevant
*last-token state*; crucially, we show that injectivity is not an initialization artifact
but *persists under training*.

##### Inverse problems in language modeling.

Model inversion asks whether one can reconstruct a model’s input prompt from outputs or internal signals (Sun et al., 2021).
In the context of language models, this question has motivated a growing body of work exploring practical inversion strategies.
Output-to-prompt methods infer prompts from generated continuations but yield only approximate
reconstructions (Zhang et al., 2024). Recent work shows that even black-box outputs are information-rich:
Morris et al. (2023b) train a separate inverter to map next-token
probability vectors to text, and Nazir et al. (2025) extend this by taking sequences of logprobs,
applying a linear compression to embedding dimension, and training an encoder-decoder inverter;
this achieves higher exact-match rates but still without guarantees. Complementarily,
Morris et al. (2023a) reconstruct text from encoder logits via a trained iterative inverter.
These contributions highlight privacy risks when probabilities or embeddings are exposed, but they
differ from our setting: they rely on trained inverters, remain approximate, and do not invert
*hidden states* of decoder-only LMs.

A related line of work frames the task as automated prompt optimization, casting prompt design as discrete sequence optimization aligned with downstream performance (Guo et al., 2025; Sun et al., 2022; Deng et al., 2022); methods such as AutoPrompt (Shin et al., 2020) and Hard Prompts Made Easy (Wen et al., 2023) use gradient signals to discover effective, but approximate, prompts. Most closely related to ours, Thomas et al. (2025) recover prompts from hidden states via a sequential algorithm that uses an LLM-based policy to rank candidates; lacking injectivity guarantees, however, it must score all vocabulary tokens before committing, with no formal exactness guarantees.

Unlike all prior work, our approach is training-free, efficient, and comes with *provable* linear-time guarantees for *exact* recovery from internal states.

## 6 Discussion and conclusions

This work establishes that decoder-only Transformers are almost surely injective: distinct prompts produce distinct hidden states under standard initialization and training. Building on this structural result, we introduced SipIt, the first algorithm that can recover the *exact* input sequence from hidden activations, with provable linear-time guarantees. Together, these contributions move injectivity from an informal belief to a rigorously grounded and operational property of language models.

The scientific impact is clear. Our findings reconcile two competing views in the community: Transformers as “lossy” due to nonlinearities, normalization, and many-to-one attention, versus language models as injective in their hidden representations. We advocate viewing language models as maps on the *sequence* space rather than the embedding space; under this perspective, we prove that all information about the input sequence is almost surely preserved end-to-end. The constructive inversion offered by SipIt strengthens this point in practice, establishing a clean baseline for interpretability and auditing: if probes or inversion methods fail, it is not because the information is missing. For mechanistic interpretability in particular, injectivity guarantees that last-token states faithfully encode the full input, giving a sound foundation for causal and probing analyses.

Beyond theory, the findings carry practical and legal implications. Hidden states are not abstractions
but the prompt in disguise: any system that stores or transmits them is effectively handling user text
itself, with direct consequences for privacy, deletion, and compliance (Miranda et al., 2025). The evolving regulatory landscape
has not yet fully reckoned with this fact. The Hamburg Data Protection Commissioner, for instance, argued
that LLM *parameters* do not constitute personal data because training data is transformed into
abstract mathematical representations during learning, and that it “remains doubtful whether any
extractable data records constitute personal data” (HmbBfDI, 2024). That analysis, however, concerns
training data encoded in model weights; it does not address the hidden representations computed at
inference time. Our results reveal that these representations are lossless encodings of the user’s
exact input, recoverable in full via SipIt. Consequently, any system that stores, caches,
or transmits hidden states is effectively handling the user’s verbatim text, and the corresponding pipelines should be subject to the same data-protection obligations as the raw prompts they encode.

Finally, this work opens several directions. Extending the analysis to multimodal architectures such as music and vision Transformers is an open problem. Studying approximate inversion under noise or quantization will clarify how robust invertibility remains in practice. Bridging these technical insights with evolving regulatory frameworks will be crucial for safe and responsible deployment.

## Reproducibility statement

The experimental setup (hardware, software versions, and dataset construction) is described in §[4](#S4); the 100k-prompt benchmark uses uniform sampling from four public datasets detailed in §[4.1](#S4.SS1). On the theory side, every theorem stated in the main text is accompanied by a complete proof in the appendix: analytic preliminaries in Appendix [A](#A1), the formal definition of the Transformer language model and the proof that it is real-analytic in Appendix [B](#A2), almost-sure injectivity and its preservation under training in Appendix [C](#A3), and SipIt correctness and robustness in Appendix [D](#A4). Appendix [E](#A5) provides Implementation Details and Additional Experiments, and Appendix [F](#A6) verifies that all activation functions used in all modern LLMs satisfy the analyticity assumption.

## Acknowledgments

[Figure 1](#S1.F1) is adapted from Autoencoder Diagrams by Keenan Crane (2025), used under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/). We further acknowledge Adam Barla for the initial discussions on LLMs invertibility.

This work has been supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the NextGenerationEU Program, the MUR FIS2 grant n. FIS-2023-00942 “NEXUS” (cup B53C25001030001), and partly by Sapienza University of Rome via the Seed of ERC grant “MINT.AI” (cup B83C25001040001).

## References

- W. E. Aitken (2020)
General topology. part 4: metric spaces.
External Links: [Link](https://public.csusm.edu/aitken_html/Essays/Topology/metric_spaces.pdf)
Cited by: [item 1](#A1.I5.i1.p1.1).
- S. Arora, H. Browne, and D. Daners (2021)
An alternative approach to fréchet derivatives.
Journal of the Australian Mathematical Society 111 (2), pp. 202–220.
Cited by: [Proposition A.14](#A1.Thmproposition14.p1.4.4).
- J. L. Ba, J. R. Kiros, and G. E. Hinton (2016)
Layer normalization.
arXiv preprint arXiv:1607.06450.
External Links: [Link](https://arxiv.org/abs/1607.06450)
Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1).
- J. E. Chacón and T. Duong (2020)
Higher order differential analysis with vectorized derivatives.
arXiv preprint arXiv:2011.01833.
Cited by: [Definition A.3](#A1.Thmdefinition3.p2.1.1).
- M. Deng, J. Wang, C. Hsieh, Y. Wang, H. Guo, T. Shu, M. Song, E. P. Xing, and Z. Hu (2022)
RLPrompt: optimizing discrete text prompts with reinforcement learning.
External Links: 2205.12548,
[Link](https://arxiv.org/abs/2205.12548)
Cited by: [§5](#S5.SS0.SSS0.Px2.p2.1).
- Y. Dong, J. Cordonnier, and A. Loukas (2021)
Attention is not all you need: pure attention loses rank doubly exponentially with depth.
In Proceedings of the 38th International Conference on Machine Learning (ICML),
Proceedings of Machine Learning Research, Vol. 139.
External Links: [Link](https://proceedings.mlr.press/v139/dong21a.html)
Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1).
- R. Eldan and Y. Li (2023)
TinyStories: how small can language models be and still speak coherent english?.
External Links: 2305.07759,
[Link](https://arxiv.org/abs/2305.07759)
Cited by: [§4.1](#S4.SS1.p2.1).
- G. B. Folland (1999)
Real analysis.
2 edition, Pure and Applied Mathematics: A Wiley Series of Texts,
Monographs and Tracts, John Wiley & Sons, Nashville, TN (en).
Cited by: [Definition A.11](#A1.Thmdefinition11),
[Theorem C.4](#A3.Thmtheorem4).
- L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima, S. Presser, and C. Leahy (2020)
The pile: an 800gb dataset of diverse text for language modeling.
External Links: 2101.00027,
[Link](https://arxiv.org/abs/2101.00027)
Cited by: [§4.1](#S4.SS1.p1.1).
- A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, A. Yang, A. Fan, A. Goyal, A. Hartshorn, A. Yang, A. Mitra, A. Sravankumar, A. Korenev, A. Hinsvark, A. Rao, A. Zhang, A. Rodriguez, A. Gregerson, A. Spataru, B. Roziere, B. Biron, B. Tang, B. Chern, C. Caucheteux, C. Nayak, C. Bi, C. Marra, C. McConnell, C. Keller, C. Touret, C. Wu, C. Wong, C. C. Ferrer, C. Nikolaidis, D. Allonsius, D. Song, D. Pintz, D. Livshits, D. Wyatt, D. Esiobu, D. Choudhary, D. Mahajan, D. Garcia-Olano, D. Perino, D. Hupkes, E. Lakomkin, E. AlBadawy, E. Lobanova, E. Dinan, E. M. Smith, F. Radenovic, F. Guzmán, F. Zhang, G. Synnaeve, G. Lee, G. L. Anderson, G. Thattai, G. Nail, G. Mialon, G. Pang, G. Cucurell, H. Nguyen, H. Korevaar, H. Xu, H. Touvron, I. Zarov, I. A. Ibarra, I. Kloumann, I. Misra, I. Evtimov, J. Zhang, J. Copet, J. Lee, J. Geffert, J. Vranes, J. Park, J. Mahadeokar, J. Shah, J. van der Linde, J. Billock, J. Hong, J. Lee, J. Fu, J. Chi, J. Huang, J. Liu, J. Wang, J. Yu, J. Bitton, J. Spisak, J. Park, J. Rocca, J. Johnstun, J. Saxe, J. Jia, K. V. Alwala, K. Prasad, K. Upasani, K. Plawiak, K. Li, K. Heafield, K. Stone, K. El-Arini, K. Iyer, K. Malik, K. Chiu, K. Bhalla, K. Lakhotia, L. Rantala-Yeary, L. van der Maaten, L. Chen, L. Tan, L. Jenkins, L. Martin, L. Madaan, L. Malo, L. Blecher, L. Landzaat, L. de Oliveira, M. Muzzi, M. Pasupuleti, M. Singh, M. Paluri, M. Kardas, M. Tsimpoukelli, M. Oldham, M. Rita, M. Pavlova, M. Kambadur, M. Lewis, M. Si, M. K. Singh, M. Hassan, N. Goyal, N. Torabi, N. Bashlykov, N. Bogoychev, N. Chatterji, N. Zhang, O. Duchenne, O. Çelebi, P. Alrassy, P. Zhang, P. Li, P. Vasic, P. Weng, P. Bhargava, P. Dubal, P. Krishnan, P. S. Koura, P. Xu, Q. He, Q. Dong, R. Srinivasan, R. Ganapathy, R. Calderer, R. S. Cabral, R. Stojnic, R. Raileanu, R. Maheswari, R. Girdhar, R. Patel, R. Sauvestre, R. Polidoro, R. Sumbaly, R. Taylor, R. Silva, R. Hou, R. Wang, S. Hosseini, S. Chennabasappa, S. Singh, S. Bell, S. S. Kim, S. Edunov, S. Nie, S. Narang, S. Raparthy, S. Shen, S. Wan, S. Bhosale, S. Zhang, S. Vandenhende, S. Batra, S. Whitman, S. Sootla, S. Collot, S. Gururangan, S. Borodinsky, T. Herman, T. Fowler, T. Sheasha, T. Georgiou, T. Scialom, T. Speckbacher, T. Mihaylov, T. Xiao, U. Karn, V. Goswami, V. Gupta, V. Ramanathan, V. Kerkez, V. Gonguet, V. Do, V. Vogeti, V. Albiero, V. Petrovic, W. Chu, W. Xiong, W. Fu, W. Meers, X. Martinet, X. Wang, X. Wang, X. E. Tan, X. Xia, X. Xie, X. Jia, X. Wang, Y. Goldschlag, Y. Gaur, Y. Babaei, Y. Wen, Y. Song, Y. Zhang, Y. Li, Y. Mao, Z. D. Coudert, Z. Yan, Z. Chen, Z. Papakipos, A. Singh, A. Srivastava, A. Jain, A. Kelsey, A. Shajnfeld, A. Gangidi, A. Victoria, A. Goldstand, A. Menon, A. Sharma, A. Boesenberg, A. Baevski, A. Feinstein, A. Kallet, A. Sangani, A. Teo, A. Yunus, A. Lupu, A. Alvarado, A. Caples, A. Gu, A. Ho, A. Poulton, A. Ryan, A. Ramchandani, A. Dong, A. Franco, A. Goyal, A. Saraf, A. Chowdhury, A. Gabriel, A. Bharambe, A. Eisenman, A. Yazdan, B. James, B. Maurer, B. Leonhardi, B. Huang, B. Loyd, B. D. Paola, B. Paranjape, B. Liu, B. Wu, B. Ni, B. Hancock, B. Wasti, B. Spence, B. Stojkovic, B. Gamido, B. Montalvo, C. Parker, C. Burton, C. Mejia, C. Liu, C. Wang, C. Kim, C. Zhou, C. Hu, C. Chu, C. Cai, C. Tindal, C. Feichtenhofer, C. Gao, D. Civin, D. Beaty, D. Kreymer, D. Li, D. Adkins, D. Xu, D. Testuggine, D. David, D. Parikh, D. Liskovich, D. Foss, D. Wang, D. Le, D. Holland, E. Dowling, E. Jamil, E. Montgomery, E. Presani, E. Hahn, E. Wood, E. Le, E. Brinkman, E. Arcaute, E. Dunbar, E. Smothers, F. Sun, F. Kreuk, F. Tian, F. Kokkinos, F. Ozgenel, F. Caggioni, F. Kanayet, F. Seide, G. M. Florez, G. Schwarz, G. Badeer, G. Swee, G. Halpern, G. Herman, G. Sizov, Guangyi, Zhang, G. Lakshminarayanan, H. Inan, H. Shojanazeri, H. Zou, H. Wang, H. Zha, H. Habeeb, H. Rudolph, H. Suk, H. Aspegren, H. Goldman, H. Zhan, I. Damlaj, I. Molybog, I. Tufanov, I. Leontiadis, I. Veliche, I. Gat, J. Weissman, J. Geboski, J. Kohli, J. Lam, J. Asher, J. Gaya, J. Marcus, J. Tang, J. Chan, J. Zhen, J. Reizenstein, J. Teboul, J. Zhong, J. Jin, J. Yang, J. Cummings, J. Carvill, J. Shepard, J. McPhie, J. Torres, J. Ginsburg, J. Wang, K. Wu, K. H. U, K. Saxena, K. Khandelwal, K. Zand, K. Matosich, K. Veeraraghavan, K. Michelena, K. Li, K. Jagadeesh, K. Huang, K. Chawla, K. Huang, L. Chen, L. Garg, L. A, L. Silva, L. Bell, L. Zhang, L. Guo, L. Yu, L. Moshkovich, L. Wehrstedt, M. Khabsa, M. Avalani, M. Bhatt, M. Mankus, M. Hasson, M. Lennie, M. Reso, M. Groshev, M. Naumov, M. Lathi, M. Keneally, M. Liu, M. L. Seltzer, M. Valko, M. Restrepo, M. Patel, M. Vyatskov, M. Samvelyan, M. Clark, M. Macey, M. Wang, M. J. Hermoso, M. Metanat, M. Rastegari, M. Bansal, N. Santhanam, N. Parks, N. White, N. Bawa, N. Singhal, N. Egebo, N. Usunier, N. Mehta, N. P. Laptev, N. Dong, N. Cheng, O. Chernoguz, O. Hart, O. Salpekar, O. Kalinli, P. Kent, P. Parekh, P. Saab, P. Balaji, P. Rittner, P. Bontrager, P. Roux, P. Dollar, P. Zvyagina, P. Ratanchandani, P. Yuvraj, Q. Liang, R. Alao, R. Rodriguez, R. Ayub, R. Murthy, R. Nayani, R. Mitra, R. Parthasarathy, R. Li, R. Hogan, R. Battey, R. Wang, R. Howes, R. Rinott, S. Mehta, S. Siby, S. J. Bondu, S. Datta, S. Chugh, S. Hunt, S. Dhillon, S. Sidorov, S. Pan, S. Mahajan, S. Verma, S. Yamamoto, S. Ramaswamy, S. Lindsay, S. Lindsay, S. Feng, S. Lin, S. C. Zha, S. Patil, S. Shankar, S. Zhang, S. Zhang, S. Wang, S. Agarwal, S. Sajuyigbe, S. Chintala, S. Max, S. Chen, S. Kehoe, S. Satterfield, S. Govindaprasad, S. Gupta, S. Deng, S. Cho, S. Virk, S. Subramanian, S. Choudhury, S. Goldman, T. Remez, T. Glaser, T. Best, T. Koehler, T. Robinson, T. Li, T. Zhang, T. Matthews, T. Chou, T. Shaked, V. Vontimitta, V. Ajayi, V. Montanez, V. Mohan, V. S. Kumar, V. Mangla, V. Ionescu, V. Poenaru, V. T. Mihailescu, V. Ivanov, W. Li, W. Wang, W. Jiang, W. Bouaziz, W. Constable, X. Tang, X. Wu, X. Wang, X. Wu, X. Gao, Y. Kleinman, Y. Chen, Y. Hu, Y. Jia, Y. Qi, Y. Li, Y. Zhang, Y. Zhang, Y. Adi, Y. Nam, Yu, Wang, Y. Zhao, Y. Hao, Y. Qian, Y. Li, Y. He, Z. Rait, Z. DeVito, Z. Rosnbrick, Z. Wen, Z. Yang, Z. Zhao, and Z. Ma (2024)
The llama 3 herd of models.
External Links: 2407.21783,
[Link](https://arxiv.org/abs/2407.21783)
Cited by: [§4.1](#S4.SS1.p2.1).
- Q. Guo, R. Wang, J. Guo, B. Li, K. Song, X. Tan, G. Liu, J. Bian, and Y. Yang (2025)
EvoPrompt: connecting llms with evolutionary algorithms yields powerful prompt optimizers.
External Links: 2309.08532,
[Link](https://arxiv.org/abs/2309.08532)
Cited by: [§5](#S5.SS0.SSS0.Px2.p2.1).
- H. V. Henderson and S. R. Searle (1981)
The vec-permutation matrix, the vec operator and kronecker products: a review.
Linear and multilinear algebra 9 (4), pp. 271–288.
Cited by: [Definition A.3](#A1.Thmdefinition3.p1.1.1),
[§C.2.1](#A3.SS2.SSS1.7.p2.5).
- HmbBfDI (2024)
Discussion paper: large language models and personal data.
External Links: [Link](https://datenschutz-hamburg.de/fileadmin/user_upload/HmbBfDI/Datenschutz/Informationen/240715_Discussion_Paper_Hamburg_DPA_KI_Models.pdf)
Cited by: [§6](#S6.p3.1).
- R. A. Horn and C. R. Johnson (2013)
Matrix analysis.
2 edition, Cambridge University Press, Cambridge.
Cited by: [§C.2.1](#A3.SS2.SSS1.5.p1.6).
- A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed (2023)
Mistral 7b.
External Links: 2310.06825,
[Link](https://arxiv.org/abs/2310.06825)
Cited by: [§4.1](#S4.SS1.p2.1).
- H. Jiang and N. Haghtalab (2025)
On surjectivity of neural networks: can you elicit any behavior from your model?.
arXiv preprint arXiv:2508.19445.
External Links: [Link](https://arxiv.org/abs/2508.19445)
Cited by: [§5](#S5.SS0.SSS0.Px1.p2.2).
- T. G. Kolda and B. W. Bader (2009)
Tensor decompositions and applications.
SIAM Review 51 (3), pp. 455–500.
External Links: [Document](https://dx.doi.org/10.1137/07070111X),
[Link](https://doi.org/10.1137/07070111X),
https://doi.org/10.1137/07070111X
Cited by: [§A.1](#A1.SS1.p3.2).
- S. G. Krantz and H. R. Parks (2002)
A primer of real analytic functions.
Springer Science & Business Media.
Cited by: [§A.2.3](#A1.SS2.SSS3.4.p2.5).
- A. D. Lewis (2014)
Chapter 1: holomorphic and real analytic calculus.
Note: Notes on Global Analysis, Vol. 1, Queen’s UniversityVersion: 2014-02-28
External Links: [Link](https://mast.queensu.ca/~andrew/teaching/math942/pdf/1chapter1.pdf)
Cited by: [Definition A.1](#A1.Thmdefinition1),
[Proposition A.1](#A1.Thmproposition1),
[Proposition A.2](#A1.Thmproposition2),
[§C.2.1](#A3.SS2.SSS1.16.p1.2).
- D. G. Luenberger (1997)
Optimization by vector space methods.
Wiley-Interscience.
Cited by: [Definition A.5](#A1.Thmdefinition5).
- J. R. Magnus and H. Neudecker (2019)
Matrix differential calculus with applications in statistics and econometrics.
John Wiley & Sons, Inc.
Cited by: [Definition A.6](#A1.Thmdefinition6),
[Proposition A.14](#A1.Thmproposition14.p1.5.1),
[§C.2.1](#A3.SS2.SSS1.3.p3.3).
- Microsoft, :, A. Abouelenin, A. Ashfaq, A. Atkinson, H. Awadalla, N. Bach, J. Bao, A. Benhaim, M. Cai, V. Chaudhary, C. Chen, D. Chen, D. Chen, J. Chen, W. Chen, Y. Chen, Y. Chen, Q. Dai, X. Dai, R. Fan, M. Gao, M. Gao, A. Garg, A. Goswami, J. Hao, A. Hendy, Y. Hu, X. Jin, M. Khademi, D. Kim, Y. J. Kim, G. Lee, J. Li, Y. Li, C. Liang, X. Lin, Z. Lin, M. Liu, Y. Liu, G. Lopez, C. Luo, P. Madan, V. Mazalov, A. Mitra, A. Mousavi, A. Nguyen, J. Pan, D. Perez-Becker, J. Platin, T. Portet, K. Qiu, B. Ren, L. Ren, S. Roy, N. Shang, Y. Shen, S. Singhal, S. Som, X. Song, T. Sych, P. Vaddamanu, S. Wang, Y. Wang, Z. Wang, H. Wu, H. Xu, W. Xu, Y. Yang, Z. Yang, D. Yu, I. Zabir, J. Zhang, L. L. Zhang, Y. Zhang, and X. Zhou (2025)
Phi-4-mini technical report: compact yet powerful multimodal language models via mixture-of-loras.
External Links: 2503.01743,
[Link](https://arxiv.org/abs/2503.01743)
Cited by: [§4.1](#S4.SS1.p2.1).
- M. Miranda, E. S. Ruzzetti, A. Santilli, F. M. Zanzotto, S. Bratières, and E. Rodolà (2025)
Preserving privacy in large language models: a survey on current threats and solutions.
Transactions on Machine Learning Research.
Note:
External Links: ISSN 2835-8856,
[Link](https://openreview.net/forum?id=Ss9MTTN7OL)
Cited by: [§6](#S6.p3.1).
- B. Mityagin (2015)
The zero set of a real analytic function.
arXiv preprint arXiv:1512.07276.
Cited by: [Theorem A.1](#A1.Thmtheorem1),
[Remark 6](#Thmremark6.p1.3.3).
- J. X. Morris, V. Kuleshov, V. Shmatikov, and A. M. Rush (2023a)
Text embeddings reveal (almost) as much as text.
External Links: 2310.06816,
[Link](https://arxiv.org/abs/2310.06816)
Cited by: [§4.2](#S4.SS2.p2.1),
[§5](#S5.SS0.SSS0.Px2.p1.1).
- J. X. Morris, W. Zhao, J. T. Chiu, V. Shmatikov, and A. M. Rush (2023b)
Language model inversion.
External Links: 2311.13647,
[Link](https://arxiv.org/abs/2311.13647)
Cited by: [§4.2](#S4.SS2.p2.1),
[§5](#S5.SS0.SSS0.Px2.p1.1).
- J. R. Munkres (2000)
Topology.
2 edition, Prentice Hall, Upper Saddle River, NJ.
Cited by: [item 2](#A1.I5.i2.p1.2),
[item 3](#A1.I5.i3.p1.2),
[item 4](#A1.I5.i4.p1.1.1),
[item 5](#A1.I5.i5.p1.1.1),
[Definition A.9](#A1.Thmdefinition9).
- M. Nazir, M. Finlayson, J. X. Morris, X. Ren, and S. Swayamdipta (2025)
Better language model inversion by compactly representing next-token distributions.
External Links: 2506.17090,
[Link](https://arxiv.org/abs/2506.17090)
Cited by: [§4.2](#S4.SS2.p2.1),
[§5](#S5.SS0.SSS0.Px2.p1.1).
- J. Pérez, J. Marinković, and P. Barceló (2019)
On the turing completeness of modern neural network architectures.
In International Conference on Learning Representations,
External Links: [Link](https://openreview.net/forum?id=HyGBdo0qFm)
Cited by: [Appendix F](#A6.SS0.SSS0.Px1.p2.4).
- A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever (2019)
Language models are unsupervised multitask learners.
External Links: [Link](https://api.semanticscholar.org/CorpusID:160025533)
Cited by: [§E.2.2](#A5.SS2.SSS2.Px2.p1.1),
[§4.1](#S4.SS1.p2.1).
- C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu (2020)
Exploring the limits of transfer learning with a unified text-to-text transformer.
Journal of Machine Learning Research 21 (140), pp. 1–67.
External Links: [Link](http://jmlr.org/papers/v21/20-074.html)
Cited by: [§4.1](#S4.SS1.p1.1).
- A. Razzhigaev, M. Mikhalchuk, E. Goncharova, I. Oseledets, D. Dimitrov, and A. Kuznetsov (2024)
The shape of learning: anisotropy and intrinsic dimensions in transformer-based models.
External Links: 2311.05928,
[Link](https://arxiv.org/abs/2311.05928)
Cited by: [§E.5](#A5.SS5.p1.1).
- A. Razzhigaev, M. Mikhalchuk, T. Rahmatullaev, E. Goncharova, P. Druzhinina, I. Oseledets, and A. Kuznetsov (2025)
LLM-microscope: uncovering the hidden role of punctuation in context memory of transformers.
External Links: 2502.15007,
[Link](https://arxiv.org/abs/2502.15007)
Cited by: [§E.5](#A5.SS5.p1.1).
- W. Rudin (1976)
Principles of mathematical analysis.
3 edition, McGraw–Hill, New York.
Cited by: [Theorem A.2](#A1.Thmtheorem2),
[§C.2.2](#A3.SS2.SSS2.2.p2.9),
[Appendix F](#A6.3.p1.8).
- T. Shin, Y. Razeghi, R. L. L. IV, E. Wallace, and S. Singh (2020)
AutoPrompt: eliciting knowledge from language models with automatically generated prompts.
External Links: 2010.15980,
[Link](https://arxiv.org/abs/2010.15980)
Cited by: [§5](#S5.SS0.SSS0.Px2.p2.1).
- R. Shwartz-Ziv and N. Tishby (2017)
Opening the black box of deep neural networks via information.
External Links: 1703.00810,
[Link](https://arxiv.org/abs/1703.00810)
Cited by: [§E.5](#A5.SS5.SSS0.Px4.p2.1).
- M. Spivak (1971)
Calculus on manifolds.
Westview Press, Philadelphia, PA.
Cited by: [Definition A.10](#A1.Thmdefinition10).
- T. Sun, Y. Shao, H. Qian, X. Huang, and X. Qiu (2022)
Black-box tuning for language-model-as-a-service.
External Links: 2201.03514,
[Link](https://arxiv.org/abs/2201.03514)
Cited by: [§5](#S5.SS0.SSS0.Px2.p2.1).
- Z. Sun, F. Latorre, T. Sanchez, and V. Cevher (2021)
A plug-and-play deep image prior.
In ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
pp. 8103–8107.
External Links: [Link](http://dx.doi.org/10.1109/ICASSP39728.2021.9414879),
[Document](https://dx.doi.org/10.1109/icassp39728.2021.9414879)
Cited by: [§5](#S5.SS0.SSS0.Px2.p1.1).
- Z. Sun and Y. Yang (2020)
An em approach to non-autoregressive conditional sequence generation.
In International Conference on Machine Learning,
External Links: [Link](https://api.semanticscholar.org/CorpusID:220265867)
Cited by: [Appendix F](#A6.SS0.SSS0.Px1.p2.4).
- D. Sutter, J. Minder, T. Hofmann, and T. Pimentel (2025)
The non-linear representation dilemma: is causal abstraction enough for mechanistic interpretability?.
External Links: 2507.08802,
[Link](https://arxiv.org/abs/2507.08802)
Cited by: [§5](#S5.SS0.SSS0.Px1.p2.2),
[Remark 13](#Thmremark13.p1.2).
- G. Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Perrin, T. Matejovicova, A. Ramé, M. Rivière, L. Rouillard, T. Mesnard, G. Cideron, J. Grill, S. Ramos, E. Yvinec, M. Casbon, E. Pot, I. Penchev, G. Liu, F. Visin, K. Kenealy, L. Beyer, X. Zhai, A. Tsitsulin, R. Busa-Fekete, A. Feng, N. Sachdeva, B. Coleman, Y. Gao, B. Mustafa, I. Barr, E. Parisotto, D. Tian, M. Eyal, C. Cherry, J. Peter, D. Sinopalnikov, S. Bhupatiraju, R. Agarwal, M. Kazemi, D. Malkin, R. Kumar, D. Vilar, I. Brusilovsky, J. Luo, A. Steiner, A. Friesen, A. Sharma, A. Sharma, A. M. Gilady, A. Goedeckemeyer, A. Saade, A. Feng, A. Kolesnikov, A. Bendebury, A. Abdagic, A. Vadi, A. György, A. S. Pinto, A. Das, A. Bapna, A. Miech, A. Yang, A. Paterson, A. Shenoy, A. Chakrabarti, B. Piot, B. Wu, B. Shahriari, B. Petrini, C. Chen, C. L. Lan, C. A. Choquette-Choo, C. Carey, C. Brick, D. Deutsch, D. Eisenbud, D. Cattle, D. Cheng, D. Paparas, D. S. Sreepathihalli, D. Reid, D. Tran, D. Zelle, E. Noland, E. Huizenga, E. Kharitonov, F. Liu, G. Amirkhanyan, G. Cameron, H. Hashemi, H. Klimczak-Plucińska, H. Singh, H. Mehta, H. T. Lehri, H. Hazimeh, I. Ballantyne, I. Szpektor, I. Nardini, J. Pouget-Abadie, J. Chan, J. Stanton, J. Wieting, J. Lai, J. Orbay, J. Fernandez, J. Newlan, J. Ji, J. Singh, K. Black, K. Yu, K. Hui, K. Vodrahalli, K. Greff, L. Qiu, M. Valentine, M. Coelho, M. Ritter, M. Hoffman, M. Watson, M. Chaturvedi, M. Moynihan, M. Ma, N. Babar, N. Noy, N. Byrd, N. Roy, N. Momchev, N. Chauhan, N. Sachdeva, O. Bunyan, P. Botarda, P. Caron, P. K. Rubenstein, P. Culliton, P. Schmid, P. G. Sessa, P. Xu, P. Stanczyk, P. Tafti, R. Shivanna, R. Wu, R. Pan, R. Rokni, R. Willoughby, R. Vallu, R. Mullins, S. Jerome, S. Smoot, S. Girgin, S. Iqbal, S. Reddy, S. Sheth, S. Põder, S. Bhatnagar, S. R. Panyam, S. Eiger, S. Zhang, T. Liu, T. Yacovone, T. Liechty, U. Kalra, U. Evci, V. Misra, V. Roseberry, V. Feinberg, V. Kolesnikov, W. Han, W. Kwon, X. Chen, Y. Chow, Y. Zhu, Z. Wei, Z. Egyed, V. Cotruta, M. Giang, P. Kirk, A. Rao, K. Black, N. Babar, J. Lo, E. Moreira, L. G. Martins, O. Sanseviero, L. Gonzalez, Z. Gleicher, T. Warkentin, V. Mirrokni, E. Senter, E. Collins, J. Barral, Z. Ghahramani, R. Hadsell, Y. Matias, D. Sculley, S. Petrov, N. Fiedel, N. Shazeer, O. Vinyals, J. Dean, D. Hassabis, K. Kavukcuoglu, C. Farabet, E. Buchatskaya, J. Alayrac, R. Anil, Dmitry, Lepikhin, S. Borgeaud, O. Bachem, A. Joulin, A. Andreev, C. Hardin, R. Dadashi, and L. Hussenot (2025)
Gemma 3 technical report.
External Links: 2503.19786,
[Link](https://arxiv.org/abs/2503.19786)
Cited by: [§4.1](#S4.SS1.p2.1).
- R. Thomas, L. Zahran, E. Choi, A. Potti, M. Goldblum, and A. Pal (2025)
Hidden no more: attacking and defending private third-party LLM inference.
In Proceedings of the 42nd International Conference on Machine Learning (ICML),
External Links: 2505.18332,
[Link](https://arxiv.org/abs/2505.18332)
Cited by: [§5](#S5.SS0.SSS0.Px2.p2.1).
- Y. Wen, N. Jain, J. Kirchenbauer, M. Goldblum, J. Geiping, and T. Goldstein (2023)
Hard prompts made easy: gradient-based discrete optimization for prompt tuning and discovery.
External Links: 2302.03668,
[Link](https://arxiv.org/abs/2302.03668)
Cited by: [§E.1](#A5.SS1.SSS0.Px3.p1.2),
[§4.2](#S4.SS2.p1.2),
[§5](#S5.SS0.SSS0.Px2.p2.1).
- Z. Yang, Z. Dai, R. Salakhutdinov, and W. W. Cohen (2018)
Breaking the softmax bottleneck: a high-rank rnn language model.
In International Conference on Learning Representations (ICLR),
External Links: [Link](https://arxiv.org/abs/1711.03953)
Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1).
- C. Yun, S. Bhojanapalli, A. S. Rawat, S. Reddi, and S. Kumar (2020)
Are transformers universal approximators of sequence-to-sequence functions?.
In International Conference on Learning Representations,
External Links: [Link](https://openreview.net/forum?id=ByxRM0Ntvr)
Cited by: [Appendix F](#A6.SS0.SSS0.Px1.p2.4).
- C. Zhang, J. X. Morris, and V. Shmatikov (2024)
Extracting prompts by inverting llm outputs.
External Links: 2405.15012,
[Link](https://arxiv.org/abs/2405.15012)
Cited by: [§5](#S5.SS0.SSS0.Px2.p1.1).

## Appendix Overview

## Appendix A Preliminaries

This section fixes the notation used throughout the main paper and the appendix ([subsection A.1](#A1.SS1)), and it introduces *real-analyticity* as the organizing theme ([subsection A.2](#A1.SS2)). We first review the vector-space notion and its basic closure/composition properties ([subsubsection A.2.1](#A1.SS2.SSS1)), together with a zero-set principle used in measure-zero arguments. We then extend these ideas to maps between matrix spaces ([subsubsection A.2.2](#A1.SS2.SSS2)) via vectorization/matricization and note that analyticity is preserved under matrix compositions. To streamline later proofs, we summarize real-analytic building blocks commonly used in transformer layers–polynomials, exponential/logarithm, softmax, row normalization, matrix products, Hadamard scaling, and stacking ([subsubsection A.2.3](#A1.SS2.SSS3)). Finally, in [subsection A.3](#A1.SS3), we collect differential and topological tools–Fréchet derivatives and the Hessian, standard facts on $\mathbb{R}^{p}$, the inverse function theorem, and pushforwards/absolute continuity–which we use for local invertibility and absolute-continuity arguments. Readers already comfortable with these topics can skim now and return to specific subsections as needed.

### A.1 Notation

For arbitrary $T\in\mathbb{N}$, we write $[T]=\{1,2,\ldots,T\}$ to denote the set of positive integers up to $T$. Additionally, we denote the strictly positive real numbers as $\mathbb{R}^{+}=(0,\infty)$ and the non-negative real numbers as $\mathbb{R}^{+}_{0}=[0,\infty)$. Similarly, we let $\mathbb{N}_{0}=\mathbb{N}\cup\{0\}$.

Discrete sets are denoted by uppercase calligraphic letters $\mathcal{V}$, and a sequence of length $K$ is denoted by lowercase letters: $\mathrm{s}=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{K}\rangle\in\mathcal{V}^{K}$. We write $|\mathrm{s}|=K$ to denote the length of the sequence. The set of non-empty sequences of length at most $K$ is denoted as $\mathcal{V}^{\leq K}=\bigcup_{k=1}^{K}\mathcal{V}^{k}$. Non-discrete sets are denoted by uppercase calligraphic bold-face letters $\bm{\mathcal{B}}$.

###### Remark 1 .

We will often refer to a discrete set $\mathcal{V}$ as the vocabulary and to an element $\mathrm{s}\in\mathcal{V}^{\leq K}$ as an input, context, or prompt.

Matrices (vectors) are denoted by uppercase (lowercase) bold-face letters: $\mathbf{X}\in\mathbb{R}^{d_{1}\times d_{2}}$ ($\mathbf{x}\in\mathbb{R}^{d}$). For vectors and matrices, we frequently use standard norms and common matrix operations. The Hadamard and Kronecker products are defined following Kolda and Bader (2009):

- •
$p$-norm: For a vector $\mathbf{x}\in\mathbb{R}^{d}$, the $\ell_{p}$ norm is defined as
$\|\mathbf{x}\|_{p}=\left(\sum_{i=1}^{d}|\mathbf{x}_{i}|^{p}\right)^{\tfrac{1}{p}}.$
- •
Frobenius norm: For a matrix $\mathbf{X}\in\mathbb{R}^{d_{1}\times d_{2}}$, the Frobenius norm is defined as
$\|\mathbf{X}\|_{\mathrm{F}}=\sqrt{\operatorname{tr}(\mathbf{X}\mathbf{X}^{\top})}=\sqrt{\sum_{i=1}^{d_{1}}\sum_{j=1}^{d_{2}}\mathbf{X}_{ij}^{2}}.$
- •
Hadamard product: The Hadamard (element-wise) product is defined for vectors and matrices of the same shape:
$\displaystyle(\mathbf{x}\odot\mathbf{y})_{i}$
$\displaystyle=\mathbf{x}_{i}\mathbf{y}_{i},\quad$
$\displaystyle\text{for all }i\in[d],$
$\displaystyle(\mathbf{X}\odot\mathbf{Y})_{ij}$
$\displaystyle=\mathbf{X}_{ij}\mathbf{Y}_{ij},\quad$
$\displaystyle\text{for all }i\in[d_{1}],\,j\in[d_{2}],$
where $\mathbf{x},\mathbf{y}\in\mathbb{R}^{d}$ and $\mathbf{X},\mathbf{Y}\in\mathbb{R}^{d_{1}\times d_{2}}$.
- •
Kronecker product: The Kronecker product of $\mathbf{X}\in\mathbb{R}^{d_{1}\times d_{2}}$ and $\mathbf{Z}\in\mathbb{R}^{d_{3}\times d_{4}}$ is denoted $\mathbf{X}\otimes\mathbf{Z}$ and defined blockwise as
$\mathbf{X}\otimes\mathbf{Z}=\begin{bmatrix}\mathbf{X}_{11}\mathbf{Z}&\cdots&\mathbf{X}_{1d_{2}}\mathbf{Z}\\
\vdots&\ddots&\vdots\\
\mathbf{X}_{d_{1}1}\mathbf{Z}&\cdots&\mathbf{X}_{d_{1}d_{2}}\mathbf{Z}\end{bmatrix}\in\mathbb{R}^{(d_{1}d_{3})\times(d_{2}d_{4})}.$

We denote the all-zeros matrix of size $m\times n$ as $\mathbf{0}_{m\times n}$, and the all-zeros vector of length $m$ as $\mathbf{0}_{m}$. Similarly, we write $\mathbf{1}_{m}$ for the all-ones vector of length $m$, and $\mathbf{I}_{m}$ (or $\mathbf{I}_{m\times m}$ when dimensions must be explicit) for the $m\times m$ identity matrix.

Let $f:\mathcal{V}^{\leq K}\times\mathbb{R}^{p}\to\mathbb{R}^{d}$ be a function over a finite vocabulary $\mathcal{V}$ and $K\in\mathbb{N}$. We refer to $f$ as the model, to its first argument as the input sequence, and to its second argument as the parameters.

###### Remark 2 .

Throughout our analysis, we assume a finite set of possible input sequences, reflecting the practical limitations and design choices of modern LLMs, specifically the bounded context length.

###### Remark 3 .

We take the codomain of the model to be $\mathbb{R}^{d}$, corresponding to the space of token embeddings. This allows us to study how the final embedding (typically used to compute next-token probabilities) depends on both the input sequence and the model parameters.

###### Remark 1 .

###### Remark 2 .

###### Remark 3 .

### A.2 Real-Analyticity

We now introduce the central notion for our analysis: real-analyticity.
In its standard form, real-analyticity is defined for functions
$f:\bm{\mathcal{U}}\to\mathbb{R}^{n}$, where $\bm{\mathcal{U}}\subseteq\mathbb{R}^{m}$ is an open set.
Since the transformer architecture is naturally expressed in terms of matrices, it will be convenient to extend this notion to maps of the form $f:\mathbb{R}^{m\times n}\to\mathbb{R}^{a\times b}$.

Multi-index notation.
We use multi-index notation for both vectors and matrices.

*Vector case.*
Let $\bm{\alpha}=(\alpha_{1},\ldots,\alpha_{m})^{\top}\in\mathbb{N}_{0}^{m}$ and $\mathbf{x},\mathbf{y}\in\mathbb{R}^{m}$. Define:

$$ $|\bm{\alpha}|=\sum_{j=1}^{m}\alpha_{j},\qquad\bm{\alpha}!=\prod_{j=1}^{m}\alpha_{j}!,\qquad(\mathbf{x}-\mathbf{y})^{\bm{\alpha}}=\prod_{j=1}^{m}(\mathbf{x}_{j}-\mathbf{y}_{j})^{\alpha_{j}}.$ $$

*Matrix case.*
Let $\mathbf{A}=(\alpha_{uv})\in\mathbb{N}_{0}^{m\times n}$ and $\mathbf{X},\mathbf{Y}\in\mathbb{R}^{m\times n}$. Define:

$$ $|\mathbf{A}|=\sum_{u=1}^{m}\sum_{v=1}^{n}\alpha_{uv},\qquad\mathbf{A}!=\prod_{u=1}^{m}\prod_{v=1}^{n}\alpha_{uv}!,\qquad(\mathbf{X}-\mathbf{Y})^{\mathbf{A}}=\prod_{u=1}^{m}\prod_{v=1}^{n}(\mathbf{X}_{uv}-\mathbf{Y}_{uv})^{\alpha_{uv}}.$ $$

Given an open set $\bm{\mathcal{U}}\subseteq\mathbb{R}^{m}$ and a map $f:\bm{\mathcal{U}}\to\mathbb{R}$, we write

$$ $\mathbf{d}^{\bm{\alpha}}f(\mathbf{x})\;:=\;\frac{\partial^{|\bm{\alpha}|}f}{\partial\mathbf{x}_{1}^{\alpha_{1}}\cdots\partial\mathbf{x}_{m}^{\alpha_{m}}}(\mathbf{x})$ $$

for the mixed partial derivative (when it exists). Unless stated otherwise, we assume $f\in C^{\infty}(\bm{\mathcal{U}})$, so $\mathbf{d}^{\bm{\alpha}}f$ exists and is continuous for all $\bm{\alpha}\in\mathbb{N}_{0}^{m}$; for vector-valued maps $f=(f_{1},\ldots,f_{n})$ the operator $\mathbf{d}^{\bm{\alpha}}$ acts componentwise. We also use the convention $\mathbf{d}^{\mathbf{0}}f=f$.

#### A.2.1 Real-Analytic Functions with Vector Inputs

We begin with the standard vector-space definition and its basic algebraic properties. These are the building blocks from which all later analyticity arguments are assembled.

###### Definition A.1 (Real-analytic functions, Lewis 2014 , Definition 1.1.3 ) .

Let $\bm{\mathcal{U}}\subseteq\mathbb{R}^{m}$ be open. A function
$f:\bm{\mathcal{U}}\to\mathbb{R}$ is
real-analytic on $\bm{\mathcal{U}}$ if, for every $\mathbf{y}\in\bm{\mathcal{U}}$, there exist
coefficients $\{c_{\bm{\alpha}}\in\mathbb{R}\}_{\bm{\alpha}\in\mathbb{N}_{0}^{m}}$
and $r>0$ such that

$$ $f(\mathbf{x})=\sum_{\bm{\alpha}\in\mathbb{N}_{0}^{m}}c_{\bm{\alpha}}\,(\mathbf{x}-\mathbf{y})^{\bm{\alpha}}$ $$

for all $\mathbf{x}\in\bm{\mathcal{U}}$ with $\|\mathbf{x}-\mathbf{y}\|_{2}<r$. The set of real-analytic functions on $\bm{\mathcal{U}}$
is denoted by $C^{\omega}(\bm{\mathcal{U}})$.

A map $f:\bm{\mathcal{U}}\to\mathbb{R}^{n}$ is real-analytic on $\bm{\mathcal{U}}$ if each of its components
$f_{1},\dots,f_{n}:\bm{\mathcal{U}}\to\mathbb{R}$ is real-analytic.
The set of such maps is denoted $C^{\omega}(\bm{\mathcal{U}}\,;\,\mathbb{R}^{n})$.

###### Remark 4 .

To establish real-analyticity of a vector-valued mapping (e.g., an MLP, attention mechanism, or LayerNorm), it suffices to prove real-analyticity of each scalar component.

###### Proposition A.1 (Closure properties, Lewis 2014 , Proposition 1.2.1 ) .

Let $f,g:\mathbb{R}^{m}\to\mathbb{R}$ be real-analytic maps. Then, the following hold:

- 1.
Addition: $f+g\in C^{\omega}(\mathbb{R}^{m})$.
- 2.
Product: $fg\in C^{\omega}(\mathbb{R}^{m})$.
- 3.
Quotient: If $g(\mathbf{x})\neq 0$ for all $\mathbf{x}\in\mathbb{R}^{m}$, then $f/g\in C^{\omega}(\mathbb{R}^{m})$.

###### Proposition A.2 (Composition, Lewis 2014 , Proposition 1.2.2 ) .

Let $f:\mathbb{R}^{m}\to\mathbb{R}^{n}$ and $g:\mathbb{R}^{n}\to\mathbb{R}^{k}$ be real-analytic maps. Then, the composition $g\circ f:\mathbb{R}^{m}\to\mathbb{R}^{k}$ is real-analytic.

###### Remark 5 .

For simplicity, we do not state the closure properties in their most general form, where $f$ and $g$ may be defined on different open subsets of $\mathbb{R}^{m}$.
This avoids additional notation involving intersections of domains.
Since every function of interest in our later analysis is defined on the whole space $\mathbb{R}^{m}$, this restriction entails no loss of generality.

###### Theorem A.1 (Zero sets of nontrivial real-analytic maps Mityagin 2015 ) .

Let $\bm{\mathcal{U}}\subseteq\mathbb{R}^{m}$ be connected and open, and let $f\in C^{\omega}(\bm{\mathcal{U}}\,;\,\mathbb{R}^{n})$.
If $f\not\equiv\mathbf{0}_{n}$, then its zero set

$$ $Z(f)\;:=\;f^{-1}(\{\mathbf{0}_{n}\})\;=\;\{\mathbf{x}\in\bm{\mathcal{U}}:f(\mathbf{x})=\mathbf{0}_{n}\}$ $$

has Lebesgue measure zero in $\mathbb{R}^{m}$ (i.e. $\mathrm{Leb}_{m}\big(Z(f)\big)=0$).
Equivalently, if there exists $\mathbf{x}\in\bm{\mathcal{U}}$ with $f(\mathbf{x})\neq\mathbf{0}_{n}$, then $\mathrm{Leb}_{m}\big(f^{-1}(\{\mathbf{0}_{n}\})\big)=0$.

###### Remark 6 .

The result in Mityagin (2015) is stated for scalar-valued maps $f:\bm{\mathcal{U}}\to\mathbb{R}$. The extension to vector-valued maps $f=(f_{1},\ldots,f_{n}):\bm{\mathcal{U}}\to\mathbb{R}^{n}$ is immediate: the zero set of $f$ is the intersection of the zero sets of its scalar components,

$$ $Z(f)=\bigcap_{i=1}^{n}Z(f_{i}),$ $$

and if $f\not\equiv\mathbf{0}_{n}$, then at least one component $f_{j}\not\equiv 0$, so $Z(f)\subseteq Z(f_{j})$, which has measure zero by the scalar case.

###### Definition A.1 (Real-analytic functions, Lewis 2014 , Definition 1.1.3 ) .

###### Remark 4 .

###### Proposition A.1 (Closure properties, Lewis 2014 , Proposition 1.2.1 ) .

###### Proposition A.2 (Composition, Lewis 2014 , Proposition 1.2.2 ) .

###### Remark 5 .

###### Theorem A.1 (Zero sets of nontrivial real-analytic maps Mityagin 2015 ) .

###### Remark 6 .

#### A.2.2 Real-Analytic Functions with Matrix Inputs

Since transformer layers operate on matrices (e.g., $\mathbf{X}\in\mathbb{R}^{T\times d}$), we need to extend real-analyticity from vector spaces to matrix spaces. The key tool is the vectorization operator, which lets us reduce matrix-analytic questions to the vector case treated above.

###### Definition A.2 (Real-analyticity on matrix spaces) .

Let $\bm{\mathcal{U}}\subseteq\mathbb{R}^{m\times n}$ be open. A function $f:\bm{\mathcal{U}}\to\mathbb{R}$ is real-analytic on $\bm{\mathcal{U}}$ if, for every $\mathbf{Y}\in\bm{\mathcal{U}}$, there exist coefficients $\{c_{\mathbf{A}}\in\mathbb{R}\}_{\mathbf{A}\in\mathbb{N}_{0}^{m\times n}}$ and $r>0$ such that

$$ $f(\mathbf{X})=\sum_{\mathbf{A}\in\mathbb{N}_{0}^{m\times n}}c_{\mathbf{A}}(\mathbf{X}-\mathbf{Y})^{\mathbf{A}}$ $$

for all $\mathbf{X}\in\bm{\mathcal{U}}$ with $\|\mathbf{X}-\mathbf{Y}\|_{\mathrm{F}}<r$.

A map $f:\bm{\mathcal{U}}\to\mathbb{R}^{a\times b}$ is real-analytic on $\bm{\mathcal{U}}$ if each of its components $f_{ij}:\bm{\mathcal{U}}\to\mathbb{R}$ is real-analytic. The set of such maps is denoted $C^{\omega}(\bm{\mathcal{U}}\,;\,\mathbb{R}^{a\times b})$.

###### Remark 7 .

In the special case where $n=b=1$, the domain and codomain reduce to $\mathbb{R}^{m}$ and $\mathbb{R}^{a}$, respectively. Then [Definition A.2](#A1.Thmdefinition2) recovers [Definition A.1](#A1.Thmdefinition1). Thus, [Definition A.2](#A1.Thmdefinition2) generalizes real-analyticity to functions between matrix spaces.

###### Definition A.3 (Vectorization and matricization Operators) .

Let $\mathrm{vec}_{m,n}:\mathbb{R}^{m\times n}\to\mathbb{R}^{mn}$ denote the standard vectorization operator, which stacks the columns of a matrix into a single column vector (Henderson and Searle, 1981).

We also define the corresponding matricization operator $\mathrm{mat}_{m,n}:\mathbb{R}^{mn}\to\mathbb{R}^{m\times n}$. As shown in Chacón and Duong 2020, the vectorization and matricization operators are mutual inverses:

$$ $\displaystyle\mathrm{mat}_{m,n}\big(\mathrm{vec}_{m,n}(\mathbf{X})\big)$ $\displaystyle=\mathbf{X}\quad\forall\,\mathbf{X}\in\mathbb{R}^{m\times n}$ (7) $\displaystyle\mathrm{vec}_{m,n}\big(\mathrm{mat}_{m,n}(\mathbf{x})\big)$ $\displaystyle=\mathbf{x}\quad\;\forall\,\mathbf{x}\in\mathbb{R}^{mn}$ (8) $$

Furthermore, if $\mathbf{x}\in\mathbb{R}^{mn}$ and $\mathbf{X}\in\mathbb{R}^{m\times n}$ are related by vectorization and matricization, i.e., $\mathbf{x}=\mathrm{vec}_{m,n}(\mathbf{X})$ and $\mathbf{X}=\mathrm{mat}_{m,n}(\mathbf{x})$, then their norms coincide:

$$ $\|\mathbf{x}\|_{2}=\|\mathbf{X}\|_{\mathrm{F}}.$ $$

###### Definition A.4 (Vectorized Form of Function) .

Let $\bm{\mathcal{U}}\subseteq\mathbb{R}^{m\times n}$ be open and $\tilde{\bm{\mathcal{U}}}=\mathrm{vec}_{m,n}(\bm{\mathcal{U}})$ (also open since $\mathrm{vec}$ is a linear homeomorphism). We denote the vectorized form of a function $f:\bm{\mathcal{U}}\to\mathbb{R}^{a\times b}$ as

$$ $\tilde{f}:=\mathrm{vec}_{a,b}\circ f\circ\mathrm{mat}_{m,n}:\tilde{\bm{\mathcal{U}}}\to\mathbb{R}^{ab}.$ $$

Equivalently, for all $\mathbf{X}\in\bm{\mathcal{U}}$:

$$ $f(\mathbf{X})=\mathrm{mat}_{a,b}\bigg(\tilde{f}\big(\mathrm{vec}_{m,n}(\mathbf{X})\big)\bigg)$ (9) $$

###### Lemma A.1 (Equivalence real-analyticity) .

Let $\bm{\mathcal{U}}\subseteq\mathbb{R}^{m\times n}$ be open, $\tilde{\bm{\mathcal{U}}}=\mathrm{vec}_{m,n}(\bm{\mathcal{U}})$, and let $f:\bm{\mathcal{U}}\to\mathbb{R}^{a\times b}$ with its vectorized form $\tilde{f}:\tilde{\bm{\mathcal{U}}}\to\mathbb{R}^{ab}$.

Fix $\mathbf{Y}\in\bm{\mathcal{U}}$ and set $\mathbf{y}=\mathrm{vec}_{m,n}(\mathbf{Y})\in\tilde{\bm{\mathcal{U}}}$.
Then the following are equivalent:

- 1.
$f$ is real-analytic at $\mathbf{Y}$ (in the sense of [Definition A.2](#A1.Thmdefinition2)).
- 2.
$\tilde{f}$ is real-analytic at $\mathbf{y}$ (in the sense of [Definition A.1](#A1.Thmdefinition1)).

###### Proof.

We begin by establishing the correspondence between matrix and vector indices in $\mathbb{R}^{k\times\ell}$ and $\mathbb{R}^{k\ell}$. For $s\in[k\ell]$, define:

$$ $\displaystyle u(s)$ $\displaystyle:=1+(s-1)\bmod k$ (row index) $\displaystyle v(s)$ $\displaystyle:=1+\left\lfloor\frac{s-1}{k}\right\rfloor$ (column index) $$

Then $(u(s),v(s))\in[k]\times[\ell]$ gives the matrix coordinates corresponding to the $s$th entry of the vectorization. Conversely, for $(u,v)\in[k]\times[\ell]$, define:

$$ $s(u,v):=u+(v-1)k\in[k\ell]$ $$

to recover the linear index.

When clear from context, we omit arguments and simply write $u$, $v$, or $s$ for readability.

Let $\mathbf{X},\mathbf{Y}\in\mathbb{R}^{m\times n}$, with vectorizations $\mathbf{x}=\mathrm{vec}_{m,n}(\mathbf{X})$ and $\mathbf{y}=\mathrm{vec}_{m,n}(\mathbf{Y})$. For a vector multi-index $\bm{\alpha}\in\mathbb{N}_{0}^{mn}$, define the corresponding matrix multi-index $\mathbf{A}_{\bm{\alpha}}:=\mathrm{mat}_{m,n}(\bm{\alpha})$, so that:

$$ $(\mathbf{x}-\mathbf{y})^{\bm{\alpha}}=\prod_{s=1}^{mn}(\mathbf{x}_{s}-\mathbf{y}_{s})^{\bm{\alpha}_{s}}=\prod_{u=1}^{m}\prod_{v=1}^{n}(\mathbf{X}_{uv}-\mathbf{Y}_{uv})^{(\mathbf{A}_{\bm{\alpha}})_{uv}}=(\mathbf{X}-\mathbf{Y})^{\mathbf{A}_{\bm{\alpha}}}.$ (10) $$

Similarly, for a matrix multi-index $\mathbf{A}\in\mathbb{N}_{0}^{m\times n}$, define the corresponding vector multi-index $\bm{\alpha}_{\mathbf{A}}:=\mathrm{vec}_{m,n}(\mathbf{A})$, giving:

$$ $(\mathbf{X}-\mathbf{Y})^{\mathbf{A}}=\prod_{u=1}^{m}\prod_{v=1}^{n}(\mathbf{X}_{uv}-\mathbf{Y}_{uv})^{\mathbf{A}_{uv}}=\prod_{s=1}^{mn}(\mathbf{x}_{s}-\mathbf{y}_{s})^{(\bm{\alpha}_{\mathbf{A}})_{s}}=(\mathbf{x}-\mathbf{y})^{\bm{\alpha}_{\mathbf{A}}}.$ (11) $$

Now let $\mathbf{M}\in\bm{\mathcal{U}}$, and let $\mathbf{m}=\mathrm{vec}_{m,n}(\mathbf{M})\in\tilde{\bm{\mathcal{U}}}$. By definition of the vectorization,

$$ $f_{uv}(\mathbf{M})=\tilde{f}_{s}(\mathbf{m}),\quad\text{where }s=s(u,v).$ $$

This coordinate-wise correspondence underlies the equivalence stated in the lemma.

($\Rightarrow$) Assume $f$ is real-analytic at $\mathbf{Y}$.
Then by [Definition A.2](#A1.Thmdefinition2), there exists $r>0$ and, for each $(u,v)$, coefficients $\{c^{(uv)}_{\mathbf{A}}\}_{\mathbf{A}\in\mathbb{N}_{0}^{m\times n}}$ such that:

$$ $f_{uv}(\mathbf{X})=\sum_{\mathbf{A}\in\mathbb{N}_{0}^{m\times n}}c^{(uv)}_{\mathbf{A}}(\mathbf{X}-\mathbf{Y})^{\mathbf{A}},\qquad\forall\,\mathbf{X}\in\bm{\mathcal{U}}:\|\mathbf{X}-\mathbf{Y}\|_{\mathrm{F}}<r.$ (12) $$

Using [Equation 11](#A1.E11), each component $\tilde{f}_{s}$ of $\tilde{f}$ can be expressed as:

$$ $\tilde{f}_{s}(\mathbf{x})=\sum_{\bm{\alpha}\in\mathbb{N}_{0}^{mn}}\tilde{c}^{(s)}_{\bm{\alpha}}(\mathbf{x}-\mathbf{y})^{\bm{\alpha}},\quad\text{where }\tilde{c}^{(s)}_{\bm{\alpha}_{\mathbf{A}}}:=c^{(u(s),v(s))}_{\mathbf{A}}.$ $$

This series converges for all $\mathbf{x}\in\tilde{\bm{\mathcal{U}}}$ with $\|\mathbf{x}-\mathbf{y}\|_{2}=\|\mathbf{X}-\mathbf{Y}\|_{\mathrm{F}}<r$. Hence, each scalar component of $\tilde{f}$ has a convergent power series at $\mathbf{y}$, proving that $\tilde{f}$ is real-analytic there.

($\Leftarrow$) The reverse direction follows by symmetry: assume $\tilde{f}$ is real-analytic at $\mathbf{y}$, write the expansion at $\mathbf{y}$ using definition [Definition A.1](#A1.Thmdefinition1), and repeat the argument using [Equation 10](#A1.E10) to construct component-wise expansions for $f_{uv}$ at $\mathbf{Y}$.
∎

###### Remark 8 .

Consider the function $f=\mathrm{vec}_{m,n}:\mathbb{R}^{m\times n}\to\mathbb{R}^{mn\times 1}$, which vectorizes an $m\times n$ matrix by stacking its columns. Its corresponding vectorized form is

$$ $\tilde{f}(\mathbf{x})=(\mathrm{vec}_{mn,1}\circ\mathrm{vec}_{m,n}\circ\mathrm{mat}_{m,n})(\mathbf{x})=\mathrm{vec}_{mn,1}(\mathbf{x})=\mathbf{x},$ $$

since $\mathbf{x}\in\mathbb{R}^{mn}$ is already a column vector . This composition yields the identity map on $\mathbb{R}^{mn}$, which is clearly real analytic. Therefore, by [Lemma A.1](#A1.Thmlemma1), both $\mathrm{vec}_{m,n}$ is real analytic, and similarly, so is $\mathrm{mat}_{m,n}$. It is now evident that the composition of two matrix-valued real-analytic function is real-analytic, and we will prove it.

###### Proposition A.3 (Composition on matrix spaces is real-analytic) .

Suppose $f:\mathbb{R}^{m\times n}\to\mathbb{R}^{a\times b}$ and
$g:\mathbb{R}^{a\times b}\to\mathbb{R}^{p\times q}$ are real-analytic
(in the sense of [Definition A.2](#A1.Thmdefinition2)). Then
$g\circ f:\mathbb{R}^{m\times n}\to\mathbb{R}^{p\times q}$ is real-analytic.

###### Proof.

Consider the vectorized forms

$$ $\tilde{f}:=\mathrm{vec}_{a,b}\circ f\circ\mathrm{mat}_{m,n}:\mathbb{R}^{mn}\to\mathbb{R}^{ab},\qquad\tilde{g}:=\mathrm{vec}_{p,q}\circ g\circ\mathrm{mat}_{a,b}:\mathbb{R}^{ab}\to\mathbb{R}^{pq}.$ $$

By [Lemma A.1](#A1.Thmlemma1), $f$ is real-analytic iff $\tilde{f}$ is, and $g$ is real-analytic
iff $\tilde{g}$ is. Hence $\tilde{f}$ and $\tilde{g}$ are real-analytic maps between Euclidean spaces.

The vectorized form of the composition is

$$ $\widetilde{g\circ f}=\mathrm{vec}_{p,q}\circ(g\circ f)\circ\mathrm{mat}_{m,n}=\underbrace{\big(\mathrm{vec}_{p,q}\circ g\circ\mathrm{mat}_{a,b}\big)}_{\tilde{g}}\circ\underbrace{\big(\mathrm{vec}_{a,b}\circ f\circ\mathrm{mat}_{m,n}\big)}_{\tilde{f}}=\tilde{g}\circ\tilde{f},$ $$

where we inserted the identity $(\mathrm{mat}_{a,b}\circ\mathrm{vec}_{a,b})(\mathbf{X})=\mathbf{X}$.
By the vector-space composition property ([Proposition A.2](#A1.Thmproposition2)), $\tilde{g}\circ\tilde{f}$ is real-analytic on $\mathbb{R}^{mn}$.
Applying [Lemma A.1](#A1.Thmlemma1) once more, we get that $g\circ f$ is real-analytic.
∎

###### Definition A.2 (Real-analyticity on matrix spaces) .

###### Remark 7 .

###### Definition A.3 (Vectorization and matricization Operators) .

###### Definition A.4 (Vectorized Form of Function) .

###### Lemma A.1 (Equivalence real-analyticity) .

###### Proof.

###### Remark 8 .

###### Proposition A.3 (Composition on matrix spaces is real-analytic) .

###### Proof.

#### A.2.3 Real Analyticity of Common Components

We now catalogue the specific functions that appear inside transformer layers, proving each one is real-analytic. These building blocks—polynomials, exponentials, softmax, row normalization, matrix products, Hadamard scaling, and stacking—will be composed in [Appendix B](#A2) to establish the real-analyticity of the full model. Throughout, all maps are defined on $\mathbb{R}^{m\times n}$ (or an open subset thereof), so [Definition A.2](#A1.Thmdefinition2) applies.

###### Proposition A.4 (Polynomials are real-analytic) .

Let $p:\mathbb{R}^{m}\to\mathbb{R}$ be a polynomial in the coordinates of $\mathbf{x}\in\mathbb{R}^{m}$, i.e.,
$p(\mathbf{x})=\sum_{|\bm{\alpha}|\leq d}a_{\bm{\alpha}}\,\mathbf{x}^{\bm{\alpha}}$
for some $d\in\mathbb{N}_{0}$ and coefficients $a_{\bm{\alpha}}\in\mathbb{R}$. Then $p\in C^{\omega}(\mathbb{R}^{m})$.

###### Proof.

Polynomials are $C^{\infty}$, and $\mathbf{d}^{\bm{\alpha}}p\equiv 0$ whenever $|\bm{\alpha}|>d$. Hence the Taylor expansion of $p$ at any $\mathbf{y}\in\mathbb{R}^{m}$ truncates:

$$ $p(\mathbf{x})\;=\;\sum_{|\bm{\alpha}|\leq d}\frac{\mathbf{d}^{\bm{\alpha}}p(\mathbf{y})}{\bm{\alpha}!}\,(\mathbf{x}-\mathbf{y})^{\bm{\alpha}},$ $$

which holds for all $\mathbf{x}\in\mathbb{R}^{m}$ (radius $r=+\infty$). Therefore $p$ is real-analytic.
∎

###### Proposition A.5 (The exponential is real-analytic) .

The map $\exp:\mathbb{R}\to(0,\infty)$ is real-analytic on $\mathbb{R}$.

###### Proof.

Define $E(x):=\sum_{k=0}^{\infty}\frac{x^{k}}{k!}$. By the ratio test this power series has infinite radius of convergence, hence converges absolutely for all $x\in\mathbb{R}$. Standard results on power series imply that $E$ is $C^{\infty}$ on $\mathbb{R}$ and can be differentiated termwise within its radius of convergence; in particular, for every $j\in\mathbb{N}_{0}$,

$$ $E^{(j)}(x)=\sum_{k=j}^{\infty}\frac{k(k-1)\cdots(k-j+1)}{k!}\,x^{k-j}=\sum_{r=0}^{\infty}\frac{x^{r}}{r!}=E(x).$ $$

Fix $y\in\mathbb{R}$. Taylor’s theorem for power series then yields

$$ $E(x)=\sum_{j=0}^{\infty}\frac{E^{(j)}(y)}{j!}(x-y)^{j}=E(y)\sum_{j=0}^{\infty}\frac{(x-y)^{j}}{j!},$ $$

which is a convergent power series in $x-y$ with infinite radius of convergence. Hence $E$ is real-analytic at every $y\in\mathbb{R}$. As $E$ is the usual exponential function defined by its power series, $\exp$ is real-analytic on $\mathbb{R}$.
∎

###### Proposition A.6 (The logarithm is real-analytic) .

The map $\log:(0,\infty)\to\mathbb{R}$ is real-analytic on $(0,\infty)$.

###### Proof.

For brevity, we present only a proof sketch;

The exponential map $\exp:\mathbb{R}\to(0,\infty)$ is real-analytic with $\exp^{\prime}(y)\neq 0$ for all $y$.
By the real-analytic inverse function theorem (see Krantz and Parks 2002, Thm. 2.3.1),
its local inverse $\log$ is real-analytic on $(0,\infty)$.
∎

The next three results handle the attention-specific operations: softmax, row normalization (used in the causal projection form), and entrywise matrix polynomials.

###### Proposition A.7 (Softmax is real-analytic) .

The map $\mathrm{softmax}:\mathbb{R}^{m}\to\mathbb{R}^{m}$ with components

$$ $\mathrm{softmax}_{i}(\mathbf{x})\;=\;\frac{e^{\mathbf{x}_{i}}}{\sum_{j=1}^{m}e^{\mathbf{x}_{j}}},\qquad i=1,\dots,m,$ $$

is real-analytic on $\mathbb{R}^{m}$.

###### Proof.

Fix $i$. The numerator $\mathbf{x}\mapsto e^{\mathbf{x}_{i}}$ is the composition of the coordinate projection $\pi_{i}(\mathbf{x})=\mathbf{x}_{i}$ (a linear, hence real-analytic, map) with $\exp$; by [Proposition A.5](#A1.Thmproposition5) and the composition rule in [Proposition A.1](#A1.Thmproposition1), it is real-analytic. The denominator

$$ $H(\mathbf{x})=\sum_{j=1}^{m}e^{\mathbf{x}_{j}}$ $$

is a finite sum of real-analytic functions, hence real-analytic. Moreover, $H(\mathbf{x})>0$ for all $\mathbf{x}\in\mathbb{R}^{m}$ because $e^{x_{j}}>0$. Therefore, by the quotient rule in [Proposition A.1](#A1.Thmproposition1), the map

$$ $\mathbf{x}\mapsto\frac{e^{\mathbf{x}_{i}}}{H(\mathbf{x})}$ $$

is real-analytic on $\mathbb{R}^{m}$. Since this holds for each $i=1,\dots,m$, the vector-valued map $\mathrm{softmax}$ is real-analytic.
∎

###### Proposition A.8 (Row normalization is real-analytic on positive row-sum domain) .

Let

$$ $\bm{\mathcal{D}}_{T}:=\big\{\mathbf{Y}\in\mathbb{R}^{T\times T}:\mathbf{Y}\mathbf{1}_{T}\in(0,\infty)^{T}\big\}.$ $$

Define $\mathrm{RN}(\mathbf{Y})=\mathrm{diag}(\mathbf{Y}\mathbf{1}_{T})^{-1}\mathbf{Y}$ on $\bm{\mathcal{D}}_{T}$.
Then $\mathrm{RN}:\bm{\mathcal{D}}_{T}\to\mathbb{R}^{T\times T}$ is real-analytic (in the sense of [Definition A.2](#A1.Thmdefinition2)).

###### Proof.

The map $\mathbf{Y}\mapsto\mathbf{s}:=\mathbf{Y}\mathbf{1}_{T}$ is linear, hence real-analytic.
On $(0,\infty)^{T}$, the entrywise reciprocal $\mathbf{s}\mapsto\mathbf{s}^{\odot(-1)}$ is real-analytic (componentwise $t\mapsto 1/t$).
The map $\mathbf{s}\mapsto\mathrm{diag}(\mathbf{s})$ is linear. Matrix multiplication $(\mathbf{A},\mathbf{Y})\mapsto\mathbf{A}\mathbf{Y}$ is real-analytic ([Proposition A.10](#A1.Thmproposition10)). Composing these gives $\mathrm{RN}(\mathbf{Y})=\mathrm{diag}(\mathbf{Y}\mathbf{1}_{T})^{-1}\mathbf{Y}$ real-analytic on the open set $\bm{\mathcal{D}}_{T}$.
∎

###### Proposition A.9 (Entrywise matrix polynomials are real-analytic) .

Fix $m,n\in\mathbb{N}$. For coefficients $\{c_{\mathbf{A}}\in\mathbb{R}\}_{\mathbf{A}\in\mathbb{N}_{0}^{m\times n}}$
and some $d\in\mathbb{N}_{0}$, define the function $p:\mathbb{R}^{m\times n}\to\mathbb{R}$ by:

$$ $p(\mathbf{X})=\sum_{|\mathbf{A}|\leq d}c_{\mathbf{A}}\,\mathbf{X}^{\mathbf{A}},$ (13) $$

where $\mathbf{X}^{\mathbf{A}}=\prod_{u=1}^{m}\prod_{v=1}^{n}\mathbf{X}_{uv}^{\mathbf{A}_{uv}}$ as defined in the multi-index notation above. Then $p$ is real-analytic on $\mathbb{R}^{m\times n}$ (in the sense of [Definition A.2](#A1.Thmdefinition2)).

Moreover, if $f:\mathbb{R}^{m\times n}\to\mathbb{R}^{a\times b}$ has component functions $f_{ij}$ of the form [Equation 13](#A1.E13), then $f$ is real-analytic.

###### Proof.

Consider the vectorized form
$\tilde{p}:=p\circ\mathrm{mat}_{m,n}:\mathbb{R}^{mn}\to\mathbb{R}$.
Using the coordinate identification from equation [11](#A1.E11)-equation [10](#A1.E10), each monomial satisfies

$$ $\big(\mathrm{mat}_{m,n}(\mathbf{x})\big)^{\mathbf{A}}=\mathbf{x}^{\bm{\alpha}_{\mathbf{A}}},$ $$

where $\bm{\alpha}_{\mathbf{A}}=\mathrm{vec}_{m,n}(\mathbf{A})$. Hence:

$$ $\tilde{p}(\mathbf{x})=\sum_{|\mathbf{A}|\leq d}c_{\mathbf{A}}\,\mathbf{x}^{\bm{\alpha}_{\mathbf{A}}},$ $$

which is a standard multivariate polynomial in $\mathbf{x}\in\mathbb{R}^{mn}$. By [Proposition A.4](#A1.Thmproposition4), such functions are real-analytic on all of $\mathbb{R}^{mn}$, so $\tilde{p}\in C^{\omega}(\mathbb{R}^{mn})$.
By [Lemma A.1](#A1.Thmlemma1), this implies $p$ is real-analytic on $\mathbb{R}^{m\times n}$.

For the second claim, observe that if each $f_{ij}$ is a scalar polynomial of the form [Equation 13](#A1.E13), then each $f_{ij}$ is real-analytic by the argument above. Hence, by [Definition A.2](#A1.Thmdefinition2), $f$ is real analytic.
∎

Finally, we record the algebraic operations—matrix multiplication, Hadamard scaling, concatenation, and noncommutative matrix polynomials—that allow us to compose the above primitives into full transformer layers.

###### Proposition A.10 (Matrix product of real-analytic factors) .

Let the functions $f:\mathbb{R}^{m\times n}\to\mathbb{R}^{p\times r}$ and $g:\mathbb{R}^{m\times n}\to\mathbb{R}^{r\times q}$ be real-analytic. Then, $h:\mathbb{R}^{m\times n}\to\mathbb{R}^{p\times q}$ defined as $h(\mathbf{X})=f(\mathbf{X})\,g(\mathbf{X})$, is real-analytic on $\mathbb{R}^{m\times n}$.

###### Proof.

For each $(i,j)\in[p]\times[q]$, it holds that $h_{ij}(\mathbf{X})=\sum_{k=1}^{r}f_{ik}(\mathbf{X})\,g_{kj}(\mathbf{X})$.

Each factor $f_{ik}$ and $g_{kj}$ is a real-analytic scalar map by assumption; their product is real-analytic by [Proposition A.1](#A1.Thmproposition1), and a finite sum of real-analytic functions is real-analytic. Thus every $h_{ij}$ is real-analytic, hence $h$ is real-analytic.
∎

###### Proposition A.11 (Hadamard (element-wise) scaling) .

Let $\mathbf{A}\in\mathbb{R}^{m\times n}$ be a fixed matrix. Then, the map $f:\mathbb{R}^{m\times n}\to\mathbb{R}^{m\times n}$ defined as $f(X)=\mathbf{A}\odot\mathbf{X}$ is real-analytic on $\mathbb{R}^{m\times n}$.

###### Proof.

Componentwise, $(\mathbf{A}\odot\mathbf{X})_{ij}=\mathbf{A}_{ij}\,\mathbf{X}_{ij}$ is a product of a constant and a coordinate function, hence a polynomial (degree $\leq 1$) and thus real-analytic.
∎

###### Proposition A.12 (Concatenation/stacking of real-analytic blocks) .

Let $f_{\ell}:\mathbb{R}^{m\times n}\to\mathbb{R}^{p\times q_{\ell}}$ be real-analytic for $\ell\in[L]$. The horizontal concatenation operation $g:\mathbb{R}^{m\times n}\to\mathbb{R}^{p\times(q_{1}+\cdots+q_{L})}$, defined as:

$$ $g(\mathbf{X})=\big[\,f_{1}(\mathbf{X})\;\;f_{2}(\mathbf{X})\;\;\cdots\;\;f_{L}(\mathbf{X})\,\big]$ $$

is real-analytic. Likewise, if $f_{\ell}:\mathbb{R}^{m\times n}\to\mathbb{R}^{p_{\ell}\times q}$ are real-analytic, then the vertical stacking operation $h:\mathbb{R}^{m\times n}\to\mathbb{R}^{(p_{1}+\cdots+p_{L})\times q}$, defined as:

$$ $h(\mathbf{X})=\big[\,f_{1}(\mathbf{X})^{\top}\;\;f_{2}(\mathbf{X})^{\top}\;\;\cdots\;\;f_{L}(\mathbf{X})^{\top}\,\big]^{\top}$ $$

is real-analytic.

###### Proof.

Each scalar component of $g$ (respectively $h$) is exactly one scalar component of some $f_{\ell}$, hence real-analytic. Therefore $g$ and $h$ are real-analytic by definition [Definition A.2](#A1.Thmdefinition2).
∎

###### Proposition A.13 (Noncommutative matrix polynomials are real-analytic) .

Let $n,p,q\in\mathbb{N}$, let $\mathbf{X}\in\mathbb{R}^{n\times n}$, and fix coefficient matrices
$\mathbf{A}_{k}\in\mathbb{R}^{p\times n}$ and $\mathbf{B}_{k}\in\mathbb{R}^{n\times q}$ for $k=0,\ldots,d$.
Define

$$ $f(\mathbf{X})\;:=\;\sum_{k=0}^{d}\mathbf{A}_{k}\,\mathbf{X}^{k}\,\mathbf{B}_{k}\;\in\;\mathbb{R}^{p\times q},\qquad\mathbf{X}^{0}:=\mathbf{I}_{n},\;\;\mathbf{X}^{k+1}:=\mathbf{X}^{k}\mathbf{X}.$ $$

Then $f$ is real analytic in the sense of [Definition A.2](#A1.Thmdefinition2).

###### Proof.

The identity map $\mathbf{X}\mapsto\mathbf{X}$ is linear, hence a degree-$1$ entrywise polynomial; by [Proposition A.9](#A1.Thmproposition9) it is real-analytic.
Assume $\mathbf{X}\mapsto\mathbf{X}^{k}$ is real-analytic. With $f(\mathbf{X})=\mathbf{X}^{k}$ and $g(\mathbf{X})=\mathbf{X}$, [Proposition A.10](#A1.Thmproposition10) yields $\mathbf{X}^{k+1}=f(\mathbf{X})g(\mathbf{X})$ real-analytic; by induction, all powers $\mathbf{X}\mapsto\mathbf{X}^{k}$ are real-analytic.

For each $k$, left/right multiplication by fixed matrices preserves real-analyticity via [Proposition A.10](#A1.Thmproposition10): since the constant maps $\mathbf{X}\mapsto\mathbf{A}_{k}$ and $\mathbf{X}\mapsto\mathbf{B}_{k}$ are real-analytic (components are constant polynomials), the composition
$\mathbf{X}\mapsto\mathbf{A}_{k}\,\mathbf{X}^{k}\,\mathbf{B}_{k}$ is real-analytic.
Finally, $f$ is a finite sum of real-analytic maps, hence real-analytic by closure under addition (apply [Proposition A.1](#A1.Thmproposition1) componentwise).
∎

###### Remark 9 .

We highlight several standard constructions that yield real-analytic maps, omitting proofs for brevity:

- •
Affine and bilinear maps. Functions of the form $\mathbf{X}\mapsto\mathbf{A}\mathbf{X}\mathbf{B}+\mathbf{C}$ are real-analytic, as they are obtained via matrix multiplication and addition of constant matrices ([Proposition A.10](#A1.Thmproposition10), [Proposition A.1](#A1.Thmproposition1)).
- •
Algebraic expressions in $\mathbf{X}$. Any expression constructed from $\mathbf{X}$ using finitely many additions and matrix multiplications with fixed coefficient matrices, e.g.
$\mathbf{A}_{0}+\mathbf{A}_{1}\mathbf{X}\mathbf{B}_{1}+\mathbf{A}_{2}\mathbf{X}\mathbf{B}_{2}\mathbf{X}\mathbf{C}_{2}$-
defines a real-analytic map. This follows from repeated application of [Proposition A.10](#A1.Thmproposition10) and closure under addition.
- •
Scalar polynomial invariants. Coordinate functions $\mathbf{X}_{ij}$, the trace $\mathrm{tr}(\mathbf{X})$, all principal and non-principal minors, and the determinant $\det(\mathbf{X})$ are scalar polynomials in the entries of $\mathbf{X}$, and hence real-analytic by [Proposition A.9](#A1.Thmproposition9).

###### Proposition A.4 (Polynomials are real-analytic) .

###### Proof.

###### Proposition A.5 (The exponential is real-analytic) .

###### Proof.

###### Proposition A.6 (The logarithm is real-analytic) .

###### Proof.

###### Proposition A.7 (Softmax is real-analytic) .

###### Proof.

###### Proposition A.8 (Row normalization is real-analytic on positive row-sum domain) .

###### Proof.

###### Proposition A.9 (Entrywise matrix polynomials are real-analytic) .

###### Proof.

###### Proposition A.10 (Matrix product of real-analytic factors) .

###### Proof.

###### Proposition A.11 (Hadamard (element-wise) scaling) .

###### Proof.

###### Proposition A.12 (Concatenation/stacking of real-analytic blocks) .

###### Proof.

###### Proposition A.13 (Noncommutative matrix polynomials are real-analytic) .

###### Proof.

###### Remark 9 .

### A.3 Differential, Measure-Theoretic, and Topological Tools

This subsection collects the minimal calculus, measure, and topology we will use later. In finite dimensions, Fréchet derivatives let us speak uniformly about Jacobians and Hessians; basic Euclidean topology lets us control neighborhoods and compactness; the inverse function theorem gives local invertibility; and pushforwards/absolute continuity formalize how distributions transform under measurable maps.

###### Definition A.5 (Fréchet derivative (Luenberger, 1997 , §7.2-§7.3) ) .

Let $\bm{\mathcal{U}}\subseteq\mathbb{R}^{m}$ open, and consider a function
$f:\bm{\mathcal{U}}\to\mathbb{R}^{n}$. We say that $f$ is *Fréchet differentiable* at $\mathbf{x}\in\bm{\mathcal{U}}$ if there exists a bounded linear map
$\mathbf{A}:\mathbb{R}^{m}\to\mathbb{R}^{n}$ such that

$$ $\lim_{\|\mathbf{h}\|_{2}\to 0}\frac{\|f(\mathbf{x}+\mathbf{h})-f(\mathbf{x})-\mathbf{A}\mathbf{h}\|_{2}}{\|\mathbf{h}\|_{2}}=0.$ $$

The unique operator $\mathbf{A}$ is denoted by $Df(\mathbf{x})$ and called the (Fréchet) derivative of $f$ at $\mathbf{x}$.

###### Definition A.6 (Second Fréchet derivative (Magnus and Neudecker, 2019 , Ch. 18) ) .

Let $\bm{\mathcal{U}}\subseteq\mathbb{R}^{m}$ open, and consider a function $f:\bm{\mathcal{U}}\to\mathbb{R}^{n}$. Suppose $f$ is Fréchet differentiable at $\mathbf{x}$. The *second Fréchet derivative* of $f$ at $\mathbf{x}$ is the bounded bilinear map
$D^{2}f(\mathbf{x}):\mathbb{R}^{m}\times\mathbb{R}^{m}\to\mathbb{R}^{n}$ defined as:

$$ $D^{2}f(\mathbf{x})[\mathbf{h},\mathbf{k}]:=\lim_{t\to 0}\frac{Df(\mathbf{x}+t\mathbf{h})[\mathbf{k}]-Df(\mathbf{x})[\mathbf{k}]}{t}.$ $$

###### Proposition A.14 (Connection to the Hessian) .

If $f:\bm{\mathcal{U}}\to\mathbb{R}$ is $C^{2}$, then $D^{2}f(\mathbf{x})$ is symmetric (Arora et al., 2021, Thm. 5.1) and can represented by the Hessian matrix $\nabla^{2}f(\mathbf{x})$:

$$ $D^{2}f(\mathbf{x})[\mathbf{h},\mathbf{k}]\;=\;\mathbf{h}^{\top}\big(\nabla^{2}f(\mathbf{x})\big)\,\mathbf{k},$ $$

as noted in Magnus and Neudecker 2019, Ch. 18.

###### Definition A.5 (Fréchet derivative (Luenberger, 1997 , §7.2-§7.3) ) .

###### Definition A.6 (Second Fréchet derivative (Magnus and Neudecker, 2019 , Ch. 18) ) .

###### Proposition A.14 (Connection to the Hessian) .

##### Euclidean topology.

The following standard definitions and facts about $\mathbb{R}^{p}$ are used in the local-to-global measure argument in [Appendix C](#A3).

###### Definition A.7 (Closure of a set in $\mathbb{R}^{p}$ ) .

Let $\bm{\mathcal{U}}\subseteq\mathbb{R}^{p}$. The closure of $\bm{\mathcal{U}}$, denoted $\overline{\bm{\mathcal{U}}}$, is the smallest closed subset of $\mathbb{R}^{p}$ containing $\bm{\mathcal{U}}$.

###### Definition A.8 (Euclidean balls in $\mathbb{R}^{p}$ ) .

Fix $p\in\mathbb{N}$ and equip $\mathbb{R}^{p}$ with the Euclidean norm $\|\cdot\|_{2}$.
For $\mathbf{x}\in\mathbb{R}^{p}$ and $r>0$ we define:

$$ $\displaystyle B(\mathbf{x},r)$ $\displaystyle:=\{\,\mathbf{y}\in\mathbb{R}^{p}:\|\mathbf{y}-\mathbf{x}\|_{2}<r\,\}$ $\displaystyle\overline{B}(\mathbf{x},r)$ $\displaystyle:=\{\,\mathbf{y}\in\mathbb{R}^{p}:\|\mathbf{y}-\mathbf{x}\|_{2}\leq r\,\}$ $$

In $\mathbb{R}^{p}$ with the Euclidean topology one has $\overline{B}(\mathbf{x},r)=\overline{B(\mathbf{x},r)}$, i.e. the closed ball equals the topological closure of the open ball.

###### Definition A.9 (Second-countable subspace of $\mathbb{R}^{p}$ (Munkres, 2000 , §30) ) .

Let $\bm{\mathcal{X}}\subseteq\mathbb{R}^{p}$ be equipped with the subspace topology
$\tau_{\bm{\mathcal{X}}}:=\{\bm{\mathcal{U}}\cap\bm{\mathcal{X}}:\bm{\mathcal{U}}\text{ open in }\mathbb{R}^{p}\}$.
We say $\bm{\mathcal{X}}$ is second-countable if there exists a countable family
$\mathcal{F}\subseteq\tau_{X}$ such that every $\bm{\mathcal{O}}\in\tau_{\bm{\mathcal{X}}}$ is a union of members of $\mathcal{F}$.
Equivalently, the countable family

$$ $\mathcal{F}_{\mathbb{Q}}\;:=\;\big\{\,B(\mathbf{x},r)\cap\bm{\mathcal{X}}:\mathbf{x}\in\mathbb{Q}^{p},r\in\mathbb{Q}_{>0}\,\big\},$ $$

is a basis for $\tau_{\bm{\mathcal{X}}}$.

###### Proposition A.15 (Standard facts for $\mathbb{R}^{p}$ ) .

Fix $p\in\mathbb{N}$. The following hold:

- 1.
Hausdorff (Aitken, 2020, Prop. 18): $\mathbb{R}^{p}$ with its Euclidean metric is Hausdorff.
- 2.
Heine-Borel (Munkres, 2000, Thm. 27.3): A subset of $\mathbb{R}^{p}$ is compact iff it is closed and bounded; in particular, each closed Euclidean ball $\overline{B}(x,r)$ is compact.
- 3.
Second countability (Munkres, 2000, §13 and Thm. 30.2) : $\mathbb{R}$ has a countable base (intervals with rational endpoints); hence $\mathbb{R}^{p}$, being a finite product of second-countable spaces, is second-countable. Moreover, subspaces of second-countable spaces are second-countable.
- 4.
Lindelöf consequence(Munkres, 2000, Thm. 30.3(a)): Every second-countable space is Lindelöf; consequently, every open cover of any subspace of $\mathbb{R}^{p}$ admits a countable subcover.
- 5.
Local compactness of $\mathbb{R}^{p}$(Munkres, 2000, Thm. 29.2): For any $\mathbf{x}\in\mathbb{R}^{p}$ and open neighborhood $\bm{\mathcal{W}}\ni\mathbf{x}$, there exists $\varepsilon>0$ with $\overline{B}(\mathbf{x},\varepsilon)\subseteq\bm{\mathcal{W}}$, and $\overline{B}(\mathbf{x},\varepsilon)$ is compact by Heine-Borel; hence $\mathbb{R}^{p}$ is locally compact. Furthermore, in a Hausdorff space, local compactness is equivalent to shrinking neighborhoods with compact closures: for every neighborhood $\bm{\mathcal{W}}\ni\mathbf{x}$ there exists an open $\bm{\mathcal{V}}$ with $\mathbf{x}\in\bm{\mathcal{V}}\subseteq\overline{\bm{\mathcal{V}}}\subseteq\bm{\mathcal{W}}$ and $\overline{\bm{\mathcal{V}}}$ compact.

###### Definition A.7 (Closure of a set in ℝ p \mathbb{R}^{p} ) .

###### Definition A.8 (Euclidean balls in ℝ p \mathbb{R}^{p} ) .

###### Definition A.9 (Second-countable subspace of ℝ p \mathbb{R}^{p} (Munkres, 2000 , §30) ) .

###### Proposition A.15 (Standard facts for ℝ p \mathbb{R}^{p} ) .

##### Invertibility and measure transport.

The inverse function theorem and the pushforward formalism are the two tools that connect local diffeomorphism charts to global measure-preservation statements.

###### Definition A.10 ( $C^{k}$ diffeomorphism Spivak 1971 , Ch. 5 ) .

Let $U,V\subseteq\mathbb{R}^{p}$ be open sets and let $k\in\mathbb{N}\cup\{\infty\}$.
A map $f:U\to V$ is a $C^{k}$ diffeomorphism if:

- 1.
$f$ is bijective;
- 2.
$f$ is $C^{k}$ (all partial derivatives up to order $k$ exist and are continuous);
- 3.
the inverse map $f^{-1}:V\to U$ is $C^{k}$.

When $k=1$ we simply say *diffeomorphism*.
Equivalently, a $C^{k}$ diffeomorphism is a bijective $C^{k}$ map whose inverse is also $C^{k}$.

###### Theorem A.2 (Inverse Function Theorem Rudin 1976 , Thm. 9.24 ) .

Let $\bm{\mathcal{U}}\subset\mathbb{R}^{p}$ be open and $f:\bm{\mathcal{U}}\to\mathbb{R}^{p}$ be $C^{1}$.
Suppose $\mathbf{a}\in\bm{\mathcal{U}}$ satisfies $\det Df(\mathbf{a})\neq 0$. Then there exist open sets
$\bm{\mathcal{U}}_{0}\subset\bm{\mathcal{U}}$ with $\mathbf{a}\in\bm{\mathcal{U}}_{0}$ and $\bm{\mathcal{V}}_{0}\subset\mathbb{R}^{p}$ with $f(\mathbf{a})\in\bm{\mathcal{V}}_{0}$ such that

$$ $f\big|_{\bm{\mathcal{U}}_{0}}:\bm{\mathcal{U}}_{0}\to\bm{\mathcal{V}}_{0}$ $$

is a $C^{1}$-diffeomorphism. Moreover, the inverse $f^{-1}:\bm{\mathcal{V}}_{0}\to\bm{\mathcal{U}}_{0}$ is $C^{1}$ and

$$ $D\big(f^{-1}\big)(f(\mathbf{x}))\;=\;\big(Df(\mathbf{x})\big)^{-1}\qquad\forall\,\mathbf{x}\in\bm{\mathcal{U}}_{0}.$ $$

###### Remark 10 .

In [Theorem A.2](#A1.Thmtheorem2) we assume $f:\bm{\mathcal{U}}\subseteq\mathbb{R}^{p}\to\mathbb{R}^{p}$, so the Jacobian $Df(\mathbf{a})$ is a $p\times p$ (square) matrix. In this setting,

$$ $\det Df(\mathbf{a})\neq 0\quad\Longleftrightarrow\quad Df(\mathbf{a})\;\text{is invertible},$ $$

and this is exactly the hypothesis that yields a local $C^{1}$ inverse.

###### Definition A.11 (Pushforward and absolute continuity (Folland, 1999 , §3.2) ) .

Consider a Borel-measurable map $T:\mathbb{R}^{p}\to\mathbb{R}^{p}$ and let $\mu$ be a Borel measure on $\mathbb{R}^{p}$.
The pushforward measure $T_{\#}\mu$ is the Borel measure on $\mathbb{R}^{p}$ defined by

$$ $T_{\#}\mu(\bm{\mathcal{U}})\;:=\;\mu\left(T^{-1}(\bm{\mathcal{U}})\right),\qquad\bm{\mathcal{U}}\in\mathcal{B}(\mathbb{R}^{p}).$ $$

If $\nu$ is another Borel measure on $\mathbb{R}^{p}$, we say $T_{\#}\mu$ is absolutely continuous with respect to $\nu$,
and write $T_{\#}\mu\ll\nu$, if for every Borel set $\bm{\mathcal{U}}\in\mathcal{B}(\mathbb{R}^{p})$:

$$ $\nu(\bm{\mathcal{U}})=0\Longrightarrow T_{\#}\mu(\bm{\mathcal{U}})=0.$ $$

In particular, for Lebesgue measure $\mathrm{Leb}_{p}$, to prove $T_{\#}\mu\ll\mathrm{Leb}_{p}$ for every $\mu\ll\mathrm{Leb}_{p}$,
it suffices to verify that

$$ $\mathrm{Leb}_{p}(\bm{\mathcal{U}})=0\ \Longrightarrow\ \mathrm{Leb}_{p}\big(T^{-1}(\bm{\mathcal{U}})\big)=0\quad\text{for all Borel }\bm{\mathcal{U}}\subseteq\mathbb{R}^{p}.$ $$

###### Definition A.10 ( C k C^{k} diffeomorphism Spivak 1971 , Ch. 5 ) .

###### Theorem A.2 (Inverse Function Theorem Rudin 1976 , Thm. 9.24 ) .

###### Remark 10 .

###### Definition A.11 (Pushforward and absolute continuity (Folland, 1999 , §3.2) ) .

## Appendix B Transformer Language Model

This appendix section gives a concise, shape-accurate specification of the decoder-only Transformer we analyze. We include it both to keep the paper self-contained and because the measure-zero arguments later hinge on architecture-dependent witnesses and exact dimension bookkeeping. We begin with token and positional embeddings ([Definition B.3](#A2.Thmdefinition3)), define self-attention and its causal variants ([Definition B.5](#A2.Thmdefinition5), [Definition B.6](#A2.Thmdefinition6), [Definition B.7](#A2.Thmdefinition7)), assemble multi-head attention, layer normalization, and an MLP into a pre-LN residual block ([Definition B.8](#A2.Thmdefinition8), [Definition B.9](#A2.Thmdefinition9), [Definition B.4](#A2.Thmdefinition4), [Definition B.11](#A2.Thmdefinition11)), stack $L$ such blocks to obtain the model ([Definition B.12](#A2.Thmdefinition12)), and conclude with the unembedding+softmax head ([Definition B.10](#A2.Thmdefinition10)), isolating the last-token representation used in downstream proofs ([Equation 29](#A2.E29)).

##### Input processing.

The first stage of the model maps a discrete token sequence into a continuous matrix representation via learned embeddings.

###### Definition B.1 (Token Embedding Layer) .

Let $\mathcal{V}$ be a vocabulary, and let $d\in\mathbb{N}$ be the embedding dimension. For any input sequence $\mathrm{s}=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{T}\rangle\in\mathcal{V}^{\leq K}$, the Token Embedding Layer is the function defined as:

$$ $\mathrm{E}(\mathrm{s})=\left(\mathbf{E}_{\mathrm{s}_{1}},\ldots,\mathbf{E}_{\mathrm{s}_{T}}\right)^{\top}\in\mathbb{R}^{T\times d},$ (14) $$

where $\mathbf{E}\in\mathbb{R}^{|\mathcal{V}|\times d}$ is a trainable embedding matrix indexed by elements of $\mathcal{V}$, and $\mathbf{E}_{\mathrm{s}_{i}}\in\mathbb{R}^{d}$ denotes the embedding vector for token $\mathrm{s}_{i}$.

This mapping is applied element-wise and is independent of the sequence length $T$.

###### Definition B.2 (Positional Embedding Layer) .

Let $\mathcal{V}$ be a vocabulary, and let $d\in\mathbb{N}$ be the embedding dimension.
For any input sequence $\mathrm{s}=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{T}\rangle\in\mathcal{V}^{\leq K}$ with $T=|\mathrm{s}|$, the (learned absolute) Positional Embedding Layer is the function defined as:

$$ $\mathrm{PE}(\mathrm{s})\;=\;\left(\mathbf{P}_{1},\ldots,\mathbf{P}_{T}\right)^{\top}\in\mathbb{R}^{T\times d},$ (15) $$

where $\mathbf{P}\in\mathbb{R}^{K\times d}$ is a trainable matrix indexed by positions $i\in[K]$, and $\mathbf{P}_{i}\in\mathbb{R}^{d}$ denotes the embedding vector for position $i$. This mapping depends only on positions (not on token identities) and returns the first $T$ rows of $\mathbf{P}$.

###### Definition B.3 (Embedding Layer) .

Let $\mathcal{V}$ be a vocabulary, $K\in\mathbb{N}$ a context bound, and $d\in\mathbb{N}$ the embedding width.
For any input sequence $\mathrm{s}=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{T}\rangle\in\mathcal{V}^{\leq K}$ with $T=|\mathrm{s}|$, define the embedding layer as the sum of the token and positional embeddings:

$$ $\mathrm{Emb}(\mathrm{s}):=\mathrm{E}(\mathrm{s})+\mathrm{PE}(\mathrm{s})=\big(\,\mathbf{E}_{\mathrm{s}_{1}}+\mathbf{P}_{1},\;\ldots,\;\mathbf{E}_{\mathrm{s}_{T}}+\mathbf{P}_{T}\,\big)^{\top}\in\mathbb{R}^{T\times d},$ (16) $$

where $\mathbf{E}\in\mathbb{R}^{|\mathcal{V}|\times d}$ is the trainable token-embedding matrix and
$\mathbf{P}\in\mathbb{R}^{K\times d}$ is the trainable positional-embedding matrix.

###### Definition B.1 (Token Embedding Layer) .

###### Definition B.2 (Positional Embedding Layer) .

###### Definition B.3 (Embedding Layer) .

##### Sub-layer modules.

The Transformer block is built from four reusable sub-layers—an MLP, (causal) self-attention, multi-head attention, and layer normalization—each defined next.

###### Definition B.4 (Multi-Layer Perceptron) .

A Multi-Layer Perceptron (MLP) with $M$ layers is a function $\mathrm{mlp}_{M}:\mathbb{R}^{d_{0}}\to\mathbb{R}^{d_{M}}$, defined recursively as:

$$ $\displaystyle\mathbf{h}^{(1)}$ $\displaystyle=\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)}$ (17) $\displaystyle\mathbf{h}^{(m)}$ $\displaystyle=\mathbf{W}^{(m)}\,\sigma\big(\mathbf{h}^{(m-1)}\big)+\mathbf{b}^{(m)},\;m\geq 2$ (18) $\displaystyle\mathrm{mlp}_{M}(\mathbf{x})$ $\displaystyle=\mathbf{h}^{(M)}$ (19) $$

where $\mathbf{x}\in\mathbb{R}^{d_{0}}$ is the input, $\{\mathbf{W}^{(m)}\in\mathbb{R}^{d_{m}\times d_{m-1}}\}_{m=1}^{M}$ and $\{\mathbf{b}^{(m)}\in\mathbb{R}^{d_{m}}\}_{m=1}^{M}$ are trainable parameters and $\sigma$ is an activation function.

###### Definition B.5 (Self-Attention) .

A Self-Attention module is a function $\bm{\eta}:\mathbb{R}^{T\times d_{\mathrm{in}}}\to\mathbb{R}^{T\times d_{\eta}}$, defined as:

$$ $\displaystyle\bm{\eta}(\mathbf{X}\,;\mathbf{Q},\mathbf{K},\mathbf{V})=\mathrm{softmax}\left(\frac{\left(\mathbf{X}\mathbf{Q}\right)\left(\mathbf{X}\mathbf{K}\right)^{\top}}{\sqrt{d_{\eta}}}\right)\mathbf{X}\mathbf{V},$ (20) $$

where $\mathbf{X}\in\mathbb{R}^{T\times d_{\mathrm{in}}}$ is the input, $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\eta}}$ are trainable parameters (query, key, and value matrices), $\mathrm{softmax}$ is applied row-wise, $d_{\eta}$ is the attention dimension (typically $d_{\eta}<d_{\mathrm{in}}$), and $T$ is the sequence length.

###### Definition B.6 (Causal Self-Attention, masked form) .

Define the “causal mask” $\mathbf{M}\in\overline{\mathbb{R}}^{T\times T}$ as:

$$ $\mathbf{M}_{ij}=\begin{cases}0,&j\leq i,\\ -\infty,&j>i\end{cases}$ $$

Then, a Causal Self-Attention module is a function $\tilde{\bm{\eta}}:\mathbb{R}^{T\times d_{\mathrm{in}}}\to\mathbb{R}^{T\times d_{\eta}}$, defined as:

$$ $\displaystyle\tilde{\bm{\eta}}(\mathbf{X}\,;\mathbf{Q},\mathbf{K},\mathbf{V})=\mathrm{softmax}\left(\frac{\left(\mathbf{X}\mathbf{Q}\right)\left(\mathbf{X}\mathbf{K}\right)^{\top}}{\sqrt{d_{\eta}}}+\mathbf{M}\right)\mathbf{X}\mathbf{V},$ (21) $$

where $\mathbf{X}\in\mathbb{R}^{T\times d_{\mathrm{in}}}$ is the input, $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\eta}}$ are trainable parameters (query, key, and value matrices), $\mathrm{softmax}$ is applied row-wise, $d_{\eta}$ is the attention dimension (typically $d_{\eta}<d_{\mathrm{in}}$), and $T$ is the sequence length.

###### Definition B.7 (Causal Self-Attention, projection form) .

Define the unit lower-triangular matrix $\mathbf{L}\in\mathbb{R}^{T\times T}$ as $\mathbf{L}_{ij}=\mathbb{I}_{\{j\leq i\}}$ and consider the row normalization operation $\mathrm{RN}:\bm{\mathcal{D}}_{T}\to\mathbb{R}^{T\times T}$ of [Proposition A.8](#A1.Thmproposition8). Then, a Causal Self-Attention module is a function $\tilde{\bm{\eta}}:\mathbb{R}^{T\times d_{\mathrm{in}}}\to\mathbb{R}^{T\times d_{\eta}}$, defined as:

$$ $\displaystyle\tilde{\bm{\eta}}(\mathbf{X}\,;\mathbf{Q},\mathbf{K},\mathbf{V})=\mathrm{RN}\left(\mathbf{L}\odot\exp{\left(\frac{\left(\mathbf{X}\mathbf{Q}\right)\left(\mathbf{X}\mathbf{K}\right)^{\top}}{\sqrt{d_{\eta}}}\right)}\right)\mathbf{X}\mathbf{V},$ (22) $$

where $\mathbf{X}\in\mathbb{R}^{T\times d_{\mathrm{in}}}$ is the input, $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\eta}}$ are trainable parameters (query, key, and value matrices), $\mathrm{RN}$ is applied row-wise, $d_{\eta}$ is the attention dimension (typically $d_{\eta}<d_{\mathrm{in}}$), and $T$ is the sequence length.

###### Remark 11 .

Consider $\mathbf{Z}=\frac{1}{\sqrt{d_{\eta}}}\left(\mathbf{X}\mathbf{Q}\right)\left(\mathbf{X}\mathbf{K}\right)^{\top}$. Since $\mathbf{L}_{ii}=1$ for all $i\in[T]$, we have that $\big[\mathbf{L}\odot\exp{\mathbf{Z}}\big]_{ii}=e^{\mathbf{Z}_{ii}}>0$, hence the row sum
$\sum_{j\leq i}e^{\mathbf{Z}_{ij}}\geq e^{\mathbf{Z}_{ii}}>0$ and $\mathrm{RN}$ is well-defined.

###### Definition B.8 (Multi-Head Self-Attention) .

A Multi-Head Self-Attention module with $H$ heads is a function $\mathrm{attn}_{H}:\mathbb{R}^{T\times d_{\mathrm{in}}}\to\mathbb{R}^{T\times d_{\mathrm{out}}}$, defined using the Self-Attention map from [Definition B.5](#A2.Thmdefinition5) or [Definition B.7](#A2.Thmdefinition7) with different parameter sets per head:

$$ $\displaystyle\bm{\eta}_{h}(\mathbf{X})$ $\displaystyle=\bm{\eta}(\mathbf{X}\,;\mathbf{Q}^{(h)},\mathbf{K}^{(h)},\mathbf{V}^{(h)}),\quad h\in[H],$ (23) $\displaystyle\mathrm{attn}_{H}(\mathbf{X})$ $\displaystyle=\big[\bm{\eta}_{1}(\mathbf{X}),\ldots,\bm{\eta}_{H}(\mathbf{X})\big]\mathbf{W}^{O},$ (24) $$

where $\{\mathbf{Q}^{(h)},\mathbf{K}^{(h)},\mathbf{V}^{(h)}\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\eta}}\}_{h=1}^{H}$ are the head-specific parameters and $\mathbf{W}^{O}\in\mathbb{R}^{Hd_{\eta}\times d_{\mathrm{out}}}$ is the output projection matrix.

###### Definition B.9 (Layer Normalization) .

Layer Normalization is a function $\mathrm{LN}:\mathbb{R}^{d}\to\mathbb{R}^{d}$, defined as:

$$ $\mathrm{LN}(\mathbf{x})=\bm{\gamma}\odot\frac{\mathbf{x}-\mu_{\mathbf{x}}\mathbf{1}_{d}}{\sqrt{\sigma_{\mathbf{x}}^{2}+\varepsilon}}+\bm{\beta},$ (25) $$

where $\mathbf{x}\in\mathbb{R}^{d}$ is the input, $\mu_{\mathbf{x}}=\frac{1}{d}\sum_{i=1}^{d}\mathbf{x}_{i}$ and $\sigma_{\mathbf{x}}^{2}=\frac{1}{d}\sum_{i=1}^{d}(\mathbf{x}_{i}-\mu_{\mathbf{x}})^{2}$ are the mean and variance of $\mathbf{x}$, vectors $\bm{\beta},\bm{\gamma}\in\mathbb{R}^{d}$ are learnable parameters, and $\varepsilon\in\mathbb{R}^{+}$ is a small constant that ensures we don’t divide by zero.

###### Definition B.10 (Unembedding Layer) .

Let $\mathcal{V}$ be a vocabulary and $d\in\mathbb{N}$ and $\mathbf{U}\in\mathbb{R}^{|\mathcal{V}|\times d}$ be a trainable projection matrix. Define the unembedding map $\mathrm{UnEmb}:\mathbb{R}^{d}\to\mathbb{R}^{|\mathcal{V}|}$ by

$$ $\mathrm{UnEmb}(\mathbf{h})\;:=\;\mathrm{softmax}\big(\,\mathbf{U}\,\mathrm{LN}(\mathbf{h})\,\big),\qquad\mathbf{h}\in\mathbb{R}^{d}.$ $$

###### Definition B.4 (Multi-Layer Perceptron) .

###### Definition B.5 (Self-Attention) .

###### Definition B.6 (Causal Self-Attention, masked form) .

###### Definition B.7 (Causal Self-Attention, projection form) .

###### Remark 11 .

###### Definition B.8 (Multi-Head Self-Attention) .

###### Definition B.9 (Layer Normalization) .

###### Definition B.10 (Unembedding Layer) .

##### Full architecture assembly.

With all sub-layers in place, we assemble them into a single pre-LN residual block, stack $L$ such blocks into the Transformer backbone, and append the unembedding head to form the complete language model.

###### Definition B.11 (Transformer Block) .

A Transformer Block consists of a composition of a Multi-Head Self-Attention layer with $H$ heads ([Definition B.8](#A2.Thmdefinition8)) and an MLP with $M$ layers ([Definition B.4](#A2.Thmdefinition4)), each preceded by layer normalization ([Definition B.9](#A2.Thmdefinition9)) and wrapped with residual connections. Given an input $\mathbf{X}\in\mathbb{R}^{T\times d}$, the output $\mathrm{TB}(\mathbf{X})\in\mathbb{R}^{T\times d}$ is computed as:

$$ $\displaystyle\mathbf{H}$ $\displaystyle=\mathbf{X}+\mathrm{attn}_{H}(\overline{\mathbf{X}})$ (26) $\displaystyle\mathrm{TB}(\mathbf{X})$ $\displaystyle=\mathbf{H}+\mathrm{mlp}_{M}(\overline{\mathbf{H}}),$ (27) $$

where $\overline{\mathbf{X}},\overline{\mathbf{H}}\in\mathbb{R}^{T\times d}$ are the results of applying layer normalization row-wise to $\mathbf{X}$ and $\mathbf{H}$, respectively, each with its own set of learnable parameters and $\mathrm{mlp}_{M}$ is applied row-wise. All sub-layer parameters are dimensioned appropriately.

###### Definition B.12 (Transformer) .

Fix $L\in\mathbb{N}$. For each $\ell\in[L]$, let
$\mathrm{TB}^{(\ell)}:\mathbb{R}^{T\times d}\to\mathbb{R}^{T\times d}$ denote a Transformer Block ([Definition B.11](#A2.Thmdefinition11)) with its own parameters.
Define the module

$$ $\mathrm{Tr}_{T}\;:=\;\mathrm{TB}^{(L)}\circ\cdots\circ\mathrm{TB}^{(1)}.$ $$

Each $\mathrm{TB}^{(\ell)}$ maps $\mathbb{R}^{T\times d}\to\mathbb{R}^{T\times d}$, so the residual additions in [Definition B.11](#A2.Thmdefinition11) are dimensionally valid at every depth.

###### Definition B.13 (Transformer Language Model) .

Let $\mathcal{V}$ denote a finite vocabulary and $K\in\mathbb{N}$ a fixed context length.
A *Transformer Language Model* with $L$ layers is the composition of an embedding layer ([Definition B.3](#A2.Thmdefinition3)), a Transformer with $L$ blocks ([Definition B.12](#A2.Thmdefinition12)), and an Unembedding Layer ([Definition B.10](#A2.Thmdefinition10)).

Formally, it is a parameterized function

$$ $f:\mathcal{V}^{\leq K}\times\mathbb{R}^{p}\;\to\;\Delta^{|\mathcal{V}|-1}$ $$

defined as follows. Without loss of generality, consider $\bm{\theta}=(\bm{\theta}_{1}\in\mathbb{R}^{p_{1}},\bm{\theta}_{2}\in\mathbb{R}^{p_{2}},\bm{\theta}_{3}\in\mathbb{R}^{p_{3}})\in\mathbb{R}^{p}$, which collects all the model parameters.

For an input sequence $\mathrm{s}=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{T}\rangle$ with $T\leq K$:

$$ $\displaystyle\mathbf{H}(\mathrm{s}\,;\,\bm{\theta})$ $\displaystyle=\mathrm{Emb}(\mathrm{s}\,;\,\bm{\theta}_{1})\quad$ (embedding) (28) $\displaystyle\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})$ $\displaystyle=\bigg(\mathrm{Tr}_{|\mathrm{s}|}\Big(\mathbf{H}(\mathrm{s}\,;\,\bm{\theta})\,;\,\bm{\theta}_{2}\Big)\bigg)_{|\mathrm{s}|}$ (last-token representation) (29) $\displaystyle f(\mathbf{s}\,;\,\bm{\theta})$ $\displaystyle=\mathrm{UnEmb}\Big(\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})\,;\,\bm{\theta}_{3}\Big)$ (next-token prediction) (30) $$

Then, the probability of the next-token being $\mathcal{V}_{i}$ is given by:

$$ $\Pr\,[\;s_{T+1}=\mathcal{V}_{i}\mid\mathrm{s}\;]\;=\;\big(f(\mathrm{s}\,;\,\bm{\theta})\big)_{i},\quad\forall i\in[|\mathcal{V}|].$ (31) $$

###### Definition B.11 (Transformer Block) .

###### Definition B.12 (Transformer) .

###### Definition B.13 (Transformer Language Model) .

##### Verification of real-analyticity.

We close this section by showing that every module defined above is jointly real-analytic in its inputs and parameters. This is the technical property that lets the measure-zero arguments in [Appendix C](#A3) go through. We first record the equivalence between the two causal-softmax formulations, then verify analyticity of the embedding layer and of each sub-layer and their compositions.

###### Proposition B.1 (Equivalence of masked and projection causal softmax) .

For any logits $\mathbf{Z}\in\mathbb{R}^{T\times T}$, let $\mathbf{M}$ and $\mathbf{L}$ be as in
Definitions [B.6](#A2.Thmdefinition6)–[B.7](#A2.Thmdefinition7). Then, row-wise,

$$ $\mathrm{softmax}(\mathbf{Z}+\mathbf{M})\;=\;\mathrm{RN}\big(\mathbf{L}\odot\exp{\mathbf{Z}}\big).$ $$

Consequently, the two definitions of the Causal Self-Attention are identical.

###### Proof.

Fix a row $i$. By the mask:

$$ $\big[\mathrm{softmax}(\mathbf{Z}+\mathbf{M})\big]_{ij}=\begin{cases}\dfrac{e^{\mathbf{Z}_{ij}}}{\sum_{k\leq i}e^{\mathbf{Z}_{ik}}},&j\leq i,\[8.0pt] 0,&j>i,\end{cases}$ $$

interpreting $-\infty$ via a limit. On the other hand, it holds that:

$$ $[\mathbf{L}\odot\exp{\mathbf{Z}}]_{ij}=\mathbb{I}_{j\leq i}\,e^{\mathbf{Z}_{ij}}.$ $$

Therefore, $\mathbf{L}\odot\exp{\mathbf{Z}}$ keeps exactly the entries with $j\leq i$.
Then, for each row, row normalization divides the kept entries by the same positive sum $\sum_{k\leq i}e^{\mathbf{Z}_{ik}}$ and leaves the others at 0, yielding the same row as above. This holds for every row $i$, proving the identity.
∎

###### Proposition B.2 (Embedding layer is real-analytic in the parameters) .

Fix a sequence $\mathrm{s}=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{T}\rangle\in\mathcal{V}^{\leq K}$ with $T=|\mathrm{s}|$.
Consider the map

$$ $(\mathbf{E},\mathbf{P})\;\longmapsto\;\mathrm{Emb}(\mathrm{s})\;=\;\mathrm{E}(\mathrm{s})+\mathrm{PE}(\mathrm{s})\;\in\;\mathbb{R}^{T\times d},\qquad\mathbf{E}\in\mathbb{R}^{|\mathcal{V}|\times d},\;\mathbf{P}\in\mathbb{R}^{K\times d}.$ $$

Then this map is real-analytic on $\mathbb{R}^{|\mathcal{V}|\times d}\times\mathbb{R}^{K\times d}$ (in the sense of [Definition A.2](#A1.Thmdefinition2)).

###### Proof.

Let $S_{\mathrm{s}}\in\{0,1\}^{T\times|\mathcal{V}|}$ select rows $\{\mathrm{s}_{i}\}_{i=1}^{T}$, and $R_{T}\in\{0,1\}^{T\times K}$ select the first $T$ rows. Then

$$ $\mathrm{E}(\mathrm{s})=S_{\mathrm{s}}\mathbf{E},\qquad\mathrm{PE}(\mathrm{s})=R_{T}\mathbf{P},\qquad\mathrm{Emb}(\mathrm{s})=S_{\mathrm{s}}\mathbf{E}+R_{T}\mathbf{P}.$ $$

Each map $(\mathbf{E},\mathbf{P})\mapsto S_{\mathrm{s}}\mathbf{E}$ and $(\mathbf{E},\mathbf{P})\mapsto R_{T}\mathbf{P}$ is a matrix product of a *constant* matrix with the variable (*constant maps are real-analytic* as degree-0 polynomials by [Proposition A.9](#A1.Thmproposition9); the product is real-analytic by [Proposition A.10](#A1.Thmproposition10)). Their sum is real-analytic by closure under addition ([Proposition A.1](#A1.Thmproposition1)). Hence $(\mathbf{E},\mathbf{P})\mapsto\mathrm{Emb}(\mathrm{s})$ is real-analytic.
∎

###### Proposition B.3 (Joint real-analyticity of core modules and stacks) .

Assume the pointwise activation $\sigma:\mathbb{R}\to\mathbb{R}$ used in the MLP is real-analytic (e.g., $\tanh$, $\mathrm{GELU}$).
Fix $T\in[K]$. For notational convenience define the parameter tuples

$$ $\Theta_{\mathrm{attn}}:=\Big(\{\mathbf{Q}^{(h)},\mathbf{K}^{(h)},\mathbf{V}^{(h)}\}_{h=1}^{H},\;\mathbf{W}^{O}\Big),\quad\Theta_{\mathrm{LN}}^{(1)}:=(\bm{\gamma}^{(1)},\bm{\beta}^{(1)}),\quad\Theta_{\mathrm{LN}}^{(2)}:=(\bm{\gamma}^{(2)},\bm{\beta}^{(2)}),$ $$

$$ $\Theta_{\mathrm{mlp}}:=\big(\{\mathbf{W}^{(m)},\mathbf{b}^{(m)}\}_{m=1}^{M}\big),\qquad\Theta_{\mathrm{TB}}:=\big(\Theta_{\mathrm{attn}},\Theta_{\mathrm{LN}}^{(1)},\Theta_{\mathrm{LN}}^{(2)},\Theta_{\mathrm{mlp}}\big),\quad\Theta_{\mathrm{Tr},T}:=\big(\Theta_{\mathrm{TB}}^{(1)},\ldots,\Theta_{\mathrm{TB}}^{(L)}\big).$ $$

Then the following maps are jointly real-analytic in their inputs and parameters:

- 1.
MLP.
$(\mathbf{x},\Theta_{\mathrm{mlp}})\mapsto\mathrm{mlp}_{M}(\mathbf{x})$ is real-analytic: each affine layer
$(\mathbf{W},\mathbf{b},\mathbf{x})\mapsto\mathbf{W}\mathbf{x}+\mathbf{b}$ is a matrix product plus addition ([Proposition A.10](#A1.Thmproposition10) and [Proposition A.1](#A1.Thmproposition1)); the activation $\sigma$ is real-analytic by assumption, and composition preserves real-analyticity ([Proposition A.2](#A1.Thmproposition2)). Iteration over $M$ layers is repeated composition ([Proposition A.2](#A1.Thmproposition2)).
- 2.
Layer Normalization.
$(\mathbf{x},\bm{\gamma},\bm{\beta})\mapsto\mathrm{LN}(\mathbf{x})=\bm{\gamma}\odot\frac{\mathbf{x}-\mu_{\mathbf{x}}}{\sqrt{\sigma^{2}_{\mathbf{x}}+\varepsilon}}+\bm{\beta}$ is real-analytic:
$\mu_{\mathbf{x}}$ and $\sigma^{2}_{\mathbf{x}}$ are (entrywise) polynomials in $\mathbf{x}$ ([Proposition A.9](#A1.Thmproposition9));
$g(\mathbf{x})=\sigma^{2}_{\mathbf{x}}+\varepsilon$ satisfies $g(\mathbf{x})>0$ (definition of $\varepsilon>0$), and the scalar map $h(t)=t^{-1/2}$ is real-analytic on $(0,\infty)$ (classical binomial series). Thus $h\circ g$ is real-analytic ([Proposition A.2](#A1.Thmproposition2)); division by $g^{1/2}$ is a quotient by a nonvanishing real-analytic function ([Proposition A.1](#A1.Thmproposition1)); Hadamard scaling by $\bm{\gamma}$ and addition of $\bm{\beta}$ preserve real-analyticity ([Proposition A.11](#A1.Thmproposition11) and [Proposition A.1](#A1.Thmproposition1)). Row-wise application is handled by stacking ([Proposition A.12](#A1.Thmproposition12)) and the vectorization equivalence ([Lemma A.1](#A1.Thmlemma1)).
- 3.
Unembedding.
$(\mathbf{h},\mathbf{U},\bm{\gamma},\bm{\beta})\mapsto\mathrm{softmax}\big(\mathbf{U}\,\mathrm{LN}(\mathbf{h})\big)$ is real-analytic:
$\mathrm{LN}$ is real-analytic by (2); multiplication by $\mathbf{U}$ is real-analytic ([Proposition A.10](#A1.Thmproposition10));
$\mathrm{softmax}$ is real-analytic ([Proposition A.7](#A1.Thmproposition7)); the overall map is a composition ([Proposition A.2](#A1.Thmproposition2)) and stacking across coordinates ([Proposition A.12](#A1.Thmproposition12)).
- 4.
Self-Attention (vanilla or causal) and Multi-Head.
Let $\mathbf{Z}=\frac{1}{\sqrt{d_{\eta}}}\left(\mathbf{X}\mathbf{Q}\right)\left(\mathbf{X}\mathbf{K}\right)^{\top}$.
*(a) Vanilla SA:*
$(\mathbf{X},\mathbf{Q},\mathbf{K},\mathbf{V})\mapsto\mathrm{softmax}(\mathbf{Z})\mathbf{X}\mathbf{V}$ is real-analytic by:
matrix products ([Proposition A.10](#A1.Thmproposition10)), scaling, row-wise softmax ([Proposition A.7](#A1.Thmproposition7) with stacking, [Proposition A.12](#A1.Thmproposition12), and [Lemma A.1](#A1.Thmlemma1)), and a final matrix product.
*(b) Causal SA (projection form):*
With $\mathbf{L}$ unit lower-triangular and using [Definition B.7](#A2.Thmdefinition7),
$(\mathbf{X},\mathbf{Q},\mathbf{K},\mathbf{V})\longmapsto\mathrm{RN}\big(\mathbf{L}\odot\exp{\mathbf{Z}}\big)\mathbf{X}\mathbf{V}$
is real-analytic:
$\exp$ is real-analytic ([Proposition A.5](#A1.Thmproposition5));
Hadamard scaling by fixed $\mathbf{L}$ is real-analytic ([Proposition A.11](#A1.Thmproposition11));
by [Remark 11](#Thmremark11), every row of $\mathbf{L}\odot\exp(\mathbf{Z})$ sums to a strictly positive value (the diagonal term), so the argument lies in the domain $\bm{\mathcal{D}}_{T}$ of [Proposition A.8](#A1.Thmproposition8); hence $\mathrm{RN}$ is real-analytic there; the final multiplication by $\mathbf{X}\mathbf{V}$ is real-analytic ([Proposition A.10](#A1.Thmproposition10)).
Therefore, each *single* attention head is real-analytic whether it is vanilla or causal (projection).
For Multi-Head Self-Attention ([Definition B.8](#A2.Thmdefinition8)), horizontal concatenation across heads is real-analytic ([Proposition A.12](#A1.Thmproposition12)), and the output projection by $\mathbf{W}^{O}$ is a matrix product ([Proposition A.10](#A1.Thmproposition10)). Hence $(\mathbf{X},\Theta_{\mathrm{attn}})\mapsto\mathrm{attn}_{H}(\mathbf{X})$ is real-analytic regardless of which attention variant each head uses.
- 5.
Transformer Block (fixed $T$).
$(\mathbf{X},\Theta_{\mathrm{TB}})\mapsto\mathrm{TB}(\mathbf{X})\in\mathbb{R}^{T\times d}$ is real-analytic:
apply LN row-wise to get $\overline{\mathbf{X}}$ (item 2 with stacking, [Proposition A.12](#A1.Thmproposition12), and [Lemma A.1](#A1.Thmlemma1));
apply attention (item 4) to $\overline{\mathbf{X}}$; add the residual (closure under addition, [Proposition A.1](#A1.Thmproposition1));
apply LN row-wise to get $\overline{\mathbf{H}}$ (item 2 with stacking and [Lemma A.1](#A1.Thmlemma1));
apply the row-wise MLP (item 1 with stacking, [Proposition A.12](#A1.Thmproposition12));
add the residual again ([Proposition A.1](#A1.Thmproposition1)).
All intermediate matrix multiplications use [Proposition A.10](#A1.Thmproposition10), and the overall structure is a composition ([Proposition A.3](#A1.Thmproposition3) via [Lemma A.1](#A1.Thmlemma1)).
- 6.
Transformer (fixed $T$).
$(\mathbf{X},\Theta_{\mathrm{Tr},T})\mapsto\mathrm{Tr}_{T}(\mathbf{X})=\mathrm{TB}^{(L)}\circ\cdots\circ\mathrm{TB}^{(1)}(\mathbf{X})$ is a composition of real-analytic maps from (5), hence real-analytic by [Proposition A.3](#A1.Thmproposition3).

All statements extend from vector-valued to matrix-valued, row-wise applications via [Proposition A.12](#A1.Thmproposition12) and [Lemma A.1](#A1.Thmlemma1), and every sum/product/quotient/composition step above invokes [Proposition A.1](#A1.Thmproposition1), [Proposition A.10](#A1.Thmproposition10), and [Proposition A.3](#A1.Thmproposition3) as indicated.

###### Proposition B.1 (Equivalence of masked and projection causal softmax) .

###### Proof.

###### Proposition B.2 (Embedding layer is real-analytic in the parameters) .

###### Proof.

###### Proposition B.3 (Joint real-analyticity of core modules and stacks) .

## Appendix C Almost Sure Injectivity

This section establishes a foundational structural result: for causal Transformer Language Models with standard architectural widths and at least one attention head per block, the final hidden state at the last token is almost surely injective with respect to the input sequence, assuming the model parameters are drawn from any absolutely continuous distribution at initialization. Crucially, we show this injectivity is preserved after any finite number of gradient descent (GD) updates.

We organize the section in two parts; (i) Measure-zero collisions via real-analyticity and a witness construction and (ii) Preservation of absolute continuity under gradient descent. Each piece builds toward the main theorem, which asserts that under mild width and head assumptions, the Transformer map from input sequences to last-token representations is injective almost surely, even after multiple rounds of training. The main theorem follows.

###### Assumption C.1 (Minimum Embedding Dimension) .

We assume the embedding dimension satisfies $d\geq 4$ and $d_{\eta}\geq 1$. Furthermore, we assume that each transformer block has at least one attention head.
These conditions are trivially satisfied in practice: for modern large language models, embedding dimensions are typically in the hundreds or thousands, and each layer has multiple attention heads, so the assumptions impose no practical restrictions on the models under consideration.

###### Theorem C.1 (Finite-horizon a.s. injectivity under GD) .

Fix a finite vocabulary $\mathcal{V}$, a context bound $K\in\mathbb{N}$, a time horizon $T\in\mathbb{N}$, and consider the causal Transformer Language Model (TLM) of [Definition B.13](#A2.Thmdefinition13) under [Assumption C.1](#A3.Thmassumption1).
Let $\left\{\left(\mathrm{s}_{t}\in\mathcal{V}^{\leq K},\mathbf{p}_{t}\in\Delta^{|\mathcal{V}|-1}\right)\right\}_{t=1}^{T}$ be any sequence of samples and let $\{\eta_{t}\in(0,1)\}_{t=1}^{T}$ be any sequence of step-sizes.
Assume the parameters are randomly initialized and updated by gradient descent:

$$ $\displaystyle\bm{\theta}_{0}$ $\displaystyle\sim\mu,\qquad\mu\ll\mathrm{Leb}_{p},$ $\displaystyle\bm{\theta}_{t+1}$ $\displaystyle=\bm{\theta}_{t}-\eta_{t}\nabla\mathcal{L}_{\mathrm{s}_{t},\mathbf{p}_{t}}(\bm{\theta}_{t}),$ $$

where $\mathrm{Leb}_{p}$ denotes Lebesgue measure on $\mathbb{R}^{p}$ and $\mathcal{L}_{\mathrm{s},\mathbf{p}}:\mathbb{R}^{p}\to\mathbb{R}$ is the standard cross-entropy loss

$$ $\mathcal{L}_{\mathrm{s},\mathbf{p}}(\bm{\theta})=\mathrm{CrossEntropy}\big(f(\mathrm{s}\,;\,\bm{\theta}),\,\mathbf{p}\big).$ $$

Then, with probability one over the draw of $\bm{\theta}_{0}$, the last-token, last-layer representation map

$$ $\mathcal{V}^{\leq K}\ni\mathrm{s}\ \longmapsto\ \mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T})\in\mathbb{R}^{d}$ $$

is injective. Equivalently,

$$ $\Pr\left[\exists\,\mathrm{s}\neq\mathrm{t}\in\mathcal{V}^{\leq K}:\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T})=\mathbf{r}(\mathrm{t}\,;\,\bm{\theta}_{T})\right]=0,$ $$

where $\mathbf{r}(\cdot\,;\,\bm{\theta}_{T})$ denotes the last-token representation defined in [Equation 29](#A2.E29).

###### Proof.

Let $\bm{\theta}_{0}\sim\mu$ with $\mu\ll\mathrm{Leb}_{p}$.
For a fixed training horizon $T$, define the *GD update map*

$$ $\Phi:\mathbb{R}^{p}\to\mathbb{R}^{p},\qquad\Phi(\bm{\theta}_{0})\;=\;\bm{\theta}_{T},$ $$

i.e. $\Phi$ is the composition of $T$ gradient-descent steps with step sizes
$\{\eta_{t}\}_{t=1}^{T}\subset(0,1)$ on the loss $\mathcal{L}$.

1) Absolute continuity after $T$ steps.
By [Corollary C.5.1](#A3.Thmtheorem5.Thmcorollary1), since $\mu\ll\mathrm{Leb}_{p}$,
the pushforward law $\Phi_{\#}\mu$ of $\bm{\theta}_{T}$ remains absolutely continuous:

$$ $\bm{\theta}_{T}\;\sim\;\Phi_{\#}\mu\;\ll\;\mathrm{Leb}_{p}.$ $$

2) Global almost-sure distinctness.
Let $\mathcal{S}:=\mathcal{V}^{\leq K}$, which is finite.
By [Corollary C.2.1](#A3.Thmtheorem2.Thmcorollary1), under any absolutely continuous parameter law,

$$ $\Pr\Big[\,\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T})\neq\mathbf{r}(\mathrm{t}\,;\,\bm{\theta}_{T})\;\;\;\forall\,\mathrm{s}\neq\mathrm{t}\in\mathcal{V}^{\leq K}\,\Big]\;=\;1.$ $$

Thus the map $\mathrm{s}\mapsto\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T})$
is injective almost surely, as claimed.
∎

###### Assumption C.1 (Minimum Embedding Dimension) .

###### Theorem C.1 (Finite-horizon a.s. injectivity under GD) .

###### Proof.

### C.1 Absolute continuity ensures almost sure injectivity

We begin by fixing two distinct sequences and asking when their last-token representations can coincide. As before, in this subsection we will consider a finite vocabulary $\mathcal{V}$ and a finite context window $K\in\mathbb{N}$. Additionally, recall that for $\bm{\theta}=(\bm{\theta}_{1},\bm{\theta}_{2},\bm{\theta}_{3})\in\mathbb{R}^{p}$:

$$ $\mathbf{r}(\mathrm{u}\,;\,\bm{\theta}):=\Big(\mathrm{Tr}_{|\mathrm{u}|}\big(\mathrm{Emb}(\mathrm{u}\,;\,\bm{\theta}_{1})\,;\,\bm{\theta}_{2}\big)\Big)_{|\mathrm{u}|}\in\mathbb{R}^{d},$ $$

and for $\mathrm{s}\neq\mathrm{t}$, we define the discrepancy:

$$ $h(\bm{\theta}):=\big\|\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})-\mathbf{r}(\mathrm{t}\,;\,\bm{\theta})\big\|_{2}^{2}.$ $$

By [Proposition B.3](#A2.Thmproposition3), this map is real-analytic. To invoke the zero-set theorem, it suffices to show that $h\not\equiv 0$. We construct a parameter configuration $\bm{\theta}_{\star}$ such that $\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{\star})\neq\mathbf{r}(\mathrm{t}\,;\,\bm{\theta}_{\star})$, treating two exhaustive cases:

- •
Case A: If the sequences differ at their final token or in length, we isolate this distinction via selective initialization of embeddings and positional encodings.
- •
Case B: If they differ earlier, we construct orthogonal embeddings and exploit attention heads to differentiate the contributions to the final representation.

In both cases, we demonstrate explicit parameter settings under which the discrepancy is nonzero. This confirms $h\not\equiv 0$, and the zero set $\big\{\bm{\theta}:\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})=\mathbf{r}(\mathrm{t}\,;\,\bm{\theta})\big\}$ has measure zero by [Theorem A.1](#A1.Thmtheorem1). Hence, if the parameter distribution is absolutely continuous, the probability of a collision is zero. A union bound extends this to any finite set of inputs.

###### Theorem C.2 (Almost-sure pairwise distinctness of last-token representations) .

Let the parameter vector $\bm{\theta}\in\mathbb{R}^{p}$ be drawn from any distribution absolutely continuous with respect to Lebesgue measure. Then, for any fixed $\mathrm{s}\neq\mathrm{t}$,

$$ $\mathrm{Pr}\left[\,\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})=\mathbf{r}(\mathrm{t}\,;\,\bm{\theta})\,\right]=0.$ $$

###### Proof.

Let $T_{\mathrm{s}}=|\mathrm{s}|$ and $T_{\mathrm{t}}=|\mathrm{t}|$, and $h(\bm{\theta}):=\big\|\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})-\mathbf{r}(\mathrm{t}\,;\,\bm{\theta})\big\|_{2}^{2}$.
Since $h$ is real-analytic ([Proposition B.3](#A2.Thmproposition3)), it suffices to show that it is not the zero function on $\mathbb{R}^{p}$; then $h^{-1}(\{0\})$ has Lebesgue measure zero by [Theorem A.1](#A1.Thmtheorem1), and absolute continuity transfers this to probability zero.

We construct a parameter setting $\bm{\theta}_{\star}$ for which $h(\bm{\theta}_{\star})>0$, treating two exhaustive cases:

Case A: $T_{\mathrm{s}}\neq T_{\mathrm{t}}$ or $\mathrm{s}_{T_{\mathrm{s}}}\neq\mathrm{t}_{T_{\mathrm{t}}}$.
Set all Transformer parameters to zero so that the network acts as the identity: $\mathrm{Tr}_{T}(\mathbf{X})=\mathbf{X}$.

- •
If $\mathrm{s}_{T_{\mathrm{s}}}\neq\mathrm{t}_{T_{\mathrm{t}}}$, set $\mathbf{E}_{\mathrm{s}_{T_{\mathrm{s}}}}=\mathbf{e}_{1}$, $\mathbf{E}_{\mathrm{t}_{T_{\mathrm{t}}}}=\mathbf{e}_{2}\neq\mathbf{e}_{1}$, and all other rows of $\mathbf{E}$ to zero. Set $\mathbf{P}=\mathbf{0}_{K\times d}$. Then $\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{\star})=\mathbf{e}_{1}$, $\mathbf{r}(\mathrm{t}\,;\,\bm{\theta}_{\star})=\mathbf{e}_{2}$, so $h(\bm{\theta}_{\star})=\|\mathbf{e}_{1}-\mathbf{e}_{2}\|_{2}^{2}>0$.
- •
If $T_{\mathrm{s}}\neq T_{\mathrm{t}}$, set $\mathbf{E}=\mathbf{0}_{|\mathcal{V}|\times d}$ and $\mathbf{P}_{T_{\mathrm{s}}}=\mathbf{e}_{1}$, $\mathbf{P}_{T_{\mathrm{t}}}=\mathbf{e}_{2}\neq\mathbf{e}_{1}$ (all others zero). Then, again, $\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{\star})=\mathbf{e}_{1}$, $\mathbf{r}(\mathrm{t}\,;\,\bm{\theta}_{\star})=\mathbf{e}_{2}$, so $h(\bm{\theta}_{\star})>0$.

Case B: $T:=T_{\mathrm{s}}=T_{\mathrm{t}}$ and $\mathrm{s}_{T}=\mathrm{t}_{T}$, but $\mathrm{s}_{i}\neq\mathrm{t}_{i}$ for some $i\in[T-1]$.
Let $i^{\star}$ be the smallest such index. Note $T\geq 2$.

We construct a model with (i) all blocks after the first set to identity (zero parameters), (ii) in the first block, all heads set to zero except head 1 and the MLP is zero.

We explicitly construct embeddings and head-1 parameters $(\mathbf{Q},\mathbf{K},\mathbf{V})$, as well as the output projection $\mathbf{W}^{O}$, so that $\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{\star})\neq\mathbf{r}(\mathrm{t}\,;\,\bm{\theta}_{\star})$.

1) Embedding Construction.
Choose orthogonal vectors $\mathbf{e},\mathbf{p},\mathbf{q}\in\mathbb{R}^{d}$ satisfying:

$$ $\langle\mathbf{e},\mathbf{p}\rangle=\langle\mathbf{e},\mathbf{q}\rangle=\langle\mathbf{p},\mathbf{q}\rangle=0,\quad\langle\mathbf{1}_{d},\mathbf{e}\rangle=\langle\mathbf{1}_{d},\mathbf{p}\rangle=\langle\mathbf{1}_{d},\mathbf{q}\rangle=0,\quad\|\mathbf{e}\|_{2}=\|\mathbf{p}\|_{2}=\|\mathbf{q}\|_{2}=1.$ $$

Such vectors exist due to [Assumption C.1](#A3.Thmassumption1) (requires $d\geq 4$). Set embeddings:

$$ $\mathbf{E}_{v}=\begin{cases}\mathbf{e},&v\in\{\mathrm{s}_{i^{\star}},\mathrm{s}_{T}\}\\ \mathbf{0}_{d},&\text{otherwise}\end{cases},\qquad\mathbf{P}_{j}=\begin{cases}\mathbf{p},&j=i^{\star}\\ \mathbf{q},&j=T\\ \mathbf{0}_{d},&\text{otherwise}\end{cases}.$ $$

Thus, the input rows before LayerNorm are:

$$ $\Big[\mathbf{H}(\mathrm{s}\,;\,\bm{\theta}_{\star})\Big]_{j}=\begin{cases}\mathbf{e}+\mathbf{p},&j=i^{\star}\\ \mathbf{e}+\mathbf{q},&j=T\\ \in\{\mathbf{e},\mathbf{0}_{d}\},&\text{otherwise}\end{cases},\qquad\Big[\mathbf{H}(\mathrm{t}\,;\,\bm{\theta}_{\star})\Big]_{j}=\begin{cases}\mathbf{p},&j=i^{\star}\\ \mathbf{e}+\mathbf{q},&j=T\\ \in\{\mathbf{e},\mathbf{0}_{d}\},&\text{otherwise}\end{cases}.$ $$

2) LayerNorm Output.
Use LayerNorm with $(\bm{\gamma},\bm{\beta})=(\mathbf{1},\mathbf{0})$. Since all components have zero mean, the normalization is:

$$ $\mathrm{LN}(\mathbf{x})=\frac{\mathbf{x}}{\sqrt{\frac{1}{d}\|\mathbf{x}\|^{2}+\varepsilon}}=:c(\mathbf{x})\mathbf{x}.$ $$

Define:

$$ $c_{ep}:=\left(\tfrac{2}{d}+\varepsilon\right)^{-1/2},\qquad c_{e}:=\left(\tfrac{1}{d}+\varepsilon\right)^{-1/2}.$ $$

Then:

$$ $\Big[\overline{\mathbf{H}}(\mathrm{s}\,;\,\bm{\theta}_{\star})\Big]_{j}=\begin{cases}c_{ep}(\mathbf{e}+\mathbf{p}),&j=i^{\star}\\ c_{ep}(\mathbf{e}+\mathbf{q}),&j=T\\ \in\{\mathbf{0}_{d},c_{e}\mathbf{e}\},&\text{otherwise}\end{cases},\quad\Big[\overline{\mathbf{H}}(\mathrm{t}\,;\,\bm{\theta}_{\star})\Big]_{j}=\begin{cases}c_{e}\mathbf{p},&j=i^{\star}\\ c_{ep}(\mathbf{e}+\mathbf{q}),&j=T\\ \in\{\mathbf{0}_{d},c_{e}\mathbf{e}\},&\text{otherwise}\end{cases}.$ $$

3) Head Parameters.
Let $\mathbf{e}_{1}\in\mathbb{R}^{d_{\eta}}$ be the first standard basis vector. Set:

$$ $\mathbf{Q}=\alpha\mathbf{e}\mathbf{e}_{1}^{\top},\qquad\mathbf{K}=\beta\mathbf{p}\mathbf{e}_{1}^{\top},\qquad\mathbf{V}=\mathbf{e}\mathbf{e}_{1}^{\top},$ $$

where $\alpha,\beta>0$ are scalars to be chosen.

Then for any $j$, attention vectors are:

$$ $\mathbf{q}_{j}=\alpha\left\langle\Big[\overline{\mathbf{H}}(\cdot\,;\,\bm{\theta}_{\star})\Big]_{j},\;\mathbf{e}\right\rangle\mathbf{e}_{1},\quad\mathbf{k}_{j}=\beta\left\langle\Big[\overline{\mathbf{H}}(\cdot\,;\,\bm{\theta}_{\star})\Big]_{j},\;\mathbf{p}\right\rangle\mathbf{e}_{1},\quad\mathbf{v}_{j}=\left\langle\Big[\overline{\mathbf{H}}(\cdot\,;\,\bm{\theta}_{\star})\Big]_{j},\;\mathbf{e}\right\rangle\mathbf{e}_{1}.$ $$

At row $T$, $\mathbf{q}_{T}^{(\mathrm{s})}=\mathbf{q}_{T}^{(\mathrm{t})}=\alpha c_{ep}\mathbf{e}_{1}$.
Only the key at $i^{\star}$ is nonzero:

$$ $\mathbf{k}_{i^{\star}}^{(\mathrm{s})}=\beta c_{ep}\mathbf{e}_{1},\quad\mathbf{k}_{i^{\star}}^{(\mathrm{t})}=\beta c_{e}\mathbf{e}_{1}.$ $$

Value vectors at $i^{\star}$ differ:

$$ $\mathbf{v}_{i^{\star}}^{(\mathrm{s})}=c_{ep}\mathbf{e}_{1},\quad\mathbf{v}_{i^{\star}}^{(\mathrm{t})}=\mathbf{0}_{d}.$ $$

And $\mathbf{v}_{T}^{(\mathrm{s})}=\mathbf{v}_{T}^{(\mathrm{t})}=c_{ep}\mathbf{e}_{1}$.

4) Attention Weights.
The only nonzero score is at $i^{\star}$:

$$ $\mathbf{S}_{T,i^{\star}}^{(\mathrm{s})}=\frac{\alpha\beta}{\sqrt{d_{\eta}}}c_{ep}^{2},\quad\mathbf{S}_{T,i^{\star}}^{(\mathrm{t})}=\frac{\alpha\beta}{\sqrt{d_{\eta}}}c_{ep}c_{e},\quad\mathbf{S}_{T,j}^{(\cdot)}=0\text{ for }j\neq i^{\star}.$ $$

Fix $\delta\in(0,\tfrac{1}{2})$ and define $L:=\log\left(\frac{1-\delta}{\delta}(T-1)\right)$.
Set $\alpha\beta=\sqrt{d_{\eta}}L/c_{ep}^{2}$, so $\mathbf{S}_{T,i^{\star}}^{(\mathrm{s})}=L$ and $\mathbf{S}_{T,i^{\star}}^{(\mathrm{t})}>L$. Then:

$$ $\mathbf{A}_{T,i^{\star}}^{(\mathrm{s})}\geq 1-\delta,\quad\mathbf{A}_{T,i^{\star}}^{(\mathrm{t})}>1-\delta,\quad\mathbf{A}_{T,j}^{(\cdot)}\leq\frac{\delta}{T-1}\ \text{for }j\neq i^{\star}.$ $$

5) Self-Attention Output.

$$ $\mathbf{y}_{T}^{(\mathrm{s})}=(1-\delta)c_{ep}\mathbf{e}_{1}+\sum_{j\neq i^{\star}}\mathbf{A}_{T,j}^{(\mathrm{s})}\mathbf{v}_{j}^{(\mathrm{s})},\quad\mathbf{y}_{T}^{(\mathrm{t})}=\sum_{j\neq i^{\star}}\mathbf{A}_{T,j}^{(\mathrm{t})}\mathbf{v}_{j}^{(\mathrm{t})}.$ $$

Tails are bounded by:

$$ $\left\|\sum_{j\neq i^{\star}}\mathbf{A}_{T,j}^{(\cdot)}\mathbf{v}_{j}^{(\cdot)}\right\|_{2}\leq\delta c_{e}.$ $$

Since both outputs lie in $\mathrm{span}\{\mathbf{e}_{1}\}$, we compare:

$$ $\langle\mathbf{y}_{T}^{(\mathrm{s})}-\mathbf{y}_{T}^{(\mathrm{t})},\mathbf{e}_{1}\rangle\geq(1-\delta)c_{ep}-2\delta c_{e}.$ $$

Choosing $\delta<\frac{c_{ep}}{c_{ep}+2c_{e}}$ makes this strictly positive.

6) Output Projection and Propagation.
Let $\mathbf{W}^{O}$ be the matrix with $(\mathbf{W}^{O})_{1,1}=1$ and all other entries zero. Then the head output is projected into coordinate 1, making the last row of the first transformer block differ between $\mathrm{s}$ and $\mathrm{t}$ in the first coordinate. Since the original rows at $T$ were identical and the rest of the network is identity, this difference propagates to the final output, and we get $\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{\star})\neq\mathbf{r}(\mathrm{t}\,;\,\bm{\theta}_{\star})$.

∎

###### Remark 12 (Causal Self-Attention) .

The same construction works for causal self-attention. In our setup, attention at position $T$ only needs to consider tokens at positions $j\leq T$, and we only rely on attention from $T$ to $i^{\star}<T$. All nonzero scores occur at these allowable indices, so causal masking does not affect the computation or the argument.

###### Corollary C.2.1 (Almost-sure global distinctness over a finite input family) .

Let $\mathcal{S}\subseteq\mathcal{V}^{\leq K}$ be any finite collection of inputs. If $\bm{\theta}$ is drawn from a law absolutely continuous w.r.t. $\mathrm{Leb}_{p}$, then

$$ $\mathrm{Pr}\big[\ \mathbf{r}(\mathrm{s}\,;\,\bm{\theta})\neq\mathbf{r}(\mathrm{t}\,;\,\bm{\theta})\ \text{ for all distinct }\mathrm{s},\mathrm{t}\in\mathcal{S}\ \big]\;=\;1.$ $$

In particular, the last-token representations are pairwise distinct almost surely across all inputs.

###### Proof.

For each unordered pair $\{\mathrm{s},\mathrm{t}\}\subset\mathcal{S}$ with $\mathrm{s}\neq\mathrm{t}$, [Theorem C.2](#A3.Thmtheorem2) gives $\mathrm{Pr}[\,\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})=\mathbf{r}(\mathrm{t}\,;\,\bm{\theta})\,]=0$. By the union bound over the finitely many pairs ($\binom{|\mathcal{S}|}{2}$ in total),

$$ $\mathrm{Pr}\Big[\,\exists\,\mathrm{s}\neq\mathrm{t}\in\mathcal{S}:\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})=\mathbf{r}(\mathrm{t}\,;\,\bm{\theta})\,\Big]\leq\sum_{\mathrm{s},\mathrm{t}}\mathrm{Pr}\big[\,\mathbf{r}(\mathrm{s}\,;\,\bm{\theta})=\mathbf{r}(\mathrm{t}\,;\,\bm{\theta})\,\big]=0.$ $$

Hence the complement event has probability $1$.
∎

###### Remark 13 (Pointwise vs. last-token injectivity) .

Sutter et al. (2025) establish a related but distinct guarantee. They analyze the mapping from a prompt to the *entire* sequence (matrix) of hidden states, which already rules out collisions for inputs of different lengths. Their result is *pointwise injectivity*: if two prompts differ at position $t$, then the $t$-th hidden state (row) differs. This does not, by itself, imply injectivity of the map to the final hidden state / last-token embedding that we study, so two different prompts could still coincide at the last token–our quantity of operational interest.

###### Theorem C.2 (Almost-sure pairwise distinctness of last-token representations) .

###### Proof.

###### Remark 12 (Causal Self-Attention) .

###### Corollary C.2.1 (Almost-sure global distinctness over a finite input family) .

###### Proof.

###### Remark 13 (Pointwise vs. last-token injectivity) .

### C.2 Absolute continuity of the parameter distribution is preserved under GD

Our goal in this subsection is to explain why absolute continuity of the parameter law at initialization survives any finite number of gradient–descent (GD) steps, thereby allowing the almost-sure injectivity argument from the previous subsection to persist throughout training. The argument proceeds in four steps.

##### Step 1: Regularity of the GD map.

By [Propositions B.3](#A2.Thmproposition3) and [A.6](#A1.Thmproposition6), the loss $\mathcal{L}_{\mathrm{s},\mathbf{p}}$ is real-analytic, and real-analyticity is closed under differentiation and composition. Consequently the GD map $\phi(\bm{\theta})=\bm{\theta}-\eta\nabla\mathcal{L}_{\mathrm{s},\mathbf{p}}(\bm{\theta})$ is real-analytic, its Jacobian $D\phi(\bm{\theta})=\mathbf{I}_{p}-\eta\nabla^{2}\mathcal{L}_{\mathrm{s},\mathbf{p}}(\bm{\theta})$ is real-analytic, and so is $\bm{\theta}\mapsto\det D\phi(\bm{\theta})$ (the determinant is a polynomial in the matrix entries).

##### Step 2: Witness and measure-zero critical set.

We rule out the degenerate case by a witness: at $\bm{\theta}_{\star}=\mathbf{0}_{p}$, our Hessian calculation ([Lemma C.4](#A3.Thmlemma4)) shows $\det D\phi(\bm{\theta}_{\star})>0$, hence $\det D\phi$ is not identically zero and its zero set $\mathcal{C}:=\{\det D\phi=0\}$ has Lebesgue measure zero by the real-analytic zero–set theorem ([Theorem A.1](#A1.Thmtheorem1); summarized in [Theorem C.3](#A3.Thmtheorem3)).

##### Step 3: Local-to-global via countable chart covers.

On the complement $\mathbb{R}^{p}\setminus\mathcal{C}$, the Inverse Function Theorem ([Theorem A.2](#A1.Thmtheorem2)) provides, for every $\bm{\theta}$, a neighborhood on which $\phi$ is a $C^{1}$ diffeomorphism. Although these neighborhoods form an a priori uncountable cover, the second countability of $\mathbb{R}^{p}$ (and of its subspaces) ensures a *countable* subcover of such charts ([Proposition A.15](#A1.Thmproposition15), [Lemma C.5](#A3.Thmlemma5)). This countability is crucial because it lets us pass from local statements to a global measure statement via countable unions. With this cover in hand, the change-of-variables formula on each chart ([Theorem C.4](#A3.Thmtheorem4)) implies that the image under the local inverse of any null set remains null; piecing the charts together and adding the null set $\mathcal{C}$ shows that preimages of Lebesgue-null sets under $\phi$ are null ([Lemma C.6](#A3.Thmlemma6)).

##### Step 4: Preservation of absolute continuity and conclusion.

Equivalently, $\phi$ pushes absolutely continuous laws to absolutely continuous laws ([Theorem C.5](#A3.Thmtheorem5)); iterating across finitely many GD steps preserves absolute continuity ([Corollary C.5.1](#A3.Thmtheorem5.Thmcorollary1)). Finally, combining this preservation with the almost-sure pairwise distinctness of last-token representations over any finite input family ([Corollary C.2.1](#A3.Thmtheorem2.Thmcorollary1)) yields the main consequence we need for training: the last-token representation map remains injective almost surely after any finite GD horizon.

#### C.2.1 Witness Construction

The goal of this subsubsection is to show that the GD Jacobian determinant is not identically zero by evaluating it at the all-zeros witness $\bm{\theta}_{\star}=\mathbf{0}_{p}$. The argument proceeds bottom-up: we first establish a “zero-gate” lemma that zeroes out most Hessian blocks when the gate matrix vanishes ([Lemma C.1](#A3.Thmlemma1)), derive the resulting spectrum ([Lemma C.2](#A3.Thmlemma2), [Lemma C.3](#A3.Thmlemma3)), and then assemble these into the full Hessian at the witness ([Lemma C.4](#A3.Thmlemma4)).

###### Lemma C.1 (Zero-gate through scalar loss) .

Let $\bm{\mathcal{U}}\subseteq\mathbb{R}^{m+q}$ be open and write points as
$\mathbf{v}=(\bm{\xi},\bm{\psi})$ with $\bm{\xi}\in\mathbb{R}^{m}$ and $\bm{\psi}\in\mathbb{R}^{q}$.
Let $\pi:\mathbb{R}^{m+q}\to\mathbb{R}^{m}$ be the projection $\pi(\bm{\xi},\bm{\psi})=\bm{\xi}$.
Consider

$$ $g\in C^{2}(\mathbb{R}^{m}\,;\,\mathbb{R}^{n\times r}),\qquad h\in C^{2}(\bm{\mathcal{U}}\,;\,\mathbb{R}^{r}),$ $$

and define $f:\bm{\mathcal{U}}\to\mathbb{R}^{n}$ by

$$ $f(\bm{\xi},\bm{\psi}):=g(\bm{\xi})\,h(\bm{\xi},\bm{\psi})=g\big(\pi(\bm{\xi},\bm{\psi})\big)\,h(\bm{\xi},\bm{\psi}).$ $$

Let $\mathcal{L}\in C^{2}(\mathbb{R}^{n};\mathbb{R})$ and set

$$ $R:=\mathcal{L}\circ f:\bm{\mathcal{U}}\to\mathbb{R},\qquad R(\bm{\xi},\bm{\psi})=\mathcal{L}\big(g(\bm{\xi})\,h(\bm{\xi},\bm{\psi})\big).$ $$

Fix $\mathbf{v}_{0}=(\bm{\xi}_{0},\bm{\psi}_{0})\in\bm{\mathcal{U}}$ and assume $g(\bm{\xi}_{0})=\mathbf{0}_{n\times r}$.
Then the Hessian of $R$ at $\mathbf{v}_{0}$ has block form

$$ $\nabla^{2}R(\mathbf{v}_{0})=\begin{pmatrix}\nabla_{\bm{\xi}\bm{\xi}}^{2}\,R(\mathbf{v}_{0})&\nabla_{\bm{\xi}\bm{\psi}}^{2}\,R(\mathbf{v}_{0})\[2.0pt] \nabla_{\bm{\psi}\bm{\xi}}^{2}\,R(\mathbf{v}_{0})&\nabla_{\bm{\psi}\bm{\psi}}^{2}\,R(\mathbf{v}_{0})\end{pmatrix}=\begin{pmatrix}\nabla_{\bm{\xi}\bm{\xi}}^{2}R(\mathbf{v}_{0})&\mathbf{0}_{m\times q}\[2.0pt] \mathbf{0}_{q\times m}&\mathbf{0}_{q\times q}\end{pmatrix}.$ $$

i.e. all mixed and $\bm{\psi}$–only second partials vanish.

###### Proof.

1) Introduce the bilinear multiplication map $\mu:\mathbb{R}^{n\times r}\times\mathbb{R}^{r}\to\mathbb{R}^{n}$, $\mu(\mathbf{M},\mathbf{y})=\mathbf{M}\mathbf{y}$, and the $C^{2}$ map
$H:\bm{\mathcal{U}}\to\mathbb{R}^{n\times r}\times\mathbb{R}^{r}$, $H(\bm{\xi},\bm{\psi})=(g(\bm{\xi}),h(\bm{\xi},\bm{\psi}))$. Then $f=\mu\circ H$ and we write:

$$ $g_{0}:=g(\bm{\xi}_{0})=\mathbf{0}_{n\times r}\qquad h_{0}:=h(\bm{\xi}_{0},\bm{\psi}_{0})\qquad H(\mathbf{v_{0}})=(g_{0},h_{0}).$ $$

Because $\mu$ is bilinear, $D\mu(\mathbf{M},\mathbf{y})[(\Delta\mathbf{M},\Delta\mathbf{y})]=\Delta\mathbf{M}\,\mathbf{y}+\mathbf{M}\,\Delta\mathbf{y}$. By the chain rule:

$$ $\displaystyle Df(\mathbf{v}_{0})\big[(\mathbf{h}_{\bm{\xi}},\mathbf{h}_{\bm{\psi}})\big]$ $\displaystyle=D\mu(g_{0},h_{0})\Big[\,Dg(\bm{\xi}_{0})[\mathbf{h}_{\bm{\xi}}],\;Dh(\mathbf{v}_{0})[(\mathbf{h}_{\bm{\xi}},\mathbf{h}_{\bm{\psi}})]\,\Big]$ $\displaystyle=Dg(\bm{\xi}_{0})[\mathbf{h}_{\bm{\xi}}]\,h_{0}+\underbrace{g_{0}}_{\mathbf{0}_{n\times r}}\,Dh(\mathbf{v}_{0})[(\mathbf{h}_{\bm{\xi}},\mathbf{h}_{\bm{\psi}})]$ $\displaystyle=Dg(\bm{\xi}_{0})[\mathbf{h}_{\bm{\xi}}]\,h_{0}.$ $$

In particular, $Df(\mathbf{v}_{0})\big[(\mathbf{0}_{m},\;\cdot\;)\big]=\mathbf{0}_{n}$. The second-order chain rule for Fréchet derivatives (e.g. Magnus and Neudecker 2019, Thm. 18.4) yields:

$$ $D^{2}f(\mathbf{v}_{0})[\mathbf{h},\mathbf{k}]=D^{2}\mu\big(H(\mathbf{v}_{0})\big)\big[\,DH(\mathbf{v}_{0})[\mathbf{h}],\,DH(\mathbf{v}_{0})[\mathbf{k}]\,\big]+D\mu\big(H(\mathbf{v}_{0})\big)\big[\,D^{2}H(\mathbf{v}_{0})[\mathbf{h},\mathbf{k}]\,\big].$ $$

Because $\mu$ is bilinear, $D^{2}\mu\equiv\mathbf{0}$ and the first term is 0. Furthermore,

$$ $D^{2}H(\mathbf{v}_{0})[\mathbf{h},\mathbf{k}]=\Big(\,D^{2}g(\bm{\xi}_{0})[\mathbf{h}_{\bm{\xi}},\mathbf{k}_{\bm{\xi}}],\;D^{2}h(\mathbf{v}_{0})\big[(\mathbf{h}_{\bm{\xi}},\mathbf{h}_{\bm{\psi}}),(\mathbf{k}_{\bm{\xi}},\mathbf{k}_{\bm{\psi}})\big]\,\Big),$ $$

and it holds that:

$$ $\displaystyle D^{2}f(\mathbf{v}_{0})[\mathbf{h},\mathbf{k}]$ $\displaystyle=D\mu(g_{0},h_{0})\Big[\,D^{2}g(\bm{\xi}_{0})[\mathbf{h}_{\bm{\xi}},\mathbf{k}_{\bm{\xi}}],\;D^{2}h(\mathbf{v}_{0})\big[(\mathbf{h}_{\bm{\xi}},\mathbf{h}_{\bm{\psi}}),(\mathbf{k}_{\bm{\xi}},\mathbf{k}_{\bm{\psi}})\big]\,\Big]$ $\displaystyle=\Big(D^{2}g(\bm{\xi}_{0})[\mathbf{h}_{\bm{\xi}},\mathbf{k}_{\bm{\xi}}]\Big)\,h_{0}+\underbrace{g_{0}}_{\mathbf{0}_{n\times r}}\Big(D^{2}h(\mathbf{v}_{0})\big[(\mathbf{h}_{\bm{\xi}},\mathbf{h}_{\bm{\psi}}),(\mathbf{k}_{\bm{\xi}},\mathbf{k}_{\bm{\psi}})\big]\Big)$ $\displaystyle=\Big(D^{2}g(\bm{\xi}_{0})[\mathbf{h}_{\bm{\xi}},\mathbf{k}_{\bm{\xi}}]\Big)\,h_{0}.$ $$

If at least one of the two directions has $\bm{\xi}$–component zero, then
$D^{2}g(\bm{\xi}_{0})[\mathbf{h}_{\bm{\xi}},\mathbf{k}_{\bm{\xi}}]=\mathbf{0}$, so the bilinear form vanishes.

2) Apply the second-order chain rule to $R=\mathcal{L}\circ f$ at $\mathbf{v}_{0}$:

$$ $D^{2}R(\mathbf{v}_{0})[\mathbf{h},\mathbf{k}]=D^{2}\mathcal{L}\big(f(\mathbf{v}_{0})\big)\big[\,Df(\mathbf{v}_{0})[\mathbf{h}],\,Df(\mathbf{v}_{0})[\mathbf{k}]\,\big]+D\mathcal{L}\big(f(\mathbf{v}_{0})\big)\big[\,D^{2}f(\mathbf{v}_{0})[\mathbf{h},\mathbf{k}]\,\big].$ ( $\star$ ) $$

By (1), if at least one of the two directions is pure $\bm{\psi}$, both terms on the right-hand side of vanish.
Therefore

$$ $D^{2}R(\mathbf{v}_{0})[\mathbf{h},\mathbf{k}]=0\qquad\text{whenever at least one of }\mathbf{h},\mathbf{k}\text{ is of the form }(\mathbf{0}_{m},\;\cdot\;).$ $$

Invoking [Proposition A.14](#A1.Thmproposition14),
this is exactly the statement that the $\bm{\xi}\bm{\psi}$, $\bm{\psi}\bm{\xi}$ and
$\bm{\psi}\bm{\psi}$ Hessian blocks are $\mathbf{0}$. The remaining block
$\nabla_{\bm{\xi}\bm{\xi}}^{2}R(\mathbf{v}_{0})$ is whatever is induced by $(\star)$ for pairs

$$ $(\mathbf{h},\mathbf{k})=\big((\mathbf{h}_{\bm{\xi}},\mathbf{0}_{q}),(\mathbf{k}_{\bm{\xi}},\mathbf{0}_{q})\big).$ $$

∎

###### Lemma C.2 (Spectrum under block-diagonal extension) .

Let $f\in C^{2}(\mathbb{R}^{m+q}\,;\,\mathbb{R})$, and fix
$\mathbf{v}=(\bm{\xi}_{0},\bm{\psi}_{0})\in\mathbb{R}^{m+q}$.
Assume the Hessian of $f$ at $\mathbf{v}$ has the block form

$$ $\mathbf{H}:=\nabla^{2}f(\mathbf{v})\ =\ \begin{pmatrix}\mathbf{B}&\mathbf{0}_{m\times q}\\ \mathbf{0}_{q\times m}&\mathbf{0}_{q\times q}\end{pmatrix},\qquad\mathbf{B}\in\mathbb{R}^{m\times m}.$ $$

Then the characteristic polynomial factorizes as

$$ $\chi_{\mathbf{H}}(\lambda):=\det\big(\lambda\mathbf{I}_{m+q}-\mathbf{H}\big)=\det\big(\lambda\mathbf{I}_{m}-\mathbf{B}\big)\,\lambda^{q}.$ $$

Consequently,

$$ $\sigma(\mathbf{H})=\sigma(\mathbf{B})\cup\{0\},\quad\text{and}\quad\mathrm{mult}_{\mathbf{H}}(0)\ =\ \mathrm{mult}_{\mathbf{B}}(0)\,+\,q,$ $$

i.e., the spectrum of $H$ consists of the eigenvalues of $B$ together with $q$ additional zeros,
and the algebraic multiplicity of the eigenvalue 0 for $H$ equals that for $B$ plus $q$.

###### Proof.

Since $\mathbf{H}$ is block diagonal,

$$ $\lambda\mathbf{I}_{m+q}-\mathbf{H}=\begin{pmatrix}\lambda\mathbf{I}_{m}-\mathbf{B}&\mathbf{0}_{m\times q}\\ \mathbf{0}_{q\times m}&\lambda\mathbf{I}_{q}\end{pmatrix}.$ $$

The determinant of a block triangular (in particular block diagonal) matrix equals the product of
the determinants of its diagonal blocks (e.g. Horn and Johnson 2013, Cor. 0.8.5). Hence

$$ $\chi_{\mathbf{H}}(\lambda)=\det(\lambda\mathbf{I}_{m}-\mathbf{B})\cdot\det(\lambda\mathbf{I}_{q})=\det(\lambda\mathbf{I}_{m}-\mathbf{B})\cdot\lambda^{\,q}.$ $$

The zeros of $\chi_{\mathbf{H}}$ are the eigenvalues of $\mathbf{H}$ counted with algebraic multiplicity, which yields
$\sigma(\mathbf{H})=\sigma(\mathbf{B})\cup\{0\}$ and
$\mathrm{mult}_{\mathbf{H}}(0)=\mathrm{mult}_{\mathbf{B}}(0)+q$.
∎

###### Remark 14 .

If $0\in\sigma(\mathbf{B})$, then 0 appears in $\sigma(\mathbf{H})$ with multiplicity strictly larger than $q$; the
statement above accounts for this by adding $q$ to the algebraic multiplicity of 0 carried over
from $\mathbf{B}$.

###### Lemma C.3 (Hessian of $\mathcal{L}$ w.r.t. $\mathbf{U},\bm{\beta}$ at $\bm{\theta}_{\star}=\mathbf{0}$ and its spectrum) .

Let $n:=|\mathcal{V}|$ and $d$ be the embedding width.
Fix $(\mathrm{s},\mathbf{p})\in\mathcal{V}^{\leq K}\times\Delta^{n-1}$, and consider the Transformer Language Model of [Definition B.13](#A2.Thmdefinition13). In the unembedding layer, set the LayerNorm scale to zero, $\bm{\gamma}=\mathbf{0}_{d}$.
Let the parameter be ordered as

$$ $\bm{\theta}=\big(\mathbf{u},\bm{\beta},\bm{\gamma},\bm{\theta}^{\prime}\big),\qquad\mathbf{u}:=\mathrm{vec}_{n,d}(\mathbf{U})\in\mathbb{R}^{nd},\;\bm{\beta}\in\mathbb{R}^{d}.$ $$

Restrict attention to the $(\mathbf{u},\bm{\beta})$-coordinates and the base point

$$ $\bm{\theta}_{\star}=\mathbf{0}_{p}\quad\text{i.e.}\quad\mathbf{U}=\mathbf{0}_{n\times d},\,\bm{\beta}=\mathbf{0}_{d},\,\bm{\gamma}=\mathbf{0}_{d},\,\bm{\theta}^{\prime}=\mathbf{0}.$ $$

Write $\mathbf{b}:=\tfrac{1}{n}\mathbf{1}_{n}$ and $\mathbf{w}:=\mathbf{b}-\mathbf{p}\in\mathbb{R}^{n}$.

Then the Hessian of the cross-entropy loss

$$ $\mathcal{L}(\bm{\theta})=\mathrm{CrossEntropy}\big(f(\mathrm{s}\,;\,\bm{\theta}),\mathbf{p}\big)$ $$

with respect to $(\mathbf{u},\bm{\beta})$ at $\bm{\theta}_{\star}$ is the symmetric block matrix

$$ $\nabla^{2}_{(\mathbf{u},\bm{\beta})}\mathcal{L}(\bm{\theta}_{\star})=\begin{pmatrix}\mathbf{0}_{nd\times nd}&\ \ \mathbf{I}_{d}\otimes\mathbf{w}\[4.0pt] \mathbf{I}_{d}\otimes\mathbf{w}^{\top}&\ \ \mathbf{0}_{d\times d}\end{pmatrix}.$ $$

The spectrum of this Hessian is

$$ $\mathrm{spec}\big(\nabla^{2}_{(\mathbf{u},\bm{\beta})}\mathcal{L}(\bm{\theta}_{\star})\big)=\{\,\underbrace{+\|\mathbf{w}\|_{2},\ldots,+\|\mathbf{w}\|_{2}}_{d},\,\underbrace{-\|\mathbf{w}\|_{2},\ldots,-\|\mathbf{w}\|_{2}}_{d},\ \underbrace{0,\ldots,0}_{d(n-1)}\,\}.$ $$

###### Proof.

1) Logits in vectorized form.
With $\bm{\gamma}=\mathbf{0}_{d}$, the LayerNorm output at the unembedding is constant:
$\mathrm{LN}(\mathbf{h})\equiv\bm{\beta}$ ([Definition B.9](#A2.Thmdefinition9)).
Thus the logits before the final softmax are

$$ $\mathbf{Z}=\mathbf{U}\,\bm{\beta}\in\mathbb{R}^{n}.$ $$

Using $\mathrm{vec}(\mathbf{A}\mathbf{X}\mathbf{b})=(\mathbf{b}^{\top}\otimes\mathbf{A})\,\mathrm{vec}(\mathbf{X})$
(standard identity for vectorization, cf. Henderson and Searle (1981)), with $\mathbf{A}=\mathbf{I}_{n}$ and $\mathbf{b}=\bm{\beta}$,

$$ $\mathbf{z}=\mathrm{vec}(\mathbf{Z})=\mathrm{vec}(\mathbf{U}\bm{\beta})=(\bm{\beta}^{\top}\otimes\mathbf{I}_{n})\,\mathbf{u}.$ $$

Therefore, near $(\mathbf{u},\bm{\beta})=(\mathbf{0}_{nd},\mathbf{0}_{d})$, the logits map is the bilinear function

$$ $z(\mathbf{u},\bm{\beta}):=(\bm{\beta}^{\top}\otimes\mathbf{I}_{n})\,\mathbf{u}\in\mathbb{R}^{n}.$ $$

2) First and second differentials.
Let $(\mathbf{h},\bm{\eta})$ and $(\mathbf{k},\bm{\xi})$ be directions in $\mathbb{R}^{nd}\times\mathbb{R}^{d}$.
Differentiating $z(\mathbf{u},\bm{\beta})=(\bm{\beta}^{\top}\otimes\mathbf{I}_{n})\mathbf{u}$ gives

$$ $Dz(\mathbf{u},\bm{\beta})[\mathbf{h},\bm{\eta}]=(\bm{\beta}^{\top}\otimes\mathbf{I}_{n})\mathbf{h}+(\bm{\eta}^{\top}\otimes\mathbf{I}_{n})\mathbf{u}.$ $$

At $(\mathbf{u},\bm{\beta})=(\mathbf{0}_{nd},\mathbf{0}_{d})$,

$$ $Dz(\mathbf{0}_{nd},\mathbf{0}_{d})[\mathbf{h},\bm{\eta}]=\mathbf{0}_{n\times(nd+d)}$ $$

(since both terms are multiplied by $\mathbf{u}$ or $\bm{\beta}$).
Differentiating once more (or, equivalently, using bilinearity of $z$) yields the constant symmetric bilinear form

$$ $D^{2}z(\mathbf{0}_{nd},\mathbf{0}_{n})\big[(\mathbf{h},\bm{\eta}),(\mathbf{k},\bm{\xi})\big]=(\bm{\xi}^{\top}\otimes\mathbf{I}_{n})\,\mathbf{h}+(\bm{\eta}^{\top}\otimes\mathbf{I}_{n})\,\mathbf{k}.$ $$

3) Gradient of the CE-in-softmax at the origin.
Let $F(\mathbf{z}):=\mathrm{CrossEntropy}(\mathrm{softmax}(\mathbf{z}),\mathbf{p})$.
A standard computation (softmax Jacobian) gives

$$ $\nabla_{\mathbf{z}}F(\mathbf{z})=\mathrm{softmax}(\mathbf{z})-\mathbf{p}.$ $$

At $\mathbf{z}=\mathbf{0}_{n}$, $\mathrm{softmax}\left(\mathbf{0}_{n}\right)=\frac{1}{n}\mathbf{1}_{n}=:\mathbf{b}$, hence

$$ $\nabla_{\mathbf{z}}F(\mathbf{0}_{n})=\mathbf{b}-\mathbf{p}=:\mathbf{w}.$ $$

4) Second-order chain rule for $F\circ Z$ at $(\mathbf{0},\mathbf{0})$.
Similarly to the proof of [Lemma C.1](#A3.Thmlemma1), the second differential of a composition is

$$ $D^{2}(F\circ z)(\mathbf{v})[\mathbf{h},\mathbf{k}]=D^{2}F(z(\mathbf{v}))\big[Dz(\mathbf{v})\mathbf{h},\,Dz(\mathbf{v})\mathbf{k}\big]+DF(z(\mathbf{v}))\big[D^{2}z(\mathbf{v})[\mathbf{h},\mathbf{k}]\big].$ $$

At $\mathbf{v}=(\mathbf{0}_{nd},\mathbf{0}_{d})$, $Dz(\mathbf{v})=\mathbf{0}_{n\times(nd+d)}$ and $DF(z(\mathbf{v}))=\nabla_{\mathbf{z}}F(\mathbf{0}_{n})^{\top}=\mathbf{w}^{\top}$, so

$$ $\displaystyle D^{2}\mathcal{L}(\mathbf{v})\big[(\mathbf{h},\bm{\eta}),(\mathbf{k},\bm{\xi})\big]$ $\displaystyle=\mathbf{w}^{\top}\,D^{2}z(\mathbf{v})\big[(\mathbf{h},\bm{\eta}),(\mathbf{k},\bm{\xi})\big]$ $\displaystyle=\mathbf{w}^{\top}\big((\bm{\xi}^{\top}\otimes\mathbf{I}_{n})\mathbf{h}+(\bm{\eta}^{\top}\otimes\mathbf{I}_{n})\mathbf{k}\big)$ $\displaystyle=\mathbf{h}^{\top}(\mathbf{I}_{d}\otimes\mathbf{w})\,\bm{\xi}\;+\;\mathbf{k}^{\top}(\mathbf{I}_{d}\otimes\mathbf{w})\,\bm{\eta},$ $$

where we used the mixed-product rule for Kronecker products and the identity

$$ $\mathbf{w}^{\top}(\bm{\xi}^{\top}\otimes\mathbf{I}_{n})=\bm{\xi}^{\top}\otimes\mathbf{w}^{\top}.$ $$

5) Identification of the Hessian blocks.
By definition of the Hessian as a bilinear form,

$$ $D^{2}\mathcal{L}(\mathbf{v})\big[(\mathbf{h},\bm{\eta}),(\mathbf{k},\bm{\xi})\big]=\begin{pmatrix}\mathbf{h}^{\top}&\bm{\eta}^{\top}\end{pmatrix}\begin{pmatrix}\mathbf{0}_{nd\times nd}&\frac{\partial^{2}\mathcal{L}}{\partial\mathbf{u}\,\partial\bm{\beta}}\[2.0pt] \frac{\partial^{2}\mathcal{L}}{\partial\bm{\beta}\,\partial\mathbf{u}}&\mathbf{0}_{d\times d}\end{pmatrix}\begin{pmatrix}\mathbf{k}\\ \bm{\xi}\end{pmatrix}.$ $$

Comparing with the expression obtained in Step 4 for arbitrary $(\mathbf{h},\bm{\eta})$ and $(\mathbf{k},\bm{\xi})$ forces

$$ $\frac{\partial^{2}\mathcal{L}}{\partial\mathbf{u}\,\partial\bm{\beta}}(\bm{\theta}_{\star})=\mathbf{I}_{d}\otimes\mathbf{w},\qquad\frac{\partial^{2}\mathcal{L}}{\partial\bm{\beta}\,\partial\mathbf{u}}(\bm{\theta}_{\star})=\big(\mathbf{I}_{d}\otimes\mathbf{w}\big)^{\top}=\mathbf{I}_{d}\otimes\mathbf{w}^{\top},$ $$

and, because $Dz(\mathbf{v})=\mathbf{0}_{n\times(nd+d)}$ (so no quadratic term survives in either $\mathbf{u}$ or $\bm{\beta}$ alone),

$$ $\frac{\partial^{2}\mathcal{L}}{\partial\mathbf{u}\,\partial\mathbf{u}}(\bm{\theta}_{\star})=\mathbf{0}_{nd\times nd},\qquad\frac{\partial^{2}\mathcal{L}}{\partial\bm{\beta}\,\partial\bm{\beta}}(\bm{\theta}_{\star})=\mathbf{0}_{d\times d}.$ $$

This gives exactly the claimed block matrix.

6) Spectrum.
Let

$$ $\mathbf{H}:=\nabla_{(\mathbf{u},\bm{\beta})}^{2}\mathcal{L}(\bm{\theta}_{\star})=\begin{pmatrix}\mathbf{0}_{nd\times nd}&\ \ \mathbf{I}_{d}\otimes\mathbf{w}\[4.0pt] \mathbf{I}_{d}\otimes\mathbf{w}^{\top}&\ \ \mathbf{0}_{d\times d}\end{pmatrix}.$ $$

Then

$$ $\mathbf{H}^{2}=\begin{pmatrix}(\mathbf{I}_{d}\otimes\mathbf{w})(\mathbf{I}_{d}\otimes\mathbf{w}^{\top})&\mathbf{0}_{nd\times d}\\ \mathbf{0}_{d\times nd}&(\mathbf{I}_{d}\otimes\mathbf{w}^{\top})(\mathbf{I}_{d}\otimes\mathbf{w})\end{pmatrix}=\begin{pmatrix}\mathbf{I}_{d}\otimes(\mathbf{w}\mathbf{w}^{\top})&\mathbf{0}_{nd\times d}\\ \mathbf{0}_{d\times nd}&\mathbf{I}_{d}\otimes(\mathbf{w}^{\top}\mathbf{w})\end{pmatrix}.$ $$

The eigenvalues of $\mathbf{w}\mathbf{w}^{\top}$ are $\|\mathbf{w}\|_{2}^{2}$ (multiplicity $1$) and 0 (multiplicity $n-1$); the eigenvalues of $\mathbf{w}^{\top}\mathbf{w}$ equal $\|\mathbf{w}\|_{2}^{2}$ (scalar). Therefore the eigenvalues of $\mathbf{H}^{2}$ are

$$ $\underbrace{\|\mathbf{w}\|_{2}^{2},\ldots,\|\mathbf{w}\|_{2}^{2}}_{2d\ \text{times}},\quad\underbrace{0,\ldots,0}_{d(n-1)\ \text{times}}.$ $$

Because $\mathbf{H}$ is symmetric, its eigenvalues are the real square-roots of those of $\mathbf{H}^{2}$, namely
$\pm\|\mathbf{w}\|_{2}$ (each with multiplicity $d$) and 0 (with multiplicity $d(n-1)$).
This is exactly the set stated in the lemma.
∎

###### Lemma C.4 (Full Hessian at the witness: block form and spectrum) .

Let $n:=|\mathcal{V}|$ and $d$ be the embedding width. Write the parameter as

$$ $\bm{\theta}\;=\;\big((\mathbf{u},\bm{\beta}),\,(\bm{\gamma},\bm{\theta}^{\prime})\big),\qquad\mathbf{u}=\mathrm{vec}_{n,d}(\mathbf{U})\in\mathbb{R}^{nd},\;\bm{\beta},\bm{\gamma}\in\mathbb{R}^{d},\;\bm{\theta}^{\prime}\in\mathbb{R}^{p^{\prime}},$ $$

so $p=nd+2d+p^{\prime}$. Consider the witness point

$$ $\bm{\theta}_{\star}=\mathbf{0}_{p}\quad(\mathbf{U}=\mathbf{0}_{n\times d},\ \bm{\beta}=\mathbf{0}_{d},\ \bm{\gamma}=\mathbf{0}_{d},\ \bm{\theta}^{\prime}=\mathbf{0}_{d}).$ $$

Let $\mathbf{b}:=\tfrac{1}{n}\mathbf{1}_{n}$ and $\mathbf{w}:=\mathbf{b}-\mathbf{p}\in\mathbb{R}^{n}$.
Then the Hessian of the cross-entropy loss $\mathcal{L}(\bm{\theta})$ at $\bm{\theta}_{\star}$
admits the block-diagonal decomposition

$$ $\nabla^{2}\mathcal{L}(\bm{\theta}_{\star})\;=\;\begin{pmatrix}\mathbf{B}&\mathbf{0}\[2.0pt] \mathbf{0}&\mathbf{0}\end{pmatrix},\qquad\mathbf{B}\;=\;\begin{pmatrix}\mathbf{0}_{nd\times nd}&\mathbf{I}_{d}\otimes\mathbf{w}\[2.0pt] \mathbf{I}_{d}\otimes\mathbf{w}^{\top}&\mathbf{0}_{d\times d}\end{pmatrix}.$ $$

Consequently,

$$ $\mathrm{spec}\big(\nabla^{2}\mathcal{L}(\bm{\theta}_{\star})\big)\;=\;\Big\{\underbrace{+\|\mathbf{w}\|_{2},\ldots,+\|\mathbf{w}\|_{2}}_{d},\ \underbrace{-\|\mathbf{w}\|_{2},\ldots,-\|\mathbf{w}\|_{2}}_{d},\ \underbrace{0,\ldots,0}_{\,p-2d}\Big\}.$ $$

###### Proof.

Set $\bm{\gamma}=\mathbf{0}_{d}$. Then the unembedding LayerNorm output is constant,
$\mathrm{LN}(\mathbf{h})\equiv\bm{\beta}$, so the logits equal $\mathbf{z}=\mathbf{U}\,\bm{\beta}$.
Hence, in a neighborhood of $\bm{\theta}_{\star}$, the loss depends only on
$(\mathbf{u},\bm{\beta})$ and is independent of $(\bm{\gamma},\bm{\theta}^{\prime})$.

We will apply [Lemma C.1](#A3.Thmlemma1) with the open set $\bm{\mathcal{U}}=\mathbb{R}^{nd+2d+p^{\prime}}$, coordinates $\bm{\xi}=(\mathbf{u},\bm{\beta})$ and $\bm{\psi}=(\bm{\gamma},\bm{\theta}^{\prime})$ and with $n=|\mathcal{V}|$, $r=d$. Define

$$ $g(\bm{\xi}):=\mathrm{mat}_{n,d}(\mathbf{u})\in\mathbb{R}^{n\times d},\qquad h(\bm{\xi},\bm{\psi}):=\bm{\beta}\in\mathbb{R}^{d},$ $$

so that

$$ $f(\bm{\xi},\bm{\psi}):=g(\bm{\xi})\,h(\bm{\xi},\bm{\psi})\;=\;\mathbf{U}\,\bm{\beta}\in\mathbb{R}^{n},$ $$

and, with $\mathcal{L}(\mathbf{z}):=\mathrm{CrossEntropy}\big(\mathrm{softmax}(\mathbf{z}),\mathbf{p}\big)$,

$$ $R(\bm{\xi},\bm{\psi}):=\mathcal{L}\big(f(\bm{\xi},\bm{\psi})\big)=\mathrm{CrossEntropy}\big(\mathrm{softmax}(\mathbf{U}\bm{\beta}),\mathbf{p}\big).$ $$

At the witness $\mathbf{v}_{0}=(\bm{\xi}_{0},\bm{\psi}_{0})$ we have $g(\bm{\xi}_{0})=\mathbf{0}_{n\times d}$, so by [Lemma C.1](#A3.Thmlemma1) all mixed and $\bm{\psi}$–only second partials of $R$ vanish at $\mathbf{v}_{0}$, i.e.

$$ $\nabla^{2}R(\mathbf{v}_{0})=\begin{pmatrix}\nabla^{2}_{(\mathbf{u},\bm{\beta})}R(\mathbf{v}_{0})&\mathbf{0}\\ \mathbf{0}&\mathbf{0}\end{pmatrix}.$ $$

Identifying $R(\bm{\xi},\bm{\psi})\equiv\mathcal{L}(\bm{\theta})$ under the correspondence above yields

$$ $\nabla^{2}\mathcal{L}(\bm{\theta}_{\star})=\begin{pmatrix}\nabla^{2}_{(\mathbf{u},\bm{\beta})}\mathcal{L}(\bm{\theta}_{\star})&\mathbf{0}\\ \mathbf{0}&\mathbf{0}\end{pmatrix}.$ $$

Combining, [Lemmas C.2](#A3.Thmlemma2) and [C.3](#A3.Thmlemma3), we get that

$$ $\displaystyle\mathrm{spec}\big(\nabla^{2}\mathcal{L}(\bm{\theta}_{\star})\big)$ $\displaystyle=\mathrm{spec}\big(\nabla^{2}_{(\mathbf{u},\bm{\beta})}\mathcal{L}(\bm{\theta}_{\star})\big)\ \cup\ \{0\}^{\,d+p^{\prime}}$ $\displaystyle=\Big\{\pm\|\mathbf{w}\|_{2}\ \text{(each mult.\ $d$)},\ 0\ \text{(mult.\ $d(n-1)+d+p^{\prime}$)}\Big\}.$ $$

Since $p=nd+2d+p^{\prime}$, the multiplicity of 0 equals $p-2d$, which yields the claimed spectrum.
∎

###### Theorem C.3 (GD Jacobian is nondegenerate a.e.) .

Fix a finite vocabulary $\mathcal{V}$, a context bound $K\in\mathbb{N}$, and the Transformer language model $f$ of [Definition B.13](#A2.Thmdefinition13). For any sample $(\mathrm{s},\mathbf{p})\in\mathcal{V}^{\leq K}\times\Delta^{|\mathcal{V}|-1}$ and any learning rate $\eta\in(0,1)$, let $\phi:\mathbb{R}^{p}\to\mathbb{R}^{p}$ be the gradient-descent update, defined as:

$$ $\phi(\bm{\theta})\;=\;\bm{\theta}\;-\;\eta\,\nabla_{\bm{\theta}}\mathcal{L}_{\mathrm{s},\mathbf{p}}(\bm{\theta}),$ $$

where $\mathcal{L}_{\mathrm{s},\mathbf{p}}:\mathbb{R}^{p}\to\mathbb{R}$ is the standard
Cross Entropy loss:

$$ $\mathcal{L}_{\mathrm{s},\mathbf{p}}(\bm{\theta})=\mathrm{CrossEntropy}\big(f(\mathrm{s}\,;\,\bm{\theta}),\mathbf{p}\big).$ $$

Then the critical set

$$ $\mathcal{C}\;:=\;\{\bm{\theta}\in\mathbb{R}^{p}:\det{D\phi(\bm{\theta})}=0\}$ $$

has Lebesgue measure zero in $\mathbb{R}^{p}$.

###### Proof.

By [Propositions B.3](#A2.Thmproposition3) and [A.6](#A1.Thmproposition6) and the closure properties of real analyticity,
$\mathcal{L}_{\mathrm{s},\mathbf{p}}$ is real-analytic; hence so are its gradient and Hessian.
Therefore $\phi$ is real-analytic (Lewis, 2014, Thm. 1.1.15) and

$$ $D\phi(\bm{\theta})=\mathbf{I}_{p}-\eta\,\nabla^{2}_{\bm{\theta}}\mathcal{L}_{\mathrm{s},\mathbf{p}}(\bm{\theta}).$ $$

Since the determinant is a polynomial in the entries,
$\bm{\theta}\mapsto\det{D\phi(\bm{\theta})}$ is real-analytic.

It is not identically zero: at the witness $\bm{\theta}_{\star}=\mathbf{0}_{p}$,
[Lemma C.4](#A3.Thmlemma4) gives

$$ $\mathrm{spec}\big(\nabla^{2}\mathcal{L}(\bm{\theta}_{\star})\big)=\{\underbrace{+\|\mathbf{w}\|_{2},\ldots,+\|\mathbf{w}\|_{2}}_{d},\underbrace{-\|\mathbf{w}\|_{2},\ldots,-\|\mathbf{w}\|_{2}}_{d},\underbrace{0,\ldots,0}_{p-2d}\},\quad\mathbf{w}:=\tfrac{1}{n}\mathbf{1}-\mathbf{p}.$ $$

Hence the eigenvalues of $D\phi(\bm{\theta}_{\star})=\mathbf{I}_{p}-\eta\,\nabla^{2}\mathcal{L}(\bm{\theta}_{\star})$ are

$$ $\underbrace{1-\eta\|\mathbf{w}\|_{2}}_{d\,\text{times}},\quad\underbrace{1+\eta\|\mathbf{w}\|_{2}}_{d\,\text{times}},\quad\underbrace{1}_{p-2d\,\text{times}},$ $$

so

$$ $\det D\phi(\bm{\theta}_{\star})=\left(1-\eta^{2}\|\mathbf{w}\|_{2}^{2}\right)^{d}>0.$ $$

Thus $\det D\phi$ is a nontrivial real-analytic function.
By [Theorem A.1](#A1.Thmtheorem1), its zero set has Lebesgue measure 0.
∎

###### Lemma C.1 (Zero-gate through scalar loss) .

###### Proof.

###### Lemma C.2 (Spectrum under block-diagonal extension) .

###### Proof.

###### Remark 14 .

###### Lemma C.3 (Hessian of ℒ \mathcal{L} w.r.t. 𝐔 , 𝜷 \mathbf{U},\bm{\beta} at 𝜽 ⋆ = 𝟎 \bm{\theta}_{\star}=\mathbf{0} and its spectrum) .

###### Proof.

###### Lemma C.4 (Full Hessian at the witness: block form and spectrum) .

###### Proof.

###### Theorem C.3 (GD Jacobian is nondegenerate a.e.) .

###### Proof.

#### C.2.2 Gradient Descent preserves absolute continuity

With the witness in hand and the critical set shown to be measure-zero ([Theorem C.3](#A3.Thmtheorem3)), we now carry out Steps 3–4 of the roadmap: cover the non-critical region by countably many diffeomorphic charts ([Lemma C.5](#A3.Thmlemma5)), use the change-of-variables formula to show preimages of null sets remain null ([Lemma C.6](#A3.Thmlemma6)), and conclude that one GD step preserves absolute continuity ([Theorem C.5](#A3.Thmtheorem5)). Iterating yields the finite-horizon corollary ([Corollary C.5.1](#A3.Thmtheorem5.Thmcorollary1)).

###### Lemma C.5 (Countable chart cover of $\mathbb{R}^{p}\setminus\mathcal{C}$ ) .

Consider the setup of [Theorem C.5](#A3.Thmtheorem5).
In particular, let $\phi:\mathbb{R}^{p}\to\mathbb{R}^{p}$ be the one-step GD map from that theorem:

$$ $\phi(\bm{\theta})=\bm{\theta}-\eta\,\nabla_{\bm{\theta}}\mathcal{L}_{\mathrm{s},\mathbf{p}}(\bm{\theta}),$ (32) $$

with stepsize $\eta\in(0,1)$, and the measure-zero critical-set ([Theorem C.3](#A3.Thmtheorem3)):

$$ $\mathcal{C}\;:=\;\{\bm{\theta}\in\mathbb{R}^{p}:\det{D\phi(\bm{\theta})}=0\}.$ $$

Then there exist open sets $(\bm{\mathcal{U}}_{k})_{k\geq 1}$ covering $\bm{\mathcal{X}}:=\mathbb{R}^{p}\setminus\mathcal{C}$ such that, for each $k$,
the restriction $\phi_{k}:=\phi|_{\bm{\mathcal{U}}_{k}}:\bm{\mathcal{U}}_{k}\to\bm{\mathcal{V}}_{k}:=\phi(\bm{\mathcal{U}}_{k})$ is a $C^{1}$ diffeomorphism with $C^{1}$ inverse
$\psi_{k}:=\phi_{k}^{-1}$.

###### Proof.

1) $\bm{\mathcal{X}}$ is open:
By [Propositions B.3](#A2.Thmproposition3) and [A.6](#A1.Thmproposition6) and the closure rules of real-analyticity,
$\mathcal{L}_{\mathrm{s},\mathbf{p}}$ is $C^{2}$, hence $\phi$ is $C^{1}$.
The map $\bm{\theta}\mapsto D\phi(\bm{\theta})$ is continuous, and the determinant is a continuous polynomial in the entries, so $g(\bm{\theta}):=\det D\phi(\bm{\theta})$ is continuous.
Therefore $\mathcal{C}=g^{-1}(\{0\})$ is closed (Rudin, 1976, Thm. 4.8) and $\bm{\mathcal{X}}=\mathbb{R}^{p}\setminus\mathcal{C}$ is open.

2) Local diffeomorphisms by the Inverse Function Theorem:
Fix $\bm{\theta}\in\bm{\mathcal{X}}$. Then $g(\bm{\theta})\neq 0$, so by the Inverse Function Theorem
([Theorem A.2](#A1.Thmtheorem2)) there exist open neighborhoods
$\bm{\mathcal{U}}_{\bm{\theta}}\ni\bm{\theta}$ and $\bm{\mathcal{V}}_{\bm{\theta}}\ni\phi(\bm{\theta})$ such that

$$ $\phi_{\bm{\theta}}:=\phi|_{\bm{\mathcal{U}}_{\bm{\theta}}}:\bm{\mathcal{U}}_{\bm{\theta}}\to\bm{\mathcal{V}}_{\bm{\theta}}$ $$

is a $C^{1}$ diffeomorphism with $C^{1}$ inverse $\psi_{\bm{\theta}}:=\phi_{\bm{\theta}}^{-1}$.
Moreover,

$$ $D\psi_{\bm{\theta}}(\phi(\mathbf{x}))=\big(D\phi(\mathbf{x})\big)^{-1}\qquad\forall\,\mathbf{x}\in\bm{\mathcal{U}}_{\bm{\theta}}.$ $$

In particular $D\phi(\mathbf{x})$ is invertible for all $\mathbf{x}\in\bm{\mathcal{U}}_{\bm{\theta}}$, whence $\bm{\mathcal{U}}_{\bm{\theta}}\subset\bm{\mathcal{X}}$.
Thus $\{\bm{\mathcal{U}}_{\bm{\theta}}\}_{\bm{\theta}\in\bm{\mathcal{X}}}$ is an open cover of $\bm{\mathcal{X}}$ by IFT charts.

3) Select a countable subcover:
By [Proposition A.15](#A1.Thmproposition15)(3), $\mathbb{R}^{p}$ is second-countable; subspaces of second-countable spaces are second-countable, hence $\bm{\mathcal{X}}$ is second-countable. By [Proposition A.15](#A1.Thmproposition15)(4),
every open cover of a second-countable space admits a countable subcover. Therefore there exist
points $\bm{\theta}_{1},\bm{\theta}_{2},\ldots\in\bm{\mathcal{X}}$ such that $\bm{\mathcal{X}}=\bigcup_{k=1}^{\infty}\bm{\mathcal{U}}_{\bm{\theta}_{k}}$.

Set $\bm{\mathcal{U}}_{k}:=\bm{\mathcal{U}}_{\bm{\theta}_{k}}$, $\bm{\mathcal{V}}_{k}:=\bm{\mathcal{V}}_{\bm{\theta}_{k}}$, and $\phi_{k}:=\phi|_{\bm{\mathcal{U}}_{k}}=\phi_{\bm{\theta}_{k}}$, $\psi_{k}:=\psi_{\bm{\theta}_{k}}$.
Each $\phi_{k}$ is a $C^{1}$ diffeomorphism with $C^{1}$ inverse $\psi_{k}$ by Step 2.
This yields the desired countable chart cover of $\bm{\mathcal{X}}$.
∎

###### Theorem C.4 (Change of Variables Folland 1999 , Thm. 2.47(b) ) .

Let $\bm{\mathcal{U}},\bm{\mathcal{V}}\subseteq\mathbb{R}^{p}$ be open and $\psi:\bm{\mathcal{V}}\to\bm{\mathcal{U}}$ a $C^{1}$ diffeomorphism. If $\bm{\mathcal{E}}\subseteq\bm{\mathcal{V}}$ is Lebesgue measurable, then

$$ $\mathrm{Leb}_{p}\big(\psi(\bm{\mathcal{E}})\big)=\int_{\bm{\mathcal{E}}}\big|\det{D\psi(\mathbf{y})}\big|\,d\mathbf{y}.$ $$

###### Lemma C.6 (Pre-images of null sets are null) .

Consider the setup of [Theorem C.5](#A3.Thmtheorem5), in particular the $C^{1}$ gradient descent map:

$$ $\phi(\bm{\theta})=\bm{\theta}-\eta\nabla_{\bm{\theta}}\mathcal{L}_{\mathrm{s},\mathbf{p}}(\bm{\theta}),\qquad\eta\in(0,1),$ $$

and its critical set $\mathcal{C}:=\{\bm{\theta}\in\mathbb{R}^{p}:\det{D\phi(\bm{\theta})}=0\}$.
Then, for every measurable $\bm{\mathcal{A}}\subseteq\mathbb{R}^{p}$,

$$ $\mathrm{Leb}_{p}(\bm{\mathcal{A}})=0\implies\mathrm{Leb}_{p}\big(\phi^{-1}(\bm{\mathcal{A}})\big)=0.$ $$

###### Proof.

Let $\bm{\mathcal{X}}=\mathbb{R}^{p}\setminus\mathcal{C}$ and decompose the pre-image:

$$ $\phi^{-1}(\bm{\mathcal{A}})=\big(\phi^{-1}(\bm{\mathcal{A}})\cap\mathcal{C}\big)\cup\big(\phi^{-1}(\bm{\mathcal{A}})\cap\bm{\mathcal{X}}\big).$ $$

The first set is contained in $\mathcal{C}$, a measure zero set ([Theorem C.3](#A3.Thmtheorem3)), hence has $\mathrm{Leb}_{p}$–measure 0.
By [Lemma C.5](#A3.Thmlemma5), cover $\bm{\mathcal{X}}$ by countably many charts $\{\bm{\mathcal{U}}_{k}\}$ on which $\phi_{k}:=\phi|_{\bm{\mathcal{U}}_{k}}$ is a $C^{1}$ diffeomorphism onto $\bm{\mathcal{V}}_{k}:=\phi(\bm{\mathcal{U}}_{k})$ with inverse $\psi_{k}\in C^{1}(\bm{\mathcal{V}}_{k}\,;\,\bm{\mathcal{U}}_{k})$.
Then, it holds that:

$$ $\phi^{-1}(\bm{\mathcal{A}})\cap\bm{\mathcal{U}}_{k}=\psi_{k}\big(\bm{\mathcal{A}}\cap\bm{\mathcal{V}}_{k}\big).$ $$

Since $\mathrm{Leb}_{p}(\bm{\mathcal{A}})=0$ and both $\bm{\mathcal{A}}$ and $\bm{\mathcal{V}}_{k}$ are measurable, $\bm{\mathcal{A}}\cap\bm{\mathcal{V}}_{k}$ is measurable and has measure 0.
By [Theorem C.4](#A3.Thmtheorem4) applied to $\psi_{k}$ with $\bm{\mathcal{E}}=\bm{\mathcal{A}}\cap\bm{\mathcal{V}}_{k}$,

$$ $\mathrm{Leb}_{p}\big(\psi_{k}(\bm{\mathcal{A}}\cap\bm{\mathcal{V}}_{k})\big)=\int_{\bm{\mathcal{A}}\cap\bm{\mathcal{V}}_{k}}\big|\det{D\psi_{k}(\mathbf{y})}\big|\,d\mathbf{y}=0.$ $$

Therefore, each $\phi^{-1}(\bm{\mathcal{A}})\cap\bm{\mathcal{U}}_{k}$ is null and because a countable union of null sets is null, it holds that:

$$ $\mathrm{Leb}_{p}\big(\phi^{-1}(\bm{\mathcal{A}})\big)=0.$ $$

∎

###### Theorem C.5 (Preservation of absolute continuity under one GD step) .

Consider the setup of [Theorem C.3](#A3.Thmtheorem3).
In particular, let $\phi:\mathbb{R}^{p}\to\mathbb{R}^{p}$ be the one-step GD map from that theorem:

$$ $\phi(\bm{\theta})=\bm{\theta}-\eta\,\nabla_{\bm{\theta}}\mathcal{L}_{\mathrm{s},\mathbf{p}}(\bm{\theta}),$ (33) $$

with stepsize $\eta\in(0,1)$. Then, gradient-descent preserves absolute continuity: for every absolutely continuous probability law $\mu$ on $\mathbb{R}^{p}$, its image under $\phi$ remains absolutely continuous:

$$ $\phi_{\#}\mu\;\ll\;\mathrm{Leb}_{p}.$ $$

Therefore, the updated parameters $\bm{\theta}^{\prime}:=\phi(\bm{\theta})$ are absolutely continuous.

###### Proof.

By [Proposition B.3](#A2.Thmproposition3) and closure properties, $\mathcal{L}_{\mathrm{s},\mathbf{p}}$ is $C^{2}$, hence $\phi\in C^{1}$ and is Borel-measurable.
From [Theorem C.3](#A3.Thmtheorem3) the critical set

$$ $\mathcal{C}\;:=\;\{\bm{\theta}\in\mathbb{R}^{p}:\det D\phi(\bm{\theta})=0\}$ $$

has $\mathrm{Leb}_{p}$-measure 0. Therefore, the hypothesis of [Lemma C.6](#A3.Thmlemma6) holds, and we have the property:

$$ $\mathrm{Leb}_{p}(\bm{\mathcal{A}})=0\quad\Longrightarrow\quad\mathrm{Leb}_{p}\big(\phi^{-1}(\bm{\mathcal{A}})\big)=0\qquad\text{for every measurable }\bm{\mathcal{A}}\subseteq\mathbb{R}^{p}.$ ( $\dagger$ ) $$

Let $\bm{\mathcal{A}}$ be any Borel set with $\mathrm{Leb}_{p}(\bm{\mathcal{A}})=0$. Then

$$ $\phi_{\#}\mu(\bm{\mathcal{A}})\;=\;\mu\big(\phi^{-1}(\bm{\mathcal{A}})\big)\;=\;0,$ $$

because $\mu\ll\mathrm{Leb}_{p}$ and $\mathrm{Leb}_{p}\big(\phi^{-1}(\bm{\mathcal{A}})\big)=0$ by $(\dagger)$. Since this holds for every $\mathrm{Leb}_{p}$-null set $\bm{\mathcal{A}}$, we conclude $\phi_{\#}\mu\ll\mathrm{Leb}_{p}$.
∎

###### Corollary C.5.1 (Preservation of absolute continuity under finitely many GD steps) .

Fix a finite vocabulary $\mathcal{V}$, a context bound $K\in\mathbb{N}$, and the Transformer language model $f$ of [Definition B.13](#A2.Thmdefinition13).
For $t=1,\ldots,T$, let $(\mathrm{s}_{t},\mathbf{p}_{t})\in\mathcal{V}^{\leq K}\times\Delta^{|\mathcal{V}|-1}$ and $\eta_{t}\in(0,1)$, and define the $t$-th GD update

$$ $\phi_{t}(\bm{\theta})\;=\;\bm{\theta}\;-\;\eta_{t}\,\nabla_{\bm{\theta}}\mathcal{L}_{\mathrm{s}_{t},\mathbf{p}_{t}}(\bm{\theta}),\qquad\mathcal{L}_{\mathrm{s}_{t},\mathbf{p}_{t}}(\bm{\theta})=\mathrm{CrossEntropy}\big(f(\mathrm{s}_{t}\,;\,\bm{\theta}),\mathbf{p}_{t}\big).$ $$

Let the $T$-step update map be the composition

$$ $\Phi\;:=\;\phi_{T}\circ\cdots\circ\phi_{1}\;:\;\mathbb{R}^{p}\to\mathbb{R}^{p}.$ $$

Then, for every absolutely continuous probability law $\mu$ on $\mathbb{R}^{p}$, its image under $\Phi$ remains absolutely continuous:

$$ $\Phi_{\#}\mu\;\ll\;\mathrm{Leb}_{p}.$ $$

Equivalently, if $\bm{\theta}^{(0)}\sim\mu$ with $\mu\ll\mathrm{Leb}_{p}$ and

$$ $\bm{\theta}^{(t+1)}\;=\;\phi_{t}\big(\bm{\theta}^{(t)}\big),\quad t=0,\ldots,T-1,$ $$

then the $T$-step parameters $\bm{\theta}^{(T)}=\Phi\big(\bm{\theta}^{(0)}\big)$ are absolutely continuous.

###### Proof.

Since the result of [Lemma C.6](#A3.Thmlemma6) holds for each $\phi_{t}$, for any null set $\bm{\mathcal{A}}$, repeated preimages remain null:

$$ $\mathrm{Leb}_{p}\big((\phi_{T}\circ\cdots\circ\phi_{1})^{-1}(\bm{\mathcal{A}})\big)=0.$ $$

The same argument as in the proof of [Theorem C.5](#A3.Thmtheorem5) then yields the claim.
∎

###### Lemma C.5 (Countable chart cover of ℝ p ∖ 𝒞 \mathbb{R}^{p}\setminus\mathcal{C} ) .

###### Proof.

###### Theorem C.4 (Change of Variables Folland 1999 , Thm. 2.47(b) ) .

###### Lemma C.6 (Pre-images of null sets are null) .

###### Proof.

###### Theorem C.5 (Preservation of absolute continuity under one GD step) .

###### Proof.

###### Corollary C.5.1 (Preservation of absolute continuity under finitely many GD steps) .

###### Proof.

## Appendix D Left-Invertibility Via SipIt

##### Goal.

We study when and how the hidden states of a causal decoder-only Transformer admit a *left inverse*: given the layer-$\ell$ representation at position $t$ and the true prefix $\pi=\mathrm{s}_{1:t-1}$, can we recover the next token $\mathrm{s}_{t}$?

##### Main idea.

Under mild randomness in the parameters and causal masking, the *one-step last-token map* that sends a candidate token $v$ to the layer-$\ell$ representation at position $t$ (conditioning on $\pi$) is almost-surely injective, and in fact has a positive separation margin. This yields a simple verifier: declare $v$ correct iff the observed hidden state lies in a small ball around $F(v;\pi,t)$.

##### Algorithmic consequence.

Because causality localizes the dependence to $(\pi,\mathrm{s}_{t})$, we can invert an entire sequence sequentially with a single pass over the vocabulary per position. We call this procedure SipIt (Sequential Inverse Prompt via ITerative updates), and we show exact (and robust) recovery holds almost surely, with worst-case time $\Theta(T|\mathcal{V}|)$.

##### Standing conventions for this section.

Fix a layer index $\ell\in[L]$. For any input sequence $\mathrm{s}=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{T}\rangle$, define the layer outputs row-wise by

$$ $\mathbf{H}^{(0)}(\mathrm{s}):=\mathrm{Emb}(\mathrm{s}),\qquad\mathbf{H}^{(\ell)}(\mathrm{s}):=\mathrm{TB}^{(\ell)}\!\big(\mathbf{H}^{(\ell-1)}(\mathrm{s})\big)\ \in\ \mathbb{R}^{T\times d},$ $$

and write $\mathbf{h}_{t}(\mathrm{s})$ to denote the row of $\mathbf{H}^{(\ell)}(\mathrm{s})$ at position $t$. Furthermore, we use $\oplus$ for sequence concatenation: if $s=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{t-1}\rangle$ and $v\in\mathcal{V}$, then $s\oplus v=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{t-1},v\rangle$.

The parameters $\bm{\theta}$ and target layer $\ell$ are considered fixed and omitted for simplicity.

###### Assumption D.1 (Causal self-attention throughout) .

Every attention layer in every block is *causal* in the sense of [Definitions B.6](#A2.Thmdefinition6) and [B.7](#A2.Thmdefinition7). Consequently, for any $\mathrm{s}$ and any $t\in[T]$,

$$ $\mathbf{h}_{t}(\mathrm{s})\;\text{depends only on the prefix}\;\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{t}\rangle.$ (34) $$

###### Assumption D.2 (Injectivity Assumption) .

SipIt is applied to models initialized with parameters drawn from an absolutely continuous distribution and trained via (mini-batch) gradient descent with step sizes in $(0,1)$, as described in Appendix [C](#A3). Under these conditions, any network considered in the sequel is almost-surely injective ([Theorem C.1](#A3.Thmtheorem1)).

###### Assumption D.1 (Causal self-attention throughout) .

###### Assumption D.2 (Injectivity Assumption) .

### D.1 One-Step Last-Token Maps

We first isolate the positionwise map that drives inversion. Fix a position $t$ and prefix $\pi\in\mathcal{V}^{t-1}$. The *one-step map* $F(\cdot;\pi,t)$ sends a candidate token $v$ to the layer-$\ell$ hidden state at position $t$ obtained when the prefix is $\pi$ and the token at $t$ is $v$. Causality implies that $\mathbf{h}_{t}$ depends only on $(\pi,v)$ (not on any future tokens), and we show that, for almost all parameter settings, $F$ is injective with a strictly positive pairwise margin over $\mathcal{V}$.

###### Definition D.1 (One-step map at time $t$ under prefix $\pi$ ) .

Let $\pi\in\mathcal{V}^{t-1}$ be a fixed prefix (possibly $t=1$, when $\pi$ is empty). Define

$$ $F:\ \mathcal{V}\longrightarrow\mathbb{R}^{d},\qquad F(v\,;\,\pi,t)\ :=\ \mathbf{h}_{t}(\mathrm{\pi}\oplus v).$ $$

###### Remark 15 .

$F$ is simply a function that returns the hidden output of token $v$ at the $\ell$ transformer block given that $\pi$ is used a fixed prefix. This map allows us to have a convenient notation for introducing results about inversion. Furthermore, since $F$ is built using $\ell$ transformer blocks, it is parameterized by $\bm{\theta}$. Nevertheless, for the sake of simplicity, we will refer to $F_{\ell,{\bm{\theta}}}$ simply as $F$.

Once the One-step map ([Definition D.1](#A4.Thmdefinition1)) is introduced, one can present its a.s. injectivity through an application of the previously obtained result ([Theorem C.1](#A3.Thmtheorem1)). Furthermore, one can deploy the common prefix to introduce a stronger notion of injectivity: margin separation ([Lemma D.1](#A4.Thmlemma1)).

###### Theorem D.1 (A.s. one-step injectivity) .

Fix $t$ and the prefix $\pi\in\mathcal{V}^{t-1}$.
Under Assumptions [D.1](#A4.Thmassumption1) and [D.2](#A4.Thmassumption2), it holds that:

$$ $\Pr\big[\;\exists v\neq v^{\prime}\in\mathcal{V}:F(v\,;\,\pi,t)=F(v^{\prime}\,;\,\pi,t)\;\big]\ =\ 0.$ $$

Equivalently, $F$ is injective almost-surely.

###### Proof.

Set the finite family
$\mathcal{S}_{t,\pi}:=\{\pi\oplus v:\ v\in\mathcal{V}\}\subseteq\mathcal{V}^{t}$ and view $\mathbf{h}_{t}(\mathrm{s})$ as the last-token representation of the *truncated* Transformer consisting of the first $\ell$ blocks. All assumptions used in [Corollary C.2.1](#A3.Thmtheorem2.Thmcorollary1) remain valid for this truncated model. Applying the corollary with $\mathcal{S}=\mathcal{S}_{t,\pi}$
yields, almost-surely, $\mathbf{h}_{t}(\mathrm{\pi}\oplus v)\neq\mathbf{h}_{t}(\mathrm{\pi}\oplus v^{\prime})$ whenever $v\neq v^{\prime}$. This is exactly the injectivity of $F$.
∎

###### Lemma D.1 (Strict separation margin a.s.) .

Under the conditions of [Theorem D.1](#A4.Thmtheorem1), define the (data-dependent) margin

$$ $\Delta_{\pi,t}\ :=\ \min_{v\neq v^{\prime}\in\mathcal{V}}\big\|F(v\,;\,\pi,t)-F(v^{\prime}\,;\,\pi,t)\big\|_{2}$ $$

Then,

$$ $\Pr[\Delta_{\pi,t}>0]=1.$ $$

###### Proof.

By [Theorem D.1](#A4.Thmtheorem1), with probability $1$ the set

$$ $\{F(v\,;\,\pi,t):v\in\mathcal{V}\}$ $$

consists of $|\mathcal{V}|$ distinct points in $\mathbb{R}^{d}$. On this event of full probability, every pairwise distance among these finitely many points is strictly positive, so their minimum is strictly positive as well.

Thus, the event $\{\Delta_{\pi,t}>0\}$ coincides with the event that $F$ is injective on $\mathcal{V}$. Since injectivity holds almost-surely by assumption, we conclude that $\Pr[\Delta_{\pi,t}>0]=1$.
∎

###### Definition D.1 (One-step map at time t t under prefix π \pi ) .

###### Remark 15 .

###### Theorem D.1 (A.s. one-step injectivity) .

###### Proof.

###### Lemma D.1 (Strict separation margin a.s.) .

###### Proof.

### D.2 The Core Routines: Local Verifiers, Acceptance Regions, and Policies

Given $F(\cdot\,;\,\pi,t)$, inversion reduces to a local hypothesis test: for an observed $\widehat{\mathbf{h}}_{t}$, which token’s predicted representation is closest? We formalize this with *acceptance regions*–closed balls around $F(v\,;\,\pi,t)$–and a *verifier* that accepts $v$ iff $\widehat{\mathbf{h}}_{t}$ lies in its ball. Almost-sure injectivity yields uniqueness at radius 0, and a positive margin yields uniqueness for any $\varepsilon<\Delta_{\pi,t}/2$. To explore candidates efficiently, we couple the verifier with any *policy* that enumerates untried tokens (e.g., uniform without replacement or a gradient-guided ranking).

###### Definition D.2 (Local verifier and acceptance tolerance) .

Given a tolerance $\varepsilon\geq 0$, define the acceptance region for symbol $v$ as the closed ball ([Definition A.8](#A1.Thmdefinition8)):

$$ $\mathcal{A}_{\pi,t}(v\,;\,\varepsilon)\ :=\ \overline{B}\big(F(v\,;\,\pi,t),\varepsilon\big).$ $$

A candidate token $v\in\mathcal{V}$ is *verified* for observation $\widehat{\mathbf{h}}_{t}$ if and only if $\;\widehat{\mathbf{h}}_{t}\in\mathcal{A}_{\pi,t}(v\,;\,\varepsilon)$.

###### Remark 16 (Decoding via acceptance regions) .

Given a prefix $\pi\in\mathcal{V}^{t-1}$ and the observation $\widehat{\mathbf{h}}_{t}$ at position $t$, we identify the next token by checking in which acceptance region $\widehat{\mathbf{h}}_{t}$ lies: declare $v$ *verified* iff $\widehat{\mathbf{h}}_{t}\in\mathcal{A}_{\pi,t}(v;\varepsilon)$.
By [Lemma D.1](#A4.Thmlemma1), for any $\varepsilon<\nicefrac{{\Delta_{\pi,t}}}{{2}}$ the regions $\{\mathcal{A}_{\pi,t}(v;\varepsilon)\}_{v\in\mathcal{V}}$ are pairwise disjoint; hence there is at most one verified token (and in the noiseless case $\varepsilon=0$, exactly one).

Building on the intuition in [Remark 16](#Thmremark16), we introduce two radii to define acceptance regions that avoid collisions:

###### Proposition D.1 (Probabilistic soundness and uniqueness of the local verifier) .

Fix position $t$ and prefix $\pi\in\mathcal{V}^{t-1}$. Under Assumptions [D.1](#A4.Thmassumption1) and [D.2](#A4.Thmassumption2), for all $v^{\star}\in\mathcal{V}$, the following hold with probability one:

- 1.
Noiseless soundness. If $\varepsilon=0$ and $\widehat{\mathbf{h}}_{t}=F(v^{\star}\,;\,\pi,t)$, then $v^{\star}$ is the unique verified symbol.
- 2.
Robust uniqueness. If $\varepsilon<\nicefrac{{\Delta_{\pi,t}}}{{2}}$ and $\widehat{\mathbf{h}}_{t}\in\mathcal{A}_{\pi,t}(v^{*}\,;\,\varepsilon)$, then $v^{\star}$ is the unique verified symbol.

###### Proof.

Recall that under Assumptions [D.1](#A4.Thmassumption1) and [D.2](#A4.Thmassumption2), $F$ is injective and $\Delta_{\pi,t}>0$ almost-surely.

*(1) Noiseless soundness.*
For any $v\in\mathcal{V}$, $\mathcal{A}_{\pi,t}(v\,;\,0)=\{F(v\,;\,\pi,t)\}$.
If $\widehat{\mathbf{h}}_{t}=F(v^{\star}\,;\,\pi,t)$ and some $v\neq v^{\star}$ were also verified at $\varepsilon=0$, we would have
$F(v\,;\,\pi,t)=F(v^{\star}\,;\,\pi,t)$, which is a probability zero event under the assumptions made. Hence $v^{\star}$ is uniquely verified almost-surely.

*(2) Robust uniqueness.*
Assume $\varepsilon<\nicefrac{{\Delta_{\pi,t}}}{{2}}$ and $\|\widehat{\mathbf{h}}_{t}-F(v^{\star}\,;\,\pi,t)\|_{2}<\varepsilon$.
If some $v\neq v^{\star}$ were also verified, then $\|\widehat{\mathbf{h}}_{t}-F(v\,;\,\pi,t)\|_{2}\leq\varepsilon$.
By the triangle inequality,

$$ $\big\|F(v\,;\,\pi,t)-F(v^{\star}\,;\,\pi,t)\big\|_{2}\ \leq\ \big\|\widehat{\mathbf{h}}_{t}-F(v\,;\,\pi,t)\big\|_{2}+\big\|\widehat{\mathbf{h}}_{t}-F(v^{\star}\,;\,\pi,t)\big\|_{2}\ <\ 2\varepsilon\ <\ \Delta_{\pi,t},$ $$

contradicting the definition of $\Delta_{\pi,t}$ (again, valid under the assumptions made). Thus $v^{\star}$ is uniquely verified almost-surely.
∎

###### Definition D.2 (Local verifier and acceptance tolerance) .

###### Remark 16 (Decoding via acceptance regions) .

###### Proposition D.1 (Probabilistic soundness and uniqueness of the local verifier) .

###### Proof.

##### Candidate enumeration.

Finally, we introduce the last conceptual block required to build the inversion algorithm: a *policy algorithm* that systematically enumerates candidate tokens so that the verifier is guaranteed to encounter the true one.

###### Definition D.3 (Policy algorithm) .

Let $\mathcal{V}$ be a finite vocabulary. A *policy algorithm* is a (possibly randomized) map

$$ $\Pi:\ \{\,\mathcal{C}\subsetneq\mathcal{V}\,\}\ \longrightarrow\ \mathcal{V}\qquad\text{such that}\qquad\Pi(\mathcal{C})\in\mathcal{V}\setminus\mathcal{C}\ \ \text{for all }\mathcal{C}\subsetneq\mathcal{V}.$ $$

(When $\mathcal{C}=\mathcal{V}$ the map is undefined.)

###### Remark 17 (Enumeration property) .

Intuitively, a policy chooses any token not tried yet. Starting from $\mathcal{C}_{0}=\varnothing$ and iterating

$$ $v_{i}:=\Pi(\mathcal{C}_{i-1}),\qquad\mathcal{C}_{i}:=\mathcal{C}_{i-1}\cup\{v_{i}\}\quad(i=1,\dots,|\mathcal{V}|),$ $$

produces a sequence $(v_{1},\dots,v_{|\mathcal{V}|})$ that is a (possibly random) permutation of $\mathcal{V}$.
Thus, in exactly $|\mathcal{V}|$ steps, every token is output once with no repetitions.

###### Definition D.3 (Policy algorithm) .

###### Remark 17 (Enumeration property) .

##### Two examples of policy algorithms.

We give (i) a *uniform-random without replacement* policy and (ii) a *gradient-guided* policy.

Figure: Algorithm 2 Policy (Random)

Figure: Algorithm 3 Policy (Gradient-based)

###### Remark 18 (Bypassing the embedding layer) .

We slightly overload notation and write $F(\mathbf{e};\pi,t)$.
Here we bypass the token embedding lookup and inject a continuous vector at the current position:
the first $t\!-\!1$ rows of $\mathbf{H}^{(0)}$ are set to $\mathrm{Emb}(\pi)$ and the $t$-th row is set to $\mathbf{e}$.
This extension is used only to guide the search (e.g., in Policy-Gradient).
All theoretical guarantees are stated for $F(v;\pi,t)$ with $v\in\mathcal{V}$ and are unaffected by allowing $F$ to accept a continuous proxy during candidate scoring.
Any extra inputs/side outputs used by a policy (such as the updated proxy) are orthogonal to the correctness statements.

###### Remark 19 (Practical choice of policy) .

Both [Algorithms 2](#alg2) and [3](#alg3) satisfy [Definition D.3](#A4.Thmdefinition3).
In practice we use the gradient-guided policy with standard gradient descent updates,
as it tends to find the verified token with far fewer proposals: the next token is chosen by ranking $\mathcal{V}$ by the distance
$\|\mathbf{E}_{v}-\mathbf{e}^{(j)}\|_{2}$ to the updated proxy $\mathbf{e}^{(j)}$.
This preserves the same worst-case guarantees (single pass over $\mathcal{V}$) while improving empirical efficiency.

###### Remark 18 (Bypassing the embedding layer) .

###### Remark 19 (Practical choice of policy) .

### D.3 Global Inversion via SipIt

We now compose the local verifier into a sequential decoder. At step $t$, causality ensures $\mathbf{h}_{t}(\mathrm{s})=F(\mathrm{s}_{t};\pi,t)$ for the true prefix $\pi=\mathrm{s}_{1:t-1}$. Since the verifier uniquely accepts $\mathrm{s}_{t}$ (noiselessly, and robustly under perturbations below half the margin), any covering policy must encounter and accept the true token within a single pass over $\mathcal{V}$. Iterating from $t=1$ to $T$ yields exact recovery almost surely; we also quantify robustness and the worst-case runtime.

We are now ready to introduce our inversion algorithm: SipIt ([Algorithm 1](#alg1)). The algorithms applies to decoder-only transformers with *causal* self-attention ([Assumption D.1](#A4.Thmassumption1)), and assumes injectivity, which occurs with almost-surely ([Assumption D.2](#A4.Thmassumption2)). We assume access to the layer-$\ell$ hidden states per position $\left\{\widehat{\mathbf{h}}_{t}\right\}_{t=1}^{T}$ and to the parameters needed to evaluate the local verifier from [Definition D.2](#A4.Thmdefinition2) for arbitrary $(t,\pi,j)$, as well as the gradient (when needed), namely to the model up to layer $\ell$. A policy algorithm is fixed (e.g., [Algorithm 3](#alg3)).

We begin by recording the following standard lemma and omitting the proof, as it is immediate from causal masking: under causal self-attention, the representation at position $t$ is independent of future tokens.

###### Lemma D.2 (Causal factorization and prefixwise identifiability) .

Under Assumptions [D.1](#A4.Thmassumption1) and [D.2](#A4.Thmassumption2), fix position $t\in[T]$. For any $\mathrm{s}=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{T}\rangle$ with $\pi=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{t-1}\rangle$,

$$ $\mathbf{h}_{t}(\mathrm{s})\;=\;F(\mathrm{s}_{t}\,;\,\pi,t),$ $$

where $F$ is the one-step map from [Definition D.1](#A4.Thmdefinition1).

###### Proof.

With causal masking, position $t$ attends only to positions $\leq t$.
Evaluating the network up to layer $\ell$ therefore yields a representation at $t$ that is a function of the prefix $\pi$ and the current token $\mathrm{s}_{t}$ only, i.e. $F(\mathrm{s}_{t}\,;\,\pi,t)$, as claimed.
∎

###### Proposition D.2 (The verifier is the right primitive) .

Fix $t$ and a true prefix $\pi=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{t-1}\rangle$. Under [Assumption D.1](#A4.Thmassumption1), the observed hidden state at step $t$ satisfies
$\mathbf{h}_{t}(\mathrm{s})=F(\mathrm{s}_{t}\,;\,\pi,t)$ ([Lemma D.2](#A4.Thmlemma2)).
In addition, under [Assumption D.2](#A4.Thmassumption2), $F$ is injective and has positive margin $\Delta_{\pi,t}>0$ almost-surely ([Theorems D.1](#A4.Thmtheorem1) and [D.1](#A4.Thmlemma1)).
Consequently, for the local verifier of [Definition D.2](#A4.Thmdefinition2), the following hold with probability one:

- 1.
(*Noiseless*) With $\varepsilon=0$ and observation $\widehat{\mathbf{h}}_{t}=\mathbf{h}_{t}(\mathrm{s})$, the unique verified token is $\mathrm{s}_{t}$.
- 2.
(*Robust*) If $\widehat{\mathbf{h}}_{t}=\mathbf{h}_{t}(\mathrm{s})+\mathbf{e}_{t}$ with
$\|\mathbf{e}_{t}\|_{2}<\varepsilon<\nicefrac{{\Delta_{\pi,t}}}{{2}}$, then $\mathrm{s}_{t}$ is the unique verified token.

###### Proof.

Immediate from [Lemma D.2](#A4.Thmlemma2) and [Proposition D.1](#A4.Thmproposition1) applied with $v^{\star}=\mathrm{s}_{t}$, which holds almost-surely by [Theorem D.1](#A4.Thmtheorem1) and [Lemma D.1](#A4.Thmlemma1).
∎

###### Proposition D.3 (Eventual acceptance under increasing enumeration) .

Fix a position $t$ and the true prefix $\pi=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{t-1}\rangle$. Under [Assumption D.1](#A4.Thmassumption1) and [Assumption D.2](#A4.Thmassumption2),
let $\varepsilon\geq 0$ and work on the probability-one event where the local verifier uniquely accepts the true token $\mathrm{s}_{t}$
(e.g., $\varepsilon=0$ or $\varepsilon<\Delta_{\pi,t}/2$; see [Proposition D.2](#A4.Thmproposition2)).

Let $\Pi$ be any policy algorithm ([Definition D.3](#A4.Thmdefinition3)). Define the increasing visited sets by
$\mathcal{C}_{0}=\varnothing$, $v_{i}:=\Pi(\mathcal{C}_{i-1})$, and $\mathcal{C}_{i}:=\mathcal{C}_{i-1}\cup\{v_{i}\}$ for $i\geq 1$,
and stop at the first index

$$ $\tau:=\min\big\{\,i\geq 1:\ \widehat{\mathbf{h}}_{t}\in\mathcal{A}_{\pi,t}(v_{i}\,;\,\varepsilon)\,\big\}.$ $$

Then $(v_{i})_{i\geq 1}$ enumerates $\mathcal{V}$ without replacement and $\tau\leq|\mathcal{V}|$ almost surely.
In particular, for the fixed prefix $\pi$, the policy’s increasingly expanding search over $\mathcal{V}$ eventually proposes the unique verified token $\mathrm{s}_{t}$ and accepts it with probability $1$.

###### Proof.

Work on the probability-one event of [Proposition D.2](#A4.Thmproposition2) (under [Assumptions D.1](#A4.Thmassumption1) and [D.2](#A4.Thmassumption2) with the stated $\varepsilon$), on which the local verifier at step $t$ uniquely accepts the true token $\mathrm{s}_{t}$. Equivalently,

$$ $\widehat{\mathbf{h}}_{t}\in\mathcal{A}_{\pi,t}(v\,;\,\varepsilon)\ \Longleftrightarrow\ v=\mathrm{s}_{t}.$ (35) $$

###### Lemma D.2 (Causal factorization and prefixwise identifiability) .

###### Proof.

###### Proposition D.2 (The verifier is the right primitive) .

###### Proof.

###### Proposition D.3 (Eventual acceptance under increasing enumeration) .

###### Proof.

##### Enumeration without replacement.

By the definition of a policy algorithm ([Definition D.3](#A4.Thmdefinition3)), $v_{i}=\Pi(\mathcal{C}_{i-1})\in\mathcal{V}\setminus\mathcal{C}_{i-1}$ and $\mathcal{C}_{i}=\mathcal{C}_{i-1}\cup\{v_{i}\}$. Hence $v_{i}\notin\mathcal{C}_{i-1}$ and $|\mathcal{C}_{i}|=|\mathcal{C}_{i-1}|+1$. Inducting on $i$ yields that $(v_{i})_{i\geq 1}$ has no repetitions and $\mathcal{C}_{i}$ contains exactly $i$ distinct tokens. Since $\mathcal{V}$ is finite, after $|\mathcal{V}|$ steps we have $\mathcal{C}_{|\mathcal{V}|}=\mathcal{V}$, i.e., $(v_{i})_{i=1}^{|\mathcal{V}|}$ is a permutation of $\mathcal{V}$ (this holds pathwise, for any realization of the policy’s internal randomness).

##### Eventual acceptance.

Because $(v_{i})$ is a permutation of $\mathcal{V}$, there exists a unique index $j\in\{1,\dots,|\mathcal{V}|\}$ with $v_{j}=\mathrm{s}_{t}$. By equation [35](#A4.E35),

$$ $\tau=\min\{\,i\geq 1:\ \widehat{\mathbf{h}}_{t}\in\mathcal{A}_{\pi,t}(v_{i}\,;\,\varepsilon)\,\}=\min\{\,i\geq 1:\ v_{i}=\mathrm{s}_{t}\,\}=j,$ $$

so $\tau\leq|\mathcal{V}|$ and the process accepts $\mathrm{s}_{t}$.

Since the event on which equation [35](#A4.E35) holds has probability $1$, the conclusion (eventual acceptance at finite $\tau$) holds almost surely.
∎

###### Theorem D.2 (Correctness of SipIt (noiseless & robust)) .

For each $t\in\{1,\ldots,T\}$ let $\pi_{t}=\langle\mathrm{s}_{1},\ldots,\mathrm{s}_{t-1}\rangle$ and let $\Delta_{\pi_{t},t}>0$ be the margin of the one-step map $F(\cdot;\pi_{t},t)$ from [Lemma D.1](#A4.Thmlemma1).
Under [Assumptions D.1](#A4.Thmassumption1) and [D.2](#A4.Thmassumption2), run SipIt ([Algorithm 1](#alg1)) with a tolerance $\varepsilon\geq 0$ and observations

$$ $\widehat{\mathbf{h}}_{t}=\mathbf{h}_{t}(\mathrm{s})+\mathbf{e}_{t}\qquad(t=1,\ldots,T),$ $$

where the perturbations satisfy $\|\mathbf{e}_{t}\|_{2}\leq\varepsilon$ for all $t$ and

$$ $\varepsilon\;<\;\tfrac{1}{2}\,\Delta_{\pi_{t},t}\qquad\text{for all }t.$ $$

Then, with probability $1$ over the model parameters:
(i) for every $t$, the *inner for-loop over $j$* (the loop over vocabulary candidates) terminates within $|\mathcal{V}|$ iterations by accepting the true token $\mathrm{s}_{t}$; and
(ii) after the *outer for-loop over $t$* (the loop over positions) finishes, the algorithm outputs the exact sequence $\widehat{\mathrm{s}}=\mathrm{s}$.

In particular, this covers the noiseless case by taking $\varepsilon=0$ and $\widehat{\mathbf{h}}_{t}=\mathbf{h}_{t}(\mathrm{s})$, and the robust case with any uniform $\varepsilon$ such that $\max_{t}\|\mathbf{e}_{t}\|_{2}\leq\varepsilon<\tfrac{1}{2}\min_{t}\Delta_{\pi_{t},t}$.

###### Proof.

By [Assumptions D.2](#A4.Thmassumption2) and [D.1](#A4.Thmtheorem1), and [Lemma D.1](#A4.Thmlemma1), there is a probability-one event on which, for all $t$, $F(\cdot;\pi_{t},t)$ is injective with strictly positive margin $\Delta_{\pi_{t},t}$. Intersecting across finitely many $t$ preserves probability 1. Work on this event.

By [Assumptions D.1](#A4.Thmassumption1) and [D.2](#A4.Thmlemma2),
$\mathbf{h}_{t}(\mathrm{s})=F(\mathrm{s}_{t};\pi_{t},t)$. Since $\|\mathbf{e}_{t}\|_{2}\leq\varepsilon$,

$$ $\widehat{\mathbf{h}}_{t}=F(\mathrm{s}_{t};\pi_{t},t)+\mathbf{e}_{t}\in\overline{B}\!\big(F(\mathrm{s}_{t};\pi_{t},t),\varepsilon\big)=\mathcal{A}_{\pi_{t},t}(\mathrm{s}_{t};\varepsilon),$ $$

so the local verifier *accepts* $\mathrm{s}_{t}$. Moreover, because $\varepsilon<\tfrac{1}{2}\Delta_{\pi_{t},t}$, [Proposition D.1](#A4.Thmproposition1)(2) implies *robust uniqueness*:

$$ $\widehat{\mathbf{h}}_{t}\in\mathcal{A}_{\pi_{t},t}(v;\varepsilon)\quad\Longleftrightarrow\quad v=\mathrm{s}_{t}.$ (36) $$

When $\varepsilon=0$, equation [36](#A4.E36) also holds by [Proposition D.1](#A4.Thmproposition1)(1). We now analyze SipIt and proceed by induction on $t$.

*Base case ($t=1$).* The *outer for-loop over $t$* begins with $\widehat{\mathrm{s}}=\langle\,\rangle=\pi_{1}$.
Inside the *inner for-loop over $j$* (the loop over vocabulary candidates), the policy ([Definition D.3](#A4.Thmdefinition3)) enumerates $\mathcal{V}$ without replacement. By [Proposition D.3](#A4.Thmproposition3), there exists $j^{\star}\leq|\mathcal{V}|$ such that $v_{j^{\star}}=\mathrm{s}_{1}$, which is accepted and triggers the break; the algorithm appends $\mathrm{s}_{1}$.

*Inductive step.* Suppose after completing the inner loop at step $t-1$ the algorithm has appended $\mathrm{s}_{t-1}$, so the prefix entering step $t$ is $\widehat{\mathrm{s}}=\pi_{t}$. By equation [36](#A4.E36), within the inner loop the verifier accepts exactly when $v_{j}=\mathrm{s}_{t}$. Because the policy enumerates $\mathcal{V}$ without replacement, some $j\leq|\mathcal{V}|$ satisfies $v_{j}=\mathrm{s}_{t}$, which is accepted, appended, and the inner loop breaks.

Thus for every $t$, the inner loop terminates by accepting $\mathrm{s}_{t}$ within $|\mathcal{V}|$ iterations, and after the outer loop finishes we have appended $(\mathrm{s}_{1},\ldots,\mathrm{s}_{T})$, i.e., $\widehat{\mathrm{s}}=\mathrm{s}$.
Since the reasoning holds on a probability-one event (independent of the policy’s internal randomness), the conclusion is almost sure.
∎

###### Theorem D.2 (Correctness of SipIt (noiseless & robust)) .

###### Proof.

##### Termination and complexity.

Having established correctness, we record the worst-case iteration count and discuss its relation to wall-clock time.

###### Proposition D.4 (Termination and linear step bound) .

Run SipIt ([Algorithm 1](#alg1)) on a length-$T$ sequence with any policy that enumerates $\mathcal{V}$ without replacement.
Then the algorithm halts after a finite number of iterations. Moreover, in the worst case the
*inner for-loop over $j$* executes at most $|\mathcal{V}|$ iterations at each position $t$, so the total number of verifier tests across the entire run is at most $T\,|\mathcal{V}|$. In particular, the number of loop iterations grows linearly with $T\cdot|\mathcal{V}|$.

###### Proof.

Fix a position $t$. The *inner for-loop over $j$* proposes unvisited tokens and stops when a candidate verifies, or after exhausting $\mathcal{V}$. Because the policy enumerates without replacement, the loop can execute at most $|\mathcal{V}|$ iterations at step $t$. The *outer for-loop over $t$* runs for exactly $T$ positions, hence the total number of inner-loop iterations (i.e., verifier tests) is at most $\sum_{t=1}^{T}|\mathcal{V}|=T|\mathcal{V}|<\infty$. Therefore the algorithm halts and the total number of tests is linear in $T\cdot|\mathcal{V}|$.
∎

###### Remark 20 (Iterations vs. wall–clock time) .

[Proposition D.4](#A4.Thmproposition4) bounds the *number of iterations/tests*: the inner loop performs at most $|\mathcal{V}|$ verifier tests per position, so the total is $\Theta(T|\mathcal{V}|)$. This is an *iteration complexity* statement that holds for any policy satisfying the “enumerate $\mathcal{V}$ without replacement” property. Actual *wall–clock time* also depends on the per–test cost (one call to $F(v;\pi,t)$ plus a distance) and on any policy overhead (e.g., forward/backward proxy updates, scoring, sorting). A generic decomposition is

$$ $\text{time}\;=\;\Theta\!\big(T|\mathcal{V}|\cdot C_{\text{test}}\big)\;+\;\sum_{t=1}^{T}C_{\text{policy}}(t),$ $$

where $C_{\text{test}}$ is the cost of one membership test and $C_{\text{policy}}(t)$ captures policy-specific work at step $t$. Thus, if $|\mathcal{V}|$ is treated as fixed and $C_{\text{test}},\,C_{\text{policy}}(t)$ are bounded (e.g., a constant number of proxy updates and at most one ranking per update), wall–clock time is $O(T)$. If $|\mathcal{V}|$ grows or the policy sorts per update, additional factors like $|\mathcal{V}|$ or $\log|\mathcal{V}|$ may appear in the time, but the termination and the $\Theta(T|\mathcal{V}|)$ *iteration* bound remain unchanged.

###### Remark 21 (Choosing the tolerance $\varepsilon$ ) .

Theory guarantees uniqueness whenever $\varepsilon<\tfrac{1}{2}\Delta_{\pi,t}$ ([Proposition D.1](#A4.Thmproposition1)).
Since $\Delta_{\pi,t}$ is unknown, two practical choices work well:
(i) *backoff*: start with a small $\varepsilon$ and increase only if no token verifies;
(ii) *calibration*: set $\varepsilon$ from held-out hidden states at layer $\ell$.
In all cases the decision rule remains a simple yes/no membership test.

###### Remark 22 (Why SipIt is sequential) .

The algorithm never solves a global assignment.
At position $t$ it conditions on the current prefix $\pi$ and queries the local verifier for a single token.
Causality ([Assumption D.1](#A4.Thmassumption1)) ensures $\mathbf{h}_{t}$ depends only on $(\pi,\mathrm{s}_{t})$, so these local, prefixwise decisions compose to recover the full sequence.

###### Proposition D.4 (Termination and linear step bound) .

###### Proof.

###### Remark 20 (Iterations vs. wall–clock time) .

###### Remark 21 (Choosing the tolerance ε \varepsilon ) .

###### Remark 22 (Why SipIt is sequential) .

## Appendix E Implementation Details and Additional Experiments

Figure: Figure 7: Seeking collisions in a large-scale prompt set (§[4.1](#S4.SS1)). For each layer, boxplots show the distribution (log scale) of the *minimum pairwise* $\ell_{2}$ distances between last-token states across prompts for the GPT-2 model family (Small, Medium, and Large); red bars mark medians and the dashed line indicates the collision threshold $10^{-6}$.
Refer to caption: https://arxiv.org/html/2510.15511/2510.15511v4/x2.png

This section collects implementation details (§[E.1](#A5.SS1)), additional ablations on collision experiments and SipIt (§[E.2](#A5.SS2)), controlled experiments with identical next-token predictions (§[E.3](#A5.SS3)), qualitative inspection of the closest hidden-state pairs (§[E.4](#A5.SS4)), and connections to anisotropy and intrinsic dimension (§[E.5](#A5.SS5)).

### E.1 Implementation Details

##### What is a collision in practice.

In the theoretical parts of the paper we use “collision” in the usual functional sense: two distinct prompts
$\mathrm{s}\neq\mathrm{s}^{\prime}$ such that their last-token representations coincide exactly,

$$ $\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T})=\mathbf{r}(\mathrm{s}^{\prime}\,;\,\bm{\theta}_{T}).$ $$

This is the event whose probability is controlled in [theorems 2.2](#S2.Thmtheorem2) and [2.3](#S2.Thmtheorem3) and in [Appendix C](#A3), and all proofs are
carried out at the level of exact equality (no numerical threshold is required).

In the experiments, however, representations are stored in floating-point format, so exact equality of
$\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T})$ and $\mathbf{r}(\mathrm{s}^{\prime}\,;\,\bm{\theta}_{T})$ may not be a meaningful or robust criterion. We therefore adopt a numerical proxy: given two prompts $\mathrm{s},\mathrm{s}^{\prime}$ and their embeddings
$\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T}),\mathbf{r}(\mathrm{s}^{\prime}\,;\,\bm{\theta}_{T})\in\mathbb{R}^{d}$, we declare a *practical collision* only if

$$ $\texttt{torch.allclose}\bigl(\mathbf{r}(\mathrm{s}\,;\,\bm{\theta}_{T}),\mathbf{r}(\mathrm{s}^{\prime}\,;\,\bm{\theta}_{T})\bigr)=\texttt{True},$ $$

i.e., every coordinate falls within PyTorch’s prescribed tolerances, namely $10^{-5}$ and $10^{-8}$ for relevant and absolute tolerance respectively. Across all of the billions to trillions of empirical checks, every pair of distinct prompts $\mathrm{s}\neq\mathrm{s}^{\prime}$ failed this
criterion: torch.allclose always returned False, and the observed $\ell_{2}$ distances were
consistently bounded away from zero. Thus, at the resolution of our numerical precision, we did not observe any
collisions in practice.

##### SipIt implementation.

We implement SipIt exactly as in Alg. [1](#alg1) with the gradient-guided policy. To stabilize the continuous proxy used for ranking, we apply gradient clipping (capping the gradient norm at 1) and we periodically project it back to the nearest token embedding every $K\!=\!50$ candidate proposals:

$$ $\mathbf{e}^{(j)}\;\leftarrow\;\mathbf{E}_{v^{\dagger}},\qquad v^{\dagger}\;=\;\arg\min_{v\in\mathcal{V}\setminus\mathcal{C}}\big\|\mathbf{E}_{v}-\mathbf{e}^{(j)}\big\|_{2},$ $$

without taking gradients through this projection. These heuristics affect efficiency only; the verifier and all correctness guarantees remain unchanged.

##### HardPrompts implementation.

The original HardPrompts method Wen et al. (2023) targets multimodal vision-language models and optimizes prompts via a CLIP-based similarity objective. In our text-only setting we lack the vision branch and CLIP loss, so we adapt Algorithm 1 of Wen et al. (2023) to language models by replacing the objective with the same $\ell_{2}$ loss used in SipIt’s gradient calculation, and setting the optimization steps $T=\tfrac{1}{4}\text{\# tokens}\cdot|\mathcal{V}|$. All other details (step sizes, stopping rules) mirror our SipIt setup to ensure a fair comparison.

Figure: Figure 8: Seeking collisions in a large-scale prompt set (§[4.1](#S4.SS1)). For each layer, boxplots (log scale) show the distribution of *minimum pairwise* $\ell_{2}$ distances between last-token states across prompts for the Gemma-3 model family (1B, 4B, 12B); red bars denote medians and the dashed line marks the collision threshold $10^{-6}$.
Refer to caption: https://arxiv.org/html/2510.15511/2510.15511v4/x3.png

### E.2 Additional Ablations

#### E.2.1 Collision Experiments

We report three complementary ablations that probe how separation behaves across depth, length, and model family.

GPT-2 family across depth.
For GPT-2 Small, GPT-2 Medium, and GPT-2 Large, the per-layer boxplots (log scale) of the *minimum pairwise* $\ell_{2}$ distances between last-token states in [Figure 7](#A5.F7) show that all minima sit orders of magnitude above the collision threshold $10^{-6}$ at every depth, and the typical separation *increases with depth* (median red bars drift upward). This rules out collisions in practice and indicates that deeper blocks monotonically sharpen last-token distinctions in these models.

Figure: Figure 9: Sequence length versus distance over all pairs of distinct prompts for Gemma-1B.
Refer to caption: https://arxiv.org/html/2510.15511/2510.15511v4/x4.png

Gemma-3 family across depth and scale.
Across Gemma3-1B, Gemma3-4B, and Gemma3-12B, the layerwise boxplots (log scale) in [Figure 8](#A5.F8) again show minima far above $10^{-6}$ at all depths. Both depth *and* model size trend positively with separation: medians and lower whiskers move upward in deeper layers and larger models, indicating progressively stronger margins and no observed collisions.

Effect of sequence length (Gemma-1B).
Varying the prompt length reveals that min/mean/max pairwise distances rise quickly for short sequences and then plateau, with the minimum never approaching zero (see [Figure 9](#A5.F9)). This suggests that beyond a modest context size, additional tokens do not erode separability; margins stabilize rather than collapse, making collisions unlikely for any prompt length explored.

Overall, these ablations corroborate the main text: last-token states remain well-separated across architectures and depths, separation typically grows with depth (and scale for Gemma), and margins stabilize with sequence length, aligning with our almost-sure injectivity guarantees and with SipIt’s exact recovery behavior.

#### E.2.2 SipIt

**Table 6: Performance of SipIt on different vocabulary sizes**
| Model | Vocab size | Inversion performance |  |  |
| --- | --- | --- | --- | --- |
| Accuracy | Time (s) | Vocab explored (%) |  |  |
| Mistral-7B-v0.1 | 32000 | 100% | 72.99 $\pm$ 37.57 | 0.21 $\pm$ 0.11 % |
| Llama-3.1-8B | 128255 | 100% | 345.35 $\pm$ 181.30 | 0.22 $\pm$ 0.12 % |

##### Vocabulary Size.

To further validate our findings (as presented in [Section 4](#S4)) regarding the scaling of SipIt with vocabulary size, we conducted additional experiments on the two models with substantially different vocabulary sizes, Mistral-7B-v0.1 ($\approx 32K$ vocabulary) and Llama-3.1-8B ($\approx 128K$). For a fair comparison, we construct sentences that tokenize to exactly the same sequence of tokens across both models. The results are reported in the [Table 6](#A5.T6). We observe that, in practice, the inversion time grows linearly with vocabulary size, as expected, reflected by the nearly constant percentage of tokens explored between the small-vocabulary model (Mistral) and the larger-vocabulary model (Llama). Importantly, for both models, the fraction of tokens explored remains below $0.25\%$, indicating that the gradient-based heuristic is both robust and highly efficient.

**Table 7: Performance of SipIt on in-distribution vs. out-of-distribution data**
| Dataset | Inversion Time (s) | Accuracy |
| --- | --- | --- |
| Train Data | 146.48 $\pm$ 91.52 | 100% |
| Test Data | 128.62 $\pm$ 83.40 | 100% |
| OOD | 106.87 $\pm$ 39.10 | 100% |

##### Robustness of SipIt on unseen and random sequences.

Based on GPT-2, we constructed three datasets, which we refer to as Train, Test, and OOD (Out-of-Distribution). The Train set is formed by sampling sentences from WebText (the dataset used to train GPT-2 Radford et al. (2019)); the Test set contains sentences sampled from Wikipedia (not in the training set); and the OOD set consists of random token sequences. Each dataset contains 50 prompts of length 100 tokens. We report the findings in [Table 7](#A5.T7). Interestingly, the OOD samples are significantly faster to invert than the Train and Test samples. We hypothesize that this difference stems from the geometry of the hidden representations: natural language sentences (Train and Test) tend to lie on a structured, clustered manifold, which can make the inversion landscape locally flatter and less well-conditioned. In contrast, random token sequences produce more dispersed and isolated hidden states, yielding clearer descent directions and effectively stronger gradient signals, which accelerates convergence. Across all three datasets, we obtain exact recovery for every sequence, further supporting the theoretical guarantees of SipIt.

### E.3 Identical Next-Token

The collision and ablation experiments above use generic prompt sets. A natural stress test is to ask: what happens when we *deliberately* construct prompt pairs that produce the same next-token prediction? To answer this question we designed a set of new experiments where two different prompt are specifically constructed to yield the exact same target answer. First, we focused on word-to-word machine translation (google/smol) and math tasks (ProCreations/SimpleMath) on Llama-3.1-8B, Mistral-7B, and Phi-4-mini-instruct. From these datasets, we built few shot prompts that differed only in their delimiters (e.g. -> vs :) while preserving identical translations or arithmetic solutions. Some qualitative examples are shown below:

We then assessed collisions involving four different separator token embeddings across all dataset pairs, specifically ->, :, =, and -. Despite producing the exact same answer the corresponding embeddings remain clearly distinct (no “collision”) since the minimum $\ell_{2}$ distance is well above the collision threshold over the $\approx 140K$ possible pairs, as seen in [tables 9](#A5.T9) and [9](#A5.T9).

**Table 8: Distances for Translation (En–Fr) separator-token embeddings across layers.**
| Model | $\bm{\ell}_{\mathbf{2}}$ Distance (min) |  |  |
| --- | --- | --- | --- |
| layer 1 | layer $L/2$ | layer $L$ |  |
| Llama-3.1-8B | 0.694 | 1.632 | 4.202 |
| Mistral-7B-v0.1 | 0.207 | 1.056 | 2.348 |
| Phi-4-mini-instruct | 4.375 | 6.974 | 17.328 |

Furthermore, we constructed a dataset of random prefixes sampled from internet text, each followed by the fixed suffix “Complete this: The quick brown fox jumps over the lazy”. To build the dataset, we sampled 10K prefix sequences of length 50 tokens from Wikipedia and appended the tokenized suffix to each. The minimum $\ell_{2}$ distances obtained are reported in [Table 10](#A5.T10). Even in this setting, although the next token prediction is exactly “dog”, all last-token embeddings remain far above the tolerance threshold.

**Table 10: Distances for random-prefix dataset with fixed “quick brown fox” suffix.**
| Model | $\bm{\ell}_{\mathbf{2}}$ Distance (min) |  |  |
| --- | --- | --- | --- |
| layer 1 | layer $L/2$ | layer $L$ |  |
| Mistral-7B-v0.1 | 0.012 | 0.265 | 3.494 |
| Llama-3.1-8B | 0.046 | 0.733 | 6.227 |
| Phi-4-mini-instruct | 0.087 | 2.302 | 18.913 |

### E.4 Prompts with Similar Representations

To complement the quantitative injectivity results in the main text, we inspected qualitative examples of sequences whose last-token hidden states are among the closest we observed. For a given model, we computed the Euclidean distance between last-layer representations $h_{L}(s)$ and $h_{L}(t)$ of the final token in two sequences $s$ and $t$, and manually examined pairs with the smallest $\ell_{2}$ distances.

For both Llama-3.1-8B and Mistral-7B-v0.1, the closest pairs correspond to Python code snippets that are almost identical, typically differing only by a small shift such as one or more trailing newline tokens. In most of the close pairs we examined, the two sequences satisfy

$$ $\mathrm{s}\;=\;\mathrm{t}\circ\langle\text{new line token}\rangle^{k}$ $$

for some small $k\geq 1$. Even in these extremal cases, however, the last-token representations remain clearly separated in $\ell_{2}$ distance.

##### Llama-3.1-8B.

One of the closest pairs we found for Llama-3.1-8B is shown below. The only difference between the two sequences is the presence of several trailing newline characters at the end of the second snippet.
The last-token $\ell_{2}$ distance at the final layer for this pair is $1.274$, which is small relative to typical distances but still far from zero, and thus consistent with the absence of collisions observed in our exhaustive tests.

##### Mistral-7B-v0.1.

A similar phenomenon occurs for Mistral-7B-v0.1. Again, one of the closest pairs consists of two almost identical code snippets, where the second sequence appends a single newline token:
For this pair, the last-token $\ell_{2}$ distance at the last layer is $1.146$. As in the Llama example, the nearest neighbors arise from almost identical contexts differing only in trailing whitespace tokens, and even these extremal cases exhibit a non-negligible separation in representation space.

##### Summary.

Across all models and pairs we inspected, we did not observe qualitatively different prompts whose last-layer, last-token embeddings were comparably close. Instead, the nearest neighbors consistently involved near-duplicate snippets (often code or documentation) differing only by whitespace or other minor formatting tokens. These qualitative observations align with the injectivity margins reported in the main text and support the view that small perturbations in formatting do not lead to collisions in the representations used by Sipit.

### E.5 Relation with Anisotropy and Intrinsic Dimension

**Table 11: Layer-wise anisotropy, intrinsic dimension, and injectivity margin.**
| Layer | Anisotropy (mean) | ID (mean) | Margin (min) |
| --- | --- | --- | --- |
| 1 | 0.089579 | 20.754620 | 1.850306 |
| 2 | 0.076049 | 17.565538 | 1.956753 |
| 3 | 0.071429 | 16.765265 | 2.064488 |
| 4 | 0.075067 | 16.679382 | 2.241199 |
| 5 | 0.083282 | 17.183246 | 2.382355 |
| 6 | 0.089542 | 17.697870 | 2.499817 |
| 7 | 0.088463 | 17.018419 | 2.704958 |
| 8 | 0.083261 | 16.296431 | 2.886434 |
| 9 | 0.081803 | 16.040713 | 3.025268 |
| 10 | 0.083083 | 15.730601 | 3.330774 |
| 11 | 0.090206 | 15.635035 | 3.918343 |
| 12 | 0.288352 | 16.434897 | 4.640457 |

As part of our broader investigation, we also examined connections to the analyses presented in the works of Razzhigaev et al. (2025) (LLM-Microscope) and Razzhigaev et al. (2024), and ran a targeted experiment in this spirit.

##### Experimental setup.

We performed a proof-of-concept study using GPT-2 Small. We sampled 100 natural-language prompts of fixed length $K$ and, for each prompt, generated 1000 single-token continuations by appending each token from a fixed vocabulary subset of size 1000. For every layer $\ell$, we extracted the hidden representation of the last token for all 1000 continuations, producing a $1000\times d$ matrix for each (layer, prompt) pair. On each matrix we computed (i) anisotropy and intrinsic dimension as in LLM-Microscope, and (ii) simple “injectivity margin” statistics: the minimum pairwise Euclidean distance between continuation embeddings, averaged over prompts. Aggregating over the 100 prompts yields, for each layer, a triple consisting of anisotropy, intrinsic dimension, and injectivity margin.

##### Experiment 1: anisotropy vs. injectivity margin.

Across layers, we correlated mean anisotropy with the mean injectivity margin. The resulting Pearson correlation is 0.72, and the Spearman correlation is 0.45. In this setting, layers with higher anisotropy tend to exhibit larger injectivity margins: continuation clouds become both more structured (anisotropic) and farther from collisions. This suggests that anisotropy is compatible with, and may even reinforce, numerically robust injectivity.

##### Experiment 2: intrinsic dimension vs. injectivity margin.

Repeating the analysis with intrinsic dimension, we observe a Pearson correlation of -0.60 and a Spearman correlation of -0.79 between intrinsic dimension and injectivity margin. Thus, layers with lower intrinsic dimensionality tend to have larger margins: compressed-looking manifolds are, if anything, more separated. This aligns with our theorem that injectivity rules out information-destroying collapses.

##### Discussion.

This line of analysis is highly complementary to our injectivity framework. Whereas our results establish that internal representations are almost surely lossless, LLM-Microscope offers fine-grained geometric diagnostics of how these representations evolve across depth and training. Particularly notable is the observation that anisotropy and intrinsic dimension follow a reverse-U profile: representations become more anisotropic and lower-dimensional in intermediate layers, then partially re-expand near the output, offering a concrete geometric picture of how structure is carved into aligned directions and low-dimensional manifolds.

This is especially relevant given that our paper challenges classic accounts of learning via bottleneck compression (e.g. Shwartz-Ziv and Tishby (2017)). If information is preserved along the residual stream, learning cannot proceed layer by layer purely through compression. Our preliminary experiments suggest a different picture: as depth increases, margins grow, intrinsic dimension decreases, and anisotropy follows a concave trajectory with a late spike. Early layers expand and reorganize, intermediate layers carve information into low-dimensional directional manifolds, and upper layers sharpen this structure. Overall, this is consistent with a network that preserves injectivity while funneling information into increasingly structured, well-separated representations.

## Appendix F Real-Analytic Activation Functions in Modern LLMs

**Table 12: Activation functions used in the feed-forward networks of representative modern LLMs.**
| Model (HF example) | Activation in FFN | Real-analytic? |
| --- | --- | --- |
| Llama-2 | SwiGLU | Yes |
| Llama-3 | SwiGLU | Yes |
| Mistral-7B-v0.1 | SiLU | Yes |
| Mixtral-8x7B-v0.1 | SiLU | Yes |
| Gemma | GeGLU | Yes |
| Gemma-2 | GELU | Yes |
| Qwen2MoE | SwiGLU | Yes |
| Qwen-2 | SiLU | Yes |
| Qwen3MoE | SiLU | Yes |
| Qwen-3 | SiLU | Yes |
| Phi | GELU | Yes |
| Phi-3 | SiLU | Yes |
| GPT-2 | GELU | Yes |
| GPT-J | GELU | Yes |
| GptOss | SiLU | Yes |
| Grok-1 | GELU | Yes |
| DeepSeek-V2 | SiLU | Yes |
| DeepSeek-V3 | SiLU | Yes |

A natural question raised by our analysis is to what extent modern large language models actually use real-analytic activation functions in their feed-forward networks. Since our results apply most directly when the non-linearities are real-analytic, it is important to check whether this assumption holds in practice.

To get a concrete picture, we surveyed a set of widely used open-source and proprietary-style architectures and recorded the activation function used in their feed-forward blocks. The models and their reported activations are summarized in [Table 12](#A6.T12). For each model, we also indicate whether the activation is real-analytic. Activations such as SiLU/Swish, SwiGLU, GeGLU, and GELU are all real-analytic, being compositions and products of elementary analytic functions (e.g., linear maps, exponentials, and the error function).

Across this representative sample, we find that *all* models (18 out of 18) use real-analytic activations in their feed-forward blocks. In other words, the analyticity assumption is not merely a technical convenience but accurately reflects common design practice. This supports the relevance of our theoretical results for real-world large language models: the vast majority of modern transformers already operate in a regime where the non-linearities are real-analytic, and hence fall directly within the scope of our analysis. We now formally prove that SiLU and GELU are real-analytic scalar functions, and
that the corresponding gated constructions SwiGLU and GeGLU define real-analytic vector-valued
maps. The proofs build from elementary ingredients upward: sigmoid ([Proposition F.1](#A6.Thmproposition1)) $\to$ SiLU ([Proposition F.2](#A6.Thmproposition2)); error function ([Proposition F.3](#A6.Thmproposition3)) $\to$ GELU ([Proposition F.4](#A6.Thmproposition4)); then coordinatewise lifting ([Proposition F.5](#A6.Thmproposition5)) and GLU gating ([Proposition F.6](#A6.Thmproposition6)).

###### Proposition F.1 (Logistic sigmoid is real-analytic) .

The logistic sigmoid

$$ $\sigma(x)\;:=\;\frac{1}{1+e^{-x}},\qquad x\in\mathbb{R},$ $$

is real-analytic on $\mathbb{R}$.

###### Proof.

By [Proposition A.5](#A1.Thmproposition5), the map $x\mapsto e^{-x}$ is real-analytic on $\mathbb{R}$.
By [Proposition A.1](#A1.Thmproposition1), the sum $x\mapsto 1+e^{-x}$ is real-analytic; moreover
$1+e^{-x}>0$ for all $x\in\mathbb{R}$, so it never vanishes.
By the quotient rule in [Proposition A.1](#A1.Thmproposition1), the reciprocal

$$ $x\mapsto\frac{1}{1+e^{-x}}$ $$

is therefore real-analytic on $\mathbb{R}$.
∎

###### Proposition F.2 (SiLU / Swish is real-analytic) .

The SiLU (or Swish) activation

$$ $\mathrm{SiLU}(x)\;:=\;x\,\sigma(x)\;=\;\frac{x}{1+e^{-x}},\qquad x\in\mathbb{R},$ $$

is real-analytic on $\mathbb{R}$.

###### Proof.

The identity map $x\mapsto x$ is a polynomial, hence real-analytic by [Proposition A.4](#A1.Thmproposition4).
By [Proposition F.1](#A6.Thmproposition1), $\sigma$ is real-analytic.
The product of two real-analytic functions is real-analytic by [Proposition A.1](#A1.Thmproposition1), so
$x\mapsto x\,\sigma(x)$ is real-analytic on $\mathbb{R}$.
∎

###### Proposition F.3 (Error function is real-analytic) .

The error function

$$ $\operatorname{erf}(x)\;:=\;\frac{2}{\sqrt{\pi}}\int_{0}^{x}e^{-t^{2}}\,dt,\qquad x\in\mathbb{R},$ $$

is real-analytic on $\mathbb{R}$.

###### Proof.

By [Proposition A.5](#A1.Thmproposition5), $\exp$ is real-analytic on $\mathbb{R}$ with power series
$e^{z}=\sum_{k=0}^{\infty}\frac{z^{k}}{k!}$ and infinite radius of convergence.
Substituting $z=-t^{2}$ yields

$$ $e^{-t^{2}}=\sum_{k=0}^{\infty}\frac{(-1)^{k}}{k!}t^{2k},\qquad t\in\mathbb{R}.$ $$

This series has infinite radius of convergence, so it converges uniformly on every bounded interval.
By standard results on termwise integration of power series (e.g. Rudin 1976), we may integrate termwise:

$$ $\int_{0}^{x}e^{-t^{2}}\,dt=\sum_{k=0}^{\infty}\frac{(-1)^{k}}{k!}\int_{0}^{x}t^{2k}\,dt=\sum_{k=0}^{\infty}\frac{(-1)^{k}}{k!(2k+1)}\,x^{2k+1}.$ $$

Multiplying by $2/\sqrt{\pi}$ we obtain

$$ $\operatorname{erf}(x)=\frac{2}{\sqrt{\pi}}\sum_{k=0}^{\infty}\frac{(-1)^{k}}{k!(2k+1)}\,x^{2k+1},$ $$

a power series with infinite radius of convergence. Hence $\operatorname{erf}$ is real-analytic on $\mathbb{R}$ by [Definition A.1](#A1.Thmdefinition1).
∎

###### Proposition F.4 (GELU is real-analytic) .

Let

$$ $\Phi(x)\;:=\;\frac{1}{2}\Big(1+\operatorname{erf}\!\big(\tfrac{x}{\sqrt{2}}\big)\Big)$ $$

be the CDF of a standard normal random variable.
The (exact) GELU activation

$$ $\mathrm{GELU}(x)\;:=\;x\,\Phi(x)$ $$

is real-analytic on $\mathbb{R}$.

###### Proof.

By [Proposition F.3](#A6.Thmproposition3), $\operatorname{erf}$ is real-analytic.
The map $x\mapsto\tfrac{x}{\sqrt{2}}$ is linear, hence real-analytic; by [Proposition A.2](#A1.Thmproposition2), the composition
$x\mapsto\operatorname{erf}\big(\tfrac{x}{\sqrt{2}}\big)$ is real-analytic.
Adding the constant $1$ and scaling by $\tfrac{1}{2}$ preserves real-analyticity by [Proposition A.1](#A1.Thmproposition1), so $\Phi$ is real-analytic.
The identity map $x\mapsto x$ is a polynomial ([Proposition A.4](#A1.Thmproposition4)), hence real-analytic; their product
$x\mapsto x\,\Phi(x)$ is therefore real-analytic by [Proposition A.1](#A1.Thmproposition1).
∎

Having established that SiLU and GELU are real-analytic as scalar functions, we now lift them to the vector-valued setting and show that the GLU-style gating used in modern architectures preserves real-analyticity.

###### Proposition F.5 (Vector-valued SiLU and GELU are real-analytic) .

Let $m\in\mathbb{N}$. Define the coordinatewise maps

$$ $\mathrm{SiLU}_{m}(\mathbf{x}):=\big(\mathrm{SiLU}(\mathbf{x}_{1}),\ldots,\mathrm{SiLU}(\mathbf{x}_{m})\big)^{\top},\quad\mathrm{GELU}_{m}(\mathbf{x}):=\big(\mathrm{GELU}(\mathbf{x}_{1}),\ldots,\mathrm{GELU}(\mathbf{x}_{m})\big)^{\top},$ $$

for $\mathbf{x}\in\mathbb{R}^{m}$, where SiLU and GELU are as in [Proposition F.2](#A6.Thmproposition2) and [Proposition F.4](#A6.Thmproposition4).
Then both $\mathrm{SiLU}_{m}$ and $\mathrm{GELU}_{m}$ are real-analytic maps $\mathbb{R}^{m}\to\mathbb{R}^{m}$.

###### Proof.

Each scalar component $\mathbf{x}\mapsto\mathrm{SiLU}(\mathbf{x}_{i})$ (resp. $\mathrm{GELU}(\mathbf{x}_{i})$) is the composition of the projection onto coordinate $i$ (a linear map) with the real-analytic scalar function SiLU (resp. GELU). By [Proposition A.2](#A1.Thmproposition2), each component is real-analytic.
Therefore, by [Definition A.1](#A1.Thmdefinition1), the vector-valued maps $\mathrm{SiLU}_{m}$ and $\mathrm{GELU}_{m}$ are real-analytic.
∎

###### Proposition F.6 (GLU-style blocks are real-analytic) .

Let $d_{\mathrm{in}},d_{\mathrm{hid}}\in\mathbb{N}$ and consider affine maps

$$ $A_{1}(\mathbf{x})=\mathbf{W}_{1}\mathbf{x}+\mathbf{b}_{1},\qquad A_{2}(\mathbf{x})=\mathbf{W}_{2}\mathbf{x}+\mathbf{b}_{2},$ $$

with $\mathbf{W}_{1},\mathbf{W}_{2}\in\mathbb{R}^{d_{\mathrm{hid}}\times d_{\mathrm{in}}}$ and
$\mathbf{b}_{1},\mathbf{b}_{2}\in\mathbb{R}^{d_{\mathrm{hid}}}$.
Let $\phi:\mathbb{R}^{d_{\mathrm{hid}}}\to\mathbb{R}^{d_{\mathrm{hid}}}$ be either $\mathrm{SiLU}_{d_{\mathrm{hid}}}$ or $\mathrm{GELU}_{d_{\mathrm{hid}}}$ from [Proposition F.5](#A6.Thmproposition5).
Define the GLU-style block

$$ $\mathrm{GLU}_{\phi}(\mathbf{x})\;:=\;A_{1}(\mathbf{x})\odot\phi\big(A_{2}(\mathbf{x})\big),\qquad\mathbf{x}\in\mathbb{R}^{d_{\mathrm{in}}},$ $$

where $\odot$ denotes the Hadamard product.

Then $\mathrm{GLU}_{\phi}:\mathbb{R}^{d_{\mathrm{in}}}\to\mathbb{R}^{d_{\mathrm{hid}}}$ is real-analytic.
In particular:

- •
Taking $\phi=\mathrm{SiLU}_{d_{\mathrm{hid}}}$ recovers SwiGLU, which is real-analytic.
- •
Taking $\phi=\mathrm{GELU}_{d_{\mathrm{hid}}}$ recovers GeGLU, which is real-analytic.

###### Proof.

Each affine map $A_{j}$ is real-analytic as a matrix product plus addition ([Proposition A.10](#A1.Thmproposition10), [Proposition A.1](#A1.Thmproposition1)).
By [Proposition F.5](#A6.Thmproposition5), $\phi$ is real-analytic, so $\mathbf{x}\mapsto\phi(A_{2}(\mathbf{x}))$ is a composition of real-analytic maps ([Proposition A.2](#A1.Thmproposition2)), hence real-analytic.
The map $\mathbf{x}\mapsto A_{1}(\mathbf{x})\odot\phi(A_{2}(\mathbf{x}))$ is a Hadamard product of two real-analytic vector-valued functions; componentwise this is just the product of real-analytic scalars, so it is real-analytic by [Proposition A.1](#A1.Thmproposition1) (equivalently, by [Proposition A.11](#A1.Thmproposition11)).
Thus $\mathrm{GLU}_{\phi}$ is real-analytic. The SwiGLU and GeGLU cases follow by choosing $\phi$ accordingly.
∎

###### Proposition F.1 (Logistic sigmoid is real-analytic) .

###### Proof.

###### Proposition F.2 (SiLU / Swish is real-analytic) .

###### Proof.

###### Proposition F.3 (Error function is real-analytic) .

###### Proof.

###### Proposition F.4 (GELU is real-analytic) .

###### Proof.

###### Proposition F.5 (Vector-valued SiLU and GELU are real-analytic) .

###### Proof.

###### Proposition F.6 (GLU-style blocks are real-analytic) .

###### Proof.

##### Relation to universal-approximation and expressivity results.

The material above concerns only the analyticity of the non-linearities used in our analysis. For completeness, we also record here how our injectivity theorem fits alongside existing expressivity results for Transformers; this discussion is logically independent of the real-analyticity assumptions.

Classical expressivity results for Transformers are primarily *existential*.
Universal-approximation theorems (e.g. Yun et al. (2020); Sun and Yang (2020)) show that for any continuous sequence-to-sequence function $f$ on a compact domain and any $\varepsilon>0$, there exists a Transformer with suitable depth and width whose outputs are within $\varepsilon$ of those of $f$.
Turing-completeness results for encoder–decoder Transformers (e.g. Pérez et al., 2019) similarly establish the existence of parameter settings that simulate any Turing machine.
Taken together, these works characterise what the architecture can represent *in principle*: they do not model random initialization or gradient-based training, and they are not formulated in our discrete setting with finite context length, fixed decoder-only architecture, and real-analytic activations.

Our results are complementary and instead concern what happens *typically* under standard training.
We fix a concrete decoder-only architecture and a finite prompt set, and study the map from prompts to last-token representations.
In this setting we prove that (i) for any fixed architecture, the set of parameters for which this map is non-injective has Lebesgue measure zero, and (ii) gradient-based training from standard random initializations preserves absolute continuity of the parameter distribution and therefore almost surely avoids this “collision set”.
Non-injective Transformers certainly exist (we explicitly construct such failure cases in [Section 2](#S2)), but our results show that they form a thin subset that typical optimization trajectories do not reach.

Our contribution is thus orthogonal to prior expressivity theory.
We do *not* claim that Transformers can only represent injective functions.
Rather, within the specific regime we study (decoder-only, real-analytic activations, cross-entropy loss, GD-type training from standard initialization), we show that the resulting last-token map is injective with probability one over initialization and training.
In short, classical expressivity results describe what is mathematically *possible* for the Transformer function class, while our analysis characterizes what is *almost surely implemented* when that class is explored via standard training procedures.