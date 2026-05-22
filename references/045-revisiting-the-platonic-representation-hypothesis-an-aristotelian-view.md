# Revisiting the Platonic Representation Hypothesis: An Aristotelian View

Source: https://arxiv.org/abs/2602.14486
Fetched URL: https://arxiv.org/pdf/2602.14486
Source type: arxiv-pdf

---

## Page 1

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
Fabian Gr¨oger * 1 2 3 Shuo Wen * 1 Maria Brbi´c 1
Abstract
The Platonic Representation Hypothesis suggests
that representations from neural networks are con-
verging to a common statistical model of reality.
We show that the existing metrics used to mea-
sure representational similarity are confounded by
network scale: increasing model depth or width
can systematically inflate representational similar-
ity scores. To correct these effects, we introduce
a permutation-based null-calibration framework
that transforms any representational similarity
metric into a calibrated score with statistical guar-
antees. We revisit the Platonic Representation Hy-
pothesis with our calibration framework, which re-
veals a nuanced picture: the apparent convergence
reported by global spectral measures largely disap-
pears after calibration, while local neighborhood
similarity, but not local distances, retains signifi-
cant agreement across different modalities. Based
on these findings, we propose the Aristotelian Rep-
resentation Hypothesis: representations in neural
networks are converging to shared local neighbor-
hood relationships.
1. Introduction
Quantifying the similarity between neural network rep-
resentations is central to understanding the geometry of
learned representation spaces (Raghu et al., 2017; Nguyen
et al., 2021), guiding transfer learning decisions (Korn-
blith et al., 2019; Neyshabur et al., 2020), and relating
artificial representations to neural measurements in neuro-
science (Schrimpf et al., 2018). The Platonic Representation
Hypothesis (Huh et al., 2024) posits that as neural networks
scale, representations across different modalities become
increasingly similar, suggesting convergence to a shared
statistical model of reality. This hypothesis has motivated
a growing literature that uses representational similarity to
Project Page: brbiclab.epfl.ch/aristotelian
Code: github.com/mlbio-epfl/aristotelian
*Equal contribution 1EPFL 2University of Basel 3HSLU. Corre-
spondence to: Maria Brbi´c <maria.brbic@epfl.ch>.
The Aristotelian Representation Hypothesis
Neural networks, trained with different objectives
on different data and modalities, are converging to
shared local neighborhood relationships.
Local alignment
Space  
Space
Figure 1. The Aristotelian Representation Hypothesis: Local
relations (“who is near whom”), rather than distances between
data points, are preserved across different representation spaces.
Representation learning algorithms will converge to shared local
neighborhood relationships.
study whether scaling produces universal structure across
models (Huh et al., 2024; Maniparambil et al., 2024; Tjan-
drasuwita et al., 2025; Zhu et al., 2026). To measure repre-
sentational similarity across models, different metrics have
been proposed, such as Centered Kernel Alignment (Korn-
blith et al., 2019), Canonical Correlation Analysis (Weenink,
2003), Representational Similarity Analysis (Kriegeskorte
et al., 2008), and mutual k-Nearest Neighbors (Huh et al.,
2024).
In this work, we identify two pervasive confounders that
distort representational similarity measurements. The first is
the model width: when the embedding dimension increases
relative to the sample size, interaction-matrix-based sim-
ilarity metrics exhibit a systematic positive baseline even
when representations are independent. This spurious similar-
ity is a general consequence of dimensionality-driven null
inflation: the expected similarity under independence does
not vanish but instead depends on both the representation
dimensionality and the sample size (Figure 2a). As a re-
sult, wider models can appear more aligned simply because
their representations live in higher-dimensional spaces. The
second confounder is the model depth. Many analyses do
not compare individual layer pairs, because it is unknown
where similarity arises (Schrimpf et al., 2018; Huh et al.,
1
arXiv:2602.14486v1  [cs.LG]  16 Feb 2026

## Page 2

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
Language model capacity
0.08
0.09
0.10
0.11
0.12
0.13
0.14
0.15
0.16
Alignment score
Small vision model
Medium vision model
Large vision model
Language model capacity
0.20
0.25
0.30
0.35
0.40
0.45
Alignment score
Small vision model
Medium vision model
Large vision model
(a) Width confounder:
# samples
Raw 
Calibrated
 Model depth (# comparisons)
Raw score 
Calibrated score
(c) Revisit the Platonic Representation Hypothesis 
Global similarity (CKA)
Local similarity (mutual KNN)
Model width
(b) Depth confounder:
Score
0.5
0.0
Calibration
Calibration
Model width
Language model capacity
Language model capacity
Alignment score
Figure 2. Null calibration removes width and depth con-
founders. (a) Width confounder: raw scores exhibit positive null
baselines that increase with the ratio of dimension (width) of the
spaces and the number of samples; calibration collapses them to
zero. (b) Depth confounder: selection-based summaries (max over
layers) inflate with search space size; aggregation-aware calibra-
tion removes this. (c) After calibration, global metrics lose their
convergence trend, while local metrics retain significant alignment.
2024). Instead, they search over all pairs and report a sum-
mary statistic such as the maximum. Taking a maximum
over many comparisons inflates the reported score even if
there is no similarity, since the expected maximum of inde-
pendent draws exceeds the mean. This inflation grows with
the number of comparisons, so deeper models can appear
more aligned simply because more layer pairs are com-
pared (Figure 2b). Together, these confounders undermine
the comparative use of representational similarity without
calibration.
To address these issues, we introduce the null-calibration
for representational similarity, a general permutation-based
framework that transforms any similarity metric into a cali-
brated score with a principled null reference, here defined as
no relationship (Figure 2). The core idea is to measure how
extreme an observed similarity is relative to an empirical
null distribution obtained by breaking sample correspon-
dences. For scalar comparisons (i.e., width confounder), we
estimate a critical threshold from the null distribution and
define a calibrated score that is zero when the observed sim-
ilarity falls below this threshold and rescaled to preserve the
maximum at one. For selection-based summaries (i.e., depth
confounder), we apply aggregation-aware calibration. We
compute the null distribution of the same aggregate statistic
that is ultimately reported (e.g., the maximum over all layer
pairs), thereby calibrating the selection step itself.
These observations raise a question: Does the Platonic Rep-
resentation Hypothesis still hold once similarity is cali-
brated? We find that, after calibration, the previously re-
ported convergence in global metrics (Huh et al., 2024; Ma-
niparambil et al., 2024; Tjandrasuwita et al., 2025) largely
disappears, suggesting it was driven primarily by width and
depth confounders, whereas local neighborhood-based met-
rics retain significant cross-modal alignment (Figure 2c).
However, we also observe that the convergence in local dis-
tances is not preserved, suggesting that only local neighbor-
hood relationships are aligned. Motivated by these results,
we refine the original Platonic Representation Hypothesis
and propose the Aristotelian Representation Hypothesis1:
Neural networks, trained with different objectives on differ-
ent data and modalities, converge to shared local neighbor-
hood relationships (Figure 1). We name it after the Greek
philosopher Aristotle, who was a student of Plato and, in his
Categories, established the principles of relatives (Aristotle,
ca. 350 B.C.E).
2. Related work
Representational similarity metrics.
A long line of
work compares representation spaces using a variety
of similarity measures. Canonical Correlation Analysis
(CCA) (Hotelling, 1992) and variants such as Singular
Vector Canonical Correlation Analysis (SVCCA) (Raghu
et al., 2017) and Projection Weighted Canonical Correlation
Analysis (PWCCA) (Morcos et al., 2018) compare
subspaces up to linear transformations, while Procrustes-
and shape-based distances compare representations up to
restricted alignment classes (Ding et al., 2021; Williams
et al., 2021). Centered Kernel Alignment (CKA) (Kornblith
et al., 2019) has become a dominant tool for comparing
deep representations, with kernelized variants extending to
nonlinear similarity. Representational Similarity Analysis
(RSA) (Kriegeskorte et al., 2008), originating in neuro-
science, compares representational dissimilarity matrices
rather than feature bases. Neighborhood-based approaches,
such as mutual k-Nearest Neighbors (mKNN) (Huh et al.,
2024), capture local topological consistency rather than
global alignment. However, recent evaluations stress that
different metrics encode different invariances and can yield
qualitatively different conclusions, motivating more robust
reporting practices (Klabunde et al., 2025; Ding et al., 2021;
Harvey et al., 2024; Bo et al., 2024).
Reliability of representational similarity metrics.
In
finite-sample, high-dimensional regimes, raw similarity
scores can be systematically biased. Recent works (Murphy
et al., 2024; Chun et al., 2025) propose debiased CKA, but
1Calling this refinement Aristotelian: it emphasizes learned
representations converging on relations among instances (who is
near whom) rather than the idea of convergence toward a globally
matching structure.
2

## Page 3

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
these corrections are metric-specific. For neighborhood-
based metrics, no analogous debiasing methods exist despite
distance concentration effects that inflate random k-NN
overlap (Beyer et al., 1999; Aggarwal et al., 2001). Other
approaches address confounding from input population
structure. For instance, Cui et al. (2022) propose regression-
style deconfounding to remove effects of shared input
statistics on RSA/CKA. A separate reliability issue arises
from layer search, where max or top-k aggregation across
many layer pairs introduces multiple-comparison inflation.
While resampling-based “maxT” procedures (Westfall &
Young, 1993; Nichols & Holmes, 2002) can calibrate such
aggregates, this has not yet been applied in representational
similarity studies. Our calibration framework addresses
both finite-sample bias and selection inflation in a unified,
metric-agnostic way.
The Platonic Representation Hypothesis.
A growing
body of work examines whether neural networks trained
under different conditions converge toward similar repre-
sentations. The Platonic Representation Hypothesis (Huh
et al., 2024) posits that as models scale, their representa-
tions increasingly converge across architectures and even
across modalities such as vision and language, with con-
vergence reported under both global and local similarity
measures. Follow-up work has examined factors influencing
these trends, including model size, training duration, and
data distribution (Raugel et al., 2025), and has explored
analogous convergence effects in broader settings such as
video models (Zhu et al., 2026) and comparisons to bio-
logical vision (Marcos-Manch´on & Fuentemilla, 2025). In
this work, we revisit the Platonic Representation Hypothesis
using our null-calibration framework that controls for width
and depth confounders.
3. Problem setup
3.1. Representation spaces and similarity score
Let X ⊆Rdx and Y ⊆Rdy be two representation spaces,
where dx and dy are the respective space dimensions. For
a set of n input samples, let X ∈Rn×dx and Y ∈Rn×dy
be the corresponding embeddings in X and Y. We as-
sume row-wise alignment such that the i-th row of X and
Y correspond to paired inputs. We use a similarity score
s(X, Y) ∈R to quantify the agreement between X and Y.
In practice, we compute it from X, Y and, by a slight abuse
of notation, denote it with s(X, Y).
We consider a generic similarity function s(X, Y) ∈R. Our
focus covers three families of metrics: (i) spectral: metrics
defined on the spectrum of cross-covariance or Gram matri-
ces (e.g., CKA, CCA), (ii) neighborhood: metrics measuring
local topological overlap (e.g., mKNN), and (iii) geometric:
second-order isomorphism metrics (e.g., RSA). Section C
provides definitions of the metrics used in this paper.
3.2. The null hypothesis of independence
We claim that a similarity score s(X, Y) is uninterpretable
without a baseline. To provide this baseline, we define the
null hypothesis H0 as the absence of a relationship between
X and Y beyond their marginal statistics. We operationalize
H0 via a permutation group Πn acting on sample indices:
draw π ∼Unif(Πn) independently of (X, Y) and evaluate
s(X, π(Y)), where π(Y) permutes the rows of Y.
Assumption 3.1 (Exchangeability under the null). Under
H0, the joint distribution of paired samples is invariant to
relabeling of correspondences. For any permutation π ∈Πn,
PH0(X, Y) = PH0(X, π(Y)).
This assumption implies that if no true relationship exists,
the observed pairing is statistically indistinguishable from
a random shuffling of the data. It allows us to construct an
empirical null distribution by holding X fixed and shuffling
the rows of Y.
3.3. Baseline problem: non-zero null expectations
Ideally, under H0, we desire Eπ[s(X, π(Y))] ≈0. How-
ever, for commonly used raw or biased estimators, the ex-
pected similarity under the null is not zero,
µ0(n, dx, dy) := Eπ[s(X, π(Y))].
(1)
This baseline µ0 is metric- and preprocessing-dependent and
can deviate from zero in finite samples. It also varies with
sample size and dimension, thus acting as a confounding
variable in comparative studies.
4. Theoretical motivation: spurious alignment
We motivate and formalize why raw representational sim-
ilarity metrics fail in cross-scale model comparisons. We
identify two distinct sources of confounding: (i) the width
confounder driven by representation dimension, and (ii) the
depth confounder driven by the number of layers considered
when comparing models.
4.1. The width confounder
Many spectral-family similarity metrics, e.g., linear/ker-
nel CKA, the RV coefficient, and CCA-based scores
(CCA/SVCCA/PWCCA), can be written as functionals of an
interaction operator constructed from two representations.
One such operator is the (normalized) cross-covariance
eC =
1
n −1X⊤
c Yc ∈Rdx×dy,
(2)
where Xc and Yc denote row-centered representations (Sec-
tion C.1).
3

## Page 4

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
A common but misleading intuition is that if X and Y are
independent, then eC ≈0 and therefore spectral aggregates
should be near zero. In high dimension this fails: the null
interaction energy is typically non-zero.
Proposition 4.1 (Non-vanishing null interaction energy).
Assume the rows are i.i.d. with E[xi] = E[yi] = 0,
Cov(xi) = Idx, Cov(yi) = Idy, and xi and yi are in-
dependent. Then
EH0
h
∥eC∥2
F
i
=
dxdy
n −1.
(3)
Proof. See Section D.4.
Since CKA is scaled by the normalized self-similarity terms,
which each scale as O(
√
d), the resulting null baseline for
the metric is thus O(d/n).
This aligns with insights from random matrix theory: in
high-dimensional regimes (d ∼n), the null singular spec-
trum of interaction operators (after centering/whitening)
concentrates into a non-trivial “noise bulk” whose upper
edge depends on d/n and preprocessing, rather than collaps-
ing to zero (Wachter, 1978; M¨uller, 2002). Our framework
estimates this null baseline directly via permutation, pro-
viding a metric- and pipeline-independent alternative to
asymptotic formulas.
Neighborhood metrics follow a different regime.
While
spectral metrics have null baselines scaling as O(d/n),
neighborhood-based metrics such as mutual k-NN exhibit
different behavior, as they rely on set comparisons rather
than interactions.
Proposition 4.2 (Null baseline for neighborhood metrics).
Assume the rows are i.i.d. with xi and yi independent, and
that pairwise distances are almost surely distinct (e.g., under
absolutely continuous distributions). Then for any k < n,
EH0

mKNN(X, Y)

=
k
n −1.
(4)
Proof. See Section D.6.
In particular, neighborhood metrics have null baselines scal-
ing as O(k/n).
The difference in null baseline between spectral and neigh-
borhood metrics is substantial: (i) The neighborhood scale
k can be fixed consistently across experiments, whereas the
embedding dimension d is determined by the model archi-
tecture, making it difficult to control in comparison studies.
(ii) The neighborhood metrics are much less confounded
since k ≪d in typical settings.
4.2. The depth confounder
A subtle yet pervasive issue is the comparison of
selection-based alignment summaries across models. Let
Sℓ,ℓ′ := s(X(A)
ℓ
, Y(B)
ℓ′ ) be the similarity between layer ℓ
of model A and layer ℓ′ of model B. It is common to sum-
marize the similarity between two models by the maximum
alignment score Tmax = maxℓ,ℓ′ Sℓ,ℓ′. Let M = LALB
be the number of layer pairs searched, where LA and LB
are the depths of models A and B. Even under H0, taking
a maximum over M comparisons inflates the reported
score, a “look-elsewhere” effect. This is an instance of
the classical multiple comparisons problem (Benjamini &
Hochberg, 1995; Bonferroni, 1936): as M increases, the
probability that at least one null similarity exceeds any
fixed threshold grows, inflating the expected maximum.
Consequently, when alignment is summarized via a max or
top-k statistic without correction, unrelated representations
can exhibit spuriously high reported similarity, as the
inflation depends on model depth, making raw summaries
non-comparable across architectures.
Characterizing this inflation does not require independence
across pairs. It follows from a uniform right-tail bound.
Assume there exist a common mean µ ∈R and σ > 0 such
that the null fluctuations satisfy, for all (ℓ, ℓ′) and all t ≥0,
P(Sℓ,ℓ′ −µ ≥t) ≤exp

−t2
2σ2

.
(5)
For bounded similarities Sℓ,ℓ′ ∈[smin, smax], Hoeffding’s
inequality implies a sub-Gaussian right-tail bound of the
form Equation (5) with σ ≤(smax −smin)/2. This covers
many common bounded metrics (e.g., CKA/RSA/mKNN).
Crucially, only the right tail is needed for bounding the
maximum. Then a union bound gives
P(Tmax −µ ≥t) ≤M exp

−t2
2σ2

,
(6)
and consequently for a constant C
EH0 [Tmax] ≤µ + C σ
p
log M.
(7)
Proof. See Section D.5.
This creates a depth confounder. Deeper models (larger
M = LALB) can attain higher raw “max-alignment” scores
purely because of a larger search space. Correlations across
neighboring layers reduce the effective number of compar-
isons, but the inflation remains monotone in the search space
size in typical workflows. Therefore, raw scaling plots of
Tmax (or top-k summaries) are not comparable across archi-
tectures unless the selection step itself is calibrated.
5. Representational similarity calibration
To overcome the issues of the width and depth confounders,
we introduce the null-calibration for representational simi-
larity. The key idea is to compare observed similarity scores
4

## Page 5

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
against an empirical null distribution obtained by permuting
sample correspondences, thereby establishing a principled
zero point that accounts for finite-sample, high-dimensional
artifacts.
5.1. Null-calibrated similarity
We propose null-calibrated similarity measures to correct
for width and depth confounders by transforming raw simi-
larity scores into an effect size with a principled zero point.
Given representations X ∈Rn×dx and Y ∈Rn×dy aligned
by rows, we operationalize the null hypothesis H0 (no rela-
tionship beyond marginal statistics) by permuting sample
correspondences. For permutations πk ∈Πn drawn i.i.d.
uniformly from Πn and independently of (X, Y), we form
null scores
s(k) = s(X, πk(Y)),
k = 1, . . . , K.
(8)
Let sobs := s(X, Y) denote the observed score. Let s(1) ≤
s(2) ≤· · · ≤s(K+1) denote the order statistics of the com-
bined multiset {sobs, s(1), . . . , s(K)} (with ties allowed).
We define a right-tail rank-based critical value:
τα := s(⌈(1−α)(K+1)⌉),
(9)
where ⌈(1 −α)(K + 1)⌉is the (1 −α)-quantile of the
(K + 1)-sized multiset and the empirical right-tail p-value:
p = 1 + #{k ∈{1, . . . , K} : s(k) ≥sobs}
K + 1
.
(10)
The critical value τα defines a robust zero point: values
below τα are typical under H0 at level α, while p provides
an evidence measure that can be combined with multiple-
testing correction when many comparisons are performed.
The proposed calibration framework relies on randomiza-
tion (permutation) to construct a null distribution for any
similarity statistic. This yields finite-sample guarantees un-
der an exchangeability condition (Assumption 3.1), and it
implies useful invariances that make calibrated scores com-
parable across metrics and implementations.
The permutation p-value in Equation (10) is super-uniform
under H0 (i.e., PH0(p ≤α) ≤α for all α ∈[0, 1]), a
standard consequence of randomization inference (Nichols
& Holmes, 2002; Phipson & Smyth, 2010; Good, 2005) (see
Section D.1 for formal definitions and proofs).
Corollary 5.1 (Type-I control for calibrated scores). Let
sobs
= s(X, Y) and s(k)
= s(X, πk(Y)) for k =
1, . . . , K. Define the add-one permutation p-value p as in
Equation (10), and equivalently define the rank-based criti-
cal value τα := s(⌈(1−α)(K+1)⌉) from the sorted combined
set {sobs, s(1), . . . , s(K)}. Under Assumption 3.1,
PH0
p ≤α

≤α
and hence
PH0
sobs > τα

≤α,
(11)
so the gating rule “scal > 0” (where scal is the calibrated
score defined in Equation (12), which implies p ≤α) is a
finite-sample α-level declaration of similarity above chance.
Proof. Follows directly from Lemma D.2; see Section D.1.
Calibrated score (scalar case).
While p-values and null
percentiles are rank-based and therefore invariant under
monotone transformations of the raw score (Proposition D.3;
see Section D.2), effect sizes serve a complementary pur-
pose: they quantify how much similarity exceeds chance on
an interpretable scale. The calibrated score achieves this by
rescaling the excess over the null threshold τα to the interval
[0, 1]. This rescaling is not monotone-invariant, and this by
design. A purely rank-based calibration would be equiva-
lent to a score shift and would be unable to correct for the
scale-dependent null baselines identified in Section 4. The
calibrated score instead adapts to the actual null distribution,
providing a meaningful zero point.
For bounded similarity metrics with known maximum smax
(often smax = 1), we define a max-preserving calibrated
score
scal = max
 sobs −τα
smax −τα
, 0

.
(12)
This calibrated score depends on the chosen level α through
τα (Equation (9)). We therefore also report the correspond-
ing permutation p-value and/or null percentile for an α-free
summary. This score satisfies scal = 0 whenever sobs ≤τα
(i.e., below the estimated right-tail critical value of the per-
mutation null), and scal = 1 when sobs = smax (i.e., perfect
similarity remains 1). When smax is unknown, or the metric
is unbounded, we default to the unnormalized effect size
[s −τα]+ = max(s −τα, 0).
5.2. Aggregation-aware null-calibration
To analyze the similarity between two models A and B with
depths LA and LB, a common approach is to compute a
layer-by-layer similarity matrix S ∈RLA×LB by evaluating
a similarity score for every pair of layers:
Sℓ,ℓ′ = s

X(A)
ℓ
, Y(B)
ℓ′

,
(13)
where X(A)
ℓ
∈Rn×dℓand Y(B)
ℓ′
∈Rn×dℓ′ are the represen-
tations of models A and B at layers ℓand ℓ′ respectively,
evaluated on n samples, and s(·, ·) is a similarity metric.
A common practice is then to summarize S by a selection-
based aggregation operator, such as taking the maximum.
These summaries are attractive because they support state-
ments such as “there exists a layer in A that matches some
layer in B” or “each layer of A best matches a layer in B”.
However, selection introduces a statistical effect: even under
the null hypothesis of no relationship between representa-
tions, selection-based summaries are systematically inflated.
5

## Page 6

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
As analyzed in Section 4.2, this inflation grows with the
number of layer pairs and makes na¨ıve post-selection p-
values anti-conservative. Our aggregation-aware calibration
addresses this by calibrating the reported statistic directly:
the null distribution must match the entire analysis pipeline.
Let the aggregate score be T(S) (e.g., a maximum), then the
appropriate null is the distribution of T(S) under a valid null
transformation (e.g., permuting sample correspondences).
We therefore define an aggregation-aware permutation null.
Consistency of permutations across layers.
For each
draw πk ∈Πn, we apply the same sample permutation to
all layers of model B and define
S(k)
ℓ,ℓ′ := s

X(A)
ℓ
, πk

Y(B)
ℓ′

,
(14)
ℓ= 1, . . . , LA,
ℓ′ = 1, . . . , LB,
then compute T (k)
:=
T(S(k)). Let Tobs
:=
T(S)
denote the observed aggregate. Let T(1)
≤
· · ·
≤
T(K+1) denote the order statistics of the combined set
{Tobs, T (1), . . . , T (K)} (with ties allowed). We define
τ agg
α
:= T(⌈(1−α)(K+1)⌉),
(15)
where ⌈(1 −α)(K + 1)⌉is the (1 −α)-quantile of the
(K + 1)-sized multiset. We report the right-tail permutation
p-value
pagg = 1 + #{k ∈{1, . . . , K} : T (k) ≥Tobs}
K + 1
,
(16)
By the same exchangeability argument as for scalar calibra-
tion, pagg is super-uniform under H0 (see Proposition D.4).
Calibrated score (aggregate case).
For bounded similar-
ities with maximum smax (often smax = 1), we report a
max-preserving calibrated aggregate
Tcal = max
 Tobs −τ agg
α
smax −τ agg
α
, 0

.
(17)
This score satisfies Tcal = 0 when Tobs ≤τ agg
α
and Tcal = 1
when Tobs = smax. As above, Tcal depends on α via τ agg
α
;
we therefore report both Tcal (magnitude above null) and
pagg (evidence against null), applying multiplicity correc-
tion (Holm, 1979; Benjamini & Hochberg, 1995) when
many model pairs are evaluated.
5.3. Summary
To compute a calibrated similarity score: (i) fix a signifi-
cance level α (e.g., α = 0.05); (ii) generate K null scores
by permuting sample correspondences; (iii) compute critical
value τ as the ⌈(1 −α)(K + 1)⌉-th order statistic of the
combined set (observed + null scores); (iv) return calibrated
score, either scal or Tcal.
Use scalar calibration (Section 5.1) when comparing a sin-
gle pair of representations. Use aggregation-aware calibra-
tion (Section 5.2) when reporting a summary statistic (e.g.,
maximum) over multiple layer pairs. Section E provides
pseudocode for both procedures.
6. Experiments
We quantify the effects of the width and depth confounders
in controlled synthetic experiments and show that our cali-
bration framework effectively removes them. We then revisit
the Platonic Representation Hypothesis using our calibra-
tion framework, assessing which convergence trends remain
robust after controlling for these confounding factors.
6.1. Null-calibration removes width confounder
We validate that our calibration eliminates width-related
inflation of similarity across metrics, regimes, and noise
distributions, without metric-specific derivations.
We design controlled synthetic experiments as follows.
Under H0, we draw X, Y
∈
Rn×d independently
from Gaussian and heavy-tailed (Student-t, Laplace) dis-
tributions. We sweep the number of samples n
∈
{128, 256, 512, 1024, 2048, 4096} and the dimension d ∈
{128, 256, 512, 1024, 2048}. Under H1, we inject a shared
low-rank signal component and vary the signal-to-noise
ratio. We evaluate representative metrics spanning three
families. For spectral similarity, we use linear and RBF
CKA, as well as CCA/SVCCA/PWCCA; for neighborhood
similarity, we use mKNN (with k = 10); and for geometric
similarities, we use RSA and Procrustes. Figure 3 reports a
subset of these metrics for readability; additional metrics are
reported in Section F.6. For calibration, we use K = 200
permutations with α = 0.05.
Under H0, uncalibrated scores increase with d/n, while our
calibrated scores stay at zero across settings (Figure 3). This
confirms that the similarity scores of wider models can arise
Uncalibrated
Calibrated

	
	


















	

	

Figure 3. Calibration eliminates spurious similarity across met-
rics. Raw scores (top) drift with d/n; calibrated scores (bottom)
collapse to zero. Results for heavy-tailed distributions and addi-
tional metrics are in Section F.6.
6

## Page 7

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
purely from high-dimensional finite-sample effects, and our
calibration removes this spurious baseline. Importantly, the
magnitude of the null baseline is metric-dependent, con-
sistent with our theory: CKA’s baseline scales as O(d/n)
(Proposition 4.1), while mKNN’s baseline scales as O(k/n)
(Proposition 4.2). Intuitively, mKNN compares local neigh-
borhood overlap at a fixed k, thus only comparing relation-
ships instead of local distances, making its null baseline
insensitive to representation width d, which explains the
order-of-magnitude gap observed in raw scores. The same
pattern holds for heavy-tailed noise (Section F.6).
Next, we verify the statistical guarantees of our empirical
null calibration. For Type-I error control, rejection rates
stay at or below the nominal α = 0.05 across (n, d/n)
configurations (Figure 4a). Crucially, our calibration does
not sacrifice sensitivity to real alignment: detection rates
increase rapidly with signal strength (Figure 4b). Overall,
our calibration preserves signal structure: in the high-signal
regime, raw and calibrated scores show the same pattern,
while in the low-signal regime, calibration correctly gates
scores to zero (Section F.2).
Furthermore, we verify that our empirical calibration closely
matches existing analytical bias corrections for CKA (Mur-
phy et al., 2024), recovering the width correction without
metric-specific derivation (Section F.4).
Additionally, we perform ablations on different noise dis-
tributions used in the synthetic experiments (Section F.1),
different calibration approaches (Section F.3), and an abla-
tion on the influence of the number of permutations K used
for calibration (Section F.5).
0.25
0.50
0.75
1.00
1.25
1.50
1.75
2.00
d/n
0.03
0.04
0.05
0.06
0.07
0.08
0.09
0.10
Type I error rate
α = 0.05
1.0
1.5
2.0
2.5
3.0
3.5
4.0
Signal strength
0.0
0.2
0.4
0.6
0.8
1.0
Power (detection rate)
CKA (linear)
CKA (RBF)
mKNN
RSA
Figure 4. Statistical guarantees. (Left) Type-I error stays at or
below α across configurations. (Right) Power increases rapidly
with signal strength; calibration does not sacrifice sensitivity.
6.2. Null-calibration removes depth confounder
We validate that our aggregation-aware null-calibration elim-
inates the depth confounder. To build a controlled synthetic
setting, we construct two synthetic models, A and B, each
with L layers. Under H0, we sample layer representations
{Xℓ}L
ℓ=1 and {Yℓ′}L
ℓ′=1, where each Xℓ, Yℓ′ ∈Rn×d
has i.i.d. N(0, 1) entries (independent across layers and
between models), using d/n = 8 to match the upper
range of the Platonic Representation Hypothesis setting.
We then compute the layerwise similarity matrix Sℓ,ℓ′ =
CKAlin(Xℓ, Yℓ′) and summarize it with standard aggrega-
tion operators.
The uncalibrated max-aggregated scores inflate with layer
count even under H0 (Figure 5): raw max-scores are sys-
tematically higher at L = 128 than at L = 1, despite no
true signal. Our aggregation-aware calibration eliminates
this bias: calibrated aggregates remain stable regardless of
depth. We further show that naively calibrating each scalar
comparison still leads to inflation, highlighting the impor-
tance of calibrating the final statistic. Furthermore, since
deeper models tend to be wider as well, raw comparisons
are doubly confounded.
0
20
40
60
80
100
120
Number of layers
0.890
0.895
0.900
0.905
0.910
0.915
Uncalibrated score
0
20
40
60
80
100
120
Number of layers
0.000
0.025
0.050
0.075
0.100
0.125
Calibrated score
entry-wise calibration
max
row-max mean
col-max mean
top-5 mean
top-10 mean
Figure 5. Aggregation-aware calibration removes depth con-
founding. Raw max-aggregates of linear CKA scores inflate with
layer count under the null; calibrated aggregates are stable and
show that naive entry-wise calibration still leads to inflation.
6.3. Revisiting the Platonic Representation Hypothesis
A central claim behind the Platonic Representation Hypoth-
esis is that, as models become more capable, their repre-
sentations begin to converge across modalities. We revisit
this claim through our calibration framework to determine
whether the observed alignment reflects genuine shared rep-
resentation structure or instead arises from width and depth
confounders.
We follow the experimental protocol of Huh et al. (2024)
using n = 1024 image–text pairs (WIT; Srinivasan et al.
(2021)) and embeddings from three language model
families (Bloomz, OpenLLaMA, LLaMA) and five vision
model families (ImageNet-21K, MAE, DINOv2, CLIP,
CLIP-finetuned) across multiple scales. This yields 204
vision–language model pairs spanning d/n ∈[0.75, 8]. For
each pair, we compute layer-wise similarity and report the
maximum across layers, as in the original work. We evaluate
both global spectral metrics (CKA linear/RBF) and local
neighborhood metrics (mKNN, cycle-kNN, CKNNA). Fol-
lowing Huh et al. (2024), we evaluate mKNN, cycle-kNN,
and CKNNA with k = 10. We further apply Benjamini-
Hochberg FDR correction (Benjamini & Hochberg, 1995)
to control for multiple comparisons across model pairs.
For the global similarity, we find that uncalibrated CKA
scores increase with model scale (dotted lines in Figure 6a),
reproducing the trend interpreted as evidence of cross-modal
7

## Page 8

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.1
0.2
0.3
0.4
0.5
0.6
Alignment to DINOv2
BLOOM
OpenLLaMA LLaMA
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.3
0.4
0.5
0.6
Alignment to CLIP
BLOOM
OpenLLaMA LLaMA
base
large
huge
calibrated
uncalibrated
(a) CKA RBF: Global spectral alignment.
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.08
0.10
0.12
0.14
0.16
Alignment to DINOv2
BLOOM OpenLLaMA LLaMA
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.12
0.14
0.16
0.18
0.20
Alignment to CLIP
BLOOM OpenLLaMA LLaMA
base
large
huge
calibrated
uncalibrated
(b) mKNN: Local neighborhood overlap.
Figure 6. Revisiting the Platonic Representation Hypothesis. Models are ranked according to their language performance (Huh et al.,
2024). Solid lines connect the models within the same family, while semi-transparent lines connect the models across different families.
(a) Global spectral metrics lose their convergence trend; calibrated scores show no systematic increase with scale. (b) Local neighborhood
metrics keep their trend even after calibration. Full results for all vision families and metrics in Section F.7.
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.2
0.4
0.6
CKA RBF
BLOOM OpenLLaMA LLaMA
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.05
0.10
0.15
0.20
mKNN
BLOOM OpenLLaMA LLaMA
base
large
huge
calibrated
uncalibrated
Figure 7. Video–language alignment. Extending the Platonic
Representation Hypothesis analysis to video encoders (VideoMAE
base/large/huge) yields the same pattern: calibrated CKA drops
substantially while mKNN retains alignment.
convergence (Huh et al., 2024). However, this trend disap-
pears after our calibration (solid lines): calibrated CKA
shows no systematic increase with model size. This indi-
cates that global convergence in uncalibrated CKA is largely
attributable to width and depth confounders rather than a
genuine increase in representational similarity.
In contrast, for the local similarity, evidence of cross-modal
convergence remains strong for neighborhood-based metrics
even under our calibration (Figure 6b). The same qualitative
conclusion holds for other neighborhood-based measures
(cycle-kNN and CKNNA; Section F.7) and different choices
of α (Section F.10). Further analysis (Section F.9) reveals
that models converge in local neighborhood structure: mod-
els increasingly agree on which points are neighbors, but do
not agree on the pairwise distances, since CKA-RBF with a
small bandwidth shows no alignment after calibration.
To test whether these findings generalize beyond images and
text, we extend our analysis to video–language alignment
following Zhu et al. (2026). We compare video encoders
(VideoMAE base/large/huge) against the same language
model families. Consistent with our previous findings, the
global similarity (CKA) shows no trend with model capac-
ity (Figure 7). In contrast, for local similarity (mKNN), a
clear scaling trend emerges with VideoMAE-Huge, whereas
smaller video encoders appear to act as a bottleneck, lim-
iting alignment regardless of language model size. This
confirms that local neighborhood convergence extends to
video–language alignment, provided that representations are
sufficiently powerful. Section F.8 further compares a variety
of image models at the frame level on the same dataset,
showing the same trend.
Taken together, these results suggest a refined version of
the Platonic Representation Hypothesis. After calibration,
we find little evidence that representations converge in
global spectral structure as models scale, at least under the
considered setting. What reliably persists is local geometric
alignment: different models preserve similar neighborhood
relationships among inputs. We therefore propose the
alternative Aristotelian Representation Hypothesis: As
models become capable, their representations converge to
shared local neighborhood relationships.
7. Conclusion
Representational similarity metrics are widely used to study
learned features, but their interpretation is systematically
distorted by two artifacts: width-dependent null baselines
and depth-dependent selection inflation. We introduced a
unified null-calibration framework that corrects both, turn-
ing similarity scores into effect sizes with principled zero
points and valid p-values. Applying our framework to the
Platonic Representation Hypothesis reveals that previously
reported global spectral convergence is largely confounded
by width and depth, whereas local neighborhood alignment
remains significant, motivating an Aristotelian Representa-
tion Hypothesis.
8

## Page 9

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
Acknowledgements
We thank Artyom Gadetsky, Siba Smarak Panigrahi, Deba-
jyoti Dasgupta, David Fr¨uhbuss, Shin Matsushima, Rishubh
Singh, Adriana Moreno Castan, and Gioele La Manno
for their valuable suggestions, which helped improve the
manuscript. We are especially grateful to Simone Lionetti
for additional input and support. We gratefully acknowl-
edge the support of the Swiss National Science Founda-
tion (SNSF) starting grant TMSGI2 226252/1, SNSF grant
IC00I0 231922, and the Swiss AI Initiative. M.B. is a CI-
FAR Fellow in the Multiscale Human Program.
References
Aggarwal, C. C., Hinneburg, A., and Keim, D. A. On the
surprising behavior of distance metrics in high dimen-
sional space. In International Conference on Database
Theory. Springer, 2001.
Aristotle. Categories, ca. 350 B.C.E.
Benjamini, Y. and Hochberg, Y. Controlling the false dis-
covery rate: a practical and powerful approach to multiple
testing. Journal of the Royal Statistical Society, 1995.
Beyer, K., Goldstein, J., Ramakrishnan, R., and Shaft, U.
When is “nearest neighbor” meaningful? In International
Conference on Database Theory. Springer, 1999.
Bo, Y., Soni, A., Srivastava, S., and Khosla, M. Evaluating
representational similarity measures from the lens of func-
tional correspondence. arXiv preprint arXiv:2411.14633,
2024.
Bolya, D., Huang, P.-Y., Sun, P., Cho, J. H., Madotto, A.,
Wei, C., Ma, T., Zhi, J., Rajasegaran, J., Rasheed, H. A.,
Wang, J., Monteiro, M., Xu, H., Dong, S., Ravi, N., Li, S.-
W., Dollar, P., and Feichtenhofer, C. Perception encoder:
The best visual embeddings are not at the output of the
network. Advances in Neural Information Processing
Systems, 2025.
Bonferroni, C. Teoria statistica delle classi e calcolo delle
probabilita.
Pubblicazioni del R istituto superiore di
scienze economiche e commericiali di firenze, 1936.
Cai, M. B., Schuck, N. W., Pillow, J. W., and Niv, Y. Repre-
sentational structure or task structure? Bias in neural rep-
resentational similarity analysis and a Bayesian method
for reducing bias. PLoS Computational Biology, 2019.
Cho, J. H., Madotto, A., Mavroudi, E., Afouras, T., Na-
garajan, T., Maaz, M., Song, Y., Ma, T., Hu, S., Jain, S.,
Martin, M., Wang, H., Rasheed, H. A., Sun, P., Huang,
P.-Y., Bolya, D., Ravi, N., Jain, S., Stark, T., Moon, S.,
Damavandi, B., Lee, V., Westbury, A., Khan, S., Krae-
henbuehl, P., Dollar, P., Torresani, L., Grauman, K., and
Feichtenhofer, C. PerceptionLM: Open-access data and
models for detailed visual understanding. Advances in
Neural Information Processing Systems, 2025.
Chun, C., Canatar, A., Chung, S., and Lee, D. D. Estimating
Neural Representation Alignment from Sparsely Sampled
Inputs and Features. arXiv preprint arXiv:2502.15104,
2025.
Cram´er, H. Mathematical methods of statistics. Princeton
University Press, 1999.
Cui, T., Kumar, Y., Marttinen, P., and Kaski, S. Decon-
founded representation similarity for comparison of neu-
ral networks. Advances in Neural Information Processing
Systems, 2022.
Diedrichsen, J., Berlot, E., Mur, M., Sch¨utt, H. H., Shahbazi,
M., and Kriegeskorte, N. Comparing representational
geometries using whitened unbiased-distance-matrix sim-
ilarity. arXiv preprint arXiv:2007.02789, 2020.
Ding, F., Denain, J.-S., and Steinhardt, J. Grounding repre-
sentation similarity through statistical testing. Advances
in Neural Information Processing Systems, 2021.
Embrechts, P., Kl¨uppelberg, C., and Mikosch, T. Modelling
extremal events: for insurance and finance. Springer
Science & Business Media, 2013.
Good, P. Permutation, parametric and bootstrap tests of
hypotheses. Springer, 2005.
Harvey, S. E., Lipshutz, D., and Williams, A. H. What
Representational Similarity Measures Imply about De-
codable Information. In Proceedings of UniReps: the
Second Edition of the Workshop on Unifying Representa-
tions in Neural Models. PMLR, 2024.
Holm, S. A simple sequentially rejective multiple test pro-
cedure. Scandinavian Journal of Statistics, 1979.
Hotelling, H. Relations between two sets of variates. In
Breakthroughs in Statistics: Methodology and Distribu-
tion. Springer, 1992.
Huh, M., Cheung, B., Wang, T., and Isola, P. Position:
The platonic representation hypothesis. In International
Conference on Machine Learning, 2024.
Klabunde, M., Schumacher, T., Strohmaier, M., and Lem-
merich, F. Similarity of neural network models: A survey
of functional and representational measures. ACM Com-
puting Surveys, 2025.
Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. Sim-
ilarity of neural network representations revisited. In
International Conference on Machine Learning, 2019.
9

## Page 10

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
Kriegeskorte, N., Mur, M., and Bandettini, P. A. Represen-
tational similarity analysis–connecting the branches of
systems neuroscience. Frontiers in Systems Neuroscience,
2008.
Lehmann, E. L. and Romano, J. P. Testing statistical hy-
potheses. Springer, 2005.
Maniparambil, M., Akshulakov, R., Djilali, Y. A. D.,
El Amine Seddik, M., Narayan, S., Mangalam, K., and
O’Connor, N. E. Do Vision and Language Encoders Rep-
resent the World Similarly? In Conference on Computer
Vision and Pattern Recognition, 2024.
Marcos-Manch´on, P. and Fuentemilla, L. Convergent trans-
formations of visual representation in brains and models.
arXiv preprint arXiv:2507.13941, 2025.
Morcos, A., Raghu, M., and Bengio, S. Insights on repre-
sentational similarity in neural networks with canonical
correlation. Advances in Neural Information Processing
Systems, 2018.
M¨uller, R. R. A random matrix model of communication
via antenna arrays. IEEE Transactions on information
theory, 2002.
Murphy, A., Zylberberg, J., and Fyshe, A. Correcting biased
centered kernel alignment measures in biological and arti-
ficial neural networks. arXiv preprint arXiv:2405.01012,
2024.
Neyshabur, B., Sedghi, H., and Zhang, C. What is being
transferred in transfer learning? Advances in Neural In-
formation Processing Systems, 2020.
Nguyen, T., Raghu, M., and Kornblith, S. Do Wide and Deep
Networks Learn the Same Things? Uncovering How Neu-
ral Network Representations Vary with Width and Depth.
In International Conference on Learning Representations,
2021.
Nichols, T. E. and Holmes, A. P. Nonparametric permutation
tests for functional neuroimaging: a primer with examples.
Human Brain Mapping, 2002.
Phipson, B. and Smyth, G. K. Permutation P-values Should
Never Be Zero: Calculating Exact P-values When Permu-
tations Are Randomly Drawn. Statistical Applications in
Genetics & Molecular Biology, 2010.
Raghu, M., Gilmer, J., Yosinski, J., and Sohl-Dickstein, J.
SVCCA: Singular Vector Canonical Correlation Anal-
ysis for Deep Learning Dynamics and Interpretability.
In Advances in Neural Information Processing Systems,
2017.
Raugel, J., Szafraniec, M., Vo, H. V., Couprie, C., Labatut, P.,
Bojanowski, P., Wyart, V., and King, J.-R. Disentangling
the factors of convergence between brains and computer
vision models. arXiv preprint arXiv:2508.18226, 2025.
Robert, P. and Escoufier, Y. A unifying tool for linear multi-
variate statistical methods: the RV-coefficient. Journal of
the Royal Statistical Society, 1976.
Schrimpf, M., Kubilius, J., Hong, H., Majaj, N. J., Rajaling-
ham, R., Issa, E. B., Kar, K., Bashivan, P., Prescott-Roy, J.,
Geiger, F., et al. Brain-score: Which artificial neural net-
work for object recognition is most brain-like? BioRxiv,
2018.
Smilde, A. K., Kiers, H. A., Bijlsma, S., Rubingh, C., and
Van Erk, M. Matrix correlations for high-dimensional
data: the modified RV-coefficient. Bioinformatics, 2009.
Song, L., Smola, A., Gretton, A., Bedo, J., and Borgwardt, K.
Feature Selection via Dependence Maximization. Journal
of Machine Learning Research, 2012.
Srinivasan, K., Raman, K., Chen, J., Bendersky, M., and Na-
jork, M. Wit: Wikipedia-based image text dataset for mul-
timodal multilingual machine learning. In International
ACM SIGIR Conference on Research and Development
in Information Retrieval, 2021.
Tjandrasuwita, M., Ekbote, C., Ziyin, L., and Liang, P. P.
Understanding the Emergence of Multimodal Representa-
tion Alignment. In International Conference on Machine
Learning, 2025.
Tong, Z., Song, Y., Wang, J., and Wang, L. VideoMAE:
Masked autoencoders are data-efficient learners for self-
supervised video pre-training. Advances in Neural Infor-
mation Processing Systems, 2022.
Wachter, K. W. The strong limits of random matrix spectra
for sample matrices of independent elements. The Annals
of Probability, 1978.
Weenink, D. Canonical correlation analysis. In Proceedings
of the Institute of Phonetic Sciences of the University of
Amsterdam. University of Amsterdam Amsterdam, 2003.
Westfall, P. H. and Young, S. S. Resampling-based multiple
testing: Examples and methods for p-value adjustment.
John Wiley & Sons, 1993.
Williams, A. H., Kunz, E., Kornblith, S., and Linderman,
S. W. Generalized Shape Metrics on Neural Represen-
tations. In Advances in Neural Information Processing
Systems, 2021.
Zhu, T., Han, T., Guibas, L., P˘atr˘aucean, V., and Ovsjanikov,
M. Dynamic Reflections: Probing Video Representations
with Text Alignment. In International Conference on
Learning Representations, 2026.
10

## Page 11

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
A. Limitations
Permutation calibration is finite-sample valid under Assumption 3.1, which treats the n row pairs as exchangeable units.
In practice, exchangeability can be violated even without a sequential structure (e.g., grouped/clustered samples). In such
settings, validity is recovered by using restricted permutations that preserve the dependence structure (e.g., permuting within
blocks or permuting block labels) and by re-running under each restricted permutation.
B. Existing calibration approaches for representational similarity metrics
Table 1. Comparison of prior works. Y=yes, N=no, P=partial/indirect. “Debias” indicates an explicit null correction of the reported
similarity. “Bounded” indicates whether the corrected score preserves an interpretable upper bound (e.g., 1 for perfect alignment).
“Agg-aware” indicates calibration of selection-based aggregates (e.g., max over layer pairs).
Ref
Metric(s)
Debias?
Bounded?
Agg-aware?
Murphy et al. (2024)
CKA
Y
N
N
Chun et al. (2025)
CKA
Y
N
N
Cui et al. (2022)
RSA/CKA
P
N
N
Diedrichsen et al. (2020)
RSA (cv/WUC)
Y
P
N
Cai et al. (2019)
RSA (Bayes)
P
N
N
Smilde et al. (2009)
RV / adj. RV
Y
N
N
Ours
Any bounded metric
Y
Y
Y
C. Metrics and score definitions
This appendix gives the definitions of the similarity metrics s(X, Y) used throughout the paper. The main text focuses on
the calibration procedure (Sections 5 and 5.2). Here we provide concrete instantiations of the metrics referenced in Section 3
and Section 6.
C.1. Preprocessing and basic notation
Let X ∈Rn×dx and Y ∈Rn×dy denote row-aligned representations evaluated on the same n inputs. We use the centering
matrix
H = In −1
n1n1⊤
n ,
(18)
where In ∈Rn×n is the identity matrix and 1n ∈Rn is the all-ones vector. We define row-centered representations
Xc = HX and Yc = HY. Unless stated otherwise, similarities are computed on centered representations.
C.2. Raw similarity metrics
This section provides formal definitions of the similarity metrics used throughout the paper. In the main text, we primarily use
CKA (linear and RBF kernel) (Kornblith et al., 2019), RSA (Kriegeskorte et al., 2008), and mutual k-NN (Huh et al., 2024)
as representative metrics from the spectral, geometric, and neighborhood families, respectively. Additional metrics (SVCCA,
PWCCA, cycle-kNN, CKNNA, RV coefficient, Procrustes) are included for completeness and used in supplementary
experiments.
C.2.1. SPECTRAL METRICS
Linear Centered Kernel Alignment (CKA).
Linear CKA (Kornblith et al., 2019) can be written as a normalized
Frobenius energy of the sample cross-covariance operator. With Xc, Yc as above, define the sample (cross-)covariances
eΣXX :=
1
n −1X⊤
c Xc,
eΣY Y :=
1
n −1Y⊤
c Yc,
eC := eΣXY :=
1
n −1X⊤
c Yc.
(19)
The biased linear Hilbert-Schmidt Independence Criterion (HSIC) energy equals ∥eC∥2
F . The commonly used linear CKA
normalization can be written as
CKAlin(X, Y) =
∥eC∥2
F
∥eΣXX∥F ∥eΣY Y ∥F
=
∥X⊤
c Yc∥2
F
∥X⊤
c Xc∥F ∥Y⊤
c Yc∥F
∈[0, 1],
(20)
11

## Page 12

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
where the second equality follows by cancellation of common 1/(n −1) factors.
Kernel Centered Kernel Alignment.
Kernel CKA (Kornblith et al., 2019) generalizes linear CKA by replacing dot
products with kernel functions. Let kX : Rdx × Rdx →R and kY : Rdy × Rdy →R be positive semidefinite kernel
functions (e.g., RBF kernel kX(x, x′) = exp(−∥x −x′∥2/2σ2)). Let KX ∈Rn×n and KY ∈Rn×n be Gram matrices
with entries (KX)ij = kX(xi, xj) and (KY )ij = kY (yi, yj). Let eKX = HKXH and eKY = HKY H denote centered
Gram matrices. Kernel CKA is defined as:
CKAkX,kY (X, Y) =
⟨eKX, eKY ⟩F
∥eKX∥F ∥eKY ∥F
.
(21)
where ⟨A, B⟩F = tr(A⊤B). With positive semidefinite kernels and the biased HSIC estimator, the numerator is nonnegative,
and kernel CKA typically lies in [0, 1].
Unbiased Centered Kernel Alignment.
The biased HSIC estimator can yield inflated similarity scores at finite sample
sizes. Song et al. (2012) derived an unbiased HSIC estimator by recognizing that HSIC can be formulated as a U-statistic.
Following Kornblith et al. (2019), we substitute the unbiased estimator into the CKA formula. Let eKX = HKXH be the
centered Gram matrix with diagonal set to zero. The unbiased HSIC estimator is:
HSICu(KX, KY ) =
1
n(n −3)
 
tr( eKX eKY ) + 1⊤eKX1 · 1⊤eKY 1
(n −1)(n −2)
−
2
n −21⊤eKX eKY 1
!
.
(22)
Unbiased CKA replaces both numerator and denominator of Equation (21) with this estimator. Unlike the biased version,
unbiased CKA can take small negative values at finite n.
Canonical Correlation Analysis (CCA)-based similarity.
CCA (Weenink, 2003) measures linear subspace alignment.
The sample canonical correlations {ρi}r
i=1 (with r = rank(eΣXY )) are the singular values of the whitened cross-covariance
operator
eTCCA = eΣ
−1
2
XX eΣXY eΣ
−1
2
Y Y .
(23)
Common scalar summaries include the mean canonical correlation
1
r
Pr
i=1 ρi or a weighted average as used in
SVCCA (Raghu et al., 2017) and PWCCA (Morcos et al., 2018).
Singular Vector Canonical Correlation Analysis (SVCCA).
SVCCA (Raghu et al., 2017) combines dimensionality
reduction via singular value decomposition (SVD) with CCA. First, truncated SVD is applied to each representation to retain
the top principal components, yielding X′ ∈Rn×p and Y′ ∈Rn×q. Then CCA is applied to the reduced representations,
yielding canonical correlations {ρi}r
i=1. The SVCCA similarity is the mean canonical correlation:
SVCCA(X, Y) = 1
r
r
X
i=1
ρi.
(24)
Projection Weighted Canonical Correlation Analysis (PWCCA).
PWCCA (Morcos et al., 2018) improves upon
SVCCA by weighting canonical correlations according to their importance in explaining the original representations. Let
hX
i and hY
i denote the i-th canonical variables (projections onto canonical directions). The weight for the i-th canonical
correlation is proportional to how much variance it explains:
αi =
dx
X
j=1
|⟨hX
i , X:,j⟩|,
(25)
where X:,j is the j-th column of X. The PWCCA similarity is the weighted mean:
PWCCA(X, Y) =
Pr
i=1 αiρi
Pr
i=1 αi
.
(26)
This weighting ensures that canonical correlations corresponding to principal directions receive higher weight than those
corresponding to noise dimensions.
12

## Page 13

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
RV coefficient.
The RV (“Relation between two sets of Variables”) coefficient (Robert & Escoufier, 1976; Smilde et al.,
2009) is a multivariate generalization of the squared Pearson correlation. It measures the similarity between two configuration
matrices via their inner-product (Gram) matrices. Let WX = XX⊤and WY = YY⊤be the sample inner-product matrices.
The RV coefficient is:
RV(X, Y) =
tr(WXWY )
p
tr(W2
X) tr(W2
Y )
∈[0, 1].
(27)
C.2.2. GEOMETRIC METRICS
Representational Similarity Analysis (RSA) via Spearman correlation of dissimilarity matrices.
RSA (Kriegeskorte
et al., 2008) compares the geometry induced by pairwise dissimilarities. Let δ(·, ·) be a dissimilarity on representation
vectors (e.g., correlation distance δ(u, v) = 1−corr(u, v), cosine distance). Define Representational Dissimilarity Matrices
(RDMs)
(DX)ij = δ(xi, xj),
(DY )ij = δ(yi, yj),
(28)
and let vec△(D) ∈Rn(n−1)/2 denote vectorization of the strict upper triangle. RSA is then computed as a rank correlation
between the two RDM vectors:
RSA(X, Y) = ρS(vec△(DX), vec△(DY )) ,
(29)
where Spearman’s ρ can be expressed as Pearson correlation of ranks,
ρS(u, v) = corr(rank(u), rank(v)) .
(30)
Procrustes distance.
The orthogonal Procrustes distance (Williams et al., 2021) measures the minimal Euclidean distance
between two representations after optimal orthogonal alignment. Assuming dx = dy = d, the optimal orthogonal matrix
Q∗∈O(d) is:
Q∗= argmin
Q∈O(d)
∥X −YQ∥2
F ,
(31)
which has the closed-form solution Q∗= VU⊤where UΣV⊤= X⊤Y is the SVD. The Procrustes distance is:
dProc(X, Y) = ∥X −YQ∗∥F .
(32)
To convert to a similarity in [0, 1], one can use 1 −d2
Proc/(∥X∥2
F + ∥Y∥2
F ) after appropriate normalization.
C.2.3. NEIGHBORHOOD METRICS
Mutual k-Nearest Neighbors (mKNN).
mKNN (Huh et al., 2024) focuses on local topology. For each anchor sample i,
define the set of its k nearest neighbors according to a distance measure dist(·, ·) in X and Y,
NX(i) = KNNk(i; X),
NY(i) = KNNk(i; Y),
(33)
where KNNk(i; X) denotes the indices of the k samples (excluding i) that minimize dist(xi, xj). mKNN is then defined as
the average fraction of shared neighbors:
mKNNk(X, Y) = 1
n
n
X
i=1
|NX(i) ∩NY(i)|
k
∈[0, 1].
(34)
Cycle-kNN (bidirectional k-NN).
While mKNN measures one-directional neighborhood overlap, cycle-kNN enforces
bidirectional consistency (Huh et al., 2024). A pair (i, j) forms a cycle if j ∈NX(i) and i ∈NX(j) (mutual neighbors in
X), and similarly for Y. Define the set of bidirectional neighbors:
CX(i) = {j : j ∈NX(i) and i ∈NX(j)}.
(35)
Cycle-kNN measures the overlap of these symmetric neighborhoods:
cycle-kNNk(X, Y) = 1
n
n
X
i=1
|CX(i) ∩CY(i)|
max(|CX(i)|, 1) ∈[0, 1].
(36)
This metric is stricter than mKNN, requiring that shared neighbors be mutually recognized in both representation spaces.
13

## Page 14

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
CKA with Neighborhood Alignment (CKNNA).
CKNNA (Huh et al., 2024) combines the kernel-based formulation of
CKA with local neighborhood structure. Instead of computing CKA on full Gram matrices, CKNNA restricts interaction to
k-nearest neighbor graphs. Let AX ∈{0, 1}n×n be the adjacency matrix of the k-NN graph on X, with (AX)ij = 1 if
j ∈NX(i) or i ∈NX(j). CKNNA applies the CKA formula (Equation (21)) to the centered adjacency matrices:
CKNNAk(X, Y) =
⟨HAXH, HAY H⟩F
∥HAXH∥F ∥HAY H∥F
.
(37)
D. Theoretical Derivations
In this section, we provide the theoretical justification for the confounding factors identified in Section 4.
D.1. Permutation validity, super-uniformity, and gating
This section formalizes the finite-sample validity of permutation calibration.
Definition D.1 (Super-uniformity). A p-value p is super-uniform under H0 if for all t ∈[0, 1],
PH0(p ≤t) ≤t.
(38)
Equivalently, p-values under H0 are stochastically larger than Unif(0, 1), which is sufficient for valid Type-I error control.
Lemma D.2 (Permutation p-values are super-uniform). Under Assumption 3.1, the permutation p-value in Equation (10)
satisfies super-uniformity: PH0(p ≤α) ≤α for all α ∈[0, 1] (finite-sample validity).
Proof of Lemma D.2. Let sobs = s(X, Y) be the observed statistic and let s(k) = s(X, πk(Y)) for k = 1, . . . , K be the
statistics computed on permuted pairings. Under Assumption 3.1, the vector (sobs, s(1), . . . , s(K)) is exchangeable: its joint
distribution is invariant to permutations of the indices. Consider the (upper) rank
R = 1 + #{k ∈{1, . . . , K} : s(k) ≥sobs} ∈{1, . . . , K + 1}.
(39)
If the scores are almost surely distinct, exchangeability implies that the rank of sobs among {sobs, s(1), . . . , s(K)} is uniform
on {1, . . . , K + 1}. With possible ties, the add-one p-value of Phipson & Smyth (2010),
p =
R
K + 1,
(40)
is conservative, implying PH0(p ≤α) ≤α for all α ∈[0, 1].
Proof of Corollary 5.1. Let sobs = s(X, Y) and s(k) = s(X, πk(Y)) for k = 1, . . . , K. Under Assumption 3.1, the vector
(sobs, s(1), . . . , s(K)) is exchangeable. Let
τα := s(⌈(1−α)(K+1)⌉)
be the (1 −α)-quantile defined via the order statistic of the combined multiset {sobs, s(1), . . . , s(K)}. Define the (upper)
rank
R = 1 + #{k ∈{1, . . . , K} : s(k) ≥sobs} ∈{1, . . . , K + 1},
and the corresponding add-one p-value p = R/(K + 1). By construction of τα, the rejection event {sobs > τα} implies that
sobs lies among the largest ⌊α(K + 1)⌋values of {sobs, s(1), . . . , s(K)}, hence R ≤α(K + 1) and therefore p ≤α. By
Lemma D.2, PH0(p ≤α) ≤α, which yields
PH0(sobs > τα) ≤PH0(p ≤α) ≤α.
14

## Page 15

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
D.2. Monotone invariance of rank-based calibration
The following proposition is a standard result in randomization inference; we state it here for completeness and to clarify its
role in justifying the calibrated score design.
Proposition D.3 (Monotone invariance of rank-based calibration (Lehmann & Romano, 2005)). Let g : R →R be strictly
increasing. Define pg by applying Equation (10) to the transformed statistic g ◦s using the same permutations. Then pg = p,
and likewise the null percentile (the rank of sobs among the combined set) is invariant under g.
Proof. Let g be strictly increasing. For any two real numbers a, b, we have a ≥b if and only if g(a) ≥g(b). Therefore, for
each permutation draw k,
1{s(k) ≥sobs} = 1{g(s(k)) ≥g(sobs)}.
(41)
Summing over k shows that the permutation rank R (and thus the add-one p-value) is unchanged by applying g to both
the observed and permuted statistics. The same argument applies to the null percentile, since the ordering of samples is
preserved under g.
D.3. Post-selection inflation and aggregation-aware validity
Proposition D.4 (Validity for aggregation-aware calibration). Let T be any measurable aggregation operator applied to
a layer-wise similarity matrix S (e.g., max, row-max, top-k). If Tobs = T(S) is calibrated against the permutation null
{T(S(k))}K
k=1 as in Equation (16), then the resulting pagg is super-uniform under H0.
Proof of Proposition D.4. Let T be any measurable functional of the full data (representations across all layers), producing
the scalar report Tobs. Under Assumption 3.1 and consistent layer-wise permutation of sample correspondences, the vector
(Tobs, T (1), . . . , T (K)) is exchangeable. Applying the same rank argument as in Section D.1 yields super-uniformity for the
add-one p-value in Equation (16).
D.4. The width confounder
This appendix provides concrete calculations that justify the width confounder using Random Matrix Theory (RMT): even
under independence, interaction operators have non-trivial magnitude and spectrum when d is not negligible relative to n.
Proof of Proposition 4.1. Let X ∈Rn×dx and Y ∈Rn×dy have i.i.d. rows with mean 0, identity covariance, and xi and
yi independent. Let H = In −1
n1n1⊤
n be the centering matrix, so Xc = HX and Yc = HY. Since H is symmetric and
idempotent (H2 = H), the sample cross-covariance is
eC =
1
n −1X⊤
c Yc =
1
n −1X⊤HY.
(42)
Denote entry (a, b) as eCab. Expanding via Hij = δij −1
n:
eCab =
1
n −1


n
X
i=1
XiaYib −1
n
 n
X
i=1
Xia
 n
X
j=1
Yjb


.
(43)
We compute E[ eC2
ab] using independence of xi and yj for all i, j, zero means, and identity covariance.
Term 1: E

(P
i XiaYib)2
= P
i,j E[XiaXja]E[YibYjb]. For i ̸= j, independence across rows and zero mean give
E[XiaXja] = E[Xia]E[Xja] = 0. For i = j, we have E[X2
ia]E[Y 2
ib] = 1. Thus E

(P
i XiaYib)2
= n.
Term 2: E

(P
i XiaYib)(P
j Xja)(P
k Ykb)

= P
i,j,k E[XiaXja]E[YibYkb]. This is nonzero only when i = j and i = k,
yielding P
i 1 · 1 = n.
Term 3: E

(P
i Xia)2(P
j Yjb)2
= E[(P
i Xia)2]E[(P
j Yjb)2] = n · n = n2.
15

## Page 16

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
Combining:
E
h
eC2
ab
i
=
1
(n −1)2

n −2
n · n + n2
n2

=
1
(n −1)2 (n −2 + 1) =
1
n −1.
(44)
Summing over all entries:
E
h
∥eC∥2
F
i
=
dx
X
a=1
dy
X
b=1
E[ eC2
ab] = dxdy
n −1.
(45)
Interpretation.
The null interaction energy is O(dxdy/n). In the common regime dx, dy ≍n, the null energy is O(n) and
therefore does not vanish. Since many spectral similarity metrics aggregate singular values (e.g., via ∥eC∥2
F = P
i σ2
i (eC)),
this already explains a positive baseline under H0 and its dependence on (n, dx, dy).
Why we use permutation rather than closed forms.
Closed-form bulk edges are ensemble- and normalization-specific
and are brittle to the preprocessing used in practice (e.g., centering, whitening, kernelization). Moreover, finite-n corrections
can be non-negligible. We therefore estimate the relevant right-tail behavior nonparametrically via permutation. This yields
a conservative, implementation-faithful estimate of chance fluctuations without relying on fragile analytical formulas.
D.5. The depth confounder
Here we formalize why selection-based summaries (e.g., maximum similarity over layer pairs) inflate with the size of the
search space using Extreme Value Theory (EVT).
Let S = {Sℓ,ℓ′ : 1 ≤ℓ≤LA, 1 ≤ℓ′ ≤LB} denote the collection of null similarity fluctuations under H0, and let
M = LALB.
Assumption D.5 (Uniform sub-Gaussian right tails and integrability). There exist µ ∈R and σ > 0 such that for all (ℓ, ℓ′)
and all t ≥0,
P(Sℓ,ℓ′ −µ ≥t) ≤exp

−t2
2σ2

.
(46)
Moreover, each Sℓ,ℓ′ is integrable: E|Sℓ,ℓ′| < ∞for all (ℓ, ℓ′).
Proposition D.6 (Maximal inequality, no independence required). Under Assumption D.5 and for M ≥2,
E
h
max
ℓ,ℓ′ Sℓ,ℓ′
i
≤µ + C σ
p
log M,
(47)
where C > 0 is a constant (e.g., one can take C = 3).
Proof. Let Z := maxℓ,ℓ′ Sℓ,ℓ′ −µ. Since M < ∞and E|Sℓ,ℓ′| < ∞for all (ℓ, ℓ′), we have
E|Z| ≤E
h
max
ℓ,ℓ′ |Sℓ,ℓ′|
i
+ |µ| ≤
X
ℓ,ℓ′
E|Sℓ,ℓ′| + |µ| < ∞,
(48)
so Z is integrable, and the tail-integration formula applies. By the union bound and Assumption D.5,
P(Z ≥t) ≤M exp

−t2
2σ2

for all t ≥0.
(49)
Using the tail-integration formula for an integrable real-valued random variable Z,
E[Z] =
Z ∞
0
P(Z ≥t) dt −
Z ∞
0
P(Z ≤−t) dt ≤
Z ∞
0
P(Z ≥t) dt,
(50)
and the bound P(Z ≥t) ≤1, we obtain
E[Z] ≤
Z ∞
0
min

1, M exp

−t2
2σ2

dt.
(51)
16

## Page 17

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
Let t0 = σ√2 log M. This value of t0 is the solution of M exp
−t2
0/2σ2
= 1, i.e., the crossover where the bound
min{1, ·} switches. Splitting the integral at t0 yields
E[Z] ≤t0 + M
Z ∞
t0
exp

−t2
2σ2

dt.
(52)
Applying the standard Gaussian tail bound
R ∞
t0 e−t2/(2σ2)dt ≤(σ2/t0)e−t2
0/(2σ2) gives
E[Z] ≤σ
p
2 log M +
σ
√2 log M .
(53)
For M ≥2, the right-hand side is at most 3σ√log M, proving the claim with C = 3.
Remark.
When the Sℓ,ℓ′ are i.i.d. (or weakly dependent), classical Extreme Value Theory yields sharper asymptotics. For
example, if Sℓ,ℓ′ ∼N(µ0, σ2
0) i.i.d., the centered maximum converges to a Gumbel distribution and
E[Tmax] ≈µ0 + σ0
√
2 ln M −ln ln M + ln 4π
2
√
2 ln M

,
(54)
as stated in standard references (Cram´er, 1999; Embrechts et al., 2013). Real layer-wise similarities are dependent, so the
approximation above should be treated as heuristic; Proposition D.6 provides a dependence-robust upper bound.
D.6. Null Baselines for Neighborhood Metrics
The preceding analysis focused on spectral metrics whose null baselines scale with d/n. Neighborhood-based metrics such
as mutual k-NN follow a fundamentally different regime, which we now characterize.
Definition D.7 (Mutual k-NN overlap). For representations X ∈Rn×dx, Y ∈Rn×dy and neighborhood size k < n, let
NX(i) ⊆{1, . . . , n} \ {i} denote the indices of the k nearest neighbors of sample i in X (e.g., Euclidean or cosine), and
similarly for NY(i). The mutual k-NN overlap is
mKNN(X, Y) = 1
n
n
X
i=1
|NX(i) ∩NY(i)|
k
.
(55)
Proposition D.8 (Uniformity of k-NN index sets under i.i.d. sampling). Fix an anchor index i ∈{1, . . . , n}. Let
x1, . . . , xn ∈Rd be i.i.d. and define the k-NN set NX(i) ⊆{1, . . . , n} \ {i} using a fixed distance dist(·, ·). Assume either
(i) {dist(xi, xj)}j̸=i are almost surely distinct, or (ii) ties are broken by selecting a uniformly random k-subset among the
set of minimizers. Then NX(i) is uniformly distributed over the
n−1
k

k-subsets of {1, . . . , n} \ {i}.
Proof. Let I := {1, . . . , n} \ {i} be the candidate-neighbor index set. For any permutation π of I, i.i.d. sampling implies
(xj)j∈I
d= (xπ(j))j∈I.
The k-NN selection rule depends on the candidate points only through their distances to xi, so permuting the candidate
indices permutes the resulting neighbor set. Under either the no-ties assumption or the stated uniform tie-break rule, for any
two k-subsets S, S′ ⊆I there exists a permutation π with π(S) = S′ and hence
P
NX(i) = S

= P
NX(i) = S′
.
Since the events {NX(i) = S} over all |S| = k partition the sample space, each has probability
n−1
k
−1.
Theorem D.9 (Null baseline for mutual k-NN). Let X, Y ∈Rn×d have i.i.d. rows, with X independent of Y. Define NX(i)
and NY(i) as in Definition D.7, using either almost sure absence of distance ties or uniform random tie-breaking. Then
EH0

mKNN(X, Y)

=
k
n −1.
17

## Page 18

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
Proof. Fix an anchor i. By Proposition D.8, NX(i) and NY(i) are each uniform random k-subsets of the (n −1)-element
set {1, . . . , n} \ {i}. Moreover, since X and Y are independent and NX(i) (resp. NY(i)) is a measurable function of X
(resp. Y), the sets NX(i) and NY(i) are independent.
Therefore |NX(i) ∩NY(i)| has a hypergeometric distribution with population size n −1, number of “successes” k, and
draws k, giving
EH0

|NX(i) ∩NY(i)|

=
k2
n −1.
Substituting into the definition of mKNN,
EH0

mKNN(X, Y)

= 1
n
n
X
i=1
EH0
|NX(i) ∩NY(i)|
k

= 1
n
n
X
i=1
k
n −1 =
k
n −1.
Proposition D.10 (Per-anchor variance and generic bounds for mKNN under the null). Under the assumptions of The-
orem D.9, for each anchor i the intersection size Hi := |NX(i) ∩NY(i)| is hypergeometric with mean k2/(n −1) and
variance
Var[Hi] =
k2(n −1 −k)2
(n −1)2(n −2).
Moreover, since mKNN(X, Y) ∈[0, 1] deterministically, we have the fully general bound
Var[mKNN(X, Y)] ≤1
4.
If one additionally assumes that the per-anchor terms {|NX(i) ∩NY(i)|/k}n
i=1 are independent (this is a modeling
assumption, not a consequence of H0), then Var[mKNN(X, Y)] = O(1/n).
Proof. The hypergeometric variance formula gives
Var[Hi] = k ·
k
n −1

1 −
k
n −1

· (n −1) −k
(n −1) −1 = k2(n −1 −k)2
(n −1)2(n −2).
The bound Var[mKNN] ≤1/4 follows from mKNN ∈[0, 1]. Under the stated additional independence assumption across
anchors,
Var

mKNN(X, Y)

= 1
n Var
H1
k

=
1
nk2 Var[H1],
which is O(1/n) for fixed k.
E. Implementation
A key advantage of null calibration is its simplicity: the framework can be applied to any similarity metric with minimal
code changes. This section provides pseudocode for the two main calibration procedures described in the paper.
Scalar null calibration.
Algorithm 1 shows the complete procedure for calibrating a single similarity comparison. The
only requirement is a function similarity(X,Y) that computes the raw metric. The algorithm returns both a permutation
p-value and a calibrated score with a principled zero point.
Aggregation-aware calibration for layer-wise comparisons.
When comparing models with multiple layers and reporting
a summary statistic (e.g., maximum similarity across layer pairs), the aggregation step must also be calibrated. Algorithm 2
shows how to extend scalar calibration to this setting. The key insight is that the same sample permutation must be applied
consistently across all layers.
Computational cost.
Scalar calibration requires K additional similarity computations. Aggregation-aware calibration
requires K×LA×LB computations, which can be parallelized across permutations. In practice, K = 200–500 permutations
suffice for stable p-values and threshold estimation.
18

## Page 19

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
Algorithm 1 Scalar Null Calibration
Require: Representations X ∈Rn×dx, Y ∈Rn×dy
Require: Similarity function sim(·, ·), permutations K, significance level α
Ensure: Calibrated score scal, p-value p
1: sobs ←sim(X, Y) {Observed similarity}
2: null scores ←[]
3: for k = 1 to K do
4:
π ←random permutation(n) {Permute sample indices}
5:
Yπ ←Y[π, :] {Permute rows of Y}
6:
null scores[k] ←sim(X, Yπ)
7: end for
8: combined ←[sobs] ∪null scores {Combined set}
9: τα ←quantile(combined, 1 −α) {Critical threshold from combined set}
10: p ←1+PK
k=1 1[null scores[k]≥sobs]
K+1
{Permutation p-value}
11: scal ←max

sobs−τα
smax−τα , 0

{Calibrated score (use smax = 1 for bounded metrics)}
12: return scal, p
Algorithm 2 Aggregation-Aware Null Calibration
Require: Layer representations {X(ℓ)}LA
ℓ=1, {Y(ℓ′)}LB
ℓ′=1 (all n samples)
Require: Similarity function sim(·, ·), aggregator T (e.g., max), permutations K, level α
Ensure: Calibrated aggregate Tcal, p-value pagg
1: {Compute observed similarity matrix}
2: for ℓ= 1 to LA do
3:
for ℓ′ = 1 to LB do
4:
S[ℓ, ℓ′] ←sim(X(ℓ), Y(ℓ′))
5:
end for
6: end for
7: Tobs ←T(S) {e.g., maxℓ,ℓ′ S[ℓ, ℓ′]}
8: null aggregates ←[]
9: for k = 1 to K do
10:
π ←random permutation(n) {Single permutation for all layers}
11:
for ℓ= 1 to LA do
12:
for ℓ′ = 1 to LB do
13:
S(k)[ℓ, ℓ′] ←sim(X(ℓ), Y(ℓ′)[π, :]) {Same π for all ℓ′}
14:
end for
15:
end for
16:
null aggregates[k] ←T(S(k)) {Aggregate under null}
17: end for
18: combined ←[Tobs] ∪null aggregates {Combined set}
19: τ agg
α
←quantile(combined, 1 −α) {Critical threshold from combined set}
20: pagg ←1+PK
k=1 1[null aggregates[k]≥Tobs]
K+1
21: Tcal ←max

Tobs−τ agg
α
smax−τ agg
α
, 0

22: return Tcal, pagg
19

## Page 20

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
F. Additional Experimental Results
This appendix provides additional analyses that support the main text claims.
F.1. Phase diagrams across different noise distributions
The theoretical analysis in Section 4 assumes Gaussian entries for tractability, but real neural network activations rarely
follow Gaussian distributions. Instead, they often exhibit heavy tails, sparsity, or multimodality. A critical question is whether
our calibration, which makes no distributional assumptions, remains effective under such deviations.
Figure 8 shows phase diagrams under different noise distributions: Gaussian, Student-t (ν = 3), Laplace, and Gaussian
mixtures. Each panel shows raw scores (left) and calibrated scores (right) across the (d/n, σ) grid, where σ controls the
noise level added to a fixed shared signal. At low σ, the signal dominates and both raw and calibrated scores correctly
indicate high similarity. At high σ, noise overwhelms the signal, and similarity should approach zero. The key finding
is that raw scores remain elevated (around 0.4–0.6) even at high noise levels where no detectable signal remains, while
calibrated scores correctly collapse to near-zero. This pattern holds across all noise distributions tested, confirming that
permutation-based calibration adapts to the data-generating process without requiring explicit distributional modeling.
0.25
0.50
1.00
d/n
0.0
0.5
1.0
1.5
2.0
2.5
3.0
Noise level σ
Raw score
0.25
0.50
1.00
d/n
Noise level σ
Calibrated score
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
(a) Gaussian
0.25
0.50
1.00
d/n
0.0
0.5
1.0
1.5
2.0
2.5
3.0
Noise level σ
Raw score
0.25
0.50
1.00
d/n
Noise level σ
Calibrated score
0.2
0.4
0.6
0.8
1.0
0.2
0.4
0.6
0.8
1.0
(b) Student-t (ν = 3)
0.25
0.50
1.00
d/n
0.0
0.5
1.0
1.5
2.0
2.5
3.0
Noise level σ
Raw score
0.25
0.50
1.00
d/n
Noise level σ
Calibrated score
0.2
0.4
0.6
0.8
1.0
0.2
0.4
0.6
0.8
1.0
(c) Laplace
0.25
0.50
1.00
d/n
0.0
0.5
1.0
1.5
2.0
2.5
3.0
Noise level σ
Raw score
0.25
0.50
1.00
d/n
Noise level σ
Calibrated score
0.2
0.4
0.6
0.8
1.0
0.2
0.4
0.6
0.8
1.0
(d) Gaussian mixture
Figure 8. Phase diagrams under different noise types. Calibrated scores (right) collapse to near-zero at high noise levels across the
(d/n, σ) grid, while raw scores (left) exhibit systematic positive bias. Calibration remains effective regardless of tail behavior.
F.2. SNR sweep heatmaps
The experiments of the main paper (Figure 4) demonstrated that calibration eliminates false positives under H0 while
preserving sensitivity to fixed signals. This section extends the analysis by characterizing how calibrated similarity varies
jointly with signal strength, noise level, and dimensionality ratio, thereby delineating the regimes in which similarity
estimation remains reliable.
Figure 9 presents heatmaps of raw scores (top row) and calibrated scores (bottom row) across the (Noise level, Signal
strength) grid for three signal ranks (r ∈{1, 5, 10}). The results reveal a clear phase transition structure. Raw scores (top)
show uniformly high values across most of the grid, obscuring the true detection boundary. Calibrated scores (bottom)
reveal the underlying signal: high scores concentrate in the low-noise, high-signal corner (bottom-left), while scores
correctly collapse to zero as noise increases (moving right) or signal weakens (moving down). The detection boundary shifts
rightward (tolerating higher noise) as signal rank increases. This phase structure is meaningful: it delineates when similarity
measurements carry information about shared structure versus when they reflect only finite-sample artifacts.
Figure 10 provides a complementary view by collapsing the 2D heatmaps into 1D curves, plotting calibrated score against
noise level for different signal strengths s. As expected, calibrated scores decrease monotonically with noise level: at low
20

## Page 21

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
0
1
2
3
Noise level σ
0.25
0.5
1
1.5
2
3
4
Signal strength
Raw score
0.0
0.2
0.4
0.6
0.8
1.0
(a) Rank r = 1
0
1
2
3
Noise level σ
0.25
0.5
1
1.5
2
3
4
Signal strength
Raw score
0.0
0.2
0.4
0.6
0.8
1.0
(b) Rank r = 5
0
1
2
3
Noise level σ
0.25
0.5
1
1.5
2
3
4
Signal strength
Raw score
0.0
0.2
0.4
0.6
0.8
1.0
(c) Rank r = 10
0
1
2
3
Noise level σ
0.25
0.5
1
1.5
2
3
4
Signal strength
Calibrated score
0.0
0.2
0.4
0.6
0.8
1.0
(d) Rank r = 1
0
1
2
3
Noise level σ
0.25
0.5
1
1.5
2
3
4
Signal strength
Calibrated score
0.0
0.2
0.4
0.6
0.8
1.0
(e) Rank r = 5
0
1
2
3
Noise level σ
0.25
0.5
1
1.5
2
3
4
Signal strength
Calibrated score
0.0
0.2
0.4
0.6
0.8
1.0
(f) Rank r = 10
Figure 9. SNR sweep heatmaps (calibrated scores). Higher-rank signals are detected at higher noise levels. The clear gradient confirms
calibration preserves sensitivity to genuine structure.
noise, scores are high (reflecting the detectable shared signal), while at high noise, scores collapse to zero (reflecting that the
signal is buried). Stronger signals (larger s) maintain elevated scores across a wider range of noise levels before eventually
succumbing. Higher-rank signals (r = 5, 10) show more gradual decay compared to r = 1, consistent with their greater
statistical detectability. All curves converge to zero at high noise, confirming that the null floor is correctly calibrated
regardless of signal strength or rank.
0
1
2
3
Noise level σ
0.0
0.2
0.4
0.6
0.8
1.0
Calibrated score
r = 1
s = 0.25
s = 0.5
s = 1
s = 1.5
s = 2
s = 3
s = 4
0
1
2
3
Noise level σ
r = 5
s = 0.25
s = 0.5
s = 1
s = 1.5
s = 2
s = 3
s = 4
0
1
2
3
Noise level σ
r = 10
s = 0.25
s = 0.5
s = 1
s = 1.5
s = 2
s = 3
s = 4
Figure 10. Calibrated scores decay with noise level. Each curve shows calibrated score versus noise level for a fixed signal strength s.
Stronger signals maintain elevated scores across wider noise ranges; all curves converge to zero at high noise.
F.3. Comparing calibration approaches
A natural question is whether the choice of calibration summary affects the correction. We consider several approaches:
(i) gated score, which thresholds at a significance level and rescales (α ∈{0.05, 0.1}); (ii) null-centered, subtracting the null
mean; (iii) z-score, standardizing by null mean and standard deviation; and (iv) ARI-style, applying the chance-correction
formula (s −E[s])/(smax −E[s]). Figure 11 evaluates these variants across metrics as d/n increases.
The results demonstrate that the gated score, null-centered, and ARI-style corrections all successfully collapse to appropriate
null baselines across all metrics, regardless of whether the raw metric exhibits severe inflation (CKA, approaching 0.8) or
21

## Page 22

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
mild inflation (RSA and mKNN, below 0.1). The z-score calibration, while correcting the mean, can exhibit artifacts when
the null distribution is skewed, as occurs for bounded metrics like CKA at high d/n, making it less suitable as a universal
correction.
128
256
512
1024
2048
d
128
256
512
1024
2048
4096
n
CKA (lin)
128
256
512
1024
2048
d
gated q90
128
256
512
1024
2048
d
gated q95
128
256
512
1024
2048
d
null-centered
128
256
512
1024
2048
d
z-score
128
256
512
1024
2048
d
ARI-adjusted
0.00
0.25
0.50
0.75
0.00
0.25
0.50
0.75
0.00
0.25
0.50
0.75
−1
0
1
−1
0
1
−1
0
1
(a) CKA linear
128
256
512
1024
2048
d
128
256
512
1024
2048
4096
n
CKA (rbf)
128
256
512
1024
2048
d
gated q90
128
256
512
1024
2048
d
gated q95
128
256
512
1024
2048
d
null-centered
128
256
512
1024
2048
d
z-score
128
256
512
1024
2048
d
ARI-adjusted
0.00
0.25
0.50
0.75
0.00
0.25
0.50
0.75
0.00
0.25
0.50
0.75
−1
0
1
−1
0
1
−1
0
1
(b) CKA RBF
128
256
512
1024
2048
d
128
256
512
1024
2048
4096
n
RSA
128
256
512
1024
2048
d
gated q90
128
256
512
1024
2048
d
gated q95
128
256
512
1024
2048
d
null-centered
128
256
512
1024
2048
d
z-score
128
256
512
1024
2048
d
ARI-adjusted
−0.025
0.000
0.025
0.050
−0.025
0.000
0.025
0.050
−0.025
0.000
0.025
0.050
−1
0
1
−1
0
1
−1
0
1
(c) RSA (Spearman)
128
256
512
1024
2048
d
128
256
512
1024
2048
4096
n
mKNN
128
256
512
1024
2048
d
gated q90
128
256
512
1024
2048
d
gated q95
128
256
512
1024
2048
d
null-centered
128
256
512
1024
2048
d
z-score
128
256
512
1024
2048
d
ARI-adjusted
0.000
0.025
0.050
0.075
0.000
0.025
0.050
0.075
0.000
0.025
0.050
0.075
−1
0
1
−1
0
1
−1
0
1
(d) Mutual k-NN
Figure 11. Comparing calibration approaches across metrics. Each panel shows raw scores alongside four calibration variants (gated
score, null-centered, z-score, ARI-style) as d/n increases. Gated score, null-centered, and ARI-style corrections collapse to appropriate
baselines; z-score exhibits artifacts for skewed null distributions.
F.4. Comparison with analytical debiasing
We validate our empirical null calibration by comparing it to existing analytical bias corrections for CKA. Figure 12 shows
the difference between our calibrated CKA and two existing estimators: the debiased CKA of Murphy et al. (2024) and the
dep-cols CKA of Chun et al. (2025).
Our calibrated CKA closely matches the debiased CKA estimator, indicating that our calibration automatically corrects the
dominant width-induced bias without requiring a metric-specific derivation. In contrast, dep-cols CKA is designed to correct
column dependence, which is not a confound in our experimental setup (where columns are independent by construction),
and as a result, it attenuates the true signal under H1.
22

## Page 23

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
2
−2
2
0
2
2
2
4
d/n
−0.6
−0.4
−0.2
0.0
0.2
0.4
0.6
Difference under signal
Calibrated − Debiased
Calibrated − Dep-cols
2
−2
2
0
2
2
2
4
d/n
−0.004
−0.002
0.000
0.002
0.004
Difference under null
Calibrated − Debiased
Calibrated − Dep-cols
Figure 12. Calibration recovers analytical debiasing. Difference between calibrated CKA and existing estimators (n = 1024, d/n
swept). (Left) Under signal. (Right) Under null.
F.5. Permutation budget analysis
Permutation-based calibration introduces a computational-statistical tradeoff: more permutations yield more stable threshold
estimates but increase runtime. Practitioners need guidance on the minimum budget required for reliable inference.
We analyze the stability of threshold estimates τα and calibrated scores as a function of the permutation budget K across
50 random seeds. Figure 13 shows two panels: the left panel displays threshold estimates, while the right panel shows
calibrated scores under H0. Threshold estimates (left) stabilize rapidly, reaching stable values by approximately K = 50 for
all metrics tested. Calibrated scores (right) exhibit more variability at very low budgets (K < 50), with occasional spikes
due to unstable threshold estimation, but converge to near-zero by K ≈100–200.
Based on these results, we recommend K ≥200. The computational cost scales linearly with K, so this recommendation
represents a favorable tradeoff between precision and efficiency.
0
100
200
300
400
500
Number of Permutations K
0.0
0.1
0.2
0.3
0.4
Threshold τ
Permutation budget
CKA (linear)
CKA (RBF)
mKNN
RSA
0
100
200
300
400
500
Number of Permutations K
0.0000
0.0025
0.0050
0.0075
0.0100
0.0125
0.0150
Calibrated score
Permutation budget
CKA (linear)
CKA (RBF)
mKNN
RSA
Figure 13. Permutation budget analysis. Left: threshold τα stabilizes by K ≈50. Right: calibrated scores under H0 converge to
near-zero by K ≈100–200. Shaded regions show variability across random seeds.
F.6. Full null drift results
The main text presents null drift results for a representative subset of metrics under Gaussian noise. Here, we present
additional results across all metrics evaluated in this work, including RSA, the RV coefficient, and Procrustes distance, as
well as results under heavy-tailed noise distributions.
Figure 14 presents results under Gaussian noise for all metrics. The severity of null drift varies substantially across metric
families: CKA variants exhibit the most severe inflation, followed by RV coefficient and CCA-variants, with neighborhood
metrics showing the mildest drift. This reflects the structural sensitivity of the metrics to high-dimensional spurious
correlations. Critically, calibration eliminates drift across all metrics, collapsing scores to zero regardless of the raw bias
magnitude.
Figure 15 extends these results to heavy-tailed noise (Student-t, ν = 3). The qualitative pattern is preserved: all metrics
23

## Page 24

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
exhibit positive drift under the null, and calibration eliminates this drift. The magnitude of raw bias is generally higher under
heavy-tailed noise, consistent with increased finite-sample variability, yet calibration adapts automatically without requiring
distributional knowledge.
128
256
512
1024
2048
4096
n
CKA (lin)
CKA (rbf)
mKNN
RSA
CCA
SVCCA
PWCCA
RV
Procrustes
128
256
512
1024
2048
d
128
256
512
1024
2048
4096
n
128
256
512
1024
2048
d
128
256
512
1024
2048
d
128
256
512
1024
2048
d
128
256
512
1024
2048
d
128
256
512
1024
2048
d
128
256
512
1024
2048
d
128
256
512
1024
2048
d
128
256
512
1024
2048
d
0.00
0.25
0.50
0.75
0.00
0.25
0.50
0.75
0.00
0.25
0.50
0.75
0.00
0.25
0.50
0.75
0.000
0.025
0.050
0.075
0.000
0.025
0.050
0.075
−0.025
0.000
0.025
0.050
−0.025
0.000
0.025
0.050
0.0
0.5
1.0
1.5
0.0
0.5
1.0
1.5
0.0
0.1
0.2
0.0
0.1
0.2
0
2
4
0
2
4
0.0
0.5
1.0
0.0
0.5
1.0
0.0
0.5
0.0
0.5
Figure 14. Full null drift results (Gaussian). Raw scores (top) exhibit systematic positive bias; calibrated scores (bottom) collapse to
zero.
128
256
512
1024
2048
4096
n
CKA (lin)
CKA (rbf)
mKNN
RSA
CCA
SVCCA
PWCCA
RV
Procrustes
128
256
512
1024
2048
d
128
256
512
1024
2048
4096
n
128
256
512
1024
2048
d
128
256
512
1024
2048
d
128
256
512
1024
2048
d
128
256
512
1024
2048
d
128
256
512
1024
2048
d
128
256
512
1024
2048
d
128
256
512
1024
2048
d
128
256
512
1024
2048
d
0.00
0.25
0.50
0.75
0.00
0.25
0.50
0.75
0.00
0.25
0.50
0.75
0.00
0.25
0.50
0.75
0.000
0.025
0.050
0.075
0.000
0.025
0.050
0.075
−0.05
0.00
0.05
−0.05
0.00
0.05
0
1
2
0
1
2
0.0
0.1
0.2
0.0
0.1
0.2
0
5
10
15
0
5
10
15
0.00
0.25
0.50
0.75
0.00
0.25
0.50
0.75
0.0
0.5
0.0
0.5
Figure 15. Full null drift results (heavy-tailed). Student-t (ν = 3) noise. The pattern is consistent across all metrics: calibration
eliminates spurious similarity regardless of noise distribution.
F.7. Extended PRH alignment results (image–text)
The main text establishes a divergence between local and global similarity metrics when applied to the Platonic Repre-
sentation Hypothesis (PRH): neighborhood-based metrics retain significant cross-modal alignment after calibration, while
spectral metrics lose their apparent convergence trend. A natural question is whether this finding is robust across model
families and metric variants.
Here we present comprehensive results across all five vision model families in the PRH setting (DINOv2, CLIP, ImageNet-
21K, MAE, and CLIP-finetuned) and a broad range of metrics spanning the local-to-global spectrum (Figures 16 and 17).
The results reinforce and extend the main text findings. Neighborhood metrics (mKNN, cycle-kNN, CKNNA) show a
consistent alignment trend across all vision families with a neighborhood size of 10. This pattern holds for both self-
supervised (DINOv2, MAE) and supervised (ImageNet-21K) pretraining objectives, as well as for both CLIP-aligned and
CLIP-finetuned variants. Spectral metrics (CKA linear, CKA RBF, unbiased CKA) show a different pattern: raw scores
suggest increasing alignment with model scale, but calibrated scores show no such scaling trend.
Statistical significance.
Beyond calibrated scores, we report permutation p-values to quantify statistical evidence against
the null hypothesis of no cross-modal alignment (Figure 18). All 204 vision–language model pairs are significant at p < 0.05,
with most achieving p < 0.005 (the minimum achievable with K = 200 permutations) for both local and global metrics.
This confirms that cross-modal similarity is statistically significant (i.e., has some alignment) across all model pairs. The
critical distinction between local and global metrics lies not in statistical significance but in the magnitude and trends of
calibrated scores. Local metrics show substantial alignment above the null threshold that persists across scales, whereas
global metrics, although significant, show no convergence in calibrated effect sizes.
F.8. Extended video–language alignment results
The main text extends the PRH analysis to video–language alignment following Zhu et al. (2026). Here, we provide
additional results to verify that the local-vs-global pattern observed for image–language alignment extends to the video
24

## Page 25

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.08
0.10
0.12
0.14
Alignment score
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.04
0.06
0.08
0.10
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.075
0.100
0.125
0.150
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.125
0.150
0.175
0.200
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.100
0.125
0.150
0.175
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(a) mKNN: Neighborhood overlap.
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
0.35
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.3
0.4
0.5
0.6
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(b) CKA RBF: Spectral alignment.
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.4
0.5
0.6
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.4
0.5
0.6
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.5
0.6
0.7
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.5
0.6
0.7
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(c) cycle-kNN: Bidirectional consistency.
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.0
0.2
0.4
0.6
0.8
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.15
0.20
0.25
0.30
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
0.35
0.40
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.30
0.35
0.40
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
0.35
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(d) Unbiased CKA.
Figure 16. PRH alignment results (all vision families). All five vision model families are shown (DINOv2, CLIP, ImageNet-21K, MAE,
CLIP-finetuned). The divergence between local and global metrics is consistent across all families.
25

## Page 26

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
0.35
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(a) CKA linear.
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.100
0.125
0.150
0.175
Alignment score
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.06
0.08
0.10
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.100
0.125
0.150
0.175
0.200
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.15
0.20
0.25
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.15
0.20
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(b) CKNNA.
Figure 17. Additional PRH metrics (all vision families). CKA linear (a) shows the same loss of convergence trend as CKA RBF.
CKNNA (b) shows consistent local alignment across all vision families.
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
p-value (log scale)
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
α = 0.05
(a) mKNN (k = 10).
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
p-value (log scale)
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
α = 0.05
(b) CKA linear.
Figure 18. Permutation p-values for PRH alignment. All model pairs are significant at p < 0.05, with most achieving p < 0.005 for
both local (a) and global (b) metrics. The difference between metric families lies in calibrated effect sizes, not significance.
26

## Page 27

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
modality.
We use 1024 samples from the PVD (Bolya et al., 2025; Cho et al., 2025) test set. We evaluate both video-native models
(VideoMAE (Tong et al., 2022)) and image models (DINOv2 and CLIP) applied to the middle frame of each video.
We compare these against the same three language model families used in the image–language experiments (BLOOM,
OpenLLaMA, LLaMA) at multiple scales. Figure 19 shows results for both spectral (CKA RBF) and neighborhood (mKNN)
metrics.
The pattern mirrors the image–language findings. For spectral metrics, raw scores suggest alignment, whereas calibrated
scores drop significantly, indicating that much of the apparent alignment is attributable to width and depth confounders. In
contrast, neighborhood metrics retain significant alignment after calibration, confirming that video and language representa-
tions share local topological structure.
F.9. Characterizing the locality of cross-modal alignment
The main text establishes that local neighborhood metrics retain significant alignment after calibration, while global
spectral metrics do not. A natural follow-up question is: how local is this alignment? Both mKNN and CKA-RBF
have hyperparameters that control their sensitivity to local versus global structure. By varying these parameters, we can
characterize the scale at which cross-modal alignment emerges.
Experimental setup.
We vary two locality parameters: the neighborhood size k in mKNN, testing k ∈{10, 20, 50, 100}
where smaller values focus on immediate neighbors and larger values consider broader local structure, and the RBF kernel
bandwidth σ in CKA-RBF, testing σ ∈{0.1, 0.5, 2.0, 5.0}, which controls the length scale over which the kernel assigns
significant weight.
RBF bandwidth.
The RBF (radial basis function) kernel is defined as k(x, y) = exp
−∥x −y∥2/2σ2
. The bandwidth
σ determines the length scale of similarity. When σ is small (e.g., 0.1), the kernel is sharply peaked: only very close points
contribute significantly to the Gram matrix, making the similarity measure sensitive to exact pairwise distances in the
immediate neighborhood. When σ is large (e.g., 5.0), the kernel is broad: even moderately distant points contribute, and the
similarity measure aggregates information over larger neighborhoods, becoming sensitive to coarser geometric structure.
Neighborhood size.
For mKNN, the parameter k controls how many nearest neighbors are considered when measuring
overlap. Small k (e.g., 10) measures agreement on immediate neighbors, i.e., the closest points to each sample, capturing
fine-grained local topology. Large k (e.g., 100) measures agreement on a broader neighborhood. With n = 1000 samples
and k = 100, we ask whether the 10% closest points agree across representations. Crucially, mKNN is a rank-based metric:
it asks which points are neighbors (ordinal information), not how close they are (cardinal information).
mKNN across k values.
Figure 20 shows the PRH alignment results for mKNN with varying k. A consistent pattern
emerges: all k values show significant alignment after calibration, with calibrated scores remaining well above zero even at
k = 100. However, the scaling trend is most pronounced at small k. For k = 10, raw scores show a clear upward trend with
model capacity that persists after calibration. At large k, this trend flattens even in raw scores. For k = 100, raw scores
plateau for larger models, suggesting that broader neighborhood agreement is already saturated across model scales. This
pattern indicates that scaling-driven improvement in alignment is concentrated at the finest topological level.
CKA-RBF across bandwidth values.
Figure 21, and the accompanying p-values in Figure 22, shows results for CKA-
RBF with varying bandwidth σ, revealing a different pattern from mKNN. At σ = 0.1 (very local), there is no significant
alignment after calibration: raw scores are near 1.0, reflecting the high similarity of any high-dimensional representations
under a sharply peaked kernel. However, calibrated scores collapse to approximately zero with p-values exceeding 0.05
for most model pairs, indicating that the observed similarity is indistinguishable from chance. At σ = 0.5, alignment
emerges, but with a flattening trend after calibration. Calibrated scores initially rise with model scale, then plateau and
slightly decline for the largest models. At σ = 2.0 and σ = 5.0, significant alignment persists, but the calibrated trend also
flattens, resembling the pattern observed for large-k mKNN: alignment exists, but scaling-driven improvement disappears
after calibration.
27

## Page 28

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.2
0.4
0.6
Alignment score
BLOOM
OpenLLaMA
LLaMA
VideoMAE
base
large
huge
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.2
0.4
0.6
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.2
0.3
0.4
0.5
0.6
0.7
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
giant
calibrated
uncalibrated
(a) CKA RBF: spectral alignment.
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.05
0.10
0.15
0.20
Alignment score
BLOOM
OpenLLaMA LLaMA
VideoMAE
base
large
huge
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.16
0.18
0.20
0.22
0.24
0.26
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.20
0.22
0.24
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
giant
calibrated
uncalibrated
(b) mKNN (k = 10): neighborhood overlap.
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.05
0.10
0.15
0.20
0.25
0.30
Alignment score
BLOOM
OpenLLaMA LLaMA
VideoMAE
base
large
huge
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.24
0.26
0.28
0.30
0.32
0.34
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.26
0.28
0.30
0.32
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
giant
calibrated
uncalibrated
(c) mKNN (k = 50): neighborhood overlap.
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.05
0.10
0.15
0.20
Alignment score
BLOOM
OpenLLaMA LLaMA
VideoMAE
base
large
huge
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.175
0.200
0.225
0.250
0.275
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
560M
1B1
1B7
3B
7B1
3B
7B
13B
7B
13B
30B
65B
0.20
0.22
0.24
0.26
0.28
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
giant
calibrated
uncalibrated
(d) CKNNA (k = 10): neighborhood overlap.
Figure 19. Video–language alignment results. (a) Spectral alignment drops substantially after calibration. (b–d) Neighborhood alignment
trend remains after calibration.
28

## Page 29

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.08
0.10
0.12
0.14
Alignment score
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.04
0.06
0.08
0.10
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.075
0.100
0.125
0.150
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.125
0.150
0.175
0.200
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.100
0.125
0.150
0.175
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(a) mKNN (k = 10)
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.08
0.10
0.12
0.14
0.16
Alignment score
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.06
0.08
0.10
0.12
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.100
0.125
0.150
0.175
0.200
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.15
0.20
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.100
0.125
0.150
0.175
0.200
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(b) mKNN (k = 20)
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.100
0.125
0.150
0.175
0.200
Alignment score
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.075
0.100
0.125
0.150
0.175
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.15
0.20
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.15
0.20
0.25
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.15
0.20
0.25
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(c) mKNN (k = 50)
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.15
0.20
0.25
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.10
0.15
0.20
0.25
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.15
0.20
0.25
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.15
0.20
0.25
0.30
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(d) mKNN (k = 100)
Figure 20. PRH alignment with varying neighborhood size k for mKNN. All k values show significant alignment after calibration. The
scaling trend is clearest at small k and flattens at large k, suggesting scaling improvements are concentrated at the finest local scale.
29

## Page 30

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.00
0.25
0.50
0.75
1.00
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.00
0.25
0.50
0.75
1.00
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.00
0.25
0.50
0.75
1.00
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.00
0.25
0.50
0.75
1.00
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.00
0.25
0.50
0.75
1.00
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(a) CKA-RBF (σ = 0.1)
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
0.8
1.0
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.1
0.2
0.3
0.4
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
0.8
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
0.8
1.0
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
0.8
1.0
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(b) CKA-RBF (σ = 0.5)
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.1
0.2
0.3
0.4
0.5
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
0.35
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(c) CKA-RBF (σ = 2.0)
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
0.35
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(d) CKA-RBF (σ = 5.0)
Figure 21. PRH alignment with varying bandwidth σ for CKA-RBF. At very small σ (a), no significant alignment remains after
calibration. Larger σ values (b–d) show significant alignment, but the scaling trend flattens after calibration.
30

## Page 31

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−1
100
p-value (log scale)
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
10−1
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
10−1
100
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
10−1
100
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
α = 0.05
(a) CKA-RBF (σ = 0.1)
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
p-value (log scale)
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
α = 0.05
(b) CKA-RBF (σ = 0.5)
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
p-value (log scale)
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
α = 0.05
(c) CKA-RBF (σ = 2.0)
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
p-value (log scale)
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
α = 0.05
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
10−2
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
α = 0.05
(d) CKA-RBF (σ = 5.0)
Figure 22. Significance of PRH alignment with varying bandwidth σ for CKA-RBF. Alignment with σ = 0.1 (a) is not significant for
multiple models where larger bandwidths have significance (b–d).
31

## Page 32

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
Topological versus metric alignment.
The contrasting behavior of mKNN and small-σ CKA-RBF reveals a fundamental
distinction in what “local alignment” means. On one hand, mKNN measures topological alignment: do the representations
agree on which points are neighbors? This captures ordinal information where the ranking of distances matters but not their
absolute values. On the other hand, small-σ CKA-RBF measures metric alignment: do the representations agree on how
close neighbors are? This captures cardinal information where exact distance values matter.
The fact that mKNN shows alignment at all k values while small-σ CKA-RBF shows no alignment reveals that cross-modal
representations agree on neighborhood identity (which points are close) but not on exact local distances (how close they
are). This finding is consistent with the observation that different training objectives and architectures induce different
distance scales in representation space while preserving the relative ordering of neighbors. The Aristotelian Representation
Hypothesis should therefore be understood as convergence to shared topological structure rather than shared metric structure.
F.10. Sensitivity to significance level α
The main text uses a significance level of α = 0.05 throughout. A natural concern is whether the conclusions of the PRH
analysis depend on this particular choice. We repeat the PRH evaluation from Section 6.3 with α ∈{0.01, 0.05, 0.10} for
representative global (CKA linear, CKA RBF) and local (mKNN with k = 10) metrics.
Figures 23 to 25 show that the conclusions are entirely invariant to the choice of α. For global metrics, calibrated scores
show no convergence trend at any significance level. For local metrics, calibrated scores retain their alignment trend across
all three α values. Stricter thresholds (α = 0.01) produce slightly lower calibrated scores, while more permissive thresholds
(α = 0.10) produce slightly higher ones, but the qualitative pattern is unchanged. This confirms that our findings are not an
artifact of a particular significance level.
32

## Page 33

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
0.35
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(a) α = 0.01
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
0.35
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(b) α = 0.05 (default)
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
0.35
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.3
0.4
0.5
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(c) α = 0.10
Figure 23. Sensitivity to α for CKA linear. Calibrated scores show no convergence trend regardless of significance level.
33

## Page 34

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
0.35
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.3
0.4
0.5
0.6
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(a) α = 0.01
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
0.35
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.3
0.4
0.5
0.6
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(b) α = 0.05 (default)
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
Alignment score
BLOOM
OpenLLaMA
LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.20
0.25
0.30
0.35
BLOOM
OpenLLaMA
LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
BLOOM
OpenLLaMA
LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.3
0.4
0.5
0.6
BLOOM
OpenLLaMA
LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.2
0.4
0.6
BLOOM
OpenLLaMA
LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(c) α = 0.10
Figure 24. Sensitivity to α for CKA RBF. The same pattern holds: no convergence trend at any significance level.
34

## Page 35

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.08
0.10
0.12
0.14
Alignment score
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.04
0.06
0.08
0.10
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.075
0.100
0.125
0.150
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.125
0.150
0.175
0.200
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.100
0.125
0.150
0.175
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(a) α = 0.01
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.08
0.10
0.12
0.14
Alignment score
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.04
0.06
0.08
0.10
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.075
0.100
0.125
0.150
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.125
0.150
0.175
0.200
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.100
0.125
0.150
0.175
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(b) α = 0.05 (default)
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.08
0.10
0.12
0.14
Alignment score
BLOOM
OpenLLaMA LLaMA
INet21K
tiny
small
base
large
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.04
0.06
0.08
0.10
BLOOM
OpenLLaMA LLaMA
MAE
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.075
0.100
0.125
0.150
BLOOM
OpenLLaMA LLaMA
DINOv2
small
base
large
giant
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.125
0.150
0.175
0.200
BLOOM
OpenLLaMA LLaMA
CLIP
base
large
huge
560m
1b1
1b7
3b
7b1
3b
7b
13b
7b
13b
30b
65b
0.100
0.125
0.150
0.175
BLOOM
OpenLLaMA LLaMA
CLIP (INet ft)
base
large
huge
calibrated
uncalibrated
(c) α = 0.10
Figure 25. Sensitivity to α for mKNN (k = 10). Local alignment and its scaling trend persist across all significance levels.
35
