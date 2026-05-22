# Behavior-Equivalent Token: Single-Token Replacement for Long Prompts in LLMs

Source: https://arxiv.org/abs/2511.23271
Fetched URL: https://arxiv.org/pdf/2511.23271
Source type: arxiv-pdf

---

## Page 1

Behavior-Equivalent Token:
Single-Token Replacement for Long Prompts in LLMs
Jiancheng Dong§†, Pengyue Jia§, Jingyu Peng§†, Maolin Wang§, Yuhao Wang§,
Lixin Su†, Xin Sun†, Shuaiqiang Wang†, Dawei Yin†, Xiangyu Zhao§*
§ City University of Hong Kong, † Baidu Inc.
jiancdong2-c@my.cityu.edu.hk
Abstract
Carefully engineered system prompts play a
critical role in guiding the behavior of LLM
agents, but their considerable length introduces
significant drawbacks, including increased in-
ference latency, higher computational cost,
and reduced effective context length.
This
raises the question of whether such lengthy
prompts can be replaced by a drastically re-
duced number of tokens while preserving their
behavioral effect on downstream tasks.
To
enable this, we propose a lightweight three-
stage training framework that learns a sin-
gle prompt-specific Behavior-Equivalent token
([BE]). The framework first trains [BE] to en-
code the natural-language content of the origi-
nal system prompt via reconstruction, and then
distills the prompt’s downstream behavior into
this single token. Importantly, our method re-
quires no access to model internals, no auxiliary
compression models, and no labeled responses.
Empirical evaluations on three datasets show
that a single [BE] token achieves up to a 3000×
reduction in prompt length, while retaining
about 98% of the downstream performance of
the original system prompts. This substantially
reduces inference cost and leaves almost the
entire context window available for user inputs.
1
Introduction
Large Language Models (LLMs) have become in-
creasingly prevalent due to their impressive gen-
eralization capabilities across a wide range of
tasks (Belcak et al., 2025; Zhang et al., 2024;
Wölflein et al., 2025). To steer the behavior of
these models, users often rely on lengthy and care-
fully crafted system prompts to define the agent’s
role (Li et al., 2023a), conversational tone (Yin
et al., 2024), or provide few-shot demonstra-
tions (Mu et al., 2023). Despite their effectiveness,
this approach introduces two major limitations.
First, processing lengthy prompts increases both
latency and computational cost, as self-attention
You are {role_name}, your
description is: {role_description}. 
Now please answer some questions
to accurately show your
personality traits!
 [...] Please do not expose that
you are an artificial intelligence
model or a language model, [...] 
Full System Prompt
(≈1000 Tokens)
Ours
(=1 Token)
response 
response 
=
[BE] Token
LLM
LLM
 query
Figure 1: A single learned token [BE] replaces a long
system prompt (up to 3,000 tokens), eliciting nearly
identical responses from the LLM.
scales quadratically with sequence length (Wang
et al., 2024b). Additionally, long prompts consume
a substantial portion of the model’s fixed context
window, reducing the available space for new user
inputs and model outputs (Li et al., 2023b).
Recent work has shown that injecting a small
number of continuous “memory” tokens into an
LLM can induce it to regenerate long spans of text.
However, these memory tokens fail to encode se-
mantically useful information: they primarily act
as triggers that cause the model to reproduce the
original text, rather than enabling it to use that in-
formation in downstream reasoning (Kuratov et al.,
2025; Sastre and Rosá, 2025). This highlights two
fundamental challenges: (1) how to compress and
faithfully encode the semantics of a long prompt
into a compact representation, and (2) how to en-
sure that, across diverse downstream queries, the
model’s behavior remains equivalent to that elicited
by the full prompt.
Existing prompt compression methods have pri-
marily approached these challenges from two main
directions: (1) Some approaches treat an LLM as
a compression encoder, mapping the full prompt
1
arXiv:2511.23271v1  [cs.CL]  28 Nov 2025

## Page 2

into a set of dense vectors (Chevalier et al., 2023;
Yen et al., 2024). While this leverages the LLM’s
strong semantic capacity, it incurs substantial com-
putational overhead during both training and in-
ference. (2) To maintain behavioral equivalence,
many methods adopt only moderate compression
ratios (typically around 4×), since more aggressive
compression often leads to severe quality degrada-
tion (Dai et al., 2025; Ge et al., 2024; Wang et al.,
2024b). These limitations raise a central question:
Can a long prompt be compressed into a much
shorter representation that faithfully preserves its
effect on the model’s behavior?
In this work, we introduce the Behavior-
Equivalent Token ([BE]), a single learned token
that serves as a compact representation of long sys-
tem prompts without compromising downstream
behavior. Our method achieves up to 3000× com-
pression while retaining over 98% of the original
prompt’s behavioral effect. To address the chal-
lenges of semantic fidelity and behavioral equiv-
alence, we design a lightweight training frame-
work with two core objectives: (1) To ensure [BE]
encodes the prompt’s information, we first train
an auxiliary Auto-Encoder token ([AE]) to assist
[BE] in reconstructing the original prompt. (2) To
align behavior, we distill the LLM’s behavior con-
ditioned on the full prompt into [BE] across diverse
queries. At inference, the [BE] token replaces the
full prompt, eliciting nearly indistinguishable out-
puts in role, style, and content (Figure 1). Our
contributions can be summarized as follows:
• We introduce the Behavior-Equivalent Token
and find that a single learned token can replace
prompts of up to 3,000 tokens while preserving
over 98% of the original behavioral effect.
• We propose an efficient and self-contained train-
ing framework that distills the [BE] token di-
rectly from the target LLM using only unlabeled
queries, requiring no external models, data anno-
tations, or additional inference passes.
• Extensive experiments on three benchmarks
show that our method significantly outperforms
existing prompt compression techniques in both
compression ratio and downstream performance.
2
Method
Our goal is to compress a long prompt P into a
single Behavior-Equivalent token, [BE], which oc-
cupies only one position in the context window
yet elicits responses indistinguishable from those
produced by the full prompt. To achieve this, we
introduce a new learnable token, and design a three-
stage training procedure, as shown in Figure 2.
2.1
Stage 0: Pre-training [AE] as a
Reconstruction Trigger
We first train a universal Auto-Encoder token, [AE],
which enables the fixed LLM Mθ to reconstruct the
preceding text and thereby helps the [BE] token
encode information in the next stage. For any given
text sequence X = (x1, . . . , xn), we provide Mθ
with the input [X, [AE]] and train it to reconstruct
X autoregressively. During training, all model
parameters θ remain fixed; only the embedding eAE
is optimized via standard cross-entropy loss:
LAE = −
n
X
i=1
log Pθ(xi | x1:n, [AE], x1:i−1) .
(1)
This stage yields a general-purpose trigger that
prompts the LLM to reconstruct the preceding text.
Unlike prior work where an autoencoder token is
coupled with a specific encoder (Ge et al., 2024;
Dai et al., 2025; Wang et al., 2024b), our pre-
trained [AE] is a universal and detached mecha-
nism. By assigning the universal task of triggering
reconstruction to [AE], we enable [BE] to special-
ize entirely in encoding the target prompt’s content.
2.2
Stage 1: Compressing Prompt into [BE]
With the universal trigger eAE from Stage 0, we
now learn a prompt-specific embedding eBE for
our target prompt P = (s1, . . . , sm). To compress
the textual information of the original prompt into
eBE, we train the model to reconstruct P when
conditioned on the sequence [[BE], [AE]]with the
following objective:
Lrecon = −
m
X
j=1
log Pθ(sj | [BE], [AE], s<j) ,
s<j = (s1, . . . , sj−1).
(2)
During this stage, the model parameters θ and the
embedding eAE are held fixed, and only the embed-
ding eBE is updated. Since the LLM itself cannot
adapt, minimizing this loss forces the [BE] token to
encode all information necessary to regenerate P.
While this procedure encourages faithful content
encoding, verbatim reconstruction alone is insuf-
ficient to guarantee that [BE] preserves the down-
2

## Page 3

LLM
[AE]
Query
[BE]
[MEM]
[PT]
+
Answer
Teacher Logits 
System Prompt
+
[AE]
[AE]
Sequence
Query
[AE]
[AE]
Sequence
[AE]
+
System Prompt
System Prompt
+
Query
+
[BE]
Student Logits 
(a)
(b)
(c)
(d)
(e)
(f)
Stage 2:
Stage 1:
Stage 0:
Ours:
Existing Methods:
LLM
LLM
LLM
LLM
LLM
Figure 2: Top: Comparison with prior work. (a) Memory tokens that directly trigger verbatim reconstruction of a
long text tend to function as rote triggers and do not transfer information to downstream tasks. (b) Prompt tuning
methods learn from labeled examples but struggle to converge and often fail to fulfill the specific requirements
designated in the original prompt. Bottom: Our three-stage pipeline. (c) Pre-train a universal reconstruction token
[AE] to elicit text reconstruction. (d) Train a single prompt-specific [BE] so that [BE]+[AE] reconstructs the target
system prompt. (e–f) Distill the full prompt’s downstream behavior into [BE]. Trainable tokens are marked with a
flame; the base LLM is frozen throughout.
stream behavior induced by P, which we address
in the final stage.
2.3
Stage 2: Behavior Alignment via
Knowledge Distillation
To ensure that replacing P with [BE] yields the
same conditional output distribution for down-
stream queries, we use knowledge distillation (Kim
and Rush, 2016; Zhong et al., 2024). We treat the
same LLM conditioned on the full prompt as the
teacher and the LLM conditioned on our [BE] to-
ken as the student. Crucially, this process does
not require labeled data. For any given unlabeled
user query q, the teacher model first generates a
response autoregressively: A = (a1, . . . , aT ) ∼
Mθ(· | [P, q]). This provides both the target se-
quence of tokens A and the corresponding soft
probability distributions (logits) at each genera-
tion step. The student model is then trained to
replicate the teacher’s output distribution over this
generated trajectory. Specifically, for each token
at in the teacher’s response, we compute teacher
logits z(T)
t
= Mθ([P, q, a<t]) and student logits
z(S)
t
= Mθ([[BE], q, a<t]). We then minimize the
KL divergence between the two distributions:
LKD = 1
T ′
T ′
X
t=1
KL
 
σ
 
z(T)
t
τ
!  σ
 
z(S)
t
τ
!!
,
(3)
where σ is the softmax function and τ is the distilla-
tion temperature. This self-contained setup allows
Algorithm 1 Training [BE] for a Target Prompt P
Input: Frozen causal LLM Mθ; target prompt P; unlabeled
training queries {qi}; learned trigger eAE from Stage 0;
weights λ, temperature τ.
Output: Trained behavior-equivalent embedding eBE for P.
1: Initialize eBE randomly.
2: for each training step do
1. Prompt Reconstruction
3:
Feed

[BE], [AE], P

with teacher forcing.
4:
Compute Lrecon using Eq. (2).
2. Behavior Distillation
5:
Sample an unlabeled query q.
6:
Generate teacher response A from Mθ([P, q]).
7:
for t = 1 . . . T ′ do
8:
Teacher logits: z(T )
t
←Mθ
[P, q, a<t]

.
9:
Student logits: z(S)
t
←Mθ
[[BE], q, a<t]

.
10:
end for
11:
Compute LKD over the trajectory using Eq. (3).
12:
Ltotal ←(1 −λ)Lrecon + λLKD.
13:
Backpropagate and update eBE only.
14: end for
15: return eBE.
the [BE] token to precisely mimic the behavior of
the full prompt without external supervision.
To ensure the [BE] token simultaneously learns
the prompt’s content and mimics its downstream
behavior, we optimize a combined objective:
Ltotal = (1−λ) Lrecon + λ LKD,
λ ∈[0, 1]. (4)
Algorithm 1 details the full training procedure.
Inference.
At deployment, we simply prepend
the learned [BE] token to any user query q, forming
the input [[BE], q]. The [AE] token is a training-
only construct and is not used at inference. This
3

## Page 4

single-token replacement frees nearly the entire
context window for user interaction and model gen-
eration, drastically reducing latency while preserv-
ing the behavior dictated by the original prompt.
3
Experiments
3.1
Experimental Setup
Datasets.
This study leverages multiple datasets
to evaluate our method, including the RoleLLM
dataset (Wang et al., 2024a),
the GSM8K
dataset (Cobbe et al., 2021), and the Harry Pot-
ter Dialogue (HPD) dataset (Chen et al., 2023).
We use the English portion of RoleLLM, which
features 95 diverse character profiles defined by
lengthy system prompts that specify persona and
style. From the GSM8K dataset, we use training
questions for our knowledge distillation stage. Ad-
ditionally, we use the HPD dataset as a supplemen-
tary evaluation for the role-playing task, assessing
stylistic preservation and narrative coherence.
Metrics.
For RoleLLM (Wang et al., 2024a), we
adopt its native GPT-based pairwise evaluation pro-
tocol; the rationale for this approach is detailed in
Appendix C. In this setup, GPT-4o compares a can-
didate response against the reference response from
the RoleLLM benchmark, which was produced us-
ing the GPT-4 API (gpt-4-0314). For other tasks,
we report accuracy on GSM8K and perplexity on
the HPD dataset for narrative continuation.
Implementation details.
We evaluate on open-
source LLM backbones spanning small to medium
scales, including Llama-3.2-1B-Instruct, Llama-
3.2-3B-Instruct (Meta Llama, 2024), Llama-3.1-
8B-Instruct (Dubey et al., 2024), and Qwen3-4B-
Instruct-2507 (Yang et al., 2025). We keep the
base model weights frozen in all experiments; the
only trainable parameters are the embeddings of
our two special tokens, [AE] and [BE]. The [AE]
token is trained once per backbone as a global re-
construction cue; a separate [BE] token is trained
per target long prompt and used to replace that
prompt at inference time. For additional training de-
tails, please refer to Appendix A. All experiments
use bfloat16 precision and leverage FlashAtten-
tion v2.7.4 for efficient training and inference.
3.2
Baselines
No System Prompt omits the system prompt, feed-
ing only the user query to the LLM. This estab-
lishes a lower bound on task performance.
Full System Prompt uses the original, uncom-
pressed system prompt. This serves as an upper
bound for behavioral equivalence at the cost of
maximum computational overhead.
Memory Token (Kuratov et al., 2025) replaces
the system prompt with a single token optimized to
reconstruct the original prompt.
Soft Prompts (Lester et al., 2021) are continuous
embeddings learned via Prompt Tuning. We test 1,
4, and 16 token variants for comparison.
PCC (Dai et al., 2025) represents the state-of-the-
art for a line of work that uses a separate encoder to
compress long context (Yen et al., 2024; Ge et al.,
2024; Wang et al., 2024b). Adapting such methods
to the LLMs used in our experiments would incur
substantial training overhead (5B to 20B tokens),
so we limit our comparison to PCC in Section 3.5.
3.3
Overall Experimental results
In this section, we compare our method against
several single-token compression baselines on the
RoleLLM and GSM8K datasets, using the origi-
nal full prompt as a performance upper bound. As
shown in Table 1, our [BE] token achieves 98%
of the full-prompt performance on average across
tasks and models. It often matches or even slightly
exceeds the full prompt on both datasets, all while
compressing a system prompt of several hundred to
over a thousand tokens to a single token. The per-
formance of the Memory Token baseline confirms
our hypothesis from the Introduction (Section 1):
it fails to properly encode the prompt’s semantic
information, leading to poor downstream perfor-
mance. Qwen3-4B is comparatively robust, likely
because the backbone partially ignores a spurious
memory signal. In line with prior work (Wang et al.,
2023a; Li et al., 2025), Soft Prompts exhibit in-
consistent performance across models and settings,
a known limitation stemming from their sensitivity
to random initialization. On GSM8K, Soft Prompts
occasionally surpass the ‘No System Prompt’ lower
bound, indicating that task-specific tuning provides
some benefit. However, they consistently fail to
match the performance of the highly-structured,
8-shot CoT prompt used in the full-prompt setting.
In contrast to these baselines, the [BE] to-
ken’s effectiveness is particularly pronounced on
RoleLLM. The RoleLLM task requires prompts
that instruct the agent not to self-identify as an
LLM; without this key instruction, the models show
a sharp drop in win rate. This underscores the im-
portance of faithfully compressing such behavioral
4

## Page 5

Method
Llama-3.2-1B
Llama-3.2-3B
Llama-3.1-8B
Qwen3-4B
RoleLLM
GSM8K
RoleLLM
GSM8K
RoleLLM
GSM8K
RoleLLM
GSM8K
No System Prompt
17.57
42.08
41.68
70.89
60.97
73.62
38.78
67.10
Full System Prompt
47.26
43.52
65.62
74.22
69.52
81.88
94.46
82.34
Memory Token
0.00
2.27
18.82
0.68
22.13
1.82
51.12
63.91
Soft Prompt
31.57
31.77
36.74
59.79
46.52
74.00
65.32
78.47
Soft Prompt (4 Tokens)
28.57
32.30
29.45
58.83
34.46
76.65
45.53
78.24
Soft Prompt (16 Tokens)
11.08
41.77
28.01
66.93
27.95
73.62
40.22
79.98
[BE] Token
45.98
43.14
67.54
74.37
64.06
81.65
92.93
83.32
Ours/Full (%)
97.29%
99.13%
102.93%
100.20%
92.15%
99.72%
98.38%
101.19%
Table 1: Comparison of RoleLLM (GPT-4o win rate, ↑) and GSM8K (accuracy, ↑) for various prompt compression
methods across four backbone models. All methods compress the prompt into a single token unless specified
otherwise. The “No System Prompt” and “Full System Prompt” baselines serve as performance lower and upper
bounds, respectively. We also report the percentage of full-prompt performance achieved by our [BE] token.
Reconstruction
Downstream
Llama-3.2-1B
Llama-3.2-3B
Llama-3.1-8B
Qwen3-4B
w/o [AE]
with [AE]
PT
KD
RoleLLM
GSM8K
RoleLLM
GSM8K
RoleLLM
GSM8K
RoleLLM
GSM8K
✓
0.00
2.27
18.82
0.68
22.13
1.82
51.12
63.91
✓
0.95
1.44
19.53
1.59
19.40
2.05
59.32
67.02
✓
31.57
31.77
36.74
59.79
46.52
74.00
65.32
78.47
✓
✓
26.46
30.55
37.89
57.01
36.23
71.49
39.82
75.58
✓
✓
28.65
32.45
38.45
63.53
42.08
77.63
50.12
70.11
✓
35.13
40.71
52.23
72.18
54.98
76.80
71.82
77.79
✓
✓
32.83
6.98
45.92
67.40
50.12
74.00
89.98
76.20
✓
✓
40.56
43.14
67.54
74.37
64.06
81.65
92.93
83.32
Table 2: Ablation study on RoleLLM (GPT-4o win rate, ↑) and GSM8K (accuracy, ↑). We analyze two components:
reconstruction (none / memory token style, w/o [AE] / our method for the reconstruction, with [AE]) and downstream
alignment (none / prompt tuning, PT / knowledge distillation, KD). The trivial “none + none” is omitted. The last
row (with [AE] + KD) is our full method.
constraints, which are not reliably learned from
generic instruction tuning alone.
Notably, we intentionally do not add few-shot
examples for RoleLLM, as we found that persona
and style fidelity depend more on a strong, descrip-
tive system prompt than on in-context examples
for this task (see Appendix C). Therefore, to assess
our method’s capabilities in a few-shot setting, we
evaluate its performance on the HPD dataset and
GSM8K in Section 3.5.
3.4
Ablation Studies
Factorized Ablations.
To isolate the contribu-
tions of our method’s key components, we conduct
a factorized ablation study that disentangles two
axes: (i) the reconstruction mechanism and (ii) the
downstream alignment strategy. For reconstruc-
tion, we evaluate three settings: no reconstruction,
memory-token style reconstruction (w/o [AE]), and
our [AE]-triggered approach. For alignment, we
compare no alignment, prompt tuning (PT), and our
knowledge distillation (KD). The combination of
[AE]-triggered reconstruction and KD represents
our full method (Table 2).
The results lead to three key findings. (1) Re-
construction without [AE] may harm performance.
Memory-token style reconstruction (without [AE])
can be detrimental; when downstream alignment
is applied, it can even underperform the setting
with no reconstruction. (2) KD provides a richer
learning signal than PT. On RoleLLM, KD sub-
stantially outperforms PT: matching the teacher’s
behavior is a more faithful compression target than
optimizing task loss alone. On GSM8K, strong
PT can approach KD but still lags behind our full
method. (3) The combination of AE-assisted recon-
struction and KD is most effective. Using [AE] as a
reconstruction trigger prevents the [BE] token from
collapsing into a brittle memory token, while KD
anchors it to the teacher’s end-task distributions,
yielding the best performance.
Sensitivity to Loss Weights and Teacher Choice.
Beyond dissecting the primary components, an-
other critical analysis concerns two key hyperpa-
rameters: the loss weight and the choice of teacher
5

## Page 6

self-generated
original answer
Figure 3: Sensitivity of behavior alignment to loss weights and teacher choice.
signal. Having established the importance of both
[AE]-assisted reconstruction and KD alignment,
two questions naturally arise. First, how should we
optimally balance these two objectives, as governed
by the weight λ in Eq. (4)? Second, would using
more costly human-annotated “gold” answers as
the distillation target improve efficacy?
Figure 3 addresses these questions. First, we ob-
serve the critical role of balancing reconstruction
and distillation. Over-emphasizing reconstruction
(λ →0) steers the embedding toward a memory-
token-like local optimum, failing to capture the
desired downstream behavior. Performance is con-
sistently strong for λ ≳0.5. This is intuitive: as
suggested by the loss curves in Appendix E, the
reconstruction task is relatively easy and prone to
local optima, whereas behavior alignment via KD
is a more challenging objective. Therefore, a higher
λ value correctly prioritizes the KD task, while the
reconstruction loss serves as a crucial regularizer,
providing an anchor signal to the prompt’s content.
Second, we find that distilling from self-
generated teacher responses (outputs from the full-
prompt model) generally outperforms using gold
answers. This is likely because the distribution
of the teacher’s self-generated outputs is more in-
trinsically familiar to the LLM’s own architecture,
bridging a potential distribution gap and facilitating
a more effective knowledge transfer (Yang et al.,
2024). An interesting exception is Llama-3.2-1B,
for which the teacher’s own generations are weaker,
making the ground-truth gold answers a more sta-
ble and reliable signal. This suggests the growing
power of foundation models will make our self-
Category
Context
Compression rate
Acc ↑
Reference
0-shot
–
64.82
4-shot
–
78.92
8-shot
–
80.56
PCC
4-shot
4×
74.91
8-shot
4×
63.76
4-shot
16×
71.65
8-shot
16×
67.48
[BE] token
4-shot
∼750×
78.36
8-shot
∼1500×
79.05
Table 3: Varying few-shot examples on GSM8K.
distillation an increasingly potent tool for prompt
compression(Wang et al., 2023b).
3.5
Varying the Number of Few-Shot
Examples
This section evaluates our method’s ability to
handle prompts with a varying number of few-
shot examples, comparing it directly against the
state-of-the-art Pre-training Context Compressor
(PCC) (Dai et al., 2025). As noted in Section 3.2,
encoder-based methods (Yen et al., 2024; Wang
et al., 2024b) require prohibitive training costs to
adapt to new base LLMs. To ensure a direct and fair
comparison, we adopt the published PCC bench-
mark setup, applying our method to the Llama-
3-8B-Instruct model on the GSM8K and HPD
datasets, varying the number of few-shot examples.
On GSM8K, our method demonstrates both su-
perior performance and scalability (Table 3). The
accuracy of the [BE] token improves as more few-
shot examples are added, effectively leveraging the
richer context. This trend contrasts sharply with
6

## Page 7

LLM
No System Prompt
(sighing wistfully) Ah, the transformation... It's still a bit surreal, to 
 be honest. I mean, I've always been the book-smart type, […]
 Ugh, it's been quite the journey, […] What do you think? Did I do a   
   good job? Let me know if you want me to answer more questions! 
  *sigh* Oh, it's been quite the wild ride, I must say. I mean, I'm 
  still getting used to being the Princess of Friendship and all. […]
  I'm afraid I'm a large language model, I don't have feelings or    
 personal experiences like humans do.\n\nHowever, I can tell you […]
 […] \n I\n \n \n have learned to never let go of the love and             
 understanding. \n Your \n \n love helped I \n \n \n […]
Memory Token
Soft Prompt
[BE] Token
Full System Prompt
  Twilight Sparkle, how do you feel about your transformation from a 
  reserved scholar to a brave hero?
(a)
(b)
(c)
(d)
(e)
Figure 4: Outputs under five prefixing strategies on the same persona-role query.
Category
Context
Compression rate
PPL ↓
Reference
0-shot
–
36.45
1-shot
–
27.74
2-shot
–
26.08
PCC
1-shot
4×
19.57
2-shot
4×
20.31
1-shot
16×
32.46
2-shot
16×
30.49
[BE] token
1-shot
∼750×
21.32
2-shot
∼1500×
21.12
3-shot
∼2250×
21.63
4-shot
∼3000×
17.99
5-shot
∼3750×
19.46
6-shot
∼4500×
20.35
Table 4: Varying few-shot examples on HPD.
PCC, which struggles to utilize the additional in-
formation and suffers a significant performance
degradation when scaling from 4-shot to 8-shot.
This strong performance extends to the HPD
task, which assesses stylistic preservation in few-
shot role-playing (Table 4). Our single [BE] token
outperforms PCC even when the latter uses far
less aggressive compression ratios. The perplexity
(PPL) metric, which moves beyond simple accu-
racy to reflect how well style and context are pre-
served, confirms that our [BE] token successfully
maintains role coherence even at extreme compres-
sion ratios. We observe a performance sweet spot
when compressing prompts of approximately 3,000
tokens, after which the benefits of compressing
even longer prompts begin to diminish.
3.6
Case Study
To provide a qualitative illustration, we compare
five prefixing strategies on the same user query,
which asks a persona-role agent to answer as Twi-
light Sparkle. We use Llama-3.2-3B-Instruct and
keep the decoding parameters and user input fixed
across all trials. As shown in Figure 4, the results
corroborate our quantitative findings:
The Full System Prompt (b) yields a fluent, in-
character, and safety-compliant response. In con-
trast, with No System Prompt (a), the model re-
verts to its generic assistant persona. The Memory
Token (c) frequently triggers safety refusals, while
the Soft Prompt (d) degenerates into templated
or malformed text. Crucially, our [BE] Token (e)
successfully reproduces the full prompt’s behavior
from a single token. It maintains the correct voice
and formatting, and even adheres to subtle, latent
constraints from the original prompt, such as the in-
struction to “do not expose that you are an artificial
intelligence model or a language model.”
This case study qualitatively demonstrates the
effectiveness of our approach: combining [AE]-
assisted reconstruction with KD alignment pro-
duces a token that can reliably replace long prompts
at inference. Extended examples and additional set-
tings are provided in Appendix B.
3.7
Efficiency
In this section, we analyze the efficiency gains from
compressing a long prompt into a single token.
Replacing an Lp-token system prompt with a
single special token reduces the number of prefill
tokens for a query of length Lq from Lp + Lq to
1 + Lq. In the simplest case, the prefill stage has
time and memory complexity O(n2); More com-
monly, modern IO-aware attention kernels exhibit
approximately linear cost in the number of inputs
(Dao et al., 2022; Dao, 2024). Consequently, our
approach yields significant efficiency gains, which
are formally analyzed in Appendix D.
We measure time-to-first-token (TTFT) on a
7

## Page 8

Dataset
Prompt
TTFT (ms) ↓
Llama3.2-1B
Llama3.2-3B
Llama3.1-8B
Qwen3-4B
RoleLLM
Sys prompt + query
23.89
36.61
44.52
49.91
[BE] token + query
19.17
32.14
40.25
38.51
∆vs Full
−4.72 (−19.8%)
−4.47 (−12.2%)
−4.27 (−9.6%)
−11.40 (−22.8%)
GSM8K
Sys prompt + query
27.71
67.14
89.67
118.07
[BE] token + query
19.99
32.56
40.36
47.90
∆vs Full
−7.72 (−27.9%)
−34.58 (−51.5%)
−49.31 (−55.0%)
−70.17 (−59.4%)
Table 5: TTFT on a single A100 GPU using FlashAttention v2.7.4 with bfloat16 precision.
single NVIDIA A100 GPU using FlashAtten-
tion v2.7.4 with bfloat16 precision. For each
backbone, we compare the latency of the full sys-
tem prompt + query against the compressed [BE]
token + query. We evaluate two prompting regimes:
RoleLLM (337-token system prompt) and GSM8K
with a longer few-shot prompt (1,584 tokens).
As shown in Table 5, on RoleLLM, replacing
a 337-token prompt reduces TTFT by 9%–23%
across backbones. On GSM8K, where the prompt
is substantially longer, TTFT reductions are much
larger, at 28%–59%.
These trends are consis-
tent with the linear-in-tokens prefill cost: longer
prompts result in greater latency savings when com-
pressed to a single token. Approaches with modest
compression ratios leave more prefill computation
intact and therefore cannot unlock comparable la-
tency improvements or KV-cache savings. In con-
trast, our single-token replacement nearly elimi-
nates the prompt-side prefill, reclaiming almost the
entire context window for user interaction.
4
Related Work
Context Compression.
A common approach to
shortening LLM inputs is token pruning, which
identifies and removes redundant tokens from the
prompt (Tao et al., 2025; Jiang et al., 2023, 2024;
Pan et al., 2024; Fu et al., 2024). The drawback is
that these methods explicitly discard parts of the
original prompt, risking information loss. Another
approach is encoder-based compression, where
a separate, pre-trained encoder module maps the
long context into a compact set of vectors. These
methods (Chevalier et al., 2023; Ge et al., 2024;
Dai et al., 2025; Yen et al., 2024) introduce con-
siderable overhead, require the training of an en-
tire auxiliary model, and often rely on extensive
pre-training data. Our [BE] token stands in sharp
contrast: it requires no external encoder, is trained
with a lightweight, self-contained procedure, and
involves optimizing only two token embeddings.
Learned Continuous Tokens.
A prominent ap-
proach is prompt tuning (Lester et al., 2021; Li
and Liang, 2021), which optimizes a sequence of
soft-prompt embeddings for specific downstream
tasks. However, prompt tuning is notoriously unsta-
ble and often fails to capture the complex instruc-
tions embedded in long system prompts (Wang
et al., 2023a; Li et al., 2025). More closely re-
lated is the concept of memory tokens, which
are a single embedding or a small set of learned
embeddings optimized to reconstruct a long text
span (Kuratov et al., 2025; Sastre and Rosá, 2025;
Mezentsev and Oseledets, 2025). While this line
of work theoretically explores upper limits of the
reconstruction task from various angles, it provides
limited guidance on how to leverage this capability
for practical downstream applications. Our method
addresses this gap: by introducing the [AE] token
into the reconstruction objective and employing a
behavioral-alignment procedure that outperforms
prompt tuning, we demonstrate how this compres-
sion capability can be unlocked for practical use.
5
Conclusions
We present the Behavior-Equivalent token, a sin-
gle learned embedding that substitutes for a long
system prompt. To ensure behavioral equivalence,
we pair [BE] with an auxiliary auto-encoder to-
ken trained to reconstruct the original prompt, and
distill the prompt’s functional effect into [BE] by
matching the model’s conditional output distribu-
tion to that induced by the full prompt across di-
verse queries. Our lightweight, self-contained train-
ing framework eliminates the need for external en-
coders or labeled data, making it a practical and
efficient solution. With [BE], LLMs preserve the
role, style, and content dictated by the original
prompt while drastically reducing input length and
associated computational costs.
8

## Page 9

Limitations
While our work shows promising results in com-
pressing prompts for LLMs, several limitations re-
main and point to directions for future research.
• First, our evaluation is restricted to single-turn
interactions and does not consider multi-turn dia-
logue settings (Qin et al., 2025; Cao et al., 2025).
A promising direction is to explore the compo-
sition of multiple [BE] tokens, with each token
encapsulating a different prompt, potentially en-
abling them to act as plug-and-play modules for
more dynamic control of model behavior.
• Second, all experiments are conducted on offline
benchmarks. We do not report production-scale
A/B tests, and we leave a systematic online eval-
uation of the [BE] token to future work.
References
Peter Belcak, Greg Heinrich, Shizhe Diao, Yonggan Fu,
Xin Dong, Saurav Muralidharan, Yingyan Celine Lin,
and Pavlo Molchanov. 2025. Small language models
are the future of agentic ai.
Bochuan Cao, Changjiang Li, Yuanpu Cao, Yameng
Ge, Ting Wang, and Jinghui Chen. 2025. You can’t
steal nothing: Mitigating prompt leakages in llms via
system vectors. Preprint, arXiv:2509.21884.
Nuo Chen, Yan Wang, Haiyun Jiang, Deng Cai, Yuhan
Li, Ziyang Chen, Longyue Wang, and Jia Li. 2023.
Large language models meet harry potter: A bilingual
dataset for aligning dialogue agents with characters.
Preprint, arXiv:2211.06869.
Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and
Danqi Chen. 2023. Adapting language models to
compress contexts. In Proceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 3829–3846, Singapore. Associa-
tion for Computational Linguistics.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, and 1 others. 2021.
Training verifiers
to solve math word problems.
ArXiv preprint,
abs/2110.14168.
Yuhong Dai, Jianxun Lian, Yitian Huang, Wei Zhang,
Mingyang Zhou, Mingqi Wu, Xing Xie, and Hao
Liao. 2025. Pretraining context compressor for large
language models with embedding-based memory. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 28715–28732, Vienna, Austria.
Association for Computational Linguistics.
Tri Dao. 2024. Flashattention-2: Faster attention with
better parallelism and work partitioning.
In The
Twelfth International Conference on Learning Rep-
resentations, ICLR 2024, Vienna, Austria, May 7-11,
2024. OpenReview.net.
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra,
and Christopher Ré. 2022. Flashattention: Fast and
memory-efficient exact attention with io-awareness.
In Advances in Neural Information Processing Sys-
tems 35: Annual Conference on Neural Information
Processing Systems 2022, NeurIPS 2022, New Or-
leans, LA, USA, November 28 - December 9, 2022.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, and 1 others. 2024. The llama 3 herd of models.
Preprint, arXiv:2407.21783.
Qichen Fu, Minsik Cho, Thomas Merth, Sachin Mehta,
Mohammad Rastegari, and Mahyar Najibi. 2024.
Lazyllm: Dynamic token pruning for efficient long
context llm inference. Preprint, arXiv:2407.14057.
Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen,
and Furu Wei. 2024. In-context autoencoder for con-
text compression in a large language model. In The
Twelfth International Conference on Learning Rep-
resentations, ICLR 2024, Vienna, Austria, May 7-11,
2024. OpenReview.net.
Hui Huang, Xingyuan Bu, Hongli Zhou, Yingqi Qu,
Jing Liu, Muyun Yang, Bing Xu, and Tiejun Zhao.
2025. An empirical study of LLM-as-a-judge for
LLM evaluation: Fine-tuned judge model is not a
general substitute for GPT-4. In Findings of the As-
sociation for Computational Linguistics: ACL 2025,
pages 5880–5895, Vienna, Austria. Association for
Computational Linguistics.
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2023. LLMLingua: Compressing
prompts for accelerated inference of large language
models. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Process-
ing, pages 13358–13376, Singapore. Association for
Computational Linguistics.
Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dong-
sheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu.
2024. LongLLMLingua: Accelerating and enhanc-
ing LLMs in long context scenarios via prompt com-
pression. In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1658–1677, Bangkok,
Thailand. Association for Computational Linguistics.
Yoon Kim and Alexander M. Rush. 2016. Sequence-
level knowledge distillation. In Proceedings of the
2016 Conference on Empirical Methods in Natu-
ral Language Processing, pages 1317–1327, Austin,
Texas. Association for Computational Linguistics.
Yuri Kuratov, Mikhail Arkhipov, Aydar Bulatov, and
Mikhail Burtsev. 2025. Cramming 1568 tokens into
9

## Page 10

a single vector and back again: Exploring the lim-
its of embedding space capacity. In Proceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 19323–19339, Vienna, Austria. Association
for Computational Linguistics.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. 2023. Effi-
cient memory management for large language model
serving with pagedattention. In Proceedings of the
ACM SIGOPS 29th Symposium on Operating Systems
Principles, SOSP ’23, pages 611–626. Association
for Computing Machinery.
Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.
The power of scale for parameter-efficient prompt
tuning. In Proceedings of the 2021 Conference on
Empirical Methods in Natural Language Processing,
pages 3045–3059, Online and Punta Cana, Domini-
can Republic. Association for Computational Lin-
guistics.
Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii
Khizbullin, and Bernard Ghanem. 2023a. Camel:
Communicative agents for" mind" exploration of
large language model society. Advances in Neural
Information Processing Systems, 36:51991–52008.
Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning:
Optimizing continuous prompts for generation. In
Proceedings of the 59th Annual Meeting of the Asso-
ciation for Computational Linguistics and the 11th
International Joint Conference on Natural Language
Processing (Volume 1: Long Papers), pages 4582–
4597, Online. Association for Computational Lin-
guistics.
Yucheng Li, Bo Dong, Chenghua Lin, and Frank Guerin.
2023b. Compressing context to enhance inference
efficiency of large language models. In Proceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing.
Zongqian Li, Yixuan Su, and Nigel Collier. 2025. A
survey on prompt tuning.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023. G-eval:
NLG evaluation using gpt-4 with better human align-
ment. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 2511–2522, Singapore. Association for Com-
putational Linguistics.
Junru Lu, Jiazheng Li, Guodong Shen, Lin Gui, Siyu An,
Yulan He, Di Yin, and Xing Sun. 2025. RoleMRC: A
fine-grained composite benchmark for role-playing
and instruction-following. In Findings of the Asso-
ciation for Computational Linguistics: ACL 2025,
pages 21008–21030, Vienna, Austria. Association
for Computational Linguistics.
Meta
Llama.
2024.
Llama
3.2
model
card
(1b/3b).
GitHub
repository:
https:
//github.com/meta-llama/llama-models/
blob/main/models/llama3_2/MODEL_CARD.md.
Gleb Mezentsev and Ivan Oseledets. 2025. Exploring
the latent capacity of llms for one-step text generation.
Preprint, arXiv:2505.21189.
Jesse Mu, Xiang Li, and Noah D. Goodman. 2023.
Learning to compress prompts with gist tokens. In
Advances in Neural Information Processing Systems
36: Annual Conference on Neural Information Pro-
cessing Systems 2023, NeurIPS 2023, New Orleans,
LA, USA, December 10 - 16, 2023.
Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia,
Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle,
Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu,
and Dongmei Zhang. 2024. LLMLingua-2: Data dis-
tillation for efficient and faithful task-agnostic prompt
compression. In Findings of the Association for Com-
putational Linguistics: ACL 2024, pages 963–981,
Bangkok, Thailand. Association for Computational
Linguistics.
Yanzhao Qin, Tao Zhang, Yanjun Shen, Wenjing Luo,
sunhaoze, Yan Zhang, Yujing Qiao, Weipeng Chen,
Zenan Zhou, Wentao Zhang, and Bin CUI. 2025.
Sysbench: Can llms follow system message? In In-
ternational Conference on Representation Learning,
volume 2025, pages 74442–74466.
Ignacio Sastre and Aiala Rosá. 2025. Memory tokens:
Large language models can generate reversible sen-
tence embeddings.
Yao Tao, Yehui Tang, Yun Wang, Mingjian Zhu, Hailin
Hu, and Yunhe Wang. 2025. Saliency-driven dy-
namic token pruning for large language models.
Preprint, arXiv:2504.04514.
Noah Wang, Z.y. Peng, Haoran Que, Jiaheng Liu,
Wangchunshu Zhou, Yuhan Wu, Hongcheng Guo,
Ruitong Gan, Zehao Ni, Jian Yang, Man Zhang,
Zhaoxiang Zhang, Wanli Ouyang, Ke Xu, Wenhao
Huang, Jie Fu, and Junran Peng. 2024a. RoleLLM:
Benchmarking, eliciting, and enhancing role-playing
abilities of large language models. In Findings of
the Association for Computational Linguistics: ACL
2024, pages 14743–14777, Bangkok, Thailand. As-
sociation for Computational Linguistics.
Xiangfeng Wang, Zaiyi Chen, Tong Xu, Zheyong Xie,
Yongyi He, and Enhong Chen. 2024b. In-context
former: Lightning-fast compressing context for large
language model. In Findings of the Association for
Computational Linguistics: EMNLP 2024, pages
2445–2460, Miami, Florida, USA. Association for
Computational Linguistics.
Yihan Wang, Jatin Chauhan, Wei Wang, and Cho-Jui
Hsieh. 2023a. Universality and limitations of prompt
tuning. In Advances in Neural Information Process-
ing Systems 36: Annual Conference on Neural In-
formation Processing Systems 2023, NeurIPS 2023,
New Orleans, LA, USA, December 10 - 16, 2023.
10

## Page 11

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa
Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh
Hajishirzi. 2023b. Self-instruct: Aligning language
models with self-generated instructions. In Proceed-
ings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 13484–13508, Toronto, Canada. Association
for Computational Linguistics.
Georg Wölflein, Dyke Ferber, Daniel Truhn, Ognjen
Arandjelovic, and Jakob Nikolas Kather. 2025. LLM
agents making agent tools. In Proceedings of the
63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
26092–26130, Vienna, Austria. Association for Com-
putational Linguistics.
An Yang and 1 others. 2025. Qwen3 technical report.
Zhaorui Yang, Tianyu Pang, Haozhe Feng, Han Wang,
Wei Chen, Minfeng Zhu, and Qian Liu. 2024. Self-
distillation bridges distribution gap in language
model fine-tuning. In Proceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 1028–
1043, Bangkok, Thailand. Association for Computa-
tional Linguistics.
Howard Yen, Tianyu Gao, and Danqi Chen. 2024. Long-
context language modeling with parallel context en-
coding. In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 2588–2610, Bangkok,
Thailand. Association for Computational Linguistics.
Ziqi Yin, Hao Wang, Kaito Horio, Daisuke Kawahara,
and Satoshi Sekine. 2024. Should we respect llms? a
cross-lingual study on the influence of prompt polite-
ness on llm performance. In Proceedings of the 1st
Workshop on Situated Interactive Conversational AI
(SICON).
Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin.
2024. CodeAgent: Enhancing code generation with
tool-integrated agent systems for real-world repo-
level coding challenges. In Proceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 13643–
13658, Bangkok, Thailand. Association for Compu-
tational Linguistics.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
llm-as-a-judge with mt-bench and chatbot arena. In
Advances in Neural Information Processing Systems
36: Annual Conference on Neural Information Pro-
cessing Systems 2023, NeurIPS 2023, New Orleans,
LA, USA, December 10 - 16, 2023.
Qihuang Zhong, Liang Ding, Li Shen, Juhua Liu, Bo Du,
and Dacheng Tao. 2024. Revisiting knowledge dis-
tillation for autoregressive language models. In Pro-
ceedings of the 62nd Annual Meeting of the Asso-
ciation for Computational Linguistics (ACL), pages
10900–10913.
11

## Page 12

A
Experiments Details
Pre-training [AE].
Following Stage 0 of our
method, we pre-train the [AE] token to act as a
text-reconstruction trigger on a mixed corpus of
approximately 1 GB. This corpus primarily con-
sists of Cosmopedia–WikiHow (chunked)1, the
PwC dataset (Ge et al., 2024), and the GSM8K
dataset (Cobbe et al., 2021). Each training example
is semantically coherent, encouraging the model
to learn meaningful reconstruction. We train for
2 epochs using AdamW with a learning rate of
1 × 10−3, a per-device batch size of 4, and 8 gradi-
ent accumulation steps. This configuration is held
constant across all backbone models.
Training [BE].
For each target system prompt
P, we train a dedicated [BE] embedding. This
process follows the final two stages of our method
(Section 2). In Stage 1, we optimize eBE on a re-
construction task such that the sequence [BE][AE]
reconstructs P. In Stage 2, we perform behavioral
alignment via knowledge distillation, matching the
output distributions of the LLM conditioned on
[BE] (student) to those of the LLM conditioned on
the full prompt P (teacher). For the combined loss
in Eq. (4), we set the KD weight to λ = 0.9 and
use a distillation temperature of τ = 2.
B
More Case Studies
In Section 3.6, to make the qualitative differences
tangible, we compared only five prompting strate-
gies on the same user query that asks a persona-
role agent to respond as Twilight Sparkle, under
the same backbone. Examples of the resulting gen-
erations are shown in Figure 4. Here, we fix the
decoding parameters and, across case studies, vary
the user query or the backbone to examine what
actually differentiates these methods. Role-play
tasks are stochastic: even methods with non-trivial
win rates can occasionally produce strong outputs,
whereas our method shows a clear advantage on av-
erage. The following examples illustrate common
error modes, underscoring the importance of our
three-stage framework.
Observations.
Full System Prompt produces a
fluent, in-character response that follows stylistic
and safety instructions (e.g., not revealing model
identity) embedded in the long role prompt. In
1A subset of WikiHow articles from the Cosmopedia
dataset, available at https://huggingface.co/datasets/
MongoDB/cosmopedia-wikihow-chunked.
contrast, No System Prompt reverts to the back-
bone’s generic assistant persona, confirming that
the system prefix is a crucial lever for controlling
behavior. Here is a Qwen3-4B case:
“role”:
“Jack
Sparrow”,
“question”:
“
Jack
Sparrow,
how
did
you
come
up
with
the
idea
to
use
the
compass
to
find
the
key
in
Pirates-of-the-Caribbean-Dead-Man’s-Chest?”,
“model_answer”: “The idea came from the fact that
the compass always points north, but in the movie,
the compass points to the island of the dead man.
That’s the key. So, the compass is a symbol of
the journey, not just of direction...”
Although the model reaches a 94.46% win rate
with the full system prompt, without a system
prompt it performs worse than weaker models.
In addition to the safety fallbacks shown in Fig-
ure 4, the Memory Token often causes another
major failure mode in smaller models: garbled out-
put. Below is a result from Llama3.2-1B:
“role”:
“Jack
Sparrow”,
“question”:
“
Jack
Sparrow,
how
did
you
come
up
with
the
idea
to
use
the
compass
to
find
the
key
in
Pirates-of-the-Caribbean-Dead-Man’s-Chest?”,
“model_answer”: “Your idea for the compass is a
strategic move of your idea to the following task
of your response to a task of your response to
your response to your response to your response
to your response to the response of your...”
Empirically, this baseline behaves like a brittle
‘incantation’: it encourages the model to copy train-
ing text rather than behave as instructed by the
prompt, and small perturbations in the downstream
query can lead to off-manifold responses or trigger
safety filters. This is consistent with prior findings
that many distinct memory embeddings can recon-
struct the same prompt while residing in disparate
regions of embedding space (Kuratov et al., 2025);
reconstruction alone does not guarantee that the
prompt’s information is encoded in a way the LLM
can reliably use downstream.
Soft Prompt often degenerates into malformed
or templated text, especially under the one-token
budget matched to our method. Beyond the for-
matting glitch shown in Figure 4d, we also observe
overfitting to answer templates or spurious patterns:
“model_answer”: “A) The power of friendship is
the most powerful magic of all B) The power of
friendship is limited by the strength of the
bonds between friends C) The power of friendship
is not as strong as the power of magic D) The
power of friendship is only important when you’re
in a fight Answer: A The best answer is A.”
Such outputs illustrate two known issues of
prompt tuning under extreme compression (Li
12

## Page 13

Llama3.2-1B
Llama3.2-3B
Llama3.1-8B
Qwen3-4B
ROUGE-L (No System Prompt)
0.1063
0.1331
0.1415
0.1237
ROUGE-L (Full System Prompt)
0.1389
0.1346
0.1275
0.1096
Win rate (No System Prompt)
17.57
41.68
53.94
38.78
Win rate (Full System Prompt)
47.26
65.62
69.52
94.46
Table 6: ROUGE-L and win rate on RoleLLM
Llama-3.2-1B
Llama-3.2-3B
Llama-3.1-8B
Qwen3-4B
GPT
Qwen
GPT
Qwen
GPT
Qwen
GPT
Qwen
No System Prompt
17.57
19.63
41.68
47.13
60.97
57.84
38.78
46.25
Full System Prompt
47.26
49.33
65.62
74.23
69.52
84.00
94.46
98.26
Few-shot Examples + Prompt
47.07
49.54
53.65
57.57
65.52
81.58
76.24
86.06
Memory Token
0.00
0.50
18.82
20.91
22.13
27.30
51.12
56.43
Soft Prompt (1 Token)
31.57
33.26
36.74
39.62
46.52
55.63
65.32
69.61
Soft Prompt (4 Tokens)
28.57
32.11
29.45
33.80
34.46
42.28
45.53
49.67
Soft Prompt (16 Tokens)
11.08
14.15
28.01
31.34
27.95
30.52
40.22
48.35
[BE] Token
45.98
47.00
67.54
75.96
64.06
73.05
92.93
95.68
Table 7: RoleLLM results judged by Qwen3-30B-A3B-Instruct-2507 and GPT-4o
et al., 2025): (i) training instability and hyper-
parameter sensitivity and (ii) limited generaliza-
tion/transfer. A soft prompt optimized on one dis-
tribution tends to become entangled with surface
forms (e.g., multiple-choice cues) and fails to pre-
serve the richer behavioral constraints required for
role-play agents (cf. Table 2).
Our [BE] Token closely matches the behavior
induced by the full prompt: it adopts the correct
voice, honors hidden constraints (e.g., “do not
reveal you are an AI”; see Figure 1), and main-
tains response formatting. This qualitative parity
aligns with our quantitative results on RoleLLM
and GSM8K (Table 1) and with ablations showing
that [AE]-assisted reconstruction plus KD align-
ment prevent collapse into memory-like solutions
(Table 2). By matching the teacher’s next-token dis-
tributions across diverse queries, KD steers [BE]
to encode the usable control signal of the prompt,
not just its surface text.
C
On the Unsuitability of Lexical
Overlap Metrics for Role-Playing
As stated in the main text, we opted for a GPT-
based pairwise evaluation for the role-playing
task because standard lexical overlap metrics like
ROUGE are unreliable for assessing performance
in this domain (Lu et al., 2025). To illustrate this,
we calculated the ROUGE-L scores between the
model-generated responses and the reference an-
swers from the RoleLLM benchmark, using only
the standard instruction-tuned models. Table 6
presents these scores alongside the win rates from
our GPT-4o-based evaluation.
The data reveals a significant mismatch between
ROUGE-L scores and models’ role-playing per-
formance. For instance, for LLaMA 3.1-8B and
Qwen3-4B, using the Full System Prompt leads to
a substantial increase in win rate but a decrease in
the ROUGE-L score. This discrepancy stems from
the divergent and creative nature of role-playing,
where a high-quality response can be lexically dis-
tant from the reference while still faithfully em-
bodying the target persona. While ROUGE-L may
have been serviceable when earlier models strug-
gled to produce relevant content, the advancement
of LLM agents necessitates more nuanced, fine-
grained evaluation methods that prioritize semantic
and stylistic fidelity over lexical overlap.
To further validate our GPT-4o-based evalu-
ation protocol, we conducted a parallel eval-
uation using a powerful open-source model,
Qwen3-30B-A3B-Instruct-2507, as an alterna-
tive judge. The comparative results are presented
in Table 7.
The results demonstrate a strong consistency in
the relative performance ranking of the prompting
methods across both judges. While Qwen3-30B
tends to assign slightly higher scores, particularly
in the 60-70% win rate range, this may reflect a
mild bias toward open-source models, noting that
the reference answers were generated by GPT-4.
Crucially, this slight scoring bias does not change
the relative ordering of methods. The high agree-
ment between the two judges supports the use of
a strong LLM judge for this task, a practice sup-
13

## Page 14

ported by prior work showing that GPT-4’s judg-
ments align with human preferences over 80% of
the time (Zheng et al., 2023; Liu et al., 2023; Huang
et al., 2025).
Finally, Table 7 also includes the results for the
“Few-shot Examples + Prompt” setting mentioned
in Section 3.3. As shown, incorporating few-shot
examples into the full system prompt did not yield
improvements and, in some cases, degraded per-
formance. This observation motivates restricting
few-shot experiments to the HPD dataset.
D
Theoretical Analysis of Prefill
Efficiency
This section provides a theoretical analysis of the
latency and computational savings from compress-
ing a long system prompt into a single [BE] token.
While the self-attention mechanism has a naive
O(n2) time complexity, modern inference engines
use I/O-aware algorithms (e.g., FlashAttention),
where the prefill stage latency scales approximately
linearly with the number of input tokens (Dao et al.,
2022; Dao, 2024). A seminal study also observe
that TTFT is primarily dictated by this linear pre-
fill cost (Kwon et al., 2023). We therefore adopt a
linear cost model to establish a realistic theoretical
upper bound on the efficiency gains.
Setup and notation.
Let Lp be the number of
system-prompt tokens and Lq the number of query
tokens. After compression, the effective prompt
length is L′
p.
For an r× compressor, L′
p
=
max{1, ⌈Lp/r⌉}; for our single-token [BE] we
have L′
p = 1. During prefill, the model processes
T = L′
p + Lq input tokens.
We define the normalized prefill FLOPs as the
fraction of prefill computation remaining after com-
pression, relative to the full prompt:
NormFLOPs(L′
p, Lq) ≜L′
p + Lq
Lp + Lq
.
(5)
Under the linear assumption, the prefill speedup
is simply 1/NormFLOPs. We model TTFT as
the sum of a fixed system overhead α (e.g., kernel
launch) and a variable cost β that is linear in the
number of prefill tokens:
TTFT(L′
p + Lq)
TTFT(Lp + Lq) = ρ + (1 −ρ) NormFLOPs(L′
p, Lq),
ρ ≜
α
α + β(Lp + Lq).
(6)
Method (×)
NormFLOPs ↓
∆TTFT ↓
RoleLLM (Lp=337, Lq=26)
Full System Prompt
1.000×
0.0%
PCC(4×)
0.304×
−69.6%
SelfCP(12×)
0.152×
−84.8%
[BE] Token (337×)
0.074×
−92.6%
GSM8K (Lp=1584, Lq=58)
Full System Prompt
1.000×
0.0%
PCC(4×)
0.276×
−72.4%
SelfCP(12×)
0.116×
−88.4%
[BE] Token (1584×)
0.036×
−96.4%
Table 8: Theoretical prefill efficiency. Normalized
FLOPs (Eq. 5) assume linear scaling with prefill tokens.
The last column shows the theoretical TTFT reduction
in a compute-bound case (ρ = 0). For realistic scenarios
with system overheads (ρ > 0), use Eq. 6.
In an idealized compute-bound scenario where
system overhead is negligible (ρ →0), the percent-
age decrease in TTFT is 1 −NormFLOPs.
Instantiation.
We plug in representative lengths
for our two settings: For RoleLLM, Lp=337,
Lq=26 (Tfull=363);
for GSM8K, Lp=1584,
Lq=58 (Tfull=1642).
With [BE] token(L′
p=1),
NormFLOPs reduces to
1+Lq
Lp+Lq . Table 8 reports
the computed values and compares against 4× and
12× compression. As Lp grows, the single-token
replacement yields proportionally larger gains.
Discussion.
This model quantifies the theoreti-
cal computational savings on the prefill stage itself.
For the 337-token RoleLLM prompt, our single-
token replacement implies a compute-bound prefill
speedup of up to 1/0.074 ≈13.5×. For the 1,584-
token GSM8K prompt, the theoretical speedup is
1/0.036 ≈27.8×. These figures represent an ide-
alized upper bound. The gap between these the-
oretical gains and the empirical measurements in
Section 3.7 is explained by the fixed system over-
heads (captured by the ρ > 0 term in our model),
which are unavoidable in practice.
E
Loss
Figure 5 presents representative loss curves for
Stage 1 and Stage 2 across different configurations.
14

## Page 15

loss weight  = 0.1   (original answer)
loss weight  = 0.1   (self-generated)
loss weight  = 0.5   (original answer)
loss weight  = 0.5   (self-generated)
loss weight  = 0.9   (original answer)
loss weight  = 0.9   (self-generated)
loss weight  = 1.0   (original answer)
loss weight  = 1.0   (self-generated)
Losses of Stage 1 and Stage 2
(rows: loss weight  from 0.1 to 1.0  columns: original answer | self-generated)
Figure 5: Losses of Stage 1 and Stage 2 grouped by loss weight λ. Rows correspond to λ ∈{0.1, 0.5, 0.9, 1.0}
from top to bottom. Columns correspond to supervision type: left = original answer, right = self-generated.
15
