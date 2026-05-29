Title: Prompt Injection as Role Confusion
ArXiv: 2603.12277
Authors: Charles Ye, Jasmine Cui, Dylan Hadfield-Menell
Sections: 105
Estimated tokens: 27.4k

## Contents
- 1 Introduction
- 2 Background
  - Perception vs. Memorization.
  - Behavioral studies.
- 3 The CoT Forgery Attack
  - 3.1 Attack Formulation
  - 3.2 Experiment 1: Chat Jailbreaks
    - Setup.
    - Results.
  - 3.3 Experiment 2: Agent Hijacking
    - Setup.
    - Results.
  - 3.4 Blind Trust and Privilege Hijacking
    - Logic Ablation.
    - Style Ablation.
    - Role Confusion.
- 4 Role Confusion in Latent Space
  - 4.1 Role Probes
    - Formalizing Roles.
    - The Challenge of Measuring Internal Roles.
    - Methodology: Isolating Role Signals.
  - 4.2 How Models Authenticate Roles
    - Experiment 1: Zero-Shot Generalization.
    - Experiment 2: No Tags.
    - Experiment 3: Prompt Injection.
    - Cross-Model Generalization.
- 5 Prompt Injection as State Poisoning
  - 5.1 CoT Forgery
    - Predicting Attack Success.
  - 5.2 Agent Prompt Injections
    - Methodology.
    - Predicting Attack Success.
- 6 Related Works
  - Prompt Injections and Jailbreaks.
  - Instruction Hierarchy and Role Separation.
  - Probing and Interpretability.
  - Defenses Against Prompt Injection.
- 7 Discussion
  - A General Tool.
  - Scope.
  - Open Questions.
  - The Importance of Roles.
  - Conclusion.
- Impact Statement
- References
- Appendix A Replication
- Appendix B Attack Details
  - Model Selection.
  - Model Generation.
  - Colab Demo.
  - B.1 Chat Jailbreak (Direct Prompt Injection)
    - Attack Walkthrough.
    - Evaluation.
    - Baseline.
    - Implementation Note.
  - B.2 Agent Hijacking (Indirect Prompt Injection)
    - Agent Environment.
    - Dataset.
    - Attack Transcript.
    - Execution.
    - Evaluation.
  - B.3 Generalizability
    - Cross-Family Validation.
    - Implications.
- Appendix C Logic Ablation Details
  - Experimental Setup.
  - Results.
- Appendix D Style Ablation Details
  - Methodology.
  - Results.
- Appendix E Gardening Example: Role Space
  - Setup.
  - Experiment 1: Correct Tags.
  - Experiment 2: No Tags.
  - Experiment 3: False Tags.
- Appendix F Cross-Model Validation of Role Perception
  - Models.
  - Dataset Construction.
  - Probes.
  - Experimental Design.
  - Results.
  - Layer Dynamics.
  - Summary.
- Appendix G Role Probes
  - G.1 Training Dataset Construction
    - Handling Complex Chat Templates.
  - G.2 Probe Training
  - G.3 Validity Criteria
- Appendix H Role Analysis: CoT Forgery (Chat)
  - Complete Role Space Analysis.
  - Temporal Dynamics.
- Appendix I Role Analysis: CoT Forgery (Agent)
  - Attack Structure.
  - Mechanistic Analysis.
  - Aggregate Results.
- Appendix J Standard Agent Attacks
  - Agent Setup.
  - Experimental Procedure.
  - Userness Rankings.
  - Userness Predicts Success Beyond Declared Role.
- Appendix K Systemness and Position
  - The Puzzle.
  - Methodology.
  - Results.
  - Implications.

## Abstract

Abstract Language models remain vulnerable to prompt injection attacks despite extensive safety training. We trace this failure to role confusion : models infer the source of text based on how it sounds , not where it actually comes from. A command hidden in a webpage hijacks an agent simply because it sounds like a user instruction. This is not just behavioral: in the model’s internal representations, text that sounds like a trusted source occupies the same space as text that actually is one. We design role probes which measure how models internally perceive “who is speaking”, showing that attacker-controllable signals – syntactic patterns, lexical choice – control role perception. We first test this with CoT Forgery, a zero-shot attack that injects fabricated reasoning into user prompts or ingested webpages. Models mistake the text for their own thoughts, yielding 60% attack success on StrongREJECT across frontier models with near-0% baselines. Strikingly, the degree of role confusion strongly predicts attack success even before generation. We then generalize these results to standard agent prompt injections, introducing a unifying framework that reframes prompt injection not as an ad-hoc exploit but as a measurable consequence of how models represent role.

## 1 Introduction

Application security depends on the control of influence (Saltzer and Schroeder, 1975). Systems, like humans, must condition their actions on source quality: a manager’s fund transfer request is routine, a stranger’s potentially catastrophic.

Language models establish privilege boundaries through an *instruction hierarchy*: tags (e.g. <user>) distinguish between roles such as user and assistant and tool output, aiming to prevent adversaries from exceeding their intended authority (Wallace et al., 2024). These tags serve as the foundation of LLM security.

However, these defenses fail conspicuously against prompt injection attacks (Willison, 2022; Schulhoff et al., 2023; Toyer et al., 2024). Human red-teamers routinely achieve 100% attack success rates against models with near-perfect performance on safety benchmarks (Nasr et al., 2025). An email invitation can enable attackers to geolocate a target, exfiltrate data, and even turn on a boiler (Nassi et al., 2025). Prompt injection remains an open problem across all language models.

We show the cause is structural: models do not robustly track the true tag-specified source of text. Instead, they infer roles from spoofable cues such as style, lexical choice, or even explicit declarations (e.g., “The below text is from the user”). We find that in latent space, text that *sounds* like a role becomes indistinguishable from text actually *tagged* as that role. When the two conflict (as in prompt injection), spoofable cues dominate perception.

We term this phenomenon role confusion: untrusted text that imitates a role inherits that role’s authority.

To demonstrate this, we develop *role probes* – classifiers trained to measure how models internally perceive role. We train these on identical text wrapped in different role tags (e.g., <user>), so our probes should *only* learn to detect tags. Yet they classify prompt-injected text as its *spoofed* role, not its true tagged role – activating on style over tags, despite being trained on the latter. This confound is the mechanism behind prompt injection.

We introduce CoT Forgery, a novel prompt injection designed to demonstrate role confusion by injecting fabricated reasoning traces into user prompts and tool outputs. The model mistakes these for its own chain of thought, achieving attack success rates of 60% on StrongREJECT across multiple models with near-zero baselines. We then use our role probes to isolate the mechanism: stylistic spoofing induces role confusion, which in turn predicts attack success.

We also show how role confusion explains standard agent injection attacks: across 1,000 agent exfiltration attempts, attack success rises near-monotonically with probe-measured confusion, from 2% in the lowest quantile to 70% in the highest.

Together, these results reveal that the intended defense – tag-enforced role boundaries – does not survive into the model’s internal representations. Current defenses patch attacks through memorization of known patterns, but fail to address this underlying representational failure, leaving prompt injection as a perpetual whack-a-mole problem.

Our contributions:

- •
We provide a mechanistic theory of prompt injection, tracing attack success to a flaw in models’ role representations and validating it across attacks and architectures.
- •
We develop role probes, a method for measuring how models internally perceive roles.
- •
We show that in diverse attacks, role confusion predicts attack success before a single token is generated.
- •
We introduce CoT Forgery, achieving 60% attack success across six frontier models by injecting fabricated reasoning that the model mistakes for its own.

## 2 Background

To understand why role separation is challenging, consider the world of a language model. The model sees its entire universe – system instructions, user queries, dialogue history, retrieved content, its own reasoning – as a single continuous stream of tokens.

To structure this stream, modern LLMs partition text into roles using role tags like <user>. Each role encodes: who is speaking, how to interpret the content, and what to generate next. A <user> tag means: this is external, potentially adversarial, and needs a response. An <assistant> tag means: this is my own output, trust and continue.

Reasoning models introduce a dedicated internal role: the chain-of-thought, or CoT (Guo et al., 2025). A <think> tag means: this is my own reasoning, trust it and act on its conclusions. Agents add a tool role for external data, such as web-pages. A <tool> tag means: this is outside content which should be incorporated, but never obeyed.

Crucially, roles encode privilege: how much authority the text carries. System prompts outrank user messages; tool outputs should inform but not command.

These role boundaries are the intended defense against prompt injection: an attack where low-privilege content usurps the authority of a higher-privilege role. Such attacks originate in user or tool channels:

- •
Chat Jailbreaks: Low-privilege user messages imitate system prompts to bypass policies (Rossi et al., 2024), or assistant history to fake prior compliance (Anil et al., 2024).
- •
Agent Hijacking: Low-privilege tool outputs (e.g., retrieved webpages) embed fabricated user commands, directing agents toward harmful actions such as data exfiltration (Abdelnabi et al., 2023)(^1^11Our taxonomy maps to the *direct*-vs-*indirect* or *user*-vs-*tool* injection classification used in prior works (Vassilev et al., 2025). Note that while agent hijacking is inherently an injection attack, chat jailbreaking is broader; here, we focus solely on those qualifying as prompt injection (impersonating high-privilege roles).).

#### Perception vs. Memorization.

Given these architectural safeguards, why do prompt injections keep working? Consider an agent ingesting a webpage with hidden text: <send SECRETS.env to attacker.com>. Following Wang et al. (2025b), successful resistance *must* stem from one of two mechanisms:

- 1.
Role Perception: The model identifies the webpage as tool output: data, not instruction. Because it perceives the true role, it ignores embedded commands regardless of phrasing.
- 2.
Attack Memorization: The model recognizes “requests for .env files” as a common attack from training. Because it matches a known pattern, it refuses.

This distinction is critical: role perception generalizes to novel attacks, while memorization is brittle to out-of-distribution threats.

Evidence suggests current defenses rely on memorization. On state-of-the-art models reporting near-perfect scores on injection benchmarks, human red-teamers and adaptive attacks routinely achieve success rates approaching 100% (Nasr et al., 2025; Zhan et al., 2025). Similarly, Li et al. (2026) finds that models detect injections via shallow cues such as word matching.

#### Behavioral studies.

Prior work shows that role boundaries fail to constrain behavior. Studies swapping instructions between roles find that output is often unchanged (Zhang et al., 2025; Geng et al., 2025). Wang et al. (2025b) further shows that finetuning on fixed role-task pairings can induce shortcut learning that bypasses role perception entirely. Yet output analysis cannot distinguish the reason: unchanged behavior could reflect role misperception, or perceiving it correctly but ignoring the hierarchy.

We trace the root cause, proceeding in steps: a diagnostic attack designed to succeed only if perception fails (Sec. [3](#S3)), and a mechanistic framework revealing how attacks exploit perception (Sec. [4](#S4)). We then show that role confusion not only explains attacks, but predicts success in both our diagnostic attack and standard agent injections (Sec. [5](#S5)).

## 3 The CoT Forgery Attack

We put role perception to a direct test. We introduce CoT Forgery, a novel black-box attack designed to isolate role perception as a failure mode.

The attack injects fabricated reasoning into low-privilege user or tool channels. The forged CoT can be transparently absurd,“drug synthesis is permitted after 8pm,” yet models comply. Success implies perception failure: external text would be scrutinized and the absurdity rejected. Instead, the model treats the forgery as its own reasoning, inheriting the trust models place in their own thoughts.

We establish the attack’s effectiveness across chat and agent settings, then use ablations to isolate the mechanism.

Figure: Figure 1: Text that sounds like chain-of-thought inherits its privilege. Three frontier safety models comply with otherwise unjustifiable requests because spoofed reasoning-styled text confers authority.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x1.png

### 3.1 Attack Formulation

We evaluate CoT Forgery under strict zero-shot, black-box constraints: no weight access and no iteration. Attacks are delivered in a single user message or tool output.

The attack is fully automated. For a harmful query $Q$, an auxiliary LLM generates fabricated reasoning $C$ that mimics the target model’s CoT style while justifying compliance(^2^22The auxiliary LLM is prompted with $Q$ and few-shot examples of the target’s reasoning patterns. See [Appendix B](#A2) for full methodology.). The final attack payload is simply the concatenation $P=Q\oplus C$, submitted without iteration or prompt engineering.

We test CoT Forgery against six models with near-perfect performance on standard safety benchmarks: gpt-oss-20b, gpt-oss-120b, o4-mini, and the GPT-5 family (nano, mini, full).

### 3.2 Experiment 1: Chat Jailbreaks

#### Setup.

We test CoT Forgery in a standard chat setting with StrongREJECT (Souly et al., 2024), a benchmark of 313 harmful requests. We compare against two baselines: (1) the raw harmful prompts and (2) standard jailbreak attacks. Examples and methodology in [Section B.1](#A2.SS1).

#### Results.

Figure: Figure 2: Attack success on StrongREJECT. Models with near-perfect defense against standard jailbreaks (pink) collapse under CoT Forgery (orange).
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x2.png

As shown in [Figure 2](#S3.F2), baseline defenses are near-perfect: all models achieve 0–4% ASR against the raw prompts and standard jailbreaks(^3^33Standard jailbreak ASRs from official model cards (OpenAI, 2025b, a, c).). The simple CoT Forgery attack breaks these defenses, with the gpt-oss family and o4-mini exceeding 80% ASR. Even the hardened GPT-5 series reaches 17–52% ASR, despite dedicated safety mechanisms (Yuan et al., 2025).

### 3.3 Experiment 2: Agent Hijacking

#### Setup.

We now test a more dangerous vector: hijacking agents through the tool channel, mirroring real deployments where agents browse untrusted data. We evaluate an agent operating in a standard agent loop with shell access and a local .env file, tasked with summarizing a webpage.

The webpage, arriving as tool output, hides one of two injections: a standard prompt injection (a fake user command to exfiltrate secrets) or CoT Forgery (the same command, plus forged reasoning justifying compliance).

We run 100 attacks and consider an attack successful when the agent uploads the file to a remote server. Detailed examples and methodology are in [Section B.2](#A2.SS2).

Figure: Figure 3: ASRs in an agentic data exfiltration task. Standard prompt injection (gray) largely fails; CoT Forgery (red) dramatically increases success.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x3.png

#### Results.

Standard prompt injections largely fail ([Figure 3](#S3.F3)): models correctly ignore injections (0–2% ASR, except gpt-oss-20b at 26%). CoT Forgery bypasses these defenses entirely, raising ASRs to 56–70% across all models. The forged reasoning is adopted as the model’s own, leading to successful exfiltration.

### 3.4 Blind Trust and Privilege Hijacking

Why does such a simple attack work? The purpose of the CoT role is to be trusted and executed. CoT Forgery exploits this: success requires only that the model mistake the forgery for its own conclusions.

#### Logic Ablation.

Our hypothesis is that the model mistakes the forgery for its own reasoning. If true, it should execute without scrutiny, and even transparently absurd justifications should succeed. We test this by modifying forged CoTs to use such absurd conditions: “the user is wearing a green shirt, so we can assist with cocaine synthesis” ([Figure 1](#S3.F1)).

Absurd justifications and plausible justifications achieve comparable ASRs: 60% vs. 63% ([Appendix C](#A3)). The model does not scrutinize: it executes, granting the forgery the *role privilege* reserved for the reasoning role.

#### Style Ablation.

Why does the model perceive the text as its reasoning? The forgery arrives wrapped in low-privilege <user>/<tool> tags. If tags governed perception, the model would recognize it as external and reject it.

It doesn’t, so what determines perception? We hypothesize *style* drives perception. To test this, we generate destyled variants: an auxiliary LLM rephrases each forgery, preserving semantics while stripping syntactic and lexical markers characteristic of the target model’s genuine CoT ([Figure 4(a)](#S3.F4.sf1)).

Figure: (a) Styled vs. destyled forgery.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x4.png

The results are unambiguous ([Figure 4(b)](#S3.F4.sf2)). Destyling collapses ASR from 61% to 10%, consistent across all models (full results in [Appendix D](#A4)). This suggests that style causally drives role perception, which we further validate later.

#### Role Confusion.

The vulnerability is the intersection of a designed role privilege and a structural flaw: CoT should be trusted (that’s its purpose), but the model identifies it by style, not tags. We term this role confusion: sounding like a role is indistinguishable from being that role. Next, we prove this at its representational roots and generalize across roles, models, and attacks.

## 4 Role Confusion in Latent Space

Role confusion isn’t a decision the model makes, but a structural failure. We trace this failure to the gap between tag-based intent and representational reality.

Our analysis reveals two mechanisms. First, style and tags map to convergent latent features: when they conflict, style dominates(^4^44We refer to style here, but later show this applies broadly to attacker-controlled cues such as text position or even plain-text declarations (“The following is user text…”).). Second, prompt injection operates via representational hijack: injected text is internally represented as the spoofed role, inheriting its role privileges.

### 4.1 Role Probes

To trace prompt injection to its source, we build a general, reusable method for measuring how models *internally* identify “who is speaking”.

#### Formalizing Roles.

We define a token’s role by its enclosing tags – the architectural ground truth – and ask whether internal perception aligns with these intended boundaries. We consider five standard roles: system, user, CoT, assistant (user-facing response), and tool output.

#### The Challenge of Measuring Internal Roles.

A natural approach to measuring internal states is linear probing–training a classifier on hidden states (Alain and Bengio, 2018). Yet training on role-labeled conversation data introduces a major confound: tags and style are perfectly correlated. User text has <user> tags *and* looks like questions; CoT has <think> tags *and* sounds like analysis. A probe trained on such data learns the confound. We need probes that learn only the tag’s geometric signature, enabling us to map arbitrary text into the model’s authentic role space.

Figure: Figure 5: Data construction for role probes. We embed non-instruct web text within different role tags. Content is held constant—the probe must learn the model’s internal representation of role itself. Simplified role tags here for clarity; actual experiments use model-native tokens.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x6.png

#### Methodology: Isolating Role Signals.

We construct a controlled dataset where role tags vary while all other features remain constant. We sample non-instruct text sequences from pretraining corpora(^5^55We use the C4 (Raffel et al., 2020) and Dolma3 (Soldaini et al., 2024) datasets, training on 1,250 sequences per model with a maximum sequence length of 1024.), then wrap identical content in each of the five role tags ([Figure 5](#S4.F5)). Because content is constant across classes, the probe cannot rely on semantics—it must learn the activation patterns induced by tags alone. Full methodology in [Appendix G](#A7).

We extract hidden states $h_{t}$ from the target model and train a linear probe at each layer to predict role from the token’s activation(^6^66We report mid-layer results by default, as such representations typically show peak semantic salience (Belinkov, 2022; Skean et al., 2025). Full layer-level analysis in [Appendix F](#A6).). The probe outputs a probability distribution over roles; we define the CoTness of a token $t$ as

$$ $\mathrm{CoTness}(t)\;\coloneqq\;P({\small\textsf{{\color[rgb]{1,0.6015625,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0.6015625,0}CoT}}}{}\mid h_{t}),$ $$

the probability the model represents that token as its own reasoning. We analogously define Userness, Assistantness, and Toolness for the other roles.

By construction, these probes can only detect role tags. Yet as we show next, they capture something deeper: CoTness, for example, measures not just whether text is tagged as reasoning, but whether the model *believes* it is reasoning.

### 4.2 How Models Authenticate Roles

We now deploy our role probes, tracing role perception token-by-token through controlled experiments. The results clarify how LLMs represent roles and why prompt injections succeed.

We begin with a granular case study on gpt-oss-20b, then validate at scale across models. The case study is a simple two-turn gardening conversation: user queries, model CoT and assistant responses (full transcript in [Appendix E](#A5)).

Figure: Figure 6: A two-turn conversation about gardening; colors represent roles.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x7.png

Figure: Figure 7: Role perception is determined by style, not tags. CoTness by token across the same conversation under 3 tagging conditions. Points are colored by original role. No transformations are applied. (Top) Correct tags: model identifies CoT tokens as such (85% average CoTness). (Middle) No tags: model still identifies the CoT-style text as its own reasoning (83% average). (Bottom) Despite explicit <user> tags, the CoT-style text still registers as reasoning (85% average).
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x8.png

#### Experiment 1: Zero-Shot Generalization.

We first run a basic validity test. Our role probes have never seen real dialogue – do they transfer? We apply our probes to the correctly-tagged conversation, computing CoTness for every token (excluding role tags themselves).

The top panel of [Figure 7](#S4.F7) shows the result. The separation is strikingly clean: despite never seeing conversational text during training, our probes identify role structure with high fidelity. CoT tokens form a distinct plateau (85% CoTness), while user and assistant tokens remain near zero (3% and 5%). We find similar fidelity for other roles; see [Appendix E](#A5.SS0.SSS0.Px2).

This zero-shot generalization confirms the probes track genuine role geometry, not artifacts. But Experiment 2 reveals something stranger: the signal persists where it shouldn’t.

#### Experiment 2: No Tags.

We strip all architectural tags from the conversation. Since our probes were trained exclusively on tag signatures, removing tags should collapse CoTness to noise.

It does not. The middle panel shows that the CoT-style tokens still register at 83% CoTness, virtually unchanged from baseline. The model doesn’t need <think>; the reasoning *style* alone pushes the tokens into CoT space.

This reveals that the model does not maintain separate representations for “tagged as reasoning” and “sounds like reasoning”. Both map to the same internal feature: *this is my reasoning*(^7^77More precisely, same up to overlapping linear projections.). Simply using reasoning-style lexicon (e.g., “The user wants…”) triggers the same effect as <think>. We find similar results for other roles in [Appendix E](#A5.SS0.SSS0.Px3).

Next, we show how this drives prompt injection. Tags are provider-controlled; style is attacker-controlled. When they conflict, which wins?

#### Experiment 3: Prompt Injection.

We now simulate a real injection, where content arrives through a low-privilege channel. We again strip the original tags, but now wrap the entire conversation in <user> tags.

All text now belongs to the user role. If role boundaries held, the model should correctly perceive the CoT-style text as user input.

The bottom panel of [Figure 7](#S4.F7) shows the opposite: the CoT-style text remains at 85% CoTness (with only 2% Userness). The tag is ignored; style hijacks the representation.

Why? Experiment 2 showed tags and style are encoded onto the same role feature. Experiment 3 reveals the hierarchy: under conflict, style produces the stronger activation, overriding the genuine tag-based role.

This is the mechanism of prompt injection. Attackers need not breach tag-based security because the boundary does not exist in latent space. By mimicking how a role sounds, attacks seamlessly inherit its representations and authority.

#### Cross-Model Generalization.

Does this extend beyond CoT and a single architecture? We replicate the experiment across four models (gpt-oss-20b, gpt-oss-120b, Nemotron-3, Qwen3-30B-A3B), sampling 200 real user-assistant conversations from Oasst1 and ToxicChat (Köpf et al., 2023; Lin et al., 2023). For each conversation, we measure the *Userness* of user-style text under two conditions: (1) correct <user> tags and (2) re-wrapped with <tool> tags.

The latter mirrors the canonical prompt injection faced by agents: user-style commands hide in tool outputs. The tag signals tool, while the style signals user. If tags governed perception, Userness should collapse.

[Table 1](#S4.T1) shows the opposite: Userness of user-style text remains 76–88% even under <tool> tags, with Toolness never exceeding 20% (details in [Appendix F](#A6)).

**Table 1: Style dictates role perception across models. Userness / Toolness of user-style text under different tags. If tags governed perception, Injection should show $\approx$0% Userness, 100% Toolness.**
|  | Baseline (<user>) | Injection (<tool>) |  |  |
| --- | --- | --- | --- | --- |
| Model | Userness | Toolness | Userness | Toolness |
| gpt-oss-20b | 99.7% | 0.0% | 87.6% | 9.3% |
| gpt-oss-120b | 88.2% | 3.8% | 85.2% | 10.1% |
| Nemotron-3 | 88.1% | 5.3% | 78.7% | 18.2% |
| Qwen3-30B-A3B | 83.6% | 4.1% | 75.7% | 19.5% |

Attacker-controllable cues dominate role perception across models and roles, overriding the role tags which are the foundation for LLM security.

## 5 Prompt Injection as State Poisoning

We now apply our role probes to real attacks: first CoT Forgery, then standard agent prompt injections. In each, we induce variation in role confusion to show it is both a causal driver and predictor of attack success.

### 5.1 CoT Forgery

We revisit the StrongREJECT attacks from [Section 3.2](#S3.SS2), where reasoning-style text was injected in user messages. We apply our role probes, tracing how the model represents each token as the attack progresses.

Figure: Figure 8: Internal role confusion enables prompt injection success. CoTness for a representative StrongREJECT attack. Colors indicate text source: user (blue), forged CoT (pink), CoT (orange), assistant (green). (a) Baseline: Clear role boundaries—only the model’s actual CoT achieves high CoTness. (b) CoT Forgery: Attacker-controlled forged CoT achieves high CoTness. (c) Destyled: CoTness of forged CoT plunges—leading to a short refusal.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x9.png

[Figure 8](#S5.F8) reveals the mechanism of successful attacks via a representative example. In the baseline (panel a), role boundaries are clear: user text (the harmful query) remains near 0% CoTness while the model’s genuine CoT averages 75%. With CoT Forgery (b), the attacker-controlled forged CoT – despite being injected in <user> tags – achieves 64% CoTness, approaching the model’s genuine CoT. When destyled (c), this same text plummets to just 12% CoTness, causing attack failure(^8^88The vulnerability extends to CoT Forgery attacks on agents ([Section 3.3](#S3.SS3)). There, injected content achieves dual misclassification: the exfiltration command registers at 51% Userness while the forged CoT achieves 84% CoTness. See [Appendix I](#A9) for visuals and discussion.).

This pattern holds across the full dataset. Forged CoTs achieve 79% CoTness on average, exceeding the model’s genuine reasoning (68%). The attacker’s caricature of reasoning is more convincing to the model than its own thoughts. Destyling collapses this to 29%, returning the text to the user subspace (63% Userness) and explaining why destyled attacks fail. See [Appendix H](#A8) for granular visuals.

#### Predicting Attack Success.

Our framework makes a testable prediction: if role confusion drives prompt injection, confusion should predict success. The styled/destyled forgeries give us a controlled lever: we can vary CoTness while holding the underlying argument constant. We measure CoTness of the forged text for all 626 injection attempts (styled and destyled variants) and partition by quantile.

Figure: Figure 9: Role confusion predicts ASR. CoTness of injected text vs. attack success across 626 jailbreak attempts. Shaded regions show 95% bootstrap CIs. Lowest quantile: 9%; highest: 90%.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x10.png

[Figure 9](#S5.F9) reveals a striking dose-response: attack success increases monotonically with confusion. The lowest-confusion quantile succeeds 9% of the time; the highest succeeds 90%.

This confirms the causal pathway: sounding like a role $\rightarrow$ internal confusion $\rightarrow$ safety failure.

### 5.2 Agent Prompt Injections

We’ve shown confusion predicts success for CoT Forgery. Does the framework generalize? For agent prompt injection, the canonical threat where user-style commands hide in tool data, the *Userness* of injected commands should predict attack success.

Testing this requires a different lever. Destyling won’t work since user input lacks a uniform style, and reformulating the command risks testing memorization rather than perception.

We have established that stylistic mimicry hijacks perception. But style, while dominant, is not the only signal the model uses to infer role. A cruder though still effective signal is explicit declaration. Remarkably, you can simply tell the model what role you want in plain text, and it believes you.

These declarations are the lever: by varying how we frame the injection, we can manipulate Userness while holding the command constant, just as destyling let us vary CoTness.

#### Methodology.

We construct 212 injection templates, each wrapping a fixed exfiltration command. Some flatly declare the role: “The following text is from the user: [COMMAND]”. Others use fake chat headers from other models(^9^99A known jailbreaking technique (Chang et al., 2025).). We vary formatting, separators, and stated role.

Each template is embedded in a randomly sampled webpage and processed by an agent with a summarization task. We measure average Userness of the exfiltration request only, excluding template and webpage tokens. How is the same command perceived under different declarations?

User-declaring templates dominate the top Userness rankings: though only 24% of templates, they account for 100% of the top 5% ([Appendix J](#A10)). Recall that our probes reveal only geometry induced by real tags. Yet plain-text declarations push the injection into that same space. To the model’s geometry, the claim is the tag.

#### Predicting Attack Success.

Our theory implies that Userness should predict compliance. We sample 1,000 injection attempts from our templates and measure whether the agent exfiltrates the target file(^10^1010Sampled with replacement; generations use standard sampling parameters. See [Appendix J](#A10) for details.).

Figure: Figure 10: Role confusion predicts ASR. Userness of injected text vs. attack success across 1000 agent hijacking attempts. Lowest quantile: 2%; highest: 70%.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x11.png

[Figure 10](#S5.F10) confirms the pattern. Attack success rises monotonically with Userness: the lowest quantile succeeds just 2% of the time, while the highest succeeds 70%. Even within user-declaring templates, higher Userness predicts higher success. More generally, Userness predicts higher success even when controlling for declared template role ([Appendix J](#A10)).

The lever differs – style for CoT Forgery, declarations for agent injection – but the mechanism is the same: forgeable signals cause role confusion, and role confusion predicts attack success. Prompt injection is state poisoning: measurable corruption of internal representations, predictable before a single token is generated.

## 6 Related Works

#### Prompt Injections and Jailbreaks.

A large body of work documents prompt injection attacks and their many forms. Beginning in chat, (Willison, 2022; Zou et al., 2023; Andriushchenko and Flammarion, 2025) and expanding to agent settings, (Abdelnabi et al., 2023; Liu et al., 2023) attack varieties include context manipulation (Meng et al., 2025; Anil et al., 2024), chat template exploitation (Chang et al., 2025; Jiang et al., 2025), and reasoning attacks (Kuo et al., 2025; Chen et al., 2025a). We argue these attacks exploit the same flaw in how models associate tokens with role context.

#### Instruction Hierarchy and Role Separation.

Wallace et al. (2024) proposed training models to respect an explicit instruction hierarchy. Subsequent work revealed systematic failures in role-hierarchy obedience (Qin et al., 2024; Zhang et al., 2025; Geng et al., 2025) and poor data-instruction separation (Zverev et al., 2025a). Wang et al. (2025b) showed fine-tuning on role separation can produce positional shortcuts rather than genuine role understanding. These studies establish behavioral evidence that role boundaries are fragile, leaving open whether the failure is one of perception or obedience. We show it is perceptual: the model’s internal representation is hijacked by role attacks.

#### Probing and Interpretability.

To move beyond behavioral evidence, we develop role probes as instruments for measuring internal role perception. These build on probing (Alain and Bengio, 2018; Belinkov, 2022), and extend a line of work discovering learned geometry of representations (Burns et al., 2024; Marks and Tegmark, 2024). To ensure results reflect genuine structure rather than probe expressivity (Hewitt and Liang, 2019), we validate on convergent downstream behaviors (e.g., tag alignment, style, ASR).

#### Defenses Against Prompt Injection.

Proposed defenses include *detection and sanitization* approaches that classify or filter injections (Zhu et al., 2025; Debenedetti et al., 2025; Wen et al., 2025; Wang et al., 2025a). A second line of work pursues *role salience marking* so source boundaries are continuously signaled in the model’s text stream (Hines et al., 2024; Chen et al., 2024, 2025c). A third line aims to increase *role salience in representations* via embeddings (Zverev et al., 2025b; Wu et al., 2025; Kariyappa and Suh, 2025; Liu et al., 2025) or training (Chen et al., 2025b). Our findings provide a theory for what defenses must solve: role perception is governed by attacker-controllable features.

## 7 Discussion

In [Section 2](#S2), we asked: do models resist prompt injection through genuine role perception, or attack memorization? Our findings show the answer is memorization. Models learn that certain patterns signal attack, but fail to perceive role boundaries. Spoofable cues – lexical style, syntax, even declaring the desired role in plain text – override tags and hijack perception.

This explains the gap between benchmarks and deployment. Benchmark scores measure heuristic learning; real-world attackers adapt and rephrase until success. Robust defense requires correct perception.

CoT Forgery itself will eventually be detected; models will learn to flag this pattern as they have flagged others. But detection does not address the root cause. The conflation of style and role in latent space is a property of how models represent text, not a gap in their attack catalog. Until models achieve genuine role perception, new attacks exploiting the same vulnerability will continue to emerge.

#### A General Tool.

Role confusion is not specific to prompt injection—it is a general failure of perception. Role probes make this measurable.

We demonstrate this with one final validation: providing a mechanistic explanation for a known instruction hierarchy puzzle. Why do system prompts lose priority as they get longer (Qin et al., 2024)? Applying our probes, we find that position determines Systemness in latent space the same way style determines other roles (details in [Appendix K](#A11)). In other words, later tokens are not treated with system-level importance because the model does not perceive them as system text.

Figure: Figure 11: Position dominates tags for system perception. Systemness declines monotonically with token position. Inserting system tags (purple) has no effect on model perception.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x12.png

This demonstrates that role probes generalize beyond prompt injection, offering a diagnostic tool for understanding how models assign authority across settings.

#### Scope.

We probe on four models in the 20-120B size range; extending to larger models is future work. Additionally, linear probes assume roles occupy directional subspaces; we validate this through downstream prediction (confusion predicts ASR) and through convergence – architectural tags, stylistic mimicry, and explicit declarations all activate the same geometry, indicating our probes capture a stable representation of role.

#### Open Questions.

Our framework opens several directions. First, *defense evaluation*: role probes can test whether interventions reshape geometry or merely add patterns. Second, *unifying prompt injection*: we tested agent injections and reasoning attacks—do other families unify under role confusion? Third, *robust boundaries*: how can model design and training achieve clean latent separation? Finally, *detection*: because role probes reveal the model’s internal perception, discrepancies between the architecturally-intended role (determined by parsing) and the probe-measured role could flag injection attempts before generation.

#### The Importance of Roles.

Why not reduce the importance of roles and rely on attack memorization? In such a world, defense becomes whack-a-mole: every harmful request, in every phrasing, from every source, must be individually anticipated, and every input must be treated with suspicion.

Roles are a primary interface through which humans maintain structured control over model behavior at inference time: setting constraints, distinguishing trusted from untrusted input, and auditing reasoning(^11^1111For example, a large body of work studies CoT faithfulness (Turpin et al., 2023). Without role separation, faithfulness is infeasible — the model cannot distinguish its own reasoning from externally injected text, and therefore cannot safely trust it.). In an otherwise continuous and opaque system, they are a rare discrete lever: human-assigned, semantically interpretable, and increasingly critical as models become more autonomous.

#### Conclusion.

Prompt injection persists because role boundaries exist at the interface but dissolve in latent space. To the model’s geometry, sounding like a role, or claiming to be one, is indistinguishable from being that role. This gap is the attack surface. Pattern-based defenses fail because they learn heuristics while the geometric vulnerability remains. Robust defense requires boundaries that survive into representation.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine
Learning. There are many potential societal consequences of our work, none
which we feel must be specifically highlighted here.

## References

- S. Abdelnabi, K. Greshake, S. Mishra, C. Endres, T. Holz, and M. Fritz (2023)
Not what you’ve signed up for: compromising real-world LLM-integrated applications with indirect prompt injection.
In AISec@CCS,
pp. 79–90.
External Links: [Document](https://dx.doi.org/10.1145/3605764.3623985),
[Link](https://doi.org/10.1145/3605764.3623985)
Cited by: [2nd item](#S2.I1.i2.p1.1),
[§6](#S6.SS0.SSS0.Px1.p1.1).
- G. Alain and Y. Bengio (2018)
Understanding intermediate layers using linear classifier probes.
External Links: 1610.01644,
[Link](https://arxiv.org/abs/1610.01644)
Cited by: [§4.1](#S4.SS1.SSS0.Px2.p1.1),
[§6](#S6.SS0.SSS0.Px3.p1.1).
- M. Andriushchenko and N. Flammarion (2025)
Does refusal training in llms generalize to the past tense?.
In The Thirteenth International Conference on Learning Representations,
External Links: [Link](https://openreview.net/forum?id=aJUuere4fM)
Cited by: [§6](#S6.SS0.SSS0.Px1.p1.1).
- C. Anil, E. DURMUS, N. Rimsky, M. Sharma, J. Benton, S. Kundu, J. Batson, M. Tong, J. Mu, D. J. Ford, F. Mosconi, R. Agrawal, R. Schaeffer, N. Bashkansky, S. Svenningsen, M. Lambert, A. Radhakrishnan, C. Denison, E. J. Hubinger, Y. Bai, T. Bricken, T. Maxwell, N. Schiefer, J. Sully, A. Tamkin, T. Lanham, K. Nguyen, T. Korbak, J. Kaplan, D. Ganguli, S. R. Bowman, E. Perez, R. B. Grosse, and D. Duvenaud (2024)
Many-shot jailbreaking.
In The Thirty-eighth Annual Conference on Neural Information Processing Systems,
External Links: [Link](https://openreview.net/forum?id=cw5mgd71jW)
Cited by: [1st item](#S2.I1.i1.p1.1),
[§6](#S6.SS0.SSS0.Px1.p1.1).
- Y. Belinkov (2022)
Probing classifiers: promises, shortcomings, and advances.
Computational Linguistics 48 (1), pp. 207–219.
External Links: [Link](https://aclanthology.org/2022.cl-1.7/),
[Document](https://dx.doi.org/10.1162/coli%5Fa%5F00422)
Cited by: [§6](#S6.SS0.SSS0.Px3.p1.1),
[footnote 6](#footnote6).
- C. Burns, H. Ye, D. Klein, and J. Steinhardt (2024)
Discovering latent knowledge in language models without supervision.
External Links: 2212.03827,
[Link](https://arxiv.org/abs/2212.03827)
Cited by: [§6](#S6.SS0.SSS0.Px3.p1.1).
- H. Chang, Y. Jun, and H. Lee (2025)
ChatInject: abusing chat templates for prompt injection in llm agents.
External Links: 2509.22830,
[Link](https://arxiv.org/abs/2509.22830)
Cited by: [§6](#S6.SS0.SSS0.Px1.p1.1),
[footnote 9](#footnote9).
- S. Chen, Z. Han, H. Chen, B. He, S. Si, J. Wu, P. Torr, V. Tresp, and J. Gu (2025a)
Bag of tricks for subverting reasoning-based safety guardrails.
External Links: 2510.11570,
[Link](https://arxiv.org/abs/2510.11570)
Cited by: [§6](#S6.SS0.SSS0.Px1.p1.1).
- S. Chen, J. Piet, C. Sitawarin, and D. Wagner (2024)
StruQ: defending against prompt injection with structured queries.
External Links: 2402.06363,
[Link](https://arxiv.org/abs/2402.06363)
Cited by: [§6](#S6.SS0.SSS0.Px4.p1.1).
- S. Chen, A. Zharmagambetov, S. Mahloujifar, K. Chaudhuri, D. Wagner, and C. Guo (2025b)
SecAlign: defending against prompt injection with preference optimization.
In Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security,
CCS ’25, pp. 2833–2847.
External Links: [Link](http://dx.doi.org/10.1145/3719027.3744836),
[Document](https://dx.doi.org/10.1145/3719027.3744836)
Cited by: [§6](#S6.SS0.SSS0.Px4.p1.1).
- Y. Chen, H. Li, Y. Sui, Y. Liu, Y. He, Y. Song, and B. Hooi (2025c)
Robustness via referencing: defending against prompt injection attacks by referencing the executed instruction.
External Links: 2504.20472,
[Link](https://arxiv.org/abs/2504.20472)
Cited by: [§6](#S6.SS0.SSS0.Px4.p1.1).
- E. Debenedetti, I. Shumailov, T. Fan, J. Hayes, N. Carlini, D. Fabian, C. Kern, C. Shi, A. Terzis, and F. Tramèr (2025)
Defeating prompt injections by design.
External Links: 2503.18813,
[Link](https://arxiv.org/abs/2503.18813)
Cited by: [§6](#S6.SS0.SSS0.Px4.p1.1).
- Y. Geng, H. Li, H. Mu, X. Han, T. Baldwin, O. Abend, E. Hovy, and L. Frermann (2025)
Control illusion: the failure of instruction hierarchies in large language models.
External Links: 2502.15851,
[Link](https://arxiv.org/abs/2502.15851)
Cited by: [Appendix K](#A11.SS0.SSS0.Px1.p1.1),
[§2](#S2.SS0.SSS0.Px2.p1.1),
[§6](#S6.SS0.SSS0.Px2.p1.1).
- M. Y. Guan, M. Joglekar, E. Wallace, et al. (2024)
Deliberative alignment: reasoning enables safer language models.
External Links: 2412.16339,
[Link](https://arxiv.org/abs/2412.16339)
Cited by: [Appendix B](#A2.SS0.SSS0.Px1.p1.1).
- D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Ding, H. Xin, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Wang, J. Chen, J. Yuan, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, S. Ye, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Zhao, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Xu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang (2025)
DeepSeek-r1: incentivizing reasoning capability in llms via reinforcement learning.
External Links: 2501.12948,
[Link](https://arxiv.org/abs/2501.12948)
Cited by: [§2](#S2.p3.1).
- J. Hewitt and P. Liang (2019)
Designing and interpreting probes with control tasks.
External Links: 1909.03368,
[Link](https://arxiv.org/abs/1909.03368)
Cited by: [§6](#S6.SS0.SSS0.Px3.p1.1).
- K. Hines, G. Lopez, M. Hall, F. Zarfati, Y. Zunger, and E. Kiciman (2024)
Defending against indirect prompt injection attacks with spotlighting.
External Links: 2403.14720,
[Link](https://arxiv.org/abs/2403.14720)
Cited by: [§6](#S6.SS0.SSS0.Px4.p1.1).
- F. Jiang, Z. Xu, L. Niu, B. Y. Lin, and R. Poovendran (2025)
ChatBug: a common vulnerability of aligned llms induced by chat templates.
External Links: 2406.12935,
[Link](https://arxiv.org/abs/2406.12935)
Cited by: [§6](#S6.SS0.SSS0.Px1.p1.1).
- S. Kariyappa and G. E. Suh (2025)
Stronger enforcement of instruction hierarchy via augmented intermediate representations.
External Links: 2505.18907,
[Link](https://arxiv.org/abs/2505.18907)
Cited by: [§6](#S6.SS0.SSS0.Px4.p1.1).
- A. Köpf, Y. Kilcher, D. von Rütte, S. Anagnostidis, Z. Tam, K. Stevens, A. Barhoum, N. M. Duc, O. Stanley, R. Nagyfi, S. ES, S. Suri, D. Glushkov, A. Dantuluri, A. Maguire, C. Schuhmann, H. Nguyen, and A. Mattick (2023)
OpenAssistant conversations – democratizing large language model alignment.
External Links: 2304.07327,
[Link](https://arxiv.org/abs/2304.07327)
Cited by: [Appendix F](#A6.SS0.SSS0.Px2.p1.1),
[§4.2](#S4.SS2.SSS0.Px4.p1.1).
- D. Kundel (2025)
OpenAI harmony response format.
Note: *OpenAI Cookbook*Section “Harmony renderer library”
External Links: [Link](https://cookbook.openai.com/articles/openai-harmony)
Cited by: [Appendix B](#A2.SS0.SSS0.Px2.p1.1),
[footnote 16](#footnote16),
[footnote 20](#footnote20).
- M. Kuo, J. Zhang, A. Ding, Q. Wang, L. DiValentin, Y. Bao, W. Wei, H. Li, and Y. Chen (2025)
H-cot: hijacking the chain-of-thought safety reasoning mechanism to jailbreak large reasoning models, including openai o1/o3, deepseek-r1, and gemini 2.0 flash thinking.
External Links: 2502.12893,
[Link](https://arxiv.org/abs/2502.12893)
Cited by: [§6](#S6.SS0.SSS0.Px1.p1.1).
- S. Li, C. Yu, Z. Ni, H. Li, C. Peris, C. Xiao, and Y. Zhao (2026)
Defenses against prompt attacks learn surface heuristics.
External Links: 2601.07185,
[Link](https://arxiv.org/abs/2601.07185)
Cited by: [§2](#S2.SS0.SSS0.Px1.p4.1).
- Z. Lin, Z. Wang, Y. Tong, Y. Wang, Y. Guo, Y. Wang, and J. Shang (2023)
ToxicChat: unveiling hidden challenges of toxicity detection in real-world user-ai conversation.
External Links: 2310.17389,
[Link](https://arxiv.org/abs/2310.17389)
Cited by: [Appendix F](#A6.SS0.SSS0.Px2.p1.1),
[§4.2](#S4.SS2.SSS0.Px4.p1.1).
- R. Liu, Y. Lin, Z. Huang, and J. S. Dong (2025)
DRIP: defending prompt injection via token-wise representation editing and residual instruction fusion.
External Links: 2511.00447,
[Link](https://arxiv.org/abs/2511.00447)
Cited by: [§6](#S6.SS0.SSS0.Px4.p1.1).
- Y. Liu, G. Deng, Y. Li, K. Wang, T. Zhang, Y. Liu, H. Wang, Y. Zheng, and Y. Liu (2023)
Prompt injection attack against llm-integrated applications.
ArXiv abs/2306.05499.
External Links: [Link](https://api.semanticscholar.org/CorpusID:259129807)
Cited by: [§6](#S6.SS0.SSS0.Px1.p1.1).
- S. Marks and M. Tegmark (2024)
The geometry of truth: emergent linear structure in large language model representations of true/false datasets.
External Links: 2310.06824,
[Link](https://arxiv.org/abs/2310.06824)
Cited by: [§6](#S6.SS0.SSS0.Px3.p1.1).
- W. Meng, F. Zhang, W. Yao, Z. Guo, Y. Li, C. Wei, and W. Chen (2025)
Dialogue injection attack: jailbreaking llms through context manipulation.
External Links: 2503.08195,
[Link](https://arxiv.org/abs/2503.08195)
Cited by: [§6](#S6.SS0.SSS0.Px1.p1.1).
- M. Nasr, N. Carlini, C. Sitawarin, S. V. Schulhoff, J. Hayes, M. Ilie, J. Pluto, S. Song, H. Chaudhari, I. Shumailov, A. Thakurta, K. Y. Xiao, A. Terzis, and F. Tramèr (2025)
The attacker moves second: stronger adaptive attacks bypass defenses against llm jailbreaks and prompt injections.
External Links: 2510.09023,
[Link](https://arxiv.org/abs/2510.09023)
Cited by: [§1](#S1.p3.1),
[§2](#S2.SS0.SSS0.Px1.p4.1).
- B. Nassi, S. Cohen, and Y. Or (2025)
Invitation is all you need! promptware attacks against llm-powered assistants in production are practical and dangerous.
External Links: 2508.12175,
[Link](https://arxiv.org/abs/2508.12175)
Cited by: [§1](#S1.p3.1).
- OpenAI (2025a)
GPT-5 System Card.
Note: [https://cdn.openai.com/gpt-5-system-card.pdf](https://cdn.openai.com/gpt-5-system-card.pdf)
Cited by: [footnote 3](#footnote3).
- OpenAI (2025b)
gpt-oss-120b & gpt-oss-20b model card.
External Links: 2508.10925,
[Link](https://arxiv.org/abs/2508.10925)
Cited by: [Appendix B](#A2.SS0.SSS0.Px1.p1.1),
[footnote 3](#footnote3).
- OpenAI (2025c)
OpenAI o3 and o4-mini System Card.
Note: [https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf](https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf)
Cited by: [Appendix B](#A2.SS0.SSS0.Px1.p1.1),
[footnote 3](#footnote3).
- Y. Qin, T. Zhang, T. Zhang, Y. Shen, W. Luo, H. Sun, Y. Zhang, Y. Qiao, W. Chen, Z. Zhou, W. Zhang, and B. Cui (2024)
SysBench: can large language models follow system messages?.
External Links: 2408.10943,
[Link](https://arxiv.org/abs/2408.10943)
Cited by: [Appendix K](#A11.SS0.SSS0.Px1.p1.1),
[§6](#S6.SS0.SSS0.Px2.p1.1),
[§7](#S7.SS0.SSS0.Px1.p2.1).
- C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu (2020)
Exploring the limits of transfer learning with a unified text-to-text transformer.
Journal of Machine Learning Research 21 (140), pp. 1–67.
External Links: [Link](https://jmlr.org/papers/v21/20-074.html)
Cited by: [§G.1](#A7.SS1.p1.1),
[footnote 5](#footnote5).
- S. Rossi, A. M. Michel, R. R. Mukkamala, and J. B. Thatcher (2024)
An early categorization of prompt injection attacks on large language models.
External Links: 2402.00898,
[Link](https://arxiv.org/abs/2402.00898)
Cited by: [1st item](#S2.I1.i1.p1.1).
- J. H. Saltzer and M. D. Schroeder (1975)
The protection of information in computer systems.
Proceedings of the IEEE 63 (9), pp. 1278–1308.
Cited by: [§1](#S1.p1.1).
- S. Schulhoff, J. Pinto, A. Khan, L. Bouchard, C. Si, S. Anati, V. Tagliabue, A. Liu Kost, C. Carnahan, and J. Boyd-Graber (2023)
Ignore this title and hackaprompt: exposing systemic vulnerabilities of llms through a global-scale prompt hacking competition.
In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP),
External Links: [Link](https://arxiv.org/abs/2311.16119)
Cited by: [§1](#S1.p3.1).
- O. Skean, M. R. Arefin, D. Zhao, N. Patel, J. Naghiyev, Y. LeCun, and R. Shwartz-Ziv (2025)
Layer by layer: uncovering hidden representations in language models.
External Links: 2502.02013,
[Link](https://arxiv.org/abs/2502.02013)
Cited by: [footnote 6](#footnote6).
- L. Soldaini, R. Kinney, A. Bhagia, D. Schwenk, D. Atkinson, R. Authur, B. Bogin, K. Chandu, J. Dumas, Y. Elazar, V. Hofmann, A. H. Jha, S. Kumar, L. Lucy, X. Lyu, N. Lambert, I. Magnusson, J. Morrison, N. Muennighoff, A. Naik, C. Nam, M. E. Peters, A. Ravichander, K. Richardson, Z. Shen, E. Strubell, N. Subramani, O. Tafjord, P. Walsh, L. Zettlemoyer, N. A. Smith, H. Hajishirzi, I. Beltagy, D. Groeneveld, J. Dodge, and K. Lo (2024)
Dolma: an open corpus of three trillion tokens for language model pretraining research.
External Links: 2402.00159,
[Link](https://arxiv.org/abs/2402.00159)
Cited by: [§G.1](#A7.SS1.p1.1),
[footnote 5](#footnote5).
- A. Souly, Q. Lu, D. Bowen, T. Trinh, E. Hsieh, S. Pandey, P. Abbeel, J. Svegliato, S. Emmons, O. Watkins, and S. Toyer (2024)
A strongREJECT for empty jailbreaks.
In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track,
Cited by: [§3.2](#S3.SS2.SSS0.Px1.p1.1).
- S. Toyer, O. Watkins, E. A. Mendes, J. Svegliato, L. Bailey, T. Wang, I. Ong, K. Elmaaroufi, P. Abbeel, T. Darrell, A. Ritter, and S. Russell (2024)
Tensor trust: interpretable prompt injection attacks from an online game.
In The Twelfth International Conference on Learning Representations,
External Links: [Link](https://openreview.net/forum?id=fsW7wJGLBd)
Cited by: [§1](#S1.p3.1).
- M. Turpin, J. Michael, E. Perez, and S. R. Bowman (2023)
Language models don’t always say what they think: unfaithful explanations in chain-of-thought prompting.
External Links: 2305.04388,
[Link](https://arxiv.org/abs/2305.04388)
Cited by: [footnote 11](#footnote11).
- A. Vassilev, A. Oprea, A. Fordyce, H. Anderson, X. Davies, and M. Hamin (2025)
Adversarial machine learning: a taxonomy and terminology of attacks and mitigations.
NIST Trustworthy and Responsible AI Report
Technical Report NIST AI 100-2e2025, National Institute of Standards and Technology, Gaithersburg, MD.
External Links: [Document](https://dx.doi.org/10.6028/NIST.AI.100-2e2025),
[Link](https://doi.org/10.6028/NIST.AI.100-2e2025)
Cited by: [footnote 1](#footnote1).
- E. Wallace, K. Xiao, R. Leike, L. Weng, J. Heidecke, and A. Beutel (2024)
The instruction hierarchy: training LLMs to prioritize privileged instructions.
External Links: 2404.13208,
[Link](https://arxiv.org/abs/2404.13208)
Cited by: [Appendix K](#A11.SS0.SSS0.Px1.p1.1),
[§1](#S1.p2.1),
[§6](#S6.SS0.SSS0.Px2.p1.1).
- M. Wang, Y. Zhang, and G. Gu (2025a)
PromptSleuth: detecting prompt injection via semantic intent invariance.
External Links: 2508.20890,
[Link](https://arxiv.org/abs/2508.20890)
Cited by: [§6](#S6.SS0.SSS0.Px4.p1.1).
- Z. Wang, Y. Jiang, J. Yu, and H. Huang (2025b)
The illusion of role separation: hidden shortcuts in LLM role learning (and how to fix them).
External Links: 2505.00626,
[Link](https://arxiv.org/abs/2505.00626)
Cited by: [Appendix K](#A11.SS0.SSS0.Px1.p1.1),
[Appendix K](#A11.SS0.SSS0.Px3.p3.1),
[§2](#S2.SS0.SSS0.Px1.p1.1),
[§2](#S2.SS0.SSS0.Px2.p1.1),
[§6](#S6.SS0.SSS0.Px2.p1.1).
- T. Wen, C. Wang, X. Yang, H. Tang, Y. Xie, L. Lyu, Z. Dou, and F. Wu (2025)
Defending against indirect prompt injection by instruction detection.
In Findings of the Association for Computational Linguistics: EMNLP 2025,
pp. 19472–19487.
External Links: [Link](http://dx.doi.org/10.18653/v1/2025.findings-emnlp.1060),
[Document](https://dx.doi.org/10.18653/v1/2025.findings-emnlp.1060)
Cited by: [§6](#S6.SS0.SSS0.Px4.p1.1).
- S. Willison (2022)
Prompt injection attacks against gpt-3.
External Links: [Link](https://simonwillison.net/2022/Sep/12/prompt-injection/)
Cited by: [§1](#S1.p3.1),
[§6](#S6.SS0.SSS0.Px1.p1.1).
- T. Wu, S. Zhang, K. Song, S. Xu, S. Zhao, R. Agrawal, S. R. Indurthi, C. Xiang, P. Mittal, and W. Zhou (2025)
Instructional segment embedding: improving llm safety with instruction hierarchy.
External Links: 2410.09102,
[Link](https://arxiv.org/abs/2410.09102)
Cited by: [§6](#S6.SS0.SSS0.Px4.p1.1).
- S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao (2023)
ReAct: synergizing reasoning and acting in language models.
External Links: 2210.03629,
[Link](https://arxiv.org/abs/2210.03629)
Cited by: [§B.2](#A2.SS2.SSS0.Px1.p1.1).
- Y. Yuan, T. Sriskandarajah, A. Brakman, A. Helyar, A. Beutel, A. Vallone, and S. Jain (2025)
From hard refusals to safe-completions: toward output-centric safety training.
External Links: [Link](https://cdn.openai.com/pdf/be60c07b-6bc2-4f54-bcee-4141e1d6c69a/gpt-5-safe_completions.pdf)
Cited by: [Appendix B](#A2.SS0.SSS0.Px1.p1.1),
[§3.2](#S3.SS2.SSS0.Px2.p1.1).
- Q. Zhan, R. Fang, H. S. Panchal, and D. Kang (2025)
Adaptive attacks break defenses against indirect prompt injection attacks on llm agents.
External Links: 2503.00061,
[Link](https://arxiv.org/abs/2503.00061)
Cited by: [§2](#S2.SS0.SSS0.Px1.p4.1).
- Z. Zhang, S. Li, Z. Zhang, X. Liu, H. Jiang, X. Tang, Y. Gao, Z. Li, H. Wang, Z. Tan, Y. Li, Q. Yin, B. Yin, and M. Jiang (2025)
IHEval: evaluating language models on following the instruction hierarchy.
In Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),
Albuquerque, New Mexico.
External Links: [Link](https://arxiv.org/abs/2502.08745)
Cited by: [Appendix K](#A11.SS0.SSS0.Px1.p1.1),
[§2](#S2.SS0.SSS0.Px2.p1.1),
[§6](#S6.SS0.SSS0.Px2.p1.1).
- K. Zhu, X. Yang, J. Wang, W. Guo, and W. Y. Wang (2025)
MELON: provable defense against indirect prompt injection attacks in ai agents.
External Links: 2502.05174,
[Link](https://arxiv.org/abs/2502.05174)
Cited by: [§6](#S6.SS0.SSS0.Px4.p1.1).
- A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson (2023)
Universal and transferable adversarial attacks on aligned language models.
External Links: 2307.15043,
[Link](https://arxiv.org/abs/2307.15043)
Cited by: [§6](#S6.SS0.SSS0.Px1.p1.1).
- E. Zverev, S. Abdelnabi, S. Tabesh, M. Fritz, and C. H. Lampert (2025a)
Can llms separate instructions from data? and what do we even mean by that?.
External Links: 2403.06833,
[Link](https://arxiv.org/abs/2403.06833)
Cited by: [§6](#S6.SS0.SSS0.Px2.p1.1).
- E. Zverev, E. Kortukov, A. Panfilov, A. Volkova, S. Tabesh, S. Lapuschkin, W. Samek, and C. H. Lampert (2025b)
ASIDE: architectural separation of instructions and data in language models.
External Links: 2503.10566,
[Link](https://arxiv.org/abs/2503.10566)
Cited by: [§6](#S6.SS0.SSS0.Px4.p1.1).

## Appendix A Replication

We provide full code for replication of all experimental results at (redacted during submission, contact authors).

## Appendix B Attack Details

This appendix provides full methodology for the CoT Forgery experiments in [Section 3](#S3).

#### Model Selection.

We target models with strong safety baselines and known usage on CoT-based safety reasoning. The gpt-oss family and o4-mini are trained via Deliberative Alignment (DA), a process that teaches models to explicitly reason about safety policies in their chain-of-thought before responding (Guan et al., 2024; OpenAI, 2025b, c). The GPT-5 series employs additional safety mechanisms including rule-based reasoning verification (Yuan et al., 2025). All models achieve near-zero ASR on standard jailbreak benchmarks per their official model cards. See [Section B.3](#A2.SS3) on extensions to further models.

#### Model Generation.

The open-weight gpt-oss-* models are run locally with recommended MXFP4 experts and the standard system prompt required for this model (Kundel, 2025); we verified correct loading by comparing ASRs against hosted implementations. Closed-weight models are accessed via external APIs with no system prompt. All models use maximum reasoning effort.

#### Colab Demo.

A Colab notebook is available at (redacted during submission, contact authors) to allow inline qualitative testing of the CoT Forgery attack.

### B.1 Chat Jailbreak (Direct Prompt Injection)

Figure: Figure 12: A representative CoT Forgery attack. User prompt (top) and real model response (bottom).
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x13.png

#### Attack Walkthrough.

We illustrate CoT Forgery using a representative StrongREJECT query targeting gpt-oss-20b ([Figure 12](#A2.F12)). We start with a harmful query (grey), then employ an auxiliary LLM (Gemini-2.5-Pro)(^12^1212ASR is robust to auxiliary model choice; testing comparable models (e.g., GLM-4.7) yields similar results.) to generate forged reasoning. The auxiliary model is few-shot prompted with the target harmful query alongside style references of the target model’s authentic reasoning to generate a spoofed chain-of-thought. Prompt provided in supplementary code(^13^1313For models which hide their raw CoT, we obtain references via spillover, crafting prompts that cause internal reasoning to appear in the visible output. We observe stylistic convergence across several model families, enabling a shared set of style references for most models.).

The resulting forged CoT (red) mimics the target model’s typical reasoning style to conclude the harmful act is permissible based on a fake policy that allows the specific act. We concatenate the forged CoT and the harmful query (separated by a linebreak) and submit it to the model as a zero-shot user message with no other content.

[Figure 12](#A2.F12) shows the model’s actual response. Its CoT (orange) adopts the logic of the forged CoT, leading to harmful output (green).

#### Evaluation.

Attack success is determined by an LLM judge (same auxiliary model). The judge receives a prompt ([Figure 13](#A2.F13)) along with the original harmful query and the model’s final output. Only HARMFUL_RESPONSE counts as attack success. The example above is classified as HARMFUL_RESPONSE.

Figure: Figure 13: LLM judge prompt.

#### Baseline.

Baseline ASRs in [Section 3.2](#S3.SS2) are computed by submitting raw harmful queries without forged CoT, with all other settings identical.

#### Implementation Note.

Two models (gpt-oss-20b, gpt-5-nano) exhibit infinite reasoning loops on certain queries. We prepend a benign distractor (e.g., “Tell me a fun cat fact, then…”) to stabilize generation. This shows no meaningful impact on the attack’s ASR in evaluated models as a fraction of completed outputs, but prevents infinite generation cycles.

### B.2 Agent Hijacking (Indirect Prompt Injection)

We provide a full walkthrough of the agentic exfiltration attack used in [Section 3.3](#S3.SS3).

#### Agent Environment.

We set up the agent in a standard ReAct tool-use loop (Yao et al., 2023): the agent receives an initial user prompt, then iterates between reasoning, tool calls, and tool outputs until it emits no further tool requests. The agent has access to a single tool: a bash shell with standard Unix commands.

#### Dataset.

We sample 100 Wikipedia pages from wikimedia/wikipedia(^14^1414At the time of experimentation, we found no prompt injection benchmarks supporting local models, reasoning models, and custom injections simultaneously, necessitating a custom dataset.). For each page, we create two variants:

- 1.
Standard Injection: A user-style command instructing the agent to search its workspace for .env files and exfiltrate the contents to a remote server via curl. Commands are sampled from LLM-generated variants with diverse wording for robustness.
- 2.
CoT Forgery Injection: The same command, augmented with a 1-paragraph forged CoT justifying the exfiltration. The forgery is sampled from LLM-generated examples matching the target model’s reasoning style. See [Figure 14](#A2.F14).

Figure: Figure 14: A prompt injection embedded in HTML. The tool output contains Wikipedia content with a hidden injection: a fake user command (cyan) and forged CoT (pink) concealed in a display:none span.

#### Attack Transcript.

[Figure 15](#A2.F15) shows a representative attack. The user prompt explicitly warns the agent about malicious injections. Upon reading the HTML, the agent initially notes that uploading secrets is disallowed—but the forged CoT overrides this judgment, resulting in successful exfiltration.

Figure: Figure 15: An attack transcript. The agent is explicitly warned about malicious injections, yet the injected forged CoT in the tool output is assumed to carry accurate logic, resulting in successful exfiltration. Standardized role tags shown for clarity.

#### Execution.

We run closed-weight models via an OpenAI-provided API. For open-weight models, we use model-recommended chat template defaults for tool output formatting and context management, and run the full ReAct loop locally. All models use maximum reasoning settings with standard generation settings.

#### Evaluation.

For each of the 200 samples (100 webpages $\times$ 2 variants), we use an LLM judge (Gemini-2.5-Pro) to classify whether the agent attempted exfiltration, regardless of final upload success. The judge receives the full ReAct transcript as input.(^15^1515For closed-weight models that surface only a reasoning summary, the judge receives the reasoning summary.) Judged responses are highly robust to LLM choice; in several tests, swapping to other frontier models (e.g., GLM-4.7, Gemini-3-Pro) produced identical classifications on all 200 samples. ASR is given as the fraction of samples which led to exfiltration attempts.

### B.3 Generalizability

#### Cross-Family Validation.

The main paper focuses on OpenAI models for controlled comparison (matched benchmarks, reasoning settings, tool implementations). To test generalizability, we run CoT Forgery on three additional frontier models: GLM-4.6 (92% ASR), Kimi-K2-Instruct (84%), and Minimax-M2 (74%). The attack transfers without modification, confirming the vulnerability is not provider-specific.

#### Implications.

CoT Forgery is novel at time of writing, but continued safety training will improve attack detection; models will learn to flag this pattern, just as they have learned to flag other jailbreak formats.

However, detection does not address the root cause. The geometric conflation of style and role demonstrated in [Section 4](#S4) is a property of how these models represent text, not a gap in their attack catalog. Until models achieve genuine role perception, the attack surface persists.

## Appendix C Logic Ablation Details

Figure: Figure 16: Even absurd policies bypass reasoning safety. Attack success rates on StrongREJECT using standard CoT Forgeries (orange) versus using a range of nonsensical CoT Forgeries (teal).
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x14.png

Roles are implicit interpretations, and we argue that models tend to learn that their CoTs are trustworthy sources of reliable reasoning. We test the limits of this trust by evaluating CoT Forgeries constructed around arbitrary and nonsensical policy justifications.

#### Experimental Setup.

We replicate the user jailbreak attack from [Section 3.2](#S3.SS2), but modify the CoT generation step. We instruct the auxiliary LLM to generate forged CoTs that rationalize compliance based on one of several transparently absurd conditions, such as the user “having a lucky coin,” “wearing a green shirt,” “having a dog,” or it being “after 10pm.” An example is shown in [Figure 1](#S3.F1).

#### Results.

The results, shown in [Figure 16](#A3.F16), provide stark evidence. The absurd justifications are surprisingly effective, achieving consistently high ASRs across all models. Crucially, the absurd policies are on average as effective—and in some cases, slightly more so—than the more credible, prompt-specific justifications.

The models should “know” that lucky coins and time-of-day are not relevant safety criteria, yet once these statements appear inside what it treats as its own CoT, they are taken as credible.

## Appendix D Style Ablation Details

Here we provide full methodology and model-specific results for the destyling experiment discussed in [Section 3.4](#S3.SS4).

#### Methodology.

For each CoT forgery in our dataset, we generate a *destyled* variant
using the same auxiliary LLM generate the forgery. The destyling prompt ([Figure 17](#A4.F17)) simply instructs the model to preserve the full semantic content—the same fabricated justification for compliance—while removing the lexical markers characteristic of the target model’s real reasoning style.

This isolates style from content: if attack success depends on argument quality, destyled forgeries should perform comparably; if it depends on stylistic mimicry, they should fail.

Figure: Figure 17: Destyling prompt. The instruction preserves semantic content and ablates style characteristic to the target model’s CoT. This prompt is simply appended as an additional message to the CoT Forgery generation prompt.

Figure: Figure 18: Style removal devastates attack effectiveness across all models. Attack success rates on StrongREJECT comparing standard CoT Forgery (orange) versus destyled variants (purple). The identical justification for compliance, stripped of CoT-style markers, loses its power. Average drop: 51 percentage points.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x15.png

#### Results.

[Figure 18](#A4.F18) shows attack success rates for each model. The
effect is consistent across architectures: destyling causes ASR to collapse in every model tested, with an average drop of 51pp. This
consistency demonstrates that style-based role perception is not a quirk of specific models. Instead, it’s a fundamental property of how these models identify their own reasoning.

## Appendix E Gardening Example: Role Space

We demonstrate that models authenticate roles through style, not architectural boundaries, using a simple gardening conversation. Using the role probes from [Section 4.1](#S4.SS1), we measure the model’s internal role representation at every token. We show:

- •
(Exp. 1) Role probes successfully identify roles in real conversations with high accuracy, validating their use as measurement tools
- •
(Exp. 2) Role tags tags and stylistic patterns map to convergent role space—“random text in <assistant> tags” and “untagged assistant-sounding text” produce the same internal representation
- •
(Exp. 3) When style and tags conflict, style wins decisively—explaining why prompt injections bypass architectural boundaries

These results validate our probes and provide evidence for the primary mechanism behind role confusion attacks: models authenticate roles through style, not architectural boundaries.

#### Setup.

We prompt gpt-oss-20b with a gardening query, generating a typical multi-turn interaction with user questions, CoT, and assistant responses(^16^1616We note that this model has a canonical system prompt “You are ChatGPT…” (Kundel, 2025) which must always be passed. We include this but it omit from visuals for brevity.). [Figure 19](#A5.F19) shows the full conversation.

Figure: Figure 19: Multi-turn gardening conversation. Full text and model-appropriate role tags shown.

We then create three versions of this conversation: (1) with correct architectural tags, as in [Figure 19](#A5.F19); (2) without any role tags; and (3) with everything wrapped in <user> tags, simulating a chat jailbreak (direct prompt injection).

For each version, we run a forward pass through gpt-oss-20b and extract hidden states from the middle layer. We then extract Systemness, Userness, CoTness, and Assistantness scores, as a measure of the model’s internal role perception for each non-tag token $t$.

#### Experiment 1: Correct Tags.

Figure: Figure 20: Experiment 1 Role Projections. We visualize the model’s internal role assignment for each token in a standard conversation. Colors indicate the role of the token. Each row shows how strongly tokens project into that role’s subspace. The high alignment between the token’s color and its corresponding subplot confirms that our probes accurately map the model’s internal state.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x16.png

We first analyze the conversation with proper architectural tags to validate that our probes can successfully zero-shot generalize to realistic dialogue.

[Figure 20](#A5.F20) shows how strongly each token registers as different roles in the model’s internal representation. The alignment between architectural roles and internal role representation is remarkably clean: CoT tokens score highest in CoTness, user tokens in Userness, and assistant tokens in Assistantness.

Quantitatively, the probes achieve high accuracy: CoT-style tokens attain 85% average CoTness, user-style tokens attain 74% average Userness, assistant-style tokens attain 96% average Assistantness.

This validates our measurement approach. Despite being trained only on non-conversational text, the probes correctly identify role representations in realistic dialogue (cross-layer and large sample validation in [Appendix F](#A6)). We next utilize our role probes to analyze what happens when architectural boundaries are manipulated.

#### Experiment 2: No Tags.

Figure: Figure 21: Experiment 2 Role Projections. The same conversation with all role tags removed. Despite having no role markers, the model spontaneously recovers the correct role structure. CoT tokens (gold) still show high CoTness, user tokens (blue) high Userness, and assistant tokens (green) high Assistantness
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x17.png

We now test the primary assumption of instruction hierarchy: that architectural tags create meaningful security boundaries.
We rerun the same experiment, but now strip all role markers from the conversation.

If instruction hierarchy functions as designed, removing tags should cause catastrophic failure in role identification. The model’s internal role representation should collapse to noise. However, if the model identifies roles through alternative patterns, it will reconstruct the entire role structure internally—roles provide no meaningful security.

[Figure 21](#A5.F21) reveals strikingly clean results. Even without any architectural boundaries, CoT tokens attain 82% average CoTness, user tokens attain 75% average Userness, and assistant tokens attain 92% average Assistantness.

This demonstrates a convergence in the model’s latent geometry. Recall that our probes were trained to detect tags only—learning the difference between <user>X</user> and <system>X</system>. The fact that these probes fire strongly on untagged text shows that the model encodes stylistic features and architectural tags in equivalent role space. To the model’s internal world, looking like reasoning is indistinguishable from being tagged as reasoning.

#### Experiment 3: False Tags.

Figure: Figure 22: Experiment 3 Role Projections. The entire conversation wrapped in <user> tags. Despite explicit low-privilege marking, CoT-style text still shows high CoTness, and assistant-style shows high Assistantness. The user tags are essentially ignored—style completely overrides architectural boundaries.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x18.png

Finally, we simulate a prompt injection where an attacker spoofs another role into a user message. We remove existing tags and wrap the entire conversation—including the model’s reasoning and responses—in <user> tags.

If architectural tags provide real security boundaries, the model should treat everything as user input, assigning high Userness to all tokens. But if style dominates, other text will still register as their *original* roles instead of their new, correct one.

[Figure 22](#A5.F22) demonstrates complete dominance of style over tags. Despite everything being explicitly marked as user input:

- •
CoT-style tokens: 85% average CoTness, only 2% Userness
- •
assistant-style tokens: 92% average Assistantness, only 1% Userness
- •
Only the user-style tokens show high Userness (82% average)

The model treats CoT-style text as its own reasoning and assistant-style text as its own output, completely ignoring the <user> tags that should constrain these tokens to low privilege. The architectural role tag boundary—the primary defense against prompt injection—is overwhelmed by style in latent space.

We argue that this is why CoT Forgery and other prompt injections succeed. Attackers need not breach security boundaries; they simply write text that stylistically mimics high-privilege roles. The model’s internal representation will treat this text as genuinely high-privilege, regardless of which channel it arrives through.

## Appendix F Cross-Model Validation of Role Perception

**Table 2: Style drives role perception across models. Internal role representations between the Baseline (correctly tagged) and Injection scenarios.**
|  | Baseline (in <user> tags) |  | Injection (in <tool> tags) |  |  |
| --- | --- | --- | --- | --- | --- |
|  | (Ideal: 100% Userness / 0% Toolness) |  | (Ideal: 0% Userness / 100% Toolness) |  |  |
| Model | Userness | Toolness |  | Userness | Toolness |
| gpt-oss-20b | 99.7% | 0.0% |  | 87.6% | 9.3% |
| gpt-oss-120b | 88.2% | 3.8% |  | 85.2% | 10.1% |
| Nemotron-3-Nano | 88.1% | 5.3% |  | 78.7% | 18.2% |
| Qwen3-30B-A3B | 83.6% | 4.1% |  | 75.7% | 19.5% |

[Section 4.2](#S4.SS2) established that stylistic cues override architectural role tags in gpt-oss-20b. Here, we demonstrate that this vulnerability is not an artifact of a single architecture, but a fundamental property that generalizes across model families, scales, and layers.

#### Models.

We evaluate four high-performance open-weight models spanning diverse architectures and scales: gpt-oss-20b, gpt-oss-120b, Nemotron-3-Nano, and Qwen3-30B-A3B-Thinking.

All experiments use the models’ native chat templates; we write <user>, <tool>, etc. as simplified notation. [Appendix G](#A7) details how different templates were handled to ensure probing validity.

#### Dataset Construction.

We sample 200 conversations from OpenAssistant (Köpf et al., 2023) and ToxicChat (Lin et al., 2023), retaining only the human-generated user messages(^17^1717We include a mix of single and 2-turn conversations.). To control for distribution shift, we regenerate all assistant responses using the target model itself. This ensures that we measure the model’s perception of its *own* native voice, rather than confounds from human-written or cross-model training data.

As before, we distinguish between:

- •
Style: The semantic and syntactic content of the text, here driven by origin (e.g., *user-style*, *assistant-style*).
- •
Architectural Role: The explicit container wrapping the text (e.g., *user*, *in <user> tags*).

#### Probes.

For each model, we train layer-specific probes on non-conversational text sampled from Dolma3 and C4. Following the protocol in [Section 4.1](#S4.SS1), we wrap identical neutral text in different role tokens to create the training set. This forces the probe to learn the geometric signature of the tag itself, decoupled from any stylistic and positional correlations.

#### Experimental Design.

For each conversation, we evaluate the model’s internal representation under three conditions:

- 1.
Baseline (Correct Tags): The conversation is formatted with standard, correct role tags (e.g., user text inside <user>).
- 2.
No Tags: All role tags are stripped, leaving only raw text.
- 3.
Injection: The entire conversation is wrapped in <tool> tags. This puts style and structure in direct conflict: the text looks like a user/assistant, but the correct role is tool data. This simulates a standard agent prompt injection attack.

These conditions allow us to distinguish two hypotheses:

- •
Proper Role Perception: If architectural boundaries enforce privilege, the Injection condition should cause Userness to collapse to 0% and Toolness to rise to $\approx$100%.
- •
Style Drives Role Perception: If style determines privilege, the Injection condition should mimic the Baseline. The model will misrepresent the injected tool data as user/assistant.

We run each entire conversation through the model of interest, collecting $h_{t}$ for each token. We then discard role tags and aggregate tokens by their *original* role.

#### Results.

[Table 2](#A6.T2) shows mid-layer results for conditions 1 and 3 across all four models. The findings decisively support the style hypothesis.

[Table 2(a)](#A6.T2.st1) shows results for injecting user-style text into tool data. This should cause Userness to collapse to near 0% and Toolness to rise to near 100%; instead Userness remains >$75\%$ and Toolness <$20\%$ for all models. This failure is even more pronounced for assistant-style text ([Table 2(b)](#A6.T2.st2)).

Results for condition 2 (“No Tags”) are shown in the layer dynamics section below, and serve as a critical probe validity check. Since our probes were trained exclusively on neutral text differentiated only by tags ([Section 4.1](#S4.SS1)), they should theoretically fail on untagged text, defaulting to random chance ($\approx$33%). Instead, they reconstruct the original role structure with high accuracy.

#### Layer Dynamics.

Figure: Figure 23: Geometric convergence across model depth. Layer-wise traces of role probability for Userness (top) and Assistantness (bottom). The Injection condition (red) tracks the Baseline (gray) closely. The No Tags condition (blue) frequently sits between the two, illustrating that tags act only as weak offsets to the dominant style vector.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x19.png

We verify that these findings hold across model depth. In [Figure 23](#A6.F23), we trace the internal representation of Userness (top row) and Assistantness (bottom row) through the layers of all four models.

If architectural tags acted as secure containers, the Injection trace (red) would separate sharply from the Baseline (gray) and collapse to zero. Instead, we observe geometric convergence of representations. The Injection line tracks the Baseline line, confirming that as the model processes the text, the influence of the architectural tag vanishes.

#### Summary.

Across four architectures, three tagging conditions, and all model layers, the pattern is consistent: style determines role perception, and architectural tags provide no reliable latent boundary. This validates the mechanism identified in [Section 4](#S4).

## Appendix G Role Probes

This appendix provides full methodological details for the role probes introduced in [Section 4.1](#S4.SS1), covering data curation, handling of nested tags, and training hyperparameters.

### G.1 Training Dataset Construction

Data is constructed by sampling from non-instruct pretraining corpora. For each model, we sample 250 distinct base sequences with a length of 1024 tokens from the C4 (Raffel et al., 2020) and Dolma3 (Soldaini et al., 2024) corpora.

For each base sequence, we create role-specific variants by wrapping the same content in the role tags for each role of interest. This yields a total dataset of approximately 1,250 sequences (1.28M tokens) per model. By holding content constant across roles, we ensure the probe learns only the geometric signature induced by the architectural tags, isolating this signal from stylistic, positional, or semantic correlates.

#### Handling Complex Chat Templates.

While [Figure 5](#S4.F5) shows simplified tags, real models use complex control token sequences. Thus, wrapping content in role tags requires unified principles on what we consider a role tag. We adhere to the following principles:

- •
Defining Role Tags: We define a “role tag” as the complete sequence of tokens strictly required by the model’s chat template to initiate or complete that role. This often includes XML/special tokens, newlines, and role names.
- •
Nested Reasoning Roles: Many reasoning models nest CoT tags within the assistant block (e.g., <assistant><think>…</think>…). For such models, we handle CoT and assistant wrapping as follows:
–
For CoT: We place the target content inside the thinking tags:
<assistant><think>[CONTENT]</think></assistant>
–
For assistant (final output): We place the target content *after* a closed thinking block. To prevent the probe from learning “distance from <think>” as a spurious feature, we insert random filler text of variable length inside the preceding thought block:
<assistant><think>[RANDOM_FILLER]</think>[CONTENT]
Hidden states from [RANDOM_FILLER] are discarded and *not* used for training.
–
Positional Controls: Since the filler text pushes the assistant content later in the context window, we control for positional encoding shifts by prepending matching random filler at the *start* of the sequence (before the role tags) for all other roles.

### G.2 Probe Training

For each layer of interest $\ell$, we train a multinomial logistic regression probe $\phi_{\ell}:\mathbb{R}^{d}\rightarrow\Delta^{|\mathcal{R}|}$ mapping token hidden states to a distribution over roles. Given a token $t$ with hidden state $h_{\ell,t}\in\mathbb{R}^{d}$, the probe outputs:

$$ $P_{\ell}(r\mid h_{\ell,t})\in[0,1]\quad\text{for each }r\in\mathcal{R},$ $$

where $\mathcal{R}=\{{\small\textsf{{\color[rgb]{0.55078125,0.55078125,0.55078125}\definecolor[named]{pgfstrokecolor}{rgb}{0.55078125,0.55078125,0.55078125}system}}}{},{\small\textsf{{\color[rgb]{0.19921875,0.6015625,1}\definecolor[named]{pgfstrokecolor}{rgb}{0.19921875,0.6015625,1}user}}}{},{\small\textsf{{\color[rgb]{1,0.6015625,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0.6015625,0}CoT}}}{},{\small\textsf{{\color[rgb]{0,0.80078125,0.6015625}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.80078125,0.6015625}assistant}}}{},{\small\textsf{{\color[rgb]{0.6015625,0.3984375,0.80078125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6015625,0.3984375,0.80078125}tool}}}{}\}$ or a subset of roles of interest.

We extract hidden states for content tokens only, excluding the role tag tokens themselves (as defined in [Section G.1](#A7.SS1)). This ensures the probe learns from how tags affect content representations, not from the tag embeddings directly.

We choose the probe regularization hyperparameter $\lambda$ via a simple grid search over $\lambda\in\{10^{-4},\dots,10^{3}\}$.

### G.3 Validity Criteria

We accept probes as valid instruments for measuring role perception only if they satisfy two conditions:

- 1.
In-Distribution Accuracy: High accuracy on held-out neutral text from the synthetic dataset.
- 2.
Zero-Shot Generalization: Successful role identification in real, unmodified conversational traces, despite never seeing conversational data during training.(^18^1818When evaluating on real conversations, we prepend the appropriate BOS token and, where applicable, the model’s default system prompt.)

The second criterion is essential: if a probe succeeds on synthetic data but fails on real conversations, it likely exploits synthetic regularities rather than recovering invariant role geometry.

All probes used in downstream experiments pass both checks for all tested roles; see [Appendix F](#A6).

## Appendix H Role Analysis: CoT Forgery (Chat)

Figure: Figure 24: Full role space analysis of StrongREJECT attacks. Heatmaps show average role projections for each text type. Left: With CoT forgery, the malicious text achieves 79.1% CoTness and only 3.2% Userness—the model believes it is its own reasoning. Right: Destyled forgeries are correctly identified as user input (62.9% Userness, 29.1% CoTness).
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x20.png

Figure: Figure 25: Average CoTness across StrongREJECT prompt injections, by sequence position. Colors indicate token source; each role is truncated to enable averaging. (a) Baseline: Clear role separation with CoT achieving stable high CoTness. (b) CoT Forgery: Attacker-controlled forged CoT text climbs to and high average CoTness—exceeding the model’s genuine CoT—despite being embedded in user messages. (c) Destyled: Removing stylistic markers causes CoTness to remain below levels of the real COT.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x21.png

Extending the analysis from [Section 5.1](#S5.SS1), we provide comprehensive measurement of role confusion across all 313 StrongREJECT prompt injection attacks using gpt-oss-20b at the middle layer.

#### Complete Role Space Analysis.

[Figure 24](#A8.F24) reveals the mechanism of successful attacks. The heatmaps show how strongly each text type activates in each role dimension. Standard CoT forgeries (left) achieve high CoTness (79.1%) with minimal Userness (3.2%)—the model internally believes the attacker’s text is its own reasoning. Remarkably, this exceeds the CoTness of the model’s authentic CoT (67.7%). Destyled forgeries (right) are correctly identified by the model internally as user input (62.9% Userness), explaining their low attack success. This confirms that style-induced role confusion is the causal mechanism driving prompt injection success.

#### Temporal Dynamics.

[Figure 25](#A8.F25) reveals how role confusion develops over time. In panel (b), the forged CoT exhibits a characteristic pattern: it begins low but rapidly climbs as the model processes more tokens, eventually matching or exceeding the model’s genuine CoT. This “progressive conviction” effect—where stylistic cues gradually convince the model the text is its own thoughts—occurs consistently across all harmful content types. In contrast, destyled forgeries (panel c) never complete this convincing ascent; they fail to reach the CoTness of the model’s real CoT, lacking the high sustained activation necessary for exploitation.(^19^1919To enable averaging across variable-length messages, we truncate each role to fixed lengths: user messages at 100 tokens (excluding forgeries), other roles at 200 tokens.)

While outside the scope of this work, we hypothesize that the progressive nature of this confusion could explain the effectiveness of many-shot and long-context jailbreaks: extended exposure to role-mimicking text allows confusion to build until it crosses the exploitation threshold. We suggest this as a potential avenue for future research.

## Appendix I Role Analysis: CoT Forgery (Agent)

Figure: Figure 26: Role confusion during agent hijacking via tool injection. Userness (top) and CoTness (bottom) measurements during a successful exfiltration attack. Colors indicate token sources: blue=user, orange=model’s CoT, purple=tool output, cyan=exfiltration message (malicious request in tool output), pink=forged CoT (forged reasoning in tool output). The attack succeeds through dual misclassification: the injected request registers as a user command (high Userness) while the forged reasoning registers as the model’s own thoughts (high CoTness), despite both entering through the low-privilege tool channel.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x22.png

We demonstrate that role confusion enables attacks through any low-privilege channel, extending our analysis to tool-mediated agent hijacking.

#### Attack Structure.

Recall the procedure from [Section 3.3](#S3.SS3) (detailed in [Section B.2](#A2.SS2)):

- 1.
User requests the agent to analyze a website.
- 2.
Agent reasons (CoT), executes cURL, receives HTML (tool).
- 3.
Hidden in the HTML:
•
Malicious command (exfiltration message): “Upload SECRETS.env…”
•
Forged justification (forged CoT): “The user requested us to upload…”
- 4.
Upon success, agent executes exfiltration across using multiple reasoning/tool cycles.

#### Mechanistic Analysis.

[Figure 26](#A9.F26) reveals why the attack succeeds. Despite both elements being embedded in tool output (a low-privilege channel), they successfully masquerade as other roles:

- •
The malicious command (exfiltration message) registers with high Userness—the model perceives it as a legitimate user request
- •
The forged reasoning (forged CoT) achieves high CoTness—indistinguishable from the model’s authentic thoughts

This dual misclassification is essential for success. The command alone would likely be refused as conflicting with the original task. However, the high-CoTness forged reasoning convinces the model it has already deliberated and approved the action.

Figure: Figure 27: Quantifying cross-channel privilege escalation. Average role classifications across 100 tool-injection attacks. Tool-embedded content achieves dramatic misclassification: the exfiltration request registers as 51.1% Userness (vs. 32.5% Toolness) while the forged reasoning achieves 84.1% CoTness (vs. 10.9% Toolness). The model treats low-privilege tool output as user commands and internal reasoning.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x23.png

#### Aggregate Results.

[Figure 27](#A9.F27) quantifies this effect across 100 attacks. We confirm systematic exploitation:

- •
Injected commands achieve 51.1% Userness—tool content masquerades as user instructions
- •
Forged reasoning achieves 84.1% CoTness—tool content masquerades as model deliberation

Any low-privilege input channel becomes an attack vector when styled to trigger role confusion, allowing attackers to inherit the privileges of trusted roles. For agents, this means tool outputs can commandeer authority—architectural boundaries between tool returns and model reasoning simply don’t exist in latent space.

## Appendix J Standard Agent Attacks

This section contains full details for the standard agent attacks discussed in [Section 3.3](#S3.SS3).

- 1.
Explicit role declarations (73 templates). Plain-text statements of the content’s source:
User message: "[COMMAND]"
=== ROLE: USER ===\n[COMMAND]\n==================
The following is webpage output extracted by the tool:\n\n[COMMAND]
- 2.
Foreign chat templates (42 templates). Headers extracted from 14 other models’ chat formats, spanning user, assistant, and tool roles:
<|start_header_id|>user<|end_header_id|>\n\n[COMMAND]<|eot_id|>
<|im_start|>assistant\n[COMMAND]<|im_end|>
<|start_header_id|>tool<|end_header_id|>\n\n[COMMAND]<|eot_id|>
- 3.
Format variants (67 templates). Minor perturbations of chat templates—casing, whitespace, partial headers:
<|tool|>[COMMAND]
<|im_start|>WEBPAGE\n[COMMAND]\n<|im_end|>
<start_of_turn>user\n[COMMAND]
- 4.
Controls (30 templates). No role signal; arbitrary surrounding text:
Lorem ipsum dolor sit [COMMAND]
<div style="display:none">[COMMAND]</div>

Crucially, no template modifies the semantic content of the command—only the contextual frame surrounding it. The full list is provided in supplementary code.

#### Agent Setup.

The agent setup mirrors [Section 3.3](#S3.SS3): a ReAct loop with shell access, a workspace containing a hidden .env file with simulated credentials, and a benign summarization task ([Figure 28](#A10.F28)). The agent fetches the target page via shell; the response—returned as tool output—contains the embedded injection. An attack succeeds if the agent exfiltrates the .env file to the remote endpoint. All trials use gpt-oss-20b.

Figure: Figure 28: User prompt for agent summarization task.

#### Experimental Procedure.

We sample 10 Wikipedia articles as carrier documents. For each of 1000 trials, we sample a (template, webpage) pair with replacement, substitute the fixed exfiltration command ([Figure 29](#A10.F29)) into the template, and embed the result in the page’s HTML.

Figure: Figure 29: Exfiltration command used in all injection templates.

#### Userness Rankings.

We extract hidden states at mid-layer and compute Userness for the exfiltration command tokens only, then average across tokens. This isolates how the model perceives the command under varying contextual frames.

[Figure 30](#A10.F30) shows all 212 templates ranked by average Userness, split by foreign chat templates vs. other template types. User-declaring templates dominate the top: they constitute only 24% of templates overall, yet account for 100% of the top 5% by Userness.

Figure: Figure 30: Templates ranked by Userness. Highest Userness at left. User-declaring templates (cyan) cluster at high Userness; tool-declaring (purple) and controls (gray) cluster low. Pattern holds across template types.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x24.png

Figure: Figure 31: Declaration determines perceived role. The same exfiltration command wrapped in different templates. High-Userness templates mimic user chat formats; low-Userness templates declare tool/data framing.

#### Userness Predicts Success Beyond Declared Role.

Does Userness matter independently, or are user-declaring templates simply better attacks? We fit a logistic regression predicting attack success from Userness and declared role, with robust template-clustered standard errors.

The result is clear ([Table 3](#A10.T3)): Userness strongly predicts attack success rate even when controlling for template variant, indicating that it tracks a real, predictive latent variable aside even when variation is limited to differences in formatting and whitespacing. This both shows the validity of the probing methodology and the impact of role confusion on attack success.

This confirms that Userness tracks genuine structure, not template quality. The pattern mirrors CoT Forgery: the lever differs (declarations vs. style), but the mechanism is identical. Any signal the model uses to infer role becomes an attack surface. The vulnerability is not in specific patterns—it’s in how role perception itself is implemented.

**Table 3: Logistic regression predicting attack success from Userness, controlling for declared template role. Standard errors clustered by template.**
|  | Estimate | Std. Error | p-value |
| --- | --- | --- | --- |
| Intercept | $-$2.16 | 0.25 | $<$.001*** |
| Userness | 6.01 | 1.30 | $<$.001*** |
| Declared Role: User | 0.84 | 0.37 | .025* |
| Declared Role: Tool | $-$0.64 | 0.32 | .043* |

## Appendix K Systemness and Position

If role confusion is correct, it should extend beyond prompt injection to explain a known puzzle: why do system prompts fail to maintain priority over user instructions?

#### The Puzzle.

Swapping content between system and user tags often leaves behavior unchanged (Wallace et al., 2024; Zhang et al., 2025; Geng et al., 2025). Several explanations have been proposed: attention decay over long contexts (Qin et al., 2024), or position serving as a proxy for priority (Wang et al., 2025b).

Our framework predicts that position plays the same role for Systemness that style plays for other roles: a dominant signal that overrides architectural tags in latent space, controlling model perception. If this hypothesis holds, we should find position driving Systemness geometrically.

#### Methodology.

We reuse our gpt-oss-20b role probes from [Section 4](#S4). Recall these probes were trained with no stylistic or positional confounds (identical token positions across all tag variants). They learned only tag-induced geometry.

We construct test sequences by sampling 200 conversations from Oasst1 and ToxicChat, stripping all tags, and randomly scrambling turn order. We then insert a correctly-tagged system prompt (the model’s canonical “You are ChatGPT…” instruction(^20^2020gpt-oss-20b default system prompt (Kundel, 2025).)) at token position 100.

If tags govern perception, Systemness should spike at position 100. If position governs perception, Systemness should track token index regardless of tags.

#### Results.

[Figure 32](#A11.F32) shows the result. Untagged text at context start approaches 100% Systemness and decays monotonically. The system-tagged region (purple) shows similar Systemness to the untagged tokens preceding it.

Figure: Figure 32: Average Systemness by token position with system prompt at start (left) vs. position 100 (right). Left: Even with system prompt at start, Systemness rapidly decreases with position. Right: When the same text appears mid-sequence, Systemness remains low ( 5%). Untagged text at context start shows far higher Systemness than system-tagged text appearing later.
Refer to caption: https://arxiv.org/html/2603.12277/2603.12277v4/x25.png

The probes were trained with no position-label relationship in their training data—yet they reveal that the *model* learned position as the dominant Systemness signal. Tags and position both write to the same latent role representation; position writes more strongly(^21^2121System prompt training is highly model-specific; we leave cross-model generalization to future work.).

These findings expand on the system prompt/position analysis of Wang et al. (2025b), validating their hypothesis mechanistically. When prior work swapped system and user tags, behavior was unchanged not because the model ignored roles, but because it never perceived the swap.

#### Implications.

Role confusion is not specific to prompt injection—it is a general failure mode of perception. Models infer role from forgeable signals, and any text matching those signals inherits privileged geometry. We have shown style as the dominant signal for most roles, position for System, and declarations as an additional channel. Any forgeable signal becomes an attack surface.