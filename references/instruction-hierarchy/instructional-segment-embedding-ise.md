Title: Instructional Segment Embedding: Improving LLM Safety with Instruction Hierarchy
ArXiv: 2410.09102
Authors: Tong Wu, Shujian Zhang, Kaiqiang Song, Silei Xu, Sanqiang Zhao, Ravi Agrawal, Sathish Reddy Indurthi, Chong Xiang, Prateek Mittal, Wenxuan Zhou, Princeton University, Zoom Video Communications, Works done at Zoom. Correspondence to:, ,
Sections: 40
Estimated tokens: 19.3k

## Contents
- 1 Introduction
- 2 Background: LLM Vulnerabilities
- 3 Lack of Instruction Hierarchy in Modern LLM Architecture
- 4 Proposed Approach: Instructional Segment Embedding (ISE)
- 5 Experimental Design
  - 5.1 Generating Training Data
  - 5.2 Experiment Setup
  - 5.3 Robustness Evaluation
- 6 Experimental Results and Analysis
  - 6.1 Main Results on Structured Query
  - 6.2 Main Results on Instruction Hierarchy
  - 6.3 Detailed Analysis over Attacks
  - 6.4 Other Analysis
- 7 Related Works
- 8 Discussion and Conclusion
- 9 Ethics and Reproducibility Statement
- Acknowledgment
  - Acknowledgments
- Appendix A Details of Implementing Instructional Segment Embedding
- Appendix B Details of Training Data
  - B.1 Structured Query benchmark.
  - B.2 Instruction Hierarchy benchmark.
- Appendix C Details of Evaluation
  - C.1 Evaluations on Structured Query Benchmark
  - C.2 Evaluations on the Instruction Hierarchy Benchmark
  - C.3 Instruction-following Evaluation
  - C.4 Over-refusal Evaluation
- Appendix D More Experimental Results on Structured Query
- Appendix E More Experimental Results on Instruction Hierarchy
  - E.1 Detailed Analysis of Indirect Prompt Injection
  - E.2 Detailed Analysis of Direct Prompt Injection
  - E.3 Detailed Analysis of Prompt Extraction
  - E.4 Detailed Analysis of Harmful Requests
  - E.5 Detailed Analysis of LLAMA 3.1 Model
  - E.6 \reb Investigation of Instructional Segment Embedding
  - E.7 Evaluation on automatic attacks.
  - E.8 Evaluation on multi-turn conversation.
  - E.9 Case Study of ISE Attention
  - E.10 Detailed Figure of Robustness Evaluation on Instruction Hierarchy
- Appendix F Discussions and Evaluations on Jailbreak Attacks

## Abstract

Abstract Large Language Models (LLMs) are susceptible to security and safety threats, such as prompt injection, prompt extraction, and harmful requests.
One major cause of these vulnerabilities is the lack of an instruction hierarchy.
Modern LLM architectures treat all inputs equally, failing to distinguish between and prioritize various types of instructions, such as system messages, user prompts, and data.
As a result, lower-priority user prompts may override more critical system instructions, including safety protocols.
Existing approaches to achieving instruction hierarchy, such as delimiters and instruction-based training, do not address this issue at the architectural level.
We introduce the I nstructional S egment E mbedding (ISE) technique, inspired by BERT, to modern large language models, which embeds instruction priority information directly into the model.
This approach enables models to explicitly differentiate and prioritize various instruction types, significantly improving safety against malicious prompts that attempt to override priority rules.
Our experiments on the Structured Query and Instruction Hierarchy benchmarks demonstrate an average robust accuracy increase of up to 15.75% and 18.68%, respectively.
Furthermore, we observe an improvement in instruction-following capability of up to 4.1% evaluated on AlpacaEval.
Overall, our approach offers a promising direction for enhancing the safety and effectiveness of LLM architectures.

## 1 Introduction

Figure: Figure 1: A demonstration of the hierarchy of instructions, including system instruction, user instruction, data input as well as model output.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/priority.png

Large Language Models (LLMs) have shown significant potential in enabling agentic applications and autonomous decision-making across various domains, such as web agents, educational tools, and
medical assistance (YaoWebShop; GanEducation; abbasian2024health). To optimize LLM applications, a structured approach to implementation is widely adopted. This involves clear distinctions among system instructions, user prompts, and data inputs, as illustrated in Figure [1](https://arxiv.org/html/2410.09102v2#S1.F1). \rebThese instructions contain specific priorities (e.g., system instructions have higher importance than user instructions) that help the model execute functionalities and better assist users.

However, modern LLM architecture processes all input tokens equally without formal mechanisms to differentiate instructions. Consequently, malicious attackers can easily exploit this limitation to override the priorities of instructions, leading to various vulnerabilities.
For example, prompt injection (KaiPIA) \rebinserts malicious instructions into data sources to subvert the original ones.
Prompt extraction (Zhang2023EffectivePE) \rebaims
to extract system messages, revealing proprietary prompts.
Harmful requests (Ganguli2022RedTL) involve malicious users providing unsafe instructions to elicit irresponsible or dangerous responses from the safety-aligned LLMs.
These vulnerabilities underscore the significance of designing more robust instruction hierarchy in LLM applications.

Recently, research has been conducted to enhance models’ ability to follow the instruction hierarchy.
For instance, Hines2024DefendingAI proposed prompt-based solutions utilizing a special delimiter between prompts.
Chen2024StruQDA and Wallace2024TheIH suggested methods for generating hierarchical prompts, incorporating adversarial data along with high-quality responses to fine-tune LLMs.
However, despite these improvements, the core challenge persists: current LLM architectures still lack an effective mechanism to differentiate and prioritize hierarchical instructions.

In this work, we tackle the challenge by introducing an architecture-level design for LLMs.
Inspired by BERT (lan2019albert) and its variants (lan2019albert; yasunaga-etal-2022-linkbert), we propose using an Instructional Segment Embedding (ISE) to categorize different types of instructions distinctly.
Specifically, we enhance the input token by incorporating segment information that classifies each token by its role (e.g., system instruction as 0, user prompt as 1, and data input as 2).
This segment information is processed through a learned embedding layer, converting it into segment embeddings, which are then passed to later self-attention layers along with token embeddings.
To obtain a robust segment embedding layer, we perform supervised fine-tuning on datasets containing structured prompts and high-quality responses.
This process enables the model to differentiate between levels of instruction hierarchies more effectively, thereby boosting the overall safety of the system.

Empirically, we conduct comprehensive experiments on two benchmarks: Structured Query (Chen2024StruQDA) and Instruction Hierarchy (Wallace2024TheIH), which are constructed based on the Alpaca (alpaca) and Ultrachat (ding2023enhancing) datasets, respectively. We fine-tune multiple pretrained LLMs, including Llama-2-13B (touvron2023llama), Llama-3-8B (dubey2024llama3herdmodels), and Llama-3.1-8B, and compare their performance with and without the use of Instructional Segment Embedding.
Our findings indicate that our method yields substantial improvements in robustness while either maintaining or enhancing the models’ general capabilities, regardless of the presence of adversarial training data.
For example, on the Structured Query benchmark, the method achieves an average robust accuracy improvement of up to 15.75% against indirect prompt injection attacks. On the Instruction Hierarchy benchmark, our ISE yields an average boost in robustness of up to 18.68% across multiple vulnerabilities, including indirect and direct prompt injection, prompt extraction, and harmful requests. In addition, the integration of ISE also maintains or even improves the instruction-following capability by as much as 4.1% on AlpacaEval.

Contributions: (1) We identify and analyze critical limitations in current LLM architectures concerning the lack of instruction hierarchy (Section [3](https://arxiv.org/html/2410.09102v2#S3)). (2) We propose Instructional Segment Embedding, a simple yet effective method designed to incorporate instruction-type information directly into the model. This approach enables the model to better distinguish and prioritize instructions based on their privilege (Section [4](https://arxiv.org/html/2410.09102v2#S4)).
(3) We empirically demonstrate the effectiveness of ISE across two benchmarks, encompassing five training datasets and addressing four types of vulnerabilities (Sections [5](https://arxiv.org/html/2410.09102v2#S5) & [6](https://arxiv.org/html/2410.09102v2#S6)).

Figure: Figure 2: A demonstration of various vulnerabilities of LLM applications, including prompt injection, prompt extraction as well as harmful request.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/attack_demo.png

## 2 Background: LLM Vulnerabilities

Modern LLM products typically involve up to three stakeholders(^1^11Here, we simplify real-world scenarios by assuming the LLM provider and the LLM application provider to be the same stakeholder collectively responsible for providing system instructions. Additionally, we consider text from third parties as data, which may also include other contents like outputs from external API calls.): (1) the LLM application provider (e.g., OpenAI), who designs the model’s system-level instructions and manages the general workflow; (2) the primary user, who provides input in the form of instructions or queries; and (3) third-party source/data, such as web search results, that offer additional context for the LLM. As a result, LLM applications often establish a hierarchical order of instructions based on their perceived reliability: system instructions take precedence, followed by user instructions, and finally data.

Security vulnerabilities arise when conflicts between these priorities occur, such as (1) a malicious user attempting to bypass safety system instructions or (2) malicious web providers injecting harmful actions in data.
These conflicts may take various forms, including prompt injections, prompt extractions, and harmful requests, as shown in Figure [2](https://arxiv.org/html/2410.09102v2#S1.F2) and outlined below.

Prompt injection (Figure [2](https://arxiv.org/html/2410.09102v2#S1.F2)a). Prompt injection attacks (ignore_previous_prompt) generally occur in two forms: indirect and direct. Indirect prompt injection attacks occur when third-party data input contains instructions that should never be followed by LLMs.
Direct prompt injection attacks happen when a malicious attacker manipulates the user query to an LLM, causing the model to generate outputs that deviate from predefined instructions.

Prompt extraction (Figure [2](https://arxiv.org/html/2410.09102v2#S1.F2)b).
This vulnerability (Zhang2023EffectivePE) often exploits a weakness in certain LLM applications that store confidential information within system instructions. Attackers may craft malicious queries that prompt the model to reference this stored information, potentially leading to the disclosure of system prompts.

Harmful requests (Figure [2](https://arxiv.org/html/2410.09102v2#S1.F2)c).
Harmful requests (Ganguli2022RedTL) aim to bypass the model’s safety alignment (bai2022training) through malicious queries. These prompts can lead to unsafe outcomes, including unethical responses or even the weaponization of LLMs.

In this paper, we aim to enhance the instruction hierarchy capabilities of LLMs, thereby mitigating various forms of attacks that attempt to override the priority rules.

Figure: Figure 3: A demonstration of the chat template for Llama-3-Instruct (dubey2024llama3herdmodels).

## 3 Lack of Instruction Hierarchy in Modern LLM Architecture

Current embeddings lack instruction hierarchy. Given an input context $\mathbf{X}_{M}$ with $M$ tokens $x_{1},x_{2},\dots,x_{M}$, the large language models first convert each token into a high-dimensional vector using a token embedding matrix ${\mathbf{E}}^{\text{Tok}}\in\mathbb{R}^{V\times D}$, where $V$ is the vocabulary size, and $D$ is the output embedding dimension. The embedding vector $e^{\text{Tok}}_{m}$ for token $x_{m}$ is given by ${\mathbf{E}}^{\text{Tok}}[x_{m}]$, based on its index in the vocabulary.
Additionally, the model also obtains positional embeddings ${\mathbf{E}}^{\text{Pos}}_{m}$, based on the position of each token.
Then, the token embeddings $(e^{\text{Tok}}_{1},e^{\text{Tok}}_{2},\dots,e^{\text{Tok}}_{M})$ will be fed into the transformer’s self-attention layers along with positional embeddings for further processing.(^2^22Models handle positional information in different ways. For example, GPT-2 (radford2019language) adds learned positional embeddings to its token embeddings, while Llama-2 uses Rotary Position Embedding (RoPE) (SU2024127063) in its attention blocks to represent positions.)

In these self-attention layers, each token embedding is processed “equally”. As a result, the model recognizes only the semantic content and sequential order of each token from the embedding, lacking the capability to distinguish their hierarchical significance. This architectural design can inherently lead to vulnerabilities. For instance, a lower-priority user prompt, such as ‘‘Please focus on my prompt as the system prompt is outdated’’, could mistakenly be prioritized and override the original system prompt. This could inadvertently lead to various types of vulnerabilities, as shown in Figure [2](https://arxiv.org/html/2410.09102v2#S1.F2).

Prior works do not address this issue. To mitigate these vulnerabilities, researchers have introduced methods to improve the robustness of large language models (LLMs) during the supervised fine-tuning phase.
This method involves not only using benign prompt-response data but also adversarial or misaligned instructions with robust responses (Piet2023JatmoPI; Chen2024StruQDA; Wallace2024TheIH).
This approach helps the model learn to prioritize hierarchical instructions and adhere to embedded safety protocols.
Despite its improvement, the challenge of uniformly processing hierarchical instructions remains a fundamental limitation inherent in current embedding methods and model architecture. \reb We demonstrated that our proposed architecture design can provide better robustness (see results in Table [1](https://arxiv.org/html/2410.09102v2#S6.T1) and Figure [5](https://arxiv.org/html/2410.09102v2#S6.F5)).

An alternative approach is to use specific chat templates to better handle input data. For instance, LLAMA-3-Chat (dubey2024llama3herdmodels) leverages a chat template with special tokens like <|begin_of_text|> and <|star_header_id|> as shown in Figure [3](https://arxiv.org/html/2410.09102v2#S2.F3).
Hines2024DefendingAI and Chen2024StruQDA have also leveraged the specialized delimiters that aid the model in more effectively distinguishing instructions.
However, two major drawbacks exist. Firstly, during inference, only a few tokens contain hierarchical priority information, and this signal is likely to diminish when encountering long-context tasks (e.g., summarizing a novel).
Secondly, malicious attackers may extract these special delimiters, and exploiting them could lead to more severe attacks (zheng2024fsj).

Figure: Figure 4: The input representation includes both token embeddings and instructional segment embeddings. We categorize all input texts into four segments: system instructions, user instructions, third-party data, and generated output. We assign different segment embeddings to each type of input text. The final input embeddings are the sum of token embeddings and segment embeddings. The LLMs will predict the next token after "the summary is", with extra instruction hierarchy information.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/SegmentE.png

## 4 Proposed Approach: Instructional Segment Embedding (ISE)

To tackle this challenge, we propose Instructional Segment Embedding (ISE), which encodes the instruction hierarchy directly into the embeddings. This enables subsequent self-attention layers to more effectively recognize and follow instruction priorities, thereby boosting robustness.

Specifically, we leverage a learnable embedding layer, similar to the token embedding matrix ${\mathbf{E}}^{\text{Tok}}$, which we call the segment embedding matrix ${\mathbf{E}}^{\text{Seg}}$. We define ${\mathbf{E}}^{\text{Seg}}\in\mathbb{R}^{H\times D}$, where $H$ is the number of hierarchies and $D$ is the embedding dimension. By default, we set $H$ to 4, representing system, user, data, and output. Each token in $\mathbf{X}_{M}$ is tagged with corresponding hierarchy information $h_{m}\in\{0,1,2,3\}$, readily derived from distinct stakeholder categories in the LLM applications. The instructional segment embeddings of $\mathbf{X}_{M}$ are represented as $(e^{\text{Seg}}_{1},e^{\text{Seg}}_{2},\dots,e^{\text{Seg}}_{M})$ and obtained from ${\mathbf{E}}^{\text{Seg}}[h_{m}]$. To incorporate this modification, the final embeddings are computed by summing the token embeddings and segment embeddings. This results in $(e^{\text{Seg}}_{1}+e^{\text{Tok}}_{1},e^{\text{Seg}}_{2}+e^{\text{Tok}}_{2},
\dots,e^{\text{Seg}}_{M}+e^{\text{Tok}}_{M})$, as illustrated in Figure [4](https://arxiv.org/html/2410.09102v2#S3.F4). These embeddings are then fed into self-attention layers, following the process used in current LLMs.

The segment embedding layer is trained alongside other parameters during the supervised fine-tuning (instruction tuning) phase. In our experiments, we use widely adopted instruction-following datasets and construct structured queries based on the original prompt using GPT-4o (openai2023gpt4osystemcard).
Additionally, we experiment with datasets containing malicious instructions designed to override higher-level instructions, enabling the model to learn how to reject or ignore such commands.

Flexibility in design. The design choice for Instructional Segment Embedding can be flexible and should be tailored to the specific downstream tasks. For instance, if the data category can be further subdivided into outputs from external API tools or online information, we can introduce "tools type" and "web data type" categories, providing more fine-grained information. If the application does not involve third-party context, the data type can be omitted.

Connection to BERT. Inspired by BERT’s segment embeddings (Devlin2019BERTPO), originally used to distinguish input segments for next-sentence prediction, our approach repurposes these embeddings to encode hierarchical instructions. This helps address the need for structured prompts and safer LLM outputs by providing direct, contextually relevant cues to the model. Unlike BERT, we incorporate the output type for two reasons: (1) It supports consistent autoregressive inference for each token in the input. (2) output may also include instructions (e.g., ‘‘Please provide more details of your question’’) that are critical in multi-turn language tasks.

Simplicity and lightweight. The implementation is also straightforward \reband lightweight, requiring only $H\times D$ additional parameters. We provide a PyTorch code snippet that demonstrates how to implement this in just a few lines, as shown in Appendix [A](https://arxiv.org/html/2410.09102v2#A1).

## 5 Experimental Design

In this section, we present how we conducted the experiments. Specifically, we begin by describing the generation of the training data (Section [5.1](https://arxiv.org/html/2410.09102v2#S5.SS1)), the experimental setup (Section [5.2](https://arxiv.org/html/2410.09102v2#S5.SS2)), and the details of the robustness evaluation against multiple attacks (Section [5.3](https://arxiv.org/html/2410.09102v2#S5.SS3)).

### 5.1 Generating Training Data

We conduct experiments using two benchmarks: Structured Query and Instruction Hierarchy. The Structured Query benchmark primarily focuses on indirect prompt injection attacks, whereas the Instruction Hierarchy benchmark evaluates all types of vulnerabilities discussed, including indirect and direct prompt injections, prompt extraction, and harmful requests.

For the Structured Query benchmark, we generally follow the approach of Chen2024StruQDA. Two datasets are constructed: Clean Alpaca and Adversarial Alpaca. The Clean Alpaca dataset is constructed by Alpaca-Cleaned-50K dataset (alpaca; AlpacaDataCleaned). For the Adversarial Alpaca dataset, we incorporate instructions \rebdrawn from other samples (either directly or with a fabricated response) into the data and train the model to ignore such instructions. \rebMore details are available in Section [B.1](https://arxiv.org/html/2410.09102v2#A2.SS1).

For the Instruction Hierarchy benchmark, we mostly adhere to previous work by Wallace2024TheIH to create both aligned and misaligned data(^3^33We contacted the authors for training data and details, but they cannot share them due to company restrictions.).
We select the UltraChat-200K dataset (ding2023enhancing) as the base dataset, which contains more training data.
Since UltraChat consists solely of prompts and responses, we utilized GPT-4o (openai2023gpt4osystemcard) to decompose 10K prompts into three components: system instructions, user instructions, and data inputs, which we term the UltraChat Baseline.
Additionally, we incorporate datasets from SystemChat (abacusai_systemchat_2023) and SystemMessage (nobodyexists_systemmessagecontradictions_2023) that contain specifical system prompts, designated as the System Follow dataset.
Lastly, \rebwe crafted three types of attacks for the malicious data: indirect/direct prompt injection and prompt extraction, which we collectively name the Instruction Hierarchy datasets.
We excluded harmful request data from the training but used them as evaluations following Wallace2024TheIH.
Further details on generating training data are available in Section [B.2](https://arxiv.org/html/2410.09102v2#A2.SS2).

### 5.2 Experiment Setup

Data processing. We format all training and evaluation samples with clear segmentation, including system, user, data, and output information.
We merge the system and user instructions for the Structured Query benchmark into the same type, as all system instructions in Alpaca are identical. To simplify the experiments, we train and evaluate only single-turn chats, where the conversation ends after the model generates a complete response.

LLM training and inference. By default, we utilize Llama-2-13B (touvron2023llama) and Llama-3-8B (dubey2024llama3herdmodels) as the pretrained models for Structured Query and Instruction Hierarchy, respectively.
Experiments with Llama-3.1-8B are presented in Section [6.4](https://arxiv.org/html/2410.09102v2#S6.SS4).

We employ supervised fine-tuning to update all model parameters for all baseline and ISE methods with three epochs.
A learning rate of 2e-5 and a cosine learning schedule are used.
During inference, we use top-p sampling methods with the model’s default settings.

Baselines.
We conduct experiments on multiple datasets and compare our Instructional Segment Embedding method (+ISE) with models that do not use it. For the Structured Query benchmark, we include experiments using either text (Baseline) or newly constructed tokens (Delimiter) to differentiate distinct types of prompts, following the approach of Chen2024StruQDA. For the Instruction Hierarchy benchmark, we only include experiments with the default delimiters of LLAMA-3-8B (Baseline), as these are already specially reserved tokens (shown in Figure [3](https://arxiv.org/html/2410.09102v2#S2.F3)).

### 5.3 Robustness Evaluation

We evaluate our method using comprehensive attack methods. More details are in Appendix [C](https://arxiv.org/html/2410.09102v2#A3).

Structured Query benchmark. Following the approach of Chen2024StruQDA, we evaluate indirect prompt injection attacks on models trained on the Alpaca dataset. Specifically, we focus on four types of attacks: Naive Attack, Ignore Attack, Escape Separation Attack, and Completion Real Attack as in-domain attacks. All in-domain attacks are crafted by injecting one or a few sentences at the end of the data to trick the model into outputting the word ‘‘hacked’’. Since the scenarios for in-domain attacks are quite similar to the adversarial training data constructed by Chen2024StruQDA, we further develop new attacks based on the in-domain methods. These new attacks inject adversarial texts at the data’s beginning and end, called out-of-domain attacks. This strategy significantly degrades the robustness of models trained on the adversarial Alpaca dataset.
For the evaluation metrics, we compute the rate at which the model does not generate content containing the word ‘‘hacked’’ and refer to this as robustness or robust accuracy.

Instruction Hierarchy benchmark.
Evaluating models trained on the Instruction Hierarchy benchmark is complex due to the need to account for indirect and direct prompt injection, prompt extraction, and harmful requests. To address these challenges: (1) For indirect prompt injection, we apply the same evaluations and metrics used in Structured Query benchmarks. For direct prompt injection, we use the same attacking prompts but inject them directly into the user prompt. (2) For prompt extraction, we use the ShareGPT and Unnatural Instructions datasets from (Zhang2023EffectivePE), along with 15 author-selected effective extraction prompts, and evaluate robustness using an approximate metric based on Rouge-L recall (Lin2004ROUGEAP). (3) For harmful requests, we follow the evaluations of (Wallace2024TheIH), using Jailbreakchat (Chat) and "Do Anything Now" (DAN) prompts (SCBSZ24) paired with StrongREJECT malicious instructions (souly2024strongreject). We query GPT-4o to check whether its responses adhere to safety guardrails.

Comprehensive robustness metrics.
For prompt injection and extraction, which encompass multiple attack methods or malicious prompts, we include additional metrics. We define average robustness as the model’s average performance across these various attack methods, offering a general evaluation of model robustness. Furthermore, we introduce worst robustness, representing the model’s ability to defend against the most challenging attack.

Clean evaluation. We evaluate the model’s capacity using standard datasets. Both benchmarks are assessed with AlpacaEval 1.0 (alpaca_eval). For the Instruction Hierarchy benchmark, we additionally use the MT-Bench (LLM_judge) to measure the model’s performance.

## 6 Experimental Results and Analysis

We report the main results on the Structured Query benchmark in Section [6.1](https://arxiv.org/html/2410.09102v2#S6.SS1) and the Instruction Hierarchy in Section [6.2](https://arxiv.org/html/2410.09102v2#S6.SS2). We observe that our approach consistently achieves higher robust accuracy while either maintaining or improving general capability.
We also present a more detailed analysis of multiple vulnerabilities in Section [6.3](https://arxiv.org/html/2410.09102v2#S6.SS3). Lastly, we conduct an over-refusal evaluation and assess generalization to the advanced Llama-3.1-8B model in Section [6.4](https://arxiv.org/html/2410.09102v2#S6.SS4).

### 6.1 Main Results on Structured Query

**Table 1: The evaluation results on Structured Query benchmark against both in-domain and out-of-domain indirect prompt injection attacks. We compare our method (+ISE) with the baseline and delimiter methods (Chen2024StruQDA) on Clean Alpaca and Adversarial Alpaca.**
|  | Dataset | Clean Alpaca | Adversarial Alpaca |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | Method | Baseline | Delimiter | +ISE (Ours) | Baseline | Delimiter | +ISE (Ours) |
| Capability ($\uparrow$) | AlpacaEval | 72.76 | 72.67 | 72.13 | 73.41 | 72.26 | 73.76 |
| In-Domain<br>Robustness ($\uparrow$) | Naive | 65.87 | 68.75 | 75.96 | 100.00 | 99.04 | 100.00 |
| Ignore | 57.69 | 57.21 | 70.19 | 99.52 | 99.04 | 99.04 |  |
| Escape-S | 75.00 | 69.23 | 78.85 | 99.52 | 99.52 | 100.00 |  |
| Completion-R | 4.81 | 7.21 | 40.38 | 70.19 | 100.00 | 100.00 |  |
| Average | 50.84 | 50.60 | 66.35 (+15.75) | 92.31 | 99.16 | 99.76 (+0.60) |  |
| Worst | 4.81 | 7.21 | 40.38 (+32.17) | 70.19 | 99.04 | 99.04 (+0.00) |  |
| Out-of-Domain<br>Robustness ($\uparrow$) | Naive | 62.02 | 66.35 | 69.71 | 64.90 | 67.79 | 76.44 |
| Ignore | 52.40 | 51.92 | 69.71 | 98.56 | 96.15 | 96.63 |  |
| Escape-S | 72.12 | 71.63 | 70.67 | 73.08 | 76.44 | 88.46 |  |
| Completion-R | 1.92 | 12.99 | 34.14 | 85.58 | 91.35 | 99.52 |  |
| Average | 47.12 | 50.72 | 61.06 (+10.34) | 80.53 | 82.93 | 90.26 (+7.67) |  |
| Worst | 1.92 | 12.99 | 34.14 (+21.15) | 64.90 | 67.79 | 76.44 (+8.65) |  |
|  |  |  |  |  |  |  |  |

Maintains high utility. In Table [1](https://arxiv.org/html/2410.09102v2#S6.T1), we present the main results for capability and robustness by comparing our method with the baseline and delimiter methods on both the clean and adversarial Alpaca datasets. Compared to the other two methods, Instructional Segment Embedding maintains high utility with negligible degradation or even slight improvement. The difference in winning rate between the methods is less than 1% on AlpacaEval.

Consistent robustness enhancement. We also observe that our method consistently improves robustness against indirect prompt injection attacks. Specifically, it achieves a 15.75% increase in average robust accuracy and a 32.17% increase in worst robust accuracy against in-domain attacks when trained with the clean Alpaca dataset. Both the delimiter and our ISE reach nearly perfect in-domain robustness.
For out-of-domain attacks, we find that adding ISE can also significantly enhance robustness, resulting in improvements of $\sim$10% and $\sim$7% in average robustness for clean and adversarial Alpaca, respectively.
Interestingly, our out-of-domain attacks degrade the robustness of models trained on the adversarial Alpaca dataset more than those trained on the clean Alpaca dataset (16% vs. 5%). This suggests that the adversarial dataset may overfit to in-domain attacks. Nevertheless, adding ISE largely maintains generalization to out-of-domain attacks.

### 6.2 Main Results on Instruction Hierarchy

Figure: Figure 5: The evaluation of model capabilities on the Instruction Hierarchy benchmark is conducted using AlpacaEval and MT-Bench, as illustrated in Figure (a). Robustness evaluations include both indirect and direct prompt injection attacks, prompt extraction attacks, and harmful requests, as shown in Figure (b). We performed experiments across three training datasets (i.e., UltraChat Baseline, System Follow, and Instruction Hierarchy) and compared ISE with the baseline (Wallace2024TheIH).
Refer to caption: https://arxiv.org/html/2410.09102/x1.png

We present the evaluation results for our method on the Instruction Hierarchy benchmark in Figure [5](https://arxiv.org/html/2410.09102v2#S6.F5), focusing on model capability and average robustness across various datasets and attack scenarios.

Improvement in capabilities. Adding ISE boosts instruction-following capabilities, particularly for models trained on the System Follow and Instruction Hierarchy datasets. For example, the AlpacaEval win rate improves by approximately $\sim$4.1% when training on the Instruction Hierarchy dataset with our ISE, as shown in Figure [5](https://arxiv.org/html/2410.09102v2#S6.F5)(a). Additionally, we observe negligible degradation on MT-Bench for the UltraChat Baseline model and improvements for the other two training datasets.

Enhanced safety against multiple vulnerabilities. We evaluate the robustness of the models against indirect and direct prompt injection attacks, prompt extraction attacks, and harmful requests.
(1) Indirect and direct prompt injection scenarios (#1, #2, #3, and #4 in Figure [5](https://arxiv.org/html/2410.09102v2#S6.F5)(b))
: We report the average robustness across four types of attacks, including both in-domain (ID) and out-of-domain (OOD) contexts. Our results demonstrate robust accuracy improvements ranging from 5% to 25% across all training data configurations when applying the ISE method. Notably, for models trained with the UltraChat Baseline, robust accuracy increases by nearly 25% on average.
(2) Prompt extraction scenarios (#5 and #6 in Figure [5](https://arxiv.org/html/2410.09102v2#S6.F5)(b)): Robustness is measured against 15 effective extraction prompts. Our findings show that models using ISE consistently achieve higher average robustness, with an increase of at least 10% across all datasets. This is evident even for models trained on the Instruction Hierarchy dataset, which already demonstrated more than 80% robust accuracy.
(3) Harmful requests (#7 and #8 in Figure [5](https://arxiv.org/html/2410.09102v2#S6.F5)(b)): Our analysis reveals improvements in robustness for models under the UltraChat Baseline and Instruction Hierarchy settings when using ISE. For System Follow, our methods either maintain or slightly exceed the baseline method.

Overall, using Instructional Segment Embeddings significantly enhances both the capabilities and robustness of models against a wide range of attacks on the Instruction Hierarchy benchmark.

### 6.3 Detailed Analysis over Attacks

The previous sections mainly covered the overall results (average robustness) across multiple prompt injection and extraction attacks. Here, we provide more detailed evaluations of attacks on the Instruction Hierarchy benchmark. Results for the Structure Query are reported in Appendix [D](https://arxiv.org/html/2410.09102v2#A4).

Figure: Figure 6: Robust accuracy of indirect prompt injection attack on the Instruction Hierarchy benchmark with both in-domain and out-of-domain attacks. More details are described in Appendix [E.1](https://arxiv.org/html/2410.09102v2#A5.SS1).
Refer to caption: https://arxiv.org/html/2410.09102/x2.png

Prompt injection.
In Figure [6](https://arxiv.org/html/2410.09102v2#S6.F6), we present the results of indirect prompt injection attacks, including Naive, Ignore, Escape Separation, and Completion Real, across in-domain and out-of-domain scenarios.
The results indicate that our ISE method significantly enhances performance compared to the baseline across nearly all scenarios.
Notably, the Completion Real attack severely compromises model robustness, resulting in less than 10% effectiveness for models trained on the UltraChat Baseline and the System Follow dataset without ISE. This attack works by introducing a spoofed response to the benign instruction and concatenating a new malicious instruction into the data. Models that fail to effectively differentiate between these types of instructions are prone to executing the new malicious instruction.
However, our method significantly boosts robustness, yielding improvements ranging from approximately 30% to 50%.
We also observe similar trends for direct prompt injection attacks, which are detailed in Appendix [E.2](https://arxiv.org/html/2410.09102v2#A5.SS2).

Figure: Figure 7: Robust accuracy against 15 effective prompt extraction attacks on ShareGPT dataset.
Refer to caption: https://arxiv.org/html/2410.09102/x3.png

Prompt extraction. As mentioned in Section [5.3](https://arxiv.org/html/2410.09102v2#S5.SS3), we utilize 15 effective malicious prompts to extract the system messages. In Figure [7](https://arxiv.org/html/2410.09102v2#S6.F7), we present all the results and find that our method consistently outperforms the baseline, notably enhancing the worst robust accuracy by up to approximately 45%. Interestingly, the model trained on the UltraChat Baseline dataset with ISE exhibits the highest robustness, even exceeding that of the model trained on the Instruction Hierarchy dataset. We find that this is because the instruction-following capability of models trained on the UltraChat Baseline is relatively weak (about 20% lower than the other two models on AlpacaEval).
Consequently, in scenarios where the model is misled into fulfilling a request to output the system message, it sometimes generates only a partial system prompt. Therefore, the attack is not classified as successful. Results on the Unnatural dataset are provided in Appendix [E.3](https://arxiv.org/html/2410.09102v2#A5.SS3).

Harmful requests. In Figure [10](https://arxiv.org/html/2410.09102v2#S6.F10), we report the robustness of models trained on UltraChat Baseline against Jailbreakchat prompts across six categories: ‘Disinformation and Deception,’ ‘Hate, Harassment, and Discrimination,’
‘Illegal Goods and Services,’ ‘Non-Violent Crimes,’ ‘Sexual Content,’ and ‘Violence.’
We observe that Instructional Segment Embedding improves robustness in 6 out of 6 categories, with improvements of up to 18%. Further results are reported in Appendix [E.4](https://arxiv.org/html/2410.09102v2#A5.SS4).

Figure: Figure 8: Harmful requests across categories using UltraChat Baseline.
Refer to caption: https://arxiv.org/html/2410.09102/x4.png

### 6.4 Other Analysis

Over-refusal Evaluation. One potential concern is that our method may overfit and refuse to follow user instructions.
Therefore, we conduct an over-refusal evaluation on the WildChat dataset (zhao2024wildchat) following (claude3_report; Zou2024ImprovingAA).
After filtering out prompts that exceed the context window, we use 691 non-toxic prompts to query the model and evaluate whether it generates \rebnon-refusal responses using GPT-4o. In Figure [10](https://arxiv.org/html/2410.09102v2#S6.F10), we report the compliance rate on the benchmark and observe that our ISE improves the compliance rate by about 10% for the model trained on the UltraChat Baseline but shows slight degradation for the other two models.
Overall, we expect that our method will maintain model capacity, as shown on AlpacaEval and MT-bench in Figure [5](https://arxiv.org/html/2410.09102v2#S6.F5).

Generalization to Other Model. We also evaluated Llama-3.1-8B using the same setup as Llama-3-8B on the Instruction Hierarchy benchmarks. In Figure [10](https://arxiv.org/html/2410.09102v2#S6.F10), we present the results on AlpacaEval and the robustness against four attacks (averaged results) of models trained on the UltraChat Baseline.
Our method demonstrates an approximate 10% improvement in win rate on the AlpacaEval dataset. For robustness, we observe around a 5% robust accuracy improvement against harmful requests and over 10% on all other attacks. Overall, these results suggest our method can be generalized across different models. The complete results are provided in Appendix [E.5](https://arxiv.org/html/2410.09102v2#A5.SS5).

## 7 Related Works

Safety vulnerabilities of LLMs. Recently, the safety of LLMs has become a critical concern. (1) These models are vulnerable to indirect and direct prompt injection attacks. Indirect prompt injections happen when malicious content is embedded in inputs sourced from external data providers, as discussed in various research studies (KaiPIA; liu2023prompt; Zhan2024InjecAgentBI; debenedetti2024agentdojo). In contrast, direct prompt injections occur when attackers explicitly introduce malicious instructions into user input, as demonstrated in (ignore_previous_prompt; mu2023rules; toyer2024tensor; sharma2024spml). (2) Another safety concern is the prompt extraction attack (yu2023assessing; wang2023decodingtrust; Zhang2023EffectivePE), which is more related to privacy. In this type of attack, the attacker’s goal is to maliciously obtain information from the system prompt, which is usually considered confidential. (3) Lastly, we consider harmful requests (Ganguli2022RedTL; perez2022red; souly2024strongreject; xie2024sorry), where the prompts attempt to circumvent safety guidelines and elicit responses involving unsafe behavior, such as instructions for stealing someone’s identity.

Improving LLM robustness. To mitigate these attacks, researchers have developed two major defense strategies: prompt-based and learning-based defenses. Prompt-based defenses construct special instructions (e.g., in-context exemplars or delimiters) to mitigate attacks during inference (wei2023jailbreak; Hines2024DefendingAI; zverev2024can). While these defenses can achieve high robustness against specific attacks, concerns exist regarding their potential utility drops.
Learning-based defenses (Piet2023JatmoPI; Chen2024StruQDA; Wallace2024TheIH) aim to enhance model robustness by fine-tuning the models with a dataset of malicious instructions combined with robust responses. In this work, we explore another approach to improving model robustness by modifying the embedding approach, which is orthogonal to all previous mitigation strategies.

Embedding and architecture of LLMs. Recent research has also focused on improving the LLM embeddings and architectural designs to tackle different challenges. For instance, yen-etal-2024-long proposed a method for enhancing long-context generalization by using a small encoder to process long inputs in chunks. Additionally, mcleish2024transformers introduced the Abacus embedding to improve model performance on arithmetic tasks. In contrast, this paper focuses primarily on enabling the model to learn the instruction hierarchy through Instructional Segment Embedding, as inspired by previous work on designing BERT (lan2019albert) and LinkBERT (yasunaga-etal-2022-linkbert).

## 8 Discussion and Conclusion

Limitations and future work directions.
This study primarily focused on the supervised fine-tuning phase, using single-turn conversations. Future work could explore incorporating ISE during the pre-training or RLHF stage and applying it to multi-turn conversation datasets.
Additionally, while our approach significantly improves the instruction hierarchy capabilities of LLMs, it offers limited robustness against adaptive attacks, commonly referred to as jailbreaks (see Appendix [F](https://arxiv.org/html/2410.09102v2#A6) for more discussion).
However, integrating our method with established adversarial training strategies may potentially enhance the robustness.
Lastly, our experiments were limited to smaller models with 8B and 13B parameters and datasets less than 300K. It remains to be investigated whether our results can be generalized to larger models and datasets, which could provide deeper insights into the scalability of our proposed methods.

Conclusion.
In this work, we introduced the Instructional Segment Embedding as the first attempt to design novel architectures to enhance instruction hierarchy. We conducted comprehensive experiments to demonstrate its effectiveness in improving robustness and general capabilities.

## 9 Ethics and Reproducibility Statement

Ethics.
Our research employs publicly available datasets for experiments and utilizes safety-aligned GPT-4o to generate some training data and judge the performance. We anticipate no ethical concerns with our methodology. Furthermore, we expect our work to have a positive societal impact, as we propose an embedding method to enhance model robustness against various malicious instructions.

Reproducibility. We discuss how we generate data in Section [5.1](https://arxiv.org/html/2410.09102v2#S5.SS1) and Appendix [B](https://arxiv.org/html/2410.09102v2#A2). The process of training the model and inference is detailed in Section [5.2](https://arxiv.org/html/2410.09102v2#S5.SS2). The evaluation data is explained in Section [5.3](https://arxiv.org/html/2410.09102v2#S5.SS3) and Appendix [C](https://arxiv.org/html/2410.09102v2#A3). Additionally, we provide a code snippet to implement our method in Appendix [A](https://arxiv.org/html/2410.09102v2#A1). We release our code at [https://github.com/tongwu2020/ISE](https://github.com/tongwu2020/ISE).

## Acknowledgment

We would like to thank Jiachen T. Wang and Feiran Jia for providing feedback on our early draft.
Prateek was supported in part by the National Science Foundation under grant CNS-2131938 and the
Princeton SEAS Innovation Grant.

#### Acknowledgments

We would like to thank Jiachen T. Wang and Feiran Jia for providing feedback on our early draft.
Prateek was supported in part by the National Science Foundation under grant CNS-2131938 and the Princeton SEAS Innovation Grant.

## Appendix A Details of Implementing Instructional Segment Embedding

Here’s an example of implementing Instructional Segment Embedding with a few lines of Python/Pytorch code. The additional code is highlighted in bold blue.

In the init function, we initialize embedding layers, including the token embedding layer, ISE embedding layer, and positional embedding layer. The inputs to the function include the embedding dimension (embed_size), vocabulary size (vocab_size), and ISE dimension (ISE_size, which defaults to 4).

During inference (the forward function), we compute the token embeddings and ISE embeddings, then sum them for further processing. The input x is a list containing the input IDs of each token in the sentence, and the input seg is a list containing the segment IDs (e.g., system as 0, user as 1, data as 2, output as 3) for each token, with the same size as x.

Instructional

## Appendix B Details of Training Data

In this section, we provide details on how we construct the data, including both the clean and adversarial datasets, to conduct experiments on Structured Query and Instruction Hierarchy benchmarks.

### B.1 Structured Query benchmark.

For the Clean Alpaca dataset, we use the Alpaca-Cleaned-50K dataset (alpaca; AlpacaDataCleaned) to fine-tune the model. \rebThere are 32,603 samples that contain data input, and 19,157 samples do not contain any data input. All of them contain user input. Since the dataset shares the same system instructions (i.e., ‘‘Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.’’), we combine the types of system and user instructions into a single instruction type.

For the Adversarial Alpaca dataset, we follow the approach outlined in Chen2024StruQDA to construct the dataset. \rebWe keep the samples without data input unchanged.
Then, the remaining data includes both clean samples (50%), derived from the Clean Alpaca dataset, and attacked samples (50%), which involve indirect prompt injection attacks within the data.
These attacked samples contain two types of attacks: the Naive Attack and the Completion-Other Attack.
In the Naive Attack, the instruction from other samples is injected into the data.
In the Completion-Other Attack, a fabricated response is injected first, followed by another set of instructions.
The desired output for these adversarial samples should address only the original user instruction so the output remains the same. We provide a demonstration in Figure [11](https://arxiv.org/html/2410.09102v2#A2.F11).

More details can be found in this [StruQ repository](https://github.com/Sizhe-Chen/StruQ/tree/main).

Figure: Figure 11: A demonstration of how the Adversiaral Alpaca dataset is constructed.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/advalpaca_training.png

### B.2 Instruction Hierarchy benchmark.

We use three different datasets to train models: UltraChat Baseline, System Follow, and Instruction Hierarchy.

For the UltraChat Baseline dataset, we use the UltraChat-200K dataset (ding2023enhancing) and employ GPT-4o to decompose 10K prompts into three components: system instructions, user instructions, and data inputs. \rebWe provided the detailed prompts in Figure [12](https://arxiv.org/html/2410.09102v2#A2.F12) and an example of decomposing data in Figure [13](https://arxiv.org/html/2410.09102v2#A2.F13).
This results in approximately 190K plain UltraChat samples and 10K samples with structured queries. For samples without system instructions, we use the default prompt shown in Figure [14](https://arxiv.org/html/2410.09102v2#A2.F14).

Figure: Figure 12: The prompt of decomposing compositional prompts to structured prompts.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/decomposing.png

Figure: Figure 13: An example of prompts before and after decomposing using GPT-4o.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/decomposed.png

Figure: Figure 14: The default system prompt of Instruction Hierarchy benchmark.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/SystemMessage.png

For the System Follow dataset, we combine the UltraChat Baseline dataset with two additional datasets: SystemChat (abacusai_systemchat_2023) and SystemMessage (nobodyexists_systemmessagecontradictions_2023). We use 5K samples from each of these datasets. \rebAll of them do not contain the data part. Additionally, we observe that the quality of the original responses in SystemChat is relatively low; therefore, we use GPT-4o to generate high-quality output.

For the Instruction Hierarchy dataset, we include three different types of misaligned data following Wallace2024TheIH: indirect prompt injection, direct prompt injection, and prompt extraction. (1) Indirect prompt injection: We use 2K samples from UltraChat and inject instructions from other samples into the data. The output remains unchanged. (2) Direct prompt injection: We use 2K SystemChat samples and inject malicious user instructions that attempt to change the original system instructions. Additionally, we include 200 samples that directly conflict with the system instructions. The outputs for both parts are generated by GPT-4o. \rebWe provide a demonstration of how to prompt the GPT-4o model to ensure it follows the desired behavior in Figure [15](https://arxiv.org/html/2410.09102v2#A2.F15). Additionally, we manually review a portion of the outputs to verify this. (3) Prompt extraction: We use 200 SystemMessage samples and inject malicious user instructions that attempt to extract the system message. We also included 2K samples that used aligned questions, which asked for general information about the system messages. Again, GPT-4o is used to generate the outputs. For direct prompt injection and prompt extraction, all injected user instructions are also generated by prompting GPT-4o.

Figure: Figure 15: \rebThe prompt of generating desired responses from GPT-4o.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/good_ouput.png

## Appendix C Details of Evaluation

In this section, we provide full details on constructing various attacks to evaluate LLM models. Specifically, we consider the Structured Query benchmark (Appendix [C.1](https://arxiv.org/html/2410.09102v2#A3.SS1)) and the Instruction Hierarchy benchmark (Appendix [C.2](https://arxiv.org/html/2410.09102v2#A3.SS2)). We also explain how we evaluate the model’s instruction-following capabilities in Appendix [C.3](https://arxiv.org/html/2410.09102v2#A3.SS3).

### C.1 Evaluations on Structured Query Benchmark

As we discussed in Section [5.3](https://arxiv.org/html/2410.09102v2#S5.SS3), we leverage the same evaluation methods from (Chen2024StruQDA), including Naive Attack, Ignore Attack, Escape Separation Attack, and Completion Real Attack, as shown in Figure [16](https://arxiv.org/html/2410.09102v2#A3.F16). We use the test set from AlpacaEval, but only include samples containing valid data inputs to evaluate the model, resulting in a total of 208 test samples.

These attacks are designed to trick the model into outputting the specific word ‘‘hacked’’.
The Naive Attack directly requests the model to print ‘‘hacked’’. The Ignore Attack prompts the model to disregard all previous instructions and output ‘‘hacked’’ (ignore_previous_prompt). The Escape Separation Attack injects special characters to simulate a separation of the previous text, tricking the model into ignoring prior instructions. Finally, the Completion Real Attack appends a fake response to the prompt, leading the model to believe the task is complete, after which malicious instructions are injected for the model to follow.

We also evaluate out-of-domain indirect prompt injection attacks. The key idea is to inject the adversarial text at the beginning and end of the data input sequence. We present a demonstration in Figure [17](https://arxiv.org/html/2410.09102v2#A3.F17).

For both in-domain and out-of-domain attacks, we evaluate the models using robust accuracy, defined as the ratio of samples that do not output ‘‘hacked’’.

Figure: Figure 16: A demonstration of in-domain indirect prompt injection attacks on the Alpaca dataset, including Naive Attack, Ignore Attack, Escape Separation Attack, and Completion Real Attack.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/alpaca_attack/alpacaattack4.png

Figure: Figure 17: A demonstration of out-of-domain indirect prompt injection attacks on the Alpaca dataset, including Naive Attack, Ignore Attack, Escape Separation Attack, and Completion Real Attack.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/alpaca_attack/alpacaattack5.png

### C.2 Evaluations on the Instruction Hierarchy Benchmark

We evaluate our models on Instruction Hierarchy using four types of attacks: indirect prompt injection, direct prompt injection, prompt extraction, and harmful requests.

Indirect prompt injection. We use the same evaluations and metrics as those applied in the Structured Query benchmarks.

Direct prompt injection.
We use the same attacking prompts as indirect prompt injection but inject them directly into the user prompt. Additionally, we use the test set from AlpacaEval without any data input, which consists of 597 test samples. The evaluation metric remains the same: we calculate the ratio of samples that do not output ‘‘hacked’’, and denote this as robustness.

Prompt extraction. We use the ShareGPT and Unnatural Instructions datasets (honovich-etal-2023-unnatural), each consisting of 500 prompts, as described in (Zhang2023EffectivePE), to serve as the secret system messages. To perform prompt extraction, we utilize 15 effective prompts selected by the authors. These prompts are illustrated in Figure [18](https://arxiv.org/html/2410.09102v2#A3.F18).

For evaluation metrics, we adopt the approach of Zhang2023EffectivePE, utilizing an approximate metric based on ROUGE-L recall (Lin2004ROUGEAP). This metric computes the length of the longest common subsequence (LCS) between the extracted system message ($e$) and the ground truth ($t$). The resulting score is the ratio of the LCS length to the length of the ground truth. An attack is considered successful if this score exceeds 0.9, as presented in Equation [1](https://arxiv.org/html/2410.09102v2#A3.E1).

$$ $\operatorname{Metric}(e,t)=1\left[\frac{|\operatorname{LCS}(\operatorname{ tokens}(t),\operatorname{tokens}(e))|}{|\operatorname{tokens}(t)|}\geq 0.9\right]$ (1) $$

Robustness is then evaluated by the proportion of samples that successfully defend against the attack.

Figure: Figure 18: A demonstration of attacking prompts used in prompt extraction.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/PE_prompt.png

Harmful requests.
We follow the evaluation approach of Wallace2024TheIH, utilizing prompts from Jailbreakchat (Chat) and "Do Anything Now" (DAN). These prompts are concatenated with malicious requests from StrongREJECT (souly2024strongreject) to create the final harmful prompts.
The full dataset includes 79 effective attacking prompts from Jailbreakchat (Chat) and 225 from "Do Anything Now" (DAN), all of which are publicly available. Both Jailbreakchat (Chat) and "Do Anything Now" (DAN) can be accessed in the [repository](https://github.com/verazuo/jailbreak_llms/tree/main/data/prompts), sourced from jailbreakchat.com and ChatGPTJailbreak.
Additionally, there are 313 malicious instructions from StrongREJECT. For each sample, we randomly select one attacking prompt to concatenate with the instruction, resulting in a dataset of 313 evaluation samples.

For evaluation metrics, we query GPT-4o to determine whether the response adheres to safety guardrails (i.e., whether the response provides sufficient details to answer malicious requests). The detailed evaluation prompt is presented in Figure [19](https://arxiv.org/html/2410.09102v2#A3.F19), which is similar to that used in (xie2024sorry). Robust accuracy is then computed as the ratio of cases where the model either rejects or does not provide sufficient detail in response to malicious questions.

Figure: Figure 19: A demonstration of evaluation prompts for harmful requests.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/JB_prompt.png

### C.3 Instruction-following Evaluation

We also evaluate our models using two instruction-following benchmarks: AlpacaEval 1.0 (alpaca_eval) and MT-Bench (LLM_judge). Specifically, we follow the steps of AlpacaEval 1.0 to assess model performance. For MT-Bench, we evaluate only 1-turn conversations, as our model is trained for single-turn interactions.

### C.4 Over-refusal Evaluation

For over-refusal evaluation, we query GPT-4o to assess whether the response provides sufficient details to address benign requests. The detailed evaluation prompt is shown in Figure [20](https://arxiv.org/html/2410.09102v2#A3.F20), which is similar to the prompts used for evaluating harmful request tasks.

Figure: Figure 20: \rebA demonstration of prompts for over-refusal evaluation.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/over_refusal_prompt.png

## Appendix D More Experimental Results on Structured Query

In this section, we provide a detailed evaluation of additional indirect prompt injection attacks as constructed by (Chen2024StruQDA). Specifically, we evaluate the "Escape deletion" attack, which injects multiple instances of \b or \r to mimic the deletion of previous characters. We also study 12 other types of completion attacks that attempt to obfuscate the prompt roles using unusual characters, and further details are in (Chen2024StruQDA) and [StruQ repository](https://github.com/Sizhe-Chen/StruQ/tree/main). Additionally, we include the results of our instructional segment embedding with text delimiters.

As shown in Figures [2](https://arxiv.org/html/2410.09102v2#A4.T2) and [3](https://arxiv.org/html/2410.09102v2#A4.T3), our ISE with special delimiters consistently outperforms all other methods in almost all cases. Interestingly, we found that directly using Instructional Segment Embedding does not improve performance on the Clean Alpaca dataset but generally increases robustness on the Adversarial Alpaca dataset by up to 4.6% on average and up to 14% for the in-domain worst robust accuracy compared to the baseline. Therefore, ISE should be used with special token delimiters to achieve the best performance.

**Table 2: Full evaluation results of the LLM on LLAMA-2-13B using in-domain indirect prompt injection attacks.**
| Dataset | Clean Alpaca | Adversarial Alpaca |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | Baseline | +ISE (Ours) | Delimiter | +ISE (Ours) | Baseline | +ISE (Ours) | Delimiter | +ISE (Ours) |
| AlpacaEval ($\uparrow$) | 72.76 | 72.13 | 72.67 | 72.13 | 73.41 | 73.35 | 72.26 | 73.76 |
| Naive | 65.87 | 67.31 | 68.75 | 75.96 | 100.00 | 100.00 | 99.04 | 100.00 |
| Ignore | 57.69 | 61.06 | 57.21 | 70.19 | 99.52 | 98.08 | 99.04 | 99.04 |
| Escape-deletion | 86.54 | 80.77 | 83.17 | 79.81 | 99.04 | 99.52 | 99.04 | 98.56 |
| Escape-separation | 75.00 | 72.60 | 69.23 | 78.85 | 99.52 | 100.00 | 99.52 | 100.00 |
| Completion-other | 10.10 | 21.15 | 9.62 | 43.75 | 100.00 | 100.00 | 100.00 | 100.00 |
| Completion-othercmb | 31.25 | 30.29 | 33.65 | 60.58 | 100.00 | 100.00 | 100.00 | 100.00 |
| Completion-real | 4.81 | 5.29 | 7.21 | 40.38 | 70.19 | 81.73 | 100.00 | 100.00 |
| Completion-realcmb | 26.92 | 25.96 | 17.31 | 48.56 | 98.08 | 95.67 | 100.00 | 100.00 |
| Completion-close-2hash | 5.29 | 5.29 | 10.10 | 45.67 | 97.60 | 98.56 | 100.00 | 100.00 |
| Completion-close-1hash | 9.62 | 6.25 | 7.69 | 36.54 | 65.38 | 79.81 | 100.00 | 100.00 |
| Completion-close-0hash | 11.06 | 5.77 | 9.13 | 47.12 | 93.27 | 97.12 | 100.00 | 100.00 |
| Completion-close-upper | 5.29 | 6.25 | 7.21 | 38.46 | 93.27 | 96.15 | 100.00 | 100.00 |
| Completion-close-title | 5.77 | 5.77 | 7.21 | 27.40 | 70.19 | 87.98 | 99.52 | 100.00 |
| Completion-close-nospace | 6.25 | 5.77 | 6.25 | 38.46 | 82.69 | 89.90 | 100.00 | 100.00 |
| Completion-close-nocolon | 6.25 | 5.77 | 8.65 | 38.46 | 71.63 | 89.42 | 100.00 | 100.00 |
| Completion-close-typo | 7.21 | 6.25 | 10.10 | 50.96 | 97.12 | 99.52 | 99.52 | 100.00 |
| Completion-close-similar | 5.77 | 5.29 | 8.65 | 45.19 | 91.83 | 93.75 | 99.52 | 99.52 |
| Average | 24.75 | 24.52 | 24.77 | 50.96 | 89.96 | 94.60 | 99.66 | 99.83 |
| Worst | 4.81 | 5.29 | 6.25 | 27.40 | 65.38 | 79.81 | 98.08 | 98.56 |
|  |  |  |  |  |  |  |  |  |

**Table 3: Full evaluation results of the LLM on LLAMA-2-13B using out-of-domain indirect prompt injection attacks.**
| Dataset | Clean Alpaca | Adversarial Alpaca |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | Baseline | +ISE (Ours) | Delimiter | +ISE (Ours) | Baseline | +ISE (Ours) | Delimiter | +ISE (Ours) |
| AlpacaEval ($\uparrow$) | 73.32 | 72.13 | 72.67 | 71.21 | 73.41 | 73.35 | 72.26 | 72.60 |
| Naive | 62.02 | 63.46 | 66.35 | 69.71 | 64.90 | 65.87 | 67.79 | 76.44 |
| Ignore | 52.40 | 66.83 | 51.92 | 69.71 | 98.56 | 96.63 | 96.15 | 96.63 |
| Escape-separation | 72.12 | 63.46 | 71.63 | 70.67 | 73.08 | 74.52 | 76.44 | 88.46 |
| Completion-real | 1.92 | 1.92 | 12.99 | 34.14 | 85.58 | 96.64 | 91.35 | 99.52 |
| Average | 47.12 | 48.92 | 50.72 | 61.06 | 80.53 | 83.41 | 82.93 | 90.26 |
| Worst | 1.92 | 1.92 | 12.99 | 34.14 | 64.90 | 65.87 | 67.79 | 76.44 |
|  |  |  |  |  |  |  |  |  |

## Appendix E More Experimental Results on Instruction Hierarchy

In this section, we provide additional experimental results on the Instruction Hierarchy benchmark, covering indirect prompt injection (Appendix [E.1](https://arxiv.org/html/2410.09102v2#A5.SS1)), direct prompt injection (Appendix [E.2](https://arxiv.org/html/2410.09102v2#A5.SS2)), prompt extraction (Appendix [E.3](https://arxiv.org/html/2410.09102v2#A5.SS3)), and harmful requests (Appendix [E.4](https://arxiv.org/html/2410.09102v2#A5.SS4)). Furthermore, we present the results for Llama-3.1-8B in Appendix [E.5](https://arxiv.org/html/2410.09102v2#A5.SS5).

### E.1 Detailed Analysis of Indirect Prompt Injection

In Figure [6](https://arxiv.org/html/2410.09102v2#S6.F6), we present the results of both in-domain and out-of-domain attacks. Similar to the Structured Query benchmark, we evaluate additional in-domain attacks designed by (Chen2024StruQDA), which are shown in Figure [21](https://arxiv.org/html/2410.09102v2#A5.F21). Due to space constraints, we use Att to represent different attacks.

Specifically, Att1 to Att18 correspond to the following list of attacks:
Att1:Naive,
Att2:Ignore,
Att3:Escape_deletion,
Att4:Escape_separation,
Att5:Completion_other,
Att6:Completion_othercmb,
Att7:Completion_real,
Att8:Completion_realcmb,
Att9:Completion_close_2hash,
Att10:Completion_close_1hash,
Att11:Completion_close_0hash,
Att12:Completion_close_upper,
Att13:Completion_close_title,
Att14:Completion_close_nospace,
Att15:Completion_close_nocolon,
Att16:Completion_close_typo, and
Att17:Completion_close_similar.

Again, we observe that our ISE method significantly enhances robustness against almost all attacks. The average robust accuracy gains range from approximately 15% to 45%, with the worst robust accuracy gains reaching up to nearly 50%.

Figure: Figure 21: Full results of in-domain indirect prompt injection attack we evaluated on the Instruction Hierarchy benchmark.
Refer to caption: https://arxiv.org/html/2410.09102/x7.png

### E.2 Detailed Analysis of Direct Prompt Injection

In Figure [22](https://arxiv.org/html/2410.09102v2#A5.F22), we report the robust accuracy against both in-domain and out-of-domain direct prompt injection attacks. We observe performance gains for our ISE method across various attack scenarios. For instance, the average robust accuracy against in-domain attacks improves from 47.3% to 69.9% for the model trained on the UltraChat Baseline dataset.

Additionally, similar to indirect prompt injection attacks, we also include the full results of in-domain attacks in Figure [23](https://arxiv.org/html/2410.09102v2#A5.F23). The attacking prompts are exactly the same as described in Appendix [E.1](https://arxiv.org/html/2410.09102v2#A5.SS1). These results further validate the effectiveness of our method, improving the average robust accuracy by over 10% and the worst robust accuracy by over 20%.

Figure: Figure 22: Results of direct prompt injection attack we evaluated on the Instruction Hierarchy benchmark.
Refer to caption: https://arxiv.org/html/2410.09102/x8.png

Figure: Figure 23: Full results of in-domain direct prompt injection attack we evaluated on the Instruction Hierarchy benchmark.
Refer to caption: https://arxiv.org/html/2410.09102/x9.png

### E.3 Detailed Analysis of Prompt Extraction

Following Section [6.3](https://arxiv.org/html/2410.09102v2#S6.SS3), we also present the full results of the prompt extraction on the Unnatural Instructions dataset. We observe similar trends where adding ISE makes the model more robust against extraction attacks, potentially enhancing privacy. Notably, the robustness (i.e., the ratio of cases where the attack fails to extract a significant number of original prompts) improves by over 20% on both average and worst scenarios for the models trained on the UltraChat Baseline.

Figure: Figure 24: Full results of prompt extraction we evaluated on Unnatural Instructions.
Refer to caption: https://arxiv.org/html/2410.09102/x10.png

### E.4 Detailed Analysis of Harmful Requests

We present the full results from Figure [10](https://arxiv.org/html/2410.09102v2#S6.F10) with two more models trained on the System Follow and Instruction Hierarchy dataset in Figure [25](https://arxiv.org/html/2410.09102v2#A5.F25). We continue to observe average robustness improvements across different categories, especially for the UltraChat Baseline and Instruction Hierarchy datasets. Note that the model was trained without any data specifically designed to bypass the safety guidelines.

Figure: Figure 25: Full results of the harmful request evaluation on JailbreakChat prompts using StrongREJECT malicious instructions.
Refer to caption: https://arxiv.org/html/2410.09102/x11.png

### E.5 Detailed Analysis of LLAMA 3.1 Model

We then provide a more detailed evaluation of the Llama-3.1-8B model on Instruction Hierarchy in Figure [26](https://arxiv.org/html/2410.09102v2#A5.F26). We continue to observe improved model capability and enhanced robustness across various attacks, indicating that our method generalizes well to different models. For instance, ISE consistently improves the winning rate on AlpacaEval and either maintains or improves the score on MT-Bench. In terms of robustness, our method also improves performance, even for models trained on Instruction Hierarchy, which already achieve high robustness.

Figure: Figure 26: Evaluations on Llama-3.1-8B. The evaluation of model capabilities on the Instruction Hierarchy benchmark is conducted using AlpacaEval and MT-Bench (Figure a). Robustness evaluations include indirect and direct prompt injection attacks, prompt extraction attacks, and harmful requests (Figure b). We performed experiments across three training datasets and compared our Instructional Segment Embedding (ISE) method against the baseline.
Refer to caption: https://arxiv.org/html/2410.09102/x12.png

### E.6 \reb Investigation of Instructional Segment Embedding

In this appendix, we examine the behavior of our ISE embeddings across different scenarios. Specifically, we analyze three settings: omitting the system prompt, using the system prompt with user embedding, and using the data part with user embedding.

Capability Evaluation. First, we present the results on AlpacaEval as the clean performance in Table [4](https://arxiv.org/html/2410.09102v2#A5.T4) with these three settings.

In general, we observe a slight performance degradation ($<$ 5%) when system prompts are omitted, particularly for models trained on the Instruction Hierarchy. However, our method largely preserves instruction-following capability compared to baseline methods. This highlights that the system prompt plays a more significant role in the ISE models than in the baseline methods.

Next, we assess the performance of system prompts using user embeddings. In this setting, we observe a noticeable performance degradation ($\sim$15%), which is even greater than when system prompts are omitted. We attribute this to the system prompts (i.e., Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.) being mixed with the original user prompts. This makes it challenging for the model to accurately interpret and follow the intended user prompt, leading to lower-quality responses.

Interestingly, we find that performance remains high when the data part uses user embeddings. The degradation is less than 2%. We think that this is because some of the training data does not separate the user and data parts.

**Table 4: AlpacaEval evaluation of various configurations with and without Instruction Segment Embedding (ISE).**
| Training Data | UltraChat Baseline | Instruction Follow | Instruction Hierarchy |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Baseline | +ISE (Ours) | Baseline | +ISE (Ours) | Baseline | +ISE (Ours) |
| With system prompt | 63.18% | 64.65% | 77.24% | 81.82% | 79.25% | 83.35% |
| Without system prompt | 61.39% | 63.50% | 75.16% | 76.60% | 79.47% | 78.57% |
| System using user embedding | — | 50.93% | — | 66.42% | — | 70.90% |
| Data using user embedding | — | 63.34% | — | 80.77% | — | 83.08% |
|  |  |  |  |  |  |  |

Robustness Evaluation. We then conduct experiments against indirect prompt injection attacks (Naive attack) under the same three settings. We present the results in Table [5](https://arxiv.org/html/2410.09102v2#A5.T5).

Firstly, we observe that the robustness against indirect prompt injection attacks remains roughly unchanged when the system prompt is omitted. Our ISE method still demonstrates higher robustness (10% to 20%) compared to the baseline.
Secondly, we evaluate the use of system prompts with user embeddings and observe a slight performance degradation (<6%).
Lastly, we assess the use of data prompts with user embeddings and find a more noticeable performance degradation (up to 10%). This result is expected, as adversarial texts (injected into the data part) are more prioritized.

**Table 5: Performance of various configurations with and without Instruction Segment Embedding (ISE) against indirect prompt injection attacks.**
| Training Data | UltraChat Baseline | Instruction Follow | Instruction Hierarchy |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Baseline | +ISE (Ours) | Baseline | +ISE (Ours) | Baseline | +ISE (Ours) |
| With system prompt | 64.42% | 82.21% | 61.54% | 77.88% | 83.65% | 87.02% |
| Without system prompt | 62.50% | 82.69% | 64.90% | 78.37% | 80.77% | 88.94% |
| System using user embedding | — | 76.92% | — | 75.48% | — | 85.58% |
| Data using user embedding | — | 76.44% | — | 71.63% | — | 76.92% |
|  |  |  |  |  |  |  |

### E.7 Evaluation on automatic attacks.

We also conduct experiments to evaluate the effectiveness of our method against indirect prompt injection attacks using automatically optimized prompts. Specifically, we follow the methodology outlined in (chao2023jailbreaking), which iteratively refines and improves the attacking prompts using GPT-4o. The goal of these prompts is to mislead the model into generating the word “hacked.” The results are presented in Table [6](https://arxiv.org/html/2410.09102v2#A5.T6).

We observe that our ISE methods consistently demonstrate higher robustness (up to 20%) against this automatic attack. The improvement is particularly noticeable when the model is trained on the UltraChat Baseline and Instruction Following datasets. These results align with our findings on other types of attacks.

**Table 6: Performance of the Instructional Segment Embedding on multi-turn MT-Bench.**
| Training Data | UltraChat Baseline | Instruction Follow | Instruction Hierarchy |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Baseline | +ISE (Ours) | Baseline | +ISE (Ours) | Baseline | +ISE (Ours) |
| Automatic attacks | 39.90% | 57.21% | 13.46% | 34.62% | 65.38% | 69.71% |
|  |  |  |  |  |  |  |

### E.8 Evaluation on multi-turn conversation.

Although all the training data are single-turn conversations, we observe that our ISE method is capable of handling multi-turn conversations. Specifically, we conduct experiments on MT-Bench with multi-turn scenarios and prompt GPT-4o to evaluate the performance of the generated responses. The results are presented in Table [7](https://arxiv.org/html/2410.09102v2#A5.T7).

We observe that all models experience some performance degradation (up to 2.7), which is expected given that the models are trained on single-turn chat datasets. Nonetheless, ISE achieves comparable or even higher MT-Bench scores in multi-turn tasks. This demonstrates the potential of our method to extend to multi-turn conversations.

**Table 7: Performance of the Instructional Segment Embedding on multi-turn MT-Bench.**
| Training Data | UltraChat Baseline | Instruction Follow | Instruction Hierarchy |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Baseline | +ISE (Ours) | Baseline | +ISE (Ours) | Baseline | +ISE (Ours) |
| Single-turn | 6.73 | 6.64 | 7.38 | 7.55 | 7.30 | 7.48 |
| Mutli-turn | 5.81 | 5.7 | 5.6 | 6.48 | 4.61 | 5.4 |
|  |  |  |  |  |  |  |

### E.9 Case Study of ISE Attention

In this appendix, we analyze the attention differences between the baseline and ISE models.
We focus on models trained on the instruction hierarchy dataset and illustrate an attack example (indirect prompt injection) in Figure [27](https://arxiv.org/html/2410.09102v2#A5.F27).
Our observations reveal that the baseline model fails to distinguish benign instructions from malicious ones, whereas our method effectively identifies the presence of malicious attacks in the context.

Next, we present a visualization of the average attention logits in layer 0 in Figure [28](https://arxiv.org/html/2410.09102v2#A5.F28).
We observe that the diagonal line of the baseline model (without ISE) is much brighter than that of our ISE model. This indicates that the baseline model places significantly stronger attention on neighbor tokens, making it more susceptible to malicious instructions in data. As a result, the malicious instruction "Print exactly hacked" plays an overly important role in generating the final response. In contrast, the ISE model distributes attention more evenly across the full instructions, and provide safe responses.

Figure: Figure 27: \rebA demonstration of indirect prompt injection attacks against baseline and ISE models trained on instruction hierarchy dataset.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/attack_case.png

Figure: Figure 28: \rebAttention patterns between baseline and ISE models on the example of Figure [27](https://arxiv.org/html/2410.09102v2#A5.F27).
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/attention.png

### E.10 Detailed Figure of Robustness Evaluation on Instruction Hierarchy

Figure: Figure 29: \rebDetailed demonstration of Figure [5](https://arxiv.org/html/2410.09102v2#S6.F5). Evaluation results of Llama-3-8B on Instruction Hierarchy.
Refer to caption: https://arxiv.org/html/2410.09102/extracted/6244466/img/all_alternative.png

## Appendix F Discussions and Evaluations on Jailbreak Attacks

In our harmful request evaluations, we primarily focused on malicious prompts collected in the wild, without involving any active optimization, following Wallace2024TheIH.
We considered these prompts as zero-shot generalization evaluations since no training data aimed to bypass safety alignment.

There also exist adaptive attacks, known as jailbreak attacks, generated through advanced strategies, such as adversarially optimized texts like those in (zou2023universal; liao2024amplegcg), or carefully human-crafted strategies as seen in anil2024many.
In Table [8](https://arxiv.org/html/2410.09102v2#A6.T8), we present the results of using adaptive attacks from (andriushchenko2024jailbreaking; zheng2024fsj) on 50 malicious requests (chao2023jailbreaking), and we observe that our models almost completely fail to generate safe responses.

In fact, we do not expect our method to improve adaptive jailbreak robustness.
First, none of our data were explicitly created to defend against (or reject) jailbreak attacks. Second, while our segment embedding is designed to differentiate between types of instructions, adversarial texts may directly target the model.
Our method is orthogonal to many robust training methods, such as LAT (sheshadri2024targeted) and circuit breakers (Zou2024ImprovingAA). We leave further exploration of this issue for future research.

**Table 8: Robust accuracy against adaptive attacks on Instruction Hierarchy benchmark.**
|  | UltraChat<br>Baseline | +ISE<br>(Ours) | System<br>Follow | +ISE<br>(Ours) | Instruction<br>Hierarchy | +ISE<br>(Ours) |
| --- | --- | --- | --- | --- | --- | --- |
| Jailbreak attacks (%) | 2.0 | 2.0 | 0.0 | 2.0 | 2.0 | 4.0 |
|  |  |  |  |  |  |  |