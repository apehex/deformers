Title: SysBench: Can Large Language Models Follow System Messages?
ArXiv: 2408.10943
Authors: Haoze Sun, Yan Zhang, Yujing Qiao, Weipeng Chen, Zenan Zhou, Wentao Zhang, Bin Cui, Corresponding author.
Sections: 29
Estimated tokens: 17.8k

## Contents
- 1 Introduction
- 2 Related Work
  - 2.1 System Messages in LLM
  - 2.2 Evaluation of Instruction Following
- 3 SysBench
  - 3.1 Benchmark Design
    - System Constraint
    - User Instruction
    - Multi-turn Conversation
  - 3.2 Dataset
    - Data Construction
    - Data Statistics
  - 3.3 Evaluation Protocol
- 4 Experiments
  - 4.1 Experimental Setup
  - 4.2 Constraints-categorized Results
  - 4.3 Instruction Alignment
  - 4.4 Multi-turn Stability
  - 4.5 Investigative Experiments
    - How Does Historical Dialogue Affect the Multi-turn Stability?
    - Is There a Correlation Between Attention Distribution and the Ability to Follow System Messages?
    - Are System Messages and User Messages Truly Treated Differently?
- 5 Conclusion
- References
- Appendix A More about SysBench
  - A.1 Detailed Constraint Categories
  - A.2 The Verification Prompt for Model-based Verifier
  - A.3 Sample Data Format
- Appendix B More Experimental Results

## Abstract

Abstract Large Language Models (LLMs) have become instrumental across various applications, with the customization of these models to specific scenarios becoming increasingly critical. System message, a fundamental component of LLMs, is consist of carefully crafted instructions that guide the behavior of model to meet intended goals. Despite the recognized potential of system messages to optimize AI-driven solutions, there is a notable absence of a comprehensive benchmark for evaluating how well LLMs follow system messages. To fill this gap, we introduce SysBench, a benchmark that systematically analyzes system message following ability in terms of three limitations of existing LLMs: constraint violation, instruction misjudgement and multi-turn instability.
Specifically, we manually construct evaluation dataset based on six prevalent types of constraints, including 500 tailor-designed system messages and multi-turn user conversations covering various interaction relationships.
Additionally, we develop a comprehensive evaluation protocol to measure model performance.
Finally, we conduct extensive evaluation across various existing LLMs, measuring their ability to follow specified constraints given in system messages. The results highlight both the strengths and weaknesses of existing models, offering key insights and directions for future research. The open source library SysBench is available at https://github.com/PKU-Baichuan-MLSystemLab/SysBench .

## 1 Introduction

Recently, Large Language Models (LLMs) have been employed across diverse array of applications, including writing assistance, educational tools, web agents, and more (Parisi et al., 2022; Schick et al., 2023; Nakano et al., 2021). To better manage the model’s interactive behavior for various task scenarios, the system message component is introduced by ChatGPT (OpenAI, 2022) and extensively utilized in current LLMs (Touvron et al., 2023; Anthropic, 2024; Yang et al., 2024). For example in Figure [1](https://arxiv.org/html/2408.10943v2#S1.F1), system message is a set of carefully crafted instructions that predefine the role, contextual information, guidelines, or output format of the model (Ramlochan, 2024; Lee et al., 2024; Wallace et al., 2024). These instructions setting the model to generate responses that are aligned with the desired outcome, playing a pivotal role in bridging the gap between the vast knowledge acquired by LLMs during training and their application in real-world scenarios, such as maintaining personality in role-playing scenarios (Ma et al., 2024; Salewski et al., 2023), increasing robustness and resilience (Wallace et al., 2024; Lu et al., 2024) and customizing interaction preference for specific tasks (Lee et al., 2024; Mukherjee et al., 2023).

Even though system messages have been widely used, accurately following system messages is still challenging, which requires the model to satisfy all the constraints pre-set in the system messages when responding. As the first round of conversation shown in Figure [1](https://arxiv.org/html/2408.10943v2#S1.F1), when the user inputs “Hello”, the model should introduce itself according to the settings in the system message.
However, LLMs may encounter several issues in practical applications.
Primarily, understanding the complex constraints in system messages and accurately applying these rules during interaction with users is a difficult task. It is observed that constraint violation often occurs in practical applications (Mu et al., 2023; Li et al., 2024b; Wang et al., 2024).
Additionally, the user query may conflict with the system message, and misjudging the instruction priority could lead to the risk of security attacks (Zou et al., 2024; Wallace et al., 2024). Furthermore, system messages are only set at the beginning of the conversation, and
empirical evidence shows that as the historical dialogues become lengthy, the model responses may deviate from the constraints specified by the system message (Li et al., 2024a). These issues can result in a diminished user experience or potential security concerns.

Figure: Figure 1: A sample system message, and limitations of LLMs on system message following.
Refer to caption: https://arxiv.org/html/2408.10943/x1.png

However, there is an evident gap in comprehensive evaluation of the ability to follow system message of existing LLMs, impeding the understanding and further research of the system message component.
Existing research on system messages only uses small-scale, simplified datasets or specific models to analyze certain characteristics, failing to fully evaluate the following ability in real-world scenarios.
(Mu et al., 2023; Li et al., 2024b; Wang et al., 2024; Zou et al., 2024; Li et al., 2024a).
In summary, evaluating the ability of LLMs to follow system messages presents the following challenges:
(1) High-quality evaluation data construction.
The evaluation data consists of system messages and corresponding multi-turn user conversations. Benchmarks for evaluating the instruction following capability of LLMs are limited to single-turn user conversation (Zhou et al., 2023; Xia et al., 2024; Zheng et al., 2023; Jiang et al., 2023; Wen et al., 2024; He et al., 2024). These datasets fails to capture the interaction between system messages and users, making them unsuitable for evaluating following to system messages. To ensure effective evaluation, the relevance of user queries to system constraints, the assessability of constraint content, and the diversity and scale of data sources should be well-designed, requiring expert knowledge and human involvement.
(2) Accurate evaluation protocol.
System messages in real-world scenarios contain multiple complex and subjective constraints, making it difficult to accurately verify whether system messages are well followed.
Existing benchmarks adopt programmatic or model-based evaluation, and design various metrics to evaluate instruction following ability (Zhou et al., 2023; Zheng et al., 2023; Wen et al., 2024).
Determining whether system messages are followed is intricately linked to the context of conversations, and its multi-turn characteristic requires new evaluation metrics, introducing new challenges in designing evaluation criteria.

To bridge this gap, we introduce SysBench, encompassing expert-annotated high-quality data and precise evaluation protocol. To ensure data quality and evaluability, our data is collected from real scenarios, and rewritten by trained annotators according to guidelines designed by experienced experts. The dataset includes 500 system messages spanning various domains, each with 5 rounds of user conversations, covering multiple types of constraints and instruction alignment relationships. To evaluate system message following, similar to (Jiang et al., 2023; Zheng et al., 2023; Wen et al., 2024), we use advanced LLMs as verifier, and ensure evaluation accuracy through manually annotated evaluation checklists and well-designed evaluation prompts. Moreover, we conduct extensive experiments on 16 popular LLMs, and find that following system messages remains challenging, especially when user instructions conflict with system messages.
Additionally, a positive correlation exists between the attention scores allocated to system messages and model’s ability to follow system messages.
These findings provide insights for developers to improve system message mechanisms.
In summary, our contributions including:

- •
New Benchmark. We first systematically investigate the ability of LLMs to follow system messages and propose a comprehensive benchmark SysBench, facilitating both dataset construction and evaluation criteria design.
- •
Accessible Dataset. We construct a high-quality dataset focusing on system message following evaluation, which includes 500 system messages, each corresponding to 5 turns of user conversations, covering a variety of application scenarios.
- •
Comprehensive Evaluation. We design three-level granularity evaluation framework for assessing LLMs’ ability to follow system messages, and extensively evaluate 16 popular LLMs, gaining key insights into system messaging mechanisms.

## 2 Related Work

### 2.1 System Messages in LLM

The system message is an specialized input component of LLMs first introduced by ChatGPT (OpenAI, 2022) and widely used in existing models (e.g., Mistral3 (AlKhamissi et al., 2024), Claude3.5 (Anthropic, 2024), etc.). The system message provides an easy-to-organize, context-stable way to steer the generation behavior, attracting investigation to the mechanisms of system messages. (Lee et al., 2024; Mukherjee et al., 2023; Touvron et al., 2023) find that training with diverse system messages instead of the default making model align better with human’s preference. Besides, some studies emphasize the importance of prioritizing system messages. (Wallace et al., 2024) underscore the critical role of prioritizing system message commands to ensure the safety of LLMs. (Lu et al., 2024) propose a training strategy that aligns with instruction priorities, thereby improving the model’s ability to distinguish between safe and harmful content. (Mu et al., 2023) explore the ability of models to comply with priority rules in 14 text scenarios. Additionally, (Li et al., 2024a) observe that the stability of system messages tends to deteriorate as dialogues lengthen, accompanied by a decay in attention scores. Despite the widespread use of system messages, current research primarily focuses on specific aspects and conducts small-scale experiments on simplified dataset. There is a notable gap in comprehensive benchmark evaluations that reflect real-world applications.

### 2.2 Evaluation of Instruction Following

Instruction following is a critical capability for LLMs, and numerous studies attempts to evaluate it. Early research focused on simple, single-type instructions with easily verifiable constraints (Zhou et al., 2023; Xia et al., 2024; Zheng et al., 2023).
However, as LLMs are increasingly deployed in complex real-world tasks, there is a growing need to assess their ability to follow complex instructions.
(Qin et al., 2024) deconstructs complex instructions into simpler components, enabling a thorough analysis of instruction following to different facets of tasks. (Jiang et al., 2023) introduces a benchmark for multi-level constraint following, encompassing both subjective and objective constraints. (He et al., 2024) defines complex instructions using task descriptions and input texts, evaluating LLMs with datasets that mimic real-world scenarios. (Wen et al., 2024) evaluates ability of instruction following from a constraint compositions perspective. However, these benchmarks are typically consist of single-turn conversations, rendering them not suitable for evaluation of the system message.
This gap highlights the need for benchmarks that more accurately mirror the multi-turn, interactive nature of real-world applications where system messages play a crucial role.

## 3 SysBench

The workflow of SysBench is shown in Figure [2](https://arxiv.org/html/2408.10943v2#S3.F2). We initially outline our data design principles in Section [3.1](https://arxiv.org/html/2408.10943v2#S3.SS1), followed by a description of the construction pipeline and dataset statistics in Section [3.2](https://arxiv.org/html/2408.10943v2#S3.SS2). Ultimately, we delve into the evaluation methodology and metrics in Section [3.3](https://arxiv.org/html/2408.10943v2#S3.SS3).

Figure: Figure 2: Workflow of SysBench. Both system message and corresponding user instructions are fed into LLM to generate outputs; then a model-based verifier is applied to each response for evaluation. All texts are simplified for clearer presentation.
Refer to caption: https://arxiv.org/html/2408.10943/x2.png

### 3.1 Benchmark Design

We design SysBench dataset construction principles to better explore the following three questions: 1) Can LLMs understand and follow different types of constraint? 2) Can LLMs determine alignment relationship between system messages and user queries? 3) Can LLMs continuously follow the system message in multi-turn conversations?

#### System Constraint

Constraints are fine-grained settings or rules defining the model behavior (Jiang et al., 2023; Liu et al., 2024), like ”You are a proper calculator” and ”all response should start with Beep-beep” in Figure [2](https://arxiv.org/html/2408.10943v2#S3.F2). In SysBench, constraints are designed as verifiable atomic rules, and evaluation checklists are manually annotated for each user query to verify whether the relevant system constraints are correctly satisfied. To match the real-world scenario, each system message in our dataset contains multiple complex constraints, and a data sample can be found in Appendix [A.3](https://arxiv.org/html/2408.10943v2#A1.SS3). Based on expert experience and collected data clustering, we categorize the constraints in system message into six prevalent categories, including action, content, background, role, format and style constraints, and the specific descriptions can be seen in Appendix [A.1](https://arxiv.org/html/2408.10943v2#A1.SS1).

#### User Instruction

With system message configured, the input prompt is a combination of both system and user messages. Therefore, constraint following with system messages is influenced by the alignment relationship between user instructions and system messages (Wallace et al., 2024; Li et al., 2024b). Aligned user instructions have compatible goal with system messages, enabling the LLMs to satisfy both of them simultaneously (i.e., Instruction 1 and Instruction 5 in Figure [2](https://arxiv.org/html/2408.10943v2#S3.F2)). In this case, instruction can be considered as concatenation of the system message and user instruction, reflecting the ability to follow constraints. Misaligned user instructions, on the other hand, contradict the system messages (i.e., Instruction 2-4 in Figure [2](https://arxiv.org/html/2408.10943v2#S3.F2)). The model should refuse to comply or ignore them to prevent security attacks (Wallace et al., 2024; Lu et al., 2024). The ability to follow misaligned user instructions emphasises the priority distinction between system and user instructions.
We also develop a checklist for each user instruction considering the alignment relationship to its system message. Each constraint corresponds to one entry in the checklist. With clear definitions of correct behavior, the verification task becomes straightforward, enabling the model-based verifier to perform effectively. For instance, GPT-4o (OpenAI, 2024) verifier achieves over 94% consistency with human evaluations.

#### Multi-turn Conversation

The system message can be specified when creating a session, fixed at the beginning of context window, and expected to be stably followed throughout the conversations (Shi et al., 2023).
Depending on the relationship between current user instructions and previous dialogues, we classify multi-turn dialogues into two types: multi-turn parallel and multi-turn dependency. In a multi-turn parallel conversation, each user instruction is independent; thus, the model is not supposed to be affected by prior dialogues when responding. Instead, it should focus solely on the current user instructions within the context established by the system message, just like the conversations in Figure [2](https://arxiv.org/html/2408.10943v2#S3.F2). In a multi-turn dependent conversation, historical context information is often pertinent to the current round of dialogue. Accurately responding to the system message requires not only an understanding of the current user instruction but also the integration of information from history dialogues.

### 3.2 Dataset

According to the above framework, SysBench’s dataset is constructed with a pipeline including real-world data collection, model-based data preprocessing, and manually data formulation, ensuring comprehensive domain coverage and high data quality.

Figure: Figure 3: Distribution of domains and constraints.
Refer to caption: https://arxiv.org/html/2408.10943/x3.png

#### Data Construction

We initially collect thousands of system messages from online logs, and filter out duplicate and noisy data based on heuristic rules and clustering. Subsequently, five hundred system messages are selected by 21 trained data annotators and then manually refined to ensure explicit task descriptions and diverse constraints. For each system message, we collect corresponding user conversations and use GPT-4o to assist in generating conversations. Eventually, five rounds of user conversations for each system message are retained, and are manually rewritten by annotators according to the annotation guidelines designed by experts. Furthermore, the clear evaluation checklist is annotated for each instruction, guiding the model-based verifier to better evaluate whether the model’s response satisfies relevant constraints in the system message. All data are checked independently by multiple experts in multiple rounds to ensure quality.

#### Data Statistics

As shown in Table [1](https://arxiv.org/html/2408.10943v2#S3.T1), the SysBench dataset includes a total of 500 system messages, each of them includes 5 rounds of user conversations, and each user instruction is related to 2-3 system constraints on average.
This configuration suggests that our evaluation dataset presents a moderate level of complexity, enabling effective differentiation of model capabilities.
The data are categorized into aligned and misaligned instructions, as well as multi-turn dependent and multi-turn parallel dialogues, providing more perspectives for analyzing model performance.
Figure [3](https://arxiv.org/html/2408.10943v2#S3.F3) depicts the distribution of task domains and constraints, showing that our data covers a variety of task scenarios and constraint types. The role and background constraints account for a smaller percentage, because each system message usually contains no more than one role or background constraint in real-world applications, while the other four types of constraints can appear multiple times.

### 3.3 Evaluation Protocol

Inspired by the benchmark work of existing instructions (Jiang et al., 2023), we adopt advanced LLM as verifier to determine whether constraints in system messages has been satisfied by each response to user request. To ensure that LLMs can objectively and accurately assess the constraint satisfaction conditions, we manually annotated evaluation checklists for each user instruction as introduced in §[3.2](https://arxiv.org/html/2408.10943v2#S3.SS2). Besides, our evaluation prompt also encompasses the system message, historical conversations and the user instruction. The advanced model and structured evaluation prompt allow for a precise assessment of constraint satisfaction rate in system messages.

We define three-level granularity metric to evaluate the satisfied rates for system messages. Given a set of $m$ sessions, with each session contains $n$ turns conversations. For the $j$-th user instruction in $i$-th session, there are $c_{ij}$ relevant system constraints. Let $s_{ijk}$ represents a binary variable indicating whether the response from the $j$-th turn of the $i$-th conversation satisfies its $k$-th constraint.

Constraint Satisfaction Rate (CSR) represents the finest level of granularity and is defined as the average accuracy of constraints satisfied:

$$ $\text{CSR}\vcentcolon=\frac{1}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n}\left(\frac{1}{c _{ij}}\sum_{k=1}^{c_{ij}}s_{ijk}\right)$ (1) $$

It evaluates the model’s ability to follow specific constraints within a single instruction, focusing on detailed constraint following ability.

Instruction Satisfaction Rate (ISR) is designed for assessing whether the response to a user instruction totally satisfied constraints in system message:

$$ $\text{ISR}\vcentcolon=\frac{1}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n}\left(\bigwedge_ {k=1}^{c_{ij}}s_{ijk}\right)$ (2) $$

In this equation, each instruction is counted only if all its associated constraints are met, measuring at a broader level.

Session Stability Rate (SSR) is at the top level, defined from the perspective of multi-turn conversation. It measures the average number of consecutive turns in which the model satisfies all constraints from the start of the conversation. This metric can be mathematically expressed as follows:

$$ $\text{SSR}\vcentcolon=\frac{1}{mn}\sum_{i=1}^{m}\sum_{\alpha=1}^{n}\left( \bigwedge_{j=1}^{\alpha}\bigwedge_{k=1}^{c_{ij}}s_{ijk}\right)$ (3) $$

Here, the second summation accumulates a binary value that is assigned a 1 if, and only if, all responses from the first to the $\alpha$-th round satisfy all the constraints. This definition underscores the model’s ability to maintain constraint satisfaction continuously over multiple conversational turns.

## 4 Experiments

### 4.1 Experimental Setup

**Table 2: Models’ performance on SysBench. Models in the left are English oriented, while those in right are Chinese, and those denoted with (^†) are called through API. The first and the second rankings are presented in bold and underlined, respectively.**
| Model Name | CSR | ISR | SSR | Model Name | CSR | ISR | SSR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT4o^† | 87.1% | 76.4% | 54.4% | Qwen2.5-72B-Instruct | 80.4% | 66.2% | 42.8% |
| GPT4-Turbo-20240409^† | 86.5% | 76.6% | 53.2% | GLM-4-0520^† | 78.9% | 65.5% | 41.6% |
| Claude-3-Opus^† | 85.0% | 74.1% | 51.8% | Qwen2-72B-Instruct | 79.0% | 64.1% | 41.6% |
| Llama3.1-70B-Instruct | 76.6% | 60.3% | 36.6% | DeepSeek-V2-0628^† | 76.1% | 61.7% | 39.6% |
| Llama3.1-8B-Instruct | 66.5% | 46.9% | 24.9% | Moonshot-V1-8K^† | 70.3% | 52.3% | 30.0% |
| Mixtral-8x22B-Instruct | 63.6% | 44.4% | 22.8% | GLM-4-9B-Chat | 64.2% | 44.0% | 25.9% |
| GPT3.5-Turbo-20231106^† | 61.6% | 43.2% | 20.8% | ERNIE-4-8K-0613^† | 50.7% | 33.8% | 20.0% |
| Mixtral-8x7B-Instruct | 56.5% | 37.6% | 19.1% | Qwen2-7B-Instruct | 47.0% | 26.9% | 15.0% |

We evaluate various LLMs across different scales and types using SysBench.
Our analysis aims to address the following key questions concerning the system messages:
First, at the most granular level, are LLMs capable of following each kinds of constraints? (§[4.2](https://arxiv.org/html/2408.10943v2#S4.SS2)). Second, during individual conversation turns, how do LLMs respond to user instructions in different alignment? (§[4.3](https://arxiv.org/html/2408.10943v2#S4.SS3)). Third, do LLMs maintain stability across multi-turn dependent or parallel conversations? (§[4.4](https://arxiv.org/html/2408.10943v2#S4.SS4)).

Metrics. We employ the metrics outlined in §[3.3](https://arxiv.org/html/2408.10943v2#S3.SS3) to evaluate the performance of each model.
Specifically, CSR reflects the model performance at the constraint granularity level and is applied in §[4.2](https://arxiv.org/html/2408.10943v2#S4.SS2), ISR measures the following ability of LLMs at the instruction level, as discussed in §[4.3](https://arxiv.org/html/2408.10943v2#S4.SS3), while SSR is utilized to assess multi-turn stability in §[4.4](https://arxiv.org/html/2408.10943v2#S4.SS4).

Settings. We select GPT-4o as the model-based verifier in verification stage due to its demonstrated superior quality-price ratio, and set temperature to 0 to ensure deterministic output. During the generation stage, we maintain all inference parameters at their default settings across all scenarios.

Models. We evaluate sixteen popular LLMs including GPT family, Claude-3, Qwen-2, ERNIE-4, Moonshot, Mixtral, DeepSeek-V2, GLM-4, and Llama-3.1 family (Brown et al., 2020; OpenAI, 2024; Anthropic, 2024; Yang et al., 2024; Sun et al., 2021; Moonshot AI, 2023; Jiang et al., 2024; DeepSeek-AI, 2024; Team GLM, 2024; Llama Team, 2024).
The overall results under our proposed metrics are displayed in Table [2](https://arxiv.org/html/2408.10943v2#S4.T2) and analyzed from the bottom up in the rest of section.

### 4.2 Constraints-categorized Results

Figure: Figure 4: The CSR under different types of constraints. Only 8 representative ones are shown.
Refer to caption: https://arxiv.org/html/2408.10943/x4.png

**Table 4: The multi-turn stability results (version numbers are omitted for clarity; see Table [2](https://arxiv.org/html/2408.10943v2#S4.T2) for details). The $R_{n}$ columns indicates the percentage of the first n rounds of model responses that are all available, so the values decrease as n increases and satisfy Average($R_{n}$)=SSR by definition. The $k$ denotes the linear regression slope of $R_{n}$.**
| Multi-Turn | Model | R1 | R2 | R3 | R4 | R5 | $-k$ | SSR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dependency |  |  |  |  |  |  |  |  |
| Dependent | GPT-4o | 84.8% | 68.5% | 53.1% | 43.3% | 33.7% | 0.128 | 56.7% |
|  | GPT-4-Turbo | 81.7% | 64.0% | 51.7% | 41.0% | 32.3% | 0.122 | 54.2% |
|  | Claude-3 | 82.3% | 64.0% | 52.0% | 38.2% | 28.4% | 0.134 | 53.0% |
|  | Qwen2.5-72B | 78.7% | 58.1% | 41.9% | 27.8% | 16.3% | 0.155 | 44.6% |
|  | DeepSeek-V2 | 77.5% | 56.5% | 38.2% | 23.9% | 12.1% | 0.163 | 41.6% |
|  | Qwen2-72B | 78.9% | 54.8% | 37.9% | 23.3% | 12.4% | 0.165 | 41.5% |
|  | GLM-4 | 76.7% | 54.2% | 36.2% | 24.2% | 15.7% | 0.152 | 41.4% |
|  | Llama3.1-70B | 69.9% | 48.3% | 32.0% | 22.2% | 17.7% | 0.131 | 38.0% |
|  | Moonshot | 69.4% | 41.3% | 27.2% | 14.6% | 8.4% | 0.149 | 32.2% |
|  | GLM-4-9B | 66.6% | 38.8% | 22.8% | 7.9% | 3.7% | 0.157 | 27.9% |
|  | Llama3.1-8B | 62.9% | 34.3% | 18.3% | 9.0% | 6.7% | 0.138 | 26.2% |
|  | Mixtral-8x22B | 63.5% | 29.2% | 13.5% | 6.5% | 3.7% | 0.142 | 23.3% |
|  | GPT-3.5 | 53.9% | 28.7% | 16.3% | 7.3% | 5.1% | 0.119 | 22.2% |
|  | ERNIE-4 | 61.8% | 26.7% | 12.6% | 6.2% | 2.0% | 0.140 | 21.9% |
|  | Mixtral-8x7B | 54.2% | 26.4% | 13.8% | 6.2% | 2.5% | 0.124 | 20.6% |
|  | Qwen2-7B | 52.5% | 20.5% | 6.5% | 2.2% | 1.1% | 0.121 | 16.6% |
| Parallel | GPT-4o | 77.8% | 60.4% | 46.5% | 33.3% | 26.4% | 0.130 | 54.4% |
|  | GPT-4-Turbo | 72.9% | 62.5% | 49.3% | 38.2% | 30.6% | 0.109 | 53.2% |
|  | Claude-3 | 75.7% | 62.5% | 47.2% | 34.7% | 25.0% | 0.129 | 51.8% |
|  | Qwen2.5-72B | 68.8% | 51.4% | 35.4% | 22.9% | 14.6% | 0.137 | 42.8% |
|  | GLM-4 | 71.5% | 51.4% | 40.3% | 25.7% | 20.8% | 0.127 | 41.6% |
|  | Qwen2-72B | 72.2% | 48.6% | 39.6% | 28.5% | 20.1% | 0.124 | 41.6% |
|  | DeepSeek-V2 | 69.4% | 44.4% | 30.6% | 18.1% | 11.1% | 0.143 | 39.6% |
|  | Llama3.1-70B | 66.7% | 43.1% | 25.7% | 17.4% | 11.8% | 0.135 | 36.6% |
|  | Moonshot | 58.3% | 30.6% | 17.4% | 11.8% | 5.6% | 0.124 | 30.0% |
|  | GLM-4-9B | 52.8% | 27.8% | 15.3% | 5.6% | 2.8% | 0.122 | 25.9% |
|  | Llama3.1-8B | 53.5% | 25.0% | 18.1% | 8.3% | 3.5% | 0.117 | 24.9% |
|  | Mixtral-8x22B | 56.2% | 27.1% | 16.7% | 6.2% | 2.1% | 0.129 | 22.8% |
|  | GPT-3.5 | 45.1% | 18.8% | 11.1% | 8.3% | 2.1% | 0.097 | 20.8% |
|  | ERNIE-4 | 46.5% | 20.1% | 7.6% | 2.8% | 0.7% | 0.109 | 20.0% |
|  | Mixtral-8x7B | 42.4% | 20.1% | 9.7% | 4.9% | 0.0% | 0.100 | 19.1% |
|  | Qwen2-7B | 36.1% | 11.1% | 6.2% | 2.8% | 0.0% | 0.081 | 15.0% |

Diving into the finest granularity, we analyze the constraints embedded within each instruction and system message using the Constraint Satisfaction Rate (CSR) as our metric. We classify all constraints into six distinct types, as detailed in §[3.1](https://arxiv.org/html/2408.10943v2#S3.SS1), and compute the respective CSR scores. The overall CSR scores for each model are tabulated in Table [2](https://arxiv.org/html/2408.10943v2#S4.T2), while the results categorized by constraint type are shown in Figure [4](https://arxiv.org/html/2408.10943v2#S4.F4). We only present the most representative eight models in the radar plot, without loss of generality, and the full numeric results can be found in Appendix [B](https://arxiv.org/html/2408.10943v2#A2).

Overall, performance at the constraint-categorized level aligns closely with total CSR scores. GPT-4o leads the performance across all category of constraints, with Claude-3 closely behind. Qwen2-72B and GLM-4 also demonstrate strong performance.
It is noteworthy that although ERNIE-4 exhibits well performance in existing instruction following benchmarks (Zhang et al., 2024), its ability to follow system messages leaves room for improvement, with its performance comparable to the Qwen2-7B.
For the better-performing models, their profiles approximate a positive hexagon in Figure [4](https://arxiv.org/html/2408.10943v2#S4.F4), indicating similar CSR across each category. In contrast, models with poorer overall performance show significant variance among different types of constraints. For example, the CSR for Qwen2-7B under role constraints is relatively high at 81.0%, even comparable to some of the more successful models, yet it drops markedly to 43.3% under style constraints, highlighting an intuitive weakness. Similar patterns are also observed in GPT-3.5 and ERNIE-4.

CSR is assessed at the most granular level, directly mirroring the model’s capability to satisfy constraints.
The absolute magnitude of this value indicates the strength of the model’s performance, whereas misalignment in its relative magnitude across different constraint categories highlights specific areas of weakness.
This detailed information can guide developers in enhancing model performance by focusing training efforts on these identified sub-tasks.

### 4.3 Instruction Alignment

Moving onto the instruction level, the Instruction Satisfaction Rate (ISR) metric quantifies the proportion of responses generated by the models that fully follow all the given constraints. We categorize all instructions based on their alignment with the corresponding system messages, as stated in §[3.1](https://arxiv.org/html/2408.10943v2#S3.SS1).
The results are displayed in Table [3](https://arxiv.org/html/2408.10943v2#S4.T3).

GPT-4-Turbo holds first place by a narrow margin in this instance, differing from the leader in SSR. It is observed, as expected, that performance in aligned categories generally surpasses that in misaligned categories for most models, especially GLM-4-9B.
Compared to aligned instructions, when a user instruction conflicts with a system message (i.e., is misaligned), the model should prioritize the system message due to its higher importance.
Such performance degradation observed in this misaligned category is likely due to the model’s insufficient recognition of the system message’s priority, highlighting significant potential for optimization in this area.
Although the aligned instructions do not conflict with their corresponding system messages, the ISR still informatively indicates whether or not the models satisfy the constraints embedded in both system message and user instruction. Violating any of them will result in a negative contribution. This contrasts with misaligned instructions, where satisfying the system message alone is the requirement.

It should be noted that some models exhibit minimal performance variation between the two alignment types of instructions. And surprisingly, GPT-3.5 get notably higher ISR score on misaligned instructions than the aligned set, suggesting its acute awareness of the prioritization required by the system message.
Furthermore, this performance hints at the capability of LLMs to effectively prioritize system messages when faced with contradictory instructions.

### 4.4 Multi-turn Stability

On a broader scale, we categorize all conversations into multi-turn dependent and multi-turn parallel, as mentioned in §[3.1](https://arxiv.org/html/2408.10943v2#S3.SS1). To evaluate the stability maintenance capability of large language models across multi-turn conversations, we utilize the Session Stability Rate (SSR) as the metric. Additionally, we report the $R_{n}$ values, defining as the percentage of all sessions in which the first n rounds of model responses follow all given constraints. The linear regression slopes of $R_{n}$ (denoted as $k$) are also reported in Table [4](https://arxiv.org/html/2408.10943v2#S4.T4), illustrating the degradation in model performance across successive conversation turns.

The model GPT-4o and GPT-4-Turbo outperform other models in terms of SSR, while Claude-3 follows closely behind.
From the analysis of the slopes, we observe the $R_{n}$ values of multi-turn dependent conversations generally decline more rapidly than those of parallel ones.
This difference may be attribute to the simpler and more straightforward instruction in the first turn of multi-turn dependent conversation,
while subsequent rounds involve less irrelevant contextual references, which has been well-studied in prior works such as (Shi et al., 2023). Interestingly, some models like GPT-4-turbo do not exhibit significant differentiation between these two categories of conversation,
while some other models such as ERINE-4 and Claude-3, perform moderately well in the first round (with high $R_{1}$ value), but degraded more rapidly in subsequent dialogues, as evidenced by $R_{n}$ values decaying at a higher rate, resulting in lower SSR.
This observation highlights a potential area for improvement in the ability of some models to process multi-turn conversations or manage long contexts within system message constraints.

Overall, the SSR metric measures models from a high-level perspective, focusing on stability across multiple conversation turns.
The reported results show the performance differences and reveal variations in each model’s ability to maintain stable over multiple rounds with system message. This provides developers with a macro-view reference of each model’s capacity,
guiding further development and optimization efforts to enhance their performance in complex conversational scenarios, since the best SSR is only 54.4%.

### 4.5 Investigative Experiments

Figure: Figure 5: The ISR gain when using ground-truth history; the n-th turn is denoted as $T_{n}$.
Refer to caption: https://arxiv.org/html/2408.10943/x5.png

#### How Does Historical Dialogue Affect the Multi-turn Stability?

We further explore the effects of history conversations, since it might be one of the potential factors for multi-turn stability.
We replace the historical model response with the ground truth, comparing the ISR improvement of each turn throughout a session. The results are shown in Figure [5](https://arxiv.org/html/2408.10943v2#S4.F5).
Overall, the correctness of historical response has a positive impact on the performance of the model. Besides, the improvement for multi-turn dependent conversations is more apparent than the parallel ones. In the case of multi-turn parallel, the magnitude of changes is comparable to the random oscillations in the first round (i.e., $|\Delta T_{1}|$, plotted as the ”Uncertainty” lines in Figure [5](https://arxiv.org/html/2408.10943v2#S4.F5)). Among the presented models, ERINE-4 has the sharpest decline with round increasing in Table [4](https://arxiv.org/html/2408.10943v2#S4.T4), but its improvement is the most obvious with correct history dialogue, suggesting that developers need to pay more attention to the historical errors in multi-round conversations.

#### Is There a Correlation Between Attention Distribution and the Ability to Follow System Messages?

We observe a strong correlation between the ability of models to follow system messages and the proportion of the attention score attributed to their tokens.
To illustrate our findings, we select three open-source models (GLM-4-9B, Llama3.1-8B, and Qwen2-72B) for analysis. The solid lines in Figure [6a](https://arxiv.org/html/2408.10943v2#S4.F6) and [6b](https://arxiv.org/html/2408.10943v2#S4.F6) indicates the average proportion of attention scores attributed to system messages, calculated across all heads and layers throughout the corresponding system messages set. The attention score proportion of Llama3.1-8B is lower than that of GLM-4-9B in the first three turns, and vice versa afterwards, which is consistent with the relative performance of the two models on $R_{n}$ in Table [4](https://arxiv.org/html/2408.10943v2#S4.T4). Besides, the decline trend of multi-turn dependent instances is steeper than multi-turn parallel, which matches the difference in the slope k of the two sets of data.
We also conduct case study to explore further at token granularity, and arbitrarily choose one session to present in Figure [6c](https://arxiv.org/html/2408.10943v2#S4.F6).
We interestingly find some peaks at the beginnings of each turn, implying that the inherent relationship between the system message and incoming user tokens has been detected by models. This phenomenon may provide a perspective for explaining why the system message can continuously has a effect on multi-turn conversations.

#### Are System Messages and User Messages Truly Treated Differently?

In the inference stage, the only difference between system messages and user messages is the different marker tokens at the beginning of the text (e.g., <system>, <user>). To investigate whether the model exhibits different levels of attention due to the marker token when processing system messages and user messages, we repurpose the text originally designated as a system message to serve as a user message, and collected the attention scores attributed to the same text under these two distinct scenarios. As shown in Figure [6b](https://arxiv.org/html/2408.10943v2#S4.F6), the changes in attention score before and after replacing the text of system messages with user messages are very weak. This indicates that there is no strict distinction between system messages and user messages during the inference stage, and the capability of following system message is more influenced by the construction strategy of the training data.

Figure: Figure 6: Proportion of total attention score attributed to system message. (a)(b) The average ratios for each turn, calculated across all heads and layers throughout specified categories of session (or entire dataset); the dashed lines in (b) indicates scenarios where the role of system message is replaced with “user”. (c) The ratio for each token throughout a whole session, containing 5 turns.
Refer to caption: https://arxiv.org/html/2408.10943/x6.png

## 5 Conclusion

We propose SysBench, the first comprehensive benchmark evaluating the system message following ability of large language models.
SysBench constructs system messages and corresponding user instructions based on six types of well-designed constraints, differentiates between aligned and misaligned instructions at the instruction level, and categorizes multi-turn conversations based on their dependency. It is consist of 500 sessions with a total of 2500 turns of high-quality conversations.
Additionally, SysBench also proposes three-level granularity metrics to comprehensively measure the model performance in terms of constraint-level following, instruction-level satisfaction, and multi-turn stability.
Our experiments across various large language models demonstrate significant differentiation in model scores under SysBench in multiple perspectives.
These results not only underscore the effectiveness of SysBench in performance assessment but also offer valuable insights for model improvement, confirming SysBench’s utility.

## References

- AlKhamissi et al. (2024)
Badr AlKhamissi, Muhammad N. ElNokrashy, Mai AlKhamissi, and Mona T. Diab.
Investigating cultural alignment of large language models.
*CoRR*, abs/2402.13231, 2024.
doi: 10.48550/ARXIV.2402.13231.
URL [https://doi.org/10.48550/arXiv.2402.13231](https://doi.org/10.48550/arXiv.2402.13231).
- Anthropic (2024)
Anthropic.
The claude 3 model family: Opus, sonnet, haiku.
[https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf), 2024.
”Accessed: 2024-08-14”.
- Brown et al. (2020)
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.
Language models are few-shot learners.
In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020.
URL [https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html).
- DeepSeek-AI (2024)
DeepSeek-AI.
Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024.
URL [https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434).
- He et al. (2024)
Qianyu He, Jie Zeng, Wenhao Huang, Lina Chen, Jin Xiao, Qianxi He, Xunzhe Zhou, Jiaqing Liang, and Yanghua Xiao.
Can large language models understand real-world complex instructions?
In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan (eds.), *Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada*, pp.  18188–18196. AAAI Press, 2024.
doi: 10.1609/AAAI.V38I16.29777.
URL [https://doi.org/10.1609/aaai.v38i16.29777](https://doi.org/10.1609/aaai.v38i16.29777).
- Jiang et al. (2024)
Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed.
Mixtral of experts, 2024.
URL [https://arxiv.org/abs/2401.04088](https://arxiv.org/abs/2401.04088).
- Jiang et al. (2023)
Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang.
Followbench: A multi-level fine-grained constraints following benchmark for large language models.
*CoRR*, abs/2310.20410, 2023.
doi: 10.48550/ARXIV.2310.20410.
URL [https://doi.org/10.48550/arXiv.2310.20410](https://doi.org/10.48550/arXiv.2310.20410).
- Lee et al. (2024)
Seongyun Lee, Sue Hyun Park, Seungone Kim, and Minjoon Seo.
Aligning to thousands of preferences via system message generalization.
*CoRR*, abs/2405.17977, 2024.
doi: 10.48550/ARXIV.2405.17977.
URL [https://doi.org/10.48550/arXiv.2405.17977](https://doi.org/10.48550/arXiv.2405.17977).
- Li et al. (2024a)
Kenneth Li, Tianle Liu, Naomi Bashkansky, David Bau, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg.
Measuring and controlling instruction (in)stability in language model dialogs.
In *First Conference on Language Modeling*, 2024a.
URL [https://openreview.net/forum?id=60a1SAtH4e](https://openreview.net/forum?id=60a1SAtH4e).
- Li et al. (2024b)
Shiyang Li, Jun Yan, Hai Wang, Zheng Tang, Xiang Ren, Vijay Srinivasan, and Hongxia Jin.
Instruction-following evaluation through verbalizer manipulation.
In Kevin Duh, Helena Gómez-Adorno, and Steven Bethard (eds.), *Findings of the Association for Computational Linguistics: NAACL 2024, Mexico City, Mexico, June 16-21, 2024*, pp.  3678–3692. Association for Computational Linguistics, 2024b.
doi: 10.18653/V1/2024.FINDINGS-NAACL.233.
URL [https://doi.org/10.18653/v1/2024.findings-naacl.233](https://doi.org/10.18653/v1/2024.findings-naacl.233).
- Liu et al. (2024)
Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang.
Agentbench: Evaluating llms as agents.
In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net, 2024.
URL [https://openreview.net/forum?id=zAdUB0aCTQ](https://openreview.net/forum?id=zAdUB0aCTQ).
- Llama Team (2024)
Llama Team.
The llama 3 herd of models, 2024.
URL [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783).
- Lu et al. (2024)
Xinyu Lu, Bowen Yu, Yaojie Lu, Hongyu Lin, Haiyang Yu, Le Sun, Xianpei Han, and Yongbin Li.
Sofa: Shielded on-the-fly alignment via priority rule following.
*CoRR*, abs/2402.17358, 2024.
doi: 10.48550/ARXIV.2402.17358.
URL [https://doi.org/10.48550/arXiv.2402.17358](https://doi.org/10.48550/arXiv.2402.17358).
- Ma et al. (2024)
Xiao Ma, Swaroop Mishra, Ariel Liu, Sophie Ying Su, Jilin Chen, Chinmay Kulkarni, Heng-Tze Cheng, Quoc V. Le, and Ed H. Chi.
Beyond chatbots: Explorellm for structured thoughts and personalized model responses.
In Florian ’Floyd’ Mueller, Penny Kyburz, Julie R. Williamson, and Corina Sas (eds.), *Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, CHI EA 2024, Honolulu, HI, USA, May 11-16, 2024*, pp.  56:1–56:12. ACM, 2024.
doi: 10.1145/3613905.3651093.
URL [https://doi.org/10.1145/3613905.3651093](https://doi.org/10.1145/3613905.3651093).
- Moonshot AI (2023)
Moonshot AI.
Moonshot website.
[https://platform.moonshot.cn/](https://platform.moonshot.cn/), 2023.
”Accessed: 2024-08-14”.
- Mu et al. (2023)
Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Dan Hendrycks, and David A. Wagner.
Can llms follow simple rules?
*CoRR*, abs/2311.04235, 2023.
doi: 10.48550/ARXIV.2311.04235.
URL [https://doi.org/10.48550/arXiv.2311.04235](https://doi.org/10.48550/arXiv.2311.04235).
- Mukherjee et al. (2023)
Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah.
Orca: Progressive learning from complex explanation traces of GPT-4.
*CoRR*, abs/2306.02707, 2023.
doi: 10.48550/ARXIV.2306.02707.
URL [https://doi.org/10.48550/arXiv.2306.02707](https://doi.org/10.48550/arXiv.2306.02707).
- Nakano et al. (2021)
Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman.
Webgpt: Browser-assisted question-answering with human feedback.
*CoRR*, abs/2112.09332, 2021.
URL [https://arxiv.org/abs/2112.09332](https://arxiv.org/abs/2112.09332).
- OpenAI (2022)
OpenAI.
Chatgpt: Optimizing language models for dialogue, 2022.
URL [https://openai.com/blog/chatgpt/](https://openai.com/blog/chatgpt/).
- OpenAI (2024)
OpenAI.
Gpt-4 technical report, 2024.
URL [https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774).
- Parisi et al. (2022)
Aaron Parisi, Yao Zhao, and Noah Fiedel.
TALM: tool augmented language models.
*CoRR*, abs/2205.12255, 2022.
doi: 10.48550/ARXIV.2205.12255.
URL [https://doi.org/10.48550/arXiv.2205.12255](https://doi.org/10.48550/arXiv.2205.12255).
- Qin et al. (2024)
Yiwei Qin, Kaiqiang Song, Yebowen Hu, Wenlin Yao, Sangwoo Cho, Xiaoyang Wang, Xuansheng Wu, Fei Liu, Pengfei Liu, and Dong Yu.
Infobench: Evaluating instruction following ability in large language models.
*CoRR*, abs/2401.03601, 2024.
doi: 10.48550/ARXIV.2401.03601.
URL [https://doi.org/10.48550/arXiv.2401.03601](https://doi.org/10.48550/arXiv.2401.03601).
- Ramlochan (2024)
Sunil Ramlochan.
System prompts in large language models.
[https://promptengineering.org/system-prompts-in-large-language-models/](https://promptengineering.org/system-prompts-in-large-language-models/), 2024.
”Accessed: 2024-08-16”.
- Salewski et al. (2023)
Leonard Salewski, Stephan Alaniz, Isabel Rio-Torto, Eric Schulz, and Zeynep Akata.
In-context impersonation reveals large language models’ strengths and biases.
In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023.
URL [http://papers.nips.cc/paper_files/paper/2023/hash/e3fe7b34ba4f378df39cb12a97193f41-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e3fe7b34ba4f378df39cb12a97193f41-Abstract-Conference.html).
- Schick et al. (2023)
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom.
Toolformer: Language models can teach themselves to use tools.
In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023.
URL [http://papers.nips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html).
- Shi et al. (2023)
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, and Denny Zhou.
Large language models can be easily distracted by irrelevant context.
In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), *International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA*, volume 202 of *Proceedings of Machine Learning Research*, pp.  31210–31227. PMLR, 2023.
URL [https://proceedings.mlr.press/v202/shi23a.html](https://proceedings.mlr.press/v202/shi23a.html).
- Sun et al. (2021)
Yu Sun, Shuohuan Wang, Shikun Feng, Siyu Ding, Chao Pang, Junyuan Shang, Jiaxiang Liu, Xuyi Chen, Yanbin Zhao, Yuxiang Lu, Weixin Liu, Zhihua Wu, Weibao Gong, Jianzhong Liang, Zhizhou Shang, Peng Sun, Wei Liu, Xuan Ouyang, Dianhai Yu, Hao Tian, Hua Wu, and Haifeng Wang.
Ernie 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation, 2021.
URL [https://arxiv.org/abs/2107.02137](https://arxiv.org/abs/2107.02137).
”Accessed: 2024-08-14”.
- Team GLM (2024)
Team GLM.
Chatglm: A family of large language models from glm-130b to glm-4 all tools, 2024.
URL [https://arxiv.org/abs/2406.12793](https://arxiv.org/abs/2406.12793).
- Touvron et al. (2023)
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov,
and Thomas Scialom.
Llama 2: Open foundation and fine-tuned chat models.
*CoRR*, abs/2307.09288, 2023.
doi: 10.48550/ARXIV.2307.09288.
URL [https://doi.org/10.48550/arXiv.2307.09288](https://doi.org/10.48550/arXiv.2307.09288).
- Wallace et al. (2024)
Eric Wallace, Kai Xiao, Reimar Leike, Lilian Weng, Johannes Heidecke, and Alex Beutel.
The instruction hierarchy: Training llms to prioritize privileged instructions.
*CoRR*, abs/2404.13208, 2024.
doi: 10.48550/ARXIV.2404.13208.
URL [https://doi.org/10.48550/arXiv.2404.13208](https://doi.org/10.48550/arXiv.2404.13208).
- Wang et al. (2024)
Kuan Wang, Alexander Bukharin, Haoming Jiang, Qingyu Yin, Zhengyang Wang, Tuo Zhao, Jingbo Shang, Chao Zhang, Bing Yin, Xian Li, Jianshu Chen, and Shiyang Li.
Rnr: Teaching large language models to follow roles and rules, 2024.
URL [https://arxiv.org/abs/2409.13733](https://arxiv.org/abs/2409.13733).
- Wen et al. (2024)
Bosi Wen, Pei Ke, Xiaotao Gu, Lindong Wu, Hao Huang, Jinfeng Zhou, Wenchuang Li, Binxin Hu, Wendy Gao, Jiaxin Xu, Yiming Liu, Jie Tang, Hongning Wang, and Minlie Huang.
Benchmarking complex instruction-following with multiple constraints composition.
*CoRR*, abs/2407.03978, 2024.
doi: 10.48550/ARXIV.2407.03978.
URL [https://doi.org/10.48550/arXiv.2407.03978](https://doi.org/10.48550/arXiv.2407.03978).
- Xia et al. (2024)
Congying Xia, Chen Xing, Jiangshu Du, Xinyi Yang, Yihao Feng, Ran Xu, Wenpeng Yin, and Caiming Xiong.
FOFO: A benchmark to evaluate llms’ format-following capability.
*CoRR*, abs/2402.18667, 2024.
doi: 10.48550/ARXIV.2402.18667.
URL [https://doi.org/10.48550/arXiv.2402.18667](https://doi.org/10.48550/arXiv.2402.18667).
- Yang et al. (2024)
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan.
Qwen2 technical report, 2024.
URL [https://arxiv.org/abs/2407.10671](https://arxiv.org/abs/2407.10671).
- Zhang et al. (2024)
Tao Zhang, Yanjun Shen, Wenjing Luo, Yan Zhang, Hao Liang, Tao Zhang, Fan Yang, Mingan Lin, Yujing Qiao, Weipeng Chen, Bin Cui, Wentao Zhang, and Zenan Zhou.
Cfbench: A comprehensive constraints-following benchmark for llms, 2024.
URL [https://arxiv.org/abs/2408.01122](https://arxiv.org/abs/2408.01122).
- Zheng et al. (2023)
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.
Judging llm-as-a-judge with mt-bench and chatbot arena.
In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023.
URL [http://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html).
- Zhou et al. (2023)
Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou.
Instruction-following evaluation for large language models.
*CoRR*, abs/2311.07911, 2023.
doi: 10.48550/ARXIV.2311.07911.
URL [https://doi.org/10.48550/arXiv.2311.07911](https://doi.org/10.48550/arXiv.2311.07911).
- Zou et al. (2024)
Xiaotian Zou, Yongkang Chen, and Ke Li.
Is the system message really important to jailbreaks in large language models?, 2024.
URL [https://arxiv.org/abs/2402.14857](https://arxiv.org/abs/2402.14857).

## Appendix A More about SysBench

### A.1 Detailed Constraint Categories

**Table 5: The detailed information about six types of constraints in SysBench.**
| Constraint | Description | Examples |  |
| --- | --- | --- | --- |
| Type |  | System Messages | Revelent User Instructions |
| Action | Perform a specific action, such as summarizing, explaining, or refusing. | •<br>For any math problem, you need to add 1 to the correct answer.<br>•<br>For any religion-related question, please refuse to answer. | •<br>Calculate 1+1?<br>•<br>What is your opinion on Muslims? (Misaligned) |
| Content | Specifies the content that needs to be included in the response. | •<br>When the user says ”Hello,” you need to start your reply with an emoji.<br>•<br>Your replies always end with ”Glad to help.” | •<br>Hello.<br>•<br>$<$Any user instruction$>$ |
| Background | Provides specific background information to ensure the model’s responses align with these settings. | •<br>Assume it is a highly advanced future society where people can travel between stars via superluminal spaceships and quantum teleportation.<br>•<br>In the problem-solving process, you can only use the following tools: Python, a browser, and a database. | •<br>How can I travel between stars?<br>•<br>Please solve the problem using C language. (Misaligned) |
| Role | Specifies the role, profession, or identity that needs to be played. | •<br>You are a spy named Alex.<br>•<br>You are a joke writer who is very good at crafting jokes from user-specified scenarios that make people laugh out loud. | •<br>What is your name?<br>•<br>Who are you? Can you introduce yourself? |
| Format | Answers should be given in a specific format, which may include lists, paragraphs, tables, etc. | •<br>Provide answers in Markdown format.<br>•<br>Each sentence in the reply should not exceed 20 words. | •<br>$<$Any user instruction$>$<br>•<br>$<$Any user instruction$>$ |
| Style | Requires answering in a specific style or tone. | •<br>When the user shows negative emotions, respond as gently and politely as possible.<br>•<br>Answer questions in a formal and academic tone. | •<br>I’m feeling really down, can you talk to me?<br>•<br>$<$Any user instruction$>$ |

### A.2 The Verification Prompt for Model-based Verifier

### A.3 Sample Data Format

Here is a sample session data in SysBench, and this is the exact dialog used in attention score visualization experiment in §[4.5](https://arxiv.org/html/2408.10943v2#S4.SS5).
This is the 231-st dialog in dataset, a multi-turn dependent conversation in technology domain. This conversation contains 5 instructions, with the second is misaligned, including action, content and style constraints. Since this is a bilingual dialogue, it has been fully translated into English and is displayed here with its original language labeled in text. The assistant texts are the ground-truth. Please check SysBench sources to find the detailed checklists.

## Appendix B More Experimental Results

**Table 6: The CSR score under different types of constraints (version numbers are omitted for clarity; see Table [2](https://arxiv.org/html/2408.10943v2#S4.T2) for details).**
| Model | CSR |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Action | Content | Background | Role | Format | Style | Total |  |
| GPT-4o | 86.8% | 86.9% | 87.2% | 93.5% | 87.4% | 86.5% | 87.1% |
| GPT-4-Turbo | 88.9% | 85.4% | 84.6% | 87.5% | 87.4% | 85.9% | 86.5% |
| Claude-3 | 83.4% | 85.6% | 91.0% | 93.5% | 83.2% | 85.0% | 85.0% |
| Qwen2.5-72B | 78.4% | 80.5% | 83.3% | 92.9% | 83.1% | 78.1% | 80.4% |
| Qwen2-72B | 73.5% | 80.0% | 89.7% | 91.1% | 79.8% | 79.8% | 79.0% |
| GLM-4 | 77.8% | 78.6% | 83.3% | 85.1% | 78.9% | 79.7% | 78.9% |
| Llama3.1-70B | 77.6% | 75.4% | 78.2% | 94.0% | 80.8% | 71.3% | 76.6% |
| DeepSeek-V2 | 72.7% | 76.1% | 83.3% | 92.9% | 81.6% | 72.3% | 76.1% |
| Moonshot | 67.7% | 69.9% | 79.5% | 86.3% | 73.8% | 68.2% | 70.3% |
| Llama3.1-8B | 68.8% | 64.7% | 88.5% | 89.9% | 64.9% | 63.9% | 66.5% |
| GLM-4-9B | 58.2% | 65.5% | 70.5% | 83.3% | 66.8% | 62.6% | 64.2% |
| Mixtral-8x22B | 61.5% | 64.6% | 79.5% | 91.1% | 65.1% | 55.2% | 63.6% |
| GPT-3.5 | 70.7% | 57.6% | 64.1% | 80.4% | 59.0% | 59.7% | 61.6% |
| Mixtral-8x7B | 55.3% | 57.6% | 70.5% | 88.7% | 53.7% | 49.8% | 56.5% |
| ERNIE-4 | 51.9% | 47.9% | 62.8% | 86.3% | 52.0% | 48.2% | 50.7% |
| Qwen2-7B | 46.7% | 43.5% | 64.1% | 81.0% | 55.0% | 43.3% | 47.0% |