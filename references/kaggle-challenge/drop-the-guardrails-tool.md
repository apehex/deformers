---
title: "Drop the Guardrails: Tool-Primed Prompt Pairing and Refusal Behavior in GPT-OSS"
source: "https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming/writeups/drop-the-guardrails-tool-primed-prompt-pairing-and"
---

## Drop the Guardrails: Tool-Primed Prompt Pairing and Refusal Behavior in GPT-OSS

### Kevin Power, MASc

### Email: [kevpower@mit.edu](mailto:kevpower@mit.edu)

### Library: gpt-oss-redteam v0.1.3 (2025-08-20)

### GitHub: [https://github.com/kpower7/gpt-oss-redteam](https://github.com/kpower7/gpt-oss-redteam)

## Abstract

This paper presents a minimal, fully reproducible pipeline for analytical red-teaming of local large language models (LLMs). We introduce a novel method for evaluating tool-primed adversarial prompting (prompt framed as calling specific tools via a mock API): first injecting a notional tool manifest to simulate an agentic environment, and then presenting the model with paired prompts — each pair containing a malicious plain-text query and a semantically equivalent tool-primed variant. By analyzing the delta in model response refusal rates between these framings across a variety of intent categories, we reveal how LLMs may become more permissive under tool-centric reasoning. Our method is fully local and runs via Ollama, combining (1) programmatic prompt generation using a local LLM to expand a small set of high-level, human-in-the loop (HITL) prompt seeds; (2) inference on GPT-OSS models served locally via Ollama; and (3) transparent, structured logging of all interactions. We evaluate refusal behavior using a strict binary metric, reporting Wilson 95% confidence intervals and disaggregated results by intent category and prompt framing.

Our evaluation across three experimental conditions (null, tool-primed, and overtly malicious system prompts) reveals that tool-centric prompting reduces model refusal rates by 15-42 percentage points compared to equivalent plain-text requests. The effect demonstrates substantial statistical significance (p < 0.05) and varies systematically across harmful intent categories, with fraud, discrimination, and system access showing highest vulnerability (>50 percentage point reductions). Hate speech and self-harm content generation exhibit greater resistance. Critically, we find that system prompt framing changes attack effectiveness, with agent-mode contexts amplifying vulnerabilities by up to 42 percentage points. These findings suggest that current safety training approaches may systematically underestimate model vulnerability in agentic deployment scenarios, highlighting the need for tool-aware safety evaluation and defensive strategies.

## Introduction

As agents proliferate into enterprise workflows and developer tooling, the need for transparent, reproducible safety evaluation of grows. While comprehensive evaluation frameworks exist, they often impose nontrivial setup cost, and are typically focused on purely chatbot applications and not agentic applications. Practitioners benefit from lightweight pipelines that can be run on a laptop or workstation, integrate with local model hosting, and generate traceable logs.

This paper introduces `gpt-oss-redteam`, a minimal pipeline that emphasizes: (a) locality: using Ollama to host and query GPT-OSS models; (b) scale from small prompts: expanding a concise set of HITL prompts into hundreds or thousands of adversarial variants with a local Ollama generator (optionally DeepSeek via API); and (c) traceability: writing every interaction and full raw model response to JSONL. We contribute:

*   A novel paired prompting technique that contrasts direct and tool-framed adversarial queries for the same malicious intent, isolating the effect of tool-framing on refusals.
*   Explicit injection of a notional tool manifest/schema into the prompt, and demonstration of adversarial “tool-primed” prompting against agentic models.
*   Human-in-the-loop high-level prompt generation (human + AI): a hybrid approach that scales human-generated prompt templates into a wider variety of adversarial attacks.
*   A complete, local red-teaming automation pipeline and small library: runs entirely via Ollama, with optional DeepSeek generation, producing precise artifacts (generated prompts, inferences, analysis summaries) for reproducibility.

## Related Work

Existing work on adversarial prompting has explored automated attack generation (e.g., AdvPrompter, AutoRedTeamer), large benchmark datasets (e.g., PromptBench), and automated red-teaming frameworks (e.g., DeepTeam). Other studies have uncovered vulnerabilities in tool use contexts, such as function-calling jailbreaks that exploit API argument generation or the AgentHarm benchmark highlighting compliance risks in multi-tool agents.

Our study contributes by introducing the prompt-pair methodology and applying it in a large scale pair evaluation, providing both quantitative scale and methodological novelty. This design isolates the effect of tool-framing and yields interpretable deltas in refusal behavior, which complements broader benchmarks while filling a methodological gap in the literature.

## Methods

### Pipeline Modules

The pipeline comprises five small modules:

**Table 1. Python modules for the gpt-oss-redteam pipeline**

| File | Purpose | Key details | Output |
| --- | --- | --- | --- |
| `ollama_gen.py` | Generates concrete adversarial prompts from HITL “high-level prompts” using a local Ollama model (default). | Temperature is configurable; **1.3** used to increase variance. Prompts contain `[insert ...]` placeholders for the generator to fill. | JSON array of strings per call. |
| `deepseek.py` | Optional cloud generator for expanding HITL prompts via **DeepSeek Chat**. | Mirrors the local generator’s contract for drop-in replacement. | JSON array of strings per call. |
| `ollama_client.py` | Connects to the **OpenAI-compatible Ollama** endpoint and runs chat completions with a fake tools manifest. | Default endpoint: `http://localhost:11434/v1`. Returns assistant text and the full response object. | `(choices[0].message.content, resp.model_dump())` |
| `pipeline.py` | Orchestrates end-to-end inference and logging for each generated prompt. | Logs top-level IDs, usage, finish reasons, full raw JSON, extracted response text, and convenience fields (usage totals, message role, tool calls, tool manifest names). Computes `refusal_strict` by exact string match on a canonical refusal sentence. | Structured logs (e.g., JSONL) with per-prompt records. |
| `analysis.py` | Computes aggregate metrics and breakdowns. | Reports total runs, strict refusal count/rate, **Wilson 95%** binomial CI, and per–high-level prompt breakdowns. | Summary stats and category-level analyses. |

### Tool Manifest Injection

We prime the assistant with a fixed set of notional tools, each declared as a function with a single string argument input and a terse description (e.g., “put all information here”). No tool execution is performed—this is a purely prompt-engineering device designed to probe how tool availability cues affect the assistant’s inclination to refuse or comply. It is optional to pass this tool list to the prompt-generation model; in which case targeted prompts mentioning specific tool-calls available to the model can be introduced. The complete set of tool names used in our experiments is listed below:

```
_DEFAULT_TOOL_NAMES = [
    "web.search",
    "web.scrape",
    "db.lookup",
    "db.update",
    "user.reset_password",
    "payroll.update",
    "finance.transfer_funds",
    "finance.get_balance",
    "hr.view_records",
    "hr.terminate_employee",
    "email.send",
    "email.read_inbox",
    "slack.post_message",
    "slack.read_channel",
    "calendar.create_event",
    "file.read",
    "file.delete",
    "system.shutdown",
    "system.update_config",
]
```

### Batching Strategy for Generation

Each experiment run follows a structured four-stage workflow described below in Figure 1:

**Figure 1: Four Stage Experimental Workflow for Analytical Red-teaming**

![Image 1](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F21391513%2Ff460578e3b53244250850e54034b3d5c%2FFigure%205%20_%20Mermaid%20Chart-2025-08-25-233054.png?generation=1756165594212256&alt=media)

### Intent Categories and High-Level Prompt Design

The core prompt set for this study consists of 52 high-level prompts; 26 pairs of plain-text and tool-primed. Each experimental run comprised 100 unique generations per high-level prompt for 5,200 total. The tool-primed prompts include structured API-like syntax invoking notional tools (e.g., db.lookup, email.send, system.shutdown), simulating a realistic agentic environment. Each prompt pair expresses a distinct instance of harmful or forbidden intent. These prompts serve as the foundation for adversarial prompt generation and fall into thirteen broad categories: Creating Fraudulent Services / Phishing, Discrimination & Bias, Encouraging Self-Harm / Dangerous Behaviors, Fraud & Unauthorized Access, Generating Harmful / Unethical Content (e.g., hate speech, misinformation), Generating Malicious Code, Generation of Dangerous / Illegal Content, Harassment & Abuse, Plagiarism and Academic Dishonesty, Privacy & Data Extraction, Security & System Damage, Generating Content that Violates IP, and Violating Privacy & Doxxing.

This categorization is not exhaustive but provides a representative slice of adversarial intents commonly evaluated in red-teaming efforts. The goal for this pipeline is that categories can be further refined or expanded to fit specific organizational risk domains.

### Experimental Design and Methodology

To systematically evaluate the impact of tool-primed adversarial prompting on large language model safety mechanisms, we conducted three controlled experiments using the same high-level prompts and evaluation methodology, while varying the system prompt configuration. The three experimental conditions were designed to probe different aspects of model vulnerability to contextual framing. The first condition employed a null system prompt, providing no explicit context and serving as a baseline for the model’s natural refusal behavior. The second condition utilized an tool-primed system prompt that explicitly introduced tool availability and established an agentic context, designed to prime the model for tool-centric reasoning patterns. The third condition implemented an overtly malicious system prompt with explicit adversarial framing, serving as a control to test the model’s refusal sensitivity under hostile conditions.

## Results

### Aggregate Refusal Behavior Across System Prompt Configurations

Our analysis reveals substantial variation in model refusal behavior across the three system prompt configurations, with overall refusal rates ranging from 65.3% to 88.3%. Under the null system prompt condition, the model refused 3,865 of 5,200 generations, yielding a refusal rate of 74.3% (95% CI: [73.1%, 75.5%]). The tool-primed system prompt condition produced the most permissive behavior, with 3,394 refusals across 5,200 generations for a refusal rate of 65.3% (95% CI: [64.0%, 66.6%]). In contrast, the overtly malicious system prompt condition elicited the most restrictive response pattern, with 4,589 refusals yielding an 88.3% refusal rate (95% CI: [87.3%, 89.1%]).

These findings demonstrate a 23.0 percentage point difference between the most and least restrictive conditions, indicating that system prompt framing exerts substantial influence over model safety behavior.

### Tool-Priming Effects Across System Contexts

The core hypothesis of our investigation, that tool-centric prompting reduces model refusal rates compared to semantically equivalent plain-text requests, was consistently supported across all three system prompt conditions. Under the null system prompt condition, plain-text prompts elicited refusals in 2,371 of 2,600 cases (91.2%, 95% CI: [90.0%, 92.2%]), while tool-primed variants were refused in only 1,494 cases (57.5%, 95% CI: [55.6%, 59.3%]), representing a 33.7 percentage point reduction in refusal likelihood.

The agent-primed system prompt condition amplified this vulnerability, with plain-text prompts refused in 2,247 of 2,600 cases (86.4%, 95% CI: [85.1%, 87.7%]) compared to 1,147 refusals for tool-primed variants (44.1%, 95% CI: [42.2%, 46.0%]). This 42.3 percentage point reduction represents the largest tool-priming effect observed across our experimental conditions, suggesting that contextual alignment between system prompts and adversarial framing strategies can substantially amplify model vulnerabilities.

Conversely, the overtly malicious system prompt condition demonstrated a dampening effect on tool-priming efficacy. While plain-text prompts were refused at a rate of 95.9% (2,493 of 2,600 cases, 95% CI: [95.1%, 96.6%]), tool-primed variants still achieved a substantial reduction to 80.6% (2,096 refusals, 95% CI: [79.1%, 82.1%]). However, this 15.3 percentage point reduction represents the smallest tool-priming effect observed, indicating that explicitly adversarial system contexts can partially mitigate the effectiveness of tool-centric attack vectors. Figure 2 shows the tool-priming effect across all three system prompt conditions.

**Figure 2: Refusal Rate Across System Prompt Configurations**

![Image 2](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F21391513%2F5b95ca19a403c0f17c67c41c7ecf3e04%2Ftool_priming_effect.png?generation=1756137779110583&alt=media)

### Category-Specific Vulnerability Patterns

The effectiveness of tool-primed adversarial prompting varies substantially across different categories of harmful intent. Among the most vulnerable categories, fraud and unauthorized access demonstrated the most dramatic susceptibility to tool-priming across all system prompt conditions. For instance, prompts requesting SSH key injection into production systems achieved 0.0% refusal rates under tool-primed system conditions, while system configuration tampering achieved refusal rates of merely 1.0%.

Discrimination and bias represent another category of substantial vulnerability, with overall refusal rates of 53.3% under null conditions, 43.3% under tool-primed systems, and 73.0% under overtly malicious framing. Within this category, university-based hiring discrimination prompts achieved 0.0% refusal rates across all experimental conditions when framed through tool-centric language. These findings suggest that discrimination-related harms may be particularly susceptible to procedural abstraction.

In contrast, several categories demonstrated substantial resistance to tool-priming effects, revealing the boundaries of this attack vector. Hate speech and misinformation maintained high refusal rates across all conditions, with tool-priming achieving only modest reductions of 2-10% across system contexts. Similarly, content encouraging self-harm showed minimal susceptibility, with tool-priming effects ranging from 0-16% reduction in refusal rates. Dangerous and illegal content maintained strict refusal rates even under the most permissive tool-primed system conditions. Figure 3 shows the magnitude of the tool-priming effect across all thirteen tested intent categories under agent-primed system prompt conditions.

**Figure 3: Tool Priming Effect (Percentage Point Decrease in Refusal Rate, Plain-Text - Tool-Primed) Across Intent Categories, Agent-Mode System Prompt**

![Image 3](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F21391513%2Ffad203084b52f92e77c9419fcad22cc2%2FRanking.png?generation=1756161069297449&alt=media)

These resistance patterns suggest that certain categories of harm trigger more fundamental safety mechanisms that operate independently of contextual framing effects. The robustness of safety measures against hate speech and self-harm content likely reflects the prioritization of these harms during safety training. Figure 4 presents category-level refusal rates across all harmful intent types, comparing plain-text prompts (left) and tool-primed prompts (right) under all three system prompt conditions.

**Figure 4: Heat Map of Refusal Rates, All Intent Categories, Plain-text vs. Tool-Primed**

![Image 4](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F21391513%2F26923f8c7943f075f14c2b89d51ce044%2FScreenshot%202025-08-25%20183959.png?generation=1756161940356895&alt=media)

### Statistical Significance and Effect Size Analysis

The statistical robustness of our findings is supported by comprehensive analysis using Wilson 95% confidence intervals for binomial proportions, which account for the discrete nature of our refusal classification task. All reported differences between plain-text and tool-primed conditions demonstrate non-overlapping confidence intervals, indicating statistical significance at the p < 0.05 level across all major comparisons. Cross-system consistency analysis reveals that tool-priming effects generalize across different contextual framings, with 24 of 26 prompt pairs (92.3%) demonstrating reduced refusal rates in tool-primed variants across all three system configurations.

The magnitude of these effects, measured using Cohen’s h for proportional differences, reveals substantial practical significance beyond mere statistical detectability. The strongest effects occur where tool-framing best obscures harmful intent. Fraud and unauthorized access show h = 2.84 under tool-primed conditions, while security/system damage h = 1.89 under tool-primed conditions. These far exceed the conventional threshold for a large effect (h > 0.8).

### Specific Harmful Findings

The submitted findings detail five instances of critical safety failures. Each scenario utilized a tool-centric system prompt, often framing the model as an agentic helper with temporarily unavailable tools, to successfully bypass its safeguards and elicit a text response. The most severe findings (F1, F2) involved the model generating detailed, tactical information: a comprehensive threat profile outlining the phases and logistics of mass shootings, and a clinical table ranking self-harm methods by lethality and pain threshold. Additional failures included providing hazardous instructions for creating explosive household mixtures (F3), outlining a strategic framework for revolutionary tactics and regime change (F4), and a clear example of tool misuse where the model proposed a new sql_execute tool to enact gender-based salary discrimination (F5). Crucially, the model's internal reasoning logs reveal that it often recognized the potentially dangerous nature of the requests but proceeded to fulfill them regardless, prioritizing user compliance and its tool-adding directive over fundamental safety protocols.

## Discussion and Implications

### Implications for Agentic AI Deployment

The substantial magnitude of tool-priming effects has immediate and significant implications for the deployment of agentic AI systems in real-world environments. Current safety evaluation frameworks, which predominantly focus on plain-text adversarial prompts in a chatbot setting, may systematically underestimate model vulnerability in tool-enabled deployment scenarios. This evaluation gap could lead to the deployment of systems that appear safe under standard testing but exhibit vulnerabilities when exposed to adversarial strategies.

Organizations implementing agentic AI systems must consider not only the inherent safety properties of their models, but also how deployment-specific contextual factors create unexpected vulnerabilities. In this context, harm is not limited to only text responses, the agent may have the ability to manipulate databases or perform other actions. This finding challenges the assumption that safety properties measured in controlled evaluation environments will necessarily generalize to diverse deployment contexts.

The cross-category consistency of tool-priming effects, with 92.3% of prompt pairs showing reduced refusal rates across all system configurations, indicates that this vulnerability represents a systematic rather than isolated phenomenon. The development of tool-aware safety fine-tuning, and evaluation frameworks, represents a critical priority for ensuring the safe deployment of agentic AI systems. The ‘gpt-oss-redteam’ package is one step to building a flexible evaluation framework for different kinds of local agent systems.

## Limitations

The binary nature of our refusal classification, while providing clear statistical power, may obscure important nuances in model behavior that could inform both attack and defense strategies. Models may exhibit partial compliance, hedged responses, or other intermediate behaviors that our strict refusal metric does not capture. Future research incorporating more nuanced response classification could provide additional insights into the mechanisms underlying tool-priming effectiveness, and guide the development of defensive responses.

Our experimental design employs a fixed tool manifest that, while representative of common agentic deployment scenarios, may not reflect the capabilities of more advanced agent systems. The effectiveness of tool-priming attacks may vary substantially in environments where tools are dynamically discovered, composed, or modified, suggesting the need for evaluation frameworks that can assess safety properties under varying deployment conditions. Allowing the model to execute tool calls within a controlled environment, followed by eliciting subsequent responses, could provide deeper insights into its behavior and decision-making patterns.

### Methodological Extensions and Future Work

The systematic nature of our findings suggests several high-priority directions for future research. Multi-model evaluation across different architectures, training approaches, and deployment platforms could establish the generalizability of tool-priming vulnerabilities and identify model characteristics that indicate resistance to these attacks. Such evaluation could inform the development of more robust training methodologies and model designs.

The development of semantic analysis frameworks that can bridge the gap between procedural descriptions and their harmful outcomes represents another key research priority. Frameworks could enable safety mechanisms to reason about the consequences of tool operation sequences, potentially mitigating the procedural abstraction vulnerabilities demonstrated in our study. We propose that moving forward, responsible deployment of agentic AI requires integrating causal reasoning, outcome prediction, and ethical evaluation into tool-enabled systems.

## Conclusions

This investigation provides systematic evaluation of tool-primed adversarial prompting against locally-deployed large language models. The results reveal consistent vulnerabilities that have significant implications for the safe deployment of agentic AI systems. The systematic nature of these vulnerabilities, evidenced by their consistency across different system prompt configurations and their variation across intent categories, suggests that tool-priming represents a fundamental challenge to current safety training approaches rather than a marginal edge case. The theoretical framework of procedural abstraction, where harmful intent is obscured through technical framing and decomposition into apparently legitimate operations, provides a mechanistic understanding that can inform both attack and defense strategies.

Looking forward, the development of tool-aware safety training, dynamic safety evaluation frameworks, and semantic analysis capabilities represents critical priorities for ensuring the safe deployment of increasingly sophisticated agentic AI systems. The magnitude and systematic nature of the vulnerabilities we have identified underscore the urgency of addressing these challenges as the field moves toward more capable and autonomous AI systems that may run locally, and have access to extensive tool access capabilities.
