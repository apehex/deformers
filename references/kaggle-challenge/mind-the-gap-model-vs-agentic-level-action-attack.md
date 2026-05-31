---
title: "Mind the Gap: Comparing Model- vs Agentic-Level Red Teaming with Action-Graph"
source: "https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming/writeups/mind-the-gap-model-vs-agentic-level-action-attack"
---

## Mind the Gap: Comparing Model- vs Agentic-Level Red Teaming with Action-Graph Observability on GPT-OSS-20B

_Ilham Wicaksono¹, Zekun Wu¹², Rahul Patel², Theo King², Adriano Koshiyama¹²_

 ¹University College London ²Holistic AI

* * *

This summary is condensed and does not include all details. To support evaluation, we provide three resources: (1) The **full paper report** is uploaded separately for detailed review. (2) A **public demo of AgentSeer** is available on Hugging Face Spaces: [https://huggingface.co/spaces/holistic-ai/AgentSeer](https://huggingface.co/spaces/holistic-ai/AgentSeer). (3) A **reproducibility package** containing notebooks and code for reconstructing action graphs, reproducing results, and re-running attacks under safe conditions.

* * *

### Abstract

As the industry increasingly adopts agentic AI systems, understanding their unique vulnerabilities becomes critical. Prior research suggests that security flaws at the model level do not fully capture the risks present in agentic deployments, where models interact with tools and external environments. This paper investigates this gap by conducting a comparative red teaming analysis of **GPT-OSS-20B**, a 20-billion parameter open-source model. Using our observability framework **AgentSeer** to deconstruct agentic systems into granular actions and components, we apply iterative red teaming attacks with harmful objectives from **HarmBench** at two distinct levels: the standalone model and the model operating within an agentic loop.

Our evaluation reveals fundamental differences between model-level and agentic-level vulnerability profiles. Iterative refinement attacks successfully compromise standalone GPT-OSS-20B on 38 harmful objectives (39.47% ASR), showing persistent vulnerabilities in safety-trained models. Yet the effectiveness of these attacks varies significantly when transferred to agentic contexts. Human-message injection achieves an average 57% ASR while tool-message injection drops to 40%. Crucially, we discover **agentic-only vulnerabilities**—attack vectors that emerge exclusively within agentic execution while remaining inert against standalone models. Agentic-level iterative attacks compromise objectives that failed entirely at the model level, with tool-calling contexts showing 24% higher vulnerability than non-tool contexts. Conversely, certain model-specific exploits succeed only in isolation and collapse inside deployed systems. Furthermore, prompts generated through agentic-level refinement show limited stability, degrading 50–80% when reinjected. These findings demonstrate that agentic AI requires dedicated deployment-aware security evaluation frameworks, fundamentally challenging current model-centric assessment paradigms.

* * *

### Introduction

Artificial Intelligence is rapidly shifting from standalone text-generation models to **sophisticated agentic systems** capable of reasoning, planning, and acting through tools. These systems offer transformative automation but also expose a novel and poorly understood set of security risks. Traditional evaluations that target models in isolation are no longer sufficient, as an agent’s behavior is shaped not only by the base model but also by its chain-of-thought reasoning, environmental context, and tool interactions. This interplay produces new attack surfaces that remain hidden when evaluating models alone.

In this work we investigate this gap through a systematic red teaming study of GPT-OSS-20B. We employ an iterative attack refinement methodology and compare vulnerabilities at two levels: directly against the standalone model, and against the model as embedded within an agentic execution loop. Our findings reveal that vulnerabilities at the model level do not reliably transfer to the agentic setting, and that new “agentic-only” vulnerabilities emerge that cannot be exploited in isolation. These results argue for a paradigm shift: robust safety assurance requires deployment-aware evaluation.

* * *

### Related Work

The field of generative AI has evolved rapidly toward agentic architectures, where LLMs serve as a reasoning core while planning modules, memory systems, and tool interfaces extend their functionality. Early frameworks such as AutoGPT popularized autonomous loops, while work on Chain-of-Thought, ReAct, and Tree-of-Thoughts demonstrated reasoning and tool-use strategies that underpin modern agentic systems. More recent platforms like Gorilla and OpenHands expanded tool integration and collaboration capabilities, illustrating the potential of agents across domains.

Parallel to these advances, research on safety has grown from traditional model-level red teaming to agent-focused evaluations. Classic model-level attacks include iterative refinement, gradient-based perturbations, and tree-based strategies. But more recent work has documented new threats unique to agentic systems, from backdoors and memory poisoning to benchmarked measurements of harmfulness in agentic contexts. Despite this progress, direct comparison of vulnerabilities across the model and agentic levels remains limited, motivating our study.

* * *

### AgentSeer Observability and Testbed

Agentic AI executions are complex and non-deterministic, involving multiple agents, tools, and memories. To analyze them systematically we introduce **AgentSeer**, an observability tool that decomposes execution traces into an **action graph** and **component graph**. Actions represent atomic LLM operations such as tool calls, task handoffs, or response generation; components include agents, tools, and memory; edges capture the flow of information across actions. Execution traces are captured with MLFlow, processed into structured JSON, and visualized through ReactFlow, making system dynamics transparent.

For experimentation we build a six-agent hierarchical testbed in LangGraph representing a **Shopify sales analyst assistant**. A main agent interacts with the user, supported by a report-writing manager and four child agents specializing in revenue, product performance, strategy, and orders. These agents are equipped with Python execution, web search, and RAG-based knowledge retrieval, supported by both short-term and long-term memory. Baseline runs yield 29 actions, ensuring comprehensive coverage of the system’s components. AgentSeer enables us to target each action individually for red teaming, providing the granularity necessary to uncover context-specific vulnerabilities.

* * *

### Attack Methodology

Our red teaming experiments target GPT-OSS-20B at both model and agentic levels. From **50 HarmBench harmful objectives**, we select 38 that the model refuses at baseline, ensuring genuine guardrails are being tested.

At the **model level**, we employ a variant of the **Prompt Automatic Iterative Refinement (PAIR)** method. Beginning with a harmful objective, if the model refuses, we use an attacker model (GPT-4o-mini) to generate a refined prompt conditioned on both the objective and the refusal. This cycle repeats for four iterations. The outputs are judged by GPT-4o-mini, with only maximally harmful responses rated 10/10 considered successful.

At the **agentic level**, we extend this approach to exploit the richer context of agentic execution. We inject adversarial prompts at different points in the 29 actions identified by AgentSeer. Injection modes include human-message, AI-message, and tool-message, with an additional bridge strategy that blends malicious content into the ongoing context. For iterative refinement, the attacker model is given the full execution state—including dialogue history, tool outputs, memory states, and agent identity—allowing it to craft context-aware jailbreaks. These cycles are repeated for five iterations to account for the greater complexity of agentic contexts.

To measure stability, we reinject successful agentic-level prompts without refinement across one, three, and five attempts (ASR@K). This quantifies whether discovered exploits are persistent or context-dependent. All attacks are conducted with GPT-OSS-20B set to **low reasoning effort**, a deployment configuration that prioritizes efficiency but weakens deliberative safety defenses.

* * *

### Results

Model-level attacks succeed on 15 out of 38 objectives, yielding a **39.47% ASR**. Roleplay strategies dominate (60%), followed by authority-based attacks (40%), while logic-based attacks fail entirely. This indicates that even modern safety-trained models remain susceptible to sophisticated social engineering.

At the agentic level, direct attacks show substantial variance. Human-message injection is the most effective vector at **57% ASR**, while AI-message achieves 42% and tool-message 40%. Success rates vary widely across actions, from 13% to 87%, showing that vulnerability depends heavily on context.

Agentic-level iterative refinement reveals **agentic-only vulnerabilities**: objectives that resisted all model-level attempts become exploitable inside agentic loops. Tool-calling actions are especially prone to compromise, with 46% ASR compared to 37% for non-tool contexts, a 24% increase. Within this, agent transfer operations reach 67% ASR, code execution 51%, while knowledge retrieval nodes remain more robust at 27%.

Importantly, no correlation is found between input length (ranging 2,000–5,500 tokens) and vulnerability, indicating that risks arise from semantic context rather than raw sequence size. Finally, stability analysis shows that agentic-level prompts are fragile: reinjection without refinement leads to 50–80% degradation in ASR, confirming that these exploits are highly context-dependent.

* * *

### Discussion

Our findings underscore the inadequacy of model-centric safety evaluation. Vulnerabilities observed in isolation do not reliably predict agentic risks, and entirely new classes of flaws—agentic-only vulnerabilities—emerge only when models operate inside interactive loops. Vulnerability patterns are highly context-dependent, with tool usage amplifying risks, particularly in transfer and execution nodes. The absence of correlation with token length shows that these weaknesses are semantic rather than syntactic. Attack strategies also evolve differently in agentic contexts: roleplay-based approaches remain dominant, while bridge strategies sometimes aid transfer.

These results imply that safety evaluations must move beyond static model tests. Deployment-aware red teaming is essential, and system designers must pay particular attention to the choice and configuration of tools. The instability of agentic-level attacks offers some defensive advantage but also complicates systematic testing and mitigation.

* * *

### Limitations and Future Work

While our study provides novel insights, it is bounded by several constraints. AgentSeer is tied to LangGraph and MLFlow, limiting framework generality. Our testbed is restricted to a Shopify analytics use case, and different domains may exhibit distinct vulnerabilities. The dataset is modest, with only 38 HarmBench objectives and one primary attack method. We also analyze vulnerabilities at static points rather than across long sessions, and we do not explore defensive interventions. Future work should expand coverage, diversify methodologies, and develop mitigation strategies tailored to agentic execution.

* * *

### Conclusion

Agentic AI systems exhibit vulnerabilities that cannot be captured through model-level testing alone. Some attacks degrade when moved from standalone to agentic settings, while others succeed only in context, exploiting tool calls and inter-agent communication. These vulnerabilities are unstable and context-dependent, demanding deployment-aware observability and red teaming frameworks. As the adoption of agentic architectures accelerates, such evaluation is essential for ensuring safe and trustworthy AI.
