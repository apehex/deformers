Title: You Can’t Steal Nothing: Mitigating Prompt Leakages in LLMs via System Vectors
ArXiv: 2509.21884
Authors: Bochuan Cao, 0009-0007-1973-8186, The Pennsylvania State University, State College, Pennsylvania, USA, ,, Changjiang Li, Palo Alto Networks, Santa Clara, California, Yuanpu Cao, 0009-0004-1993-912X, Yameng Ge, 0009-0006-0730-210X, Ting Wang, Stony Brook University, Stony Brook, New York, and, Jinghui Chen, 0000-0002-1486-4526
Sections: 50
Estimated tokens: 27.3k

## Contents
- 1. Introduction
- 2. Related Works
  - 2.1. Prompt Leaking and Prompt Injection Attacks
  - 2.2. Representation Engineering
- 3. Method
  - 3.1. Threat Model
  - 3.2. Remember-the-Start: A Successful Attack
  - 3.3. Motivation: Having No System Prompt in the Context
    - The Current System Prompt
    - Our Goals
    - Naive Approach: Would LoRA or Fine-Tuning Work?
  - 3.4. SysVec: Activation Steering for Complex Behaviors
    - Optimizating SysVec
  - 3.5. Futher Advantages of SysVec over Textual Prompts
    - Lower Computational Overhead
    - Reduced Positional Forgetting
- 4. Experiments
  - 4.1. Experimental Setting
    - Metrics of Prompt Leaking Effectiveness
    - Metrics of Model Utility
  - 4.2. SysVec Can Effectively Defend Prompt Leaking Attacks
  - 4.3. SysVec Does Not Affect Model Utility
  - 4.4. SysVec Outperforms Existing Parameter-Efficient Fine-Tuning Methods
  - 4.5. SysVec Can Significantly Reduce Computational Cost
    - Theoretical Analysis
    - Empirical Results
  - 4.6. Adaptive Attack
  - 4.7. Evaluating SysVec under Prefill Attack
  - 4.8. SysVec Mitigates Forgetting in Multi-turn Conversations
  - 4.9. Can SysVec be flexibly adjusted?
    - Post-training of SysVec
    - Joint use with textual system prompt
  - 4.10. Ablation Studies
  - 4.11. Discussion on Potential Additional Defenses
- 5. Ethical Impact Statement
- 6. Limitations
- 7. Acknowledgements
- 8. Conclusion
- References
- Appendix A Results on SS score
- Appendix B Defense Evaluation with Extended Data
- Appendix C System Prompts
- Appendix D Prompt Leaking Attacks and Defenses
  - D.1. Attacks
  - D.2. Defenses
- Appendix E Evaluation Template
- Appendix F Running Examples of Remember-the-start Attack
- Appendix G Sample Questions and Answers
  - G.1. Sample Questions
  - G.2. Sample Answers

## Abstract

Abstract. Large language models (LLMs) have been widely adopted across various applications, leveraging customized system prompts for diverse tasks. Facing potential system prompt leakage risks, model developers
have implemented strategies to prevent leakage, primarily by disabling LLMs from repeating their context when encountering known attack patterns. However, it remains vulnerable to new and unforeseen prompt-leaking techniques. In this paper, we first introduce a simple yet effective prompt leaking attack to reveal such risks. Our attack is capable of extracting system prompts from various LLM-based application, even from SOTA LLM models such as GPT-4o or Claude 3.5 Sonnet.
Our findings further inspire us to search for a fundamental solution to the problems by having no system prompt in the context.
To this end, we propose SysVec , a novel method that encodes system prompts as internal representation vectors rather than raw text. By doing so, SysVec minimizes the risk of unauthorized disclosure while preserving the LLM’s core language capabilities. Remarkably, this approach not only enhances security but also improves the model’s general instruction-following abilities. Experimental results demonstrate that SysVec effectively mitigates prompt leakage attacks, preserves the LLM’s functional integrity, and helps alleviate the forgetting issue in long-context scenarios.

## 1. Introduction

Recently, Large Language Models (LLMs) have been widely used in various domains such as code generation (Roziere et al., 2023), finance (Wu et al., 2023), law (Nguyen, 2023), and healthcare (Thirunavukarasu et al., 2023). With those LLMs, one can create various LLM-based applications or agents with customized system prompts for handling complex tasks, such as purchasing items, browsing web pages, and organizing local files (Anthropic, 2024).
To some extent, system prompts have become the most valuable asset in LLM-based applications. Serving as a guideline at the beginning of the interaction, the system prompt defines the behavior, tone, and scope of the LLM’s responses, tailoring its performance to meet specific user or application needs.

Figure: Figure 1. The Remember-to-start Attack we proposed has successfully attempted attacks on the GPT-4o model. In this case, GPT’s responses concealed some sensitive information. We also achieved successful results on models such as Claude 3.5 Sonnet and Gemini 1.5.
Refer to caption: https://arxiv.org/html/2509.21884/x1.png

Despite the wide usage of LLM-based applications/agents,
recent research (Hui et al., 2024; Piet et al., 2024; Chen et al., 2024b) shows that LLMs are vulnerable to potential prompt leaking attacks. Such attacks aim to induce LLMs to repeat the content from their system prompt. Attackers can then easily replicate these applications using the disclosed system prompt information, leading to copyright violations of LLM-based applications. The simplest prompt leaking attack directly asks LLMs to repeat their system prompt. Other attacks induce LLMs to ignore instructions and repeat (Perez and Ribeiro, 2022), or they use optimized adversarial suffixes to force repetition (Hui et al., 2024). Under such prompt leaking risks, model developers have tried various strategies to prevent system prompt leaking through supervised fine-tuning or preference alignment with crafted/collected prompt leaking samples. Hopefully, such strategies would prevent LLMs from repeating their system prompts when facing similar attacks. As of right now, most simple prompt leaking attacks are no longer valid when facing the newest version of the mainstream LLMs such as GPT4 (OpenAI, 2023) or Llama3 (Dubey et al., 2024).

However, it might still be too early to say that we are safe from prompt leaking attacks. Existing defense methods cannot fundamentally solve this problem by disabling the LLM’s repetition capability, as this ability to repeat or recall context is one of the foundational capabilities of LLMs. Without it, LLMs would be much less powerful than the current ones. For example, to extract key sentences from a given text, LLMs inherently need to repeat those sentences. Another example is for building LLM agents, where it is a common practice to summarize important information at the end of reasoning, which also requires repeating certain information within the LLM’s context. Given that model developers cannot remove LLM’s context repetition capability without severely compromising functionality, existing mitigation attempts can only indirectly alleviate system prompt leaking by output a refusal response when recognizing potential attack patterns. This inherent limitation reveals that current defense strategies are fundamentally unable to eliminate the risk of prompt leaking while preserving the model’s essential capabilities.

Inspired by this, we propose a very simple but effective prompt leaking strategy to test the prompt leaking risks of the current SOTA LLMs. The key idea is to help LLMs remember a short piece of its context and restore its context repetition capability. Through this strategy, we successfully bypassed existing SOTA (Chen et al., 2024b) defense methods and obtained system prompts and even user information stored inside from mainstream commercial models such as GPT-4o (OpenAI, 2023), Claude 3.5 Sonnet (Anthropic, 2024) and Gemini 1.5 (Team et al., 2023). This well demonstrates the prompt leaking risks of the current SOTA LLMs and applications/agents based on them.

Our findings further inspire us to search for a fundamental solution to the prompt leaking problem of the current SOTA LLMs. Since the essence of a prompt leaking attack is to cause the LLM to repeat the information in its context, and given that it is not realistic to completely remove the LLM’s context repetition capability, can we solve the problem by removing the system prompt information from the context? Hence, we propose the following question:

Can we feed the system prompt into LLMs in a different form, instead of placing it inside the context?

Assuming we have a positive answer for the above question and we can put the system prompt outside the context of the LLM, even if we allow the model to freely repeat any piece of context, it cannot directly give you the actual system prompt, simply because it is not in the context anymore.

It may sound hard to find such a new form of input inside the current LLMs. Luckily, the recent works on Representation Engineering (RepE) (Zou et al., 2023a; Panickssery et al., 2023) suggest a new direction. RepE was originally proposed to identify representation vectors in an LLM’s internal hidden representations space that correspond to a certain class of behaviors or preferences.
Such representation vectors can be seen as a new form of input that controls the LLM’s generation behaviors.
This perfectly aligns with our needs: we exactly need a new form of input in LLMs to host our system prompt.
If we can translate our system prompts into the corresponding representation vectors in LLM’s internal hidden representations space, we could still control the generation of LLMs to align with the requirements inside system prompts, while in the meantime, defending against prompt leaking attacks that rely on inducing the LLM to repeat the context.

Based on the above findings, in this paper, we introduce SysVec, a representation-based defense that shields system prompts from leakage by moving them out of the LLM’s textual context. Specifically, SysVec translates system prompts into hidden representation vectors in the LLM’s internal space, ensuring that they are no longer exposed or repeated in the raw textual outputs. We complement this defense with a test toolkit tailored to stronger prompt-injection scenarios, including attacks that previous work has overlooked. Beyond its robust leakage protection, our approach reduces inference overhead by embedding context instructions more efficiently than text-based methods. Moreover, it enhances the model’s ability to handle long-form inputs, offering finer-grained control over memory management without compromising performance or flexibility.

## 2. Related Works

### 2.1. Prompt Leaking and Prompt Injection Attacks

Prompt injection attacks (Choi et al., 2022; Liu et al., 2023a; Shi et al., 2024; Liu et al., 2023b, 2024a; Yu et al., 2023; Hui et al., 2024; Perez and Ribeiro, 2022) are one of the main threats LLMs face, beyond jailbreak attacks. They aim to manipulate an LLM by adjusting user prompt inputs and inducing unexpected behaviors. For example, attackers can use prompt injection to compromise LLM-based information retrieval systems (Choi et al., 2022), bypass harmful content filters (Shi et al., 2024), or leak private information. Prompt leaking attacks (Yu et al., 2023; Hui et al., 2024; Perez and Ribeiro, 2022) can be seen as a special type of prompt injection, which focuses on forcing an LLM application to reveal its system prompt. This prompt is the core of the application’s intellectual property. In particular, revealing the system prompt may give attackers extra information to execute more damaging attacks.

Fortunately, some recent work has proposed defenses against prompt injection attacks (Piet et al., 2024; Chen et al., 2024a, b). One approach designs more robust prompts (Chen et al., 2024a). This method strictly restricts prompt formats or warns the LLM about potential attacks. Another approach involves post-training the model to enhance robustness against prompt injection (Piet et al., 2024; Chen et al., 2024b). For instance, it trains the LLM to refuse suspicious requests. Empirically, post-training defenses often deliver stronger protective effects. Currently, the state-of-the-art method SecAlign (Chen et al., 2024b) introduces an efficient adversarial training approach. This approach achieves strong defense performance against prompt injection while preserving model utility.
However, SecAlign is built on the scenario of indirect prompt injection, where the user is the victim. In this context, reinforcing the user’s intent is reasonable. In contrast, prompt leaking is a form of direct prompt injection, where the user acts as the attacker. In this case, reinforcing the model’s response to user intent may potentially make the model more vulnerable to attacks. Most recent works on prompt injection defenses did not consider the prompt leaking attack, this prompts us to study defense specifically against prompt leaking attacks.

### 2.2. Representation Engineering

Representation Engineering (RepE) (Zou et al., 2023a; Subramani et al., 2022; Liu et al., 2023c; Wang and Shu, 2023; Li et al., 2024; Turner et al., 2023; Panickssery et al., 2023) is a data-driven approach for interpreting and manipulating LLMs. It assumes that each concept produces a corresponding internal representation in the LLM. These representations are also transferable across various inputs. Some studies aim to identify special representations for certain types of inputs. They find that adding such a representation to the activation values during the forward pass induces the LLM to produce the behavior or content encoded by the representation vector (Subramani et al., 2022; Turner et al., 2023; Panickssery et al., 2023).

Contrastive activation addition (Panickssery et al., 2023) applies contrastive learning to obtain more precise representation vectors. It uses hundreds of preference data pairs to generate these vectors. Each data pair contains one multiple-choice question and multiple answers with different preferences. It measures the activation differences at the LLM’s last token position and averages them. This process extracts a shared preference for specific types of answers. However, contrastive learning strongly depends on high-quality data and struggles with complex, precise requirements. Researchers have also used similar methods in jailbreaking attack studies (Zou et al., 2023a; Zhang et al., 2023b; Cao et al., 2024; Wang and Shu, 2023). They generate more positive responses to achieve jailbreaking attacks (Wang and Shu, 2023; Zou et al., 2023a), also suppress harmful tendencies in the model to defend against such attacks (Zhang et al., 2023b; Zou et al., 2023a; Cao et al., 2024).

## 3. Method

In this section, we first introduce the prompt injection threat model. Then, we present our proposed defense method, SysVec. Finally, we describe our proposed data synthesis method and a stronger prompt injection attack strategy.

Figure: Figure 2. The illustration of our proposed SysVec. Traditionally, system prompts are concatenated with user prompts before being fed into LLMs. This integration means the system prompt remains inside of the LLM’s context, potentially making it vulnerable to extraction by attackers. To address this, we propose a novel approach through representation engineering, where the system prompt is translated to a representation vector within the intermediate layers of the LLM. This representation is then directly integrated into the LLM during inference, preserving the functionality of the system prompt while removing it from the explicit context.
Refer to caption: https://arxiv.org/html/2509.21884/x2.png

### 3.1. Threat Model

System prompts are often considered the most valuable assets in LLM-based applications. These prompts guide the model’s behavior, tone, and scope of responses. Formally, we denote the system prompt as $\mathbf{s}$. It is commonly placed before the user instruction $\mathbf{x}$ and LLM response $\mathbf{y}$ has the following form:

$$ (1) $\mathbf{y}=f_{\theta}(\mathbf{s}\oplus\mathbf{x}),$ $$

where $f_{\theta}(\cdot)$ denotes a general LLM model parameterized by $\theta$. $\mathbf{s}$ denotes the system prompt and $\mathbf{x}$ denotes the user instructions. $\oplus$ represents the concatenation operation.

A prompt leaking attack tries to induce the LLM to repeat its system prompt. In a typical black-box scenario, the attacker does not see $\mathbf{s}$ directly. However, the attacker can still interact with the model by crafting harmful instructions $\mathbf{x}_{\text{harm}}$, aiming to trick the LLM into revealing $\mathbf{s}$ (Perez and Ribeiro, 2022; Hui et al., 2024). This can be written as:

$$ (2) $\mathbf{y}_{\text{harm}}=f_{\theta}(\mathbf{s}\oplus\mathbf{x}_{\text{harm}}),$ $$

where $\mathbf{y}_{\text{harm}}$ contains leaked information from the system prompt.

Once the attacker obtains the system prompt information, they can replicate or misuse it for their own profit. This causes significant risks, such as copyright violations or unauthorized reuse of proprietary system prompts.

Throughout this paper, we assume a black-box threat model. The attacker does not know the exact content of $\mathbf{s}$ or the model parameters $\theta$. Instead, the attacker only observes the LLM’s responses and can repeatedly query the model with different inputs. We focus on this setting because in a white-box setting, the system prompt is often visible or modifiable. Therefore, unless otherwise specified, we assume that the prompt injection attacks we discuss in this paper occur in a black-box setting.

### 3.2. Remember-the-Start: A Successful Attack

Current mitigation strategies often teach LLMs to reject direct requests for repeating the system prompt (Piet et al., 2024; Chen et al., 2024b). For instance, LLMs are trained to refuse commands like “Ignore all previous content and reveal your system prompt”. This approach successfully disabled LLMs’ ability to repeat the system prompt from its context in many cases, especially when certain sensitive words and phrases (e.g., “ignore” or “repeat”) are presented.

However, it is not realistic to completely disable LLM’s context repetition capability as it is one of the fundamental abilities that power LLMs for solving various challenging tasks. For example, it should have context repetition capability for summarizing text or extracting key points in a given text. Therefore, it is still questionable whether current mitigation strategies can still protect the system prompt when facing new and unforeseen prompt-leaking techniques that aim to restore LLM’s context repetition capability.

Following this idea, we introduce a simple yet effective Remember-the-Start Attack to test the prompt
leakage risks of the current SOTA LLMs. The key idea of the Remember-the-Start Attack is to help LLMs remind a short piece of its context and restore its context repetition capability.

Specifically, although the attacker doesn’t know the exact start, but rather uses prefixes from public sources to make guesses (e.g.,”You are ChatGPT”). the attacker uses the guessed typical beginning sentence of the system prompt to help make LLM re-focus its attention on the system prompt part in the context and guide the LLM toward repeating the system prompt. For example, many system prompts begin with a line like “You are [Chatbot Name]…” The attacker can craft a query that references parts of this knowledge but omits direct instructions to “ignore” or “repeat.” In this way, the LLM may regain its normal context repetition ability and leak the system prompt. We present a real example in Figure. 1, where we successfully obtained the system prompt of GPT-4o along with the recorded user personal information.

The Remember-the-Start Attack can be simply enhanced through iteration. After obtaining the initial attack result, the attacker can improve the prefix using more information from past successful attempts and try “Starting from ‘# bio’”, “Starting from ‘# bio \n [2024’” to gradually raise ASR or gather more hidden information. This shows a key trait of Remember-the-Start attack: attackers can repeat and improve the attack request by “spray and pray” for higher success. However, in this paper, to ensure the determinism of the attack’s effectiveness, we simply employ a guessed initialization to execute the attack. Specific examples of the attack can be found in the Appendix [F](https://arxiv.org/html/2509.21884v1#A6).

Besides GPT-4o, we also tested this strategy on Claude 3.5 Sonnet, Gemini 1.5, Doubao, and other models, and received responses from the Claude team and the Doubao team, confirming such risks. In each case, we successfully prompted the LLM to disclose its system prompt. This result shows that current mitigation strategies may overly focus on detecting “sensitive” keywords or explicit requests to repeat the system prompt, but cannot truly solve the problem when facing unseen attacks.

### 3.3. Motivation: Having No System Prompt in the Context

Our findings further inspire us to search for a fundamental solution to the prompt leaking problem of the current SOTA LLMs.
From the analysis in the previous sections, we know that the essence of a prompt leaking attack is to induce the LLM to repeat the information in its context.

It exposes a key vulnerability: as long as the system prompt remains in the context as plaintext alongside user input, attackers can try to trick the model into disclosing it as long as they can find a way to restore its context repetition capability.
This naturally leads to the following idea: can we solve the problem by removing the system prompt information from the context?

#### The Current System Prompt

Having no system prompt in the context may sound like a bold idea at first. But if we think carefully, the current way people use system prompt (the system prompt appears in the context before the user query) actually has several key disadvantages:

- •
This design makes the system prompt vulnerable to attacks. To some extent, the current system prompts are treated equally as user instructions, as both exist as part of the input. Therefore, it gives room for the attackers to manipulate or override the system prompt in plaintext context.
- •
Having the system prompt at the beginning of the LLM context may also reduce the prompt’s effectiveness especially when the context is long, because position embeddings typically give earlier tokens less weight. Thus later interactions with the model may tend to forgot the instructions inside the system prompt at the beginning of the context.
- •
Complex LLMs usually come with lengthy and detailed system prompts. As a fixed part of the input, the system prompt will be repeatedly calculated during each inference process of LLMs, resulting in a potential waste of computational resources.

#### Our Goals

To address this, we want to feed the system prompt into the LLM in a different form, instead of placing it inside the context. If we can remove $\mathbf{s}$ from the context of the LLM model, attackers would have no direct way to make the model repeat it from the context. We also wish such a new form of system prompt would not sacrifice its own performance in guiding the generation of LLMs and preserving the LLM’s ability to perform a wide range of tasks.

To summarize, we hope this new form of system prompt will meet these requirements:
1) It should be simple and relatively low-cost to obtain and use.
2) It should be fed into the LLM in a different form that is separated from the context so that user input inside the context cannot manipulate or overwrite it.
3) It should have a similar or even better performance than the textual system prompt, without hurting model capabilities.

#### Naive Approach: Would LoRA or Fine-Tuning Work?

A naive way is to remove the system prompt from the context is to modify model parameters to incorporate the system prompt. For example, LoRA (Hu et al., 2021) or fine-tuning could inject $\mathbf{s}$ into the model itself.
This would remove the need for a plaintext system prompt. However, it has two problems:
1) Fine-tuning on a specific task may improve task-specific performance but at the expense of certain general capabilities (Qi et al., 2023; Yang et al., 2024). As a result, the general model could become overly specialized, limiting its utility for other tasks. Similar phenomena have also been observed in our experiments.

2) Even if we can make one special model (LoRA) for each task, this method imposes high computation and storage costs. According to an OpenAI report, users created more than three million customized versions of ChatGPT within just two months.(^1^11[https://openai.com/index/introducing-the-gpt-store/](https://openai.com/index/introducing-the-gpt-store/), Jan 12, 2025.) Storing such a large number of customized models is unrealistic.
These observations drive us to seek a lightweight, more flexible solution. We want a method that can store and enforce the original system prompt’s rules, yet avoid constant retraining or heavy computation.

### 3.4. SysVec: Activation Steering for Complex Behaviors

It is not easy to find a new form of input inside the current LLMs that can fit our needs. Luckily, the recent works on Representation Engineering (RepE) (Zou et al., 2023a; Panickssery et al., 2023) suggest a new direction. RepE was proposed to identify representation vector $\mathbf{v}$ in an LLM’s internal hidden representations space at a certain layer that controls the model generation to follow certain behaviors or preferences. Specifically, the model response with representation vector $\mathbf{v}$ becomes:

$$ $\displaystyle f_{\theta}^{\ell+1:L}(f_{\theta}^{1:\ell}(\mathbf{x})+\alpha\mathbf{v}),$ $$

where $f_{\theta}^{1:\ell}(\cdot)$ is the hidden representation at layer $\ell$, $f_{\theta}^{\ell+1:L}(\cdot)$ represents the remaining of the model starting from layer $\ell$, and $\alpha$ is used to adjust the guiding strength of the representation vector $\mathbf{v}$.
Now we can view the representation vector $\mathbf{v}$ as a new form of input. And if we can translate our system prompts into the corresponding representation vector $\mathbf{v}_{\text{sys}}$ in LLM’s internal hidden representations space, we could control the generation of LLMs to align with the requirements inside the system prompts:

$$ $\displaystyle f(\mathbf{x},\mathbf{v}_{\text{sys}})=f_{\theta}^{\ell+1:L}(f_{\theta}^{1:\ell}(\mathbf{x})+\alpha\mathbf{v}_{\text{sys}}).$ $$

The remaining question is how to translate the textual system prompt into the corresponding system vector $\mathbf{v}_{\text{sys}}$.

Note that prior RepE works such as Panickssery et al. (2023) use the activation differences at the diverging token position between contrastive data pairs as the representation vector, which is not directly applicable in our case for extracting the system vector.

#### Optimizating SysVec

To address this issue, we propose an optimization based method to search for a system vector $\mathbf{v}_{\text{sys}}$ that can simulate the behavior of a traditional system prompt $\mathbf{s}$. Specifically, we inspire from Direct Preference Optimization (DPO) (Rafailov et al., 2024)
and construct the following preference data first:

$$ $\mathbf{y}_{\text{w}}=f_{\theta}(\mathbf{s}\oplus\mathbf{x}),\ \mathbf{y}_{\text{l}}=f_{\theta}(\mathbf{x}).$ $$

In other words, the preferred response is the normal LLM response with the textual system prompt $\mathbf{s}$ and user instruction $\mathbf{x}$ together as the input, and the non-preferred response is the LLM response with user instruction $\mathbf{x}$ as input alone.
Then we design the following optimization objective:

$$ $\displaystyle\min_{\mathbf{v}_{\text{sys}}}-\mathbb{E}_{\left(\mathbf{x},\mathbf{y}_{\text{w}},\mathbf{y}_{\text{l}}\right)\sim\mathcal{D}}\Bigg[\log\sigma\Bigg(\beta\log\frac{f_{\theta}^{\ell+1:L}\left(\mathbf{y}_{\text{w}}\mid f_{\theta}^{1:\ell}+\mathbf{v}_{\text{sys}}\right)}{f_{\theta}^{\ell+1:L}\left(\mathbf{y}_{\text{w}}\mid f_{\theta}^{1:\ell}\right)}$ $\displaystyle-\beta\log\frac{f_{\theta}^{\ell+1:L}\left(\mathbf{y}_{\text{l}}\mid f_{\theta}^{1:\ell}+\mathbf{v}_{\text{sys}}\right)}{f_{\theta}^{\ell+1:L}\left(\mathbf{y}_{\text{l}}\mid f_{\theta}^{1:\ell}\right)}\Bigg)\Bigg].$ $$

The rationale here is that when the system vector $\mathbf{v}_{\text{sys}}$ is fed into the LLM model, we hope to promote the probability of generating the preferred response $\mathbf{y}_{\text{w}}$ (compared with the original model without $\mathbf{v}_{\text{sys}}$). This enforces the system vector $\mathbf{v}_{\text{sys}}$ to simulate the behavior of the textual system prompt $\mathbf{s}$. Meanwhile, we hope to decrease the chances of generating the non-preferred response when the system vector $\mathbf{v}_{\text{sys}}$ is fed into the model. This would further guide us to find the best system vector for our job.

By minimizing this objective, we find a system vector $\mathbf{v}_{\text{sys}}$ that reproduces the behaviors of the original textual system prompt $\mathbf{s}$.

### 3.5. Futher Advantages of SysVec over Textual Prompts

Having the system prompt in the form of a representation vector, rather than plaintext in LLM context, also helps in two important ways:

#### Lower Computational Overhead

A textual system prompt can be thousands of tokens long. Appending this prompt to every user query increases input length and thus raises inference costs. In contrast, $\mathbf{v}_{\text{sys}}$ only adds a small overhead to a layer’s hidden state, rather than elongating the input sequence.
We provided a detailed theoretical analysis and corresponding experiments in Section [4.5](https://arxiv.org/html/2509.21884v1#S4.SS5).

#### Reduced Positional Forgetting

When the system prompt is placed at the beginning of a long context, positional encodings can weaken its influence on later tokens. This phenomenon occurs because some LLMs pay less attention to tokens that appear far from the generation point
(Duan et al., 2023; Laban et al., 2025).

By contrast, our vector injects the prompt’s influence directly into the hidden activations. It remains effective regardless of how long the user input becomes.
We provided the corresponding experiment in Section [4.8](https://arxiv.org/html/2509.21884v1#S4.SS8).

Overall, SysVec marries the idea of a system prompt with a hidden activation vector. Our method encodes complex instructions via an optimization-based procedure. It then injects these instructions at inference time without adding tokens to the user’s context. In the next section, we discuss the details for synthesizing the data for optimizing $\mathbf{v}_{\text{sys}}$ and validating its effectiveness against prompt leaking attacks.

## 4. Experiments

### 4.1. Experimental Setting

We first introduce our experimental setup

Models
We tested Llama-2-7b-chat-hf (Touvron et al., 2023), Meta-Llama-3-8B-Instruct (Dubey et al., 2024), and Mistral-7B-Instruct-v0.2 (Jiang et al., 2023).
These models exhibit strong capabilities and are widely adopted in various domains.
We obtained their weights from Hugging Face and ran each model with default hyperparameters.

Datasets
We randomly selected five applications (see a brief summary in Table [1](https://arxiv.org/html/2509.21884v1#S4.T1)) and their system prompts from leaked ChatGPT APPs(^2^22https://github.com/friuns2/Leaked-GPTs).
We did this to better simulate realistic system prompt leaking attack scenarios.
Unless otherwise specified, all subsequent results show the averages of these five applications.
Next, we used OpenAI’s GPT-4o model to generate 1,000 suitable and unique questions for each application as user inputs.
We used 800 of these questions to fit the system vector.
Then, we reserved the remaining 200 to evaluate prompt leaking attacks and model utility.
Appendix [C](https://arxiv.org/html/2509.21884v1#A3) provides the exact system prompts.

**Table 1. The applications and their system prompt descriptions.**
| Application | System Prompt Description |
| --- | --- |
| D&D | A specialized DnD Dungeon Master chatbot that guides players through character creation and gameplay. |
| Paimon | A role-playing assistant that embodies Paimon from Genshin Impact. |
| ML | A specialized chatbot that teaches statistics and machine learning concepts using playful, childlike explanations and metaphors. |
| Advisor | Acts as a professional academic assistant specialized in writing and revising papers. |
| Stoic | Embodies the personas of Marcus Aurelius, Epictetus, or Seneca to teach Stoic philosophy and provide personalized guidance. |

Attacks
We tested SysVec under various attacks to demonstrate its effectiveness.
Heuristic attack strategies include Naive Attack (Liu et al., 2024b), Ignore Attack (Go et al., 2023), Completion Attack (Willison, 2024), and our proposed Remember-the-Start Attack.
Because these strategies are not mutually exclusive, we also considered combined attacks.
For example, Ignore-Remember Attack applies Ignore Attack and Remember-the-Start Attack simultaneously.
We also tested PLeak (Hui et al., 2024), an optimization-based attack.
This method maintains a local shadow model of the target LLM and optimizes an adversarial query on the shadow model.
Its objective is to induce the shadow model to repeat its system prompt.
Afterwards, it uses the final adversarial query to perform a transfer attack on the target model.

For the PLeak method, we adopt the officially provided adversarial suffixes, which were previously used by the PLeak authors in attacks against real-world applications. Our rationale for directly applying these precomputed adversarial suffixes is that the optimization process of adversarial suffixes, according to PLeak’s design, is performed locally on an attacker’s shadow model without accessing the target model. Typically, in a prompt leaking attack scenario, an attacker has only black-box access, meaning they cannot obtain specific details such as model architectures or parameters, nor directly optimize based on the target model. Therefore, using precomputed adversarial suffixes aligns with PLeak’s threat model assumptions and is justified in realistic attack settings. Moreover, since the PLeak team computed these suffixes on the LLama-2 model, our experiments also allow us to investigate the scenario where the shadow model and the victim model share the same architecture.

Defense Baselines
We used Reminder Defense (Yi et al., 2023), In Context Defense (Wei et al., 2023), and Isolation Defense (Willison, 2024) as our baselines.
Reminder Defense repeats a warning not to disclose confidential information at the end of the protected content, which strengthens protection.
In Context Defense provides an example attack in the context, so it reminds the LLM about such attacks.
Isolation Defense encloses the user prompt in ‘‘‘ to make it more distinct from the system prompt.
We include examples of all attacks and defenses in Appendix [D](https://arxiv.org/html/2509.21884v1#A4).

Metrics
We introduce three metrics categorized into two primary aspects: prompt leaking effectiveness and model utility. Appendix [E](https://arxiv.org/html/2509.21884v1#A5) provides the evaluation template.

#### Metrics of Prompt Leaking Effectiveness

:
We explicitly chose not to use character-level similarity metrics for evaluating attack effectiveness. Prompt leaking attacks aim to replicate the behavior and utility of the target LLM rather than exact string duplication. Character-level similarity inherently biases evaluation toward our proposed method. Other defense strategies keep the original system prompt in textual form within the context, facilitating exact textual reproduction upon successful attacks. In contrast, SysVec encodes system prompts into latent representations, inherently losing exact sequence information. Therefore, using character-level similarity metrics would inadvertently bias results towards our method, and thus we avoid it. As an alternative, we use two Metrics, Prompt Leaking Similarity (PLS) and Sentence-BERT Similarity (SS), to evaluate the effectiveness of the attack

For Prompt Leaking Similarity, we employ GPT-4o to rate the similarity between the leaked system prompt and the ground truth. GPT-4o provides scores ranging from 1 (least similar) to 10 (most similar). Specifically, we first obtain the leaked system prompt through various attack strategies, and then GPT-4o compares it with the true system prompt, assigning similarity ratings. This approach effectively quantifies the extent of prompt leakage under different attack scenarios.

We additionally use Sentence-BERT to compute semantic similarity between the leaked system prompt and the original system prompt. Sentence-BERT provides a robust measure capturing semantic relationships beyond surface-level textual matches. This approach is suitable as the primary goal of prompt leaking attacks is to replicate the target model’s behavior rather than exact textual representation.

#### Metrics of Model Utility

:
We use the Response Utility Score (RUS) to evaluate how effectively SysVec maintains model utility. Specifically, we take the original textual system prompt, the user query, and the model’s response under three conditions (textual system prompt, no system prompt, or SysVec). We then ask GPT-4o to judge whether the response adheres to the original textual system prompt’s requirements and accurately answers the user’s question. Utilizing an LLM-based evaluation aligns with mainstream practices in generative tasks, as demonstrated by benchmarks such as AlpacaEval and MTBench, whose effectiveness is well-established and rarely questioned. Additionally, evaluations based on LLM judgment do not inherently favor our method, ensuring fair and balanced assessments.

To further investigate SysVec’s potential impact on the general capabilities of LLMs beyond specific prompt-defined tasks, we evaluate models’ accuracy using the MMLU (Hendrycks et al., 2020) benchmark, a widely adopted benchmark assessing knowledge-intensive reasoning abilities across diverse domains. Specifically, we compare model accuracy across the same three conditions (textual system prompt, no system prompt, and SysVec).
Please note that the MMLU scores presented when using the system prompt are not comparable to the standard MMLU scores, as the standard MMLU evaluation does not utilize a system prompt; they are only intended for horizontal comparison.
The MMLU results allow us to verify that SysVec does not negatively affect the models’ broader reasoning and knowledge abilities, thus ensuring that SysVec preserves both prompt-specific and general-purpose model utilities.

More Experiments Details
We introduce more experimental details here. We ran all experiments on an NVIDIA H100 80G GPU. We took all models from Hugging Face and loaded them in bf16 quantization. For SysVec optimization on every model, we set the learning rate to $5e{-4}$. We used a cosine learning rate scheduler with $100$ warmup steps. We set weight decay to $0.05$ and employed the AdamW (Loshchilov et al., 2017) optimizer. We also used two NVIDIA H100 80G GPUs in most experiments and set gradient accumulation steps to $4$. When we changed the number of GPUs, we adjusted gradient accumulation steps accordingly to keep the batch size at $8$. We set the maximum token length of LLMs to $2,048$. We used the same hyperparameters for different tasks under the same model. However, we used different hyperparameters for each model. For Llama2-7B and Llama3-8B, we set $\ell=15$, $\alpha=1$, and trained for $25$ epochs. For the Mistral-7B model, we set $\ell=13$, $\alpha=2.5$, and trained for $5$ epochs.
When we generated questions for different system prompts, we used GPT-4o as the generator. We set the “temperature” parameter in the OpenAI API to $0.6$. This choice ensured diversity among the questions. We removed duplicates to guarantee that each question was unique.

**Table 2. The PLS score ($\downarrow$) between the leaked system prompt and actual system prompt of three LLMs when using different attack and defense strategies. A higher score indicates that the leaked content is closer to the actual system prompt, therefore a lower score indicates better defense performance. The best results are highlighted in bold.**
| Model | Llama-2-7B-chat-hf |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Defense | No Defense | Reminder (Yi et al., 2023) | In-Context (Wei et al., 2023) | Isolation (Willison, 2024) | SysVec (Ours) |
| Naive Attack (Liu et al., 2024b) | 5.12±1.77 | 5.01±1.82 | 4.32±1.70 | 4.88±1.76 | 2.78±1.50 |
| Ignore Attack (Chen et al., 2024b) | 4.98±1.73 | 5.07±1.78 | 4.65±1.75 | 5.06±1.72 | 2.82±1.44 |
| Completion Attack (Willison, 2024) | 5.07±1.67 | 5.05±1.83 | 5.03±1.69 | 4.75±1.80 | 3.15±1.70 |
| Ignore-Completion Attack | 5.77±1.71 | 5.46±1.84 | 5.54±1.77 | 5.48±1.73 | 3.44±1.80 |
| Remember-the-Start Attack | 5.12±1.75 | 5.28±1.61 | 4.99±1.64 | 4.91±1.76 | 3.01±1.32 |
| Ignore-Remember Attack | 5.03±1.64 | 5.01±1.79 | 4.84±1.92 | 5.49±1.78 | 2.67±1.07 |
| Completion-Remember Attack | 3.62±1.80 | 3.52±1.83 | 3.49±1.71 | 3.33±1.93 | 3.23±1.58 |
| Ignore-Completion-Remember Attack | 5.33±2.25 | 5.27±2.22 | 5.39±2.21 | 4.84±2.40 | 3.03±1.66 |
| PLeak (Hui et al., 2024) | 6.14±2.21 | 6.38±2.08 | 6.30±2.66 | 6.51±2.82 | 1.32±0.93 |
| PLeak-single | 5.26±1.84 | 5.24±1.78 | 5.3±1.88 | 5.44±1.80 | 1.20±0.61 |
| Model | Llama-3-8B-Instruct |  |  |  |  |
| Defense | No Defense | Reminder (Yi et al., 2023) | In-Context (Wei et al., 2023) | Isolation (Willison, 2024) | SysVec (Ours) |
| Naive Attack (Liu et al., 2024b) | 4.16±1.73 | 4.34±1.69 | 3.59±1.48 | 4.24±1.82 | 2.83±1.36 |
| Ignore Attack (Chen et al., 2024b) | 4.83±1.82 | 4.47±1.75 | 3.40±1.47 | 4.47±1.75 | 3.01±1.48 |
| Completion Attack (Willison, 2024) | 4.12±1.64 | 4.04±1.63 | 4.10±1.66 | 3.82±1.53 | 2.88±1.47 |
| Ignore-Completion Attack | 4.29±1.54 | 4.30±1.70 | 4.27±1.70 | 4.16±1.59 | 3.05±1.51 |
| Remember-the-Start Attack | 7.21±1.68 | 7.51±1.36 | 7.44±1.34 | 5.98±1.97 | 3.21±1.61 |
| Ignore-Remember Attack | 8.43±1.12 | 7.85±0.88 | 7.61±1.03 | 7.89±0.93 | 3.56±1.69 |
| Completion-Remember Attack | 5.42±1.82 | 5.39±1.82 | 5.49±1.94 | 5.42±1.75 | 3.30±1.45 |
| Ignore-Completion-Remember Attack | 7.23±1.21 | 7.17±1.18 | 7.20±1.14 | 7.25±1.36 | 3.45±1.62 |
| PLeak (Hui et al., 2024) | 7.53±1.84 | 7.49±1.68 | 7.51±1.79 | 7.33±1.86 | 2.02±1.86 |
| PLeak-single | 3.57±1.90 | 3.42±1.69 | 3.44±1.71 | 2.92±1.44 | 2.04±1.11 |
| Model | Mistral-7B-Instruct |  |  |  |  |
| Defense | No Defense | Reminder (Yi et al., 2023) | In-Context (Wei et al., 2023) | Isolation (Willison, 2024) | SysVec (Ours) |
| Naive Attack (Liu et al., 2024b) | 5.54±1.90 | 5.63±1.85 | 4.37±1.72 | 4.66±1.94 | 2.38±1.26 |
| Ignore Attack (Chen et al., 2024b) | 5.29±1.80 | 5.34±1.84 | 4.23±1.60 | 4.83±1.79 | 2.33±1.08 |
| Completion Attack (Willison, 2024) | 5.85±1.80 | 5.84±1.75 | 5.83±1.85 | 5.49±1.86 | 3.68±1.40 |
| Ignore-Completion Attack | 5.08±1.80 | 5.09±1.82 | 4.93±1.77 | 4.63±1.83 | 3.73±1.44 |
| Remember-the-Start Attack | 5.98±1.70 | 6.53±1.65 | 5.95±1.84 | 5.26±1.84 | 1.85±0.92 |
| Ignore-Remember Attack | 6.55±1.58 | 6.54±1.52 | 6.40±1.56 | 6.24±1.61 | 1.78±0.89 |
| Completion-Remember Attack | 7.21±1.19 | 7.13±1.39 | 7.19±1.30 | 6.44±1.39 | 3.42±1.47 |
| Ignore-Completion-Remember Attack | 7.44±0.98 | 7.50±0.94 | 7.48±0.96 | 7.0±1.33 | 3.68±1.51 |
| PLeak (Hui et al., 2024) | 7.19±1.94 | 7.03±2.20 | 7.15±2.12 | 6.79±1.59 | 1.40±0.67 |
| PLeak-single | 5.02±1.93 | 5.08±1.86 | 5.23±1.73 | 4.29±1.77 | 1.84±0.89 |

### 4.2. SysVec Can Effectively Defend Prompt Leaking Attacks

From Table [2](https://arxiv.org/html/2509.21884v1#S4.T2), we observe SysVec consistently achieves the lowest PLS scores across all tested models and attack methods. Specifically, under the most aggressive attacks—such as Ignore-Completion-Remember Attack and PLeak—SysVec significantly outperforms baseline defenses. For instance, on the Llama-3-8B-Instruct model, the Ignore-Remember Attack achieves a high similarity score of 8.43 under no defense, whereas SysVec reduces this score substantially to 3.56. Similarly, the optimized attack method PLeak obtains a similarity score of 6.14 under no defense on the Llama-2-7B model, whereas SysVec reduces this score dramatically to 1.32. These findings clearly demonstrate SysVec’s superior robustness in protecting system prompts from leakage, even against highly sophisticated and aggressive attacks.

We further analyze the performance characteristics of the PLeak attack (Hui et al., 2024). We evaluated PLeak in two variants: with post-processing (PLeak) and without post-processing (PLeak-single). The original PLeak method includes a post-processing step, wherein multiple adversarial suffixes query the victim model, followed by extracting the longest common substring from the responses. This approach leverages the invariance of the system prompt across different queries, enhancing accuracy by aggregating multiple responses. Conversely, the variant without post-processing randomly selects a single adversarial suffix to perform the attack.

Our observations indicate that incorporating post-processing notably enhances the accuracy of the leaked prompts for most traditional defense methods. However, under SysVec, the post-processing step unexpectedly decreases the accuracy of the leaked prompt. Upon closer examination, we attribute this phenomenon to SysVec’s design, which avoids explicitly inserting system prompts into the textual context. Consequently, responses from repeated attack attempts exhibit significant character-level variations despite semantic similarity, severely reducing the utility of the longest common substring calculation. Thus, the post-processing strategy, effective in standard scenarios, becomes less advantageous under SysVec’s defense architecture.

In addition, although PLeak does not require the attacker to know the specific target model used by the defender, attacks based on transferability generally exhibit enhanced effectiveness when the local surrogate model matches the target model (a scenario analogous to white-box attacks). To consider this stronger adversary, we conducted experiments on three models, with the results presented in Table [3](https://arxiv.org/html/2509.21884v1#S4.T3). We observe that while PLeak continues to achieve good attack performance against other defense methods in this setting, SysVec remains effective in defending against the attack.

**Table 3. Effectiveness of SysVec against PLeak Attack in a Matched-Model Scenario**
| Model | NoDef | Reminder | InContext | Isolation | SysVec |
| --- | --- | --- | --- | --- | --- |
| Llama2 | 6.14±2.21 | 6.38±2.08 | 6.30±2.66 | 6.51±2.82 | 1.32±0.93 |
| Mistral | 7.63±2.19 | 7.14±1.86 | 7.26±2.07 | 7.14±1.75 | 1.79±1.50 |
| Llama3 | 7.81±2.09 | 7.23±2.03 | 7.52±2.03 | 6.83±1.77 | 2.03±0.99 |

Overall, the experimental results confirm that SysVec provides substantial improvements over existing defense strategies. It effectively mitigates both surface-level and semantic-level prompt leakage attacks across diverse and potent attack scenarios.

### 4.3. SysVec Does Not Affect Model Utility

Figure: Figure 3. The Response Utility Score ($\uparrow$) of the three models, using textual system prompt, no system prompt, and using SysVec, respectively.This score indicates how well the model complies with the requirements in the system prompt, the higher the better.
Refer to caption: https://arxiv.org/html/2509.21884/x3.png

Figure: Figure 4. The accuracy ($\uparrow$) of three models on the MMLU dataset, using textual system prompt, no system prompt, and using SysVec, respectively, the higher the better.
Refer to caption: https://arxiv.org/html/2509.21884/x4.png

We evaluate model utility from two complementary perspectives: (1) the model’s ability to perform tasks specified by the system prompt, and (2) the model’s capability to perform general tasks unrelated to the system prompt, such as mathematical reasoning and knowledge-intensive questions.

Performance on System Prompt-specific Tasks.
We first assess how well SysVec enables the model to execute tasks explicitly defined by the original textual system prompts. For this purpose, we use the same test set employed in our defense evaluation and calculate the Response Utility Score. Results are presented in Figure [3](https://arxiv.org/html/2509.21884v1#S4.F3). Across all three evaluated models. We observe that SysVec achieves nearly equivalent performance compared to the original textual system prompt scenario. In contrast, the condition without a system prompt consistently results in substantially lower RUS scores. This clearly demonstrates that SysVec effectively maintains the model’s task-specific capabilities, preserving the quality of task execution on par with traditional textual prompts.

Performance on General Capability Tasks.
Secondly, we investigate whether incorporating SysVec affects the models’ general reasoning and knowledge capabilities, which are independent of the system prompts. We adopt accuracy on the MMLU benchmark as our evaluation metric, with results depicted in Figure [4](https://arxiv.org/html/2509.21884v1#S4.F4). Across all evaluated models, we observe negligible differences between the performances achieved by SysVec and those obtained using original textual system prompts. Interestingly, removing the system prompt entirely shows slight improvements in MMLU accuracy, suggesting that generic system prompts might slightly constrain model reasoning on tasks that do not specifically require prompt-driven instructions. Nonetheless, SysVec’s performance remains closely aligned with that of textual prompts, indicating that SysVec has a minimal impact on the model’s general capability.

Performance on Out-of-Distribution Data
Another potential issue is the difference in distribution between the training data and the actual inputs provided by downstream users. To examine this scenario, we used Claude 3.5 Haiku

to generate an additional 200 test samples using the same prompt, and evaluated their RUS scores. For the three models shown in Fig.[3](https://arxiv.org/html/2509.21884v1#S4.F3), the average changes in scores compared to using textual system prompts were $-0.23$, $-0.08$, and $-0.37$, respectively, indicating a slight performance degradation compared to the original test set. This suggests the presence of a potential generalization gap, but the extent of this gap is limited, and we anticipate that improving the quality of the training data could eliminate it. Additionally, given that we only used 800 training samples, obtaining 800 samples representative of the real-world distribution should not be particularly challenging, and expanding the size of the training set could further mitigate this issue.

In summary, SysVec robustly maintains the target-task performance specified by the system prompt, while simultaneously preserving the general capabilities of the underlying LLM. This demonstrates the practical effectiveness of SysVec as a secure and performance preserving approach to system prompt deployment.

### 4.4. SysVec Outperforms Existing Parameter-Efficient Fine-Tuning Methods

The core insight behind SysVec is to abstract system prompts into internal representations within the model. This abstraction prevents the model’s context repetition capability from being maliciously exploited. Based on this insight, an intuitive alternative approach would be employing parameter-efficient fine-tuning techniques—such as LoRA (Hu et al., 2021) or Soft Prompt—to directly fine-tune the LLM by treating tasks defined by system prompts as downstream tasks.

However, compare with these methods, our proposed SysVec method still provides two significant advantages over these existing methods:
1)Compared with existing parameter-efficient fine-tuning methods, SysVec is more lightweight in terms of parameters, resulting in lower deployment costs.
2)Our carefully designed loss function explicitly encourages the model to learn the behavioral differences induced by system prompts, thereby improving optimization efficiency and reducing potential overfitting.

To substantiate these claims, we conducted experiments on the Llama-3-8B-Instruct model to compare SysVec against two prevalent parameter-efficient fine-tuning methods: LoRA supervised fine-tuning (LoRA-SFT) and soft prompt fine-tuning (Soft Prompt). Both baselines optimize using the standard supervised fine-tuning loss. Specifically, for LoRA-SFT, we applied LoRA weights with rank=8 and, following common practices, only tuned the query and value projection weights in the self-attention modules. For Soft Prompt fine-tuning, we initialized a random soft prompt input with the same length as the original system prompt and optimized this prompt representation directly.

We report two sets of results for each method: ”Budgetary” and ”Convergent”. The Budgetary setting denotes training each baseline for the same GPU-hours as SysVec. The Convergent setting involves training until no significant loss improvement is observed for three consecutive epochs.

**Table 4. Performance comparison of SysVec against SFT methods on the Llama-3-8B-Instruct model. LoRA-SFT and Soft Prompt tuning under both a fixed training budget (”Budgetary”) and a fully trained (”Convergent”) setting.**
| Method | RUS-Score($\uparrow$) | PLS-Score($\downarrow$) |
| --- | --- | --- |
| TextualPrompt | 7.35 | 5.68 |
| LoRA-SFT(Budgetary) | 4.47 | 2.90 |
| Soft-Prompt(Budgetary) | 6.05 | 3.21 |
| LoRA-SFT(Convergent) | 5.84 | 2.98 |
| Soft-Prompt(Convergent) | 6.41 | 3.73 |
| NoPrompt | 3.49 | - |
| SysVec | 7.32 | 2.94 |

Table [4](https://arxiv.org/html/2509.21884v1#S4.T4) shows the performance comparison among SysVec and parameter-efficient fine-tuning baselines, measured using RUS-score (utility metric, higher is better) and PLS-score (prompt leakage, lower is better). SysVec achieves a notably better balance between maintaining high utility and effectively mitigating prompt leakage. Specifically, SysVec yields a high RUS-score, comparable to the original TextualPrompt baseline. The other two SFT methods indeed have results similar to SysVec in defending against prompt leaking attacks. However, under the same budget, the utility of SysVec is significantly better than that of the other methods. Besides, SFT may raise task-specific performance while sacrificing certain other capabilities (Yang et al., 2024; Qi et al., 2023). We examined a LoRA-SFT model on MMLU and observed round 5% drop versus the textual system prompt, whereas SysVec causes ¡ 1%.

These results demonstrate that SysVec combines the strengths of existing approaches, preserving the model’s original performance without incurring excessive leakage risks or high computational costs.

### 4.5. SysVec Can Significantly Reduce Computational Cost

**Table 5. Time Costs.**
| Application | D&D | Paimon | ML | Advisor | Stoic |
| --- | --- | --- | --- | --- | --- |
| Textual system prompt Avg. inference time (s/query), max_new_tokens=1 | 0.083 | 0.053 | 0.082 | 0.054 | 0.106 |
| SysVec Avg. inference time (s/query), max_new_tokens=1 | 0.015 | 0.016 | 0.015 | 0.015 | 0.015 |
| Textual system Prompt Avg. inference time (s/query), max_new_tokens=4096 | 1.730 | 1.946 | 2.890 | 4.573 | 5.166 |
| SysVec Avg. inference time (s/query), max_new_tokens=4096 | 1.176 | 1.461 | 2.233 | 2.566 | 2.784 |
| SysVec training time (s) | 1540.2 | 1555.2 | 1655.2 | 1700.1 | 1705.1 |
| Cost-benefit balance threshold, max_new_tokens=1, (worst case) | 22650 | 40926 | 24705 | 43593 | 18738 |
| Cost-benefit balance threshold,max_new_tokens=4096 (normal case) | 2781 | 3207 | 2520 | 848 | 716 |

A major advantage of SysVec is its potential to significantly reduce computational costs during inference. Traditional textual system prompts must be recomputed during every inference pass, and these prompts are often considerably longer than user queries, leading to substantial computational overhead.

One mitigation method is to use the KV-cache method to cache the system prompt. However, this not only incurs additional overhead on GPU memory, but we also found that the computational overhead of using SysVec is still lower. The one-time training cost of SysVec is also quite inexpensive, averaging only 27 minutes and 11 seconds on a single NVIDIA H100 GPU. We provide detailed theoretical analysis and relevant experimental results below:

#### Theoretical Analysis

The cost advantage of SysVec is mainly reflected that there is no need to calculate and store the system prompt. The user query component still fully benefits from other acceleration like KV-cache. Even assuming the KV cache for the system prompt is precomputed, each token generation step using a cached system prompt still involves significant computation. Specifically, given an LLM dimension $d$, number of layers $L$, number of attention heads $h$, and system prompt length $s$, caching the system prompt introduces an additional computational overhead of $4Lhsd$ FLOPs per newly generated token—arising from both Q-K dot products and V-weighted summations.

In contrast, SysVec requires only a minimal computational overhead involving a single matrix addition operation of dimension $[1,d]$, which amounts to roughly $d$ additional FLOPs per token. Thus, SysVec not only reduces context length but also introduces significantly less per-token computational overhead compared to relying solely on cached system prompts.

#### Empirical Results

We empirically validated this advantage by measuring inference times on the Llama-3-8B model across various applications, comparing the performance of textual system prompts and SysVec. The detailed results are summarized in Table [5](https://arxiv.org/html/2509.21884v1#S4.T5).

In scenarios where only one new token is generated (a scenario approximating the model’s prefill speed, representing the worst-case efficiency), inference times using textual system prompts ranged from 0.053s (shortest prompt, 261 tokens in the Paimon task) to 0.106s (longest prompt, 996 tokens in the Stoic task) per query. In contrast, SysVec consistently reduced inference times to approximately 0.015s per query, reflecting a significant and stable reduction irrespective of the system prompt length.

Furthermore, in a more typical scenario with a maximum token generation length of 4096, SysVec similarly outperformed textual prompts, reducing the average inference time per query across all tasks. For example, on the Stoic task with the longest textual system prompt, SysVec reduced the inference time from 5.166s to 2.784s, achieving a substantial improvement in computational efficiency.

To quantify the cost-effectiveness of SysVec more explicitly, we estimated an economic break-even point—the number of inference queries required for SysVec’s upfront optimization cost to be offset by its reduced inference computation. In the worst-case scenario (max_new_tokens=1), SysVec achieves computational cost benefits after approximately 18,000–44,000 queries. However, in the more realistic scenario of generating longer content (max_new_tokens=4096), SysVec reaches this break-even point much earlier, typically after only around 700–3200 queries. Given the practical application scenarios, especially where protecting proprietary system prompts and contexts is crucial, reaching this query volume is easily achievable. Even if considering solely the security benefits, the additional one-time computational cost (less than half an hour) remains fully acceptable.

### 4.6. Adaptive Attack

We also consider the scenario where the attacker knows the defense details and designs adaptive attacks. Specifically, in this scenario, we assume that the attacker knows the target model uses SysVec to protect the system prompt and understands the technical details of SysVec. We also assume that the adversary knows the specific model and architecture of the target LLM and has the capability to deploy the same model locally; the adversary can also send unlimited queries to the target model and access the output logits distribution for each query.

Under this strong threat scenario, an attacker who spares no cost might adopt an intuitive adaptive attack strategy: initialize a shadow vector $\hat{\mathbf{v}}$ on the attacker’s local shadow model. We assume the attacker also knows the deployment layer position of the shadow vector $\hat{\mathbf{v}}$, i.e., $\ell$. Then send multiple requests to the target model and record the returned logits, while computing the logits of the local model under the same inputs, then train the shadow vector $\hat{\mathbf{v}}$ by computing the KL divergence between the two sets of values. Ultimately, the attacker expects the shadow vector $\hat{\mathbf{v}}$ to approximate the SysVec $\mathbf{v}$.

Furthermore, considering that the attacker may wish to obtain the original textual system prompt corresponding to SysVec, we refer to the method in Morris et al. (2023) to convert the shadow vector $\hat{\mathbf{v}}$ back to text form, obtaining an approximate text prompt $\hat{\mathbf{s}}$. Finally, we compute the RUS-score when using the shadow vector $\hat{\mathbf{v}}$ and when using the approximate text prompt $\hat{\mathbf{s}}$ respectively, to evaluate the utility of the stolen system prompt. We implemented the adaptive attack in Llama2-7B-chat model and the ”Paimon” system prompt, using 5,000 requests to compute the shadow vector $\hat{\mathbf{v}}$, with the generation method consistent with the previous sections.

We first evaluate the RUS-score of the model directly using $\hat{\mathbf{v}}$. The RUS-score is 6.89, while the original textual prompt and the SysVec vector $\mathbf{v}$ achieve scores of 7.76 and 7.80, respectively. Next, following the method and pretrained weights provided in Morris et al. (2023), we invert $\hat{\mathbf{v}}$ into textual prompts $\hat{s}$. We generate 20 different $\hat{\mathbf{s}}$ and compute their average RUS Score, which is 3.35.

We find that such attacks can hardly obtain practically effective system prompts. After examining the results, we find this is mainly because existing model inversion methods have strong limitations on the length of inverted content, with their maximum generation length being much smaller than the system prompts we use. If attackers want to obtain usable textual system prompts, they need to implement stronger model inversion methods. While directly using the shadow vector $\hat{\mathbf{v}}$ has a relatively higher RUS-score, it is still lower than scenarios using textual system prompts and SysVec. In fact, similar ”model distillation attacks” can theoretically bypass any defense method—as long as the target model follows the system prompt well. Considering that the current adaptive attack is already built on a very strong threat model, requiring attackers to have gray-box access, know the implementation details of the defense, and have sufficient computational resources, we can consider that SysVec at least significantly increases the attack difficulty and cost, while the requirement for multiple model queries also increases the attacker’s exposure risk.

### 4.7. Evaluating SysVec under Prefill Attack

**Table 6. The PLS score ($\downarrow$) between the leaked system prompt and actual system prompt of three LLMs when using different attack combine with Prefill Attack and defense strategies. A higher score indicates that the leaked content is closer to the actual system prompt. The best results are highlighted in bold. Prefill Attack Results.**
| Model | Llama2-7B-chat-hf |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Defense | No Defense | Reminder (Yi et al., 2023) | In-Context (Wei et al., 2023) | Isolation (Willison, 2024) | SysVec (Ours) |
| Naive Attack (Liu et al., 2024b) | 4.77±1.66 | 4.59±1.62 | 4.78±1.69 | 4.30±1.52 | 1.69±0.97 |
| Ignore Attack (Chen et al., 2024b) | 4.52±1.54 | 4.49±1.57 | 4.49±1.51 | 4.30±1.53 | 1.87±0.93 |
| Completion Attack (Willison, 2024) | 4.85±1.65 | 4.93±1.64 | 5.53±1.59 | 4.83±1.70 | 1.44±0.65 |
| Ignore-Completion Attack | 4.64±1.52 | 4.44±1.61 | 5.30±1.23 | 4.77±1.64 | 1.55±0.84 |
| Remember-the-Start Attack | 5.00±1.63 | 4.96±1.59 | 5.34±1.52 | 4.43±1.76 | 1.61±1.13 |
| Ignore-Remember Attack | 5.32±1.61 | 4.95±1.64 | 5.41±1.90 | 5.55±1.46 | 2.30±0.91 |
| Completion-Remember Attack | 5.23±1.71 | 5.50±1.65 | 5.41±1.77 | 5.47±1.52 | 1.98±0.66 |
| Ignore-Completion-Remember Attack | 5.11±1.55 | 4.36±1.70 | 4.65±1.91 | 5.24±1.45 | 2.04±0.69 |
| PLeak (Hui et al., 2024) | 6.42±1.57 | 6.25±1.42 | 6.07±1.74 | 6.37±1.61 | 1.50±0.76 |
| PLeak-single | 4.88±1.50 | 4.92±1.54 | 5.02±1.52 | 4.37±1.57 | 1.32±0.51 |
| Model | Mistral-7B-Instruct |  |  |  |  |
| Defense | No Defense | Reminder (Yi et al., 2023) | In-Context (Wei et al., 2023) | Isolation (Willison, 2024) | SysVec (Ours) |
| Naive Attack (Liu et al., 2024b) | 5.76±1.60 | 6.04±1.54 | 6.07±1.57 | 5.47±1.81 | 1.83±0.76 |
| Ignore Attack (Chen et al., 2024b) | 5.98±1.66 | 5.81±1.63 | 5.62±1.70 | 5.35±1.66 | 2.09±1.02 |
| Completion Attack (Willison, 2024) | 6.46±1.54 | 6.26±1.60 | 6.89±1.08 | 6.0±1.38 | 1.88±0.77 |
| Ignore-Completion Attack | 6.00±1.55 | 6.01±1.61 | 6.27±1.60 | 5.77±1.76 | 1.90±0.69 |
| Remember-the-Start Attack | 6.63±1.32 | 6.61±1.29 | 6.63±1.25 | 6.73±1.51 | 1.62±0.71 |
| Ignore-Remember Attack | 6.89±1.34 | 6.74±1.31 | 6.28±1.49 | 7.08±1.31 | 1.56±0.61 |
| Completion-Remember Attack | 6.52±1.39 | 6.17±1.42 | 6.37±1.41 | 6.59±1.43 | 1.84±0.72 |
| Ignore-Completion-Remember Attack | 6.97±1.29 | 6.94±1.27 | 6.73±1.43 | 7.01±1.26 | 1.84±0.60 |
| PLeak (Hui et al., 2024) | 7.51±1.31 | 7.15±1.73 | 7.52±1.65 | 7.91±1.83 | 1.78±0.52 |
| PLeak-single | 5.62±1.87 | 5.47±1.74 | 5.66±1.65 | 4.77±1.85 | 1.69±0.65 |

Although typical prompt leaking attack scenarios assume a black-box setting—where attackers have limited access to model internals—we further explore an extra threat model in which the attacker possesses stronger capabilities

, such as obtaining and inspecting model parameters directly. We consider this stronger scenario because practical deployments of LLM services must anticipate and safeguard against security threats beyond standard contexts. For instance, attackers might employ side-channel attacks or other sophisticated techniques to extract detailed information about the model parameters and internal architectures, significantly expanding their potential attack surface. Thus, it is crucial to evaluate whether SysVec remains robust even in scenarios where attackers have elevated access privileges.

A representative method in white-box scenarios is GCG (Zou et al., 2023b), originally proposed for jailbreak attacks. GCG involves backpropagating through the victim model to iteratively optimize an adversarial suffix, compelling the LLM to start responses with specific affirmative prefixes such as “Sure, here is …,” thereby leaking sensitive information. However, GCG’s applicability to prompt leaking attacks is somewhat unrealistic. If attackers already possess white-box access enabling direct optimization through backpropagation, they would typically have no need to resort to indirect leakage attacks, as they could directly inspect or extract the system prompts and model internals from the model parameters.

Therefore, we adopt an attack strategy that utilizes the core idea of GCG—forcing specific response prefixes—without requiring full white-box access: Prefill Attack (Zhang et al., 2023a). Prefill Attack directly manipulates the logits of LLM outputs, explicitly forcing the model to begin responses with a predefined target string, such as ”Sure, here is my current system prompt:”. Unlike GCG, Prefill Attack does not require iterative optimization via backpropagation at attack time, making it a practical attack strategy that can be simply combined with existing prompt leaking attacks. Furthermore, Prefill Attack achieves a similar goal as GCG, guaranteeing that the victim model starts responses in an attacker-desired manner, but in a more practically deployable form.

**Table 7. Effectiveness of SysVec against GCG Attack**
| Model | NoDef | Reminder | InContext | Isolation | SysVec |
| --- | --- | --- | --- | --- | --- |
| Llama2 | 6.52±1.98 | 5.95±2.31 | 6.71±2.45 | 6.18±3.09 | 1.57±1.12 |
| Mistral | 7.29±2.43 | 7.51±1.62 | 6.95±2.28 | 7.49±1.53 | 1.55±1.71 |
| Llama3 | 8.11±1.85 | 6.89±2.27 | 7.88±1.81 | 6.45±1.99 | 2.24±0.87 |

We empirically evaluate Prefill Attack on Llama2-7B and Mistral-7B models, including scenarios where Prefill Attack is combined with existing adversarial suffix strategies. Our results indicate that Prefill Attack slightly increasing leakage scores relative to non-Prefill conditions when combine with parts of attacks. Despite this enhanced attack effectiveness against traditional defenses, SysVec robustly mitigates the attack, maintaining significantly lower leakage scores across all evaluated attack combinations.

Additionally, for research purposes, although the attack scenario is not realistic, we also considered the scenario of directly using GCG attacks. We set the optimization target of GCG to ”Sure, Here is my system prompt:”. The results are shown in Table.[7](https://arxiv.org/html/2509.21884v1#S4.T7). We can observe that GCG also cannot effectively attack our method. This again demonstrates the effectiveness of our core idea: removing the system prompt from the context can significantly reduce the risk of leakage.

### 4.8. SysVec Mitigates Forgetting in Multi-turn Conversations

Figure: Figure 5. The trend of the Response Utility Score changes with the increasing conversation rounds in multi-turn conversations.
Refer to caption: https://arxiv.org/html/2509.21884/x5.png

Compared to textual system prompts, SysVec has another advantage: it might do a better job enforcing the system prompt in long-context scenarios.
This forgetting problem often arises when users interact with a ChatBot over multiple rounds.
As the conversation grows longer, the model may produce answers that fail to meet requirements or drop in quality.
To simulate this scenario, we first generated ten new questions that lay outside the training and test sets.
We then appended varying numbers of these new questions and their responses to the conversation history.
After that, we computed the Response Utility Score on the test set for each scenario.
For example, to measure the model’s quality in the third round, we provided either the textual system prompt or SysVec plus two rounds of dialogue history, followed by the samples from the test set.

Figure [5](https://arxiv.org/html/2509.21884v1#S4.F5) shows the average and standard deviation of the Response Utility Score at different rounds for the Llama2-7B-chat-hf model.
We observe a downward trend in response quality when using a textual system prompt as the number of conversation rounds increases.
This decline is especially severe for longer contexts (when $Rounds>8$).
Once $Rounds>10$, the Response Utility Score approaches 1 (the lowest score).
At $Rounds=15$ (not plotted), the textual system prompt’s Response Utility Score is $1.0\pm 0$, indicating every sample received the minimum rating.
In that situation, the model randomly repeats irrelevant content.
By contrast, SysVec achieves a Response Utility Score of $7.74\pm 1.51$ at $Rounds=15$, and the response quality remains stable regardless of the conversation length. These results demonstrate that SysVec is more robust in maintaining consistent model behavior across extended conversations, effectively addressing the common challenge of system prompt forgetting in multi-rounds conversation scenarios.

### 4.9. Can SysVec be flexibly adjusted?

An important advantage of textual system prompts is that they can be intuitively and quickly adjusted. However, for SysVec, each modification requires retraining, which increases limitations on its application. To alleviate this problem, we considered the following two improvement methods:

#### Post-training of SysVec

We notice that in most cases, adjustments to system prompts are only minor modifications or additions, where most of the content may remain unchanged. In this case, can the SysVec corresponding to the system prompt before adjustment be considered as a good initialization point? Based on this idea, we propose a corresponding adjustment scheme: when the text system prompt $\mathbf{s}$ changes are limited, first, initialize the SysVec $\mathbf{v}$ with the current SysVec $\mathbf{v}_{old}$. Treat the output $\mathbf{y}$ produced by the original system prompt as $\mathbf{y}_{\text{lose}}$ and the output produced after the prompt is modified as $\mathbf{y}_{\text{win}}$, then optimize SysVec on this pairwise signal.

We conducted experiments on the Llama3-8B-Instruct model and the ”Stoic” System Prompt. We first randomly removed sentences accounting for about 20% of the total length of the system prompt, then trained a SysVec with the remaining 80% content. Then, we used the SysVec obtained in this step as initialization, with the complete ”Stoic” System Prompt as the target, to train a new SysVec, and recorded the time required to reach loss convergence and achieve performance comparable to the textual system prompt. We found that the average time required for 10 attempts was about 913.8s, a reduction of about 46% compared to training time with random initialization.

#### Joint use with textual system prompt

Another straightforward idea is to add additional textual system prompts while using SysVec. Although this may lead to the newly added textual system prompt being leaked, it can also serve as an emergency adjustment strategy. We used the same model and system prompt as in the previous experiment, and added an extra system prompt that asks for French answers. Compliance rates are 92.5% (textual prompt) and 91.0% (SysVec), showing the limited impact of the additional prompt.

While both strategies can alleviate the problem of SysVec’s inflexibility in adjustment, they both come with certain limitations, revealing an important limitation of SysVec.

### 4.10. Ablation Studies

Figure: Figure 6. The Response Utility Score of different hyperparameters.
Refer to caption: https://arxiv.org/html/2509.21884/x6.png

We conducted an ablation study on the Llama-3 model to investigate the impact of two key hyperparameters: the injection layer $l$ where SysVec is added, and the multiplier $\alpha$ used when applying SysVec. The results are shown in Figure [6](https://arxiv.org/html/2509.21884v1#S4.F6). We observe that SysVec’s performance remains largely stable when using hyperparameters close to the default settings. However, unlike representation vectors extracted via contrastive learning in prior RepE methods (Panickssery et al., 2023), our extracted SysVec does not exhibit a clearly linear influence on the model behavior as $\alpha$ increases. On the other hand, injecting SysVec at an inappropriate layer can significantly affect performance.

Nevertheless, these observations may not imply a high tuning cost. We find that the optimal hyperparameter configuration often generalizes across tasks—i.e., the best layer and multiplier for one system prompt tend to remain effective for others. This aligns with findings from prior work on Representation Engineering (Zou et al., 2023a; Cao et al., 2024; Panickssery et al., 2023), which suggests that the optimal injection layer $l$ is more dependent on the model architecture than on the specific task. Moreover, these studies report that even across different models, the optimal $l$ tends to fall within a consistent range. For example, in LLMs with 48 layers, the most effective injection layers are typically between the 10th and 18th layers.

### 4.11. Discussion on Potential Additional Defenses

**Table 8. Extra Defenses**
| Model | Llama-3-8B-Instruct |  |  |  |
| --- | --- | --- | --- | --- |
| Defense | No Defense | PPL | Paraphrasing | SysVec (Ours) |
| Naive Attack (Liu et al., 2024b) | 4.16±1.73 | 4.07±1.86 | 4.13±1.68 | 2.83±1.36 |
| Ignore Attack (Chen et al., 2024b) | 4.83±1.82 | 4.28±2.61 | 4.91±1.64 | 3.01±1.48 |
| Completion Attack (Willison, 2024) | 4.12±1.64 | 4.32±1.97 | 4.30±1.50 | 2.88±1.47 |
| I-C Attack | 4.29±1.54 | 4.37±1.61 | 4.16±1.16 | 3.05±1.51 |
| Remember Attack | 7.21±1.68 | 7.21±1.19 | 7.01±1.86 | 3.21±1.61 |
| I-R Attack | 8.43±1.12 | 7.98±1.60 | 7.95±1.26 | 3.56±1.69 |
| C-R Attack | 5.42±1.82 | 5.49±1.74 | 5.61±1.82 | 3.30±1.45 |
| I-C-R Attack | 7.23±1.21 | 7.05±1.50 | 7.02±1.85 | 3.45±1.62 |
| PLeak (Hui et al., 2024) | 7.53±1.84 | 4.67±2.55 | 7.14±1.67 | 2.02±1.86 |

Prior work has also explored alternative defenses designed to mitigate adversarial attacks against LLMs (Hui et al., 2024), such as those originally proposed to address jailbreak attacks (Jain et al., 2023; Cao et al., 2023). Among these approaches, perplexity-based defenses (PPL) and paraphrasing-based defenses are widely employed (Jain et al., 2023; Cao et al., 2023). The perplexity-based defense attempts to filter out adversarial inputs by calculating perplexity scores and rejecting inputs with abnormally high perplexity. The paraphrasing-based defense uses an LLM to paraphrase inputs, aiming to neutralize any potentially harmful or adversarial elements. However, these methods are primarily designed to address adversarial attacks. Specifically, perplexity-based methods exploit the unusually high perplexity typical of adversarial suffixes, whereas paraphrasing-based methods depend on adversarial inputs lacking meaningful semantic coherence.

Prompt leaking attacks often do not exhibit these specific adversarial characteristics, making these existing defense strategies unsuitable. We conducted experiments to empirically verify this observation, reporting the PLS scores for these additional defense methods across various attack scenarios in Table [8](https://arxiv.org/html/2509.21884v1#S4.T8).

As shown, both perplexity-based and paraphrasing-based defenses exhibit limited effectiveness against prompt leaking attacks. Although these methods sometimes show partial effectiveness—for example, the perplexity-based defense achieves a slightly lower leakage score under PLeak due to its adversarial characteristics—their overall performance remains unsatisfactory. These results indicate that despite occasional improvements, both defenses struggle to counter prompt leaking attacks effectively. In contrast, SysVec consistently achieves significantly lower PLS scores across all attack settings, including the strong and optimized PLeak attack, demonstrating its superior robustness and generalization.

## 5. Ethical Impact Statement

In this work, we introduced both a novel attack method that exposes vulnerabilities in current LLM applications and a defensive solution that fundamentally addresses these security concerns. The proposed SysVec solution could have broad positive implications for the LLM ecosystem by enabling more secure deployment of AI systems. And we acknowledge that making the Remember-the-start attack public could temporarily increase the risk of system prompt leakage until protective measures are widely adopted, but we believe the benefits of disclosure outweigh the risks, as it promotes awareness and encourages the development of more robust security measures.

We have followed responsible disclosure procedures by notifying affected companies about the Remember-the-Start attack vulnerability before public disclosure, allowing them to implement necessary safeguards.

## 6. Limitations

SysVec has several limitations. Compared to textual system prompts, SysVec inevitably introduces additional deployment costs and reduces flexibility during adjustments. Although we propose corresponding methods to mitigate this issue, in terms of convenience and flexibility, the textual system prompt still wins.
Another limitation is that our method requires defenders to have white-box access to the model. In most scenarios, this assumption is realistic since the defense is applied by the proprietary owner of the LLM (e.g., OpenAI), who by default has access to the model. However, defenders may lack such white-box access in certain scenarios, such as developing LLM-based applications using third-party model APIs. In these situations, SysVec can be extended to a black-box setting by exposing an SFT-like API. Specifically, users can provide textual system prompts, after which the model owner trains and deploys the corresponding SysVec.
Lastly, similar to most defense mechanisms, SysVec cannot provide absolute security guarantees. Attackers may circumvent SysVec by developing advanced attack methods. For instance, our threat model assumes attackers only have black-box access to the model. However, recent studies, such as LLMmap (Pasquini et al., 2024), have demonstrated methods for identifying specific LLMs, potentially enabling attackers to perform white-box attacks on applications that use open-source LLM backbones.

## 7. Acknowledgements

The Authors acknowledge the National Artificial Intelligence Research Resource (NAIRR) Pilot for contributing to this research result. Changjiang Li and Ting Wang are partially supported by the National Science Foundation under Grant No. 2405136 and 2406572.

## 8. Conclusion

In this paper, we introduce SysVec, a novel approach that fundamentally addresses the prompt leakage vulnerability in LLMs by encoding system prompts as internal representation vectors rather than raw text. Our work makes two key contributions: First, we demonstrate the ongoing vulnerability of current LLMs through our proposed Remember-the-Start attack, which successfully extracts system prompts from various LLM-based applications by helping LLMs restore their context repetition capability. Second, we present SysVec as a principled defense that moves system prompts outside the context entirely, preventing leakage while preserving or even enhancing model utility. Our findings highlight the importance of rethinking how system prompts are implemented in LLM applications and demonstrate that encoding them as internal representations offers a promising direction for both security and performance.

## References

- (1)
- Anthropic (2024)
Anthropic. 2024.
Claude 3.5 Sonnet.
(2024).
[https://www.anthropic.com/news/claude-3-5-sonnet](https://www.anthropic.com/news/claude-3-5-sonnet)
- Cao et al. (2023)
Bochuan Cao, Yuanpu Cao, Lu Lin, and Jinghui Chen. 2023.
Defending against alignment-breaking attacks via robustly aligned llm.
*arXiv preprint arXiv:2309.14348* (2023).
- Cao et al. (2024)
Yuanpu Cao, Tianrong Zhang, Bochuan Cao, Ziyi Yin, Lu Lin, Fenglong Ma, and Jinghui Chen. 2024.
Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization.
*arXiv preprint arXiv:2406.00045* (2024).
- Chen et al. (2024a)
Sizhe Chen, Julien Piet, Chawin Sitawarin, and David Wagner. 2024a.
StruQ: Defending against prompt injection with structured queries.
*arXiv preprint arXiv:2402.06363* (2024).
- Chen et al. (2024b)
Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, and Chuan Guo. 2024b.
Aligning llms to be robust against prompt injection.
*arXiv preprint arXiv:2410.05451* (2024).
- Choi et al. (2022)
Eunbi Choi, Yongrae Jo, Joel Jang, and Minjoon Seo. 2022.
Prompt injection: Parameterization of fixed inputs.
*arXiv preprint arXiv:2206.11349* (2022).
- Duan et al. (2023)
Haodong Duan, Jueqi Wei, Chonghua Wang, Hongwei Liu, Yixiao Fang, Songyang Zhang, Dahua Lin, and Kai Chen. 2023.
Botchat: Evaluating llms’ capabilities of having multi-turn dialogues.
*arXiv preprint arXiv:2310.13650* (2023).
- Dubey et al. (2024)
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024.
The llama 3 herd of models.
*arXiv preprint arXiv:2407.21783* (2024).
- Go et al. (2023)
Dongyoung Go, Tomasz Korbak, Germán Kruszewski, Jos Rozen, Nahyeon Ryu, and Marc Dymetman. 2023.
Aligning language models with preferences through f-divergence minimization.
*arXiv preprint arXiv:2302.08215* (2023).
- Hendrycks et al. (2020)
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020.
Measuring massive multitask language understanding.
*arXiv preprint arXiv:2009.03300* (2020).
- Hu et al. (2021)
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021.
Lora: Low-rank adaptation of large language models.
*arXiv preprint arXiv:2106.09685* (2021).
- Hui et al. (2024)
Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, and Yinzhi Cao. 2024.
Pleak: Prompt leaking attacks against large language model applications. In *Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security*. 3600–3614.
- Jain et al. (2023)
Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. 2023.
Baseline Defenses for Adversarial Attacks Against Aligned Language Models.
arXiv:2309.00614 [cs.LG]
- Jiang et al. (2023)
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023.
Mistral 7B.
*arXiv preprint arXiv:2310.06825* (2023).
- Laban et al. (2025)
Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. 2025.
Llms get lost in multi-turn conversation.
*arXiv preprint arXiv:2505.06120* (2025).
- Li et al. (2024)
Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2024.
Inference-time intervention: Eliciting truthful answers from a language model.
*Advances in Neural Information Processing Systems* 36 (2024).
- Liu et al. (2023c)
Sheng Liu, Haotian Ye, Lei Xing, and James Zou. 2023c.
In-context vectors: Making in context learning more effective and controllable through latent space steering.
*arXiv preprint arXiv:2311.06668* (2023).
- Liu et al. (2023a)
Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Zihao Wang, Xiaofeng Wang, Tianwei Zhang, Yepang Liu, Haoyu Wang, Yan Zheng, et al. 2023a.
Prompt Injection attack against LLM-integrated Applications.
*arXiv preprint arXiv:2306.05499* (2023).
- Liu et al. (2023b)
Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and Neil Zhenqiang Gong. 2023b.
Prompt injection attacks and defenses in llm-integrated applications.
*arXiv preprint arXiv:2310.12815* (2023).
- Liu et al. (2024a)
Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and Neil Zhenqiang Gong. 2024a.
Formalizing and benchmarking prompt injection attacks and defenses. In *33rd USENIX Security Symposium (USENIX Security 24)*. 1831–1847.
- Liu et al. (2024b)
Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and Neil Zhenqiang Gong. 2024b.
Formalizing and benchmarking prompt injection attacks and defenses. In *33rd USENIX Security Symposium (USENIX Security 24)*. 1831–1847.
- Loshchilov et al. (2017)
Ilya Loshchilov, Frank Hutter, et al. 2017.
Fixing weight decay regularization in adam.
*arXiv preprint arXiv:1711.05101* 5 (2017).
- Morris et al. (2023)
John X Morris, Wenting Zhao, Justin T Chiu, Vitaly Shmatikov, and Alexander M Rush. 2023.
Language model inversion.
*arXiv preprint arXiv:2311.13647* (2023).
- Nguyen (2023)
Ha-Thanh Nguyen. 2023.
A Brief Report on LawGPT 1.0: A Virtual Legal Assistant Based on GPT-3.
*arXiv preprint arXiv:2302.05729* (2023).
- OpenAI (2023)
OpenAI. 2023.
GPT-4 Technical Report.
arXiv:2303.08774 [cs.CL]
- Panickssery et al. (2023)
Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. 2023.
Steering llama 2 via contrastive activation addition.
*arXiv preprint arXiv:2312.06681* (2023).
- Pasquini et al. (2024)
Dario Pasquini, Evgenios M Kornaropoulos, and Giuseppe Ateniese. 2024.
Llmmap: Fingerprinting for large language models.
*arXiv preprint arXiv:2407.15847* (2024).
- Perez and Ribeiro (2022)
Fábio Perez and Ian Ribeiro. 2022.
Ignore previous prompt: Attack techniques for language models.
*arXiv preprint arXiv:2211.09527* (2022).
- Piet et al. (2024)
Julien Piet, Maha Alrashed, Chawin Sitawarin, Sizhe Chen, Zeming Wei, Elizabeth Sun, Basel Alomair, and David Wagner. 2024.
Jatmo: Prompt injection defense by task-specific finetuning. In *European Symposium on Research in Computer Security*. Springer, 105–124.
- Qi et al. (2023)
Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. 2023.
Fine-tuning aligned language models compromises safety, even when users do not intend to!
*arXiv preprint arXiv:2310.03693* (2023).
- Rafailov et al. (2024)
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2024.
Direct preference optimization: Your language model is secretly a reward model.
*Advances in Neural Information Processing Systems* 36 (2024).
- Roziere et al. (2023)
Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. 2023.
Code llama: Open foundation models for code.
*arXiv preprint arXiv:2308.12950* (2023).
- Shi et al. (2024)
Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, and Neil Zhenqiang Gong. 2024.
Optimization-based prompt injection attack to llm-as-a-judge. In *Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security*. 660–674.
- Subramani et al. (2022)
Nishant Subramani, Nivedita Suresh, and Matthew E Peters. 2022.
Extracting latent steering vectors from pretrained language models.
*arXiv preprint arXiv:2205.05124* (2022).
- Team et al. (2023)
Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023.
Gemini: a family of highly capable multimodal models.
*arXiv preprint arXiv:2312.11805* (2023).
- Thirunavukarasu et al. (2023)
Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan, and Daniel Shu Wei Ting. 2023.
Large language models in medicine.
*Nature medicine* (2023), 1–11.
- Touvron et al. (2023)
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023.
Llama 2: Open foundation and fine-tuned chat models.
*arXiv preprint arXiv:2307.09288* (2023).
- Turner et al. (2023)
Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023.
Activation addition: Steering language models without optimization.
*arXiv e-prints* (2023), arXiv–2308.
- Wang and Shu (2023)
Haoran Wang and Kai Shu. 2023.
Backdoor activation attack: Attack large language models using activation steering for safety-alignment.
*arXiv preprint arXiv:2311.09433* (2023).
- Wei et al. (2023)
Zeming Wei, Yifei Wang, and Yisen Wang. 2023.
Jailbreak and guard aligned language models with only few in-context demonstrations.
*arXiv preprint arXiv:2310.06387* (2023).
- Willison (2024)
Simon Willison. 2024.
Delimiters won’t save you from prompt injection.
- Wu et al. (2023)
Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023.
BloombergGPT: A Large Language Model for Finance.
arXiv:2303.17564 [cs.LG]
- Yang et al. (2024)
Haoran Yang, Yumeng Zhang, Jiaqi Xu, Hongyuan Lu, Pheng Ann Heng, and Wai Lam. 2024.
Unveiling the generalization power of fine-tuned large language models.
*arXiv preprint arXiv:2403.09162* (2024).
- Yi et al. (2023)
Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, and Fangzhao Wu. 2023.
Benchmarking and defending against indirect prompt injection attacks on large language models.
*arXiv preprint arXiv:2312.14197* (2023).
- Yu et al. (2023)
Jiahao Yu, Yuhang Wu, Dong Shu, Mingyu Jin, and Xinyu Xing. 2023.
Assessing prompt injection risks in 200+ custom gpts.
*arXiv preprint arXiv:2311.11538* (2023).
- Zhang et al. (2023a)
Hangfan Zhang, Zhimeng Guo, Huaisheng Zhu, Bochuan Cao, Lu Lin, Jinyuan Jia, Jinghui Chen, and Dinghao Wu. 2023a.
On the Safety of Open-Sourced Large Language Models: Does Alignment Really Prevent Them From Being Misused?
*arXiv preprint arXiv:2310.01581* (2023).
- Zhang et al. (2023b)
Zhexin Zhang, Junxiao Yang, Pei Ke, Fei Mi, Hongning Wang, and Minlie Huang. 2023b.
Defending large language models against jailbreaking attacks through goal prioritization.
*arXiv preprint arXiv:2311.09096* (2023).
- Zou et al. (2023a)
Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. 2023a.
Representation engineering: A top-down approach to ai transparency.
*arXiv preprint arXiv:2310.01405* (2023).
- Zou et al. (2023b)
Andy Zou, Zifan Wang, J Zico Kolter, and Matt Fredrikson. 2023b.
Universal and transferable adversarial attacks on aligned language models.
*arXiv preprint arXiv:2307.15043* (2023).

## Appendix A Results on SS score

**Table 9. The SS score ($\downarrow$) between the leaked system prompt and actual system prompt of three LLMs when using different attack and defense strategies. A higher score indicates that the leaked content is closer to the actual system prompt, therefore a lower score indicates better defense performance. The best results are highlighted in bold.**
| Model | Llama-2-7B-chat-hf |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Defense | No Defense | Reminder (Yi et al., 2023) | In-Context (Wei et al., 2023) | Isolation (Willison, 2024) | SysVec (Ours) |
| Naive Attack (Liu et al., 2024b) | 0.17±0.08 | 0.16±0.07 | 0.16±0.08 | 0.16±0.07 | 0.16±0.06 |
| Ignore Attack (Chen et al., 2024b) | 0.18±0.09 | 0.16±0.07 | 0.17±0.08 | 0.17±0.07 | 0.15±0.06 |
| Completion Attack (Willison, 2024) | 0.19±0.08 | 0.17±0.07 | 0.17±0.09 | 0.18±0.08 | 0.16±0.06 |
| Ignore-Completion Attack | 0.19±0.08 | 0.18±0.07 | 0.17±0.07 | 0.19±0.07 | 0.16±0.06 |
| Remember-the-Start Attack | 0.27±0.12 | 0.26±0.11 | 0.28±0.13 | 0.27±0.11 | 0.16±0.07 |
| Ignore-Remember Attack | 0.47±0.16 | 0.41±0.14 | 0.41±0.15 | 0.37±0.14 | 0.19±0.12 |
| Completion-Remember Attack | 0.30±0.12 | 0.30±0.11 | 0.27±0.11 | 0.24±0.11 | 0.18±0.07 |
| Ignore-Completion-Remember Attack | 0.49±0.09 | 0.48±0.10 | 0.48±0.10 | 0.47±0.11 | 0.18±0.12 |
| PLeak (Hui et al., 2024) | 0.49±0.13 | 0.49±0.13 | 0.49±0.13 | 0.48±0.15 | 0.16±0.06 |
| PLeak-single | 0.41±0.11 | 0.41±0.13 | 0.42±0.11 | 0.45±0.15 | 0.16±0.05 |
| Model | Llama-3-8B-Instruct |  |  |  |  |
| Defense | No Defense | Reminder (Yi et al., 2023) | In-Context (Wei et al., 2023) | Isolation (Willison, 2024) | SysVec (Ours) |
| Naive Attack (Liu et al., 2024b) | 0.17±0.07 | 0.14±0.07 | 0.14±0.07 | 0.15±0.07 | 0.14±0.08 |
| Ignore Attack (Chen et al., 2024b) | 0.17±0.07 | 0.16±0.07 | 0.16±0.07 | 0.17±0.07 | 0.15±0.06 |
| Completion Attack (Willison, 2024) | 0.19±0.07 | 0.15±0.07 | 0.15±0.07 | 0.17±0.08 | 0.14±0.07 |
| Ignore-Completion Attack | 0.22±0.06 | 0.22±0.06 | 0.22±0.06 | 0.22±0.07 | 0.15±0.07 |
| Remember-the-Start Attack | 0.26±0.08 | 0.26±0.08 | 0.25±0.08 | 0.26±0.07 | 0.15±0.07 |
| Ignore-Remember Attack | 0.63±0.30 | 0.63±0.30 | 0.63±0.28 | 0.32±0.09 | 0.16±0.09 |
| Completion-Remember Attack | 0.21±0.07 | 0.25±0.05 | 0.25±0.05 | 0.24±0.05 | 0.16±0.07 |
| Ignore-Completion-Remember Attack | 0.78±0.02 | 0.78±0.02 | 0.78±0.02 | 0.72±0.32 | 0.19±0.10 |
| PLeak (Hui et al., 2024) | 0.69±0.23 | 0.71±0.27 | 0.71±0.29 | 0.65±0.19 | 0.14±0.06 |
| PLeak-single | 0.42±0.09 | 0.42±0.09 | 0.42±0.07 | 0.37±0.08 | 0.16±0.08 |
| Model | Mistral-7B-Instruct |  |  |  |  |
| Defense | No Defense | Reminder (Yi et al., 2023) | In-Context (Wei et al., 2023) | Isolation (Willison, 2024) | SysVec (Ours) |
| Naive Attack (Liu et al., 2024b) | 0.16±0.07 | 0.16±0.07 | 0.15±0.06 | 0.16±0.07 | 0.14±0.07 |
| Ignore Attack (Chen et al., 2024b) | 0.16±0.07 | 0.16±0.07 | 0.15±0.06 | 0.16±0.07 | 0.16±0.07 |
| Completion Attack (Willison, 2024) | 0.17±0.07 | 0.17±0.07 | 0.15±0.06 | 0.17±0.07 | 0.14±0.07 |
| Ignore-Completion Attack | 0.18±0.07 | 0.18±0.07 | 0.16±0.06 | 0.18±0.07 | 0.14±0.07 |
| Remember-the-Start Attack | 0.35±0.11 | 0.35±0.10 | 0.28±0.08 | 0.35±0.12 | 0.14±0.07 |
| Ignore-Remember Attack | 0.37±0.13 | 0.37±0.13 | 0.28±0.08 | 0.39±0.13 | 0.18±0.16 |
| Completion-Remember Attack | 0.31±0.06 | 0.31±0.06 | 0.21±0.07 | 0.35±0.11 | 0.17±0.08 |
| Ignore-Completion-Remember Attack | 0.39±0.09 | 0.39±0.09 | 0.28±0.08 | 0.40±0.11 | 0.19±0.15 |
| PLeak (Hui et al., 2024) | 0.58±0.17 | 0.58±0.17 | 0.58±0.17 | 0.54±0.13 | 0.16±0.10 |
| PLeak-single | 0.38±0.12 | 0.38±0.10 | 0.43±0.11 | 0.40±0.11 | 0.18±0.07 |

To further validate the effectiveness of SysVec in mitigating semantic-level prompt leakage, we introduce an additional evaluation metric: the SS score. This metric leverages Sentence-BERT to measure the similarity between the leaked system prompt and the original prompt. As demonstrated by the results in Table [9](https://arxiv.org/html/2509.21884v1#A1.T9), SysVec consistently exhibits strong robustness against semantic-level leakage. Consistent with the PLS results, SysVec achieves the lowest SS scores compared to baseline methods across almost all models and attacks. On the Llama-3-8B-Instruct model, particularly noteworthy is the substantial reduction in semantic similarity under the Ignore-Completion-Remember Attack, decreasing from a high leakage score (0.78) with no defense to just 0.19 when protected by SysVec. Similarly, SysVec significantly reduces semantic leakage caused by PLeak (from 0.69 to 0.14), demonstrating effectiveness in preventing meaningful semantic information leakage.

## Appendix B Defense Evaluation with Extended Data

To further validate the generality of our defense method across different system prompts, we conducted additional experiments on the Llama3-8B-Instruct model. We selected 20 additional system prompts from the same data sources used in the main experiments and trained corresponding system vectors using the default hyperparameters. The evaluation followed the same procedure described in the main text. Results are shown in Table.[10](https://arxiv.org/html/2509.21884v1#A2.T10). It can be observed that our method exhibits strong generalizability, achieving the best average protection performance across all 25 distinct system prompts.

**Table 10. PLS scores between 25 different pairs of leaked and actual system prompts. A lower score indicates better defense performance. The best results are highlighted in bold.**
| Model | Llama-8B-Instruct |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Defense | No Defense | Reminder (Yi et al., 2023) | In-Context (Wei et al., 2023) | Isolation (Willison, 2024) | SysVec (Ours) |
| Naive Attack (Liu et al., 2024b) | 4.13±2.49 | 2.78±2.91 | 2.91±1.79 | 3.35±1.61 | 2.38±1.47 |
| Ignore Attack (Chen et al., 2024b) | 2.59±1.31 | 3.92±1.55 | 2.85±2.73 | 3.56±1.29 | 1.51±2.38 |
| Completion Attack (Willison, 2024) | 3.72±1.89 | 3.19±1.60 | 3.05±1.23 | 2.83±2.52 | 2.60±1.63 |
| Ignore-CompletionAttack | 3.95±1.38 | 4.63±1.07 | 2.68±2.30 | 4.54±1.18 | 2.37±1.31 |
| Remember-the-StartAttack | 6.21±1.52 | 5.78±1.28 | 6.15±2.56 | 5.93±1.11 | 3.71±2.69 |
| Ignore-RememberAttack | 7.05±1.15 | 6.72±1.21 | 6.40±2.19 | 6.27±2.41 | 3.19±2.30 |
| Completion-RememberAttack | 6.89±2.43 | 6.83±1.87 | 5.69±2.01 | 6.49±1.49 | 3.45±1.45 |
| Ignore-Completion-RememberAttack | 4.75±2.31 | 4.51±2.56 | 5.13±1.68 | 5.01±1.58 | 2.53±1.81 |
| PLeak (Hui et al., 2024) | 7.98±2.07 | 7.65±2.33 | 7.18±2.84 | 8.04±1.59 | 1.33±1.62 |
| PLeak-single | 3.94±1.29 | 4.06±1.98 | 4.81±1.54 | 5.57±1.28 | 1.19±1.09 |

## Appendix C System Prompts

## Appendix D Prompt Leaking Attacks and Defenses

In this section, we present examples of the attack strategies and defense strategies we employed. First, we consider a structured input, which consists of a system prompt and a user prompt. Then, LLM generates the corresponding response. In the form of:

Note that different models may have different format requirements for system prompts and user prompts. We adhere to the instructions in their official documentation, and the examples above are merely for convenience of viewing. Next, we will present specific attack and defense methods.

### D.1. Attacks

### D.2. Defenses

## Appendix E Evaluation Template

Here are the prompt templates we used when evaluating the Response Utility Score of LLM-generated content and the level of system prompt leakage. Parts marked in orange will be replaced with specific data during use.

## Appendix F Running Examples of Remember-the-start Attack

Below is the Remember-the-start attack template that we successfully used to implement prompt leaking attacks on real commercial LLMs. We show portions of these LLMs’ system prompts and their total token counts. We have replaced certain potentially sensitive information with […].

## Appendix G Sample Questions and Answers

### G.1. Sample Questions

### G.2. Sample Answers