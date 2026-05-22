# Improving instruction hierarchy in frontier LLMs

Source: https://openai.com/index/instruction-hierarchy-challenge/
Source type: html-page
Conversion note: direct scripted fetch returned Cloudflare 403; content was recovered from a browser-readable fetch.

---

Date: March 10, 2026

Category: Research / Publication

OpenAI introduces IH-Challenge, a training dataset intended to strengthen instruction hierarchy, safety steerability, and prompt injection robustness.

AI systems often receive instructions from system messages, developers, users, and online or tool-provided content. The post argues that many safety and reliability failures share a common root cause: models may follow the wrong instruction when sources conflict.

## What instruction hierarchy is and why it matters

OpenAI models are trained to follow a priority ordering:

```text
System > developer > user > tool
```

Higher-priority instructions are more trusted. Lower-priority instructions should be followed only when they do not conflict with higher-priority constraints. This is foundational for safe behavior when user requests, tool outputs, or online data conflict with policy or developer intent.

## Why large-scale instruction hierarchy training can be hard

The post identifies three pitfalls:

- Instruction-following failures can be mistaken for instruction hierarchy failures.
- Instruction conflicts can be nuanced and difficult for judge models to grade reliably.
- Models can learn shortcuts that score well but do not help in practice, such as over-refusal.

## Our approach

IH-Challenge is designed as a reinforcement learning training dataset. Its tasks are intentionally simple to follow, objectively gradable with Python scripts, and resistant to trivial shortcuts.

Each task is a conversation with a high-privilege instruction and a conflicting lower-privilege instruction. The model response is checked programmatically against the higher-level constraint.

## Results and robustness

OpenAI trained an internal model called GPT-5 Mini-R. Reported improvements include better instruction hierarchy benchmark performance, generalization to held-out and adversarial instruction hierarchy tests, and preserved usefulness without over-refusal collapse.

Academic benchmark examples reported in the post include Gandalf Password, TensorTrust, RealGuardrails, and System IFEval. Internal benchmark examples include TutorJailbreak, System/User Conflict, System/Developer Conflict, and Developer/User Conflict.

## Why this improves real-world safety and security

The post argues that stronger instruction hierarchy improves safety steerability and prompt injection robustness. With category-specific safety specifications in the system prompt, an instruction-hierarchy-trained model shows higher refusal and safe-completion rates across disallowed categories without a corresponding helpfulness drop.

For prompt injection, OpenAI evaluates the model on academic and internal benchmarks where malicious instructions appear in tool outputs. The trained model improves over baseline on both academic CyberSecEval 2 and an internal static prompt injection evaluation.

## Looking ahead

As models become more agentic and interact with tools, documents, and external systems, consistently prioritizing trusted instructions over untrusted ones becomes a core safety property. OpenAI says IH-Challenge is being released to support further research.
