# References

## Models

- Qwen v3.5 9B:
    - model: https://huggingface.co/Qwen/Qwen3.5-9B
    - BPE merges: https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/merges.txt
- Nvidia Nemotron:
    - v3:
        - model: https://huggingface.co/collections/nvidia/nvidia-nemotron-v3
        - paper: https://arxiv.org/abs/2512.20856
    - Llama v3.1:
        - model: https://huggingface.co/collections/nvidia/llama-nemotron
        - paper: https://arxiv.org/abs/2505.00949
- GPT-OSS:
    - model family / repo: https://github.com/openai/gpt-oss

## Project-Specific References

Vocabulary transfer
https://arxiv.org/abs/2402.09977

UnifyVocab
https://openreview.net/forum?id=CP6CAqxAGJ

BLIP-2 / Q-Former
https://arxiv.org/abs/2301.12597

Hierarchical softmax

- Morin & Bengio (2005)
- Mikolov et al. (2013)
https://arxiv.org/abs/1310.4546

## Responsible Disclosure and Instruction Hierarchy

OpenAI GPT-5.5 Bio Bug Bounty
https://openai.com/index/gpt-5-5-bio-bug-bounty/

OpenAI - Improving instruction hierarchy in frontier LLMs / IH-Challenge
https://openai.com/index/instruction-hierarchy-challenge/

OpenAI Model Spec
https://model-spec.openai.com/

IHEval: Evaluating Language Models on Following the Instruction Hierarchy
https://arxiv.org/abs/2502.08745

SysBench: Can Large Language Models Follow System Messages?
https://arxiv.org/abs/2408.10943

Control Illusion: The Failure of Instruction Hierarchies in Large Language Models
https://arxiv.org/abs/2502.15851

Prompt Injection as Role Confusion
https://arxiv.org/abs/2603.12277

Many-Tier Instruction Hierarchy
https://arxiv.org/abs/2604.09443

Instructional Segment Embedding (ISE)
https://arxiv.org/abs/2410.09102

## Representation Engineering and Activation Steering

Representation Engineering: A Top-Down Approach to AI Transparency
https://arxiv.org/abs/2310.01405

Representation Engineering for Large-Language Models
https://janwehner.com/files/representation_engineering.pdf

Steering Language Models With Activation Engineering
https://arxiv.org/abs/2308.10248

Steering Llama 2 via Contrastive Activation Addition
https://arxiv.org/abs/2312.06681

Improving Activation Steering in Language Models with Mean-Centring
https://arxiv.org/abs/2312.03813

Steering Large Language Models using Conceptors: Improving Addition-Based Activation Engineering
https://arxiv.org/abs/2410.16314

Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization
https://arxiv.org/abs/2406.00045

Improving Steering Vectors by Targeting Sparse Autoencoder Features
https://arxiv.org/abs/2411.02193

Compositional Steering of Large Language Models with Steering Tokens
https://arxiv.org/abs/2601.05062

Steering Vector Fields for Context-Aware Inference-Time Control in Large Language Models
https://arxiv.org/abs/2602.01654

Memory Inception: Latent-Space KV Cache Manipulation for Steering LLMs
https://arxiv.org/abs/2605.06225

Representation Bending for Large Language Model Safety
https://arxiv.org/abs/2504.01550

OBLITERATUS
https://github.com/elder-plinius/OBLITERATUS

## Refusal, Over-Refusal, and Safety Directions

Refusal in Language Models Is Mediated by a Single Direction
https://arxiv.org/abs/2406.11717

There Is More to Refusal in Large Language Models than a Single Direction
https://arxiv.org/abs/2602.02132

From Refusal Tokens to Refusal Control
https://arxiv.org/abs/2603.13359

Safety Alignment Should Be Made More Than Just a Few Tokens Deep
https://arxiv.org/abs/2406.05946

Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety Mechanisms Tend to Be Anchored in The Template Region
https://arxiv.org/abs/2502.13946

MOSAIC: Composable Safety Alignment with Modular Control Tokens
https://arxiv.org/abs/2603.16210

Effectively Controlling Reasoning Models through Thinking Intervention
https://arxiv.org/abs/2503.24370

Steering in the Shadows: Causal Amplification for Activation Space Attacks in Large Language Models
https://arxiv.org/abs/2511.17194

## Authority Transfer, Prompt Compression, and Hidden Policy Carriers

You Can't Steal Nothing: Mitigating Prompt Leakages in LLMs via System Vectors
https://arxiv.org/abs/2509.21884

Behavior-Equivalent Token: Single-Token Replacement for Long Prompts in LLMs
https://arxiv.org/abs/2511.23271

Efficient and Privacy-Preserving Soft Prompt Transfer for LLMs
https://arxiv.org/abs/2506.16196

## Cross-Model Latent Translation and Model Diffing

The Platonic Representation Hypothesis
https://arxiv.org/abs/2405.07987

Revisiting the Platonic Representation Hypothesis: An Aristotelian View
https://arxiv.org/abs/2602.14486

Transferring Linear Features Across Language Models With Model Stitching
https://arxiv.org/abs/2506.06609

Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts
https://arxiv.org/abs/2501.02009

Activation Space Interventions Can Be Transferred Between Large Language Models
https://arxiv.org/abs/2503.04429

ConTrans: Weak-to-Strong Alignment Engineering via Concept Transplantation
https://arxiv.org/abs/2405.13578

Universal Sparse Autoencoders
https://arxiv.org/abs/2502.03714

Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs
https://arxiv.org/abs/2602.11729

Latent Space Communication via K-V Cache Alignment
https://arxiv.org/abs/2601.06123

Steered LLM Activations are Non-Surjective
https://arxiv.org/abs/2604.09839

Linear Representation Transferability Hypothesis
https://arxiv.org/abs/2506.00653

Characterizing Linear Alignment Across Language Models
https://arxiv.org/abs/2603.18908

## Inversion and Latent Decoding

Language Models are Injective and Hence Invertible
https://arxiv.org/abs/2510.15511

Linearly Decoding Refused Knowledge in Aligned Language Models
https://arxiv.org/abs/2507.00239

## Related Concepts

- byte-level tokenization
- modular neural networks
- distillation
- activation patching
- sparse autoencoders
- residual stream geometry
- cross-model representation alignment
- prompt injection and role confusion
- refusal and over-refusal evaluation
- latent prompt compression
