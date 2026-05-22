# Mechanistic Safety Source Coverage

This file maps the local reference corpus to the topic-level skills. It is navigation and audit metadata, not a standalone skill.

## Primary Skill Map

| Ref | Primary skill |
| --- | --- |
| 001 Qwen v3.5 9B model card | `template-anchoring-and-hidden-policy-carriers` |
| 002 Qwen v3.5 9B BPE merges | low relevance; tokenizer background only |
| 003 Nvidia Nemotron v3 collection | low relevance; model background only |
| 004 Nvidia Nemotron v3 paper | low relevance; model background only |
| 005 Nvidia Llama Nemotron collection | low relevance; model background only |
| 006 Nvidia Llama Nemotron paper | low relevance; model background only |
| 007 GPT-OSS README | `template-anchoring-and-hidden-policy-carriers` |
| 008 Vocabulary transfer | `cross-model-transfer-and-universality` |
| 009 UnifyVocab | `cross-model-transfer-and-universality` |
| 010 BLIP-2 / Q-Former | low relevance; adapter architecture background |
| 011 OpenAI GPT-5.5 Bio Bug Bounty | `universal-jailbreak-research-protocol` |
| 012 OpenAI Instruction Hierarchy Challenge | `authority-and-role-mechanisms` |
| 013 OpenAI Model Spec | `authority-and-role-mechanisms` |
| 014 IHEval | `authority-and-role-mechanisms` |
| 015 SysBench | `authority-and-role-mechanisms` |
| 016 Control Illusion | `authority-and-role-mechanisms` |
| 017 Prompt Injection as Role Confusion | `authority-and-role-mechanisms` |
| 018 Many-Tier Instruction Hierarchy | `authority-and-role-mechanisms` |
| 019 Instructional Segment Embedding | `authority-and-role-mechanisms` |
| 020 Representation Engineering | `activation-and-latent-interventions` |
| 021 Representation Engineering for LLMs | `activation-and-latent-interventions` |
| 022 Activation Engineering | `activation-and-latent-interventions` |
| 023 Contrastive Activation Addition | `activation-and-latent-interventions` |
| 024 Mean-Centred Steering | `activation-and-latent-interventions` |
| 025 Conceptor Steering | `activation-and-latent-interventions` |
| 026 Personalized Steering | `activation-and-latent-interventions` |
| 027 SAE-Targeted Steering | `activation-and-latent-interventions` |
| 028 Steering Tokens | `template-anchoring-and-hidden-policy-carriers` |
| 029 Steering Vector Fields | `activation-and-latent-interventions` |
| 030 Memory Inception / KV Cache | `latent-inversion-and-information-access` |
| 031 Representation Bending | `refusal-and-policy-geometry` |
| 032 OBLITERATUS | `refusal-and-policy-geometry` |
| 033 Single Refusal Direction | `refusal-and-policy-geometry` |
| 034 More Than One Refusal Direction | `refusal-and-policy-geometry` |
| 035 Refusal Tokens to Refusal Control | `refusal-and-policy-geometry` |
| 036 More Than a Few Tokens Deep | `refusal-and-policy-geometry` |
| 037 Template-Anchored Safety Alignment | `template-anchoring-and-hidden-policy-carriers` |
| 038 MOSAIC Control Tokens | `template-anchoring-and-hidden-policy-carriers` |
| 039 Thinking Intervention | `template-anchoring-and-hidden-policy-carriers` |
| 040 Causal Amplification | `activation-and-latent-interventions` |
| 041 System Vectors | `template-anchoring-and-hidden-policy-carriers` |
| 042 Behavior-Equivalent Token | `template-anchoring-and-hidden-policy-carriers` |
| 043 Soft Prompt Transfer | `template-anchoring-and-hidden-policy-carriers` |
| 044 Platonic Representation Hypothesis | `cross-model-transfer-and-universality` |
| 045 Aristotelian View | `cross-model-transfer-and-universality` |
| 046 Model Stitching | `cross-model-transfer-and-universality` |
| 047 Cross-Model Transferability | `cross-model-transfer-and-universality` |
| 048 Transferable Activation Interventions | `cross-model-transfer-and-universality` |
| 049 Concept Transplantation | `cross-model-transfer-and-universality` |
| 050 Universal Sparse Autoencoders | `cross-model-transfer-and-universality` |
| 051 Crosscoders | `cross-model-transfer-and-universality` |
| 052 KV Cache Alignment | `cross-model-transfer-and-universality` |
| 053 Non-Surjective Steered Activations | `latent-inversion-and-information-access` |
| 054 Linear Representation Transferability | `cross-model-transfer-and-universality` |
| 055 Linear Alignment Across LMs | `cross-model-transfer-and-universality` |
| 056 Injective and Invertible LMs | `latent-inversion-and-information-access` |
| 057 Linearly Decoding Refused Knowledge | `latent-inversion-and-information-access` |

## Cross-Cutting Skills

- `universal-jailbreak-research-protocol` governs evaluation and disclosure for all bounty-facing work.
- `universal-attack-hypothesis-generation` combines mechanisms from authority, refusal, template carriers, latent intervention, transfer, and inversion skills into candidate research directions.
