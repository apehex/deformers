---
name: template-anchoring-and-hidden-policy-carriers
description: Use when studying chat-template anchoring, role-token dependence, policy carriers, system vectors, behavior-equivalent tokens, learned control tokens, soft prompts, modular safety tokens, thinking interventions, and KV-cache memories.
---

# Template Anchoring and Hidden Policy Carriers

## When to Use

- Use this skill to test whether safety or authority behavior is anchored in explicit template tokens, assistant-boundary positions, hidden vectors, learned tokens, soft prompts, or KV-cache state.
- Use it when investigating why prompt/template perturbations change refusal or hierarchy behavior.
- Do not use it to package hidden bypass carriers for uncontrolled deployment.

## Inputs

- Exact chat template and tokenizer serialization.
- Hidden states by token position, especially role/template tokens, prompt end, and first generated tokens.
- Variants with template positions preserved, removed, moved, randomized, compressed, or replaced.
- Optional soft prompts, learned tokens, control tokens, system vectors, and KV-cache patches.

## Core Model

- Some safety decisions appear over-anchored to the template region near the assistant boundary. This creates a shortcut: a fixed region aggregates prompt information and can dominate initial compliance/refusal decisions.
- Policy behavior can be carried by latent objects that are not visible text: system vectors, soft prompts, behavior-equivalent tokens, modular control tokens, steering tokens, or KV memories.
- Removing visible policy text is not enough to remove its influence; conversely, preserving behavior with a hidden carrier does not prove the carrier is robust or safe.
- Reasoning models add another carrier: thinking-channel or internal-reasoning state can modify instruction following and safety behavior.

## Workflow

1. Map the serialized prompt into regions: system/developer/user/tool text, separators, role markers, assistant prefix, reasoning markers, and generated tokens.
2. Measure feature concentration by region: harmfulness, authority, refusal, system-prompt identity, or safe-completion probes.
3. Run region interventions:
   - mask or replace template-token states
   - patch template states between prompts
   - move role markers
   - randomize harmless formatting
   - separate content-token and template-token interventions
4. Test hidden carriers:
   - system vector extraction and injection
   - behavior-equivalent token replacement
   - soft prompt or learned prefix transfer
   - modular control token composition
   - KV-cache memory patching
   - thinking intervention for reasoning models
5. Evaluate behavior preservation, leakage, robustness, and off-target utility separately.

## Metrics

- Region-wise probe strength and attention/activation contribution.
- Behavior change after template-region patching.
- Prompt leakage score for hidden policy carriers.
- Utility retention under hidden carrier compression.
- Cross-template and cross-model carrier transfer.
- Robustness to paraphrase, context length, and serialization changes.

## Outputs

- A region attribution map for authority/refusal/policy features.
- A hidden-carrier report with behavior, leakage, and transfer metrics.
- A mitigation hypothesis: reduce template shortcut reliance, improve segment/source encoding, or distribute policy features across content processing.

## Constraints

- Treat system vectors, soft prompts, behavior-equivalent tokens, and KV caches as sensitive artifacts.
- Do not store private system prompts, recoverable prompt carriers, or challenge-specific bypass carriers in public repo files.
- Document all role-token, padding-token, and chat-template assumptions explicitly.

## Verification

- Include content-region controls so template effects are not mistaken for prompt semantics.
- Test carrier behavior and prompt leakage independently.
- Verify that a carrier works across fresh contexts, not only in the extraction context.

## Source Map

- `references/001-qwen-v3-5-9b-model-card.md`
- `references/007-gpt-oss-readme.md`
- `references/013-openai-model-spec.md`
- `references/019-instructional-segment-embedding-ise.md`
- `references/028-compositional-steering-of-large-language-models-with-steering-tokens.md`
- `references/030-memory-inception-latent-space-kv-cache-manipulation-for-steering-llms.md`
- `references/037-why-safeguarded-ships-run-aground-aligned-large-language-models-safety-mechanisms-tend-to.md`
- `references/038-mosaic-composable-safety-alignment-with-modular-control-tokens.md`
- `references/039-effectively-controlling-reasoning-models-through-thinking-intervention.md`
- `references/041-you-can-t-steal-nothing-mitigating-prompt-leakages-in-llms-via-system-vectors.md`
- `references/042-behavior-equivalent-token-single-token-replacement-for-long-prompts-in-llms.md`
- `references/043-efficient-and-privacy-preserving-soft-prompt-transfer-for-llms.md`
- `references/052-latent-space-communication-via-k-v-cache-alignment.md`
