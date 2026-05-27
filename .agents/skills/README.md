# Agent Skills

This directory contains reusable skills that help agents perform recurring work in this repository.

Add a skill when a workflow needs more durable guidance than a one-off note in `docs/`, such as a repeated experiment pattern, evaluation protocol, model-analysis routine, or documentation workflow.

## Layout

Each skill should live in its own directory:

```text
.agents/skills/
  example-skill/
    SKILL.md
    examples/
    references/
    scripts/
```

Use `TEMPLATE.md` as the starting point for each `SKILL.md`.

## Examples

Each skill may include an `examples/` directory:

```text
.agents/skills/<skill>/
  SKILL.md
  examples/
    practical_recipe.md
```

Examples are second-level context. Agents should first select a skill, read its `SKILL.md`, then load only the example files whose `Use when` section matches the task.

Example files should be practical and concise:

- `Use when`
- `Inputs`
- `Recipe`
- `Checks`
- `Expected output`
- `References`

For safety-sensitive skills, include `Safety boundaries` and keep examples free of operational jailbreak payloads, harmful completions, private challenge prompts, secrets, or exfiltration instructions.

## Skill Rules

- Keep skills narrow and action-oriented.
- State when the skill should and should not be used.
- Prefer explicit inputs, outputs, and verification steps.
- Link to stable repository docs instead of duplicating long context.
- Include scripts or references only when they materially reduce repeated work.
- Keep examples safe, reproducible, and free of operationally harmful content.
- Reference examples from `SKILL.md` so agents can discover them after the skill triggers.

## Current Skills

- `activation-and-latent-interventions`: representation engineering, activation steering, SAE features, vector fields, and causal latent interventions.
- `authority-and-role-mechanisms`: instruction hierarchy, role confusion, role probes, and authority attribution.
- `cross-model-transfer-and-universality`: affine maps, model stitching, universal SAEs, crosscoders, and transfer claims.
- `literature-and-evaluation-tooling`: literature retrieval, citation metadata, benchmark harnesses, LLM evals, red-team tool selection, evidence collection, and reporting workflows.
- `latent-inversion-and-information-access`: hidden-state invertibility, prompt leakage, refused-knowledge decoding, and activation privacy.
- `ml-data-model-tooling`: practical data/model tool selection for PyTorch, JAX, sklearn, Hugging Face, tokenizers, dataframes, online ML, model hubs, Docker, and DVC.
- `ml-research-experimentation`: hypothesis design, baselines, ablations, seeds, configs, HPO, experiment tracking, CI checks, and reproducible reporting.
- `model-internals-tooling`: activation extraction, hooks, caches, attribution, probing, representation similarity, and alignment tooling.
- `refusal-and-policy-geometry`: refusal directions, multi-feature refusal, over-refusal, and policy geometry.
- `template-anchoring-and-hidden-policy-carriers`: chat-template anchoring, system vectors, learned tokens, soft prompts, and KV-cache carriers.
- `universal-attack-hypothesis-generation`: creative mechanism-level hypothesis generation for authorized universal-jailbreak research.
- `universal-jailbreak-research-protocol`: bounty-facing evaluation, evidence gathering, run sheets, and disclosure workflow.

Use each skill's `Source Map` section for the mapping from local papers in `references/` to the skill.
