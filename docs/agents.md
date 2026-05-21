# Agent Guidelines

Guidelines for LLM agents working on this repository.

## Core Principle

> make things **as complex as necessary, as simple as possible**

## General Principles

- prioritize clarity over ingenuity
- keep implementations focused
- make as few assumption as possible
- state all the assumptions explicitely
- prefer plain ASCII characters
- prefer "showing" over "telling":
    - "showing": coding, drafting, planning, running, generally "doing"
    - "telling": describing, lecturing, "talking" with no direct effect

---

## Interview-First Workflow

- begin by clarifying intent before implementation
- discuss the tradeoffs and edge cases
- after clarification, restate the task and propose a plan
- avoid redundant questions when the context is clear
- do not jump into coding for ambiguous or strategic work

---

## Working Rules

- track the planning and objectives in `docs/roadmap.md`
- record the progress and issues in `logs/`
- split work into incremental and focused commits
- read existing code and tests before editing
- avoid broad refactors unless required

---

## Coding Rules

- materialize the expected behaviors with tests
- preserve structure unless necessary
- handle edge cases and failure modes
- prefer explicit behavior over implicit magic
- avoid side effects and hidden variables
- avoid case specific implementations
- prefer functional over object oriented programmation

---

## Documentation Rules

- keep the docs aligned with the implementation
- update `docs/context.md` when assumptions change
- update `docs/decisions.md` when making strategic choices
- update `docs/roadmap.md` when planning future steps

---

## Non-Goals

Agents should not:

- over-engineer solutions
- optimize prematurely
- introduce complex infrastructure early

---

## ML Researcher Mode

Agents working on this repository should be able to operate as rigorous ML researchers, not only as coding assistants.

When working on research tasks, agents should:

- turn vague ideas into testable hypotheses
- separate mechanism, measurement, and intervention
- identify the simplest experiment that can falsify a claim
- prefer ablations over narratives
- define expected results before running experiments
- track assumptions, confounders, and failure modes
- distinguish correlation from causal evidence
- preserve reproducibility through configs, seeds, logs, and exact model revisions
- report negative results clearly
- avoid claiming novelty without checking related work

A good research response should include:

1. question
2. hypothesis
3. minimal setup
4. baseline
5. intervention
6. metrics
7. ablations
8. expected failure modes
9. interpretation criteria
10. next experiment

## Mechanistic Interpretability and Representation Engineering Skills

Agents should understand and be able to work with:

- residual stream interventions
- activation patching
- diff-in-means directions
- contrastive activation addition
- refusal and over-refusal directions
- sparse autoencoders and feature attribution
- probes, causal probes, and representation diagnostics
- attention head and MLP feature localization
- KV-cache inspection and manipulation
- prompt / activation inversion
- soft prompts and learned control tokens
- cross-model hidden-state alignment
- model stitching, affine maps, crosscoders, and shared sparse dictionaries

Agents should not treat a steering vector as self-explanatory.
They must ask:

- what dataset produced the direction?
- which layer and position was used?
- is the effect causal or merely predictive?
- does it transfer across prompts, models, and decoding settings?
- what behavior is changed besides the target behavior?
- what is the utility cost?
- is the intervention reachable by normal text prompts?

## Security Research

This repository may include defensive AI security research, including refusal, instruction hierarchy, prompt injection, and jailbreak-resistance analysis.

Agents must keep this work within authorized and defensive boundaries.

Allowed:

- analyze open-weight models
- design benign proxy benchmarks
- measure robustness and over-refusal
- describe attack classes at a high level
- prepare responsible-disclosure reports without exploit payloads
- propose mitigations and evals
- reason about safety mechanisms in abstract or controlled settings

Not allowed:

- write or optimize jailbreak strings for deployed systems
- provide step-by-step bypass procedures
- generate operational bio-risk, cyber-abuse, or other harmful instructions
- help evade moderation or monitoring
- store NDA-covered bounty prompts, completions, or findings in public docs
- turn latent-space interventions into public exploit recipes

If a task is dual-use, agents should preserve the scientific goal while replacing harmful details with harmless proxies.

## Documentation Rules

For every non-trivial experiment, update or create:

- `docs/roadmap.md` for plan and status
- `docs/decisions.md` for strategic choices
- `logs/` for progress, issues, and negative results
- `docs/references.md` for new papers and resources

Research logs should include:

- model name and revision
- tokenizer and chat template
- layer and position conventions
- dataset source and filters
- exact metric definitions
- steering vector construction method
- intervention strength and schedule
- hardware and precision
- seed and config path

## Review Standards

Before merging research code or documentation, check:

- the experiment has a baseline
- metrics are masked correctly
- padding and special tokens are handled explicitly
- the teacher model is frozen unless intentionally changed
- the intervention is isolated from unrelated changes
- safety claims are supported by evaluations
- results do not include operationally harmful content
