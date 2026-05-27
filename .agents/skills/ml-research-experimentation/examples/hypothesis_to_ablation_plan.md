# Hypothesis-to-Ablation Plan

## Use when

Use this to turn a vague mechanism idea into a falsifiable experiment.

## Inputs

- Research question.
- Candidate mechanism.
- Minimal dataset and model access.

## Recipe

1. State the hypothesis and the observation that would falsify it.
2. Define baseline, intervention, and negative control.
3. Choose primary metric before running experiments.
4. Add ablations for dataset, layer, position, seed, and prompt template only when they test a specific confound.
5. Pre-write interpretation criteria for success, null, and mixed results.

## Checks

- Do not add ablations that cannot change the conclusion.
- Include at least one negative control.
- Report expected failure modes before seeing results.

## Expected output

A compact experiment plan that another agent can execute without making design decisions.

## References

- https://docs.wandb.ai/models/sweeps
- https://dvc.org/doc/start/data-pipelines/data-pipelines
