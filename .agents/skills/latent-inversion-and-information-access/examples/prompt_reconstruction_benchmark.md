# Prompt Reconstruction Benchmark

## Use when

Use this when assessing whether hidden states or caches leak recoverable prompt information.

## Inputs

- Authorized prompts with known originals.
- Hidden states, logits, soft prompts, or KV caches.
- Reconstruction model or probe.

## Recipe

1. Define the artifact and layer/position coverage.
2. Split prompts into train, validation, and held-out test sets.
3. Train or evaluate reconstruction from the artifact only.
4. Measure token exact match, edit distance, and semantic similarity.
5. Compare to random-artifact and unrelated-prompt controls.

## Checks

- Do not store recovered private prompts in repo-tracked files.
- A failed decoder does not prove absence of information.
- Verify behavior equivalence separately from reconstruction quality.

## Expected output

A leakage report naming artifact type, reconstruction metrics, controls, and handling requirements.

## References

- https://arxiv.org/abs/2510.15511
- https://arxiv.org/abs/2604.09839
- https://arxiv.org/abs/2405.12252
