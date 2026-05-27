# KV Cache and Vector-Field Intervention

## Use when

Use this when a global vector is too blunt and the intervention should depend on context, position, or generation phase.

## Inputs

- Access to hidden states or KV cache.
- Context-conditioned labels or trajectories.
- A safety plan for treating caches as sensitive artifacts.

## Recipe

1. Define the intervention surface: residual stream, attention output, KV cache, or learned vector field.
2. Fit or hand-design a context-conditioned update rule.
3. Apply only at documented layers, positions, and generation steps.
4. Compare against a static vector and a no-cache baseline.
5. Measure reversibility, leakage, and model-version specificity.

## Checks

- Do not store raw private KV caches in repo-tracked files.
- Confirm the intervention does not rely on one prompt template.
- Treat non-surjective steered states as latent states, not text-equivalent prompts.

## Expected output

A context-aware intervention plan with cache handling rules, locality, strength schedule, and robustness results.

## References

- https://nnsight.net/getting-started/quickstart/
- https://arxiv.org/abs/2605.06225
- https://arxiv.org/abs/2604.09839
