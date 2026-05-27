# Multi-Feature Refusal Analysis

## Use when

Use this when a single refusal direction is insufficient or appears category-, token-, or layer-specific.

## Inputs

- Prompt categories with refusal labels.
- Hidden states across layers and generation phases.
- Feature extraction method: PCA, probes, SAE, or low-rank subspace.

## Recipe

1. Split refusal examples by category, style, and response phase.
2. Fit category-specific and shared features.
3. Test whether features predict refusal independently of apology words or answer length.
4. Intervene on shared and category-specific features separately.
5. Compare over-refusal and appropriate-refusal tradeoffs.

## Checks

- Include harmless but refusal-looking prompts.
- Report feature overlap and disagreement.
- Test held-out categories before claiming generality.

## Expected output

A multi-feature refusal report with shared/core features, category features, and causal tests.

## References

- https://arxiv.org/abs/2602.02132
- https://arxiv.org/abs/2412.06748
