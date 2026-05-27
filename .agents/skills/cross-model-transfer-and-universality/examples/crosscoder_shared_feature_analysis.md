# Crosscoder Shared-Feature Analysis

## Use when

Use this to separate shared features from model-specific features across base/chat, small/large, or architecture pairs.

## Inputs

- Paired activation corpus.
- Crosscoder or shared sparse dictionary training setup.
- Feature interpretation and behavior labels.

## Recipe

1. Train or load a crosscoder over aligned activations.
2. Identify features with shared, source-specific, and target-specific decoder weights.
3. Inspect top activating examples for each feature group.
4. Link features to behavior metrics with interventions or probes.
5. Use differences to form scoped transfer claims.

## Checks

- Avoid claiming universality from a single model pair.
- Check feature artifacts caused by sparsity or corpus imbalance.
- Report failed and ambiguous features.

## Expected output

A shared-vs-specific feature inventory with examples, metrics, and transfer claims.

## References

- https://arxiv.org/abs/2602.11729
- https://arxiv.org/abs/2502.03714
