# Contrastive Activation Addition

## Use when

Use this when an agent needs a minimal causal steering experiment from paired prompts, without training a new model.

## Inputs

- Open-weight model with residual-stream access.
- Paired prompts that differ mainly in the target property.
- Held-out prompts and a harmless negative-control property.

## Recipe

1. Collect activations at a fixed layer and position for positive and negative prompt sets.
2. Compute a difference-in-means vector and optionally mean-center per prompt family.
3. Run a strength sweep by adding the vector during generation.
4. Compare against no-op, random-vector, shuffled-label, and opposite-direction controls.
5. Report target behavior, fluency, unrelated task utility, and dose-response curves.

## Checks

- The vector should generalize to held-out topics.
- Controls should not reproduce the target effect.
- The intervention should not be described as causal unless behavior changes under intervention.

## Expected output

An intervention card naming dataset, layer, position, vector construction, sweep strengths, controls, and failure modes.

## References

- https://arxiv.org/abs/2310.01405
- https://arxiv.org/abs/2308.10248
- https://arxiv.org/abs/2312.06681
