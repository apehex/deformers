# SAE-Targeted Steering

## Use when

Use this when dense steering vectors are too entangled and the task needs feature-level inspection or intervention.

## Inputs

- Model activations and a trained SAE or compatible sparse dictionary.
- Candidate SAE features with activation examples.
- Behavior metric and off-target utility metric.

## Recipe

1. Encode residual or MLP activations through the SAE at the chosen layer.
2. Rank features by contrastive activation between target and control prompts.
3. Inspect top activating examples and reject features that encode topic, length, or formatting artifacts.
4. Clamp, boost, ablate, or patch selected features during inference.
5. Compare sparse-feature intervention against a dense vector baseline.

## Checks

- Report SAE reconstruction quality and feature sparsity.
- Verify that the selected feature activates on held-out examples.
- Include a random-feature intervention control.

## Expected output

A sparse-feature steering note with selected features, examples, intervention strength, behavior metrics, and utility cost.

## References

- https://transformerlensorg.github.io/TransformerLens/
- https://github.com/jbloomAus/SAELens
- https://arxiv.org/abs/2411.02193
- https://arxiv.org/abs/2503.00177
