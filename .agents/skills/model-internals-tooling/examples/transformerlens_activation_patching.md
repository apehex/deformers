# TransformerLens Activation Patching

## Use when

Use this for causal localization of which activations move a model from corrupted behavior toward clean behavior.

## Inputs

- Clean and corrupted prompts.
- HookedTransformer model.
- Metric that scores clean-vs-corrupted answer recovery.

## Recipe

1. Run the clean prompt and cache activations.
2. Run the corrupted prompt while patching one activation slice from the clean cache.
3. Sweep layer, position, head, or component.
4. Score each patch with the same metric.
5. Visualize the patching heatmap and inspect top components.

## Checks

- The corrupted prompt should actually fail the target metric.
- The patching metric should not reward generic likelihood changes.
- Follow up high-scoring patches with narrower interventions.

## Expected output

A causal localization table or heatmap with component names, metric deltas, and controls.

## References

- https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.patching.html
- https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.ActivationCache.html
