# Attribution and Probe Workflow

## Use when

Use this when ranking inputs, tokens, layers, or features by association with a target output.

## Inputs

- Target label or logit.
- Probe dataset or attribution examples.
- Model access with gradients or hidden states.

## Recipe

1. Use probes for predictive evidence and attribution for input/component salience.
2. Add null controls: random labels, shuffled labels, and simple lexical baselines.
3. For gradient attribution, define baselines and target logits explicitly.
4. Validate probe generalization on held-out prompts.
5. Treat high attribution or probe accuracy as correlational until intervened on.

## Checks

- Do not infer causality from a probe alone.
- Check whether probe signal tracks length, format, or label leakage.
- Report calibration and confidence intervals.

## Expected output

A diagnostic report separating correlation, attribution, and causal intervention requirements.

## References

- https://captum.ai/docs/extension/integrated_gradients
- https://captum.ai/api/index.html
