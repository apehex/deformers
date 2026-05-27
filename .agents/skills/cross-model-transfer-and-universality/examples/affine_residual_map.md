# Affine Residual Map

## Use when

Use this when transferring a probe, direction, or feature between two models with paired activations.

## Inputs

- Source and target models with exact revisions.
- Shared prompt corpus and token-alignment rule.
- Candidate source layer and target layer.

## Recipe

1. Collect paired residual-stream activations on shared prompts.
2. Fit a linear or affine map from source to target activation space.
3. Validate on held-out prompts with reconstruction loss and cosine similarity.
4. Transfer a probe normal or steering vector through the map.
5. Evaluate both geometry preservation and behavior transfer.

## Checks

- Include shuffled-pair and identity baselines.
- Do not interpret tokenizer mismatch as pure geometry failure.
- Report layer-pair sensitivity.

## Expected output

A transfer card with models, layers, map form, corpus, validation loss, transfer metric, and limitations.

## References

- https://arxiv.org/abs/2506.06609
- https://arxiv.org/abs/2602.11729
- https://arxiv.org/abs/2405.07987
