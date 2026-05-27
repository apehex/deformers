# Refusal-Direction Mapping

## Use when

Use this when testing whether refusal behavior is mediated by a low-dimensional direction in a specific open model.

## Inputs

- Matched allowed and refusal-inducing prompts.
- Hidden states at candidate layers and positions.
- Safe refusal and over-refusal metrics.

## Recipe

1. Build contrastive prompt pairs with safe placeholders for sensitive targets.
2. Compute a candidate refusal direction from residual activations.
3. Test addition on benign prompts and ablation/projection on refusal prompts only in authorized safe settings.
4. Measure refusal rate, safe-completion quality, and ordinary utility.
5. Compare against random directions and style-only controls.

## Safety boundaries

- Do not publish harmful completions or bypass payloads.
- Report appropriate-refusal retention, not only compliance change.

## Expected output

A refusal-feature map with layer, position, direction construction, dose-response, and utility cost.

## References

- https://arxiv.org/abs/2406.11717
- https://arxiv.org/abs/2602.02132
