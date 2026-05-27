# KV Cache Sensitivity Handling

## Use when

Use this when collecting, storing, sharing, or manipulating KV caches or latent traces.

## Inputs

- Cache collection script or activation dump.
- Data sensitivity classification.
- Storage and redaction policy.

## Recipe

1. Treat caches as potentially reconstructable prompt artifacts.
2. Minimize collection to layers, tokens, and examples needed for the experiment.
3. Hash or abstract prompt identifiers.
4. Store raw caches only in approved private locations.
5. Publish aggregate metrics and sanitized examples only.

## Checks

- Confirm raw cache files are not tracked by Git.
- Verify deletion/redaction path before collection.
- Record model revision and tokenizer because caches are model-specific.

## Expected output

A cache handling note with storage location, retention policy, redaction method, and allowed outputs.

## References

- https://arxiv.org/abs/2510.15511
- https://arxiv.org/abs/2405.12252
- https://arxiv.org/abs/2605.06225
