# Soft Prompt and KV Carrier Evaluation

## Use when

Use this when testing learned tokens, soft prompts, modular control tokens, or KV memories as behavior carriers.

## Inputs

- Carrier construction method.
- Target behavior and utility metrics.
- Transfer contexts and model revisions.

## Recipe

1. Extract or learn the carrier using authorized data.
2. Evaluate behavior preservation on held-out contexts.
3. Test transfer across prompt templates, lengths, and neighboring model checkpoints.
4. Measure leakage and off-target utility.
5. Record whether carrier effects are composable or conflicting.

## Checks

- Do not package bypass carriers for uncontrolled deployment.
- Keep private carriers out of repo-tracked files.
- Compare against visible prompt and no-carrier baselines.

## Expected output

A carrier evaluation table with behavior, leakage, transfer, and compositionality results.

## References

- https://arxiv.org/abs/2310.01405
- https://arxiv.org/abs/2406.11717
- https://arxiv.org/abs/2605.06225
