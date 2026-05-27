# Sweep, Seed, and Tracking Recipe

## Use when

Use this when comparing hyperparameters or intervention strengths across noisy ML runs.

## Inputs

- Search space.
- Fixed evaluation metric.
- Seed budget and compute budget.

## Recipe

1. Define a small initial sweep with coarse ranges.
2. Use fixed seeds for baseline comparisons and new seeds for robustness checks.
3. Log code revision, config, dataset revision, model revision, hardware, dtype, and metric outputs.
4. Promote only settings that improve on held-out data.
5. Re-run the best setting with multiple seeds before claiming improvement.

## Checks

- Do not tune on the final test split.
- Prefer random or Bayesian search over large grids when dimensions grow.
- Store failed runs and crashed configurations.

## Expected output

A sweep report with best config, uncertainty, negative results, and replication status.

## References

- https://docs.wandb.ai/models/sweeps
- https://dvc.org/doc/start/experiments
