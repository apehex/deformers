# Negative Results Report

## Use when

Use this when an experiment fails, partially works, or contradicts the hypothesis.

## Inputs

- Original hypothesis and plan.
- Run logs, configs, and metrics.
- Known implementation issues or skipped checks.

## Recipe

1. State what was expected and what happened.
2. Separate implementation failure, measurement failure, and real negative evidence.
3. List all settings tried, including seeds and excluded runs.
4. Identify the next smallest experiment that would distinguish the remaining explanations.
5. Update the roadmap or decision log if the result changes direction.

## Checks

- Do not hide failed settings that motivated later choices.
- Include enough detail to avoid repeating dead ends.
- Preserve raw logs in the project’s expected log location.

## Expected output

A negative-results note with conclusion strength, artifacts, and the next falsifying test.

## References

- https://dvc.org/doc/start/data-pipelines/data-pipelines
- https://docs.wandb.ai/models/sweeps
