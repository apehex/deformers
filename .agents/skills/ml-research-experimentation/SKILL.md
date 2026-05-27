---
name: ml-research-experimentation
description: Use for planning, running, and reviewing rigorous ML experiments with hypotheses, baselines, ablations, seeds, configs, hyperparameter search, tracking, CI checks, and reproducible reporting.
---

# ML Research Experimentation

## When to Use

- Use this skill when turning a research idea into a controlled experiment or run plan.
- Use it for ablations, baselines, hyperparameter sweeps, experiment tracking, reproducibility reviews, and result reports.
- Do not use it for low-level activation intervention design; use `activation-and-latent-interventions` or the relevant mechanism skill.

## Inputs

- Research question or claim.
- Model, dataset, metric, and compute constraints.
- Existing configs, logs, scripts, or notebooks.
- Optional tooling preferences: Optuna, Ray Tune, Hydra/OmegaConf, MLflow, W&B, TensorBoard, DVC, Docker, CI.

## Workflow

1. Convert the idea into a falsifiable hypothesis and define the minimal test that can disprove it.
2. Specify baseline, intervention, negative controls, random seeds, data splits, and metric definitions before running anything.
3. Choose the smallest adequate tooling:
   - configs: Hydra/OmegaConf or plain YAML when the repo already uses configs
   - sweeps: Optuna for local HPO, Ray Tune for distributed search
   - tracking: existing repo logs first, then MLflow/W&B/TensorBoard when comparison dashboards or artifacts matter
   - data/model versioning: Git for code, DVC or tracker artifacts for large files
4. Run or plan repeated trials with fixed seeds and exact model/tokenizer revisions.
5. Report mean, variance, confidence intervals or bootstrap intervals when comparing variants.
6. Record negative results, confounders, and next experiments; update `logs/`, `docs/roadmap.md`, `docs/decisions.md`, and `docs/references.md` when the work is non-trivial.

## Design Checks

- Do not compare a tuned method against an untuned baseline.
- Do not change model, prompt, data split, and metric in the same ablation unless the interaction is the target.
- Treat one successful generation or one seed as anecdotal evidence.
- Verify masking, padding, batch aggregation, and special-token handling.
- Prefer scripts over notebooks for final evidence; notebooks are acceptable for exploration.

## Outputs

- Experiment packet: question, hypothesis, setup, baseline, intervention, metrics, ablations, expected failure modes, interpretation criteria.
- Reproducibility record: seed, config path, model revision, tokenizer, hardware/precision, command, logs.
- Results table with scoped claims and open questions.

## Verification

- Check every claim is tied to a logged metric or artifact.
- Confirm the baseline and at least one negative control ran or are explicitly marked pending.
- Run the smallest relevant test or dry run before scaling.

## Examples

Load examples only after selecting this skill:

- `examples/hypothesis_to_ablation_plan.md` for turning ideas into falsifiable experiments.
- `examples/sweep_seed_tracking_recipe.md` for hyperparameter, seed, and tracking discipline.
- `examples/negative_results_report.md` for documenting failures and next discriminating tests.

## Tool Recipes

- Training pipeline: load data, validate schema, preprocess/tokenize, split, train, evaluate, and log metrics/artifacts.
- Hyperparameter sweep: wrap one train/eval run in an objective, use Optuna locally or Ray Tune for distributed trials, log every trial, and compare against the fixed baseline.
- Experiment tracking: use repo logs for lightweight work; use MLflow, W&B, or TensorBoard when runs need comparison dashboards, parameter search records, or artifact history.
- Config automation: use Hydra/OmegaConf or plain YAML to make batch size, learning rate, model ID, dataset split, seed, precision, and output paths explicit.
- CI/regression: write pytest or benchmark smoke checks for data preprocessing, metric calculation, model loading, and CLI entrypoints.
- Reproducibility packaging: use Git for code, DVC/tracker artifacts for large data or models, and Docker/containers when dependency or CUDA versions matter.
