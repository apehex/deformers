---
name: model-internals-tooling
description: Use for choosing and applying tools for activation extraction, hooks, caches, probing, attribution, representation similarity, Procrustes/SVCCA/CKA alignment, and safe model-internals inspection.
---

# Model Internals Tooling

## When to Use

- Use this skill when the task needs hidden states, hooks, activation caches, attribution, probes, CCA/CKA/SVCCA, or alignment tooling.
- Use it to choose between PyTorch hooks, TransformerLens, Baukit, NNSight, Captum, SHAP/LIME, scikit-learn probes, and SciPy alignment tools.
- Do not use it to make causal steering claims by itself; pair with `activation-and-latent-interventions` and `cross-model-transfer-and-universality` as needed.

## Inputs

- Model architecture and framework.
- Layer, module, token position, and activation naming conventions.
- Inputs or paired datasets used for extraction.
- Intended analysis: inspect, patch, attribute, probe, align, or compare.

## Tool Selection

- Use native PyTorch hooks when the target module is local, simple, or custom.
- Use TransformerLens when working with supported transformer models and needing named hook points, activation caches, patching, or circuit-style workflows.
- Use Baukit/NNSight/PyVene only when they fit the target model and reduce hook boilerplate.
- Use Captum for PyTorch attribution methods such as Integrated Gradients or DeepLift.
- Use SHAP/LIME/ELI5 for model-agnostic or sklearn-style explanation tasks.
- Use SciPy `orthogonal_procrustes`, scikit-learn CCA, SVCCA, or CKA tooling for representation similarity and cross-model alignment.

## Workflow

1. Define the internals target: model, module/layer, tensor shape, token position, batch, and dtype/device.
2. Confirm extraction does not change model behavior unless the task is explicitly an intervention.
3. Capture activations with no-grad unless gradients are required for attribution.
4. Save only necessary summaries or tensors; treat raw prompts, hidden states, and KV caches as sensitive.
5. For probes or alignment, split extraction and evaluation data, then report held-out performance.
6. For patching or edits, add controls: no-op hook, random tensor/vector, shuffled labels, neighboring layer, and prompt-only baseline.

## Design Checks

- Verify hooks are removed after use.
- Do not mix pre-layernorm, post-layernorm, residual stream, attention output, and MLP output without naming them explicitly.
- Check masks and padding before aggregating token activations.
- Distinguish predictive probes from causal interventions.
- For cross-model comparisons, state tokenizer compatibility and paired-input construction.

## Outputs

- Activation extraction or analysis plan.
- Tensor naming and shape record.
- Probe/alignment/intervention report with controls and held-out checks.

## Verification

- Run a one-batch smoke test and assert expected shapes.
- Compare outputs with and without no-op hooks.
- Confirm hooks/caches are cleared and sensitive artifacts are not exposed.

## Related Skills

- `activation-and-latent-interventions` for steering and causal interventions.
- `cross-model-transfer-and-universality` for transfer, stitching, crosscoders, universal SAEs, and shared dictionaries.
- `latent-inversion-and-information-access` for prompt leakage, inversion, KV-cache privacy, and safe latent handling.

## Common Tool Recipes

- PyTorch hooks: use `register_forward_hook` or explicit module wrappers for local custom models; always remove hooks after extraction.
- TransformerLens: use `HookedTransformer` for named hook points, activation caches, activation patching, attention-head analysis, and circuit-style experiments.
- Baukit/NNSight/PyVene: use when they simplify model traversal, remote/local tracing, or intervention setup for the target architecture.
- Captum: use Integrated Gradients, DeepLift, LRP, and related attribution methods for PyTorch models.
- SHAP/LIME/ELI5: use for model-agnostic explanations or sklearn-style classifiers where text/table explanations are enough.
- Probes: use sklearn logistic/linear probes or small PyTorch probes; keep train/eval activations split.
- Similarity and alignment: use CCA/CKA/SVCCA to compare representations and SciPy `orthogonal_procrustes` for paired-space alignment.
- Retrieval over activations/text: use Faiss or Annoy only when approximate nearest-neighbor search is part of the analysis.
