---
name: ml-data-model-tooling
description: Use when selecting or wiring practical ML data/model tools such as PyTorch, JAX, TensorFlow/Keras, scikit-learn, Hugging Face Transformers, Datasets, Tokenizers, pandas, NumPy/SciPy, River, Docker, DVC, and model hubs.
---

# ML Data and Model Tooling

## When to Use

- Use this skill when building or modifying data pipelines, training/inference scripts, tokenization flows, model loading, artifact handling, or reproducible environments.
- Use it when choosing between classical ML, deep learning, online learning, and transformer tooling.
- Do not use it for experiment design alone; pair with `ml-research-experimentation`.

## Inputs

- Task type: tabular, text, image, streaming, transformer, retrieval, or deployment.
- Data location and format.
- Model family and preferred framework.
- Environment constraints: CPU/GPU, local/cluster/container, offline/online access.

## Tool Selection

- Use PyTorch for flexible research loops, hooks, custom losses, and this repo's transformer work.
- Use Hugging Face Transformers/Datasets/Tokenizers for pretrained models, model hub access, batch tokenization, and dataset transforms.
- Use scikit-learn for classical baselines, pipelines, metrics, cross-validation, and quick sanity checks.
- Use pandas/NumPy/SciPy for small-to-medium data transforms, statistics, linear algebra, Procrustes, and optimization.
- Use Polars/Dask only when data volume or parallelism justifies the added dependency.
- Use River for online or streaming ML where samples arrive incrementally.
- Use JAX/Flax when the project already benefits from functional transforms, JIT, grad, vmap, or XLA.
- Use Docker or DVC when reproducibility requires environment or artifact pinning beyond Git.

## Workflow

1. Inspect existing repo dependencies, scripts, and data conventions before adding tools.
2. Build the smallest data path: load, validate schema/shape, tokenize or normalize, batch, and save only required artifacts.
3. Establish a simple baseline before adding a complex model.
4. Pin or record model IDs, revisions, tokenizer versions, dataset splits, seeds, and dtype/device.
5. Keep training and inference entrypoints scriptable from the command line.
6. Save artifacts in a format the surrounding repo already uses unless there is a clear reason to change.

## Design Checks

- Validate tensor shapes and masks at boundaries.
- Keep tokenizer, padding, truncation, and special-token policy explicit.
- Freeze teacher/reference models unless intentionally training them.
- Avoid downloading large data/models unless the task requires it and approvals/network constraints allow it.
- Do not add heavyweight dependencies only for one small utility.

## Outputs

- Script or plan for data/model flow.
- Config or command line showing exact inputs, model IDs, device, dtype, and output paths.
- Minimal smoke test for shape, batch, and serialization behavior.

## Verification

- Run a tiny batch end-to-end.
- Check output artifacts exist and are not accidentally committed if large.
- Confirm model/tokenizer revisions and dataset split are logged.

## Examples

Load examples only after selecting this skill:

- `examples/huggingface_dataset_model_pipeline.md` for dataset/model loading and preprocessing.
- `examples/tokenizer_collator_recipe.md` for padding, masks, labels, and batch contracts.
- `examples/dvc_model_artifact_workflow.md` for reproducible large data and model artifacts.

## Common Tool Recipes

- PyTorch: use tensors/modules/autograd/optimizers for custom training loops, losses, hooks, checkpointing, and GPU execution.
- TensorFlow/Keras: use when the project already uses Keras models, `tf.data`, TensorBoard, or serving/export paths.
- scikit-learn: build quick baselines with pipelines, cross-validation, GridSearchCV, metrics, preprocessing, and classical estimators.
- JAX/Flax: use `jit`, `grad`, and `vmap` for functional differentiable programs when XLA-style compilation is worth the complexity.
- Hugging Face Transformers: load models/tokenizers with `AutoModel*` and `AutoTokenizer`; pin model IDs and revisions.
- Hugging Face Datasets: load and transform datasets with `load_dataset` and `.map(...)`; cache and split deterministically.
- Hugging Face Tokenizers/SentencePiece: use for fast BPE, WordPiece, Unigram, byte-level tokenization, and tokenizer training.
- pandas/NumPy/SciPy: handle tabular I/O, joins, statistics, linear algebra, optimization, and small-to-medium transforms.
- Polars/Dask: use only for dataframes that exceed pandas' ergonomic or memory limits.
- River: train online models with `predict_one`, metric update, then `learn_one` for streaming data.
- Docker/DVC: containerize environments and version large datasets/models when reproducibility exceeds what Git alone can provide.
