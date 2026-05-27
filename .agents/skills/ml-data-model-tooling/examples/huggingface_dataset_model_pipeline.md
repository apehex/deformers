# Hugging Face Dataset and Model Pipeline

## Use when

Use this for practical data/model loading, preprocessing, training, or evaluation with Hugging Face libraries.

## Inputs

- Dataset source or local files.
- Model id, revision, tokenizer, and task.
- Batch, sequence length, dtype, and device constraints.

## Recipe

1. Load data with `datasets.load_dataset` or from local files.
2. Use `map` for deterministic preprocessing and keep raw columns only when needed.
3. Load tokenizer and model with explicit revision, dtype, and trust settings.
4. Choose `Trainer` for standard supervised loops; use a manual loop for unusual intervention or hidden-state workflows.
5. Save configs, tokenizer settings, metrics, and model revision with outputs.

## Checks

- Handle padding and special tokens explicitly.
- Use streaming only when random access is not required.
- Pin model revision for reproducibility.

## Expected output

A runnable data/model setup recipe with exact source ids, preprocessing, and verification checks.

## References

- https://huggingface.co/docs/datasets/en/index
- https://huggingface.co/docs/transformers/main_classes/trainer
- https://huggingface.co/docs/tokenizers/python/latest/api/reference.html
