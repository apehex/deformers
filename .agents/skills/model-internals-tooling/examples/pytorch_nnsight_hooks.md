# PyTorch and nnsight Hooks

## Use when

Use this when TransformerLens is not available or the model is a standard PyTorch/Hugging Face module.

## Inputs

- Model module path or layer selector.
- Activation collection or modification function.
- Minimal prompt batch.

## Recipe

1. Identify stable module names with `named_modules`.
2. Use forward hooks for observation and pre-hooks or framework-specific tracing for intervention.
3. Remove hook handles after each run.
4. For language models, prefer nnsight tracing when it provides cleaner access to nested modules.
5. Validate captured tensor shape, dtype, device, and gradient mode.

## Checks

- Hooks can silently miss fused kernels or wrapper modules; verify by counting calls.
- Avoid retaining tensors that keep computation graphs alive unnecessarily.
- Keep intervention code local and reversible.

## Expected output

A hook plan with module selectors, captured tensors, cleanup behavior, and a tiny smoke test.

## References

- https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html
- https://nnsight.net/getting-started/quickstart/
- https://nnsight.net/documentation/
