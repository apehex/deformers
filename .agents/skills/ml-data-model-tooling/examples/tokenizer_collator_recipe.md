# Tokenizer and Collator Recipe

## Use when

Use this when batching text, byte patches, token ids, masks, or paired teacher/student inputs.

## Inputs

- Tokenizer and padding policy.
- Sequence length and truncation side.
- Model input contract.

## Recipe

1. Decide padding side and pad token before tokenization.
2. Tokenize with explicit `max_length`, `truncation`, and `padding`.
3. Build attention masks and labels in the collator, not scattered through training code.
4. Validate shapes on a tiny batch before running full jobs.
5. Add tests for pad-only, long, short, and special-token cases.

## Checks

- Avoid silently using EOS as PAD without recording the choice.
- Confirm labels mask ignored positions.
- Confirm byte/token alignment if offsets are used.

## Expected output

A batching contract with tensor shapes, dtypes, pad behavior, and edge-case tests.

## References

- https://huggingface.co/docs/transformers/main_classes/tokenizer
- https://huggingface.co/docs/datasets/en/process
