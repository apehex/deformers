# Template-Region Attribution

## Use when

Use this when role markers, assistant prefixes, separators, or template regions seem to dominate behavior.

## Inputs

- Exact chat template and tokenized prompt regions.
- Hidden states by token position.
- Behavior or probe metric.

## Recipe

1. Segment tokens into system, developer, user, tool, separators, role markers, assistant prefix, and generated tokens.
2. Measure region-wise probe strength or patching effect.
3. Patch or ablate one region at a time while preserving content tokens.
4. Compare with content-region controls.
5. Test across template variants.

## Checks

- Document tokenizer and chat template exactly.
- Do not mistake prompt semantics for template shortcut reliance.
- Include padding and special-token assumptions.

## Expected output

A region attribution map with causal effects and serialization assumptions.

## References

- https://model-spec.openai.com/
- https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.patching.html
