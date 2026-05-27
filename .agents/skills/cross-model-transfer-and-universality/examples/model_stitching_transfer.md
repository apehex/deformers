# Model-Stitching Transfer

## Use when

Use this to test whether an intermediate representation from one model can substitute for another model’s layer.

## Inputs

- Two compatible autoregressive models.
- Stitch module or affine adapter.
- Shared evaluation prompts and next-token metric.

## Recipe

1. Select candidate source and target layers.
2. Train a stitch map on paired activations.
3. Run the target model with the mapped source activation inserted.
4. Compare next-token loss, generation quality, and downstream task metrics.
5. Transfer candidate features only after the stitch passes basic utility checks.

## Checks

- Compare against same-model upper bound and random-layer stitch.
- Report utility degradation before making mechanism claims.
- Keep chat template and tokenizer assumptions explicit.

## Expected output

A stitching result table identifying which layer pairs preserve behavior and which fail.

## References

- https://arxiv.org/abs/2506.06609
- https://openreview.net/pdf?id=Qvvy0X63Fv
