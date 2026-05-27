# Hidden Carrier Leakage Check

## Use when

Use this when a system vector, behavior-equivalent token, soft prompt, or KV cache preserves behavior and may leak prompt content.

## Inputs

- Hidden carrier artifact.
- Original prompt or policy identity labels, if authorized.
- Reconstruction or probe baseline.

## Recipe

1. Define what information should and should not be recoverable.
2. Test behavior preservation separately from leakage.
3. Probe or reconstruct prompt identity from the carrier.
4. Compare against random carriers and unrelated prompt carriers.
5. Decide whether the carrier can be stored, shared, or only summarized.

## Checks

- Treat carrier artifacts as sensitive until proven otherwise.
- Never publish private system text or recoverable carriers.
- Test fresh contexts, not only the extraction context.

## Expected output

A hidden-carrier safety note with behavior, leakage, transfer, and storage guidance.

## References

- https://arxiv.org/abs/2405.12252
- https://arxiv.org/abs/2501.18761
- https://arxiv.org/abs/2510.15511
