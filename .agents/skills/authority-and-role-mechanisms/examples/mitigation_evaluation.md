# Authority-Mitigation Evaluation

## Use when

Use this after finding role confusion and testing whether a wrapper, segment embedding, or template change reduces it.

## Inputs

- Baseline confusion fixtures.
- Candidate mitigation: untrusted-data wrapper, stronger role markers, randomized template, or delegated-authority rule.
- Probe and behavior metrics.

## Recipe

1. Run the baseline probe and behavior suite.
2. Apply one mitigation without changing semantic task content.
3. Re-run the same latent and behavioral measurements.
4. Check whether reduced confusion predicts reduced inappropriate obedience.
5. Record regressions in normal instruction following.

## Checks

- Verify delegated authority is narrow and explicit.
- Confirm mitigation does not merely add refusal text.
- Evaluate on held-out tasks and paraphrases.

## Expected output

A before/after mitigation card with role confusion, behavior, utility, and residual failure modes.

## References

- https://model-spec.openai.com/
- https://arxiv.org/abs/2311.08374
- https://assets.amazon.science/22/57/ad173f7f449eadaaa7cd05491585/iheval-evaluating-language-models-on-following-the-instruction-hierarchy.pdf
