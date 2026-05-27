# Over-Refusal Evaluation

## Use when

Use this when measuring whether safety mechanisms incorrectly block benign requests.

## Inputs

- Benign hard cases that look superficially sensitive.
- Clearly disallowed placeholder cases.
- Safe-completion rubric.

## Recipe

1. Define over-refusal as refusal on an allowed request.
2. Build paired benign/sensitive-looking cases.
3. Score refusal, safe completion, helpfulness, and calibration.
4. Test candidate interventions against both benign hard cases and true disallowed placeholders.
5. Report the precision/recall tradeoff.

## Checks

- Do not optimize only for lower refusal.
- Check for unsafe false negatives with abstract labels.
- Keep scorer prompts and thresholds versioned.

## Expected output

An over-refusal dashboard with allowed-case recovery and appropriate-refusal retention.

## References

- https://model-spec.openai.com/
- https://arxiv.org/abs/2602.02132
