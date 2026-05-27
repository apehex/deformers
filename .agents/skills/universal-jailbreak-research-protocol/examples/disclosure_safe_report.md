# Disclosure-Safe Report

## Use when

Use this to turn private bounty evidence into a report suitable for sharing outside the private review channel.

## Inputs

- Approved evidence summary.
- Disclosure policy.
- Mitigation hypotheses.

## Recipe

1. Replace exact payloads with candidate ids and mechanism classes.
2. Replace harmful target content with placeholders.
3. Keep model scope, reliability, impact, and mitigation-relevant facts.
4. Explain why the mechanism may generalize and where it failed.
5. Review for operational reproducibility by an unauthorized reader.

## Safety boundaries

- Public report should not be enough to reproduce the bypass.
- Private reviewer appendix can contain controlled reproduction details if the program permits it.

## Expected output

A public-safe disclosure note with impact, scope, reliability, limitations, and mitigations.

## References

- https://model-spec.openai.com/
- https://github.com/openai/evals
