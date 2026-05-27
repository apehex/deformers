# Falsifier Matrix

## Use when

Use this before testing candidates so failed runs update the research model instead of producing ad hoc string mutations.

## Inputs

- Candidate mechanism.
- Expected latent or behavioral signature.
- Allowed proxy tasks.

## Recipe

1. Write the strongest version of the mechanism.
2. Define at least one observation that would falsify it.
3. Define one proxy test and one private authorized test requirement.
4. Map failure modes to mechanism updates.
5. Rank by novelty, universality, measurement clarity, moderation risk, and mitigation value.

## Checks

- Falsifiers must be observable with available tools.
- Do not count scorer mismatch as model success.
- Include clean-chat retest requirements.

## Expected output

A table with mechanism, candidate class, falsifier, proxy test, private test gate, and decision.

## References

- https://github.com/openai/evals
- https://inspect.ai-safety-institute.org.uk/
