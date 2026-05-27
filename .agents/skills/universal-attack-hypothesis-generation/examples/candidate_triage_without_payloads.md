# Candidate Triage Without Payloads

## Use when

Use this when ranking candidate classes while avoiding operational bypass details in repository files.

## Inputs

- Candidate ids and abstract descriptions.
- Proxy results and risk notes.
- Disclosure constraints.

## Recipe

1. Assign opaque candidate ids.
2. Store only mechanism class, required conditions, and non-sensitive proxy metrics.
3. Keep exact strings and harmful outputs in the authorized private channel.
4. Triage candidates by evidence, transfer expectation, novelty, and mitigation value.
5. Promote only candidates with a clear falsifier and clean-chat plan.

## Safety boundaries

- Repo artifacts should be non-operational if copied alone.
- Do not include target outputs, challenge prompts, or exact bypass text.

## Expected output

A safe triage board that guides research without leaking reusable attacks.

## References

- https://model-spec.openai.com/
- https://github.com/openai/evals
