# Safe Mechanism Ideation

## Use when

Use this when brainstorming universal-jailbreak hypotheses in an authorized setting without producing payload strings.

## Inputs

- Authorized target scope.
- Current failure taxonomy.
- Mechanism families from related skills.

## Recipe

1. Pick one mechanism family: authority confusion, refusal displacement, template anchoring, hidden carrier collision, reasoning-channel confusion, latent priming, or transfer.
2. Describe why it might generalize across challenge items.
3. Generate candidate classes, not exact prompts.
4. Define a benign proxy test for the latent or behavioral signature.
5. Only escalate to private bounty items after the proxy supports the mechanism.

## Safety boundaries

- Do not write live exploit prompts.
- Use placeholders such as `DISALLOWED_DOMAIN_REQUEST`.
- Keep private candidate payloads outside repo-tracked files.

## Expected output

A mechanism hypothesis list with proxy tests and private-test requirements.

## References

- https://model-spec.openai.com/
- https://arxiv.org/abs/2406.11717
- https://arxiv.org/abs/2311.08374
