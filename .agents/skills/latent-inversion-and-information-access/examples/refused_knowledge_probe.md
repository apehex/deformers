# Refused-Knowledge Probe

## Use when

Use this when studying whether information remains decodable from aligned models even when generation refuses.

## Inputs

- Authorized dataset with safe labels or abstracted policy-boundary labels.
- Hidden states from base and instruction-tuned models.
- Linear probe and held-out evaluation split.

## Recipe

1. Define a label that can be measured safely.
2. Extract activations for prompts that trigger refusal and matched allowed controls.
3. Train linear probes on base, tuned, and cross-model settings.
4. Compare decodability against generated behavior.
5. Report whether latent prediction correlates with indirect downstream choices.

## Safety boundaries

- Do not decode or publish operational harmful answers.
- Use placeholders for policy-sensitive target content.
- Treat probe outputs as sensitive if they reveal private or restricted attributes.

## Expected output

A decodability report separating latent access, generated compliance, and indirect behavioral influence.

## References

- https://arxiv.org/abs/2507.00239
- https://arxiv.org/abs/2406.11717
