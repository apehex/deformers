# Spoofing-Factor Ablation

## Use when

Use this when isolating which formatting or language cues shift perceived authority.

## Inputs

- A benign base instruction.
- A list of candidate spoofing factors.
- Role-probe and behavior metrics.

## Recipe

1. Start from a harmless low-authority message.
2. Add only one cue per condition: header style, source declaration, quoting, foreign template marker, whitespace, or assistant-continuation framing.
3. Keep requested task constant across variants.
4. Measure probe-predicted role and behavior outcome.
5. Rank cues by authority-confusion effect and confidence interval.

## Checks

- Do not include real exfiltration commands or live prompt-injection strings.
- Include a clearly quoted/untrusted-data control.
- Test at least one unseen template.

## Expected output

An ablation table linking each cue to latent role shift, behavior shift, and mitigation hypothesis.

## References

- https://model-spec.openai.com/
- https://arxiv.org/abs/2311.08374
- https://arxiv.org/abs/2604.09443
