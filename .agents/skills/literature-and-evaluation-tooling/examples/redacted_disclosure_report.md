# Redacted Disclosure Report

## Use when

Use this when converting sensitive safety findings into a shareable report.

## Inputs

- Private run sheets.
- Program rules and disclosure constraints.
- Non-operational evidence summary.

## Recipe

1. Separate private reproduction material from public mechanism notes.
2. Replace harmful or challenge-specific content with abstract placeholders.
3. Keep model/version/date, settings, outcome counts, and mitigation-relevant details.
4. Include failures and limitations.
5. Ask whether each included detail helps reproduction by the authorized reviewer or helps misuse by an outside reader.

## Safety boundaries

- Do not publish live bypass payloads, target outputs, or private challenge prompts.
- Preserve enough evidence for authorized review in the private channel.

## Expected output

A disclosure-ready report with impact, scope, evidence summary, limitations, and mitigation recommendations.

## References

- https://github.com/openai/evals
- https://model-spec.openai.com/
