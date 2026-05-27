# Clean-Chat Run Sheet

## Use when

Use this to record authorized universal-jailbreak testing in a way that supports retest and disclosure.

## Inputs

- Candidate id.
- Model id, version, date, endpoint, and settings.
- Challenge outcomes and evaluator decisions.

## Recipe

1. Start from a new clean chat.
2. Record model/version/date, sampling settings, tools, and moderation state.
3. Run all target items, not selected examples.
4. Record pass, fail, refusal, partial, moderation, and scorer mismatch outcomes.
5. Repeat any claimed success in a new clean chat.

## Safety boundaries

- Store exact payloads and target outputs only in the authorized private channel.
- Public notes should use abstract candidate ids and outcome counts.

## Expected output

An immutable run sheet that supports all-items evaluation and clean-chat retesting.

## References

- https://github.com/openai/evals
- https://inspect.ai-safety-institute.org.uk/
