---
name: universal-jailbreak-research-protocol
description: Use for authorized universal-jailbreak research planning, OpenAI bug bounty evidence gathering, clean-chat evaluation, failure taxonomy, and responsible disclosure packaging without storing live bypass payloads or harmful procedural content.
---

# Universal Jailbreak Research Protocol

## When to Use

- Use this skill when designing, running, or documenting authorized universal-jailbreak research for a bounty or internal red-team program.
- Use it when a task needs clear success criteria, repeatable evaluation, failure analysis, or a disclosure-ready research report.
- Do not use it to publish reusable live jailbreak strings, harmful procedural outputs, or instructions for bypassing deployed systems outside authorization.

## Inputs

- Required context: target program rules, allowed model endpoint or open-weight model, challenge success criteria, and disclosure constraints.
- Required artifacts: clean-chat transcripts, exact model/version/date, sampling settings, moderation status when applicable, and evaluator outputs.
- Optional artifacts: latent probes, activation traces, failure clusters, transfer results, and source-backed hypotheses.

## Workflow

1. Define the target precisely: model, date, chat state, clean-chat requirement, number of challenge items, and scoring oracle.
2. Separate public methodology from private bounty artifacts. Keep private challenge prompts, full candidate payloads, and target outputs out of repo-tracked skills and docs unless the program explicitly permits storage.
3. State the mechanism hypothesis before testing: authority confusion, refusal suppression, template anchoring, hidden policy carrier displacement, reasoning-channel confusion, latent state priming, or transfer-based universality.
4. For each universal candidate, record one immutable run sheet: candidate id, prompt family, model/version, settings, all challenge outcomes, refusals, partial completions, moderation events, and evaluator judgement.
5. Analyze failures as mechanisms, not just strings:
   - boundary failure: prompt does not generalize across challenge items
   - authority failure: model preserves correct hierarchy
   - refusal failure: refusal remains causally active
   - content failure: output is non-refusal but does not satisfy grader
   - moderation failure: candidate triggers upstream filters
   - transfer failure: works only for one model snapshot or template
6. Convert successful private findings into disclosure evidence: minimal reproduction, impact, universality claim, model/version scope, limitations, and recommended mitigations.

## Research Heuristics

- Treat "universal" as an empirical property. Require performance across all target items from clean state, plus retests across paraphrase, decoding, and model snapshot when available.
- Prioritize mechanism-diverse candidates over string variants. Ten variants of one trick are weaker evidence than three mechanistically distinct probes.
- Keep a negative-results log. Failed candidates identify intact safeguards and prevent repeated exploration of dead ends.
- Test against benign proxy targets first when the mechanism does not require restricted content. Escalate to private challenge material only when the mechanism and measurement are ready.
- Use caricatured or synthetic forbidden targets only as calibration; do not let toy targets substitute for the actual bounty success criterion.

## Outputs

- A run sheet with exact settings and pass/fail status.
- A mechanism note explaining why the candidate should transfer across challenge items.
- A failure taxonomy update for unsuccessful candidates.
- A disclosure draft for any candidate that meets the program threshold.

## Constraints

- Keep repo-tracked artifacts non-operational. Store live candidate payloads and challenge-specific harmful content in the authorized private channel required by the program.
- Do not claim a universal jailbreak from selective examples; include all challenge attempts and failures.
- Do not mix NDA-covered bounty material into public docs unless explicitly allowed.
- If equations or algorithms from converted markdown appear corrupted, verify against the original source before implementation.

## Verification

- Confirm every run sheet includes model id, date, settings, clean-chat state, and all challenge outcomes.
- Re-run any claimed success at least once with identical settings and once with a new clean chat.
- Check that public notes contain mechanisms, metrics, and mitigations but not reusable exploit text.

## Source Map

- `references/011-openai-gpt-5-5-bio-bug-bounty.md`
- `references/012-openai-instruction-hierarchy-challenge.md`
- `references/013-openai-model-spec.md`
- `references/014-iheval-evaluating-language-models-on-following-the-instruction-hierarchy.md`
- `references/015-sysbench-can-large-language-models-follow-system-messages.md`
- `references/016-control-illusion-the-failure-of-instruction-hierarchies-in-large-language-models.md`
- `references/017-prompt-injection-as-role-confusion.md`
- `references/057-linearly-decoding-refused-knowledge-in-aligned-language-models.md`
