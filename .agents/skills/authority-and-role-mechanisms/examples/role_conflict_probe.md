# Role-Conflict Probe

## Use when

Use this when testing whether a model represents low-authority text as system, developer, user, assistant, or tool content.

## Inputs

- Role-labeled conversations with matched semantic content.
- Exact chat template and serialization.
- Hidden states or logits from candidate layers and positions.

## Recipe

1. Serialize the same instruction under multiple true roles.
2. Fit a lightweight classifier on hidden states to predict the true role.
3. Evaluate on held-out instructions and template variants.
4. Add one conflict at a time: user-vs-tool, developer-vs-user, or quoted-data-vs-user.
5. Correlate probe confusion with benign instruction-following outcomes.

## Checks

- Use held-out content so the probe does not learn task semantics.
- Report confusion matrices and calibration, not only accuracy.
- Keep public examples abstract and harmless.

## Expected output

A role-confusion report with layer/position localization and a pre-generation risk score.

## References

- https://model-spec.openai.com/
- https://assets.amazon.science/22/57/ad173f7f449eadaaa7cd05491585/iheval-evaluating-language-models-on-following-the-instruction-hierarchy.pdf
- https://arxiv.org/abs/2604.09443
