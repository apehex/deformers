# Invariants

These constraints should not change unless explicitly reconsidered.

- tokenizer boundaries must remain identical to the base model
- the hidden layers (model trunk) must be frozen during training
- training must support distillation from the base model

- safety research must be defensive, authorized, and non-operational
- public documentation must not contain live jailbreak payloads, exploit strings, or operational bio-risk content
- latent-space interventions must be evaluated against both target behavior and utility preservation
- cross-model transfer claims must include a measured alignment or transfer metric
- role-token, padding-token, and chat-template assumptions must be documented explicitly
