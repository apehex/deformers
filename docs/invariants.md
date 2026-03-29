# Invariants

These constraints should not change unless explicitly reconsidered.

- tokenizer boundaries must remain identical to the base model
- the hidden layers (model trunk) must be frozen during training
- training must support distillation from the base model
