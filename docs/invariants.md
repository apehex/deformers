# Invariants

These constraints should not change unless explicitly reconsidered.

- tokenizer boundaries must remain identical to the base model
- transformer trunk is frozen for prefix experiments
- composite embedding must output vectors of dimension 4096
- training must support distillation from the base model
