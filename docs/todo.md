# TODO

Near-term tasks.

## Documentation

- finalize the readme
- review all the docs

## Prefix experiment

- code the custom loss at depth k
- share the trunk between original and patched models
- train the prefix patch
- compare the logits of the 2 models:
    - KL divergence between the 2
    - text generation samples

## Suffix experiment

- turn the BPE merges into a binary tree
- make the training loss / task tighter

## Infrastructure

- implement model splitting utilities
- implement teacher distillation pipeline
