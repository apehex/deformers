# Context

## Repository

`apehex/deformers` on Github.

## Models

Mostly `qwen/qwen3.5-9b`.

## Purpose

Patch the layers of pretrained LLMs.

## Objectives

Experiment with model level composition and test the modularity.

## Directions

- prefix patches (input representation)
- suffix patches (output distribution)
- custom decoder (reversing the auto-regression, toward past tokens)

## Design Principles

- freeze most of the original model
- train only localized modules
- keep the same input partition and tokenization vocabulary
- align patched modules with the pretrained model via distillation or hidden-state matching

## Motivations

- remove dependency on large token embedding tables
- exploit internal structure of token strings
- allow deterministic reconstruction of embeddings
- keep the information on the token composition
- test modular training and composition of sub-models

## Workflow

1. split pretrained model into prefix / trunk / suffix
2. replace prefix or suffix with experimental module
3. train patch using teacher outputs from the original model
4. evaluate performance relative to baseline model using `scripts/benchmark.py`

## Evaluation priorities

Current evaluation focus (Stage A prefix patch):
- embedding MSE between teacher token embeddings and prefix output
- hidden-state MSE at configured trunk depth `k`
- KL divergence between teacher and student logits
- top-1 match rate, top-k set match rate, top-k exact-order match rate
- fixed sentence probe: visual comparison of teacher vs student top-k tokens
- vocab probe: deterministic (B, T) token tensor for repeatable comparison
