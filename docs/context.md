# Context

## Repository

`apehex/deformers` on Github.

## Models

Mostly `qwen/qwen3.5-9b`.

## Purpose

Patch the layers of pretrained LLMs.

Large foundation models are monolithic: embedding table, trunk, and output head are trained jointly and are tightly coupled.
The goal of this project is to expose clean modular interfaces so that individual components can be replaced or retrained in isolation, without touching the rest.

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

## Current Experiment: Prefix Patch

The active experiment trains a byte-based prefix module to replace the original embedding layer.

- input: token-piece strings encoded as UTF-8 bytes, padded or truncated to a fixed patch length of 32 bytes
- target: original teacher embedding vectors (frozen)
- trunk and lm_head remain fully frozen; only the prefix parameters are trained

Success is defined in behavioral terms: the patched prefix should produce hidden states and logits that closely match those of the original model.
Perfect embedding reconstruction is not the goal; what matters is that downstream behavior (hidden-state similarity, top-k token agreement, perplexity) is preserved.

## Workflow

1. split pretrained model into prefix / trunk / suffix
2. replace prefix or suffix with experimental module
3. train patch using teacher outputs from the original model
4. evaluate performance relative to baseline model using `scripts/benchmark.py`
