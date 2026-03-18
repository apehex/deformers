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

# Workflow

1. split pretrained model into prefix / trunk / suffix
2. replace prefix or suffix with experimental module
3. train patch using teacher outputs from the original model
4. evaluate performance relative to baseline model

## Focus Areas

- embedding compression
- vocabulary compression
- modular training
- structured output heads
