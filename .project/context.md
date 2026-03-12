# Context

Repository: `apehex/deformers`

Purpose:
Experiments with modular neural network patches applied to pretrained large language models.

Primary target model:
`qwen/qwen3.5-9b`

Main objective:
Evaluate whether large pretrained LLMs can be partially replaced by smaller structured modules while keeping most of the transformer trunk frozen.

Current patching directions:

- prefix patches (input representation)
- suffix patches (output distribution)
- reversible decoding experiments

Design principles:

- keep pretrained trunk frozen when possible
- train only localized modules
- align patched modules with the pretrained model via distillation or hidden-state matching
- maintain compatibility with the original tokenizer and vocabulary unless explicitly testing alternatives

Typical workflow:

1. split pretrained model into prefix / trunk / suffix
2. replace prefix or suffix with experimental module
3. train patch using teacher outputs from the original model
4. evaluate performance relative to baseline model

Focus areas:

- embedding compression
- vocabulary compression
- modular training
- structured output heads
