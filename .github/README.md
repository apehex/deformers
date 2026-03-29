# Deformers <img src="images/logo.png" alt="apehex logo" width="32" height="32">

[![License][shield-license]][github-license]
[![Latest][shield-release]][github-release]

Experiments with modular neural networks by patching pre-trained LLMs.

## Original Model

All experiments use the open-source model **`qwen/qwen3.5-9b`**.

Relevant configuration:

- hidden size: **4096**
- number of layers: **32**
- vocabulary size: **248320**
- embedding size: **4096**
- positional encoding: rotary
- embedding weights are **not tied** with the output head

The embedding layer and output head each contain approximately:

```
4096 × 248320 ~ 1.02B parameters
```

## Patching Layers

### Composite Embedding (Prefix Patch)

#### Frozen components

- tokenization (Qwen BPE tokenizer)
- transformer trunk (all the hidden transformer layers)
- positional encoding
- output head

#### Replaced component

The original token embedding `Embedding(V=248320, D=4096)` is replaced with a **composite embedding layer** with:

- a group dimension of 32
- an input dimension of 256 (byte values)
- an embedding dimension of 128
- for a total of 256 * 128 = 32768 parameters

So that a tensor of shape `(B, 32 * S)` is processed into `(B, S, 32 * 128) = (B, S, 4096)`.

Then a regular transformer block maps these composite embeddings with the original embeddings of the Qwen model.

Current implementation in this repository uses:

- `CompositeEmbedding(256, hidden_size / 32, group_dim=32)` for byte patches
- an extra Qwen decoder layer (`Qwen3DecoderLayer`) as prefix encoder before the frozen trunk
- frozen trunk and frozen LM head, with only prefix modules trainable

#### Input representation

Input text is tokenized using the original Qwen tokenizer.

Each token string is encoded as UTF-8 bytes.

Shorter tokens are then padded into a block of 32 bytes and the longer tokens are truncated.

#### Patch Training

The training was performed on a multilingual corpora with a custom loss:

$$L_{k} = || H_{k, patch}(x) − H_{k, qwen}(x) ||^{2}$$

Where:

- $k$ is the depth inside the original Qwen 3.5 model
- $H_{k, qwen}$ is the hidden state at depth $k$ in the original model
- $H_{k, patch}$ is the hidden state obtained when replacing the embedding layer

The code now includes a lightweight prefix-distillation loop (Colab-friendly):

- teacher and student run with shared token partition
- hidden-state MSE distillation on the frozen trunk outputs
- mixed precision (`torch.autocast`) + gradient accumulation + gradient clipping

### Hierarchical Softmax Head

#### Frozen components

- tokenizer
- embedding layer
- positional encoding
- transformer trunk (all the hidden layers)

#### Replaced component

Original output head `Linear(4096 => 248320)` is replaced by a **hierarchical softmax tree**.

Tokens are organized in a binary tree with a depth of `18 ~ log_2(248320)`

Each token corresponds to a unique path from root to leaf.

#### Patch Training

Here a simple cross entropy loss is enough.

## License

Licensed under the [aGPLv3](LICENSE.md).

[github-license]: LICENSE.md
[github-release]: https://github.com/apehex/deformers/releases/latest

[shield-license]: https://img.shields.io/badge/license-aGPLv3-green?style=flat-square
[shield-release]: https://img.shields.io/github/release/apehex/deformers.svg?style=flat-square
