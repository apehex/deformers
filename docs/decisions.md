# Decisions

## Tokenization

Keep the same partitioning as the tokenizer from the original model.

It guarantees that the input tokens are the same regardless of the model's prefix.

```python
text = 'This avoids the segmentation mismatch with fixed size patching.'
print([text[__s:__e] for (__s, __e) in tokenizer(text, return_offsets_mapping=True)['offset_mapping']])
# ['This', ' avoids', ' the', ' segmentation', ' mismatch', ' with', ' fixed', ' size', ' patch', 'ing', '.']
print([text[__s:__s + 8] for __s in range(0, len(text), 8)])
# ['This avo', 'ids the ', 'segmenta', 'tion mis', 'match wi', 'th fixed', ' size pa', 'tching.']
```

Pad the shorter tokens with null bytes and truncate the longer tokens so that the encoding is fixed to 32 bytes.

This impacts less than 3% of the tokens, which look like data (binary) tokens:

```python
print(len(tokenizer.get_vocab())) # total number of tokens
# 248077
print(len([__t for __t in tokenizer.get_vocab().keys() if len(__t.encode('utf-8')) > 32])) # number of truncated tokens
# 6365
print(100 * 6365 / 248077) # percentage of truncated tokens
# 2.565735638531585
print([__t for __t in tokenizer.get_vocab().keys() if len(__t.encode('utf-8')) > 32][:8])
# ['ĠÐ·Ð°ÑģÑĤÐ°Ð²Ð»ÑıÐµÑĤ',
#  'ĠÑĤÐµÑħÐ½Ð¸ÑĩÐµÑģÐºÐ¸Ñħ',
#  'ĠÐ¿Ð¾ÑģÑĤÐ¾ÑıÐ½Ð½Ð¾Ð³Ð¾',
#  'à¸Ĺà¸µà¹Ģà¸Ķà¸µà¸¢à¸§',
#  'ĠÐĹÐ°ÐºÐ¾Ð½Ð¾Ð´Ð°',
#  'à¸Ļà¸°à¸Ħà¸£à¸±à¸ļ',
#  'à¸£à¸¸à¹Īà¸Ļà¹ĥà¸«à¸¡à¹Ī',
#  'ĠÑģÐµÑĢÑĤÐ¸ÑĦÐ¸ÐºÐ°']
print([__t.encode('utf-8').hex() for __t in TOKENIZER_OBJ.get_vocab().keys() if len(__t.encode('utf-8')) > 32][:8]) # hex encoded, since these are likely data tokens
# ['c4a0c390c2bfc391c4a2c390c2bec391c4a7c390c2bec390c2b4c390c2b8c391c4a4c391c4ae',
#  'c3a0c2b9c4a2c3a0c2b8c4acc3a0c2b8c2b7c3a0c2b9c4aac3a0c2b8c583c3a0c2b8c2a1c3a0c2b9c4a4c3a0c2b8c2a2c3a0c2b8c4a9',
#  'c4a0c391c4a5c391c4a9c390c2b0c391c4a3c391c4a4c390c2bac390c2bec390c2b2',
#  'c3a0c2b8c2a2c3a0c2b8c2b2c3a0c2b8c2a1c3a0c2b8c4a6c3a0c2b9c4aac3a0c2b8c2b3c3a0c2b8c4a6c3a0c2b8c2b7c3a0c2b8c4bb',
#  'c4a0c3a0c2b8c4b7c3a0c2b8c2b2c3a0c2b8c2a1c3a0c2b8c2a3c3a0c2b8c2b0c3a0c2b8c4b6c3a0c2b8c2b1c3a0c2b8c4bc',
#  'c3a0c2b8c2a3c3a0c2b8c2b2c3a0c2b8c2a2c3a0c2b8c4a9c3a0c2b8c2b2c3a0c2b8c4bb',
#  'c3a0c2b8c2b3c3a0c2b8c4bbc3a0c2b8c2a7c3a0c2b8c2a2c3a0c2b8c4a3c3a0c2b8c2b2c3a0c2b8c2a3',
#  'c4a0c390c2b4c390c2b5c390c2b9c391c4a3c391c4a4c390c2b2c390c2bec390c2b2c390c2b0c391c4a4c391c4ae']
```

## Prefix Input Encoding

Token strings are encoded as follows:

- source: raw token-piece strings obtained via `tokenizer.get_vocab()` (not `decode()`), to preserve the exact byte composition of each learned BPE symbol
- encoding: UTF-8 bytes per token-piece
- fixed patch length: 32 bytes
- padding: shorter tokens are left-padded with sentinel byte value 128
- truncation: tokens longer than 32 bytes are truncated; this affects ~2.6% of the vocabulary (mostly binary/CJK data tokens)
- padding token: the tokenizer pad token (`<|endoftext|>`) is replaced by an empty string before byte encoding, so it maps to a full patch of padding bytes

Rationale:
- a fixed patch length makes the byte-to-embedding projection architecture simple and the input shape fully static
- the value 128 is not a valid single-byte UTF-8 character, which makes it suitable as an internal padding marker

## Prefix Architecture Policy

The prefix module maps fixed-size byte patches to the hidden size of the
teacher model:

```text
bytes:  (B, T, G)
output: (B, T, H)
```

where:

- `B` is the batch dimension
- `T` is the token sequence dimension
- `G` is the fixed byte-patch length, currently 32
- `H` is the hidden size of the frozen teacher trunk

The prefix must process each token patch independently.
It may attend over the byte-patch axis `G`, but it must not attend over the token sequence axis `T` when used as a strict embedding-table replacement.

Rationale:
- the original embedding table is context-independent
- preserving this property makes the prefix easier to evaluate

## Training Decomposition

- frozen: trunk transformer layers and lm_head; their weights are never updated
- trained: prefix module only (the byte-to-embedding projection)
- teacher signals are computed with a single forward pass through the frozen trunk

This decomposition keeps the training surface small and decouples prefix alignment from trunk adaptation.

## Metric Policy

- primary training loss: embedding MSE (student prefix output vs teacher embedding at depth 0) plus hidden-state MSE at a chosen trunk depth k
- token-probability KL (logit KL): tracked as an evaluation metric for comparing output distributions; not used as the primary training loss
- top-k agreement rate: tracked per step and on fixed probes

Rationale: MSE and cosine similarity provide stable, interpretable gradients for vector-space alignment; logit KL is noisy early in training and is better used as a diagnostic of downstream behavioral alignment.

## Masking Policy

All losses and metrics must exclude positions corresponding to padding tokens.

- attention mask shape: `(B,)` or `(B, T)`; broadcast to match the loss tensor
- masked MSE: multiply squared error by the mask before reduction; normalize by the count of valid (non-padding) positions
- masked KL: compute per-token KL first (sum over vocab dimension), then apply the mask and normalize by the number of valid tokens
- masked top-k: apply the mask to per-position match indicators before averaging

Rationale: including padding positions would dilute the loss and distort gradient estimates, especially when sequences have variable length.

## Normalization Axis Convention

- default: `LayerNorm` along the feature (hidden) dimension, consistent with the sequence layout `(B, T, H)`
- alternative: `RMSNorm` as a lighter drop-in replacement for `LayerNorm`; preferred when scale-only normalization is sufficient
- if using `GroupNorm` or other channel-first norms: transpose to `(B, H, T)` before applying the norm, then transpose back
- no-norm variant: removing normalization entirely is a valid ablation; rely on initialization and learning rate instead

Rationale: keeps the layout consistent with HuggingFace transformer conventions and avoids silent shape errors from channel-first vs channel-last mismatch.
