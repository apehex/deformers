# Stage A Prefix Module - Implementation Log

Date: 2026-04-05

## What was implemented

### src/deformers/layers/prefix.py

- `CompositeBytePrefix(torch.nn.Module)`: Stage A alternative prefix module.
- Accepts `(B, T, G)` long tensors (rank-3 mode, group_dim <= 0) or `(B, T*G)`
  flat tensors (rank-2 mode, group_dim > 0).
- Pipeline: `CompositeEmbedding` -> `LayerNorm` -> `Linear` -> `SiLU` ->
  `Linear` -> `LayerNorm`.
- Output shape: `(B, T, latent_dim)` float, compatible with `inputs_embeds`.
- Lazy-build: sub-layers initialized on first `forward` call, placed on the
  input tensor's device. No explicit device/dtype constructor args needed.
- `_config` dict preserved for serialization.

### src/deformers/patching/bytes.py

- `token_offsets_to_substrings(texts, offsets)`: converts tokenizer offset
  mappings to per-token substring lists.
- `encode_token_substrings_to_bytes(tokens, byte_tokenizer, max_length, ...)`:
  encodes each token substring to a fixed-length byte block (pad_id=128).
- `decode_bytes_to_text(byte_ids, byte_tokenizer)`: decodes byte-id lists back
  to text (padding bytes removed automatically).
- `encode_texts(texts, tokenizer, byte_tokenizer, max_length, ...)`: convenience
  function returning `input_ids`, `attention_mask`, `byte_ids` as aligned
  LongTensors of shapes `(B, T)`, `(B, T)`, `(B, T, G)`.

### scripts/train_prefix_stage_a.py

- Loads `qwen/qwen3.5-9b` tokenizer and model.
- Freezes all model parameters (trunk + lm_head).
- Builds `CompositeBytePrefix` with `latent_dim = model.config.hidden_size`.
- Teacher forward with `input_ids` (no grad); student forward with `inputs_embeds`.
- Loss: `w_hidden * MSE(student_h_k, teacher_h_k) + w_embed * MSE(student_emb, teacher_emb)`.
- Supports autocast + GradScaler on CUDA, gradient accumulation, grad clipping.
- Falls back to local text constant if wikitext dataset is unavailable.
- Saves checkpoint to `checkpoints/prefix_stage_a.pt`.

### tests/deformers/layers/test_prefix.py

- 12 unit tests for `CompositeBytePrefix`: shapes, dtypes, device propagation,
  assertion errors for wrong ranks, parameter registration, edge cases.

### tests/deformers/integration/test_inputs_embeds_smoke.py

- 4 integration tests using a tiny `Qwen3Config` model (no download required).
- Covers: `inputs_embeds` forward pass, `hidden_states[k]` extraction,
  teacher-student MSE loss, CUDA path (skipped if unavailable).

## Caveats

- Lazy-build: if the module is moved to CUDA with `.to('cuda')` before the first
  forward call, the device placement relies on `inputs_arr.device` being correct
  at call time. The `_build` method accepts `device` from the input tensor.
- The `group_dim <= 0` / rank-3 input mode is the primary interface for pre-split
  `(B, T, G)` tensors from `encode_texts`. The flat rank-2 mode is a secondary
  convenience if callers have not pre-split.
- About 2.6% of Qwen vocab tokens exceed 32 bytes and are silently truncated.
  This is documented in `docs/decisions.md`.
- The training script requires the full `qwen/qwen3.5-9b` weights (~18 GB).
  For Colab use, a T4 instance needs gradient checkpointing or reduced batch
  size; an A100 can run with the default config.
