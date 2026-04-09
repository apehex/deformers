# Benchmark Evaluation Utilities - Implementation Log

Date: 2026-04-09

## What was implemented

### src/deformers/pipelines/eval.py

New shared evaluation utilities module. All functions are CPU/GPU agnostic,
take plain tensors or module objects, and return Python scalars or tensors.

Metric helpers (all operate on raw logits or embedding tensors):

- `embed_mse(teacher, student)`: MSE between (B, T, H) embedding tensors.
- `hidden_mse(teacher, student)`: MSE between (B, T, H) hidden-state tensors.
- `kl_divergence(teacher_logits, student_logits, reduction)`: KL(teacher ||
  student) over (B, T, V) logits; reshapes to (B*T, V) before reduction.
  Raises AssertionError on shape mismatch.
- `top1_match_rate(teacher_logits, student_logits)`: fraction of (B, T)
  positions where argmax tokens match.
- `topk_set_match_rate(teacher_logits, student_logits, k)`: fraction of
  positions where top-k token sets are identical (order ignored).
- `topk_order_match_rate(teacher_logits, student_logits, k)`: fraction of
  positions where top-k token sequences are identical (order-sensitive).

Probe builders:

- `build_vocab_probe(vocab_size, batch_dim, seq_dim)`: deterministic (B, T)
  tensor of consecutive vocab IDs (0, 1, 2, ...) mod vocab_size.
- `build_text_probe(texts, text_tokenizer, byte_tokenizer, seq_dim, patch_dim,
  device)`: tokenizes fixed sentences and returns (tokens, mask, bytes) tensors.
- `build_vocab_probe_bytes(vocab_ids, text_tokenizer, byte_tokenizer,
  patch_dim)`: decodes each token ID to its text string and re-encodes as a
  fixed-length byte block, returning a (B, T, G) tensor.

Teacher forward helpers:

- `teacher_embed(model, input_ids)`: calls model.model.embed_tokens.
- `teacher_forward(model, embeds, attention_mask)`: runs trunk + lm_head with
  inputs_embeds; returns (residuals, logits).

Checkpoint loader:

- `load_prefix_checkpoint(local_path, hf_repo, hf_filename, device)`: loads
  a CompositeBytePrefix from a local .pt file or optionally from HF hub.
  Checkpoint format: dict with keys 'config' and 'state_dict'.

### scripts/benchmark.py

Rewritten as a true evaluation script (no training, no optimizer, no scaler).

Key changes from the original prefix.py copy:
- Dataset split changed to `train[90%:]` (small bounded eval subset).
- Batch size reduced to 4 (memory-safe default for Colab L4).
- Prefix is loaded from a checkpoint (not freshly created).
- Evaluation loop accumulates 6 metrics over a fixed number of batches.
- Summary block prints all metrics after the loop.
- Optional fixed sentence probe: prints teacher and student top-k tokens at the
  last real token position for each fixed sentence.
- Optional vocab probe: runs a deterministic (B, T) token tensor through both
  teacher and student (via byte re-encoding) and reports embed/hidden MSE, KL,
  and top-1 match.
- Uses `deformers.pipelines.eval` helpers throughout; no duplicated metric logic.

### tests/deformers/pipelines/test_eval.py

34 CPU-only unit tests for eval.py metric helpers:
- TestEmbedMse: 3 tests
- TestHiddenMse: 2 tests
- TestKlDivergence: 6 tests (zero case, positive case, shape mismatch, finite,
  reduction comparison)
- TestTop1MatchRate: 5 tests (perfect match, zero match, partial, float output,
  unit interval)
- TestTopkSetMatchRate: 6 tests (identical, disjoint, order-invariance,
  float output, unit interval, k=1 equals top1)
- TestTopkOrderMatchRate: 7 tests (identical, disjoint, order-sensitivity,
  float output, unit interval, order <= set rate, k=1 equals top1)
- TestBuildVocabProbe: 5 tests (shape, range, determinism, dtype, sequential
  fill)

### docs/roadmap.md

- Near-term tasks: marked Stage A evaluation items as done.
- Evaluation checklist: marked KL, top-1, top-k set, top-k order as done.

### docs/context.md

- Added Evaluation priorities section describing current eval focus, metrics,
  and file locations.

## Known limitations

- The benchmark script requires both a downloaded teacher model and a saved
  prefix checkpoint to run end-to-end. There is no offline/mock mode.
- Vocab probe byte re-encoding calls `text_tokenizer.decode([id])` for each
  token ID individually, which is slow for large vocab sizes. For the default
  probe size (B*T = 4*256 = 1024 tokens) this is acceptable.
- The `teacher_forward` helper in eval.py is named for the teacher use-case but
  is actually called with both teacher and student embeddings in the benchmark.
  It is a shared trunk + lm_head forward helper, usable for either.
- Mixed precision (autocast) is applied to the student forward only; teacher
  forward uses bfloat16 model weights directly without autocast to keep
  reference values stable.
- The `load_prefix_checkpoint` function uses `weights_only=True` when calling
  `torch.load`, which is safe for checkpoints saved by prefix.py but will fail
  for checkpoints containing arbitrary Python objects.
