# Stage A Monitoring Implementation Log

Date: 2026-04-09

## What was implemented

### src/deformers/pipelines/monitor.py

New generic monitoring utilities module. All functions are CPU/GPU agnostic,
stateless, and import only torch. Not specific to any experiment.

- `gpu_memory_mb(device)`: returns dict with 'allocated_mb' and 'reserved_mb'.
  Returns zeros on CPU-only environments (CUDA not available).
- `current_lr(optimizer)`: reads the learning rate from the first param group.
- `throughput(count, elapsed_sec)`: computes items/sec; returns 0.0 if elapsed <= 0.
- `log_scalars(writer, scalars, step)`: logs a dict of tag->float to a
  SummaryWriter; no-op if writer is None.

### scripts/prefix.py

Updated training script with full monitoring integration.

New config:
- LOGGING_CFG now includes 'log_dir' (default: runs/prefix) and 'tensorboard'
  toggle (default: True).

New imports: tqdm, time, deformers.monitoring, deformers.pipelines.eval.
TensorBoard writer is created at startup with graceful fallback if the
tensorboard package is not installed.

compute_loss now returns a 3-tuple (loss_scaled, embed_mse, hidden_mse) instead
of a single tensor. The caller accumulates the component losses separately.

Training loop changes:
- tqdm progress bar wraps the inner batch iterator, showing epoch description
  and optimizer-step postfix.
- Per-micro-step: accum_loss (scaled sum = mean after N steps), accum_embed_mse
  and accum_hidden_mse are accumulated.
- Per-optimizer-step:
  - KL divergence is computed via SOURCE_MOD.lm_head on the last micro-batch's
    hidden states (first item only, to avoid large (B,T,V) logit tensors).
  - Grad norm is computed from clip_grad_norm_ return value.
  - Step time is measured with time.monotonic(); throughput in tok/s is derived.
  - GPU memory stats are read via deformers.monitoring.gpu_memory_mb().
  - Stdout line: epoch/total, step/total, loss, embed, hidden, kl, lr, gnorm,
    ms, tok/s.
  - TensorBoard tags emitted: train/loss_total, train/loss_embed,
    train/loss_hidden, train/lr, train/grad_norm, train/step_time_ms, train/kl,
    train/throughput_tok_per_sec, gpu/memory_allocated_mb, gpu/memory_reserved_mb.
  - Progress bar postfix: opt, loss, embed, hidden, kl, lr.

### tests/deformers/test_monitoring.py

19 CPU-only unit tests for monitoring.py:
- TestGpuMemoryMb: 4 tests (keys, CPU zeros, float types, non-negative).
- TestCurrentLr: 4 tests (initial value, float return, after update, small LR).
- TestThroughput: 6 tests (correct rate, float return, zero elapsed, negative
  elapsed, zero count, fractional elapsed).
- TestLogScalars: 6 tests (no-op on None, call count, tag, step, value cast,
  empty dict).

### Bug fix: deformers/pipelines/eval.py

topk_rate now returns a Python float (via .item()) instead of a torch.Tensor.
The existing test test_returns_float expected float; the prior implementation
returned a Tensor causing a CI failure.

### docs/roadmap.md

Monitoring checklist updated to [x] for all implemented items.
Fixed/reference-batch hidden MSE on top-k vocab tensor remains deferred ([ ]).
TensorBoard task marked as done.

### docs/context.md

Added Observability section describing the monitoring stack, KL computation
policy, and helper module location.

## Limitations and deferred items

- Fixed/reference-batch hidden MSE on a top-k vocab tensor is intentionally
  deferred (per problem statement).
- KL is computed on the first batch item only to avoid ~1 GB logit tensors
  at full batch size with vocab_size=248077. This gives a valid but not
  batch-averaged KL signal.
- TensorBoard requires the 'tensorboard' Python package to be installed
  separately. Missing tensorboard is handled gracefully (writer=None).
- tqdm requires the 'tqdm' Python package; not added to pyproject.toml since
  it is a script-level dependency, not a package dependency.
- The train/kl and train/throughput_tok_per_sec tags are additional beyond the
  originally specified set; they provide useful complementary signals.
