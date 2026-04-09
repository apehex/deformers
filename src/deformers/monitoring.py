"""
Generic training monitoring utilities.

All helpers are stateless, CPU/GPU agnostic, and import only torch.
They are not specific to any experiment; put experiment-specific logic in scripts.

Available helpers:
- gpu_memory_mb: GPU memory stats in MB (allocated and reserved).
- current_lr: current learning rate from an optimizer param group.
- throughput: items (tokens or examples) per second from count and elapsed time.
- log_scalars: log a dict of scalars to a SummaryWriter; no-op if writer is None.
"""

from typing import Dict, Optional

import torch
import torch.optim

# GPU ##########################################################################

def gpu_memory_mb(
    device: Optional[torch.device]=None,
) -> Dict[str, float]:
    """Return current GPU memory stats in MB.

    Returns a dict with keys 'allocated_mb' and 'reserved_mb'.
    Returns zeros on CPU-only environments.
    """
    if not torch.cuda.is_available():
        return {'allocated_mb': 0.0, 'reserved_mb': 0.0}
    return {
        'allocated_mb': torch.cuda.memory_allocated(device) / 1e6,
        'reserved_mb': torch.cuda.memory_reserved(device) / 1e6,}

# OPTIMIZER ####################################################################

def current_lr(
    optimizer: torch.optim.Optimizer,
) -> float:
    """Return the current learning rate from the first optimizer param group."""
    return float(optimizer.param_groups[0]['lr'])

# THROUGHPUT ###################################################################

def throughput(
    count: int,
    elapsed_sec: float,
) -> float:
    """Compute items per second from a count and an elapsed wall time.

    Args:
        count: number of tokens or examples processed.
        elapsed_sec: elapsed wall time in seconds.

    Returns:
        Rate in count-per-second. Returns 0.0 if elapsed_sec <= 0.
    """
    if elapsed_sec <= 0.0:
        return 0.0
    return float(count) / elapsed_sec

# TENSORBOARD ##################################################################

def log_scalars(
    writer,
    scalars: Dict[str, float],
    step: int,
) -> None:
    """Log a dict of (tag -> float) scalars to a SummaryWriter.

    No-op if writer is None, which allows callers to disable TensorBoard
    by passing None without conditional logic at the call site.

    Args:
        writer: torch.utils.tensorboard.SummaryWriter instance, or None.
        scalars: dict mapping tag strings to float values.
        step: global step index for TensorBoard.
    """
    if writer is None:
        return
    for __tag, __val in scalars.items():
        writer.add_scalar(__tag, float(__val), step)
