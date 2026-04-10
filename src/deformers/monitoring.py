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

import torch
import torch.optim

# GPU ##########################################################################

def gpu_memory_mb(
    device: str='',
) -> dict[str, float]:
    """Return current GPU memory stats in MB."""
    __stats = {'allocated_mb': 0.0, 'reserved_mb': 0.0}
    # in mega bytes
    if torch.cuda.is_available() and device:
        __stats['allocated_mb'] = torch.cuda.memory_allocated(device) / float(2 ** 20)
        __stats['reserved_mb']= torch.cuda.memory_reserved(device) / float(2 ** 20)
    # defaults to 0
    return __stats

# OPTIMIZER ####################################################################

def current_lr(
    optimizer: torch.optim.Optimizer,
) -> float:
    """Return the current learning rate from the first optimizer param group."""
    return float(optimizer.param_groups[0]['lr'])

# THROUGHPUT ###################################################################

def throughput(
    count: int,
    elapsed: float,
    epsilon: float=1e-6,
) -> float:
    """Compute items per second from a count and an elapsed wall time."""
    return (float(count) / elapsed) if  (elapsed > 0.0) else 0.0

# TENSORBOARD ##################################################################

def log_scalars(
    writer,
    scalars: dict[str, float],
    step: int,
) -> None:
    """Log a dict of (tag -> float) scalars to a SummaryWriter."""
    if hasattr(writer, 'add_scalar'):
        for __tag, __val in scalars.items():
            writer.add_scalar(__tag, float(__val), step)
