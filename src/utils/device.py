"""Device detection for MPS/CUDA/CPU environments."""

import logging
import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            "Using CUDA: %s (%d device(s))",
            torch.cuda.get_device_name(0),
            torch.cuda.device_count(),
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS acceleration")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (no GPU detected)")
    return device


def get_torch_dtype() -> torch.dtype:
    """Return the best dtype for the current device."""
    if torch.cuda.is_available():
        return torch.bfloat16
    # MPS and CPU — float32 is safest
    return torch.float32


def is_cluster() -> bool:
    """Check if running on the NVWulf cluster."""
    import os

    return os.path.exists("/lustre/nvwulf")
