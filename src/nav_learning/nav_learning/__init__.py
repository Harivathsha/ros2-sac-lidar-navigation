"""SAC navigation for the original hv_bot and Jazzy square arena."""

__version__ = "1.0.0"

# SAC_CSV_LOGGING_GUARD_V1
# This project uses CSV and PNG output. Suppress SB3's optional TensorBoard
# import before it can load TensorFlow into the PyTorch/Triton process.
# This is local to the current Python process; installed packages are unchanged.
import sys as _sys
_sys.modules.setdefault("torch.utils.tensorboard", None)
_SAC_CSV_LOGGING_ONLY = True
