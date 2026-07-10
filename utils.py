import inspect
import os
import random
import socket
import sys
import traceback

import numpy as np
import torch
import transformers


def get_device() -> torch.device:
    """Determine the best available device for computation."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    transformers.set_seed(seed)


def is_cluster_node():
    return socket.gethostname().startswith("ece-")


def not_implemented(final_name: str = None):
    """Helper function to raise NotImplementedError with appropriate message."""
    stack = traceback.extract_stack()
    caller = stack[-2]
    if not final_name:
        caller_frame = inspect.currentframe().f_back
        func_name = caller_frame.f_code.co_name
        class_name = None
        for name, obj in caller_frame.f_globals.items():
            if inspect.isclass(obj) and hasattr(obj, func_name):
                class_name = name
                break
        final_name = f"{class_name}.{func_name}" if class_name else func_name
    msg = f"{final_name} not implemented (in {caller.filename}:{caller.lineno})"
    raise NotImplementedError(msg)


if __name__ == "__main__":
    sys.exit("Intended for import.")
