

import torch
import os
def _attn_window_pair(k_sw, v_sw, q, w):
    """Ensure k_sw and v_sw are valid for the given window size."""
    if w != 0:
        if k_sw is None or v_sw is None:
            raise ValueError("k_sw and v_sw must be provided when w != 0")
    # Ensure tensors are contiguous for kernel efficiency
    if k_sw is not None:
        k_sw = k_sw.contiguous()
    if v_sw is not None:
        v_sw = v_sw.contiguous()
    return k_sw, v_sw

