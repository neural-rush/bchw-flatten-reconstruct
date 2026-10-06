"""Core algorithms: BCHW <-> B x (CHW), from scratch (no reshape/view/flatten/ravel).

Mapping (row-major, W varies fastest):
    j = c*(H*W) + h*W + w
Inverse:
    c = j // (H*W);  r = j % (H*W);  h = r // W;  w = r % W
"""
import numpy as np


def flat_index(c, h, w, H, W):
    """Address transformation (c, h, w) -> j."""
    return c * (H * W) + h * W + w


def unflat_index(j, H, W):
    """Inverse address transformation j -> (c, h, w)."""
    HW = H * W
    c = j // HW
    r = j % HW
    return c, r // W, r % W


def flatten_bchw(x, B, C, H, W):
    """BCHW -> B x (CHW)."""
    out = np.zeros((B, C * H * W), dtype=x.dtype)
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    out[b, flat_index(c, h, w, H, W)] = x[b, c, h, w]
    return out


def reconstruct_bchw(flat, B, C, H, W):
    """B x (CHW) -> BCHW."""
    out = np.zeros((B, C, H, W), dtype=flat.dtype)
    for b in range(B):
        for j in range(C * H * W):
            c, h, w = unflat_index(j, H, W)
            out[b, c, h, w] = flat[b, j]
    return out
