"""Reconstruction error metrics (explicit loops)."""


def errors(a, b, B, C, H, W):
    """Return (Emax, MAE) between tensors a and b of shape BCHW."""
    e_max, total = 0.0, 0.0
    for bb in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    d = abs(float(a[bb, c, h, w]) - float(b[bb, c, h, w]))
                    total += d
                    if d > e_max:
                        e_max = d
    return e_max, total / (B * C * H * W)
