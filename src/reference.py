"""REFERENCE ONLY - independent verification using library reshape.
Not part of the submitted algorithm."""
import numpy as np


def reference_check(x, flat, back):
    ref_flat = x.reshape(x.shape[0], -1)
    return bool(np.array_equal(ref_flat, flat) and
                np.array_equal(ref_flat.reshape(x.shape), back))
