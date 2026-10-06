import numpy as np
import pytest

from src.flatten import flatten_bchw, reconstruct_bchw, flat_index, unflat_index
from src.reference import reference_check


@pytest.mark.parametrize("B,C,H,W", [(1, 2, 2, 3), (2, 1, 5, 4), (3, 7, 3, 3), (1, 500, 4, 4)])
def test_roundtrip_exact(B, C, H, W):
    x = np.random.default_rng(1).standard_normal((B, C, H, W)).astype(np.float32)
    flat = flatten_bchw(x, B, C, H, W)
    back = reconstruct_bchw(flat, B, C, H, W)
    assert flat.shape == (B, C * H * W)
    assert np.array_equal(x, back)
    assert reference_check(x, flat, back)


def test_manual_example_indices():
    H, W = 2, 3
    assert flat_index(0, 0, 2, H, W) == 2
    assert flat_index(0, 1, 0, H, W) == 3
    assert flat_index(1, 0, 0, H, W) == 6
    assert flat_index(1, 1, 2, H, W) == 11


def test_index_inverse():
    H, W = 4, 5
    for j in range(3 * H * W):
        assert flat_index(*unflat_index(j, H, W), H, W) == j
