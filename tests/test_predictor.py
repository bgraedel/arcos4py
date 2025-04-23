"""pytest unit‑tests for the Numba‑accelerated custom predictors.

Run with `pytest -q`.
"""

from __future__ import annotations

import numpy as np
from arcos4py.tools._predictor import (
    kalman_predictor,
    rigid_kalman_predictor,
)


def make_simple_cluster_map(shift=(1.0, 0.5)):
    """Return a cluster_map with exactly one cluster id==1 and two frames."""
    coords_prev = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    coords_curr = coords_prev + np.asarray(shift)
    idx = np.arange(coords_prev.shape[0])
    return {
        1: {
            -1: (coords_prev, idx),
            0: (coords_curr, idx),
        }
    }


def assert_list_of_dict(pred_out, *, expect_len):
    assert isinstance(pred_out, list), "Predictor must return a list"
    assert len(pred_out) == expect_len, f"Expected list length {expect_len}"
    for d in pred_out:
        assert isinstance(d, dict), "Each list item must be a dict"


def test_kalman_predictor_structure():
    pred_fn = kalman_predictor()
    cmap = make_simple_cluster_map()
    out = pred_fn(cmap)
    assert_list_of_dict(out, expect_len=1)
    vec = out[0][1]
    assert isinstance(vec, np.ndarray) and vec.shape == (2,)


def test_rigid_kalman_structure():
    pred_fn = rigid_kalman_predictor()
    cmap = make_simple_cluster_map()
    out = pred_fn(cmap)
    assert_list_of_dict(out, expect_len=1)
    tform = out[0][1]
    assert set(tform) == {"R", "t"}
    R, t = tform["R"], tform["t"]
    assert R.shape == (2, 2) and t.shape == (2,)


def apply_translation(coords, vec):
    return coords + vec


def test_kalman_predictor_velocity_correctness():
    shift = (0.7, -0.3)
    cmap = make_simple_cluster_map(shift=shift)
    pred_fn = kalman_predictor()
    out = pred_fn(cmap)
    v = out[0][1]
    assert np.allclose(v, shift, atol=1e-2), "Velocity vector should match centroid shift"
