from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

try:
    from numba import njit
except ModuleNotFoundError:
    def njit(*args, **kwargs):
        def _decor(f):
            return f

        return _decor


Array = np.ndarray
PredVector = Array  # shape (d,)
PredRigid = Dict[str, Array]  # keys {"R", "t"}
PredCoords = Dict[str, Array]  # key  {"coords"}
Prediction = Union[PredVector, PredRigid, PredCoords]


class Predictor:
    """Predict future coordinates of tracked clusters (new format only).

    ``prediction_map`` **must** be a ``list`` whose *i-th* element is a
    ``dict`` mapping *cluster-id → prediction* for the *i-th* stored frame.
    Supported prediction kinds per cluster:

    1. **translation** - ``np.ndarray`` of shape ``(d,)``
    2. **rigid transform** - ``{"R": ndarray((d,d)), "t": ndarray((d,))}``
    3. **explicit coords** - ``{"coords": ndarray((m,d))}`` (exact replacement)

    Arguments:
        predictor (Callable): A callable that takes a cluster map and returns a prediction map.
            If None, the default predictor is used.

    Methods:
        fit(coordinates, cluster_ids): Fits the predictor to the given coordinates and cluster IDs.
        predict(coordinates, cluster_ids, copy=True): Predicts future coordinates based on the fitted predictor.
        default_predictor(cluster_map): Default predictor that returns centroid velocities.
    """

    def __init__(self, predictor: Callable | None = None):
        """Initialize the predictor."""
        self.predictor: Callable | None = predictor or self.default_predictor
        self.prediction_map: List[Dict[int, Prediction]] = []
        self._fitted: bool = False

    def fit(self, coordinates: List[Array], cluster_ids: List[Array]) -> None:
        """Fit the predictor to the given coordinates and cluster IDs.

        Arguments:
            coordinates (List[Array]): List of coordinates for each timepoint.
            cluster_ids (List[Array]): List of cluster IDs for each timepoint.
        """
        if len(coordinates) != len(cluster_ids):
            raise ValueError("coordinates and cluster_ids must have the same length")
        if len(coordinates) < 2:
            raise ValueError("Need at least two timepoints to fit a predictor")

        cluster_map = self._create_cluster_map(coordinates, cluster_ids)
        raw_pm = self.predictor(cluster_map) if self.predictor else []

        if isinstance(raw_pm, dict):
            raw_pm = [raw_pm]
        if not (isinstance(raw_pm, list) and all(isinstance(d, dict) for d in raw_pm)):
            raise TypeError("Predictor must return list[dict[int, Prediction]] (got " f"{type(raw_pm).__name__})")
        self.prediction_map = raw_pm  # type: ignore[assignment]
        self._fitted = True

        for pm in self.prediction_map:
            for cid, pred in pm.items():
                if isinstance(pred, dict):
                    if "R" in pred and "t" in pred:
                        R, t = pred["R"], pred["t"]
                        if R.shape[0] != R.shape[1] or R.shape[0] != t.shape[0]:
                            raise ValueError(
                                f"Rigid transform for cid={cid} has inconsistent shapes: R={R.shape}, t={t.shape}"
                            )
                    elif "coords" in pred:
                        pass
                    else:
                        raise TypeError(f"Dict prediction for cid={cid} must contain ('R','t') or 'coords'")

    def predict(
        self,
        coordinates: list[Array],
        cluster_ids: list[Array],
        *,
        copy: bool = True,
    ) -> list[Array]:
        """Predict future coordinates based on the fitted predictor.

        Arguments:
            coordinates (List[Array]): List of coordinates for each timepoint.
            cluster_ids (List[Array]): List of cluster IDs for each timepoint.
            copy (bool): If True, return a copy of the coordinates.

        Returns:
            List[Array]: List of predicted coordinates for each timepoint.
        """
        if len(coordinates) != len(cluster_ids):
            raise ValueError("coordinates and cluster_ids must have the same length")

        if not self._fitted:
            warnings.warn("Predictor has not been fitted; returning input unmodified.")
            return coordinates

        coordinates = [c.copy() if copy else c for c in coordinates]

        n_frames = len(coordinates)
        pm = self.prediction_map
        if len(pm) > n_frames:
            pm_seq = pm[-n_frames:]
        elif len(pm) < n_frames:
            pm_seq = [{} for _ in range(n_frames - len(pm))] + pm
        else:
            pm_seq = pm

        for coord_f, ids_f, pm_f in zip(coordinates, cluster_ids, pm_seq):
            self._apply_prediction_map(coord_f, ids_f, pm_f)

        return coordinates

    @staticmethod
    def _apply_prediction_map(
        coordinates: Array,
        cluster_ids: Array,
        pred_map: Dict[int, Prediction],
    ) -> None:
        if not pred_map:
            return
        order = np.argsort(cluster_ids, kind="stable")
        rev_order = np.empty_like(order)
        rev_order[order] = np.arange(order.size)

        coords_sorted = coordinates[order]
        ids_sorted = cluster_ids[order]

        start, n = 0, ids_sorted.size
        while start < n:
            cid = ids_sorted[start]
            end = start + 1
            while end < n and ids_sorted[end] == cid:
                end += 1
            if cid in pred_map:
                Predictor._apply_single(pred_map[cid], coords_sorted[start:end], cid)
            start = end

        coordinates[:] = coords_sorted[rev_order]

    @staticmethod
    def _apply_single(pred: Prediction, coords_view: Array, cid: int) -> None:
        if isinstance(pred, np.ndarray):  # translation vector
            coords_view[:] = coords_view + pred
            return
        if isinstance(pred, dict) and "R" in pred and "t" in pred:  # rigid
            R, t = pred["R"], pred["t"]
            if coords_view.shape[1] != R.shape[0] or t.shape[0] != R.shape[0]:
                raise ValueError(f"Dimension mismatch rigid transform cid={cid}: R={R.shape}, t={t.shape}")
            coords_view[:] = coords_view @ R.T + t
            return
        if isinstance(pred, dict) and "coords" in pred:  # explicit coords
            new_c = pred["coords"]
            if new_c.shape != coords_view.shape:
                raise ValueError(
                    f"Explicit coords for cid={cid} have shape {new_c.shape}, expected {coords_view.shape}"
                )
            coords_view[:] = new_c
            return
        raise TypeError(f"Unsupported prediction for cid={cid}: {type(pred)}")

    @staticmethod
    def default_predictor(cluster_map: Dict[int, Dict[int, Tuple[Array, Tuple[Array]]]]) -> Dict[int, Array]:
        """Return *per-cluster* translation vectors (centroid velocity)."""

        def centroid(a: Array) -> Array:
            return a if a.shape[0] < 2 else a.mean(axis=0, keepdims=True)

        def velocity(cents: List[Array]) -> Array:
            return np.zeros_like(cents[0]) if len(cents) < 2 else np.diff(cents, axis=0).mean(axis=0)

        preds: Dict[int, Array] = {}
        for cid, tmap in cluster_map.items():
            cents = [centroid(coords) for coords, _ in tmap.values()]
            preds[cid] = velocity(cents)
        return preds

    @staticmethod
    def _create_cluster_map(
        coordinates: List[Array],
        cluster_ids: List[Array],
    ) -> Dict[int, Dict[int, Tuple[Array, Array]]]:
        result: Dict[int, Dict[int, Tuple[Array, Array]]] = defaultdict(dict)
        for t, (coords, ids) in enumerate(zip(coordinates, cluster_ids), start=-len(coordinates) + 1):
            uniques = np.unique(ids)
            if uniques.size == 0:
                result[-1][t] = (np.empty((0, coords.shape[1])), np.empty((0,), dtype=int))
                continue
            for uid in uniques:
                idx = np.where(ids == uid)[0]
                result[uid][t] = (coords[idx], idx)
        return result


@njit(cache=True, fastmath=True)
def _kalman_step(
    x: np.ndarray, P: np.ndarray, z: np.ndarray, A: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray
):
    """Const‑velocity Kalman update (2D or 3D)."""
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    y = z - H @ x_pred
    x_new = x_pred + K @ y
    P_new = (np.eye(P.shape[0]) - K @ H) @ P_pred
    return x_new, P_new


@njit(cache=True, fastmath=True)
def _kabsch(X: np.ndarray, Y: np.ndarray):
    """Return optimal R (2×2) that minimises ‖XRᵀ−Y‖² (Kabsch, 2-D)."""
    muX = X.sum(0) / X.shape[0]
    muY = Y.sum(0) / Y.shape[0]
    X0 = X - muX
    Y0 = Y - muY
    H = X0.T @ Y0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = muY - muX @ R.T
    return R, t


def kalman_predictor(q: float = 1e-3, r: float = 0.01):
    """Return a list‑format centroid CV Kalman predictor with Numba core."""
    state: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    Q = q * np.eye(4)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    Rm = r * np.eye(2)

    def _predictor(cluster_map):
        """
        Proper constant-velocity KF.

        • On first sighting of a cluster use the *displacement between the last
          two frames* to initialise the velocity state.
        • Thereafter run the normal KF prediction/update and output x[2:].
        """
        pred_last: Dict[int, np.ndarray] = {}

        for cid, frames in cluster_map.items():
            sorted_keys = sorted(frames)
            coords_new, _ = frames[sorted_keys[-1]]
            z = coords_new.mean(0)

            if cid not in state:
                if len(sorted_keys) >= 2:
                    coords_prev, _ = frames[sorted_keys[-2]]
                    v0 = z - coords_prev.mean(0)
                    print(f"Initialising {cid} with v0={v0}")
                else:
                    v0 = np.zeros(2)
                state[cid] = (np.hstack([z, v0]), np.eye(4))
                pred_last[cid] = v0
                continue

            print(f"Updating {cid} with z={z}")

            x, P = state[cid]
            x, P = _kalman_step(x, P, z, A, Q, H, Rm)
            state[cid] = (x, P)
            pred_last[cid] = x[2:]  # current velocity estimate

        return [pred_last]

    return _predictor


def rigid_kalman_predictor(
    q_pos: float = 5e-3,
    q_ori: float = 5e-3,
    r_pos: float = 1e-2,
    r_ori: float = 5e-2,
    round_thresh: float = 1.3,
):
    """
    Constant-velocity SE(2) Kalman filter per cluster.
    Returns [ {cid: {'R': R, 't': t}, … } ]       # list-of-dicts format.
    """

    state = {}  # cid -> dict(x, P)
    last_theta = defaultdict(float)

    A = np.eye(6)
    A[0, 3] = A[1, 4] = 1.0
    A[2, 5] = 1.0
    Q = np.diag([q_pos, q_pos, q_ori, q_pos, q_pos, q_ori])
    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    R_base = np.diag([r_pos, r_pos, r_ori])

    @njit(cache=True, fastmath=True)
    def _kalman_step(x, P, z, Rm):
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        S = H @ P_pred @ H.T + Rm
        K = P_pred @ H.T @ np.linalg.inv(S)
        y = z - H @ x_pred
        x = x_pred + K @ y
        P = (np.eye(6) - K @ H) @ P_pred
        return x, P

    def _predictor(cluster_map):
        pred = {}

        for cid, frames in cluster_map.items():
            t_curr = max(frames)
            coords_curr, _ = frames[t_curr]
            if coords_curr.size == 0:
                continue

            z_pos = coords_curr.mean(0)

            Xc = coords_curr - z_pos
            _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            λ1, λ2 = s[:2]

            use_theta = λ1 / λ2 >= round_thresh
            θ_meas = last_theta[cid]
            Rm = R_base.copy()

            if use_theta:
                v = Vt[0]
                θ_raw = np.arctan2(v[1], v[0])
                δ = θ_raw - last_theta[cid]
                if np.abs(δ) > np.pi / 2:
                    θ_raw += np.pi if δ < 0 else -np.pi
                θ_meas = θ_raw
            else:
                Rm[2, 2] = 1e3  # inflate orientation noise if unreliable

            z = np.array([*z_pos, θ_meas])

            if cid not in state:
                vx0 = vy0 = ω0 = 0.0
                if len(frames) >= 2:  # we have a previous frame
                    t_prev = sorted(frames)[-2]
                    coords_prev, _ = frames[t_prev]
                    z_prev = coords_prev.mean(0)
                    vx0, vy0 = z_pos - z_prev  # centroid shift

                    if use_theta:
                        # rough ω = Δθ (since Δt = 1 frame)
                        coords_prev_c = coords_prev - z_prev
                        _, s_prev, Vt_prev = np.linalg.svd(coords_prev_c, full_matrices=False)
                        v_prev = Vt_prev[0]
                        θ_prev = np.arctan2(v_prev[1], v_prev[0])
                        ω0 = θ_meas - θ_prev

                x0 = np.array([*z, vx0, vy0, ω0])
                P0 = np.eye(6)
                state[cid] = dict(x=x0, P=P0)
                last_theta[cid] = θ_meas

            x, P = state[cid]['x'], state[cid]['P']
            x, P = _kalman_step(x, P, z, Rm)
            state[cid]['x'], state[cid]['P'] = x, P
            last_theta[cid] = x[2]

            # Δ transform to go from frame t-1 → t
            vx, vy, omega = x[3], x[4], x[5]
            c, s = np.cos(omega), np.sin(omega)
            Rmat = np.array([[c, -s], [s, c]])

            z_curr = x[:2]  # current centroid estimate
            t_vec = np.array([vx, vy]) + z_curr - z_curr @ Rmat.T

            pred[cid] = {"R": Rmat, "t": t_vec}
        return [pred]

    return _predictor


def rigid_body_predictor(max_pair_dist=np.inf):
    """Return a rigid body predictor with Numba core.

    Arguments:
        max_pair_dist (float): Maximum distance for pairs to be considered.

    Returns:
        Callable: A function that takes a cluster map and returns a prediction map.
    """

    def _predictor(cluster_map):
        pred_last: Dict[int, Dict[str, np.ndarray]] = {}
        for cid, ts in cluster_map.items():
            if len(ts) < 2:
                continue
            f_prev, f_curr = sorted(ts)[-2:]
            X, _ = ts[f_prev]
            Y, _ = ts[f_curr]
            if X.size == 0 or Y.size == 0:
                continue
            tree = cKDTree(Y)
            dist, idx = tree.query(X, k=1)
            valid = dist <= max_pair_dist
            if np.sum(valid) < 3:
                continue
            Rmat, t = _kabsch(X[valid], Y[idx[valid]])
            pred_last[cid] = {"R": Rmat, "t": t}
        return [pred_last]

    return _predictor
