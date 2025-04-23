"""
vis_predictor_demo.py
Usage:  python vis_predictor_demo.py {kalman|rigid_kalman|kabsch|per_point}
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from arcos4py.tools import _predictor as cp

# ------------------------------------------------------------------  set-up
Predictor = cp.Predictor
FACT = {
    "kalman":        cp.kalman_predictor,
    "rigid_kalman":  cp.rigid_kalman_predictor,
    "kabsch":        cp.rigid_body_predictor,
    "per_point":     cp.per_point_kalman_framewise,
}
try:
    name = sys.argv[1]
    factory = FACT[name]
except Exception:
    print("Choose predictor: kalman | rigid_kalman | kabsch | per_point")
    sys.exit(1)

# ------------------------------------------------------------------  synthetic trajectory
# ------------------------------------------------------------------  synthetic trajectory  (curved path)
n_pts   = 10
theta   = np.linspace(0, np.pi, n_pts, endpoint=False)   # half-moon template
cloud0  = np.stack([np.cos(theta), np.sin(theta)], 1)    # shape at t=0

# ---------- motion parameters --------------------------------------
rot_per_frame = 0.15                 # intrinsic rotation of the cloud
angle_step    = 0.1                 # centre moves around big circle
radius        =50.0                 # radius of that circle

R_intrinsic = np.array([[np.cos(rot_per_frame), -np.sin(rot_per_frame)],
                        [np.sin(rot_per_frame),  np.cos(rot_per_frame)]])

def centre(k: int) -> np.ndarray:
    """Position of the cloud’s centroid at frame k along a circular path."""
    phi = k * angle_step
    return radius * np.array([np.cos(phi), np.sin(phi)])

def frame(k: int) -> np.ndarray:
    """Full rigid motion = intrinsic rotation + curved translation."""
    Ck = centre(k)
    return cloud0 @ np.linalg.matrix_power(R_intrinsic, k).T + Ck

# --------------------------------------------------------------- parameters
n_frames   = 15               # how many ground-truth frames you want
history    = 2                # sliding-window length for fitting
marker_seq = ["o", "v", "^", "<", ">", "s", "P", "D", "X", "h"]

# --------------------------------------------------------------- trajectory
true = [frame(k) for k in range(n_frames)]

# --------------------------------------------------------------- predictor
wrap = Predictor(factory())

coords_hist = true[:history]                 # frames 0 … history-1
ids_hist    = [np.zeros(n_pts, int)] * history

predicted = []                               # will hold n_frames-history items

for t in range(history, n_frames):
    # fit on the last `history` frames
    wrap.fit(coords_hist, ids_hist)

    # predict frame t  (one step ahead of coords_hist[-1])
    coords_pred = wrap.predict([coords_hist[-1]], [ids_hist[-1]], copy=True)[0]
    predicted.append(coords_pred)

    # slide the window: drop oldest, append true frame t
    coords_hist = coords_hist[1:] + [true[t]]
    ids_hist    = ids_hist[1:] + [np.zeros(n_pts, int)]

# --------------------------------------------------------------- plotting
fig, ax = plt.subplots(figsize=(6, 6))

for k, (truth, pred) in enumerate(zip(true[history:], predicted), start=history):
    mk = marker_seq[k % len(marker_seq)]
    ax.scatter(*truth.T, marker=mk, label=f"true t={k}", alpha=.8)
    ax.scatter(*pred.T,  marker=mk, facecolors='none', edgecolors='k',
               label=f"pred t={k}")

ax.set_aspect("equal")
ax.set_title(name)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------  plot
fig, ax = plt.subplots(figsize=(6,6))
mk = ["o", "v", "^", "<", ">", "s", "D"]

for i, (truth, pred) in enumerate(zip(true[2:], predicted), start=2):
    ax.scatter(*truth.T, marker=mk[i], label=f"true t={i}", alpha=.8)
    ax.scatter(*pred.T,  marker=mk[i], facecolors='none',
               edgecolors='k', label=f"pred t={i}")

ax.set_aspect("equal"); ax.set_title(name); ax.legend(bbox_to_anchor=(1.02,1))
plt.tight_layout(); plt.show()
