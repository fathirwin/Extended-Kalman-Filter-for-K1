import numpy as np

COLS = (["t"] + [f"alpha{i}" for i in range(12)] + ["qx", "qy", "qz", "qw"]
        + ["fx", "fy", "fz", "wx", "wy", "wz", "touch_l", "touch_r"]
        + ["gx", "gy", "gz"] + [f"R{i}" for i in range(9)])

def load_log(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return dict(t=d[:, 0], alpha=d[:, 1:13], q=d[:, 13:17], f=d[:, 17:20],
                w=d[:, 20:23], touch=d[:, 23:25], gt_p=d[:, 25:28],
                gt_R=d[:, 28:37].reshape(-1, 3, 3))

EST_COLS = ["t", "rx", "ry", "rz", "vx", "vy", "vz", "qx", "qy", "qz", "qw", "cl", "cr"]

def load_est(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return dict(t=d[:, 0], r=d[:, 1:4], v=d[:, 4:7], q=d[:, 7:11], contact=d[:, 11:13])