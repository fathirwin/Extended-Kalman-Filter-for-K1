import json
import numpy as np
from booster_k1_ekf.math_utils import rot_axis_angle, rotvec

def _trans(p):
    T = np.eye(4); T[:3, 3] = p
    return T

class Kinematics:
    """FK + Jacobian analitik dari rantai hasil extract_kinematics.py.
    s = posisi titik kontak telapak relatif titik IMU, dinyatakan di frame IMU (frame B)."""

    def __init__(self, cfg):
        T = np.array(cfg["imu"]["T"], float)
        self.R_bi, self.p_bi = T[:3, :3], T[:3, 3]
        self.chains = {}
        for side in ("left", "right"):
            ch = []
            for e in cfg["legs"][side]:
                if e["type"] == "fixed":
                    ch.append(("fixed", np.array(e["T"], float)))
                else:
                    a = np.array(e["anchor"], float)
                    ch.append(("joint", a, np.array(e["axis"], float), _trans(a), _trans(-a)))
            self.chains[side] = ch
        self.n = {s: sum(1 for e in c if e[0] == "joint") for s, c in self.chains.items()}

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def fk_full(self, side, alpha):
        """Return R (orientasi telapak), s (titik kontak), Jv, Jw; semua di frame IMU."""
        T = np.eye(4); axes, origins, k = [], [], 0
        for e in self.chains[side]:
            if e[0] == "fixed":
                T = T @ e[1]
            else:
                _, anchor, axis, Ta, Tai = e
                R = T[:3, :3]
                axes.append(R @ axis); origins.append(R @ anchor + T[:3, 3])
                Rj = np.eye(4); Rj[:3, :3] = rot_axis_angle(axis, alpha[k])
                T = T @ Ta @ Rj @ Tai
                k += 1
        p = T[:3, 3]
        Rb = self.R_bi.T
        s = Rb @ (p - self.p_bi)
        Jv = np.column_stack([Rb @ np.cross(a, p - o) for a, o in zip(axes, origins)])
        Jw = np.column_stack([Rb @ a for a in axes])
        return Rb @ T[:3, :3], s, Jv, Jw

    def fk(self, side, alpha):
        _, s, Jv, _ = self.fk_full(side, alpha)
        return s, Jv

    def ik(self, side, p_t, R_t, alpha0, iters=8, lam=1e-3, tol=1e-7):
        """Damped least squares: posisi + orientasi telapak di frame IMU."""
        a = np.array(alpha0, float)
        for _ in range(iters):
            R, s, Jv, Jw = self.fk_full(side, a)
            e = np.concatenate([p_t - s, rotvec(R_t @ R.T)])
            if np.linalg.norm(e) < tol:
                break
            J = np.vstack([Jv, Jw])
            a = a + J.T @ np.linalg.solve(J @ J.T + lam*np.eye(6), e)
        R, s, _, _ = self.fk_full(side, a)
        err = np.linalg.norm(np.concatenate([p_t - s, rotvec(R_t @ R.T)]))
        return a, err

    def jacobian_numeric(self, side, alpha, eps=1e-6):
        J = np.zeros((3, self.n[side]))
        for j in range(self.n[side]):
            ap, am = np.array(alpha, float), np.array(alpha, float)
            ap[j] += eps; am[j] -= eps
            J[:, j] = (self.fk(side, ap)[0] - self.fk(side, am)[0]) / (2*eps)
        return J

    def check_jacobian(self, n_trials=20, seed=0):
        rng = np.random.default_rng(seed); worst = 0.0
        for side in ("left", "right"):
            for _ in range(n_trials):
                a = rng.uniform(-0.8, 0.8, self.n[side])
                worst = max(worst, np.abs(self.fk(side, a)[1] - self.jacobian_numeric(side, a)).max())
        return worst