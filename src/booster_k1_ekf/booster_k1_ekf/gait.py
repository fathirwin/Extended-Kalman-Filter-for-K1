"""Gait quasi-statis open-loop di ruang kerja telapak + IK (dipakai controller Webots).
Semua posisi telapak dinyatakan di frame badan/IMU. Badan menghadap +x."""
import bisect
import numpy as np

JOINTS_L = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"]
JOINTS_R = [n.replace("left", "right") for n in JOINTS_L]
JOINTS = JOINTS_L + JOINTS_R

# batas sendi dari K1.proto: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
LIM_L = np.array([[-2.958, 2.226], [-0.375, 1.536], [-1.012, 1.012], [0.0, 2.321], [-0.87, 0.345], [-0.345, 0.345]])
LIM_R = np.array([[-2.958, 2.226], [-1.536, 0.375], [-1.012, 1.012], [0.0, 2.321], [-0.87, 0.345], [-0.345, 0.345]])
Q_INIT = np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0])

Y_NOM, X_NOM, Z_NOM = 0.0962, 0.016, -0.52   # titik kontak saat berdiri (frame IMU)

def smooth(s):
    s = min(1.0, max(0.0, s))
    return s**3*(10 - 15*s + 6*s*s)

class Gait:
    """mode 'stand': hanya berdiri. mode 'walk': berdiri t_start detik lalu berjalan sejauh dist (m)."""
    def __init__(self, kin, step_len=0.05, shift=0.01, lift=0.03, t_shift=0.2, t_step=0.30,
                 dist=3.0, t_start=4.0, ramp=2.0):
        self.kin, self.S, self.B, self.H = kin, step_len, shift, lift
        self.t_shift, self.t_step, self.t0, self.ramp = t_shift, t_step, t_start, ramp
        self.n_steps = max(3, int(np.ceil(dist/step_len)) + 1)
        self.segs = self._build()                 # [(t_a, t_b, kind, state0, state1, sup)]
        self.starts = [s[0] for s in self.segs]
        self.t_end = self.segs[-1][1]
        self.q = {"left": Q_INIT.copy(), "right": Q_INIT.copy()}
        self.q_nom = self._ik(self._pose(0.0, X_NOM, X_NOM, 0.0, 0.0))

    def _build(self):
        S, B, X, n = self.S, self.B, X_NOM, self.n_steps
        segs, t = [], self.t0
        st = dict(b=0.0, xl=X, xr=X)               # state: pergeseran lateral b, x kiri, x kanan
        def add(kind, dur, new, sup=None):
            nonlocal t, st
            segs.append((t, t + dur, kind, dict(st), dict(new), sup)); t += dur; st = dict(new)
        add("shift", self.t_shift, dict(st, b=B))
        for j in range(1, n + 1):
            sup = "left" if j % 2 == 1 else "right"
            sgn = 1.0 if sup == "left" else -1.0
            if j == 1:   xs, xw = -S/2, S/2          # langkah pertama: badan maju S/2
            elif j == n: xs, xw = 0.0, 0.0           # langkah terakhir: kedua kaki sejajar
            else:        xs, xw = -S/2, S/2
            new = dict(st)
            if sup == "left": new.update(xl=X + xs, xr=X + xw)
            else:             new.update(xr=X + xs, xl=X + xw)
            add("step", self.t_step, new, sup)
            add("shift", self.t_shift, dict(new, b=(-sgn*B if j < n else 0.0)))
        return segs

    def _pose(self, b, xl, xr, zl, zr):
        return {"left": np.array([xl, Y_NOM - b, Z_NOM + zl]),
                "right": np.array([xr, -Y_NOM - b, Z_NOM + zr])}

    def _ik(self, pose):
        out = []
        for side, lim in (("left", LIM_L), ("right", LIM_R)):
            a, _ = self.kin.ik(side, pose[side], np.eye(3), self.q[side], iters=25)
            a = np.clip(a, lim[:, 0], lim[:, 1])
            self.q[side] = a; out.append(a)
        return np.concatenate(out)

    def foot_state(self, t):
        """(b, xl, xr, zl, zr, support) pada waktu t."""
        if t < self.t0:
            return 0.0, X_NOM, X_NOM, 0.0, 0.0, "both"
        if t >= self.t_end:
            s = self.segs[-1][4]
            return s["b"], s["xl"], s["xr"], 0.0, 0.0, "both"
        i = bisect.bisect_right(self.starts, t) - 1
        ta, tb, kind, s0, s1, sup = self.segs[i]
        u = (t - ta)/(tb - ta); s = smooth(u)
        b = s0["b"] + (s1["b"] - s0["b"])*s
        xl = s0["xl"] + (s1["xl"] - s0["xl"])*s
        xr = s0["xr"] + (s1["xr"] - s0["xr"])*s
        zl = zr = 0.0
        if kind == "step":
            lift = self.H*0.5*(1 - np.cos(2*np.pi*u))
            if sup == "left": zr = lift
            else:             zl = lift
        return b, xl, xr, zl, zr, (sup or "both")

    def targets(self, mode, t):
        """dict nama_joint -> sudut (rad)."""
        if mode == "stand" or t < self.t0:
            q = min(1.0, t/self.ramp)*self.q_nom
        else:
            b, xl, xr, zl, zr, _ = self.foot_state(t)
            q = self._ik(self._pose(b, xl, xr, zl, zr))
        return dict(zip(JOINTS, q))

    def distance(self):
        return (self.n_steps - 1)*self.S