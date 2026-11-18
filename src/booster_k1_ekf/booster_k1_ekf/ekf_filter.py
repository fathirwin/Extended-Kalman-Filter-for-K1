import numpy as np
import scipy.linalg
from booster_k1_ekf.math_utils import (skew_symmetric, quat_to_matrix, zeta,
                                       quaternion_multiply, rot_axis_angle)

class BipedalEKF:
    # error state: r(0:3) v(3:6) phi(6:9) p1(9:12) p2(12:15) bf(15:18) bw(18:21)
    def __init__(self, kin, R_alpha_val=1e-6, R_s_val=1e-4, Q_f_val=1e-5, Q_w_val=1e-7,
                 Qp_stance=1e-4, Qp_swing=1e4, g=(0.0, 0.0, -9.81), freeze_yaw_bias=True):
        self.kin = kin
        self.P = np.diag([1e-6]*3 + [1e-4]*3 + [1e-4, 1e-4, 1e-6] + [1e-3]*6 + [1e-2]*3 + [1e-4]*3)
        self.r = np.zeros(3); self.v = np.zeros(3)
        self.q = np.array([0.0, 0.0, 0.0, 1.0])            # badan->dunia (konvensi Webots)
        self.p1 = np.zeros(3); self.p2 = np.zeros(3)
        self.bf = np.zeros(3); self.bw = np.zeros(3)
        self.R_alpha = np.eye(6)*R_alpha_val
        self.R_s = np.eye(3)*R_s_val
        self.Q_f, self.Q_w = np.eye(3)*Q_f_val, np.eye(3)*Q_w_val
        self.Q_bf, self.Q_bw = np.eye(3)*Q_f_val*0.1, np.eye(3)*Q_w_val*0.1
        self.Qp_stance, self.Qp_swing = Qp_stance, Qp_swing
        if freeze_yaw_bias:      # yaw tidak teramati (Bloesch): bias gyro sumbu z tidak diadaptasi
            self.P[20, 20] = 1e-12
            self.Q_bw[2, 2] = 0.0
        self.g = np.array(g, float)

    @property
    def C(self):                                           # dunia -> badan
        return quat_to_matrix(self.q).T

    def initialize(self, q_wb, alpha0):
        self.q = np.array(q_wb, float) / np.linalg.norm(q_wb)
        Cw = self.C.T
        self.p1 = self.r + Cw @ self.kin.fk("left", alpha0[0:6])[0]      # p_i0 = r0 + C0^T lkin(a0)
        self.p2 = self.r + Cw @ self.kin.fk("right", alpha0[6:12])[0]

    def predict(self, f_tilde, w_tilde, dt, contact_left, contact_right):
        C = self.C; Cw = C.T
        f_hat, w_hat = f_tilde - self.bf, w_tilde - self.bw
        a = Cw @ f_hat + self.g                            # robot diam -> a ~ 0

        G0T = rot_axis_angle(w_hat, np.linalg.norm(w_hat)*dt).T
        fx = skew_symmetric(f_hat)
        F = np.eye(21)
        F[0:3, 3:6] = np.eye(3)*dt
        F[0:3, 6:9] = -(dt**2/2)*Cw @ fx
        F[0:3, 15:18] = -(dt**2/2)*Cw
        F[3:6, 6:9] = -dt*Cw @ fx
        F[3:6, 15:18] = -dt*Cw
        F[6:9, 6:9] = G0T
        F[6:9, 18:21] = -np.eye(3)*dt

        Qp1 = np.eye(3)*(self.Qp_stance if contact_left else self.Qp_swing)
        Qp2 = np.eye(3)*(self.Qp_stance if contact_right else self.Qp_swing)
        Q = np.zeros((21, 21))
        Q[0:3, 0:3] = dt**3/3*self.Q_f
        Q[0:3, 3:6] = Q[3:6, 0:3] = dt**2/2*self.Q_f
        Q[3:6, 3:6] = dt*self.Q_f
        Q[6:9, 6:9] = dt*self.Q_w
        Q[9:12, 9:12] = dt*Cw @ Qp1 @ C
        Q[12:15, 12:15] = dt*Cw @ Qp2 @ C
        Q[15:18, 15:18] = dt*self.Q_bf
        Q[18:21, 18:21] = dt*self.Q_bw

        self.r = self.r + dt*self.v + 0.5*dt**2*a
        self.v = self.v + dt*a
        self.q = quaternion_multiply(self.q, zeta(dt*w_hat))
        self.q /= np.linalg.norm(self.q)
        self.P = F @ self.P @ F.T + Q

    def update(self, alpha):
        C = self.C
        s_l, J_l = self.kin.fk("left", alpha[0:6])
        s_r, J_r = self.kin.fk("right", alpha[6:12])
        y = np.concatenate([s_l - C @ (self.p1 - self.r), s_r - C @ (self.p2 - self.r)])

        H = np.zeros((6, 21))
        H[0:3, 0:3] = -C;  H[0:3, 6:9] = skew_symmetric(C @ (self.p1 - self.r)); H[0:3, 9:12] = C
        H[3:6, 0:3] = -C;  H[3:6, 6:9] = skew_symmetric(C @ (self.p2 - self.r)); H[3:6, 12:15] = C

        R = scipy.linalg.block_diag(self.R_s + J_l @ self.R_alpha @ J_l.T,
                                    self.R_s + J_r @ self.R_alpha @ J_r.T)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        self.r += dx[0:3]; self.v += dx[3:6]
        self.p1 += dx[9:12]; self.p2 += dx[12:15]
        self.bf += dx[15:18]; self.bw += dx[18:21]
        self.q = quaternion_multiply(self.q, zeta(dx[6:9]))
        self.q /= np.linalg.norm(self.q)

        IKH = np.eye(21) - K @ H                           # bentuk Joseph
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5*(self.P + self.P.T)