"""python3 -m booster_k1_ekf.replay_ekf --log data/walk.csv --mode ekf --out data/walk_ekf.csv
   python3 -m booster_k1_ekf.replay_ekf --log data/walk.csv --mode kin --out data/walk_kin.csv
   python3 -m booster_k1_ekf.replay_ekf --log data/walk.csv --mode kin_ref --out data/walk_kinref.csv
mode ekf = EKF Bloesch; kin = baseline joint-state odometry (orientasi dari gyro); kin_ref = idem, orientasi ideal."""
import argparse
import numpy as np
from booster_k1_ekf.data_io import load_log, EST_COLS
from booster_k1_ekf.kinematics import Kinematics
from booster_k1_ekf.ekf_filter import BipedalEKF
from booster_k1_ekf.contact import ContactDetector
from booster_k1_ekf.math_utils import quat_to_matrix, quaternion_multiply, zeta

def run_ekf(log, kin, **params):
    ekf = BipedalEKF(kin, **params)
    dl, dr = ContactDetector(), ContactDetector()
    t = log["t"]; rows = []
    for k in range(len(t)):
        cl, cr = dl.update(log["touch"][k, 0]), dr.update(log["touch"][k, 1])
        if k == 0:
            ekf.initialize(log["q"][0], log["alpha"][0])   # robot diam di awal
        else:
            ekf.predict(log["f"][k], log["w"][k], t[k] - t[k-1], cl, cr)
            ekf.update(log["alpha"][k])
        rows.append([t[k], *ekf.r, *ekf.v, *ekf.q, float(cl), float(cr)])
    return np.array(rows)

def run_kin(log, kin, attitude="gyro"):
    """Odometri sendi: r = p_kaki - R s. Kaki yang menapak dijangkar di dunia; jangkar baru dipasang saat touchdown.
    attitude='gyro': orientasi awal dari InertialUnit lalu integrasi gyro (sensor sama dengan EKF, tanpa filter).
    attitude='ref' : orientasi = InertialUnit tiap sampel (batas atas, tidak realistis)."""
    dl, dr = ContactDetector(), ContactDetector()
    t = log["t"]; r = np.zeros(3); anchor = [None, None]; prev = [False, False]; rows = []
    q = np.array(log["q"][0], float)
    for k in range(len(t)):
        if attitude == "ref":
            q = np.array(log["q"][k], float)
        elif k > 0:
            q = quaternion_multiply(q, zeta((t[k] - t[k-1])*log["w"][k])); q /= np.linalg.norm(q)
        R = quat_to_matrix(q)
        s = [kin.fk("left", log["alpha"][k, 0:6])[0], kin.fk("right", log["alpha"][k, 6:12])[0]]
        c = [dl.update(log["touch"][k, 0]), dr.update(log["touch"][k, 1])]
        if k == 0:
            anchor = [r + R @ s[0], r + R @ s[1]]
        else:
            old = [i for i in (0, 1) if c[i] and prev[i]]
            if old:
                r = np.mean([anchor[i] - R @ s[i] for i in old], axis=0)
            for i in (0, 1):
                if c[i] and not prev[i]:
                    anchor[i] = r + R @ s[i]
        prev = c
        rows.append([t[k], *r, 0.0, 0.0, 0.0, *q, float(c[0]), float(c[1])])
    out = np.array(rows)
    out[:, 4:7] = np.gradient(out[:, 1:4], t, axis=0)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True); ap.add_argument("--config", default="config/kinematics_k1.json")
    ap.add_argument("--mode", choices=["ekf", "kin", "kin_ref"], default="ekf"); ap.add_argument("--out", required=True)
    ap.add_argument("--R_alpha", type=float, default=1e-6); ap.add_argument("--R_s", type=float, default=1e-4)
    ap.add_argument("--Q_f", type=float, default=1e-5); ap.add_argument("--Q_w", type=float, default=1e-7)
    a = ap.parse_args()
    log, kin = load_log(a.log), Kinematics.load(a.config)
    if a.mode == "ekf":
        res = run_ekf(log, kin, R_alpha_val=a.R_alpha, R_s_val=a.R_s, Q_f_val=a.Q_f, Q_w_val=a.Q_w)
    else:
        res = run_kin(log, kin, "ref" if a.mode == "kin_ref" else "gyro")
    np.savetxt(a.out, res, delimiter=",", header=",".join(EST_COLS), comments="")
    print("Tersimpan:", a.out)