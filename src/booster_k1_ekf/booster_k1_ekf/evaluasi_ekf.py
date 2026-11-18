"""python3 -m booster_k1_ekf.evaluasi_ekf --log data/walk.csv --ekf data/walk_ekf.csv --kin data/walk_kin.csv --kinref data/walk_kinref.csv --tag walk
Metrik: ATE (RMSE & maks), RPE per jendela jarak, drift per meter, error roll/pitch/yaw.
Ground truth dari Supervisor, dikoreksi lever-arm ke titik IMU, lalu digeser ke origin (filter mulai di r0=0)."""
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from booster_k1_ekf.data_io import load_log, load_est
from booster_k1_ekf.kinematics import Kinematics
from booster_k1_ekf.math_utils import quat_to_matrix, rotation_angle, euler_rpy, wrap_pi

def rmse(x): return float(np.sqrt(np.mean(np.square(x))))

def rpe(p_g, p_e, R_g, R_e, idx_pairs):
    tr, ro, dist = [], [], []
    for i, j in idx_pairs:
        dg = R_g[i].T @ (p_g[j] - p_g[i]); de = R_e[i].T @ (p_e[j] - p_e[i])
        tr.append(np.linalg.norm(de - dg)); dist.append(np.linalg.norm(dg))
        ro.append(rotation_angle((R_g[i].T @ R_g[j]).T @ (R_e[i].T @ R_e[j])))
    return np.array(tr), np.array(ro), np.array(dist)

def make_pairs(s, window, step):
    """Pasangan indeks (i, j) dengan panjang lintasan ground truth s[j]-s[i] >= window."""
    pairs, j = [], 0
    for i in range(0, len(s), step):
        j = max(j, i)
        while j < len(s) and s[j] - s[i] < window: j += 1
        if j >= len(s): break
        pairs.append((i, j))
    return pairs

def analyse(name, est, p_g, R_g, s, pairs):
    R_e = np.array([quat_to_matrix(q) for q in est["q"]])
    e = est["r"] - p_g; en = np.linalg.norm(e, axis=1)
    eul = np.array([wrap_pi(euler_rpy(Re) - euler_rpy(Rg)) for Re, Rg in zip(R_e, R_g)])
    rt, rr, rd = rpe(p_g, est["r"], R_g, R_e, pairs)
    L = max(s[-1], 1e-9)
    m = {"ATE RMSE (m)": rmse(en), "ATE max (m)": en.max(), "Error posisi akhir (m)": en[-1],
         "Drift per meter (%)": 100*en[-1]/L if s[-1] > 0.5 else float("nan"),
         "RPE trans RMSE (m)": rmse(rt) if len(rt) else float("nan"),
         "RPE trans per meter (%)": 100*float(np.mean(rt/np.maximum(rd, 1e-9))) if len(rt) and s[-1] > 0.5 else float("nan"),
         "RPE rot RMSE (deg)": np.degrees(rmse(rr)) if len(rr) else float("nan"),
         "Roll RMSE (deg)": np.degrees(rmse(eul[:, 0])), "Pitch RMSE (deg)": np.degrees(rmse(eul[:, 1])),
         "Yaw RMSE (deg)": np.degrees(rmse(eul[:, 2])), "Yaw akhir (deg)": np.degrees(eul[-1, 2])}
    return m, en, eul

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True); ap.add_argument("--ekf", required=True)
    ap.add_argument("--kin"); ap.add_argument("--kinref")
    ap.add_argument("--config", default="config/kinematics_k1.json")
    ap.add_argument("--smooth", type=float, default=4.0, help="jendela perata-rataan (s) untuk jarak tempuh")
    ap.add_argument("--rpe_dist", type=float, default=1.0, help="jendela RPE (m lintasan)")
    ap.add_argument("--rpe_time", type=float, default=1.0, help="jendela RPE (s) jika robot hampir tidak berpindah")
    ap.add_argument("--tag", default="eval")
    a = ap.parse_args()
    log, kin = load_log(a.log), Kinematics.load(a.config)
    t = log["t"]
    p_g = log["gt_p"] + np.einsum("nij,j->ni", log["gt_R"], kin.p_bi)
    p_g = p_g - p_g[0]
    R_g = log["gt_R"] @ kin.R_bi
    n_sm = max(1, int(round(a.smooth/np.median(np.diff(t)))))     # ayunan lateral per langkah tidak dihitung sebagai jarak
    p_s = uniform_filter1d(p_g, n_sm, axis=0, mode="nearest")
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p_s, axis=0), axis=1))]
    step = 5
    if s[-1] > 2*a.rpe_dist:
        pairs, unit = make_pairs(s, a.rpe_dist, step), f"{a.rpe_dist:g} m"
    else:
        pairs, unit = make_pairs(t - t[0], a.rpe_time, step), f"{a.rpe_time:g} s"
    print(f"Panjang lintasan GT: {s[-1]:.3f} m | durasi {t[-1]-t[0]:.1f} s | jendela RPE: {unit}")

    runs = {"EKF": load_est(a.ekf)}
    if a.kin: runs["Odometri sendi (gyro)"] = load_est(a.kin)
    if a.kinref: runs["Odometri sendi (att. ideal)"] = load_est(a.kinref)
    table, errs, eulers = {}, {}, {}
    for name, est in runs.items():
        table[name], errs[name], eulers[name] = analyse(name, est, p_g, R_g, s, pairs)
    keys = list(next(iter(table.values())).keys())
    w = max(len(k) for k in keys) + 2
    print(f"{'Metrik':<{w}}" + "".join(f"{n:>30}" for n in table))
    for k in keys:
        print(f"{k:<{w}}" + "".join(f"{table[n][k]:>30.4f}" for n in table))
    with open(f"data/metrics_{a.tag}.csv", "w") as f:
        f.write("metrik," + ",".join(table) + "\n")
        for k in keys: f.write(k.replace(",", "") + "," + ",".join(f"{table[n][k]:.6f}" for n in table) + "\n")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9)); ax = ax.ravel()
    ax[0].plot(p_g[:, 0], p_g[:, 1], "k", lw=2, label="Ground truth")
    for n, est in runs.items(): ax[0].plot(est["r"][:, 0], est["r"][:, 1], label=n)
    ax[0].set_title("Lintasan x-y (m)"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[0].axis("equal")
    for n in runs: ax[1].plot(s, errs[n], label=n)
    ax[1].set_title("Error posisi vs jarak tempuh"); ax[1].set_xlabel("jarak tempuh GT (m)"); ax[1].set_ylabel("m")
    for n in runs: ax[2].plot(t, np.degrees(eulers[n][:, 2]), label=n)
    ax[2].set_title("Error yaw (deg)"); ax[2].set_xlabel("t (s)")
    for n in runs:
        ax[3].plot(t, np.degrees(eulers[n][:, 0]), label=f"{n} roll")
        ax[3].plot(t, np.degrees(eulers[n][:, 1]), "--", label=f"{n} pitch")
    ax[3].set_title("Error roll / pitch (deg)"); ax[3].set_xlabel("t (s)")
    for x in ax: x.grid(True); x.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"data/evaluasi_{a.tag}.png", dpi=130)
    print(f"Plot: data/evaluasi_{a.tag}.png | Tabel: data/metrics_{a.tag}.csv")

if __name__ == "__main__":
    main()