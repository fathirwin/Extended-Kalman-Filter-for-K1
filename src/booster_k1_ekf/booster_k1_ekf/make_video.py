"""python3 -m booster_k1_ekf.make_video --log data/walk.csv --ekf data/walk_ekf.csv --kin data/walk_kin.csv --out data/demo.mp4
Animasi lintasan x-y: ground truth vs EKF vs odometri sendi. Butuh ffmpeg untuk .mp4 (otomatis .gif jika tidak ada)."""
import argparse, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from booster_k1_ekf.data_io import load_log, load_est
from booster_k1_ekf.kinematics import Kinematics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True); ap.add_argument("--ekf", required=True)
    ap.add_argument("--kin"); ap.add_argument("--config", default="config/kinematics_k1.json")
    ap.add_argument("--out", default="data/demo.mp4"); ap.add_argument("--speed", type=float, default=5.0, help="kecepatan x real-time")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    log, kin = load_log(a.log), Kinematics.load(a.config)
    t = log["t"]; p_g = log["gt_p"] + np.einsum("nij,j->ni", log["gt_R"], kin.p_bi); p_g = p_g - p_g[0]
    runs = {"Ground truth": (p_g, "k", 2.5), "EKF": (load_est(a.ekf)["r"], "tab:blue", 1.8)}
    if a.kin: runs["Odometri sendi (gyro)"] = (load_est(a.kin)["r"], "tab:orange", 1.8)
    n_frames = int((t[-1] - t[0])*a.fps/a.speed)
    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
    allp = np.vstack([v[0][:, :2] for v in runs.values()])
    pad = 0.15; xmin, ymin = allp.min(0) - pad; xmax, ymax = allp.max(0) + pad
    fig, ax = plt.subplots(figsize=(8, 5)); lines, dots = {}, {}
    for n, (_, c, lw) in runs.items():
        lines[n], = ax.plot([], [], color=c, lw=lw, label=n); dots[n], = ax.plot([], [], "o", color=c)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal"); ax.grid(True)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(loc="upper left")
    txt = ax.text(0.02, 0.05, "", transform=ax.transAxes)
    def draw(f):
        k = idx[f]
        for n, (p, _, _) in runs.items():
            lines[n].set_data(p[:k+1, 0], p[:k+1, 1]); dots[n].set_data([p[k, 0]], [p[k, 1]])
        err = np.linalg.norm(runs["EKF"][0][k] - p_g[k])
        txt.set_text(f"t = {t[k]-t[0]:5.1f} s | error EKF = {err*100:5.1f} cm")
        return list(lines.values()) + list(dots.values()) + [txt]
    ani = animation.FuncAnimation(fig, draw, frames=n_frames, blit=True)
    out = a.out
    if out.endswith(".mp4") and shutil.which("ffmpeg"):
        ani.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=2000))
    else:
        out = out.rsplit(".", 1)[0] + ".gif"; ani.save(out, writer=animation.PillowWriter(fps=min(a.fps, 20)))
    print("Video:", out)

if __name__ == "__main__":
    main()