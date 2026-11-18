#!/usr/bin/env python3
# validasi_fk.py

import argparse
import numpy as np
from booster_k1_ekf.data_io import load_log
from booster_k1_ekf.kinematics import Kinematics
from booster_k1_ekf.contact import ContactDetector
from booster_k1_ekf.math_utils import quat_to_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--config", default="config/kinematics_k1.json")
    a = ap.parse_args()
    
    # 1. Pemuatan Data Log dan Parameter Geometri
    log, kin = load_log(a.log), Kinematics.load(a.config)
    
    # 2. Uji Validasi Turunan Matriks Jacobian (Analitik vs Finite Difference)
    print("Selisih Jacobian analitik vs finite-difference:", f"{kin.check_jacobian():.2e}", "(harus < 1e-6)")
    
    # 3. Transformasi Koordinat Ground Truth Menggunakan Parameter Rigid kin.R_bi dan kin.p_bi
    R_gt = log["gt_R"] @ kin.R_bi
    p_gt = log["gt_p"] + np.einsum("nij,j->ni", log["gt_R"], kin.p_bi)
    
    # POSISI REVISI: Baris Perhitungan Deviasi Dipindahkan ke Sini 
    dev = max(np.abs(quat_to_matrix(q) - Rg).max() for q, Rg in zip(log["q"][::50], R_gt[::50]))
    print(f"Selisih R(quaternion IMU) vs ground truth: {dev:.3e} (harus < ~1e-3; "
          f"jika besar tapi kecil bila ditranspos -> konvensi terbalik)")
    
    # 4. Loop Pengujian Segmen Stance Kaki Kiri dan Kanan (Minimal 20 Konfigurasi Acak)
    for side, col, sl in (("left", 0, slice(0, 6)), ("right", 1, slice(6, 12))):
        det = ContactDetector()
        c = np.array([det.update(v) for v in log["touch"][:, col]])
        pf = np.array([p_gt[k] + R_gt[k] @ kin.fk(side, log["alpha"][k, sl])[0] for k in range(len(c))])
        idx = np.flatnonzero(np.diff(np.r_[0, c.astype(int), 0]) != 0).reshape(-1, 2)
        
        print(f"\nKaki {side}: {len(idx)} segmen stance")
        for s, e in idx:
            if e - s < 25: continue
            seg = pf[s+5:e]
            print(f"  t={log['t'][s]:.2f}-{log['t'][e-1]:.2f}  z_mean={seg[:,2].mean()*1e3:7.2f} mm  "
                  f"std_xy={seg[:, :2].std(0).max()*1e3:6.2f} mm")
                  
    print("\nHarapan: z_mean ~ 0 (tinggi sol), std_xy beberapa mm. Jika besar -> FK/PROTO/titik sensor salah.")

if __name__ == "__main__":
    main()
