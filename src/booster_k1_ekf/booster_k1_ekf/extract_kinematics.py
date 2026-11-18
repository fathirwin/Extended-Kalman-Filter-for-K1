"""Membaca file .proto Webots -> config/kinematics_k1.json.
Contoh:
python3 -m booster_k1_ekf.extract_kinematics ../../../webots_project/protos/BoosterK1.proto \
    --imu imu --left-foot left_touch_sensor --right-foot right_touch_sensor
Syarat: PROTO berisi nilai literal (hasil urdf2webots) dan TouchSensor kaki sudah ada di dalam PROTO.
Binding `IS` dan blok %< >% diabaikan (dipakai nilai default 0). Asumsi: HingeJoint dengan position awal 0."""
import re, json, argparse
import numpy as np
from booster_k1_ekf.math_utils import rot_axis_angle

LEFT = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"]
RIGHT = [n.replace("left", "right") for n in LEFT]
DEVICES = {"InertialUnit", "Accelerometer", "Gyro", "TouchSensor"}

def tokenize(text):
    text = re.sub(r"%<.*?>%", " ", text, flags=re.S)
    text = re.sub(r"(?m)^\s*#.*$", "", text).replace(",", " ")
    return re.findall(r'"[^"]*"|[{}\[\]]|[^\s{}\[\]"]+', text)

def is_num(t):
    try:
        float(t); return t[0] not in "iInN"
    except ValueError:
        return False

class Parser:
    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self): return self.t[self.i] if self.i < len(self.t) else None
    def next(self): v = self.t[self.i]; self.i += 1; return v

    def value(self):
        tok = self.peek()
        if tok == "[":
            self.next(); out = []
            while self.peek() != "]": out.append(self.value())
            self.next(); return out
        if tok == "DEF":
            self.next(); name = self.next(); n = self.value()
            if isinstance(n, dict): n["def"] = name
            return n
        if tok == "USE":
            self.next(); return {"type": "USE", "name": self.next()}
        if tok.startswith('"'):
            self.next(); return tok.strip('"')
        if is_num(tok):
            nums = []
            while self.peek() is not None and is_num(self.peek()): nums.append(float(self.next()))
            return nums
        self.next()
        if self.peek() == "{": return self.node(tok)
        return tok

    def node(self, typ):
        self.next(); f = {}
        while self.peek() != "}":
            name = self.next()
            if self.peek() == "IS":
                self.next(); self.next(); continue
            f[name] = self.value()
        self.next()
        return {"type": typ, "f": f}

def parse_proto(path):
    toks = tokenize(open(path, encoding="utf-8").read())
    i = toks.index("PROTO") + 2
    assert toks[i] == "["
    depth = 0
    while True:
        depth += (toks[i] == "[") - (toks[i] == "]"); i += 1
        if depth == 0: break
    assert toks[i] == "{", "Format PROTO tidak dikenali"
    p = Parser(toks); p.i = i + 1
    return p.value()

def node_T(f):
    tr = f.get("translation", [0, 0, 0]); rot = f.get("rotation", [0, 0, 1, 0])
    if not (isinstance(tr, list) and len(tr) == 3): tr = [0, 0, 0]
    if not (isinstance(rot, list) and len(rot) == 4): rot = [0, 0, 1, 0]
    T = np.eye(4); T[:3, :3] = rot_axis_angle(rot[:3], rot[3]); T[:3, 3] = tr
    return T

def walk(node, elems, out):
    if not isinstance(node, dict) or node["type"] == "USE": return
    f, t = node["f"], node["type"]
    elems = elems + [("fixed", node_T(f))]
    if t in DEVICES and "name" in f: out[f["name"]] = (t, elems)
    ch = f.get("children", [])
    for c in (ch if isinstance(ch, list) else [ch]): walk(c, elems, out)
    if t == "HingeJoint":
        jp = f.get("jointParameters")
        jf = jp["f"] if isinstance(jp, dict) else {}
        axis, anchor = jf.get("axis", [1, 0, 0]), jf.get("anchor", [0, 0, 0])   # default Webots axis = 1 0 0
        dev = f.get("device", []); dev = [dev] if isinstance(dev, dict) else dev
        name = next((d["f"].get("name") for d in dev if isinstance(d, dict) and d["type"] == "RotationalMotor"), None)
        walk(f.get("endPoint"), elems + [("joint", name, anchor, axis)], out)

def to_json(elems, only_joints=None):
    res = []
    for e in elems:
        if e[0] == "fixed": res.append({"type": "fixed", "T": e[1].tolist()})
        else: res.append({"type": "joint", "name": e[1], "anchor": list(e[2]), "axis": list(e[3])})
    return res

def compose_fixed(elems):
    T = np.eye(4)
    for e in elems:
        assert e[0] == "fixed", "IMU berada di bawah joint, tidak didukung"
        T = T @ e[1]
    return T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto"); ap.add_argument("--imu", default="imu")
    ap.add_argument("--left-foot", default="left_touch_sensor")
    ap.add_argument("--right-foot", default="right_touch_sensor")
    ap.add_argument("--out", default="config/kinematics_k1.json")
    a = ap.parse_args()
    root = parse_proto(a.proto); out = {}
    walk(root, [], out)
    print("Device ditemukan:", {k: v[0] for k, v in out.items()})
    cfg = {"imu": {"T": compose_fixed(out[a.imu][1]).tolist()}, "legs": {}}
    for side, dev, expected in (("left", a.left_foot, LEFT), ("right", a.right_foot, RIGHT)):
        elems = out[dev][1]
        names = [e[1] for e in elems if e[0] == "joint"]
        print(side, "joint di rantai:", names)
        if names != expected:
            raise SystemExit(f"Urutan/nama joint {side} tidak cocok dengan {expected}. Ubah LEFT/RIGHT di skrip ini.")
        cfg["legs"][side] = to_json(elems)
    json.dump(cfg, open(a.out, "w"), indent=1)
    print("Tersimpan:", a.out)

if __name__ == "__main__":
    main()