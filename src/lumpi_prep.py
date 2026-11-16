"""Cross-site check on LUMPI (Hannover, Measurement 4: five roadside LiDARs, ids 34-38, already fused in one local frame).
Same front end as V2X-Real (prep_real.py), nothing tuned on LUMPI:
  background  0.2 m voxels occupied in >= THETA of every BG_STEP-th frame of the whole session (label-free);
  foreground  returns whose voxel and 26 neighbours are below THETA; DBSCAN(eps 0.5, min_samples 3) -> tag real0.5;
  GT          Label.csv (frame = round(10 t)); classes 0 person, 2 bicycle, 3 motorcycle -> VRU (0); 1, 6, 7 -> car (1);
              4, 5 -> bus/truck (2). Ownership with yaw cuboids (MARGIN / GROUND_CLEAR as V2X-Real), kept for evaluation.
Clips: CLIP frames from each start in STARTS (chosen by VRU density: three crowded, two medium, one sparse).
Writes work_lumpi/lumpi__c<start>/{gt_all.pkl, real0.5/, frames.npy (source frame indices)} and lumpi_bg.npz, lumpi_sensors.json."""
import glob, json, os, pickle, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
from scipy.optimize import least_squares
from v2x_prep import calculate_reliability, compute_bbox, MARGIN, GROUND_CLEAR
from bg_build import keys
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.environ.get("LSRC", os.path.join(HERE, "..", "data", "lumpi"))
# M6 held-out run (2026-11-12): LSRC, LLABEL, LSENS, LCACHE select another measurement; defaults = Measurement 4 (unchanged)
LLABEL = os.environ.get("LLABEL", f"{SRC}/labels_download/Label/Measurement4/Label.csv"); LCACHE = os.environ.get("LCACHE", "")
WORK = os.environ.get("LWORK", f"{HERE}/work_lumpi"); TAG = "real0.5"; THETA, EPS, BG_STEP = 0.2, 0.5, 20
STARTS = [int(s) for s in os.environ.get("STARTS", "0,300,600,1200,2400,4800").split(",")]; CLIP = int(os.environ.get("CLIP", 300))
SENSORS = tuple(int(s) for s in os.environ.get("LSENS", "34,35,36,37,38").split(",")); CAT = {0: 0, 2: 0, 3: 0, 1: 1, 6: 1, 7: 1, 4: 2, 5: 2}
OFF = [(dx << 34) + (dy << 17) + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("time", "<u4"), ("id", "u1"), ("intensity", "u1"), ("ray", "u1"),
               ("azimuth", "<u2"), ("distance", "<u2")])

def ply(i):
    b = open(f"{SRC}/lidar/{i:06d}.ply", "rb").read(); return np.frombuffer(b[b.index(b"end_header\n") + 11:], DT)

def xyz(p): return np.column_stack([p["x"], p["y"], p["z"]]).astype(np.float64)

def sensors():
    """Sensor origins from the per-point range (|p - o| = s * distance), label-free; cached."""
    f = f"{HERE}/lumpi_sensors{LCACHE}.json"
    if os.path.exists(f): return {int(k): np.array(v) for k, v in json.load(open(f)).items()}
    p = ply(100); out = {}
    for s in SENSORS:
        q = p[(p["id"] == s) & (p["distance"] > 0)]; X, d = xyz(q), q["distance"].astype(float)
        r = least_squares(lambda v: np.linalg.norm(X - v[:3], axis=1) - v[3] * d, [X[:, 0].mean(), X[:, 1].mean(), 0, 0.005])
        out[s] = r.x[:3]
    json.dump({k: v.tolist() for k, v in out.items()}, open(f, "w")); return out

def _bg_keys(i): return np.unique(keys(xyz(ply(i))))

def background():
    f = f"{HERE}/lumpi_bg{LCACHE}.npz"
    if not os.path.exists(f):
        fr = list(range(0, len(glob.glob(f"{SRC}/lidar/*.ply")), BG_STEP))
        with Pool(16) as pool: acc = pool.map(_bg_keys, fr)
        k, c = np.unique(np.concatenate(acc), return_counts=True)
        np.savez(f, keys=k, occ=(c / len(fr)).astype(np.float32), n=len(fr)); print("background", len(fr), "frames", len(k), "voxels")
    return dict(np.load(f))

def occ(b, p):
    k0 = keys(p); out = np.zeros(len(p))
    for o in OFF:
        k = k0 + o; i = np.clip(np.searchsorted(b["keys"], k), 0, len(b["keys"]) - 1)
        out = np.maximum(out, np.where(b["keys"][i] == k, b["occ"][i], 0.0))
    return out

def labels():
    L = pd.read_csv(LLABEL, header=None, skiprows=1, usecols=range(16))
    L.columns = ["t", "oid", "bx", "by", "bw", "bh", "score", "cls", "vis", "x", "y", "z", "l", "w", "h", "yaw"]
    L["f"] = np.round(L.t * 10).astype(int); return L[L.cls.isin(list(CAT))]

def clip(start):
    key = f"lumpi__c{start:04d}"; d = f"{WORK}/{key}"; os.makedirs(f"{d}/{TAG}/detection", exist_ok=True)
    BG = background(); S = sensors(); L = labels(); ids = {}
    gt, rows, mem = {}, [], []
    for i, src in enumerate(range(start, start + CLIP)):
        p = ply(src); X = xyz(p); g = L[L.f == src]
        own = np.full(len(X), -1); best = np.full(len(X), np.inf); boxes = []
        for b in g.itertuples():
            c, s = np.cos(b.yaw), np.sin(b.yaw); dx, dy, dz = X[:, 0] - b.x, X[:, 1] - b.y, X[:, 2] - b.z
            u, v = c * dx + s * dy, -s * dx + c * dy
            m = (np.abs(u) < b.l / 2 + MARGIN) & (np.abs(v) < b.w / 2 + MARGIN) & (dz > -b.h / 2 + GROUND_CLEAR) & (dz < b.h / 2 + MARGIN)
            dd = np.hypot(u, v); t = m & (dd < best); own[t] = len(boxes); best[t] = dd[t]
            boxes.append(dict(x=b.x, y=b.y, z=b.z, l=b.l, w=b.w, h=b.h, yaw=b.yaw, track_id=ids.setdefault(b.oid, len(ids)),
                              category_id=CAT[b.cls], type=str(b.cls), npts=0))
        for j in range(len(boxes)): boxes[j]["npts"] = int((own == j).sum())
        gt[i] = boxes
        fgm = occ(BG, X) < THETA; Xf, pf, of = X[fgm], p[fgm], own[fgm]
        O = np.zeros((256, 3)); O[list(S)] = np.array(list(S.values())); r = np.linalg.norm(Xf - O[pf["id"]], axis=1)
        refl = pf["intensity"].astype(np.float64); lid = pf["id"].astype(np.float64) - SENSORS[0]
        fg = np.column_stack([Xf, r, refl, np.zeros(len(Xf)), np.ones(len(Xf)), calculate_reliability(r.astype(np.float32), refl.astype(np.float32)), lid])
        lab = DBSCAN(eps=EPS, min_samples=3).fit_predict(fg[:, :3]) if len(fg) else np.zeros(0, int)
        out, cid, name = [], 0, f"{i:06d}.bin"
        for c_ in np.unique(lab[lab >= 0]):
            sel = lab == c_; q = fg[sel]; o = of[sel]; ok = o[o >= 0]
            if len(q) < 3: continue
            cats = np.bincount([boxes[k]["category_id"] for k in ok], minlength=4)
            label = int(np.argmax(cats)) if len(ok) >= 0.5 * len(q) else 3
            bb = compute_bbox(q[:, :3])
            rows.append(dict(frame_id=i, frame_name=name, cluster_id=cid, label=label, confidence=1.0, x=bb[0], y=bb[1], z=bb[2],
                             w=bb[3], l=bb[4], h=bb[5], yaw=bb[6], avg_eps=EPS, avg_reliability=float(q[:, 7].mean()),
                             num_points=len(q), zmin=float(q[:, 2].min()), zmax=float(q[:, 2].max()), owned=len(ok) / len(q)))
            mem.append(dict(frame_id=i, cluster_id=cid, owners={boxes[k]["track_id"]: int((ok == k).sum()) for k in np.unique(ok)}))
            out.append(np.column_stack([q, np.full(len(q), cid)])); cid += 1
        (np.vstack(out) if out else np.zeros((0, 10))).astype(np.float32).tofile(f"{d}/{TAG}/detection/{name}")
    pd.DataFrame(rows).to_csv(f"{d}/{TAG}/detection_bbox_results.csv", index=False)
    pickle.dump(mem, open(f"{d}/{TAG}/members.pkl", "wb")); pickle.dump(gt, open(f"{d}/gt_all.pkl", "wb"))
    np.save(f"{d}/frames.npy", np.arange(start, start + CLIP))
    nv = sum(1 for v in gt.values() for b in v if b["category_id"] == 0 and b["npts"] >= 3)
    return key, len(rows), nv

if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True); print({k: v.round(2).tolist() for k, v in sensors().items()}); background()
    with Pool(min(len(STARTS), 12)) as pool:
        for r in pool.imap_unordered(clip, STARTS): print(r, flush=True)
