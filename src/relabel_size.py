"""Label-free variant of the real front end (tag real0.5s): class from cluster geometry only.
VRU: vertical extent 0.9-2.3 m, footprint <= 2.6 x 1.3 m, >= 8 returns. Vehicle: footprint longer than 2.6 m or
wider than 1.3 m with vertical extent >= 0.5 m (truck/bus if longer than 6 m). Everything else: other.
Returns, membership and box geometry are those of real0.5."""
import glob, os
import numpy as np, pandas as pd
WORK = os.environ.get("WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work_vru"))

def label(d):
    lo, sh, H = np.maximum(d.w, d.l), np.minimum(d.w, d.l), d.zmax - d.zmin
    vru = H.between(0.9, 2.3) & (lo <= 2.6) & (sh <= 1.3) & (d.num_points >= 8)
    veh = ~vru & ((lo > 2.6) | (sh > 1.3)) & (H >= 0.5)
    return np.where(vru, 0, np.where(veh, np.where(lo > 6, 2, 1), 3))

if __name__ == "__main__":
    for src in sorted(glob.glob(f"{WORK}/*__*/real0.5/detection_bbox_results.csv")):
        d = os.path.dirname(src); dst = d + "s"; os.makedirs(dst, exist_ok=True)
        for f in ("detection", "members.pkl"):
            if not os.path.exists(f"{dst}/{f}"): os.symlink(f"{d}/{f}", f"{dst}/{f}")
        df = pd.read_csv(src); df["label_oracle"] = df.label; df["label"] = label(df); df.to_csv(f"{dst}/detection_bbox_results.csv", index=False)
    print("relabelled", len(glob.glob(f"{WORK}/*__*/real0.5s/detection_bbox_results.csv")))
