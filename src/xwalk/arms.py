import glob, pickle, numpy as np
v, c = [], []
for d in sorted(glob.glob("work_lumpi/lumpi__c*")):
    for f, bs in pickle.load(open(f"{d}/gt_all.pkl", "rb")).items():
        for b in bs: (v if b["category_id"] == 0 else c).append((b["x"], b["y"], b["yaw"]))
v, c = np.array(v), np.array(c)
def frame(p, o, th):
    t = np.deg2rad(th); dx, dy = p[:, 0] - o[0], p[:, 1] - o[1]
    return np.cos(t) * dx + np.sin(t) * dy, -np.sin(t) * dx + np.cos(t) * dy
# vehicle heading near the NE arm
m = (c[:, 0] > 15) & (c[:, 0] < 40) & (c[:, 1] > 20) & (c[:, 1] < 40)
print("NE arm vehicle yaw (deg, mod 180) median", np.median(np.rad2deg(c[m, 2]) % 180))
m = (c[:, 0] > -35) & (c[:, 0] < -15) & (c[:, 1] > 20) & (c[:, 1] < 40)
print("NW arm vehicle yaw median", np.median(np.rad2deg(c[m, 2]) % 180))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(20, 9))
for a, th in zip(ax, (47, 131)):
    for p, cm in ((c, "Reds"), (v, "Blues")):
        s, d = frame(p, (0, 0), th)
        a.hist2d(s, d, bins=[np.arange(-40, 50, 0.4), np.arange(-40, 40, 0.4)], cmin=1, cmap=cm, norm=matplotlib.colors.LogNorm(), alpha=0.7)
    a.set_xticks(range(-40, 50, 2)); a.set_yticks(range(-40, 40, 2)); a.grid(alpha=0.5); a.tick_params(labelsize=6); a.set_title(f"axis {th} deg: x=s along arm, y=d lateral")
fig.savefig("xwalk/arms.png", dpi=70, bbox_inches="tight")
